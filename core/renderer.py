# renderer.py — GPU 加速渲染导出引擎
"""USTX 可视化视频导出系统。

完整管线:
    ① CPU 预计算所有时间区间帧的视觉状态（不做视觉去重）
    ② 检测硬件 (GPU/NVENC/AMF/QSV)
    ③ 计算最优并发数 (渲染 stream/编码 worker)
    ④ 多线程并行渲染每个时间区间帧 → NV12
    ⑤ FFmpeg 逐帧真实编码（正常 GOP）
    ⑥ 音频 mux → 输出 .mp4 + 清理临时文件

对外接口:
    render_video(ust_info, output_path, ...) -> bool
"""

import os
import queue
import re
import time
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from typing import (
    Callable, List, Optional, Tuple, Dict, Any, NamedTuple,
)

import ctypes

from PySide6.QtCore import Qt, QRectF, QPointF, QSize
from PySide6.QtGui import (
    QPainter, QColor, QFont, QFontMetrics, QPen, QPolygonF,
    QImage, QFontDatabase,
)
from PySide6.QtWidgets import QApplication

from core.log import logger

# 从 main.py 导入版本号
try:
    from main import APP_VERSION
except ImportError:
    APP_VERSION = "v26h8"


# ===================== 常量 =====================

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE_LINE_WIDTH = 5
NOTE_ALPHA = 225
COPYRIGHT_ALPHA = 100

# 硬编码上限（SM 数动态计算时会覆盖此值）
MAX_RENDER_STREAMS = 8
MAX_ENCODE_WORKERS = 2
MIN_CUDA_CORES_FOR_CUDA = 512  # CUDA 核心数不足此值时禁用 CUDA 渲染
# NVENC 最快编码预设（离线导出，极致吞吐优先）
# p1 = 最快；constqp = 恒定量化参数（比 VBR 更简单，省去码率控制开销）
# bf=0 = 禁用 B 帧（只编 I/P，减少参考帧复杂度）
# 禁用 AQ/lookahead/scenecut 进一步减少编码器决策开销
NVENC_FAST_PRESET = "p1"
NVENC_FAST_OPTS = [
    "-preset", "p1",
    "-rc", "constqp", "-qp", "28",
    "-bf", "0",
    "-spatial-aq", "0", "-temporal-aq", "0",
    "-rc-lookahead", "0", "-no-scenecut", "1",
]

# 多语言 LRC 分组阈值
_LRC_GROUP_THRESHOLD = 0.020
_LRC_TIMESTAMP_RE = re.compile(r'\[(\d{1,2}):(\d{1,2})\.(\d{2,3})\]([^\[]*)')
_LRC_ENCODINGS = ['utf-8-sig', 'utf-8', 'gbk', 'gb2312', 'cp932']

# 显存安全系数
VRAM_SAFETY_FACTOR = 0.85
# 每 stream 显存开销 (GB)
VRAM_OVERHEAD_PER_STREAM = 0.15


# ===================== 渲染错误透出 =====================

_LAST_RENDER_ERROR: List[str] = []
_RENDER_ERROR_LOCK = threading.Lock()


def set_last_render_error(msg: str) -> None:
    """记录最近一次渲染失败的具体原因（渲染线程写入，UI 线程读取）。"""
    with _RENDER_ERROR_LOCK:
        _LAST_RENDER_ERROR[:] = [str(msg)]


def get_last_render_error() -> str:
    """读取最近一次渲染失败的原因，无错误时返回空字符串。"""
    with _RENDER_ERROR_LOCK:
        return _LAST_RENDER_ERROR[0] if _LAST_RENDER_ERROR else ""


def clear_last_render_error() -> None:
    """清空渲染错误记录（开始渲染前调用）。"""
    with _RENDER_ERROR_LOCK:
        _LAST_RENDER_ERROR.clear()


# ===================== 数据类型 =====================


class FrameState(NamedTuple):
    """单个帧的视觉状态（预计算结果，渲染时零 CPU 计算）。"""
    lyric: str                         # 当前歌词
    note_name: str                     # 当前音名
    bg_color: str                      # 背景色 hex
    lyric_color: Tuple[int, int, int]  # 歌词颜色 RGB
    note_color: Tuple[int, int, int]   # 音名颜色 RGB
    pitch_curve_color: str             # 音高线颜色 hex
    pitch_points: List[Tuple[float, float]]  # 音高线坐标（屏幕空间）
    show_lyric: bool                   # 是否显示歌词
    show_note_name: bool               # 是否显示音名
    show_curve: bool                   # 是否显示音高线
    # 信息文字
    song_name: str
    song_author: str
    ust_author: str
    tempo: float
    show_bpm: bool
    show_song_name: bool
    show_song_author: bool
    show_ust_author: bool
    show_copyright: bool
    show_play_time: bool
    # LRC 歌词
    lrc_lines: List[str]              # 当前显示的 LRC 行
    lrc_hidden: bool                  # 是否隐藏
    lyric_pos: str                    # "上" 或 "下"
    # 字体
    word_lyric_font_family: str
    info_font_family: str
    small_font_color: str
    # 版权
    copyright_text: str
    # 时间
    start_time: float                 # 该状态起始时间 (秒)
    duration: float                   # 持续时间 (秒)
    frame_count: int                  # 持续帧数 = duration × fps
    # 去重
    is_duplicate: bool                # 是否与上一帧视觉相同
    cache_key: str                    # 去重哈希 key


class HardwareInfo(NamedTuple):
    """硬件检测结果。"""
    gpu_name: str                     # "RTX 3060"
    gpu_vendor: str                   # "nvidia" / "amd" / "intel" / "unknown"
    cuda_cores: int                   # 0 表示无 CUDA 或非 NVIDIA
    sm_count: int                     # SM 数量（用于 CUDA 流数计算）
    vram_total_gb: float              # 总显存 (GB)
    vram_free_gb: float               # 当前空余 (GB)
    vram_usable_gb: float             # vram_free × 0.85
    nvenc_count: int                  # NVENC 编码器数量 (NVIDIA)
    nvenc_generation: int             # NVENC 代数
    has_gpu: bool                     # 是否有可用 GPU
    encoder_name: str                 # "h264_nvenc" / "h264_amf" / "h264_qsv" / "libx264"
    supports_cuda_render: bool        # 是否支持 CUDA 渲染


class WorkerConfig(NamedTuple):
    """并发分配结果。"""
    render_streams: int               # 并行渲染 stream 数
    encode_workers: int               # 并行编码 worker 数
    batch_size: int                   # 每批渲染帧数
    default_mode: str                 # "batch" / "stream"
    per_frame_gb: float               # 单帧显存占用


# ===================== 工具函数 =====================


def validate_hex_color(hex_color: str) -> str:
    """校验十六进制颜色，无效时返回 #ffffff。"""
    if re.match(r'^#([0-9A-Fa-f]{6})$', str(hex_color)):
        return hex_color.strip().lower()
    return "#ffffff"


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """#RRGGBB → (R, G, B)。"""
    try:
        h = hex_color.lstrip('#')
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
    except Exception:
        return (255, 255, 255)


def format_play_time(seconds: float) -> str:
    """秒数 → MM:SS:CC 格式。"""
    try:
        ms = int((seconds - int(seconds)) * 100)
        td = timedelta(seconds=int(seconds))
        return f"{td.seconds // 60:02d}:{td.seconds % 60:02d}:{ms:02d}"
    except Exception:
        return "00:00:00"


def midi_to_note(midi_num: int) -> str:
    """MIDI 号 → 音名 (如 C4, G#5)。"""
    try:
        midi_num = int(midi_num)
        octave = (midi_num // 12) - 1
        return f"{NOTE_NAMES[midi_num % 12]}{octave}"
    except Exception:
        return str(midi_num)


def get_pitch_text(note_num: int, placeholder: str, custom_text: str) -> str:
    """音名 + 占位符格式。"""
    try:
        ori = midi_to_note(note_num)
        pure = re.match(r'^([A-G])(\d+)$', ori)
        sharp = re.match(r'^([A-G]#)(\d+)$', ori)
        if sharp:
            return ori
        if pure:
            note, num = pure.group(1), pure.group(2)
            if placeholder == "无":
                return f"{note}{num}"
            elif placeholder == "-":
                return f"{note}-{num}"
            elif placeholder == "自定义文字":
                suffix = custom_text.strip()
                return f"{note}({suffix}){num}" if suffix else f"{note}{num}"
        return ori
    except Exception:
        return str(note_num)


def get_silent_text(silent_display: str, silent_custom: str) -> str:
    """静默音符显示文字。"""
    if silent_display == "R":
        return "R"
    if silent_display == "♪":
        return "♪"
    if silent_display == "-":
        return "-"
    if silent_display == "自定义文字":
        return silent_custom or ""
    return ""


def get_end_text(end_display: str, end_custom: str) -> str:
    """结尾显示文字。"""
    if end_display == "END":
        return "END"
    if end_display == "-":
        return "-"
    if end_display == "自定义文字":
        return end_custom or ""
    return ""


def parse_lrc_file(path: str) -> List[Tuple[float, List[str]]]:
    """解析 .lrc 文件，返回多语言分组结果。"""
    if not path or not os.path.exists(path):
        return []
    content = ""
    for enc in _LRC_ENCODINGS:
        try:
            with open(path, 'r', encoding=enc) as f:
                content = f.read()
            break
        except Exception:
            continue
    if not content:
        return []

    raw_lines: List[Tuple[float, str]] = []
    for frag in _LRC_TIMESTAMP_RE.findall(content):
        try:
            minutes, seconds, ms = int(frag[0]), int(frag[1]), int(frag[2])
            if len(frag[2]) == 2:
                ms *= 10
            timestamp = minutes * 60 + seconds + ms / 1000
            lyric = frag[3].strip()
            if lyric:
                raw_lines.append((timestamp, lyric))
        except Exception:
            continue
    if not raw_lines:
        return []

    raw_lines.sort(key=lambda x: x[0])
    multi_lines: List[Tuple[float, List[str]]] = []
    i = 0
    n = len(raw_lines)
    while i < n:
        ts0, text0 = raw_lines[i]
        langs = [text0]
        j = i + 1
        while j < n and raw_lines[j][0] - ts0 <= _LRC_GROUP_THRESHOLD:
            langs.append(raw_lines[j][1])
            j += 1
        multi_lines.append((ts0, langs))
        i = j
    return multi_lines


# ===================== 硬件检测 =====================


def _run_cmd(cmd: List[str]) -> str:
    """运行命令并返回 stdout，失败返回空字符串。"""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _detect_nvidia() -> Optional[Dict[str, Any]]:
    """检测 NVIDIA GPU。

    使用 nvidia-smi 查询:
      - GPU 名称
      - CUDA 核心数（通过 CuPy/CUDA Driver API 实时计算）
      - 显存总量/空余
      - NVENC 数量/代数
    """
    smi = _run_cmd([
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.free,count",
        "--format=csv,noheader,nounits",
    ])
    if not smi:
        return None

    try:
        lines = smi.strip().split('\n')
        if not lines:
            return None
        first = [x.strip() for x in lines[0].split(',')]
        if len(first) < 3:
            return None

        gpu_name = first[0]
        vram_total_mb = float(first[1]) if first[1] else 0
        vram_free_mb = float(first[2]) if first[2] else 0
        gpu_count = max(1, int(first[3])) if len(first) > 3 and first[3] else 1

        # 优先用 CuPy Runtime API 查询 SM 数 × 每 SM 核心数
        # 比查 GPU 名称表更准确、兼容未来显卡
        cuda_cores = _query_cuda_cores_via_cupy()
        if cuda_cores == 0:
            # CuPy 不可用时，用 CUDA Driver API 直接查询
            cuda_cores = _query_cuda_cores_via_cuda_driver()

        # NVENC 检测
        nvenc_count = gpu_count  # 每张卡 1 个 NVENC（现代显卡）
        # RTX 40/50 系列有双 NVENC 编码器
        name_upper = gpu_name.upper()
        if "RTX 40" in name_upper or "RTX 50" in name_upper or "BLACKWELL" in name_upper or "ADA" in name_upper:
            nvenc_count = 2
        # 通过 nvidia-smi 或架构名推断 NVENC 代数
        nvenc_gen = _detect_nvenc_generation(gpu_name)

        # SM 数量（从 CuPy 或 CUDA Driver API 获取）
        sm_count = _query_sm_count()

        return {
            "gpu_name": gpu_name,
            "gpu_vendor": "nvidia",
            "cuda_cores": cuda_cores,
            "sm_count": sm_count,
            "vram_total_gb": vram_total_mb / 1024,
            "vram_free_gb": vram_free_mb / 1024,
            "nvenc_count": nvenc_count,
            "nvenc_generation": nvenc_gen,
            "encoder_name": "h264_nvenc",
        }
    except Exception:
        logger.exception("NVIDIA GPU 检测失败")
        return None


# SM 核心数映射（按计算能力大版本，CUDA 规范保证永远稳定）
_SM_CORES_BY_CC = {
    (2, 0): 32,   # Fermi
    (3, 0): 192,  # Kepler
    (5, 0): 128,  # Maxwell
    (6, 0): 64,   # Pascal (GP100)
    (6, 1): 128,  # Pascal (GP10x)
    (6, 2): 128,  # Pascal (GP10x)
    (7, 0): 64,   # Volta
    (7, 5): 64,   # Turing
    (8, 0): 64,   # Ampere (A100)
    (8, 6): 128,  # Ampere (GA10x)
    (8, 7): 128,  # Ada Lovelace
    (8, 9): 128,  # Ada Lovelace
    (9, 0): 128,  # Blackwell
}


def _query_cuda_cores_via_cupy() -> int:
    """通过 CuPy Runtime API 查询 SM 数 × 每 SM 核心数。

    利用 cupy.cuda.runtime.getDeviceProperties 获取 SM 数量和计算能力，
    再按 CC 版本映射到每 SM 核心数。比查 GPU 名称表更准确、兼容未来显卡。

    Returns:
        CUDA 核心数，失败返回 0
    """
    try:
        import cupy as cp
        props = cp.cuda.runtime.getDeviceProperties(0)
        sm_count = props['multiProcessorCount']
        major = props['major']
        minor = props['minor']

        # 精确匹配 CC 版本
        key = (major, minor)
        if key in _SM_CORES_BY_CC:
            cores_per_sm = _SM_CORES_BY_CC[key]
        else:
            # 回退：按 major 版本找该系列最大已知值
            fallback = max(
                (v for (m, _), v in _SM_CORES_BY_CC.items() if m == major),
                default=128,  # 现代架构最少 128
            )
            cores_per_sm = fallback

        return sm_count * cores_per_sm
    except Exception:
        return 0


def _query_sm_count() -> int:
    """查询 GPU SM 数量（用于 CUDA 流数计算）。

    先尝试 CuPy，失败则用 CUDA Driver API。
    Returns: SM 数量，失败返回 0
    """
    try:
        import cupy as cp
        props = cp.cuda.runtime.getDeviceProperties(0)
        return props['multiProcessorCount']
    except Exception:
        pass
    try:
        import ctypes
        import sys
        if sys.platform == 'win32':
            lib = ctypes.CDLL('nvcuda.dll')
        else:
            lib = ctypes.CDLL('libcuda.so.1')
        result = lib.cuInit(0)
        if result != 0:
            return 0
        device = ctypes.c_int()
        result = lib.cuDeviceGet(ctypes.byref(device), 0)
        if result != 0:
            return 0
        sm_count = ctypes.c_int()
        result = lib.cuDeviceGetAttribute(ctypes.byref(sm_count), 16, device)
        if result != 0:
            return 0
        return sm_count.value
    except Exception:
        return 0


def _query_cuda_cores_via_cuda_driver() -> int:
    """通过 CUDA Driver API 直接查询 SM 数量 × 每 SM 核心数。

    无需 CuPy 依赖，直接调用 nvcuda.dll / libcuda.so。
    比 GPU 名称查表更准确、兼容未来显卡。

    Returns:
        CUDA 核心数，失败返回 0
    """
    try:
        import ctypes
        import sys

        if sys.platform == 'win32':
            lib = ctypes.CDLL('nvcuda.dll')
        else:
            lib = ctypes.CDLL('libcuda.so.1')

        # 初始化 CUDA Driver
        result = lib.cuInit(0)
        if result != 0:
            return 0

        # 获取设备数量
        count = ctypes.c_int()
        result = lib.cuDeviceGetCount(ctypes.byref(count))
        if result != 0 or count.value == 0:
            return 0

        # 获取第一个设备
        device = ctypes.c_int()
        result = lib.cuDeviceGet(ctypes.byref(device), 0)
        if result != 0:
            return 0

        # CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
        sm_count = ctypes.c_int()
        result = lib.cuDeviceGetAttribute(
            ctypes.byref(sm_count), 16, device,
        )
        if result != 0 or sm_count.value == 0:
            return 0

        # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
        major = ctypes.c_int()
        lib.cuDeviceGetAttribute(ctypes.byref(major), 75, device)

        # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
        minor = ctypes.c_int()
        lib.cuDeviceGetAttribute(ctypes.byref(minor), 76, device)

        # 按计算能力查找每 SM 核心数
        key = (major.value, minor.value)
        if key in _SM_CORES_BY_CC:
            cores_per_sm = _SM_CORES_BY_CC[key]
        else:
            # 回退：按 major 版本找该系列最大已知值
            fallback = max(
                (v for (m, _), v in _SM_CORES_BY_CC.items() if m == major.value),
                default=128,  # 现代架构最少 128
            )
            cores_per_sm = fallback

        return sm_count.value * cores_per_sm
    except Exception:
        return 0


def _detect_nvenc_generation(gpu_name: str) -> int:
    """检测 NVENC 代数。"""
    name_upper = gpu_name.upper()
    # Blackwell (RTX 50)
    if "RTX 50" in name_upper or "BLACKWELL" in name_upper:
        return 9
    # Ada (RTX 40)
    if "RTX 40" in name_upper or "ADA" in name_upper:
        return 8
    # Ampere (RTX 30)
    if "RTX 30" in name_upper or "AMPERE" in name_upper:
        return 7
    # Turing (RTX 20 / GTX 16)
    if "RTX 20" in name_upper or "TURING" in name_upper or "GTX 16" in name_upper:
        return 6
    # Pascal (GTX 10)
    if "GTX 10" in name_upper or "PASCAL" in name_upper:
        return 5
    return 0


def _detect_amd() -> Optional[Dict[str, Any]]:
    """检测 AMD GPU。

    通过 rocm-smi 或 Windows 注册表检测。
    AMD 不支持 CUDA 渲染，仅使用 AMF 编码。
    """
    # 尝试 rocm-smi
    rocm = _run_cmd(["rocm-smi", "--showproductinfo"])
    if rocm:
        # 简单解析 GPU 名称
        for line in rocm.split('\n'):
            if 'name' in line.lower() or 'gpu' in line.lower():
                gpu_name = line.split(':')[-1].strip()
                return {
                    "gpu_name": gpu_name or "AMD GPU",
                    "gpu_vendor": "amd",
                    "cuda_cores": 0,
                    "vram_total_gb": 0,
                    "vram_free_gb": 0,
                    "nvenc_count": 0,
                    "nvenc_generation": 0,
                    "encoder_name": "h264_amf",
                }

    # Windows 注册表检测 AMD
    try:
        import winreg
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DEVICEMAP\VIDEO",
        )
        # 尝试找 AMD 显卡
        # 简单起见，通过 D3D 或 WMI 查
        # 这里做简化处理
        winreg.CloseKey(key)
    except Exception:
        pass

    return None


def _detect_intel() -> Optional[Dict[str, Any]]:
    """检测 Intel 核显 / Arc 独显。

    Intel 不支持 CUDA 渲染，仅使用 QSV 编码。
    """
    # 尝试通过注册表或 WMI 检测
    try:
        import winreg
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Intel\Performance",
        )
        winreg.CloseKey(key)
        # 有 Intel 驱动注册表项
        return {
            "gpu_name": "Intel GPU",
            "gpu_vendor": "intel",
            "cuda_cores": 0,
            "vram_total_gb": 0,
            "vram_free_gb": 0,
            "nvenc_count": 0,
            "nvenc_generation": 0,
            "encoder_name": "h264_qsv",
        }
    except Exception:
        pass

    return None


def detect_hardware() -> HardwareInfo:
    """检测系统硬件，返回 HardwareInfo。

    检测顺序: NVIDIA → AMD → Intel → CPU fallback
    """
    # 先检测 NVIDIA
    nvidia = _detect_nvidia()
    if nvidia:
        vram_free = nvidia["vram_free_gb"]
        vram_usable = vram_free * VRAM_SAFETY_FACTOR
        cuda_cores = nvidia["cuda_cores"]
        supports_cuda = cuda_cores >= MIN_CUDA_CORES_FOR_CUDA

        if not supports_cuda:
            logger.warning(
                f"NVIDIA GPU {nvidia['gpu_name']} CUDA 核心数 ({cuda_cores}) "
                f"不足 {MIN_CUDA_CORES_FOR_CUDA}，禁用 CUDA 渲染，使用 CPU 回退"
            )

        return HardwareInfo(
            gpu_name=nvidia["gpu_name"],
            gpu_vendor="nvidia",
            cuda_cores=cuda_cores,
            sm_count=nvidia["sm_count"],
            vram_total_gb=nvidia["vram_total_gb"],
            vram_free_gb=vram_free,
            vram_usable_gb=vram_usable,
            nvenc_count=nvidia["nvenc_count"],
            nvenc_generation=nvidia["nvenc_generation"],
            has_gpu=True,
            encoder_name=nvidia["encoder_name"],
            supports_cuda_render=supports_cuda,
        )

    # 检测 AMD
    amd = _detect_amd()
    if amd:
        logger.info(f"检测到 AMD GPU: {amd['gpu_name']}，使用 AMF 编码")
        return HardwareInfo(
            gpu_name=amd["gpu_name"],
            gpu_vendor="amd",
            cuda_cores=0,
            sm_count=0,
            vram_total_gb=0,
            vram_free_gb=0,
            vram_usable_gb=0,
            nvenc_count=0,
            nvenc_generation=0,
            has_gpu=True,
            encoder_name=amd["encoder_name"],
            supports_cuda_render=False,
        )

    # 检测 Intel
    intel = _detect_intel()
    if intel:
        logger.info(f"检测到 Intel GPU: {intel['gpu_name']}，使用 QSV 编码")
        return HardwareInfo(
            gpu_name=intel["gpu_name"],
            gpu_vendor="intel",
            cuda_cores=0,
            sm_count=0,
            vram_total_gb=0,
            vram_free_gb=0,
            vram_usable_gb=0,
            nvenc_count=0,
            nvenc_generation=0,
            has_gpu=True,
            encoder_name=intel["encoder_name"],
            supports_cuda_render=False,
        )

    # 无 GPU — CPU fallback
    logger.info("未检测到可用 GPU，使用 CPU 渲染 + libx264 编码")
    return HardwareInfo(
        gpu_name="CPU (Software)",
        gpu_vendor="cpu",
        cuda_cores=0,
        sm_count=0,
        vram_total_gb=0,
        vram_free_gb=0,
        vram_usable_gb=0,
        nvenc_count=0,
        nvenc_generation=0,
        has_gpu=False,
        encoder_name="libx264",
        supports_cuda_render=False,
    )


# ===================== 并发分配 =====================


def calc_optimal_workers(
    hw: HardwareInfo, frame_states_count: int, width: int, height: int,
) -> WorkerConfig:
    """三约束取最小，算出最优并发数。

    约束:
      a) CUDA 核心数，每 stream 至少 512 核
      b) 显存，每 stream 需 overhead(0.15GB) + 1帧buffer
      c) 任务量，不超过时间区间帧数 1/10
    """
    per_frame_gb = (width * height * 4) / (1024 ** 3)

    # ---- 渲染 stream 数 ----
    if hw.supports_cuda_render and hw.cuda_cores > 0:
        max_by_cores = max(1, hw.cuda_cores // 512)
    else:
        max_by_cores = max(1, os.cpu_count() or 4)  # CPU 用 CPU 核心数

    if hw.vram_usable_gb > 0:
        max_by_vram = max(1, int(hw.vram_usable_gb / (VRAM_OVERHEAD_PER_STREAM + per_frame_gb)))
    else:
        # 无显存信息时（AMD/Intel/CPU），使用 CPU 核心数
        max_by_vram = max(1, os.cpu_count() or 4)

    max_by_task = max(1, frame_states_count // 10) if frame_states_count > 0 else 1

    render_streams = min(max_by_cores, max_by_vram, max_by_task, MAX_RENDER_STREAMS)

    # ---- 编码并发数 ----
    if hw.nvenc_count > 0:
        encode_workers = hw.nvenc_count if frame_states_count > 200 else 1
    else:
        encode_workers = 1

    # ---- 每批渲染帧数 ----
    if hw.vram_usable_gb > 0 and per_frame_gb > 0:
        batch_size = min(int(hw.vram_usable_gb / per_frame_gb), frame_states_count)
    else:
        batch_size = min(100, frame_states_count)  # CPU 无显存限制，但设合理上限

    # ---- 渲染模式判定 ----
    total_frame_volume = frame_states_count * per_frame_gb
    if hw.vram_usable_gb > 0 and total_frame_volume <= hw.vram_usable_gb:
        default_mode = "batch"  # 全缓存一次编码
    else:
        default_mode = "stream"  # 分批渲染边编边码

    return WorkerConfig(
        render_streams=render_streams,
        encode_workers=encode_workers,
        batch_size=batch_size,
        default_mode=default_mode,
        per_frame_gb=per_frame_gb,
    )


# ===================== CPU 预计算 =====================


def build_ust_info_for_render(ust_info: dict, width: int, height: int) -> dict:
    """为渲染器扩展 ust_info，添加屏幕尺寸和字体信息。"""
    ps = ust_info["player_style"]
    info = dict(ust_info)
    info["_render_width"] = width
    info["_render_height"] = height
    info["_render_fonts"] = _init_render_fonts(ps, width, height)
    return info


def _init_render_fonts(ps: dict, width: int, height: int) -> dict:
    """初始化渲染字体（与播放器 _init_fonts 一致）。"""
    wff = ps.get("word_lyric_font_family", "等线")
    iff = ps.get("info_font_family", "微软雅黑")
    note_fs = max(int(height * 2 / 3 * 0.4), 50)
    lyric_fs = max(int(height * 0.03), 10)
    ust_lyric_fs = max(int(height * 2 / 3 * 0.2), 80)

    fonts = {}
    fonts["note_font"] = QFont(iff, note_fs, QFont.Weight.Bold)
    fonts["lyric_font"] = QFont(iff, lyric_fs)
    fonts["ust_lyric_font"] = QFont(wff, ust_lyric_fs, QFont.Weight.Bold)
    fonts["small_font"] = QFont(iff, 14)
    fonts["bold_small_font"] = QFont(iff, 14, QFont.Weight.Bold)
    fonts["copyright_font"] = QFont(iff, 12)

    # 缓存 QFontMetrics
    fonts["fm_note"] = QFontMetrics(fonts["note_font"])
    fonts["fm_lyric"] = QFontMetrics(fonts["lyric_font"])
    fonts["fm_ust_lyric"] = QFontMetrics(fonts["ust_lyric_font"])
    fonts["fm_small"] = QFontMetrics(fonts["small_font"])
    fonts["fm_copyright"] = QFontMetrics(fonts["copyright_font"])

    return fonts


def precompute_frame_states(
    ust_info: dict, fps: int, width: int, height: int,
) -> List[FrameState]:
    """CPU 预计算：遍历所有音符 + LRC 变化点，生成 FrameState 列表。

    Args:
        ust_info: build_ust_info() 输出
        fps: 输出帧率
        width, height: 输出分辨率

    Returns:
        时间区间 FrameState 列表（不做视觉去重，转音完整保留）
    """
    notes = ust_info.get("notes", [])
    tempo = ust_info.get("tempo", 120)
    sc = ust_info.get("show_config", {})
    ps = ust_info.get("player_style", {})
    pi = ust_info.get("project_info", {})

    tick_per_second = (tempo * 480) / 60

    # 计算总 tick
    if notes and "position" in notes[0]:
        total_tick = max(
            n.get("position", 0) + max(n.get("length", 480), 1) for n in notes
        )
    else:
        total_tick = sum(max(n.get("length", 480), 1) for n in notes)

    note_tick_ranges = _calc_note_tick_ranges(notes)

    # 显示开关
    show_lyric = sc.get("lyric", True)
    show_lyric_autohide = sc.get("lyric_autohide", True)
    lyric_autohide_threshold = sc.get("lyric_autohide_threshold", 3.0)
    curve_show = sc.get("curve_show", False)
    show_bpm = sc.get("bpm", True)
    show_play_time = sc.get("play_time", True)
    show_song_name = sc.get("song_name", True)
    show_song_author = sc.get("song_author", True)
    show_ust_author = sc.get("ust_author", True)
    show_copyright = sc.get("copyright", True)

    # 播放器样式
    lyric_pos = ps.get("lyric_pos", "上")
    silent_display = ps.get("silent_display", "R")
    silent_custom = ps.get("silent_custom_text", "")
    end_display = ps.get("end_display", "END")
    end_custom = ps.get("end_custom_text", "")
    pitch_placeholder = ps.get("pitch_placeholder", "无")
    pitch_custom = ps.get("pitch_custom_text", "")
    global_bg_enabled = ps.get("global_bg_enabled", False)
    global_bg_color = ps.get("global_bg_color", "#00ff00")
    styles = ps.get("styles", [])
    note_styles = ps.get("note_styles", {})

    # 项目信息
    song_name = pi.get("song_name", "")
    song_author = pi.get("song_author", "")
    ust_author = pi.get("ust_author", "")

    # 字体
    wff = ps.get("word_lyric_font_family", "等线")
    iff = ps.get("info_font_family", "微软雅黑")
    small_font_color = ps.get("info_text_color", "#ffffff")

    # 版权
    copyright_text = f"ustxPlayer-preview - {APP_VERSION} © 2026 SYEternalR"

    # LRC 歌词
    lrc_path = ps.get("lrc_path", "")
    multi_lrc = parse_lrc_file(lrc_path) if show_lyric and lrc_path else []

    # 预计算 LRC 隐藏区间
    lrc_hide_intervals: List[Tuple[float, float]] = []
    if show_lyric and show_lyric_autohide and notes:
        ticks_to_sec = 1.0 / tick_per_second
        threshold = lyric_autohide_threshold
        ns = [(n.get('position', 0), n.get('position', 0) + n.get('length', 0)) for n in notes]
        for i in range(len(ns) - 1):
            gap = (ns[i+1][0] - ns[i][1]) * ticks_to_sec
            if gap > threshold:
                lrc_hide_intervals.append((
                    ns[i][1] * ticks_to_sec + threshold,
                    ns[i+1][0] * ticks_to_sec,
                ))
        lrc_hide_intervals.append((ns[-1][1] * ticks_to_sec + threshold, float('inf')))

    # 初始化字体
    fonts = _init_render_fonts(ps, width, height)
    cx, cy = width // 2, height // 2

    # 收集所有时间变化点
    # 1) 每个音符的开始时间
    # 2) 每个 LRC 行的时间
    # 3) 结尾
    time_points = set()

    # 音符变化点
    for i, note in enumerate(note_tick_ranges):
        start_tick = note[0]
        start_sec = start_tick / tick_per_second
        time_points.add(start_sec)

    # LRC 变化点
    for ts, _ in multi_lrc:
        time_points.add(ts)

    # 结尾
    end_sec = total_tick / tick_per_second
    time_points.add(end_sec)

    # 排序
    sorted_times = sorted(time_points)

    # 构建每个时间段的状态
    states: List[FrameState] = []
    last_valid_lyric = ""

    # 不做视觉去重：每个时间区间独立生成一帧。
    # 用户要求「存多少帧就渲染多少帧」——自动去重合并会把
    # 转音曲线（portamento）不同的相邻区间错误合并，导致转音丢失。
    for i, start_time in enumerate(sorted_times):
        if i + 1 >= len(sorted_times):
            break
        next_time = sorted_times[i + 1]
        duration = next_time - start_time
        if duration <= 0:
            continue

        frame_count = max(1, round(duration * fps))
        current_tick = start_time * tick_per_second

        # 匹配当前音符
        current_note = _find_note_at_tick(note_tick_ranges, current_tick)
        is_finished = current_tick >= total_tick

        # 歌词
        if is_finished:
            lyric_text = get_end_text(end_display, end_custom)
            note_name = ""
        elif current_note:
            raw_lyric = current_note.get("lyric", "")
            note_num = current_note.get("note_num", 0)
            if raw_lyric == "R":
                lyric_text = get_silent_text(silent_display, silent_custom)
                note_name = ""
            elif raw_lyric in ("-", "+"):
                lyric_text = last_valid_lyric or get_silent_text(silent_display, silent_custom)
                note_name = get_pitch_text(note_num, pitch_placeholder, pitch_custom)
            else:
                lyric_text = raw_lyric
                last_valid_lyric = raw_lyric
                note_name = get_pitch_text(note_num, pitch_placeholder, pitch_custom)
        else:
            lyric_text = get_silent_text(silent_display, silent_custom)
            note_name = ""

        # 背景色
        if is_finished or not current_note or note_name == "":
            si = 0  # 静默/结尾用样式1
        else:
            note_idx = current_note.get("index", 0)
            try:
                note_idx_int = int(note_idx) if isinstance(note_idx, str) else int(note_idx)
            except (ValueError, TypeError):
                note_idx_int = 0
            si = note_styles.get(note_idx_int, 0)

        if global_bg_enabled:
            bg_color = global_bg_color
            bg_hex = validate_hex_color(bg_color)
        else:
            if si < len(styles):
                bg_hex = validate_hex_color(styles[si].get("bg_color", "#000000"))
            else:
                bg_hex = validate_hex_color(ps.get("bg_color", "#000000"))

        # 歌词/音名颜色
        if si < len(styles):
            lyric_rgb = hex_to_rgb(validate_hex_color(styles[si].get("lyric_color", "#ffffff")))
            note_rgb = hex_to_rgb(validate_hex_color(styles[si].get("note_color", "#6c6c6c")))
            pitch_curve_hex = validate_hex_color(styles[si].get("pitch_curve_color", "#ffffff"))
        else:
            lyric_rgb = hex_to_rgb(validate_hex_color(ps.get("lyric_color", "#ffffff")))
            note_rgb = hex_to_rgb(validate_hex_color(ps.get("note_color", "#c3c3c3")))
            pitch_curve_hex = validate_hex_color(ps.get("pitch_curve_color", "#ffffff"))

        # 音高线
        pitch_points: List[Tuple[float, float]] = []
        if curve_show and current_note and note_name:
            pb_data = current_note.get("pitch_bend", [])
            note_length = current_note.get("length", 0)
            if pb_data and len(pb_data) >= 2 and note_length > 0:
                curve_width = note_length
                start_x = cx - curve_width // 2
                pb_count = len(pb_data)
                safe_top, safe_bottom = 100, height - 100
                for j in range(pb_count):
                    x = start_x + (j / (pb_count - 1)) * curve_width
                    y = cy - (pb_data[j] / 100) * (height * 0.09)
                    if y < safe_top:
                        exceed = safe_top - y
                        y = safe_top - exceed * max(0.3, 1 - (exceed / height * 2))
                    elif y > safe_bottom:
                        exceed = y - safe_bottom
                        y = safe_bottom + exceed * max(0.3, 1 - (exceed / height * 2))
                    y = max(50, min(y, height - 50))
                    pitch_points.append((x, y))

        # LRC 歌词
        lrc_lines: List[str] = []
        lrc_hidden = False
        if show_lyric and multi_lrc:
            current_lrc_idx = -1
            for li, (ts, _) in enumerate(multi_lrc):
                if ts <= start_time:
                    current_lrc_idx = li
                else:
                    break
            if 0 <= current_lrc_idx < len(multi_lrc):
                lrc_lines = multi_lrc[current_lrc_idx][1]
                if lrc_lines:
                    lrc_hidden = show_lyric_autohide and any(
                        s <= start_time <= e for s, e in lrc_hide_intervals
                    )

        # 构建唯一标识（不用于去重，仅用于调试/日志）
        cache_key = f"t{start_time:.3f}"

        state = FrameState(
            lyric=lyric_text,
            note_name=note_name,
            bg_color=bg_hex,
            lyric_color=lyric_rgb,
            note_color=note_rgb,
            pitch_curve_color=pitch_curve_hex,
            pitch_points=pitch_points,
            show_lyric=show_lyric,
            show_note_name=bool(note_name),
            show_curve=curve_show,
            song_name=song_name,
            song_author=song_author,
            ust_author=ust_author,
            tempo=tempo,
            show_bpm=show_bpm,
            show_song_name=show_song_name,
            show_song_author=show_song_author,
            show_ust_author=show_ust_author,
            show_copyright=show_copyright,
            show_play_time=show_play_time,
            lrc_lines=lrc_lines,
            lrc_hidden=lrc_hidden,
            lyric_pos=lyric_pos,
            word_lyric_font_family=wff,
            info_font_family=iff,
            small_font_color=small_font_color,
            copyright_text=copyright_text,
            start_time=start_time,
            duration=duration,
            frame_count=frame_count,
            is_duplicate=False,
            cache_key=cache_key,
        )
        states.append(state)

    logger.info(
        f"预计算完成: {len(states)} 个时间区间 "
        f"(原始 {len(sorted_times)} 个时间点, "
        f"总输出帧 {sum(s.frame_count for s in states)})"
    )
    return states


def _calc_note_tick_ranges(
    notes: List[dict],
) -> List[Tuple[int, int, dict]]:
    """计算每个音符的 tick 区间。"""
    ranges = []
    if notes and "position" in notes[0]:
        for note in notes:
            pos = note.get("position", 0)
            length = max(note.get("length", 480), 1)
            ranges.append((pos, pos + length, note))
    else:
        current_tick = 0
        for note in notes:
            length = max(note.get("length", 480), 1)
            ranges.append((current_tick, current_tick + length, note))
            current_tick += length
    return ranges


def _find_note_at_tick(
    ranges: List[Tuple[int, int, dict]], tick: float,
) -> Optional[dict]:
    """在 tick 位置查找匹配的音符。"""
    for r in ranges:
        if r[0] <= tick < r[1]:
            return r[2]
    return None


# ===================== CPU 渲染后端 =====================


def _draw_with_painter(
    painter: QPainter, state: FrameState, width: int, height: int, fonts: dict,
) -> None:
    """通用 QPainter 绘制逻辑（CPU 和 GLES 后端共用）。"""
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    cx, cy = width // 2, height // 2

    # ---- 音名 ----
    if state.show_note_name and state.note_name:
        note_c = QColor(*state.note_color)
        note_c.setAlpha(NOTE_ALPHA)
        painter.setPen(note_c)
        painter.setFont(fonts["note_font"])
        fm = fonts["fm_note"]
        tw = fm.horizontalAdvance(state.note_name)
        th = fm.height()
        pad = th * 0.2
        painter.drawText(
            QRectF(cx - tw / 2 - pad, cy - th / 2 - pad,
                   tw + pad * 2, th + pad * 2),
            Qt.AlignmentFlag.AlignCenter, state.note_name,
        )

    # ---- 音高线 ----
    if state.show_curve and state.pitch_points and len(state.pitch_points) >= 2:
        pen = QPen(QColor(state.pitch_curve_color))
        pen.setWidth(NOTE_LINE_WIDTH)
        painter.setPen(pen)
        points = [QPointF(x, y) for x, y in state.pitch_points]
        painter.drawPolyline(QPolygonF(points))

    # ---- 歌词 ----
    if state.show_lyric and state.lyric:
        lyric_c = QColor(*state.lyric_color)
        painter.setPen(lyric_c)
        painter.setFont(fonts["ust_lyric_font"])
        tw = fonts["fm_ust_lyric"].horizontalAdvance(state.lyric)
        th = fonts["fm_ust_lyric"].height()
        pad = th * 0.2
        painter.drawText(
            QRectF(cx - tw / 2 - pad, cy - th / 2 - pad,
                   tw + pad * 2, th + pad * 2),
            Qt.AlignmentFlag.AlignCenter, state.lyric,
        )

    # ---- 左上角信息 ----
    painter.setPen(QColor(state.small_font_color))
    y_off = 20
    if state.show_song_name and state.song_name:
        painter.setFont(fonts["bold_small_font"])
        painter.drawText(20, y_off + 14, state.song_name)
        painter.setFont(fonts["small_font"])
        y_off += 27
    if state.show_song_author and state.song_author:
        painter.drawText(20, y_off + 14, state.song_author)
        y_off += 25
    if state.show_ust_author and state.ust_author:
        painter.drawText(20, y_off + 14, state.ust_author)

    # BPM（右上角）
    if state.show_bpm:
        painter.setFont(fonts["small_font"])
        bpm_text = f"BPM={state.tempo}"
        bpm_w = fonts["fm_small"].horizontalAdvance(bpm_text)
        painter.drawText(width - 20 - bpm_w, 34, bpm_text)

    # LRC 歌词
    if state.show_lyric and state.lrc_lines and not state.lrc_hidden:
        anchor_y = int(height * 0.3) if state.lyric_pos == "上" else int(height * 0.7)
        painter.setFont(fonts["lyric_font"])
        line_h = fonts["fm_lyric"].height()
        step = line_h * 1.3
        n = len(state.lrc_lines)
        if state.lyric_pos == "上":
            top_baseline = anchor_y - (n - 1) * step
        else:
            top_baseline = anchor_y
        for li, text in enumerate(state.lrc_lines):
            baseline = int(top_baseline + li * step)
            text_w = fonts["fm_lyric"].horizontalAdvance(text)
            painter.drawText(width // 2 - text_w // 2, baseline, text)

    # 版权（底部居中）
    if state.show_copyright:
        copy_c = QColor(195, 195, 195)
        copy_c.setAlpha(COPYRIGHT_ALPHA)
        painter.setPen(copy_c)
        painter.setFont(fonts["copyright_font"])
        copy_w = fonts["fm_copyright"].horizontalAdvance(state.copyright_text)
        painter.drawText(width // 2 - copy_w // 2, height - 20, state.copyright_text)


def _render_frame_cpu(
    state: FrameState, width: int, height: int, fonts: dict,
) -> QImage:
    """使用 QPainter 在 QImage 上渲染一帧（CPU 渲染）。"""
    image = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(QColor(state.bg_color))

    painter = QPainter(image)
    try:
        _draw_with_painter(painter, state, width, height, fonts)
    finally:
        painter.end()

    return image


# ===================== CUDA 渲染后端 =====================


# 全局字形纹理缓存: key -> (cupy_array, width, height)
_GLYPH_CACHE: Dict[str, Tuple[Any, int, int]] = {}
_GLYPH_CACHE_LOCK = threading.Lock()


def _clear_glyph_cache():
    """清空字形缓存（每次渲染前调用，释放显存）。"""
    with _GLYPH_CACHE_LOCK:
        _GLYPH_CACHE.clear()
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass


def clear_renderer_cache():
    """释放渲染器模块级缓存，减少内存占用。

    渲染导出完成后或页面切换时调用，释放字形缓存、时间戳纹理、
    CUDA 上下文、GLES 上下文等。
    """
    _clear_glyph_cache()
    _clear_timestamp_cache()
    _clear_cuda_contexts()
    _clear_gles_context()
    # 清理渲染错误缓存
    clear_last_render_error()
    logger.debug("渲染器模块级缓存已释放")


def _rgba_to_nv12_gpu(
    rgba: Any, width: int, height: int,
) -> Any:
    """在 GPU 上将 RGBA (CuPy, H×W×4, BGRA in memory) 转为 NV12 字节流。

    NV12 布局: [Y: W×H bytes] + [UV: (H/2)×W bytes] (U/V 交错)
    转换公式 (BT.601):
        Y  = 0.299*R + 0.587*G + 0.114*B
        U  = -0.169*R - 0.331*G + 0.500*B + 128
        V  = 0.500*R - 0.419*G - 0.081*B + 128
    色度使用 [::2,::2] 降采样（最快路径，合成视频可接受）。
    """
    import cupy as cp
    # Qt ARGB32 in memory = BGRA: B=0, G=1, R=2, A=3
    b = rgba[:, :, 0].astype(cp.float32)
    g = rgba[:, :, 1].astype(cp.float32)
    r = rgba[:, :, 2].astype(cp.float32)

    # Y 平面
    y = (0.299 * r + 0.587 * g + 0.114 * b).astype(cp.uint8)

    # UV 平面（2:1 降采样，用 [::2,::2] 取每 2×2 块左上角像素）
    r_sub = r[::2, ::2]
    g_sub = g[::2, ::2]
    b_sub = b[::2, ::2]

    u = cp.clip(-0.169 * r_sub - 0.331 * g_sub + 0.500 * b_sub + 128, 0, 255).astype(cp.uint8)
    v = cp.clip(0.500 * r_sub - 0.419 * g_sub - 0.081 * b_sub + 128, 0, 255).astype(cp.uint8)

    # UV 交错: (H/2) × W, 偶数列为 U, 奇数列为 V
    uv = cp.empty((height // 2, width), dtype=cp.uint8)
    uv[:, 0::2] = u
    uv[:, 1::2] = v

    # 拼接 Y + UV → 一维字节数组
    nv12 = cp.concatenate([y.ravel(), uv.ravel()])
    return nv12


def _get_glyph_texture(
    text: str, font: QFont, color: QColor, fm: QFontMetrics,
) -> Tuple[Any, int, int]:
    """渲染文字到 GPU 纹理（缓存的）。

    用 QPainter 在小 QImage 上渲染文字 → 转 numpy → 上传 CuPy 数组。
    相同 (text, font, color) 只渲染一次，后续直接复用 GPU 纹理。
    """
    if not text:
        return None, 0, 0

    key = f"{text}|{font.family()}|{font.pointSize()}|{font.weight()}|{color.rgba()}"
    with _GLYPH_CACHE_LOCK:
        if key in _GLYPH_CACHE:
            return _GLYPH_CACHE[key]

    try:
        import cupy as cp
        import numpy as np
    except ImportError:
        return None, 0, 0

    tw = max(1, fm.horizontalAdvance(text))
    th = max(1, fm.height())

    img = QImage(tw, th, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)
    painter = QPainter(img)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setFont(font)
    painter.setPen(color)
    painter.drawText(0, fm.ascent(), text)
    painter.end()

    # QImage → numpy → cupy
    # PySide6 的 bits() 返回 memoryview，直接用 np.asarray 转换
    ptr = img.bits()
    np_arr = np.frombuffer(ptr, dtype=np.uint8).reshape((th, tw, 4))
    gpu_arr = cp.asarray(np_arr)

    with _GLYPH_CACHE_LOCK:
        _GLYPH_CACHE[key] = (gpu_arr, tw, th)
    return _GLYPH_CACHE[key]


def _blit_texture(
    fb: Any, tex: Any, tw: int, th: int,
    x0: int, y0: int, width: int, height: int,
) -> None:
    """将纹理 blit 到帧缓冲的指定位置（GPU 操作）。

    Alpha 混合：纹理 alpha > 0 的像素覆盖帧缓冲。
    """
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(width, x0 + tw)
    y1 = min(height, y0 + th)
    if x1 <= x0 or y1 <= y0:
        return

    sw = x1 - x0
    sh = y1 - y0
    src = tex[:sh, :sw]
    dst = fb[y0:y1, x0:x1]

    alpha_mask = src[:, :, 3] > 0
    dst[alpha_mask] = src[alpha_mask]


def _cuda_draw_polyline(
    fb: Any, points: List[Tuple[float, float]],
    color: QColor, line_width: int, width: int, height: int,
) -> None:
    """在 GPU 帧缓冲上画折线（全向量化，无 Python 循环）。"""
    import cupy as cp
    import numpy as np

    r = color.red()
    g = color.green()
    b = color.blue()
    half_w = line_width // 2

    # 收集所有线段的所有像素坐标，一次性写入 GPU
    all_x = []
    all_y = []
    for i in range(len(points) - 1):
        x0 = int(points[i][0])
        y0 = int(points[i][1])
        x1 = int(points[i + 1][0])
        y1 = int(points[i + 1][1])
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        xs = np.linspace(x0, x1, steps + 1).astype(np.int32)
        ys = np.linspace(y0, y1, steps + 1).astype(np.int32)
        # 线宽展开：每个点向四周偏移 half_w
        for ox in range(-half_w, half_w + 1):
            for oy in range(-half_w, half_w + 1):
                all_x.append(xs + ox)
                all_y.append(ys + oy)

    if not all_x:
        return

    # 合并所有坐标 → 一次性裁剪 + 写入 GPU
    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    xi = cp.clip(cp.asarray(all_x), 0, width - 1)
    yi = cp.clip(cp.asarray(all_y), 0, height - 1)
    fb[yi, xi, 0] = b
    fb[yi, xi, 1] = g
    fb[yi, xi, 2] = r
    fb[yi, xi, 3] = 255


class _CudaRenderContext:
    """每线程独立的 CUDA 渲染上下文（stream + 复用帧缓冲）。

    多线程渲染时每个线程持有自己的 stream 和 fb：
      - 多 stream 可真正并行执行 GPU 操作（不同 stream 不互斥）
      - fb 复用避免每帧分配 8MB 显存
    """

    def __init__(self, width: int, height: int):
        import cupy as cp
        self.width = width
        self.height = height
        self.stream = cp.cuda.Stream()
        self.fb = cp.zeros((height, width, 4), dtype=cp.uint8)
        # 预填充一次 alpha=255（背景颜色每帧重写，alpha 恒为 255）
        self.fb[:, :, 3] = 255

    def matches(self, width: int, height: int) -> bool:
        return self.width == width and self.height == height


_CUDA_LOCAL = threading.local()


def _get_cuda_ctx(width: int, height: int) -> "_CudaRenderContext":
    """获取当前线程的 CUDA 渲染上下文（懒创建 + 分辨率变化重建）。"""
    ctx = getattr(_CUDA_LOCAL, "ctx", None)
    if ctx is None or not ctx.matches(width, height):
        ctx = _CudaRenderContext(width, height)
        _CUDA_LOCAL.ctx = ctx
    return ctx


def _clear_cuda_contexts():
    """清空所有线程的 CUDA 渲染上下文（渲染结束后释放显存）。"""
    _CUDA_LOCAL.__dict__.pop("ctx", None)
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
    except ImportError:
        pass


def _render_frame_cuda(
    state: FrameState, width: int, height: int, fonts: dict,
) -> QImage:
    """CUDA 后端渲染：每线程独立 stream + 复用 GPU 帧缓冲。

    GPU 负责：背景填充、纹理 blit（alpha 混合）、折线绘制。
    多线程各自在独立 stream 上执行，可真正并行。
    每帧 CPU 仅做 GPU→CPU 下载 + 交给保存线程写盘。
    """
    try:
        import cupy as cp
        import numpy as np
    except ImportError:
        return _render_frame_cpu(state, width, height, fonts)

    try:
        ctx = _get_cuda_ctx(width, height)
        fb = ctx.fb
        stream = ctx.stream

        with stream:
            # ---- 重置帧缓冲（复用，不重新分配） ----
            bg = QColor(state.bg_color)
            fb[:, :, 0] = bg.blue()   # Qt ARGB32 = BGRA in memory
            fb[:, :, 1] = bg.green()
            fb[:, :, 2] = bg.red()

            cx, cy = width // 2, height // 2

            # ---- 音名（居中） ----
            if state.show_note_name and state.note_name:
                note_c = QColor(*state.note_color)
                note_c.setAlpha(NOTE_ALPHA)
                tex, tw, th = _get_glyph_texture(
                    state.note_name, fonts["note_font"], note_c, fonts["fm_note"],
                )
                if tex is not None:
                    x0 = cx - tw // 2
                    y0 = cy - th // 2
                    _blit_texture(fb, tex, tw, th, x0, y0, width, height)

            # ---- 音高线（GPU 画线） ----
            if state.show_curve and state.pitch_points and len(state.pitch_points) >= 2:
                _cuda_draw_polyline(
                    fb, state.pitch_points,
                    QColor(state.pitch_curve_color),
                    NOTE_LINE_WIDTH, width, height,
                )

            # ---- 歌词（居中） ----
            if state.show_lyric and state.lyric:
                lyric_c = QColor(*state.lyric_color)
                tex, tw, th = _get_glyph_texture(
                    state.lyric, fonts["ust_lyric_font"], lyric_c, fonts["fm_ust_lyric"],
                )
                if tex is not None:
                    x0 = cx - tw // 2
                    y0 = cy - th // 2
                    _blit_texture(fb, tex, tw, th, x0, y0, width, height)

            # ---- 左上角信息 ----
            small_c = QColor(state.small_font_color)
            y_off = 20
            if state.show_song_name and state.song_name:
                tex, tw, th = _get_glyph_texture(
                    state.song_name, fonts["bold_small_font"], small_c, fonts["fm_small"],
                )
                if tex is not None:
                    _blit_texture(fb, tex, tw, th, 20, y_off, width, height)
                    y_off += 27
            if state.show_song_author and state.song_author:
                tex, tw, th = _get_glyph_texture(
                    state.song_author, fonts["small_font"], small_c, fonts["fm_small"],
                )
                if tex is not None:
                    _blit_texture(fb, tex, tw, th, 20, y_off, width, height)
                    y_off += 25
            if state.show_ust_author and state.ust_author:
                tex, tw, th = _get_glyph_texture(
                    state.ust_author, fonts["small_font"], small_c, fonts["fm_small"],
                )
                if tex is not None:
                    _blit_texture(fb, tex, tw, th, 20, y_off, width, height)

            # ---- BPM（右上角） ----
            if state.show_bpm:
                bpm_text = f"BPM={state.tempo}"
                tex, tw, th = _get_glyph_texture(
                    bpm_text, fonts["small_font"], small_c, fonts["fm_small"],
                )
                if tex is not None:
                    _blit_texture(fb, tex, tw, th, width - 20 - tw, 20, width, height)

            # ---- LRC 歌词 ----
            if state.show_lyric and state.lrc_lines and not state.lrc_hidden:
                anchor_y = int(height * 0.3) if state.lyric_pos == "上" else int(height * 0.7)
                line_h = fonts["fm_lyric"].height()
                step = line_h * 1.3
                n = len(state.lrc_lines)
                if state.lyric_pos == "上":
                    top_baseline = anchor_y - (n - 1) * step
                else:
                    top_baseline = anchor_y
                lrc_c = QColor(state.small_font_color)
                for li, text in enumerate(state.lrc_lines):
                    baseline = int(top_baseline + li * step)
                    tex, tw, th = _get_glyph_texture(
                        text, fonts["lyric_font"], lrc_c, fonts["fm_lyric"],
                    )
                    if tex is not None:
                        x0 = width // 2 - tw // 2
                        y0 = baseline - fonts["fm_lyric"].ascent()
                        _blit_texture(fb, tex, tw, th, x0, y0, width, height)

            # ---- 版权（底部居中） ----
            if state.show_copyright:
                copy_c = QColor(195, 195, 195)
                copy_c.setAlpha(COPYRIGHT_ALPHA)
                tex, tw, th = _get_glyph_texture(
                    state.copyright_text, fonts["copyright_font"], copy_c, fonts["fm_copyright"],
                )
                if tex is not None:
                    x0 = width // 2 - tw // 2
                    y0 = height - 20 - th
                    _blit_texture(fb, tex, tw, th, x0, y0, width, height)

        # stream 同步（等待 GPU 完成）→ 下载到 CPU
        stream.synchronize()

        # ====== GPU 端 RGBA→NV12 转换（消除 FFmpeg CPU 颜色转换瓶颈） ======
        # 使用 CuPy 在 GPU 上直接转换，然后只下载 NV12 数据（3.1MB vs 8MB）
        nv12_gpu = _rgba_to_nv12_gpu(fb, width, height)
        nv12_cpu = cp.asnumpy(nv12_gpu)

        result_np = cp.asnumpy(fb)
        result_img = QImage(
            result_np.data, width, height, width * 4,
            QImage.Format.Format_ARGB32_Premultiplied,
        )
        # 保持 numpy 引用存活（QImage 包装不拷贝数据，保存 PNG 时需要底层缓冲有效）
        result_img._numpy_ref = result_np
        # 存储 NV12 字节供编码线程直接使用（跳过 FFmpeg 颜色转换）
        result_img._nv12_bytes = nv12_cpu.tobytes()
        return result_img

    except Exception:
        logger.exception("CUDA 渲染失败，回退 CPU")
        return _render_frame_cpu(state, width, height, fonts)


def _render_frame_nv12(
    state: FrameState, width: int, height: int, fonts: dict,
) -> Optional[bytes]:
    """CUDA 渲染帧，直接返回 NV12 字节（跳过 QImage 创建，节省 8MB RGBA 下载）。

    与 _render_frame_cuda 的区别：
        - 不创建 QImage（省去 8MB RGBA 的 GPU→CPU 下载 + 内存分配）
        - 只返回 NV12 字节（3.1MB）
        - 适合纯编码流水线使用

    Returns:
        NV12 字节流，失败时返回 None
    """
    try:
        import cupy as cp
    except ImportError:
        return None

    try:
        ctx = _get_cuda_ctx(width, height)
        fb = ctx.fb
        stream = ctx.stream

        with stream:
            # ---- 重置帧缓冲 ----
            bg = QColor(state.bg_color)
            fb[:, :, 0] = bg.blue()
            fb[:, :, 1] = bg.green()
            fb[:, :, 2] = bg.red()

            cx, cy = width // 2, height // 2

            # ---- 音名 ----
            if state.show_note_name and state.note_name:
                note_c = QColor(*state.note_color)
                note_c.setAlpha(NOTE_ALPHA)
                tex, tw, th = _get_glyph_texture(
                    state.note_name, fonts["note_font"], note_c, fonts["fm_note"],
                )
                if tex is not None:
                    _blit_texture(fb, tex, tw, th, cx - tw // 2, cy - th // 2, width, height)

            # ---- 音高线 ----
            if state.show_curve and state.pitch_points and len(state.pitch_points) >= 2:
                _cuda_draw_polyline(
                    fb, state.pitch_points,
                    QColor(state.pitch_curve_color),
                    NOTE_LINE_WIDTH, width, height,
                )

            # ---- 歌词 ----
            if state.show_lyric and state.lyric:
                lyric_c = QColor(*state.lyric_color)
                tex, tw, th = _get_glyph_texture(
                    state.lyric, fonts["ust_lyric_font"], lyric_c, fonts["fm_ust_lyric"],
                )
                if tex is not None:
                    _blit_texture(fb, tex, tw, th, cx - tw // 2, cy - th // 2, width, height)

            # ---- 左上角信息 ----
            small_c = QColor(state.small_font_color)
            y_off = 20
            if state.show_song_name and state.song_name:
                tex, tw, th = _get_glyph_texture(
                    state.song_name, fonts["bold_small_font"], small_c, fonts["fm_small"],
                )
                if tex is not None:
                    _blit_texture(fb, tex, tw, th, 20, y_off, width, height)
                    y_off += 27
            if state.show_song_author and state.song_author:
                tex, tw, th = _get_glyph_texture(
                    state.song_author, fonts["small_font"], small_c, fonts["fm_small"],
                )
                if tex is not None:
                    _blit_texture(fb, tex, tw, th, 20, y_off, width, height)
                    y_off += 25
            if state.show_ust_author and state.ust_author:
                tex, tw, th = _get_glyph_texture(
                    state.ust_author, fonts["small_font"], small_c, fonts["fm_small"],
                )
                if tex is not None:
                    _blit_texture(fb, tex, tw, th, 20, y_off, width, height)

            # ---- BPM ----
            if state.show_bpm:
                bpm_text = f"BPM={state.tempo}"
                tex, tw, th = _get_glyph_texture(
                    bpm_text, fonts["small_font"], small_c, fonts["fm_small"],
                )
                if tex is not None:
                    _blit_texture(fb, tex, tw, th, width - 20 - tw, 20, width, height)

            # ---- LRC 歌词 ----
            if state.show_lyric and state.lrc_lines and not state.lrc_hidden:
                anchor_y = int(height * 0.3) if state.lyric_pos == "上" else int(height * 0.7)
                line_h = fonts["fm_lyric"].height()
                step = line_h * 1.3
                n = len(state.lrc_lines)
                if state.lyric_pos == "上":
                    top_baseline = anchor_y - (n - 1) * step
                else:
                    top_baseline = anchor_y
                lrc_c = QColor(state.small_font_color)
                for li, text in enumerate(state.lrc_lines):
                    baseline = int(top_baseline + li * step)
                    tex, tw, th = _get_glyph_texture(
                        text, fonts["lyric_font"], lrc_c, fonts["fm_lyric"],
                    )
                    if tex is not None:
                        x0 = width // 2 - tw // 2
                        y0 = baseline - fonts["fm_lyric"].ascent()
                        _blit_texture(fb, tex, tw, th, x0, y0, width, height)

            # ---- 版权 ----
            if state.show_copyright:
                copy_c = QColor(195, 195, 195)
                copy_c.setAlpha(COPYRIGHT_ALPHA)
                tex, tw, th = _get_glyph_texture(
                    state.copyright_text, fonts["copyright_font"], copy_c, fonts["fm_copyright"],
                )
                if tex is not None:
                    x0 = width // 2 - tw // 2
                    y0 = height - 20 - th
                    _blit_texture(fb, tex, tw, th, x0, y0, width, height)

        # stream 同步（等待 GPU 完成）→ 只下载 NV12（3.1MB，跳过 8MB RGBA）
        stream.synchronize()
        nv12_gpu = _rgba_to_nv12_gpu(fb, width, height)
        nv12_cpu = cp.asnumpy(nv12_gpu)
        return nv12_cpu.tobytes()

    except Exception:
        logger.exception("CUDA→NV12 渲染失败")
        return None


def get_cuda_render_status() -> Tuple[bool, str]:
    """检测 CUDA 渲染运行环境，返回 (是否可用, 不可用原因/启用提示)。"""
    try:
        import cupy  # noqa: F401
        test = cupy.zeros((1,), dtype=cupy.float32)
        del test
        return True, ""
    except ImportError:
        return False, "未安装 cupy 库：运行 `pip install cupy-cuda12x` 后重启应用即可启用"
    except Exception as e:
        return False, f"cupy 初始化失败：{e}"


def _check_cuda_available() -> bool:
    """检查 CUDA 运行环境是否可用。"""
    ok, reason = get_cuda_render_status()
    if not ok:
        logger.info(f"CUDA 渲染不可用: {reason}")
    return ok


def _build_glyph_cache(
    frame_states: List[FrameState], fonts: dict,
) -> int:
    """预渲染所有帧中出现的唯一文字到 GPU 纹理。

    在多线程渲染前调用，一次性构建字形图集，避免渲染线程中
    并发上传纹理导致的 CuPy 线程安全问题。

    Returns: 缓存的纹理数量
    """
    count = 0
    copy_c = QColor(195, 195, 195)
    copy_c.setAlpha(COPYRIGHT_ALPHA)

    for state in frame_states:
        sc = QColor(state.small_font_color)

        if state.show_note_name and state.note_name:
            nc = QColor(*state.note_color)
            nc.setAlpha(NOTE_ALPHA)
            _get_glyph_texture(state.note_name, fonts["note_font"], nc, fonts["fm_note"])
            count += 1

        if state.show_lyric and state.lyric:
            lc = QColor(*state.lyric_color)
            _get_glyph_texture(state.lyric, fonts["ust_lyric_font"], lc, fonts["fm_ust_lyric"])
            count += 1

        if state.show_song_name and state.song_name:
            _get_glyph_texture(state.song_name, fonts["bold_small_font"], sc, fonts["fm_small"])
            count += 1
        if state.show_song_author and state.song_author:
            _get_glyph_texture(state.song_author, fonts["small_font"], sc, fonts["fm_small"])
            count += 1
        if state.show_ust_author and state.ust_author:
            _get_glyph_texture(state.ust_author, fonts["small_font"], sc, fonts["fm_small"])
            count += 1

        if state.show_bpm:
            _get_glyph_texture(f"BPM={state.tempo}", fonts["small_font"], sc, fonts["fm_small"])
            count += 1

        if state.show_lyric and state.lrc_lines and not state.lrc_hidden:
            for text in state.lrc_lines:
                _get_glyph_texture(text, fonts["lyric_font"], sc, fonts["fm_lyric"])
                count += 1

        if state.show_copyright:
            _get_glyph_texture(state.copyright_text, fonts["copyright_font"], copy_c, fonts["fm_copyright"])
            count += 1

    logger.info(f"字形图集构建完成: {len(_GLYPH_CACHE)} 个唯一纹理 ({count} 次渲染调用)")
    return len(_GLYPH_CACHE)


# ===================== OpenGL ES 渲染后端 =====================


class _OpenGLESRenderer:
    """基于 OpenGL 的离屏渲染器（FBO + QOpenGLPaintDevice + 同步回读）。

    核心流程：
      ① 绑定 FBO → glClear 清空背景
      ② QOpenGLPaintDevice 桥接 QPainter 到 GPU 绘制（复用 _draw_with_painter）
      ③ glReadPixels 同步回读 RGBA 像素
      ④ 创建 QImage（Format_RGBA8888），管线编码线程自行处理 NV12 转换

    为什么不直接 QPainter(FBO)：
      - PySide6 中 QOpenGLFramebufferObject 不被识别为 QPaintDevice
      - 改用 QOpenGLPaintDevice + FBO 绑定，效果相同

    为什么不使用 PBO 双缓冲：
      - 逐帧渲染管线中，每帧只渲染一次，PBO 的异步回读优势不明显
      - 同步 glReadPixels 更简单、正确性更高
      - NV12 转换由管线编码线程异步处理，不阻塞渲染

    线程安全：单线程渲染，内部用锁保护。
    """

    def __init__(self):
        self._context = None  # QOpenGLContext
        self._surface = None  # QOffscreenSurface
        self._fbo = None  # QOpenGLFramebufferObject
        self._gl = None  # QOpenGLFunctions
        self._paint_device = None  # QOpenGLPaintDevice
        self._lock = threading.Lock()
        self._initialized = False
        self._width = 0
        self._height = 0
        # 预分配像素缓冲区（跨帧复用）
        self._pixel_buf = None

    def ensure_init(self, width: int, height: int) -> None:
        """确保 GLES 渲染器已初始化，分辨率变化时重建。"""
        if self._initialized and self._width == width and self._height == height:
            return
        self._cleanup()
        self._init_gles(width, height)

    def _init_gles(self, width: int, height: int) -> None:
        """初始化 OpenGL 上下文 + FBO + QOpenGLPaintDevice + 像素缓冲区。"""
        from PySide6.QtGui import QSurfaceFormat, QOffscreenSurface, QOpenGLContext
        from PySide6.QtOpenGL import (
            QOpenGLPaintDevice, QOpenGLFramebufferObject, QOpenGLFramebufferObjectFormat,
        )

        self._width = width
        self._height = height

        # 1. 创建 OpenGL 上下文（桌面 OpenGL 3.3，最高兼容性）
        fmt = QSurfaceFormat()
        fmt.setRenderableType(QSurfaceFormat.OpenGL)
        fmt.setVersion(3, 3)
        fmt.setSwapInterval(0)
        fmt.setSamples(0)

        self._context = QOpenGLContext()
        self._context.setFormat(fmt)
        if not self._context.create():
            raise RuntimeError("无法创建 OpenGL 上下文")

        # 2. 创建离屏 surface
        self._surface = QOffscreenSurface()
        self._surface.setFormat(fmt)
        self._surface.create()

        if not self._context.makeCurrent(self._surface):
            raise RuntimeError("无法激活 OpenGL 上下文")

        # 获取 OpenGL 函数接口
        self._gl = self._context.functions()

        # 3. 创建 FBO（离屏渲染目标）
        fbo_fmt = QOpenGLFramebufferObjectFormat()
        fbo_fmt.setInternalTextureFormat(0x8058)  # GL_RGBA8
        fbo_fmt.setAttachment(QOpenGLFramebufferObject.NoAttachment)

        self._fbo = QOpenGLFramebufferObject(width, height, fbo_fmt)
        if not self._fbo.isValid():
            raise RuntimeError("FBO 创建失败")

        # 4. 创建 QOpenGLPaintDevice（桥接 QPainter → GPU）
        self._paint_device = QOpenGLPaintDevice(width, height)

        # 5. 预分配像素缓冲区
        self._pixel_buf = bytearray(width * height * 4)

        self._initialized = True
        logger.info(f"GLES 渲染器初始化完成: {width}x{height}")

    def render_frame(
        self, state: FrameState, width: int, height: int, fonts: dict,
    ) -> QImage:
        """渲染一帧：FBO 离屏渲染 → 同步 glReadPixels 回读 → 返回 QImage。

        NV12 转换由管线编码线程异步处理（不阻塞渲染），
        渲染线程只负责 GPU 绘制和像素回读。

        Returns:
            QImage (Format_RGBA8888)，管线编码线程自行转换 NV12
        """
        with self._lock:
            self.ensure_init(width, height)
            self._context.makeCurrent(self._surface)

            _t0 = time.monotonic()

            # ========== 1. 绑定 FBO 并清除背景 ==========
            self._fbo.bind()

            bg = QColor(state.bg_color)
            self._gl.glClearColor(bg.redF(), bg.greenF(), bg.blueF(), 1.0)
            self._gl.glClear(0x00004000)  # GL_COLOR_BUFFER_BIT

            # 通知 QOpenGLPaintDevice 尺寸变化
            if self._paint_device.size() != QSize(self._width, self._height):
                self._paint_device.setSize(QSize(self._width, self._height))

            # ========== 2. QPainter → QOpenGLPaintDevice → GPU 绘制 ==========
            painter = QPainter(self._paint_device)
            try:
                _draw_with_painter(painter, state, self._width, self._height, fonts)
            finally:
                painter.end()

            # ========== 3. 同步 glReadPixels 回读 ==========
            buf_size = self._width * self._height * 4
            if self._pixel_buf is None or len(self._pixel_buf) < buf_size:
                self._pixel_buf = bytearray(buf_size)
            self._gl.glReadPixels(
                0, 0, self._width, self._height,
                0x1908,  # GL_RGBA
                0x1401,  # GL_UNSIGNED_BYTE
                self._pixel_buf,
            )

            self._fbo.release()

            # ========== 4. 创建 QImage（拷贝像素数据，避免与编码线程的数据竞争） ==========
            import numpy as np
            # 拷贝像素数据：QImage 拥有自己的缓冲区，编码线程可安全读取
            pixel_copy = bytearray(self._pixel_buf)
            rgba_np = np.frombuffer(pixel_copy, dtype=np.uint8).reshape(
                height, width, 4)
            # OpenGL 坐标系 Y=0 在底部，需要翻转行序
            rgba_np[:] = rgba_np[::-1, :, :]

            img = QImage(rgba_np.data, width, height, QImage.Format.Format_RGBA8888)
            img._pixel_ref = pixel_copy

            _elapsed = time.monotonic() - _t0
            if _elapsed > 0.015:
                logger.debug(f"GLES 渲染帧耗时: {_elapsed*1000:.1f}ms")
            return img

    def _cleanup(self) -> None:
        """释放 OpenGL 资源（FBO、paint device、上下文、surface）。"""
        if self._initialized:
            try:
                self._paint_device = None
                self._fbo = None
                if self._context:
                    self._context.doneCurrent()
                self._context = None
                if self._surface:
                    self._surface.destroy()
                self._surface = None
            except Exception:
                logger.exception("GLES 资源清理异常")
            self._initialized = False
            self._gl = None
            self._pixel_buf = None
            logger.debug("GLES 渲染器资源已释放")

    def __del__(self):
        self._cleanup()


# 全局 GLES 渲染器实例（单例，线程安全）
_GLES_RENDERER = _OpenGLESRenderer()


def _render_frame_opengl(
    state: FrameState, width: int, height: int, fonts: dict,
) -> QImage:
    """OpenGL ES 后端渲染：FBO 离屏渲染 + QPainter 桥接 GPU。

    GPU 负责所有绘制（通过 QPainter → QOpenGLPaintDevice → GLES 管线）。
    使用 FBO 离屏渲染 + toImage() 回读像素。
    单线程渲染（GLES 非线程安全），内部锁保护。

    Returns:
        QImage (Format_ARGB32_Premultiplied)，失败时回退 CPU 渲染
    """
    try:
        return _GLES_RENDERER.render_frame(state, width, height, fonts)
    except Exception:
        logger.exception("GLES 渲染失败，回退 CPU")
        return _render_frame_cpu(state, width, height, fonts)


def _clear_gles_context() -> None:
    """释放 GLES 渲染器上下文资源（渲染完成后调用）。"""
    _GLES_RENDERER._cleanup()


# ===================== 渲染器选择 =====================


def _select_render_backend(
    hw: HardwareInfo, preferred: str,
) -> Tuple[str, Callable]:
    """选择渲染后端。

    CUDA 后端使用 CuPy 在 GPU 上渲染（背景填充 + 字形纹理混合 + 折线绘制），
    GLES 后端使用 QOpenGLFramebufferObject 离屏渲染 + QPainter 桥接 GPU，
    CPU 后端使用 QPainter 多线程渲染。
    GPU 加速还体现在编码阶段：NVIDIA 用 NVENC，AMD 用 AMF，Intel 用 QSV。
    用户选 CPU 后端时编码器也降级为 libx264（纯软件编码）。

    Args:
        hw: 硬件信息
        preferred: 用户偏好 ("auto" / "cuda" / "opengl" / "cpu")

    Returns:
        (backend_name, render_func)
    """
    if preferred == "cpu":
        return "cpu", _render_frame_cpu

    if preferred == "cuda":
        if hw.supports_cuda_render and _check_cuda_available():
            return "cuda", _render_frame_cuda
        logger.warning("CUDA 不可用，回退到 CPU 渲染")
        return "cpu", _render_frame_cpu

    if preferred == "opengl":
        # OpenGL ES 后端：FBO 离屏渲染 + QPainter 桥接 GPU
        # 兼容所有显卡，不需要 CUDA
        return "opengl", _render_frame_opengl

    # auto: 有 NVIDIA 显卡就用 CUDA，否则 GLES
    if hw.supports_cuda_render and _check_cuda_available():
        return "cuda", _render_frame_cuda
    return "opengl", _render_frame_opengl


# ===================== 渲染编码并行流水线 =====================


# 时间戳纹理缓存（避免每帧重复 QPainter 渲染）
_TIMESTAMP_CACHE: Dict[str, Tuple[Any, int, int]] = {}
_TS_CACHE_LOCK = threading.Lock()


def _clear_timestamp_cache():
    """清空时间戳纹理缓存（渲染结束后释放内存）。"""
    with _TS_CACHE_LOCK:
        _TIMESTAMP_CACHE.clear()


def _get_timestamp_overlay(
    time_str: str, font: QFont, color: QColor,
) -> Tuple[Any, int, int]:
    """获取时间戳文字的 RGBA numpy 数组（缓存）。

    用 QPainter 渲染小段文字到纹理，后续直接 numpy blit 到帧缓冲，
    避免 FFmpeg drawtext 滤镜的逐帧 CPU 开销。

    Returns:
        (numpy_array_h_w_4, width, height)
    """
    key = f"{time_str}|{font.family()}|{font.pointSize()}|{font.weight()}|{color.rgba()}"
    with _TS_CACHE_LOCK:
        if key in _TIMESTAMP_CACHE:
            return _TIMESTAMP_CACHE[key]

    import numpy as np
    fm = QFontMetrics(font)
    tw = max(1, fm.horizontalAdvance(time_str))
    th = max(1, fm.height())

    img = QImage(tw, th, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)
    painter = QPainter(img)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setFont(font)
    painter.setPen(color)
    painter.drawText(0, fm.ascent(), time_str)
    painter.end()

    ptr = img.bits()
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape((th, tw, 4)).copy()

    with _TS_CACHE_LOCK:
        _TIMESTAMP_CACHE[key] = (arr, tw, th)
    return arr, tw, th


# ===================== 逐帧渲染编码 =====================


def _rgba_to_nv12_cpu(img: QImage, width: int, height: int) -> Optional[bytes]:
    """CPU 端 RGBA→NV12 转换（整数运算，供管线编码线程使用）。

    支持两种 QImage 格式：
      - Format_ARGB32_Premultiplied（CPU 后端）：内存序 BGRA，B=0 G=1 R=2
      - Format_RGBA8888（GLES 后端）：内存序 RGBA，R=0 G=1 B=2

    使用 BT.601 整数运算（uint16，避免 float32 开销）：
        Y  = (77*R + 150*G + 29*B) >> 8
        U  = ((-43*R - 85*G + 128*B) >> 8) + 128
        V  = ((128*R - 107*G - 21*B) >> 8) + 128

    Returns:
        NV12 字节流（[Y: W×H] + [UV: (H/2)×W]，U/V 交错），失败返回 None
    """
    try:
        import numpy as np
    except ImportError:
        return None
    try:
        ptr = img.bits()
        rgba_np = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, 4))

        # 检测格式：RGBA8888 内存序为 RGBA，ARGB32 内存序为 BGRA
        # 使用 uint16（2 字节）比 int32 省一半内存带宽，实测更快
        is_rgba = (img.format() == QImage.Format.Format_RGBA8888)
        if is_rgba:
            r = rgba_np[:, :, 0].astype(np.uint16)
            g = rgba_np[:, :, 1].astype(np.uint16)
            b = rgba_np[:, :, 2].astype(np.uint16)
        else:
            b = rgba_np[:, :, 0].astype(np.uint16)
            g = rgba_np[:, :, 1].astype(np.uint16)
            r = rgba_np[:, :, 2].astype(np.uint16)

        # BT.601: Y = (77*R + 150*G + 29*B) >> 8
        # 77*255 + 150*255 + 29*255 = 65280，uint16 足够
        y = ((77 * r + 150 * g + 29 * b) >> 8).astype(np.uint8)

        # UV 子采样 + 交错
        r_sub = r[::2, ::2]
        g_sub = g[::2, ::2]
        b_sub = b[::2, ::2]

        u = np.clip(((128 * b_sub - 43 * r_sub - 85 * g_sub) >> 8) + 128, 0, 255).astype(np.uint8)
        v = np.clip(((128 * r_sub - 107 * g_sub - 21 * b_sub) >> 8) + 128, 0, 255).astype(np.uint8)

        uv = np.empty((height // 2, width), dtype=np.uint8)
        uv[:, 0::2] = u
        uv[:, 1::2] = v

        return y.tobytes() + uv.tobytes()
    except Exception as e:
        logger.exception(f"CPU RGBA→NV12 转换失败: {e}")
        return None


def _render_encode_pipeline(
    frame_states: List[FrameState],
    render_func: Callable,
    width: int, height: int,
    fonts: dict,
    num_threads: int,
    hw: HardwareInfo,
    ust_info: dict,
    output_path: str,
    fps: int,
    total_output: int,
    progress_callback: Optional[Callable] = None,
    use_nv12: bool = False,
) -> bool:
    """多线程渲染 + 逐帧真实编码（不做去重，不做比特流重复）。

    用户要求「存多少帧就渲染多少帧」——不做视觉去重，每个时间区间
    独立渲染；编码端逐帧真实编码（正常 GOP），不解析/不重复 H.264
    比特流，保证转音（portamento）曲线完整保留。

    流程：
        ① 启动 FFmpeg：rawvideo(nv12) stdin → h264（正常 GOP）→ 临时 .h264
        ② 多线程渲染每个时间区间帧 → 有序缓冲 → 编码线程按 frame_count 逐帧写入 pipe
        ③ 音频 mux → 输出 .mp4

    Args:
        frame_states: 预计算的时间区间帧状态列表（每项 frame_count = 该状态持续帧数）
        render_func: 渲染函数 (state, width, height, fonts) -> QImage
        width, height: 渲染分辨率
        fonts: 渲染用字体 dict
        num_threads: 渲染并行线程数
        hw: 硬件信息（含编码器名）
        ust_info: 完整 ust_info（用于取 audio_path）
        output_path: 输出 .mp4 路径
        fps: 输出帧率
        total_output: 总输出帧数（= sum(frame_count)），用于进度
        progress_callback: 阶段进度回调
        use_nv12: 渲染函数是否直接附带 _nv12_bytes（CUDA 后端）

    Returns:
        是否成功
    """
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        msg = "未找到 ffmpeg，无法编码视频"
        logger.error(msg)
        set_last_render_error(msg)
        return False

    output_dir = os.path.dirname(output_path)
    video_h264_path = os.path.join(
        output_dir, "_video_tmp.h264")

    try:
        # ---- 构建 FFmpeg 命令：逐帧编码（正常 GOP） ----
        cmd = [
            ffmpeg, "-y",
            "-f", "rawvideo",
            "-pixel_format", "nv12",
            "-video_size", f"{width}x{height}",
            "-framerate", str(fps),
            "-i", "-",
            "-c:v", hw.encoder_name,
        ]
        if "nvenc" in hw.encoder_name:
            cmd.extend(NVENC_FAST_OPTS)
            # 补充 profile/coder 选项（测试验证过的最快参数组合）
            cmd.extend(["-profile:v", "main", "-coder", "vlc", "-weighted_pred", "0"])
        elif "amf" in hw.encoder_name:
            cmd.extend(["-quality", "speed", "-rc", "vbr_peak", "-qp_i", "23", "-qp_p", "23", "-bf", "0"])
        elif "qsv" in hw.encoder_name:
            cmd.extend(["-preset", "veryfast", "-global_quality", "23", "-bf", "0"])
        else:
            cmd.extend(["-preset", "veryfast", "-crf", "23", "-bf", "0"])
        # 正常 GOP（关键帧间隔 2 秒），逐帧编码所有输出帧
        cmd.extend(["-g", str(fps * 2), "-pix_fmt", "yuv420p", "-f", "h264", video_h264_path])

        logger.info(f"逐帧编码命令: {' '.join(cmd)}")

        # 启动 FFmpeg 进程（stdin 管道输入）
        process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0,
        )

        # ---- stderr 读取线程（防死锁 + 错误捕获） ----
        stderr_lines: List[str] = []
        _MAX_STDERR_LINES = 300

        def _read_stderr():
            try:
                assert process.stderr is not None
                for line in process.stderr:
                    if isinstance(line, bytes):
                        line = line.decode('utf-8', errors='replace')
                    stderr_lines.append(line)
                    if len(stderr_lines) > _MAX_STDERR_LINES:
                        del stderr_lines[:len(stderr_lines) - _MAX_STDERR_LINES]
            except Exception:
                pass

        stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
        stderr_thread.start()

        # ---- 共享状态 ----
        states_count = len(frame_states)
        results: List[Optional[QImage]] = [None] * states_count
        results_lock = threading.Lock()
        cv = threading.Condition(results_lock)
        encode_error: List[Optional[Exception]] = [None]
        frames_written: List[int] = [0]

        # ---- 编码线程（消费者）：按序取渲染结果，逐帧真实编码 ----
        def _encoder_worker():
            """按 frame_count 将每帧写入 pipe（逐帧真实编码，不做比特流重复）。"""
            try:
                for frame_idx, state in enumerate(frame_states):
                    with cv:
                        while results[frame_idx] is None and encode_error[0] is None:
                            cv.wait(timeout=1.0)
                        if encode_error[0] is not None:
                            return
                        img = results[frame_idx]

                    # 获取 NV12 字节（CUDA 后端 GPU 转换；CPU 后端此处 CPU 转换）
                    nv12_bytes = getattr(img, "_nv12_bytes", None)
                    if nv12_bytes is None:
                        nv12_bytes = _rgba_to_nv12_cpu(img, width, height)
                    if nv12_bytes is None:
                        raise RuntimeError(f"帧 {frame_idx} 无法转换为 NV12")

                    # 同一状态帧按 frame_count 逐帧编码
                    assert process.stdin is not None
                    for _ in range(state.frame_count):
                        process.stdin.write(nv12_bytes)
                        frames_written[0] += 1

                    # 进度更新
                    if progress_callback:
                        pct = min(frames_written[0] / total_output * 100, 99)
                        progress_callback(pct, "GPU渲染")

                    del img  # 释放 QImage 引用

                assert process.stdin is not None
                process.stdin.close()
            except BrokenPipeError:
                process.wait(timeout=5)
                stderr_text = ''.join(stderr_lines)[-1500:]
                err = BrokenPipeError(
                    f"FFmpeg 进程意外退出 (退出码 {process.returncode})。\n"
                    f"FFmpeg stderr:\n{stderr_text}"
                )
                encode_error[0] = err
                logger.error(f"编码线程: FFmpeg 崩溃\n{stderr_text}")
            except Exception as e:
                encode_error[0] = e
                logger.exception("编码线程异常")
                try:
                    process.stdin.close()
                except Exception:
                    pass

        encoder_thread = threading.Thread(target=_encoder_worker, daemon=True)
        encoder_thread.start()

        # ---- 渲染线程（生产者）：多线程并行渲染每个时间区间帧 ----
        # GLES 后端：QOpenGLContext 必须在创建它的线程中使用，因此 inline 渲染
        # CUDA 后端：多线程并行渲染
        if num_threads == 1:
            # 单线程 inline 渲染（用于 GLES 后端，避免跨线程 QOpenGLContext 问题）
            logger.debug("单线程 inline 渲染（GLES 后端）")
            for frame_idx, state in enumerate(frame_states):
                try:
                    img = render_func(state, width, height, fonts)
                except Exception:
                    logger.exception(f"渲染帧 {frame_idx} 失败")
                    img = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
                    img.fill(QColor(state.bg_color))

                with cv:
                    results[frame_idx] = img
                    cv.notify_all()
        else:
            def _render_worker(chunk: List[Tuple[int, FrameState]]):
                for frame_idx, state in chunk:
                    try:
                        img = render_func(state, width, height, fonts)
                    except Exception:
                        logger.exception(f"渲染帧 {frame_idx} 失败")
                        img = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
                        img.fill(QColor(state.bg_color))

                    with cv:
                        results[frame_idx] = img
                        cv.notify_all()

            chunks: List[List[Tuple[int, FrameState]]] = []
            for i, state in enumerate(frame_states):
                chunk_idx = i % num_threads
                while len(chunks) <= chunk_idx:
                    chunks.append([])
                chunks[chunk_idx].append((i, state))

            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(_render_worker, chunk) for chunk in chunks]
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logger.exception("渲染线程异常")
                        if encode_error[0] is None:
                            encode_error[0] = e

        # 等待编码线程与 FFmpeg 结束
        encoder_thread.join(timeout=300)
        process.wait(timeout=300)
        stderr_thread.join(timeout=2)

        if encode_error[0] is not None:
            stderr_text = ''.join(stderr_lines)[-1500:]
            msg = f"编码异常：{type(encode_error[0]).__name__}: {encode_error[0]}"
            if stderr_text:
                msg += f"\nFFmpeg stderr:\n{stderr_text}"
            logger.error(msg)
            set_last_render_error(msg)
            return False

        if process.returncode != 0:
            stderr_text = ''.join(stderr_lines)[-1500:]
            msg = f"FFmpeg 编码失败 (退出码 {process.returncode}):\n{stderr_text}"
            logger.error(msg)
            set_last_render_error(msg)
            return False

        if progress_callback:
            progress_callback(95, "编码")

        # ==================== 音频合并 / 封装 MP4 ====================
        audio_path = ust_info.get("player_style", {}).get("audio_path", "")
        if audio_path and os.path.exists(audio_path):
            # 注意：裸 h264 输入没有 PTS，直接 -c copy 到 MP4 会导致
            # mov muxer 无法 interleave（time=N/A、audio:0KiB、无音频流）。
            # 先封装成 MPEG-TS（生成正确 PTS），再与音频转封装为 MP4。
            ts_tmp = os.path.join(output_dir, "_video_mux_tmp.ts")
            step1_cmd = [
                ffmpeg, "-y",
                "-f", "h264",
                "-r", str(fps),
                "-i", video_h264_path,
                "-c", "copy",
                "-f", "mpegts",
                ts_tmp,
            ]
            logger.info(f"FFmpeg H.264→TS 命令: {' '.join(step1_cmd)}")
            r1 = subprocess.run(
                step1_cmd, capture_output=True, text=True, encoding='utf-8', errors='replace',
                timeout=3600,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0,
            )
            if r1.returncode != 0:
                tail = r1.stderr.strip()[-1500:] if r1.stderr else "(无 stderr 输出)"
                msg = f"FFmpeg H.264→TS 失败 (退出码 {r1.returncode}):\n{tail}"
                logger.error(msg)
                set_last_render_error(msg)
                shutil.move(video_h264_path, output_path)
            else:
                mux_cmd = [
                    ffmpeg, "-y",
                    "-i", ts_tmp,
                    "-i", audio_path,
                    "-map", "0:0",
                    "-map", "1:0",
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-shortest",
                    output_path,
                ]
                logger.info(f"FFmpeg 音频合并命令: {' '.join(mux_cmd)}")
                result = subprocess.run(
                    mux_cmd, capture_output=True, text=True, encoding='utf-8', errors='replace',
                    timeout=3600,
                    creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0,
                )
                if result.returncode != 0:
                    tail = result.stderr.strip()[-1500:] if result.stderr else "(无 stderr 输出)"
                    msg = f"FFmpeg 音频合并失败 (退出码 {result.returncode}):\n{tail}"
                    logger.error(msg)
                    set_last_render_error(msg)
                    shutil.move(video_h264_path, output_path)
                else:
                    try:
                        os.unlink(video_h264_path)
                    except Exception:
                        pass
                    try:
                        os.unlink(ts_tmp)
                    except Exception:
                        pass
        else:
            shutil.move(video_h264_path, output_path)

        if progress_callback:
            progress_callback(100, "编码")

        return True

    except Exception as e:
        logger.exception("编码管道失败")
        set_last_render_error(f"编码管道异常：{type(e).__name__}: {e}")
        return False


# ===================== FFmpeg 编码 =====================


def _find_ffmpeg() -> Optional[str]:
    """查找可用的 ffmpeg 可执行文件。

    查找顺序:
      1. imageio-ffmpeg 内置（优先，功能完整，支持 concat demuxer + NVENC）
      2. PATH 环境变量
      3. 程序根目录
      4. 当前目录
    """
    # 1. imageio-ffmpeg（优先：功能完整，避免 PATH 中的精简版缺失 concat demuxer / NVENC）
    try:
        import imageio_ffmpeg
        path = imageio_ffmpeg.get_ffmpeg_exe()
        if path and os.path.exists(path):
            return path
    except Exception:
        pass

    # 2. PATH
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0,
        )
        if result.returncode == 0:
            return "ffmpeg"
    except Exception:
        pass

    # 3. 程序根目录
    import sys
    program_root = os.path.dirname(os.path.abspath(sys.argv[0]))
    ffmpeg_path = os.path.join(program_root, "ffmpeg.exe")
    if os.path.exists(ffmpeg_path):
        return ffmpeg_path

    # 4. 当前目录
    ffmpeg_path = os.path.join(os.getcwd(), "ffmpeg.exe")
    if os.path.exists(ffmpeg_path):
        return ffmpeg_path

    return None


# ===================== 主接口 =====================


def render_video(
    ust_info: dict,
    output_path: str,
    fps: int = 60,
    width: int = 1920,
    height: int = 1080,
    mode: str = "auto",
    render_backend: str = "auto",
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> bool:
    """GPU 加速渲染导出 USTX 可视化视频。

    完整流程:
        ① CPU 预计算所有时间区间帧的视觉状态（不做视觉去重）
        ② 检测硬件 (GPU/CUDA/NVENC)
        ③ 计算最优并发数
        ④ 多线程渲染每个时间区间帧 → NV12 → FFmpeg 逐帧真实编码（正常 GOP）
        ⑤ 音频合并 → 输出 .mp4

    Args:
        ust_info: build_ust_info() 生成的完整参数 dict
        output_path: 输出 .mp4 路径（必须存在父目录）
        fps: 输出帧率 (30/60/90/120)
        width, height: 输出分辨率
        mode: 已废弃（保留参数以兼容旧调用方）。渲染/编码始终使用
            "逐帧渲染 + 逐帧编码" 方案：不做去重，转音完整保留。
        render_backend: 渲染后端 ("auto" / "cuda" / "opengl" / "cpu")
        progress_callback: 进度回调 callback(pct_0_100, stage)

    Returns:
        是否成功
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    try:
        # 导出计时器：记录每个阶段耗时，最后输出到日志
        _t0 = time.monotonic()

        # 三阶段总进度：每阶段占 1/3（33%），映射到总 0-100
        _PHASE_BASE = {"预计算": 0, "GPU渲染": 33, "编码": 66}
        _PHASE_SPAN = {"预计算": 33, "GPU渲染": 33, "编码": 34}
        progress_lock = threading.Lock()
        _last_stage = [""]
        _last_pct = [0.0]

        def _emit_progress(pct: float, stage: str):
            """阶段内进度 0-100，映射到总进度 0-100（每阶段占 1/3）。"""
            if progress_callback is None:
                return
            base = _PHASE_BASE.get(stage, 0)
            span = _PHASE_SPAN.get(stage, 33)
            mapped = base + round(pct * span / 100)
            with progress_lock:
                if stage != _last_stage[0]:
                    _last_stage[0] = stage
                    _last_pct[0] = 0.0
                if mapped > _last_pct[0]:
                    _last_pct[0] = mapped
                out = _last_pct[0]
            progress_callback(int(out), stage)

        # ==================== 阶段 1: 预计算 ====================
        _emit_progress(0, "预计算")

        hw = detect_hardware()
        _emit_progress(10, "预计算")
        logger.info(
            f"硬件检测结果: GPU={hw.gpu_name}, "
            f"Vendor={hw.gpu_vendor}, "
            f"CUDA核心={hw.cuda_cores}, "
            f"可用显存={hw.vram_usable_gb:.2f}GB, "
            f"编码器={hw.encoder_name}"
        )

        # 扩展 ust_info 用于渲染
        render_info = build_ust_info_for_render(ust_info, width, height)
        fonts = render_info["_render_fonts"]

        frame_states = precompute_frame_states(ust_info, fps, width, height)
        _emit_progress(60, "预计算")
        states_count = len(frame_states)

        total_output_frames = sum(s.frame_count for s in frame_states)
        logger.info(
            f"预计算: {states_count} 个时间区间帧, "
            f"输出 {total_output_frames} 帧 (≈{total_output_frames / fps:.1f}s)"
        )

        if states_count == 0:
            msg = "预计算产生 0 帧，无法导出"
            logger.error(msg)
            set_last_render_error(msg)
            return False

        wc = calc_optimal_workers(hw, states_count, width, height)
        _emit_progress(80, "预计算")
        logger.info(
            f"并发配置: render_streams={wc.render_streams}, "
            f"encode_workers={wc.encode_workers}, "
            f"batch_size={wc.batch_size}, "
            f"mode={wc.default_mode}, "
            f"per_frame={wc.per_frame_gb:.4f}GB"
        )

        backend_name, render_func = _select_render_backend(hw, render_backend)
        logger.info(f"渲染后端: {backend_name} (逐帧渲染编码，不做去重)")

        # CPU 后端强制纯 CPU 编码，不碰任何 GPU 编码器
        if backend_name == "cpu":
            hw = hw._replace(encoder_name="libx264")
            logger.info("CPU 后端：编码器强制切换为 libx264")

        _emit_progress(100, "预计算")

        # 预计算阶段计时
        _t1 = time.monotonic()
        _precompute_elapsed = _t1 - _t0
        logger.info(f"[导出计时] 预计算阶段耗时: {_precompute_elapsed:.2f}s")

        # ==================== 阶段 2-3: 渲染 + 编码（内存管道） ====================
        _emit_progress(0, "GPU渲染")

        num_threads = wc.render_streams

        # GLES 后端：GLES 非线程安全，强制单线程渲染
        if backend_name == "opengl":
            num_threads = 1
            logger.info("GLES 后端：单线程渲染（GLES 非线程安全）")

        # CUDA 后端：预构建字形图集
        if backend_name == "cuda":
            _clear_glyph_cache()
            _build_glyph_cache(frame_states, fonts)
            logger.info(f"CUDA 后端：字形图集已预构建，开始渲染 ({num_threads} 线程)")

        # 内存管道：逐帧渲染 → FFmpeg 逐帧编码（正常 GOP）
        # 不做去重、不做比特流重复，保证转音完整
        _t2 = time.monotonic()
        success = _render_encode_pipeline(
            frame_states=frame_states,
            render_func=render_func,
            width=width, height=height,
            fonts=fonts,
            num_threads=num_threads,
            hw=hw,
            ust_info=ust_info,
            output_path=output_path,
            fps=fps,
            total_output=total_output_frames,
            progress_callback=_emit_progress,
            use_nv12=(backend_name == "cuda"),
        )
        _t3 = time.monotonic()
        _render_elapsed = _t3 - _t2

        # CUDA / GLES 后端：渲染完成后释放资源 + 时间戳纹理
        if backend_name == "cuda":
            _clear_glyph_cache()
            _clear_cuda_contexts()
        elif backend_name == "opengl":
            _clear_gles_context()
        _clear_timestamp_cache()

        _emit_progress(100 if success else 99, "GPU渲染")

        _t4 = time.monotonic()
        _encode_elapsed = _t4 - _t3
        _total_elapsed = _t4 - _t0
        _out_frames = total_output_frames if success else 0
        logger.info(
            f"[导出计时] 渲染阶段耗时: {_render_elapsed:.2f}s\n"
            f"[导出计时] 编码阶段耗时: {_encode_elapsed:.2f}s\n"
            f"[导出计时] 导出总耗时: {_total_elapsed:.2f}s\n"
            f"[导出计时] 各阶段占比: 预计算 {_precompute_elapsed:.1f}s | "
            f"渲染+编码 {_render_elapsed:.1f}s | "
            f"音频合并 {_encode_elapsed:.1f}s\n"
            f"[导出计时] 输出帧数: {_out_frames} 帧, "
            f"实际帧率: {_out_frames / _total_elapsed:.1f} fps"
        )

        # ==================== 清理 ====================
        # 显式释放大内存对象，减少内存占用
        del frame_states
        del fonts
        import gc
        gc.collect()

        return success

    except Exception as e:
        logger.exception("渲染导出失败")
        set_last_render_error(f"渲染导出异常：{type(e).__name__}: {e}")
        # 释放模块级缓存 + 强制 GC
        clear_renderer_cache()
        import gc
        gc.collect()
        return False