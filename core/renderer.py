# renderer.py — GPU 加速渲染导出引擎
"""USTX 可视化视频导出系统。

完整管线:
    ① CPU 预计算所有帧的视觉状态 + 去重
    ② 检测硬件 (GPU/NVENC/AMF/QSV)
    ③ 计算最优并发数 (渲染 stream/编码 worker)
    ④ GPU 多 stream 并行渲染唯一帧 → PNG 落盘
    ⑤ FFmpeg 帧重复编码 + drawtext 时间注入 + 音频 mux
    ⑥ 清理临时文件

对外接口:
    render_video(ust_info, output_path, ...) -> bool
"""

import os
import re
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from typing import (
    Callable, List, Optional, Tuple, Dict, Any, NamedTuple,
)

from PySide6.QtCore import Qt, QRectF, QPointF
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
    APP_VERSION = "v26h06"


# ===================== 常量 =====================

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE_LINE_WIDTH = 5
NOTE_ALPHA = 225
COPYRIGHT_ALPHA = 100

# 硬编码上限
MAX_RENDER_STREAMS = 8
MAX_ENCODE_WORKERS = 2
MIN_CUDA_CORES_FOR_CUDA = 512  # CUDA 核心数不足此值时禁用 CUDA 渲染

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
        # 通过 nvidia-smi 或架构名推断 NVENC 代数
        nvenc_gen = _detect_nvenc_generation(gpu_name)

        return {
            "gpu_name": gpu_name,
            "gpu_vendor": "nvidia",
            "cuda_cores": cuda_cores,
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
                f"不足 {MIN_CUDA_CORES_FOR_CUDA}，禁用 CUDA 渲染，使用 OpenGL 回退"
            )

        return HardwareInfo(
            gpu_name=nvidia["gpu_name"],
            gpu_vendor="nvidia",
            cuda_cores=cuda_cores,
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
    hw: HardwareInfo, unique_frames: int, width: int, height: int,
) -> WorkerConfig:
    """三约束取最小，算出最优并发数。

    约束:
      a) CUDA 核心数，每 stream 至少 512 核
      b) 显存，每 stream 需 overhead(0.15GB) + 1帧buffer
      c) 任务量，不超过唯一帧数 1/10
    """
    per_frame_gb = (width * height * 4) / (1024 ** 3)

    # ---- 渲染 stream 数 ----
    if hw.supports_cuda_render and hw.cuda_cores > 0:
        max_by_cores = max(1, hw.cuda_cores // 512)
    else:
        max_by_cores = max(1, os.cpu_count() or 4)  # CPU/OpenGL 用 CPU 核心数

    if hw.vram_usable_gb > 0:
        max_by_vram = max(1, int(hw.vram_usable_gb / (VRAM_OVERHEAD_PER_STREAM + per_frame_gb)))
    else:
        # 无显存信息时（AMD/Intel/CPU），使用 CPU 核心数
        max_by_vram = max(1, os.cpu_count() or 4)

    max_by_task = max(1, unique_frames // 10) if unique_frames > 0 else 1

    render_streams = min(max_by_cores, max_by_vram, max_by_task, MAX_RENDER_STREAMS)

    # ---- 编码并发数 ----
    if hw.nvenc_count > 0:
        encode_workers = hw.nvenc_count if unique_frames > 200 else 1
    else:
        encode_workers = 1

    # ---- 每批渲染帧数 ----
    if hw.vram_usable_gb > 0 and per_frame_gb > 0:
        batch_size = min(int(hw.vram_usable_gb / per_frame_gb), unique_frames)
    else:
        batch_size = min(100, unique_frames)  # CPU 无显存限制，但设合理上限

    # ---- 渲染模式判定 ----
    total_frame_volume = unique_frames * per_frame_gb
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
        去重后的 FrameState 列表，每项代表一个唯一视觉帧
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
    last_cache_key = ""

    # 用于去重的缓存
    seen_cache_keys: Dict[str, int] = {}  # cache_key -> index in states

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

        # 构建缓存 key
        cache_parts = [
            lyric_text, note_name, bg_hex,
            str(lyric_rgb), str(note_rgb), pitch_curve_hex,
            str(len(pitch_points)),
            str(lrc_lines), str(lrc_hidden),
            song_name, song_author, ust_author,
            str(show_bpm), str(show_song_name), str(show_song_author),
            str(show_ust_author), str(show_copyright), str(show_play_time),
            str(tempo), wff, iff, small_font_color,
        ]
        cache_key = "|".join(cache_parts)

        # 去重检测
        if cache_key in seen_cache_keys:
            # 合并到已有帧（延长持续时间）
            prev_idx = seen_cache_keys[cache_key]
            prev_state = states[prev_idx]
            new_duration = prev_state.duration + duration
            new_frame_count = prev_state.frame_count + frame_count
            # 更新为新的持续时间
            states[prev_idx] = prev_state._replace(
                duration=new_duration,
                frame_count=new_frame_count,
            )
            continue

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
        seen_cache_keys[cache_key] = len(states)
        states.append(state)

    logger.info(
        f"预计算完成: {len(states)} 个唯一帧 "
        f"(原始 {len(sorted_times)} 个时间点, "
        f"去重后 {len(states)} 帧)"
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


def _render_frame_cpu(
    state: FrameState, width: int, height: int, fonts: dict,
) -> QImage:
    """使用 QPainter 在 QImage 上渲染一帧（CPU 渲染）。"""
    image = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(QColor(state.bg_color))

    painter = QPainter(image)
    try:
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
    import cupy as cp
    _CUDA_LOCAL.__dict__.pop("ctx", None)
    cp.get_default_memory_pool().free_all_blocks()


def _render_frame_cuda(
    state: FrameState, width: int, height: int, fonts: dict,
) -> QImage:
    """CUDA 后端渲染：每线程独立 stream + 复用 GPU 帧缓冲。

    GPU 负责：背景填充、纹理 blit（alpha 混合）、折线绘制。
    多线程各自在独立 stream 上执行，可真正并行。
    每帧 CPU 仅做 GPU→CPU 下载 + BMP 保存。
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
        result_np = cp.asnumpy(fb)
        result_img = QImage(
            result_np.data, width, height, width * 4,
            QImage.Format.Format_ARGB32_Premultiplied,
        )
        # 保持 numpy 引用存活（QImage 包装不拷贝数据，保存 BMP 时需要底层缓冲有效）
        result_img._numpy_ref = result_np
        return result_img

    except Exception:
        logger.exception("CUDA 渲染失败，回退 CPU")
        return _render_frame_cpu(state, width, height, fonts)


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


# ===================== 渲染器选择 =====================


def _select_render_backend(
    hw: HardwareInfo, preferred: str,
) -> Tuple[str, Callable]:
    """选择渲染后端。

    CUDA 后端使用 CuPy 在 GPU 上渲染（背景填充 + 字形纹理混合 + 折线绘制），
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
        # OpenGL 选项保留为 CUDA 回退（UI 兼容）
        if hw.supports_cuda_render and _check_cuda_available():
            return "cuda", _render_frame_cuda
        return "cpu", _render_frame_cpu

    # auto: 有 NVIDIA 显卡就用 CUDA（GPU 渲染 + NVENC 编码），否则 CPU
    if hw.supports_cuda_render and _check_cuda_available():
        return "cuda", _render_frame_cuda
    return "cpu", _render_frame_cpu


# ===================== 多线程渲染 =====================


# 保存线程池（渲染/保存流水线分离）
# 渲染线程只做 GPU 渲染并提交保存任务，不阻塞等待保存完成，
# 这样 GPU 可以持续工作，不会因磁盘 IO 空转。
_SAVE_POOL = None
_SAVE_FUTURES: List[Any] = []
_SAVE_LOCK = threading.Lock()


def _init_save_pool(max_workers: int = 4):
    """初始化全局保存线程池。"""
    global _SAVE_POOL
    if _SAVE_POOL is None:
        _SAVE_POOL = ThreadPoolExecutor(max_workers=max_workers)


def _wait_saves() -> None:
    """等待所有已提交的保存任务完成（渲染阶段结束后调用）。"""
    global _SAVE_FUTURES
    with _SAVE_LOCK:
        futures = _SAVE_FUTURES
        _SAVE_FUTURES = []
    for f in futures:
        try:
            f.result()
        except Exception:
            logger.exception("保存线程异常")


def _shutdown_save_pool() -> None:
    """关闭保存线程池（渲染全部结束后调用）。"""
    global _SAVE_POOL
    try:
        _wait_saves()
        if _SAVE_POOL is not None:
            _SAVE_POOL.shutdown(wait=True)
    finally:
        _SAVE_POOL = None


def _save_img(img: Any, path: str, width: int, height: int) -> None:
    """保存一帧图像（在保存线程池中执行）。

    CUDA 后端返回的 QImage 包装了 numpy 数据（_numpy_ref），
    直接用 numpy 写 BMP（跳过 Qt 层，更快）。
    CPU 后端返回普通 QImage，用 QImage.save。
    """
    numpy_ref = getattr(img, "_numpy_ref", None)
    if numpy_ref is not None:
        _save_bmp_numpy(numpy_ref, path, width, height)
    else:
        img.save(path, "BMP")


def _save_bmp_numpy(bgra_arr: Any, path: str, width: int, height: int) -> None:
    """用 numpy 直接写 BMP 文件（BGRA → BGR，bottom-up）。

    比 QImage.save("BMP") 少一层 Qt 封装，写入更快。
    """
    import numpy as np

    # BMP 是 bottom-up BGR，去掉 alpha 通道并翻转
    bgr = np.ascontiguousarray(bgra_arr[::-1, :, :3])
    row_size = width * 3
    padding = (4 - row_size % 4) % 4
    stride = row_size + padding

    if padding:
        out = np.zeros((height, stride), dtype=np.uint8)
        out[:, :row_size] = bgr.reshape(height, row_size)
    else:
        out = bgr.reshape(height, stride)

    import struct
    file_size = 54 + stride * height
    header = (
        struct.pack('<2sIHHI', b'BM', file_size, 0, 0, 54)
        + struct.pack('<IiiHHIIiiII', 40, width, height, 1, 24, 0,
                       stride * height, 2835, 2835, 0, 0)
    )
    with open(path, 'wb') as f:
        f.write(header)
        f.write(out.tobytes())


def _render_chunk(
    chunk: List[Tuple[int, FrameState]],
    render_func: Callable,
    width: int, height: int, fonts: dict,
    temp_dir: str,
    progress_callback: Optional[Callable] = None,
    progress_offset: int = 0,
    total: int = 0,
) -> List[str]:
    """渲染一批帧（渲染完成后异步保存，流水线并行）。

    Args:
        chunk: [(frame_idx, state), ...]
        render_func: 渲染函数
        width, height, fonts: 渲染参数
        temp_dir: 临时目录
        progress_callback: 进度回调
        progress_offset: 进度偏移
        total: 总帧数

    Returns:
        [frame_path, ...]
    """
    paths = []
    for local_idx, (frame_idx, state) in enumerate(chunk):
        try:
            img = render_func(state, width, height, fonts)
            path = os.path.join(temp_dir, f"frame_{frame_idx:06d}.bmp")
            # 异步保存：渲染线程不阻塞，保存在线程池并行
            with _SAVE_LOCK:
                _SAVE_FUTURES.append(
                    _SAVE_POOL.submit(_save_img, img, path, width, height)
                )
            paths.append(path)
        except Exception:
            logger.exception(f"渲染帧 {frame_idx} 失败")
            # 创建一个空白帧代替
            fallback = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
            fallback.fill(QColor(state.bg_color))
            path = os.path.join(temp_dir, f"frame_{frame_idx:06d}.bmp")
            with _SAVE_LOCK:
                _SAVE_FUTURES.append(
                    _SAVE_POOL.submit(_save_img, fallback, path, width, height)
                )
            paths.append(path)

        if progress_callback:
            progress_callback(progress_offset + local_idx + 1, total, "GPU 渲染中")

    return paths


# ===================== FFmpeg 编码 =====================


def _find_ffmpeg() -> Optional[str]:
    """查找可用的 ffmpeg 可执行文件。

    查找顺序:
      1. PATH 环境变量
      2. imageio-ffmpeg 内置
      3. 程序根目录
      4. 当前目录
    """
    # 1. PATH
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0,
        )
        if result.returncode == 0:
            return "ffmpeg"
    except Exception:
        pass

    # 2. imageio-ffmpeg
    try:
        import imageio_ffmpeg
        path = imageio_ffmpeg.get_ffmpeg_exe()
        if path and os.path.exists(path):
            return path
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


def _resolve_font_path(font_family: str) -> str:
    """解析字体文件路径，用于 FFmpeg drawtext 滤镜。"""
    # 尝试从系统字体目录查找
    font_files = {
        "微软雅黑": "msyh.ttc",
        "Microsoft YaHei": "msyh.ttc",
        "等线": "Deng.ttf",
        "DengXian": "Deng.ttf",
        "黑体": "simhei.ttf",
        "SimHei": "simhei.ttf",
        "宋体": "simsun.ttc",
        "SimSun": "simsun.ttc",
    }

    if font_family in font_files:
        # 尝试在 Windows 字体目录查找
        windir = os.environ.get("WINDIR", "C:\\Windows")
        font_path = os.path.join(windir, "Fonts", font_files[font_family])
        if os.path.exists(font_path):
            return font_path

    # 回退：假设字体名就是文件名（Linux 风格）
    return font_family


def _escape_ffmpeg_path(path: str) -> str:
    """转义 FFmpeg 路径中的特殊字符。

    FFmpeg 滤镜语法中，\\: 表示转义冒号（避免被解析为选项分隔符），
    \\\\ 表示字面反斜杠。Windows 路径 C:/Windows/Fonts/msyh.ttc
    应转为 C\\:/Windows/Fonts/msyh.ttc（\\: = 字面冒号）。
    """
    # 先统一为正斜杠，再转义冒号
    path = path.replace('\\', '/')
    return path.replace(':', '\\:')


def _build_concat_file(
    frame_states: List[FrameState],
    temp_dir: str,
    fps: int,
) -> str:
    """生成 FFmpeg concat 描述文件。

    帧重复：500 帧描述文件 → 27000 帧输出
    """
    concat_path = os.path.join(temp_dir, "concat.txt")
    with open(concat_path, 'w', encoding='utf-8') as f:
        for idx, state in enumerate(frame_states):
            duration = state.frame_count / fps
            if duration <= 0:
                duration = 1.0 / fps
            f.write(f"file 'frame_{idx:06d}.bmp'\n")
            f.write(f"duration {duration:.6f}\n")
        # FFmpeg concat 要求最后一帧重复写一次
        if frame_states:
            last_idx = len(frame_states) - 1
            f.write(f"file 'frame_{last_idx:06d}.bmp'\n")
    return concat_path


def _build_drawtext_filter(
    ust_info: dict, fps: int,
) -> Optional[str]:
    """构建 FFmpeg drawtext 滤镜（显示播放时间）。

    样式从 ust_info 读取，与播放器一致。
    """
    sc = ust_info.get("show_config", {})
    ps = ust_info.get("player_style", {})

    if not sc.get("play_time", True):
        return None

    font_family = ps.get("info_font_family", "微软雅黑")
    font_color = ps.get("info_text_color", "#ffffff")
    font_file = _resolve_font_path(font_family)

    # MM:SS:CC 格式，每帧自动计算
    # FFmpeg drawtext: text='%{eif:floor(t/60):d:2}:%{eif:floor(t):d:2}:%{eif:floor(t*100):d:2}'
    filter_str = (
        f"drawtext=fontfile='{_escape_ffmpeg_path(font_file)}':"
        f"text='%{{eif\\:floor(t/60)\\:d\\:2}}\\:"
        f"%{{eif\\:mod(floor(t)\\,60)\\:d\\:2}}\\:"
        f"%{{eif\\:floor(mod(t\\,1)*100)\\:d\\:2}}':"
        f"fontcolor={font_color}:"
        f"fontsize=14:"
        f"x=20:y=h-20"
    )
    return filter_str


def _encode_video(
    frame_states: List[FrameState],
    hw: HardwareInfo,
    ust_info: dict,
    temp_dir: str,
    output_path: str,
    fps: int,
    width: int,
    height: int,
    progress_callback: Optional[Callable] = None,
) -> bool:
    """编码视频：帧重复 → drawtext 时间注入 → 音频合并。

    Args:
        frame_states: 唯一帧列表
        hw: 硬件信息
        ust_info: 配置
        temp_dir: 临时目录
        output_path: 输出 .mp4 路径
        fps: 帧率
        width, height: 分辨率
        progress_callback: 进度回调
    """
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        msg = "未找到 ffmpeg，无法编码视频"
        logger.error(msg)
        set_last_render_error(msg)
        return False

    try:
        # ---- 5.1 生成 concat.txt ----
        if progress_callback:
            progress_callback(10, 100, "编码")
        concat_path = _build_concat_file(frame_states, temp_dir, fps)

        # ---- 5.2 组装 drawtext 滤镜 ----
        drawtext_filter = _build_drawtext_filter(ust_info, fps)

        # ---- 5.3 编码视频流 ----
        video_output = os.path.join(temp_dir, "video_only.mp4")
        if progress_callback:
            progress_callback(20, 100, "编码")

        # 基础 FFmpeg 命令
        # concat 文件中的 duration 指令控制每帧持续时间，不要加 -r 覆盖
        video_cmd = [
            ffmpeg, "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_path,
        ]

        # 滤镜：先 fps 做帧率转换（让每帧有独立 PTS），再 drawtext 画时间
        if drawtext_filter:
            video_cmd.extend(["-vf", f"fps={fps},{drawtext_filter}"])

        # 编码器（不再重复 -r，fps 滤镜已控制帧率）
        video_cmd.extend([
            "-c:v", hw.encoder_name,
            "-pix_fmt", "yuv420p",
        ])

        # 编码器特定参数
        if "nvenc" in hw.encoder_name:
            video_cmd.extend([
                "-preset", "p4",
                "-rc", "vbr",
                "-cq", "23",
                "-b:v", "0",
            ])
        elif "amf" in hw.encoder_name:
            video_cmd.extend([
                "-quality", "quality",
                "-rc", "vbr_peak",
                "-qp_i", "23", "-qp_p", "23",
            ])
        elif "qsv" in hw.encoder_name:
            video_cmd.extend([
                "-preset", "medium",
                "-global_quality", "23",
            ])
        else:  # libx264
            video_cmd.extend([
                "-preset", "medium",
                "-crf", "23",
            ])

        video_cmd.append(video_output)

        # 多 NVENC 并行
        if hw.nvenc_count > 1 and len(frame_states) > 200:
            # 分段编码
            if progress_callback:
                progress_callback(30, 100, "编码")
            _encode_dual_nvenc(
                ffmpeg, frame_states, hw, temp_dir, fps,
                concat_path, drawtext_filter, video_output,
            )
            if progress_callback:
                progress_callback(75, 100, "编码")
        else:
            if progress_callback:
                progress_callback(30, 100, "编码")
            logger.info(f"FFmpeg 视频编码命令: {' '.join(video_cmd)}")
            result = subprocess.run(
                video_cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=3600,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0,
            )
            if result.returncode != 0:
                tail = result.stderr.strip()[-1500:] if result.stderr else "(无 stderr 输出)"
                msg = f"FFmpeg 视频编码失败 (退出码 {result.returncode}):\n{tail}"
                logger.error(msg)
                set_last_render_error(msg)
                return False

        if progress_callback:
            progress_callback(80, 100, "编码")

        # ---- 5.4 mux 音频 ----
        audio_path = ust_info.get("player_style", {}).get("audio_path", "")
        if audio_path and os.path.exists(audio_path):
            if progress_callback:
                progress_callback(90, 100, "编码")

            mux_cmd = [
                ffmpeg, "-y",
                "-i", video_output,
                "-i", audio_path,
                "-c", "copy",
                "-shortest",
                output_path,
            ]
            logger.info(f"FFmpeg 音频合并命令: {' '.join(mux_cmd)}")
            result = subprocess.run(
                mux_cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=3600,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0,
            )
            if result.returncode != 0:
                tail = result.stderr.strip()[-1500:] if result.stderr else "(无 stderr 输出)"
                msg = f"FFmpeg 音频合并失败 (退出码 {result.returncode}):\n{tail}"
                logger.error(msg)
                set_last_render_error(msg)
                # 尝试使用视频文件作为输出
                shutil.copy(video_output, output_path)
        else:
            # 无音频，直接复制视频
            shutil.copy(video_output, output_path)

        if progress_callback:
            progress_callback(100, 100, "编码")

        return True

    except subprocess.TimeoutExpired:
        msg = "FFmpeg 编码超时（超过 1 小时），已中止"
        logger.error(msg)
        set_last_render_error(msg)
        return False
    except Exception as e:
        logger.exception("视频编码失败")
        set_last_render_error(f"视频编码异常：{type(e).__name__}: {e}")
        return False


def _encode_dual_nvenc(
    ffmpeg: str,
    frame_states: List[FrameState],
    hw: HardwareInfo,
    temp_dir: str,
    fps: int,
    concat_path: str,
    drawtext_filter: Optional[str],
    video_output: str,
) -> bool:
    """双 NVENC 并行编码：将帧分段，两个 FFmpeg 进程并行编码，然后 concat。"""
    total = len(frame_states)
    mid = total // 2

    # 生成两个子 concat 文件
    def make_sub_concat(start: int, end: int, suffix: str) -> str:
        sub_path = os.path.join(temp_dir, f"concat_{suffix}.txt")
        with open(sub_path, 'w', encoding='utf-8') as f:
            for i in range(start, end):
                state = frame_states[i]
                duration = state.frame_count / fps
                if duration <= 0:
                    duration = 1.0 / fps
                f.write(f"file 'frame_{i:06d}.bmp'\n")
                f.write(f"duration {duration:.6f}\n")
            # 最后帧重复
            f.write(f"file 'frame_{end - 1:06d}.bmp'\n")
        return sub_path

    concat_a = make_sub_concat(0, mid, "a")
    concat_b = make_sub_concat(mid, total, "b")
    output_a = os.path.join(temp_dir, "segment_a.mp4")
    output_b = os.path.join(temp_dir, "segment_b.mp4")

    def encode_segment(concat_file: str, output: str, device: int):
        cmd = [
            ffmpeg, "-y",
            "-f", "concat", "-safe", "0", "-i", concat_file,
        ]
        if drawtext_filter:
            cmd.extend(["-vf", f"fps={fps},{drawtext_filter}"])
        cmd.extend([
            "-c:v", hw.encoder_name,
            "-pix_fmt", "yuv420p",
            "-preset", "p4",
            "-rc", "vbr",
            "-cq", "23",
            "-b:v", "0",
            "-gpu", str(device),
            output,
        ])
        subprocess.run(
            cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=3600,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0,
        )

    # 并行编码两个段
    t1 = threading.Thread(target=encode_segment, args=(concat_a, output_a, 0))
    t2 = threading.Thread(target=encode_segment, args=(concat_b, output_b, 1))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # 合并两个段
    concat_list = os.path.join(temp_dir, "concat_segments.txt")
    with open(concat_list, 'w', encoding='utf-8') as f:
        f.write(f"file '{os.path.abspath(output_a).replace(chr(92), '/')}'\n")
        f.write(f"file '{os.path.abspath(output_b).replace(chr(92), '/')}'\n")

    merge_cmd = [
        ffmpeg, "-y",
        "-f", "concat", "-safe", "0", "-i", concat_list,
        "-c", "copy",
        video_output,
    ]
    result = subprocess.run(
        merge_cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=3600,
        creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0,
    )
    return result.returncode == 0


# ===================== 主接口 =====================


def render_video(
    ust_info: dict,
    output_path: str,
    fps: int = 60,
    width: int = 1920,
    height: int = 1080,
    mode: str = "auto",
    render_backend: str = "auto",
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> bool:
    """GPU 加速渲染导出 USTX 可视化视频。

    完整流程:
        ① CPU 预计算所有帧的视觉状态 + 去重
        ② 检测硬件 (GPU/CUDA/NVENC)
        ③ 计算最优并发数 (渲染stream/编码worker)
        ④ GPU 多 stream 并行渲染唯一帧 → PNG 落盘
        ⑤ FFmpeg 帧重复编码 + drawtext 时间注入 + 音频 mux
        ⑥ 清理临时文件

    Args:
        ust_info: build_ust_info() 生成的完整参数 dict
        output_path: 输出 .mp4 路径（必须存在父目录）
        fps: 输出帧率 (30/60/90/120)
        width, height: 输出分辨率
        mode: 渲染编码模式 ("auto" / "batch" / "stream")
        render_backend: 渲染后端 ("auto" / "cuda" / "opengl" / "cpu")
        progress_callback: 进度回调 callback(current, total, stage)

    Returns:
        是否成功
    """
    import sys
    program_root = os.path.dirname(os.path.abspath(sys.argv[0]))
    temp_dir = os.path.join(program_root, "temp_render")

    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 清理旧临时目录
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    try:
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
            progress_callback(int(out), 100, stage)

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
        unique_count = len(frame_states)

        total_output_frames = sum(s.frame_count for s in frame_states)
        logger.info(
            f"预计算: {unique_count} 个唯一帧, "
            f"输出 {total_output_frames} 帧 (≈{total_output_frames / fps:.1f}s)"
        )

        if unique_count == 0:
            msg = "预计算产生 0 帧，无法导出"
            logger.error(msg)
            set_last_render_error(msg)
            return False

        wc = calc_optimal_workers(hw, unique_count, width, height)
        _emit_progress(80, "预计算")
        logger.info(
            f"并发配置: render_streams={wc.render_streams}, "
            f"encode_workers={wc.encode_workers}, "
            f"batch_size={wc.batch_size}, "
            f"mode={wc.default_mode}, "
            f"per_frame={wc.per_frame_gb:.4f}GB"
        )

        effective_mode = mode
        if effective_mode == "auto":
            effective_mode = wc.default_mode

        backend_name, render_func = _select_render_backend(hw, render_backend)
        logger.info(f"渲染后端: {backend_name}, 模式: {effective_mode}")

        # CPU 后端强制纯 CPU 编码，不碰任何 GPU 编码器
        if backend_name == "cpu":
            hw = hw._replace(encoder_name="libx264")
            logger.info("CPU 后端：编码器强制切换为 libx264")

        _emit_progress(100, "预计算")

        # ==================== 阶段 2: 渲染 ====================
        _emit_progress(0, "GPU渲染")

        # 初始化异步保存线程池：渲染只提交 GPU 任务，磁盘写入由独立线程池完成
        _init_save_pool(max_workers=4)

        def render_phase_cb(current, total, stage):
            frac = current / total if total > 0 else 0
            _emit_progress(frac * 100, stage)

        # 多线程渲染（CUDA 和 CPU 后端均使用多线程并行）
        # CUDA: 多线程并行渲染，每帧独立分配 GPU 帧缓冲，完成后释放
        # CPU:  多线程 QPainter 渲染
        num_threads = wc.render_streams

        # CUDA 后端：预构建字形图集（在主线程一次性上传所有文字纹理到 GPU，
        # 避免渲染线程中并发上传导致 CuPy 线程安全问题）
        if backend_name == "cuda":
            _clear_glyph_cache()
            _build_glyph_cache(frame_states, fonts)
            logger.info(f"CUDA 后端：字形图集已预构建，开始 GPU 渲染 ({num_threads} 线程)")

        chunks: List[List[Tuple[int, FrameState]]] = []
        for i, state in enumerate(frame_states):
            chunk_idx = i % num_threads
            while len(chunks) <= chunk_idx:
                chunks.append([])
            chunks[chunk_idx].append((i, state))

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for chunk_idx, chunk in enumerate(chunks):
                future = executor.submit(
                    _render_chunk,
                    chunk, render_func, width, height, fonts,
                    temp_dir, render_phase_cb,
                    progress_offset=sum(len(c) for c in chunks[:chunk_idx]),
                    total=unique_count,
                )
                futures.append(future)

            all_paths = []
            for future in as_completed(futures):
                try:
                    paths = future.result()
                    all_paths.extend(paths)
                except Exception:
                    logger.exception("渲染线程异常")
                    set_last_render_error("渲染线程异常，详情见日志")

        # 等待所有异步保存任务完成（渲染已全部提交，保存并行进行）
        _wait_saves()
        # 渲染/保存流水线结束，关闭保存线程池释放线程资源
        _shutdown_save_pool()

        # CUDA 后端：渲染完成后释放字形图集 + 线程上下文显存
        if backend_name == "cuda":
            _clear_glyph_cache()
            _clear_cuda_contexts()

        logger.info(f"渲染完成: {len(all_paths)} 帧, 后端={backend_name}")
        _emit_progress(100, "GPU渲染")

        # ==================== 阶段 3: 编码 ====================
        _emit_progress(0, "编码")

        def encode_phase_cb(current, total, stage):
            frac = current / total if total > 0 else 0
            _emit_progress(frac * 100, stage)

        success = _encode_video(
            frame_states, hw, ust_info, temp_dir,
            output_path, fps, width, height,
            encode_phase_cb,
        )

        _emit_progress(100 if success else 99, "编码")

        # ==================== 阶段 7: 清理 ====================
        try:
            # 保留临时文件用于调试
            if success:
                shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass

        return success

    except Exception as e:
        logger.exception("渲染导出失败")
        set_last_render_error(f"渲染导出异常：{type(e).__name__}: {e}")
        # 清理临时文件
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass
        return False