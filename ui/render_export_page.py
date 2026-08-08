# render_export_page.py — 渲染导出页面（复用主窗口内容区）
"""渲染导出设置 + 进度页面，嵌入主窗口而非独立弹窗。

布局：
    ┌─────────────────────────────────────────────────┐
    │  [← 返回]  渲染导出                              │
    ├──────────────────────┬──────────────────────────┤
    │  左栏：配置卡片       │  右栏：状态卡片           │
    │  ┌────────────────┐  │  ┌────────────────────┐  │
    │  │ 输出参数         │  │  │ 状态               │  │
    │  │ (分辨率/帧率/路径)│  │  │ (进度条/阶段/百分比)│  │
    │  └────────────────┘  │  └────────────────────┘  │
    │  ┌────────────────┐  │  ┌────────────────────┐  │
    │  │ 硬件信息         │  │  │ 操作               │  │
    │  │ (GPU/编码器/显存)│  │  │ (开始/重新生成/打开)│  │
    │  └────────────────┘  │  └────────────────────┘  │
    │  ┌────────────────┐  │  ┌────────────────────┐  │
    │  │ 渲染后端         │  │  │ 错误信息           │  │
    │  └────────────────┘  │  │ (错误/日志按钮)     │  │
    │  ┌────────────────┐  │  └────────────────────┘  │
    │  │ 预估             │  │                         │
    │  └────────────────┘  │                         │
    └──────────────────────┴──────────────────────────┘

包含：
    输出分辨率 / 输出帧率（均含自定义选项）/ 输出路径
    硬件信息 (GPU/编码器/显存)
    渲染后端 (自动 / CUDA / OpenGL / CPU，按硬件可用性禁用并说明原因)
    预估 (唯一帧 / 渲染并发 / 编码并发 / 预估耗时)
    进度条 + 阶段文字 + 失败原因 + 日志文件打开按钮

说明：
    渲染/编码方案已固定为「唯一帧 I 帧重复」：
    只编码去重后的唯一帧（-g 1 每帧 IDR），再在 H.264 比特流层面
    按帧数重复，编码量 = 唯一帧数，不再提供模式选择。
"""

import os
import threading
from typing import Optional

from PySide6.QtCore import Qt, Signal, QObject, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFileDialog, QSpinBox,
)

from qfluentwidgets import (
    BodyLabel, StrongBodyLabel, ComboBox, LineEdit, PushButton,
    PrimaryPushButton, ProgressBar,
    InfoBar, InfoBarPosition,
)

# Windows 任务栏进度支持
try:
    from PySide6.QtWinExtras import QWinTaskbarButton
    _HAS_TASKBAR = True
except ImportError:
    _HAS_TASKBAR = False

from core.log import logger, get_log_file_path
from core.renderer import (
    detect_hardware, precompute_frame_states, calc_optimal_workers,
    render_video, get_cuda_render_status, _select_render_backend,
    get_last_render_error, clear_last_render_error,
    clear_renderer_cache,
)
from ui.card_mixin import CardPageMixin

# 常见分辨率选项（末尾追加"自定义"）
RESOLUTIONS = [
    ("720P", 1280, 720),
    ("1080P", 1920, 1080),
    ("2K", 2560, 1440),
    ("4K", 3840, 2160),
]
FPS_OPTIONS = [30, 60, 90, 120]

# 各后端单帧渲染经验耗时（秒，1080P 量级，多流并行下）
# CUDA 使用 GPU 计算核心多流并行，单帧极快；GLES 单线程，略慢于 CUDA；CPU 单线程较慢
BACKEND_FRAME_TIME = {"cuda": 0.003, "opengl": 0.008, "cpu": 0.04}
# 编码耗时系数（相对视频时长）
# NVENC 为硬件编码器，速度极快；AMF/QSV 稍慢；libx264 为纯软件较慢
ENCODE_COEF = {"h264_nvenc": 0.12, "h264_amf": 0.25, "h264_qsv": 0.25, "libx264": 0.8}


class _RenderWorker(QObject):
    """后台渲染载体。QObject 信号跨线程 emit 自动排队到主线程，线程安全。"""

    progress = Signal(int, str)
    finished = Signal(bool, str, str)  # (ok, output_path, error_message)

    def __init__(self, ust_info, output_path, fps, w, h, mode, backend):
        super().__init__()
        self._ust_info = ust_info
        self._output_path = output_path
        self._fps = fps
        self._w = w
        self._h = h
        self._mode = mode
        self._backend = backend

    def run(self):
        def cb(pct: int, stage: str):
            self.progress.emit(pct, stage)

        error = ""
        try:
            ok = render_video(
                self._ust_info, self._output_path,
                fps=self._fps, width=self._w, height=self._h,
                mode=self._mode, render_backend=self._backend,
                progress_callback=cb,
            )
            if not ok:
                error = get_last_render_error()
        except Exception as e:
            logger.exception("渲染导出异常")
            ok = False
            error = f"渲染导出异常：{type(e).__name__}: {e}"
        self.finished.emit(ok, self._output_path, error)


class RenderExportPage(QWidget, CardPageMixin):
    """渲染导出页面 — 作为主窗口导航页面复用主窗口内容区展示。

    左右分栏 + 卡片布局：
        左栏：输出参数、硬件信息、渲染后端、预估
        右栏：状态（进度）、操作（按钮）、错误信息
    """

    _card_border_radius = 10

    def __init__(self, ust_info: dict, main_window: QWidget,
                 settings=None):
        super().__init__()
        self._ust_info = ust_info
        self._main_window = main_window
        self._settings = settings
        self._worker: Optional[_RenderWorker] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._last_stage = ""  # 跟踪阶段切换，重置进度条
        self._output_file = ""  # 成功后的输出文件路径

        # 硬件检测 + CUDA 可用性（一次性缓存）
        self._hw = detect_hardware()
        self._cuda_ok, self._cuda_reason = get_cuda_render_status()

        # Windows 任务栏进度
        self._taskbar_progress = None
        if _HAS_TASKBAR:
            try:
                self._taskbar_button = QWinTaskbarButton(self)
                main_win = self._main_window.windowHandle()
                if main_win:
                    self._taskbar_button.setWindow(main_win)
                    self._taskbar_progress = self._taskbar_button.progress()
                    self._taskbar_progress.setRange(0, 100)
                    self._taskbar_progress.setVisible(False)
            except Exception:
                self._taskbar_progress = None

        # 预估刷新防抖：预计算可能耗时，避免每次改参数都卡 UI
        self._estim_timer = QTimer(self)
        self._estim_timer.setSingleShot(True)
        self._estim_timer.setInterval(250)
        self._estim_timer.timeout.connect(self._refresh_estimates)

        self._build_ui()
        self._apply_card_theme()
        self._refresh_estimates()

    # ===================== UI 构建 =====================

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(10)

        # ---- 标题行 + 返回 ----
        title_row = QHBoxLayout()
        title_row.setSpacing(8)
        back_btn = PushButton("← 返回")
        back_btn.clicked.connect(self._on_back)
        title_row.addWidget(back_btn)
        title_lbl = StrongBodyLabel("渲染导出")
        title_lbl.setStyleSheet("font-size: 16px;")
        title_row.addWidget(title_lbl)
        title_row.addStretch()
        layout.addLayout(title_row)

        # ---- 左右分栏容器 ----
        split_row = QHBoxLayout()
        split_row.setSpacing(12)

        # ====== 左栏：配置卡片 ======
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        # --- 卡片：输出参数 ---
        out_card, out_layout = self._create_section_card("输出参数")

        # 分辨率
        res_row = QHBoxLayout()
        res_row.setSpacing(8)
        res_row.addWidget(BodyLabel("输出分辨率:"))
        self.res_combo = ComboBox()
        for name, w, h in RESOLUTIONS:
            self.res_combo.addItem(f"{name} ({w}×{h})")
        self.res_combo.addItem("自定义")
        self.res_combo.setCurrentIndex(1)  # 默认 1080P
        self.res_combo.currentIndexChanged.connect(self._on_res_changed)
        res_row.addWidget(self.res_combo, 1)
        out_layout.addLayout(res_row)

        # 自定义宽高
        custom_size_row = QHBoxLayout()
        custom_size_row.setSpacing(8)
        custom_size_row.addWidget(BodyLabel("宽:"))
        self.custom_w = QSpinBox()
        self.custom_w.setRange(320, 7680)
        self.custom_w.setValue(1920)
        self.custom_w.setSingleStep(160)
        self.custom_w.setSuffix(" px")
        self.custom_w.valueChanged.connect(self._schedule_refresh)
        custom_size_row.addWidget(self.custom_w, 1)
        custom_size_row.addWidget(BodyLabel("高:"))
        self.custom_h = QSpinBox()
        self.custom_h.setRange(240, 4320)
        self.custom_h.setValue(1080)
        self.custom_h.setSingleStep(90)
        self.custom_h.setSuffix(" px")
        self.custom_h.valueChanged.connect(self._schedule_refresh)
        custom_size_row.addWidget(self.custom_h, 1)
        out_layout.addLayout(custom_size_row)
        self.custom_w.setVisible(False)
        self.custom_h.setVisible(False)

        # 帧率
        fps_row = QHBoxLayout()
        fps_row.setSpacing(8)
        fps_row.addWidget(BodyLabel("输出帧率:"))
        self.fps_combo = ComboBox()
        for f in FPS_OPTIONS:
            self.fps_combo.addItem(str(f))
        self.fps_combo.addItem("自定义")
        self.fps_combo.setCurrentIndex(1)  # 默认 60
        self.fps_combo.currentIndexChanged.connect(self._on_fps_changed)
        fps_row.addWidget(self.fps_combo, 1)
        self.custom_fps = QSpinBox()
        self.custom_fps.setRange(1, 240)
        self.custom_fps.setValue(60)
        self.custom_fps.setSuffix(" fps")
        self.custom_fps.valueChanged.connect(self._schedule_refresh)
        fps_row.addWidget(self.custom_fps)
        out_layout.addLayout(fps_row)
        self.custom_fps.setVisible(False)

        # 输出路径
        path_row = QHBoxLayout()
        path_row.setSpacing(8)
        path_row.addWidget(BodyLabel("输出路径:"))
        self.path_edit = LineEdit()
        default_name = self._ust_info.get("project_info", {}).get("project_name", "") or "未命名"
        self.path_edit.setText(os.path.join(self._default_export_dir(), f"{default_name}.mp4"))
        path_row.addWidget(self.path_edit, 1)
        browse_btn = PushButton("浏览...")
        browse_btn.clicked.connect(self._on_browse)
        path_row.addWidget(browse_btn)
        out_layout.addLayout(path_row)

        left_layout.addWidget(out_card)

        # --- 卡片：硬件信息 ---
        hw_card, hw_layout = self._create_section_card("硬件信息")
        hw = self._hw
        if hw.has_gpu:
            gpu_line = f"GPU: {hw.gpu_name}"
            if hw.gpu_vendor == "nvidia":
                gpu_line += f" ({hw.cuda_cores} CUDA 核心, {hw.vram_total_gb:.1f}GB)"
            encoder_desc = {
                "h264_nvenc": f"NVENC (第{hw.nvenc_generation}代, {hw.nvenc_count}个)",
                "h264_amf": "AMF",
                "h264_qsv": "QSV",
            }.get(hw.encoder_name, hw.encoder_name)
            vram_line = f"可用显存: {hw.vram_usable_gb:.2f}GB"
        else:
            gpu_line = "GPU: 未检测到独立显卡（将使用 CPU 渲染）"
            encoder_desc = "libx264"
            vram_line = "可用显存: 无"
        self.hw_label = BodyLabel(
            f"{gpu_line}\n"
            f"编码器: {encoder_desc}\n"
            f"{vram_line}"
        )
        self.hw_label.setWordWrap(True)
        hw_layout.addWidget(self.hw_label)
        left_layout.addWidget(hw_card)

        # --- 卡片：渲染后端 ---
        backend_card, backend_layout = self._create_section_card("渲染后端")

        self.backend_combo = ComboBox()
        self.backend_combo.addItem("自动选择", userData="auto")
        self.backend_combo.addItem("CUDA (NVIDIA)", userData="cuda")
        self.backend_combo.addItem("OpenGL (通用)", userData="opengl")
        self.backend_combo.addItem("CPU (兼容)", userData="cpu")
        self.backend_combo.setCurrentIndex(0)
        self.backend_combo.currentIndexChanged.connect(self._schedule_refresh)
        backend_layout.addWidget(self.backend_combo)

        self._backend_hint = BodyLabel("")
        self._backend_hint.setWordWrap(True)
        backend_layout.addWidget(self._backend_hint)
        left_layout.addWidget(backend_card)

        left_layout.addStretch()

        # ====== 右栏：状态卡片 ======
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        # --- 卡片：状态（进度） ---
        status_card, status_layout = self._create_section_card("状态")

        # 阶段标签（大号加粗）
        self.phase_label = BodyLabel("")
        font = self.phase_label.font()
        font.setPointSize(16)
        font.setBold(True)
        self.phase_label.setFont(font)
        self.phase_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.phase_label.setVisible(False)
        status_layout.addWidget(self.phase_label)

        # 进度条
        self.progress_bar = ProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setMinimumHeight(8)
        status_layout.addWidget(self.progress_bar)

        # 百分比文字
        self.stage_label = BodyLabel("")
        self.stage_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stage_label.setVisible(False)
        status_layout.addWidget(self.stage_label)

        # 提示文字（未开始渲染时显示）
        self.idle_hint = BodyLabel("调整左侧参数后，点击「开始渲染」导出视频")
        self.idle_hint.setWordWrap(True)
        self.idle_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.idle_hint.setStyleSheet("color: #888888; padding: 20px 0;")
        status_layout.addWidget(self.idle_hint)

        status_layout.addStretch()
        right_layout.addWidget(status_card)

        # --- 卡片：操作 ---
        action_card, action_layout = self._create_section_card("操作")

        self.start_btn = PrimaryPushButton("开始渲染")
        self.start_btn.setMinimumHeight(40)
        self.start_btn.clicked.connect(self._on_start)
        action_layout.addWidget(self.start_btn)

        self.regenerate_btn = PrimaryPushButton("重新生成")
        self.regenerate_btn.clicked.connect(self._on_regenerate)
        self.regenerate_btn.setVisible(False)
        action_layout.addWidget(self.regenerate_btn)

        self.open_video_btn = PushButton("打开视频文件")
        self.open_video_btn.clicked.connect(self._on_open_video)
        self.open_video_btn.setVisible(False)
        action_layout.addWidget(self.open_video_btn)

        right_layout.addWidget(action_card)

        # --- 卡片：预估 ---
        est_card, est_layout = self._create_section_card("预估")
        self.estimate_label = BodyLabel("")
        self.estimate_label.setWordWrap(True)
        est_layout.addWidget(self.estimate_label)
        right_layout.addWidget(est_card)

        # --- 卡片：错误信息 ---
        error_card, error_layout = self._create_section_card("错误信息")

        self.error_label = BodyLabel("")
        self.error_label.setWordWrap(True)
        error_layout.addWidget(self.error_label)

        self.log_btn = PushButton("")
        self.log_btn.clicked.connect(self._open_log)
        error_layout.addWidget(self.log_btn)

        # 默认隐藏错误卡片
        self._set_error_card_visible(False)
        right_layout.addWidget(error_card)

        right_layout.addStretch()

        # ====== 组装左右分栏 ======
        split_row.addWidget(left_panel, 3)  # 左栏 60%
        split_row.addWidget(right_panel, 2)  # 右栏 40%
        layout.addLayout(split_row, 1)

        # 后端可用性提示
        self._apply_backend_availability()

    def _set_error_card_visible(self, visible: bool):
        """显示/隐藏错误卡片内容。"""
        self.error_label.setVisible(visible)
        self.log_btn.setVisible(visible)

    # ===================== 参数解析 =====================

    def _current_resolution(self) -> tuple[int, int]:
        """返回当前选择的分辨率 (w, h)。"""
        idx = self.res_combo.currentIndex()
        if 0 <= idx < len(RESOLUTIONS):
            _, w, h = RESOLUTIONS[idx]
            return w, h
        # 自定义
        return self.custom_w.value(), self.custom_h.value()

    def _current_fps(self) -> int:
        idx = self.fps_combo.currentIndex()
        if 0 <= idx < len(FPS_OPTIONS):
            return FPS_OPTIONS[idx]
        # 自定义
        return self.custom_fps.value()

    def _current_mode(self) -> str:
        """渲染/编码方案已固定为「逐帧渲染 + 逐帧编码」（不做去重）。"""
        return "frame_by_frame"

    def _current_backend(self) -> str:
        data = self.backend_combo.currentData()
        return data if data else "auto"

    def _default_export_dir(self) -> str:
        """输出目录：优先 settings.last_export_dir，否则桌面。"""
        if self._settings is not None:
            d = getattr(self._settings, "last_export_dir", None)
            if d:
                return d
        return os.path.join(os.path.expanduser("~"), "Desktop")

    # ===================== 事件处理 =====================

    def _on_res_changed(self):
        """分辨率下拉框变化：选择"自定义"时显示宽高输入框。"""
        idx = self.res_combo.currentIndex()
        is_custom = idx >= len(RESOLUTIONS)
        self.custom_w.setVisible(is_custom)
        self.custom_h.setVisible(is_custom)
        self._schedule_refresh()

    def _on_fps_changed(self):
        """帧率下拉框变化：选择"自定义"时显示帧率输入框。"""
        idx = self.fps_combo.currentIndex()
        is_custom = idx >= len(FPS_OPTIONS)
        self.custom_fps.setVisible(is_custom)
        self._schedule_refresh()

    def _on_browse(self):
        """选择输出文件路径。"""
        default_name = self._ust_info.get("project_info", {}).get("project_name", "") or "未命名"
        path, _ = QFileDialog.getSaveFileName(
            self, "选择输出文件",
            os.path.join(self._default_export_dir(), f"{default_name}.mp4"),
            "MP4 视频 (*.mp4)",
        )
        if path:
            self.path_edit.setText(path)

    def _open_log(self):
        """用系统默认程序打开日志文件。"""
        try:
            os.startfile(get_log_file_path())
        except Exception as e:
            InfoBar.error("ERcode013", f"无法打开日志文件：{e}",
                          orient=Qt.Orientation.Vertical, duration=3000,
                          parent=self._main_window, position=InfoBarPosition.TOP_RIGHT)

    def _on_back(self):
        """返回进入渲染导出页之前的页面，并释放资源。"""
        self.cleanup()
        self._main_window._exit_render_page()

    def _apply_backend_availability(self):
        """按硬件可用性显示说明，无需禁用选项（用户选了自动会回退）。"""
        hw = self._hw
        cuda_ok = hw.supports_cuda_render and self._cuda_ok

        if hw.supports_cuda_render and not self._cuda_ok:
            self._backend_hint.setText(f"提示：显卡支持 CUDA，但{self._cuda_reason}")
        elif not hw.supports_cuda_render:
            self._backend_hint.setText(
                "提示：当前显卡不支持 CUDA 渲染" +
                ("" if hw.has_gpu else "（无独立显卡，将使用 CPU）")
            )
        else:
            self._backend_hint.setText("")

    # ===================== 预估刷新（防抖） =====================

    def _schedule_refresh(self):
        """延迟触发预估刷新，避免预计算阻塞 UI。"""
        self._estim_timer.start()

    def _refresh_estimates(self):
        """根据当前参数重新预计算并更新模式可用性与预估信息。"""
        w, h = self._current_resolution()
        fps = self._current_fps()
        hw = self._hw

        try:
            frame_states = precompute_frame_states(self._ust_info, fps, w, h)
        except Exception:
            logger.exception("预计算失败")
            self.estimate_label.setText("预估失败")
            return

        state_count = len(frame_states)
        total_frames = sum(s.frame_count for s in frame_states)
        duration_s = total_frames / fps if fps else 0

        wc = calc_optimal_workers(hw, state_count, w, h)

        # ---- 预估耗时 ----
        # 通过 _select_render_backend 获取实际生效的后端（含 fallback 逻辑）
        backend_name, _ = _select_render_backend(hw, self._current_backend())
        per_frame = BACKEND_FRAME_TIME.get(backend_name, 0.02)
        # 所有后端均多线程并行渲染
        render_time = state_count / max(1, wc.render_streams) * per_frame

        # 编码器选择：CPU 后端强制 libx264
        if backend_name == "cpu":
            enc_encoder = "libx264"
        else:
            enc_encoder = hw.encoder_name
        enc_coef = ENCODE_COEF.get(enc_encoder, 1.0)

        # 逐帧渲染 + 逐帧编码方案：不做去重，编码量 = 总输出帧数。
        # 渲染与编码通过生产者-消费者模式并行，总耗时 ≈ max(渲染, 编码)。
        encode_time = total_frames / max(1, fps) * enc_coef
        estimate = max(render_time, encode_time) + 1.0

        if estimate < 60:
            estimate_str = f"≈ {estimate:.0f} 秒"
        else:
            estimate_str = f"≈ {estimate / 60:.1f} 分钟"

        # 后端名称映射
        backend_display = {
            "cuda": "CUDA (NVIDIA)",
            "opengl": "OpenGL (通用)",
            "cpu": "CPU (兼容)",
        }.get(backend_name, backend_name)

        self.estimate_label.setText(
            f"时间区间帧: {state_count}     输出帧: {total_frames} (≈{duration_s:.0f}s)\n"
            f"渲染后端: {backend_display}    方案: 逐帧编码\n"
            f"渲染并发: {wc.render_streams} stream    编码并发: {wc.encode_workers}\n"
            f"预估耗时: {estimate_str}"
        )

    # ===================== 渲染执行 =====================

    def _on_start(self):
        """开始渲染。"""
        if self._running:
            return

        output_path = self.path_edit.text().strip()
        if not output_path:
            InfoBar.error("ERcode010", "请先选择输出路径", orient=Qt.Orientation.Vertical,
                          duration=3000, parent=self._main_window, position=InfoBarPosition.TOP_RIGHT)
            return
        if not output_path.lower().endswith(".mp4"):
            output_path += ".mp4"
        try:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        except Exception:
            pass

        clear_last_render_error()
        w, h = self._current_resolution()
        fps = self._current_fps()
        mode = self._current_mode()
        backend = self._current_backend()

        self._running = True
        self._last_stage = ""
        self.idle_hint.setVisible(False)
        self.start_btn.setVisible(False)
        self.regenerate_btn.setVisible(False)
        self.open_video_btn.setVisible(False)
        self._set_error_card_visible(False)
        self.phase_label.setVisible(True)
        self.phase_label.setText("预计算")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.stage_label.setVisible(True)
        self.stage_label.setText("0%")

        # 任务栏进度
        if self._taskbar_progress:
            self._taskbar_progress.setVisible(True)
            self._taskbar_progress.setValue(0)

        self._worker = _RenderWorker(
            self._ust_info, output_path, fps, w, h, mode, backend,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._thread = threading.Thread(target=self._worker.run, daemon=True)
        self._thread.start()

    def _on_progress(self, pct: int, stage: str):
        """进度回调（主线程）。pct 为总进度 0-100，阶段切换只改标签不重置。"""
        if stage != self._last_stage:
            self._last_stage = stage
            self.phase_label.setText(stage)
        if pct > self.progress_bar.value():
            self.progress_bar.setValue(pct)
        self.stage_label.setText(f"{pct}%")
        if self._taskbar_progress:
            self._taskbar_progress.setValue(pct)

    def _on_finished(self, ok: bool, output_path: str, error_msg: str):
        """渲染完成回调（主线程）。"""
        self._running = False
        # 清理渲染线程和 Worker 引用，释放资源
        self._cleanup_render_resources()
        # 隐藏任务栏进度
        if self._taskbar_progress:
            self._taskbar_progress.setVisible(False)
        if ok:
            self._output_file = output_path
            self.progress_bar.setValue(100)
            self.stage_label.setText("100%")
            self.phase_label.setText("完成")
            # 隐藏进度，显示成功按钮
            self.progress_bar.setVisible(False)
            self.stage_label.setVisible(False)
            self.phase_label.setVisible(False)
            self.idle_hint.setVisible(False)
            self.regenerate_btn.setVisible(True)
            self.open_video_btn.setVisible(True)
            InfoBar.success("成功", f"视频已导出到：{output_path}",
                            orient=Qt.Orientation.Vertical, duration=4000,
                            parent=self._main_window, position=InfoBarPosition.TOP_RIGHT)
        else:
            log_path = get_log_file_path()
            detail = error_msg.strip() or "未知错误，详情见日志"
            self.phase_label.setText("失败")
            self.stage_label.setText("渲染失败")
            self.error_label.setText(f"渲染失败：{detail}")
            self._set_error_card_visible(True)
            self.log_btn.setText(f"打开日志文件")
            self.start_btn.setVisible(True)
            InfoBar.error("ERcode012", f"渲染导出失败，日志：{log_path}",
                          orient=Qt.Orientation.Vertical, duration=5000,
                          parent=self._main_window, position=InfoBarPosition.TOP_RIGHT)

    # ===================== 重新进入页面 =====================

    def _on_regenerate(self):
        """重新生成：隐藏成功按钮，恢复开始按钮。"""
        self.regenerate_btn.setVisible(False)
        self.open_video_btn.setVisible(False)
        self.start_btn.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.phase_label.setVisible(False)
        self.stage_label.setVisible(False)
        self._set_error_card_visible(False)
        self.idle_hint.setVisible(True)
        self._last_stage = ""

    def _cleanup_render_resources(self):
        """清理渲染线程和 Worker 引用，释放线程资源。"""
        # 清理 worker 信号连接，断开所有信号
        if self._worker is not None:
            try:
                self._worker.progress.disconnect()
            except Exception:
                pass
            try:
                self._worker.finished.disconnect()
            except Exception:
                pass
            self._worker.deleteLater()
            self._worker = None
        # 线程引用置空（threading.Thread 是 daemon，自然结束即可）
        self._thread = None

    def cleanup(self):
        """页面销毁/切换时释放所有资源，包括渲染线程和缓存数据。"""
        self._cleanup_render_resources()
        # 释放渲染器模块级缓存（字形缓存、CUDA 上下文等）
        clear_renderer_cache()
        # 释放 Settings 中的缓存数据（音符列表、解析结果等）
        if self._settings is not None:
            self._settings.clear_cached_data()
        # 释放硬件检测缓存
        self._hw = None
        # 销毁任务栏进度对象
        if self._taskbar_progress:
            try:
                self._taskbar_progress = None
            except Exception:
                pass
        # 停止预估定时器
        if self._estim_timer:
            try:
                self._estim_timer.stop()
            except Exception:
                pass

    def _on_open_video(self):
        """打开视频文件，然后重置到初始状态。"""
        if self._output_file and os.path.exists(self._output_file):
            try:
                os.startfile(self._output_file)
            except Exception as e:
                InfoBar.error("ERcode014", f"无法打开视频文件：{e}",
                              orient=Qt.Orientation.Vertical, duration=3000,
                              parent=self._main_window, position=InfoBarPosition.TOP_RIGHT)
        self.reset_state()

    def set_ust_info(self, ust_info: dict):
        """切换工程后更新数据与输出路径。"""
        self._ust_info = ust_info
        default_name = ust_info.get("project_info", {}).get("project_name", "") or "未命名"
        self.path_edit.setText(os.path.join(self._default_export_dir(), f"{default_name}.mp4"))
        self._refresh_estimates()

    def reset_state(self):
        """再次进入页面时重置渲染状态，并清理旧线程资源。"""
        self._cleanup_render_resources()
        self._running = False
        self._last_stage = ""
        self.start_btn.setVisible(True)
        self.regenerate_btn.setVisible(False)
        self.open_video_btn.setVisible(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.phase_label.setVisible(False)
        self.stage_label.setVisible(False)
        self._set_error_card_visible(False)
        self.idle_hint.setVisible(True)
        self._refresh_estimates()