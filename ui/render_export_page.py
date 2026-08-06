# render_export_page.py — 渲染导出页面（复用主窗口内容区）
"""渲染导出设置 + 进度页面，嵌入主窗口而非独立弹窗。

包含：
    输出分辨率 / 输出帧率（均含自定义选项）/ 输出路径
    硬件信息 (GPU/编码器/显存)
    渲染模式 (渲染完再编码 / 边渲染边编码 / 自动)
    渲染后端 (自动 / CUDA / OpenGL / CPU，按硬件可用性禁用并说明原因)
    预估 (唯一帧 / 渲染并发 / 编码并发 / 预估耗时)
    进度条 + 阶段文字 + 失败原因 + 日志文件打开按钮
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
)

# 常见分辨率选项（末尾追加"自定义"）
RESOLUTIONS = [
    ("720P", 1280, 720),
    ("1080P", 1920, 1080),
    ("2K", 2560, 1440),
    ("4K", 3840, 2160),
]
FPS_OPTIONS = [30, 60, 90, 120]

# 各后端单帧渲染经验耗时（秒，1080P 量级）
BACKEND_FRAME_TIME = {"cuda": 0.006, "opengl": 0.015, "cpu": 0.05}
# 编码耗时系数（相对视频时长）
ENCODE_COEF = {"h264_nvenc": 0.4, "h264_amf": 0.6, "h264_qsv": 0.6, "libx264": 1.2}


class _RenderWorker(QObject):
    """后台渲染载体。QObject 信号跨线程 emit 自动排队到主线程，线程安全。"""

    progress = Signal(int, int, str)
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
        def cb(current, total, stage):
            self.progress.emit(current, total, stage)

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


class RenderExportPage(QWidget):
    """渲染导出页面 — 作为主窗口导航页面复用主窗口内容区展示。"""

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
        self._refresh_estimates()

    # ===================== UI 构建 =====================

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(10)

        # ---- 标题行 + 返回 ----
        title_row = QHBoxLayout()
        title_row.setSpacing(8)
        title_row.addWidget(StrongBodyLabel("渲染导出"))
        title_row.addStretch()
        back_btn = PushButton("返回")
        back_btn.clicked.connect(self._on_back)
        title_row.addWidget(back_btn)
        layout.addLayout(title_row)

        # ---- 输出参数 ----
        layout.addWidget(StrongBodyLabel("输出参数"))

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
        res_row.addWidget(self.res_combo)
        res_row.addWidget(BodyLabel("宽:"))
        self.custom_w = QSpinBox()
        self.custom_w.setRange(320, 7680)
        self.custom_w.setValue(1920)
        self.custom_w.setSingleStep(160)
        self.custom_w.setSuffix(" px")
        self.custom_w.valueChanged.connect(self._schedule_refresh)
        res_row.addWidget(self.custom_w)
        res_row.addWidget(BodyLabel("高:"))
        self.custom_h = QSpinBox()
        self.custom_h.setRange(240, 4320)
        self.custom_h.setValue(1080)
        self.custom_h.setSingleStep(90)
        self.custom_h.setSuffix(" px")
        self.custom_h.valueChanged.connect(self._schedule_refresh)
        res_row.addWidget(self.custom_h)
        res_row.addStretch()
        layout.addLayout(res_row)
        # 仅在选择"自定义"时显示宽高输入框
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
        fps_row.addWidget(self.fps_combo)
        self.custom_fps = QSpinBox()
        self.custom_fps.setRange(1, 240)
        self.custom_fps.setValue(60)
        self.custom_fps.setSuffix(" fps")
        self.custom_fps.valueChanged.connect(self._schedule_refresh)
        fps_row.addWidget(self.custom_fps)
        fps_row.addStretch()
        layout.addLayout(fps_row)
        # 仅在选择"自定义"时显示帧率输入框
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
        layout.addLayout(path_row)

        # ---- 硬件信息 ----
        layout.addWidget(StrongBodyLabel("硬件信息"))
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
        layout.addWidget(self.hw_label)

        # ---- 渲染模式（下拉框，默认自动选择） ----
        layout.addWidget(StrongBodyLabel("渲染模式"))
        mode_row = QHBoxLayout()
        self.mode_combo = ComboBox()
        self.mode_combo.addItem("自动选择", "auto")
        self.mode_combo.addItem("渲染完再编码", "batch")
        self.mode_combo.addItem("边渲染边编码", "stream")
        self.mode_combo.setCurrentIndex(0)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self.mode_combo, 1)
        layout.addLayout(mode_row)
        self._batch_hint = BodyLabel("")
        self._batch_hint.setStyleSheet("color: #888888;")
        self._batch_hint.setWordWrap(True)
        layout.addWidget(self._batch_hint)
        self._batch_disabled = False

        # ---- 渲染后端（下拉框，默认自动选择） ----
        layout.addWidget(StrongBodyLabel("渲染后端"))
        backend_row = QHBoxLayout()
        self.backend_combo = ComboBox()
        self.backend_combo.addItem("自动选择", "auto")
        self.backend_combo.addItem("CUDA (NVIDIA)", "cuda")
        self.backend_combo.addItem("OpenGL", "opengl")
        self.backend_combo.addItem("CPU (兼容)", "cpu")
        self.backend_combo.setCurrentIndex(0)
        self.backend_combo.currentIndexChanged.connect(self._schedule_refresh)
        backend_row.addWidget(self.backend_combo, 1)
        layout.addLayout(backend_row)
        self._backend_hint = BodyLabel("")
        self._backend_hint.setStyleSheet("color: #888888;")
        self._backend_hint.setWordWrap(True)
        layout.addWidget(self._backend_hint)

        # ---- 预估 ----
        layout.addWidget(StrongBodyLabel("预估"))
        self.estimate_label = BodyLabel("")
        layout.addWidget(self.estimate_label)

        layout.addStretch()

        # ---- 操作区 ----
        self.start_btn = PrimaryPushButton("开始渲染")
        self.start_btn.clicked.connect(self._on_start)
        layout.addWidget(self.start_btn)

        # 阶段标签
        self.phase_label = BodyLabel("")
        self.phase_label.setVisible(False)
        font = self.phase_label.font()
        font.setPointSize(14)
        font.setBold(True)
        self.phase_label.setFont(font)
        layout.addWidget(self.phase_label)

        self.progress_bar = ProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.stage_label = BodyLabel("")
        self.stage_label.setVisible(False)
        layout.addWidget(self.stage_label)

        self.error_label = BodyLabel("")
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet("color: #c42b1c;")
        self.error_label.setVisible(False)
        layout.addWidget(self.error_label)

        self.log_btn = PushButton("")
        self.log_btn.clicked.connect(self._open_log)
        self.log_btn.setVisible(False)
        layout.addWidget(self.log_btn)

        # 成功后的两个按钮
        self.regenerate_btn = PrimaryPushButton("重新生成")
        self.regenerate_btn.clicked.connect(self._on_regenerate)
        self.regenerate_btn.setVisible(False)
        layout.addWidget(self.regenerate_btn)

        self.open_video_btn = PushButton("打开视频文件")
        self.open_video_btn.clicked.connect(self._on_open_video)
        self.open_video_btn.setVisible(False)
        layout.addWidget(self.open_video_btn)

        self._apply_backend_availability()

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
        return self.mode_combo.currentData()

    def _current_backend(self) -> str:
        return self.backend_combo.currentData()

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
        """返回进入渲染导出页之前的页面。"""
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

    def _on_mode_changed(self, index: int):
        """拦截被禁用的 batch 选项，弹回自动选择。"""
        if index == 1 and self._batch_disabled:
            # 用户尝试选择被禁用的「渲染完再编码」，弹回自动选择
            self.mode_combo.blockSignals(True)
            self.mode_combo.setCurrentIndex(0)
            self.mode_combo.blockSignals(False)
            return
        self._schedule_refresh()

    def _schedule_refresh(self):
        """延迟触发预估刷新，避免预计算阻塞 UI。"""
        self._estim_timer.start()

    def _refresh_estimates(self):
        """根据当前参数重新预计算并更新模式可用性与预估信息。"""
        w, h = self._current_resolution()
        fps = self._current_fps()
        hw = self._hw

        try:
            states = precompute_frame_states(self._ust_info, fps, w, h)
        except Exception:
            logger.exception("预计算失败")
            self.estimate_label.setText("预估失败")
            return

        unique = len(states)
        total_frames = sum(s.frame_count for s in states)
        duration_s = total_frames / fps if fps else 0

        wc = calc_optimal_workers(hw, unique, w, h)

        # ---- 模式可用性 ----
        total_volume = unique * wc.per_frame_gb
        batch_ok = hw.vram_usable_gb > 0 and total_volume <= hw.vram_usable_gb
        batch_index = 1  # 0=auto, 1=batch, 2=stream
        self._batch_disabled = not batch_ok
        if not batch_ok:
            need = max(0.0, total_volume - hw.vram_usable_gb)
            self._batch_hint.setText(
                f"「渲染完再编码」需要 ≥ {total_volume:.1f}GB 显存"
                f"（当前 {hw.vram_usable_gb:.2f}GB，不足 {need:.1f}GB），已禁用"
            )
            # 修改 batch 项文本，提示不可用
            if self.mode_combo.itemText(batch_index) != "渲染完再编码 (显存不足)":
                self.mode_combo.setItemText(batch_index, "渲染完再编码 (显存不足)")
            # 当前模式被禁用时自动切回"自动选择"
            if self.mode_combo.currentData() == "batch":
                self.mode_combo.setCurrentIndex(0)
        else:
            self._batch_hint.setText("")
            if self.mode_combo.itemText(batch_index) != "渲染完再编码":
                self.mode_combo.setItemText(batch_index, "渲染完再编码")

        # ---- 预估耗时 ----
        backend = self._current_backend()
        if backend == "auto":
            backend_name, _ = _select_render_backend(hw, "auto")
        else:
            backend_name = backend
        per_frame = BACKEND_FRAME_TIME.get(backend_name, 0.02)
        # 所有后端均多线程并行渲染
        render_time = unique / max(1, wc.render_streams) * per_frame
        enc_coef = ENCODE_COEF.get(hw.encoder_name, 1.0)
        encode_time = duration_s * enc_coef
        estimate = render_time + encode_time + 1.0  # +1s 预计算/IO 开销

        if estimate < 60:
            estimate_str = f"≈ {estimate:.0f} 秒"
        else:
            estimate_str = f"≈ {estimate / 60:.1f} 分钟"

        self.estimate_label.setText(
            f"唯一帧: {unique}     输出帧: {total_frames} (≈{duration_s:.0f}s)\n"
            f"渲染并发: {wc.render_streams} stream    编码并发: {wc.encode_workers}\n"
            f"渲染后端: {backend_name}    预估耗时: {estimate_str}"
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
        self.start_btn.setEnabled(False)
        self.start_btn.setVisible(False)
        self.regenerate_btn.setVisible(False)
        self.open_video_btn.setVisible(False)
        self.phase_label.setVisible(True)
        self.phase_label.setText("预计算")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.stage_label.setVisible(True)
        self.stage_label.setText("")
        self.error_label.setVisible(False)
        self.log_btn.setVisible(False)

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

    def _on_progress(self, current: int, total: int, stage: str):
        """进度回调（主线程）。总进度 0-100，阶段切换只改标签不重置。"""
        if stage != self._last_stage:
            self._last_stage = stage
            self.phase_label.setText(stage)
        pct = int(current / total * 100) if total > 0 else 0
        if pct > self.progress_bar.value():
            self.progress_bar.setValue(pct)
        self.stage_label.setText(f"{pct}%")
        if self._taskbar_progress:
            self._taskbar_progress.setValue(pct)

    def _on_finished(self, ok: bool, output_path: str, error_msg: str):
        """渲染完成回调（主线程）。"""
        self._running = False
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
            self.error_label.setText(f"渲染失败（ERcode012）：{detail}")
            self.error_label.setVisible(True)
            self.log_btn.setText(f"打开日志文件（{log_path}）")
            self.log_btn.setVisible(True)
            self.start_btn.setVisible(True)
            self.start_btn.setEnabled(True)
            InfoBar.error("ERcode012", f"渲染导出失败，日志：{log_path}",
                          orient=Qt.Orientation.Vertical, duration=5000,
                          parent=self._main_window, position=InfoBarPosition.TOP_RIGHT)

    # ===================== 重新进入页面 =====================

    def _on_regenerate(self):
        """重新生成：隐藏成功按钮，恢复开始按钮。"""
        self.regenerate_btn.setVisible(False)
        self.open_video_btn.setVisible(False)
        self.start_btn.setVisible(True)
        self.start_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.phase_label.setVisible(False)
        self.stage_label.setVisible(False)
        self._last_stage = ""

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
        """再次进入页面时重置渲染状态。"""
        self._running = False
        self._last_stage = ""
        self.start_btn.setVisible(True)
        self.start_btn.setEnabled(True)
        self.regenerate_btn.setVisible(False)
        self.open_video_btn.setVisible(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.phase_label.setVisible(False)
        self.stage_label.setVisible(False)
        self.error_label.setVisible(False)
        self.log_btn.setVisible(False)
        self._refresh_estimates()
