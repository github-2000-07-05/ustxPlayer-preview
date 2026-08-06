# file_page.py — "文件" 导航页
"""USTX 文件选择和解析。"""

import os

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFileDialog, QDialog, QFrame,
)
from PySide6.QtCore import Qt, QThread, QTimer, Signal, QObject

from qfluentwidgets import (
    LineEdit, PushButton, TextEdit, CheckBox,
    BodyLabel, StrongBodyLabel, HorizontalSeparator,
    InfoBar, InfoBarPosition, Theme, qconfig, themeColor,
)

from core.settings_manager import SettingsManager
from core.log import logger
import core.ustxreader as ur
from ui.track_select_dialog import TrackSelectDialog
from ui.card_mixin import CardPageMixin


# ===================== 后台解析工作线程 =====================

class _ParseWorker(QObject):
    """在后台线程中检测/解析 USTX 文件，避免阻塞 UI。"""
    tracks_ready = Signal(object)  # 音轨检测完成，携带可解析音轨列表
    finished = Signal(object)      # ust_info dict
    failed = Signal(str)           # 错误信息

    def __init__(self, ustx_path: str, parent=None):
        super().__init__(parent)
        self._path = ustx_path
        self._track_index: int | None = None

    def detect_tracks(self):
        try:
            tracks = ur.get_ustx_tracks(self._path)
            self.tracks_ready.emit(tracks)
        except Exception as e:
            logger.exception("后台检测 USTX 音轨失败")
            self.failed.emit(str(e))

    def parse(self):
        try:
            result = ur.get_ustx_info(self._path, track_index=self._track_index)
            self.finished.emit(result)
        except Exception as e:
            logger.exception("后台解析 USTX 失败")
            self.failed.emit(str(e))


class FilePage(QWidget, CardPageMixin):
    """文件选择页面 - 支持 USTX 文件解析。"""

    _card_border_radius = 10

    def __init__(self, settings: SettingsManager, parent=None):
        super().__init__(parent=parent)
        self._s = settings
        self._parse_thread: QThread | None = None  # 后台解析线程
        self._parse_worker: _ParseWorker | None = None
        self._parsing = False  # 是否正在解析
        self._pending_notes: list | None = None  # 延迟写入的音符数据
        self._setup_ui()
        self._connect_signals()
        self._apply_card_theme()

    # ===================== UI 构建 =====================

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(8)

        # ========== USTX 文件区（卡片） ==========
        ustx_card, ustx_layout = self._create_section_card("USTX 文件")

        ustx_row = QHBoxLayout()
        ustx_row.setSpacing(8)
        ustx_row.addWidget(BodyLabel("USTX 文件:"))
        self.ust_edit = LineEdit()
        self.ust_edit.setPlaceholderText("请选择或拖入 .ustx 文件路径...")
        ustx_row.addWidget(self.ust_edit, 1)
        self.select_ust_btn = PushButton("选择 USTX")
        ustx_row.addWidget(self.select_ust_btn)
        ustx_layout.addLayout(ustx_row)

        self.cb_curve = CheckBox("显示音高线变化")
        self.cb_curve.setChecked(self._s.curve_show)
        ustx_layout.addWidget(self.cb_curve)

        layout.addWidget(ustx_card)

        # ========== 音频文件区（卡片） ==========
        audio_card, audio_layout = self._create_section_card("音频文件")

        audio_row = QHBoxLayout()
        audio_row.setSpacing(8)
        audio_row.addWidget(BodyLabel("音频文件:"))
        self.audio_edit = LineEdit()
        self.audio_edit.setPlaceholderText("请选择 .mp3/.wav/.flac 音频文件...")
        audio_row.addWidget(self.audio_edit, 1)
        self.select_audio_btn = PushButton("选择音频")
        audio_row.addWidget(self.select_audio_btn)
        audio_layout.addLayout(audio_row)

        layout.addWidget(audio_card)

        # ========== 歌词文件区（卡片） ==========
        lyric_card, lyric_layout = self._create_section_card("歌词文件")

        self.cb_show_lyric = CheckBox("播放器中显示歌词")
        self.cb_show_lyric.setChecked(self._s.show_lyric)
        lyric_layout.addWidget(self.cb_show_lyric)

        # 自动隐藏歌词
        autohide_row = QHBoxLayout()
        autohide_row.setSpacing(6)
        self.cb_autohide = CheckBox("间奏自动隐藏歌词")
        self.cb_autohide.setChecked(self._s.show_lyric_autohide)
        autohide_row.addWidget(self.cb_autohide)
        autohide_row.addWidget(BodyLabel("阈值(秒):"))
        self.autohide_threshold_edit = LineEdit()
        self.autohide_threshold_edit.setPlaceholderText("3.0")
        self.autohide_threshold_edit.setText(str(self._s.lyric_autohide_threshold))
        self.autohide_threshold_edit.setFixedWidth(50)
        autohide_row.addWidget(self.autohide_threshold_edit)
        autohide_row.addStretch()
        lyric_layout.addLayout(autohide_row)

        lrc_row = QHBoxLayout()
        lrc_row.setSpacing(8)
        lrc_row.addWidget(BodyLabel("歌词文件 (.lrc):"))
        self.lyric_edit = LineEdit()
        self.lyric_edit.setPlaceholderText("请选择 .lrc 歌词文件...")
        lrc_row.addWidget(self.lyric_edit, 1)
        self.select_lyric_btn = PushButton("选择歌词")
        lrc_row.addWidget(self.select_lyric_btn)
        lyric_layout.addLayout(lrc_row)

        layout.addWidget(lyric_card)

        # ========== 操作按钮行 ==========
        action_row = QHBoxLayout()
        action_row.setSpacing(8)
        self.match_btn = PushButton("解析 USTX")
        self.match_btn.setEnabled(False)
        action_row.addWidget(self.match_btn)
        action_row.addStretch()
        layout.addLayout(action_row)

        # ========== 解析结果区（卡片） ==========
        result_card, result_layout = self._create_section_card("解析结果")

        self.result_edit = TextEdit()
        self.result_edit.setReadOnly(True)
        self.result_edit.setPlaceholderText("点击 [解析 USTX] 查看解析结果...")
        self.result_edit.setMinimumHeight(200)
        result_layout.addWidget(self.result_edit)

        layout.addWidget(result_card)

    def _connect_signals(self):
        self.ust_edit.textChanged.connect(self._on_ust_path_changed)
        self.select_ust_btn.clicked.connect(self._on_select_ust)

        self.audio_edit.textChanged.connect(
            lambda v: setattr(self._s, "audio_path", v)
        )
        self.select_audio_btn.clicked.connect(self._on_select_audio)

        self.match_btn.clicked.connect(self._on_match)

        def _on_curve(v):
            setattr(self._s, "curve_show", v == Qt.CheckState.Checked)
        self.cb_curve.checkStateChanged.connect(_on_curve)

        self.cb_show_lyric.checkStateChanged.connect(
            lambda v: setattr(self._s, "show_lyric", v == Qt.CheckState.Checked)
        )

        self.cb_autohide.checkStateChanged.connect(
            lambda v: setattr(self._s, "show_lyric_autohide", v == Qt.CheckState.Checked)
        )
        self.autohide_threshold_edit.textChanged.connect(self._on_threshold_changed)

        self.lyric_edit.textChanged.connect(
            lambda v: setattr(self._s, "lrc_path", v)
        )
        self.select_lyric_btn.clicked.connect(self._on_select_lyric)

        # 监听设置变化同步到 UI
        self._s.ustx_path_changed.connect(self._on_settings_ustx_changed)

        # 主题变化时刷新卡片样式
        try:
            self._s.theme_mode_changed.connect(self._apply_card_theme)
        except AttributeError:
            pass

    def _on_settings_ustx_changed(self, path: str):
        """Settings 端路径变化时同步到 UI。"""
        self.ust_edit.blockSignals(True)
        self.ust_edit.setText(path)
        self.ust_edit.blockSignals(False)
        self._check_match_ready()

    def _on_threshold_changed(self, text: str):
        """自动隐藏阈值：仅允许非负整数，非法输入自动取整。"""
        if not text or text == ".":
            return
        try:
            val = float(text)
            ival = round(val)
            if ival < 0:
                ival = 0
            self._s.lyric_autohide_threshold = float(ival)
            # 去除小数点显示
            if str(ival) != text:
                self.autohide_threshold_edit.blockSignals(True)
                self.autohide_threshold_edit.setText(str(ival))
                self.autohide_threshold_edit.blockSignals(False)
        except ValueError:
            pass

    def sync_all_from_settings(self):
        """从 settings 同步所有 UI 控件并自动解析。"""
        s = self._s
        self.ust_edit.blockSignals(True)
        self.ust_edit.setText(s.ustx_path)
        self.ust_edit.blockSignals(False)
        self.audio_edit.blockSignals(True)
        self.audio_edit.setText(s.audio_path)
        self.audio_edit.blockSignals(False)
        self.lyric_edit.blockSignals(True)
        self.lyric_edit.setText(s.lrc_path)
        self.lyric_edit.blockSignals(False)
        self.cb_show_lyric.setChecked(s.show_lyric)
        self.cb_curve.setChecked(s.curve_show)
        self.cb_autohide.setChecked(s.show_lyric_autohide)
        self.autohide_threshold_edit.blockSignals(True)
        self.autohide_threshold_edit.setText(str(int(s.lyric_autohide_threshold)))
        self.autohide_threshold_edit.blockSignals(False)
        self._check_match_ready()

    def _on_select_ust(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择USTX文件",
            os.path.dirname(self._s.ustx_path) if self._s.ustx_path else "",
            "USTX文件 (*.ustx);;所有文件 (*.*)",
        )
        if file_path:
            self.ust_edit.setText(file_path)
            # 自动解析
            self._on_match()

    def _on_ust_path_changed(self, path: str):
        """USTX 路径变化时更新 settings 和解析按钮状态。"""
        self._s.ustx_path = path
        self._check_match_ready()

    def _on_select_audio(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择音频文件",
            os.path.dirname(self._s.audio_path) if self._s.audio_path else "",
            "音频文件 (*.mp3 *.wav *.flac);;所有文件 (*.*)",
        )
        if file_path:
            self.audio_edit.setText(file_path)

    def _on_select_lyric(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择歌词文件",
            os.path.dirname(self._s.lrc_path) if self._s.lrc_path else "",
            "歌词文件 (*.lrc);;所有文件 (*.*)",
        )
        if file_path:
            self.lyric_edit.setText(file_path)

    def _check_match_ready(self):
        """检测是否选择了 USTX 文件，启用解析按钮。"""
        ust_ok = bool(self._s.ustx_path.strip() and os.path.exists(self._s.ustx_path.strip()))
        self.match_btn.setEnabled(ust_ok)

    def _on_match(self):
        """解析 USTX 文件：先检测音轨，多条时弹窗选择，再解析所选音轨。

        检测与解析均在后台线程执行，避免阻塞 UI。
        """
        if self._parsing:
            return
        # 项目名为空时自动使用 ustx 文件名（不含扩展名）
        if self._s.maybe_fill_project_name_from_ustx():
            InfoBar.success("提示", f"工程名为空，已自动填充为：{self._s.project_name}",
                            orient=Qt.Orientation.Vertical, duration=2000, parent=self.window(), position=InfoBarPosition.TOP_RIGHT)
        ust_path = self._s.ustx_path.strip()

        if not ust_path or not os.path.exists(ust_path):
            InfoBar.warning("提示", "USTX 文件无效", orient=Qt.Orientation.Vertical, duration=2000,
                           parent=self.window(), position=InfoBarPosition.TOP_RIGHT)
            return

        self._set_parsing_state(True)
        self._start_thread("detect")

    def _set_parsing_state(self, parsing: bool):
        """切换解析中状态：禁用/恢复解析按钮。"""
        self._parsing = parsing
        self.match_btn.setEnabled(not parsing)
        self.match_btn.setText("解析中…" if parsing else "解析 USTX")

    def _start_thread(self, mode: str, track_index: int | None = None):
        """启动后台线程执行指定任务（detect 检测音轨 / parse 解析所选音轨）。"""
        try:
            self._parse_thread = QThread(self)
            self._parse_worker = _ParseWorker(self._s.ustx_path.strip())
            self._parse_worker._track_index = track_index
            self._parse_worker.moveToThread(self._parse_thread)

            if mode == "detect":
                self._parse_thread.started.connect(self._parse_worker.detect_tracks)
                self._parse_worker.tracks_ready.connect(self._on_tracks_detected)
            else:
                self._parse_thread.started.connect(self._parse_worker.parse)
                self._parse_worker.finished.connect(self._on_parse_done)
            self._parse_worker.failed.connect(self._on_parse_failed)
            self._parse_thread.finished.connect(self._on_parse_thread_finished)

            self._parse_thread.start()
        except Exception:
            logger.exception("启动解析线程失败")
            self._set_parsing_state(False)
            InfoBar.error("ERcode004", "解析未启动，请重试", orient=Qt.Orientation.Vertical, duration=2000,
                          parent=self.window(), position=InfoBarPosition.TOP_RIGHT)

    def _on_tracks_detected(self, tracks: list):
        """音轨检测完成回调（主线程）：多条可解析音轨时弹出选择窗口。"""
        # 无任何可解析音轨（文件无音符）→ 直接结束
        if not tracks:
            self._set_parsing_state(False)
            InfoBar.warning("提示", "文件中没有音符", orient=Qt.Orientation.Vertical, duration=2000,
                           parent=self.window(), position=InfoBarPosition.TOP_RIGHT)
            return

        # 仅一条可解析音轨 → 无需选择，直接解析
        if len(tracks) == 1:
            self._start_thread("parse", track_index=tracks[0]["index"])
            return

        # 多条可解析音轨 → 弹出原生风格选择窗口
        dialog = TrackSelectDialog(tracks, parent=self.window())
        if dialog.exec() != QDialog.DialogCode.Accepted:
            # 用户取消 → 中止解析
            self._set_parsing_state(False)
            return
        track_index = dialog.selected_index()
        if track_index is None:
            self._set_parsing_state(False)
            return
        self._start_thread("parse", track_index=track_index)

    def _on_parse_thread_finished(self):
        """后台解析线程结束时清理引用。"""
        self._parse_thread = None
        self._parse_worker = None

    def _on_parse_done(self, ust_info: dict):
        """后台解析完成回调（主线程）。"""
        self._parsing = False
        self.match_btn.setEnabled(True)
        self.match_btn.setText("解析 USTX")

        notes = ust_info.get("notes", [])
        ust_path = self._s.ustx_path.strip()
        # 缓存完整解析结果，供 _on_play 复用，避免重复解析同一文件
        self._s.cached_ust_info = {"path": ust_path, "info": ust_info}

        if not notes:
            InfoBar.warning("提示", "文件中没有音符", orient=Qt.Orientation.Vertical, duration=2000,
                           parent=self.window(), position=InfoBarPosition.TOP_RIGHT)
            return

        info_lines = [
            "═══════════════════════════════════════",
            "     USTX 解析报告",
            "═══════════════════════════════════════",
            f"  文件:         {os.path.basename(ust_path)}",
            f"  版本:         {ust_info.get('version', 'unknown')}",
            f"  BPM:          {ust_info.get('tempo', 120)}",
            f"  音轨:         {ust_info.get('track_name', '全部音轨')}",
            f"  轨道数:       {ust_info.get('tracks', 1)}",
            f"  音符数:       {len(notes)}",
            "",
            "  解析完成",
            "═══════════════════════════════════════",
        ]
        self.result_edit.setPlainText("\n".join(info_lines))

        # 延迟存储音符数据，避免表格重建阻塞当前帧
        self._pending_notes = notes
        QTimer.singleShot(0, self._apply_notes)

        InfoBar.success("解析完成", f"成功解析 {len(notes)} 个音符",
                       orient=Qt.Orientation.Vertical, duration=2000, parent=self.window(), position=InfoBarPosition.TOP_RIGHT)

    def _apply_notes(self):
        """延迟应用解析结果到 Settings（表格重建在独立帧执行）。"""
        if self._pending_notes is not None:
            try:
                self._s.ustx_notes = self._pending_notes
                # 如果是 UPLR 导入的延迟解析，恢复 note_styles
                self._s._apply_deferred_uplr_styles()
            finally:
                self._pending_notes = None

    def _on_parse_failed(self, err_msg: str):
        """后台解析失败回调（主线程）。"""
        self._parsing = False
        self.match_btn.setEnabled(True)
        self.match_btn.setText("解析 USTX")
        InfoBar.error("ERcode004", f"USTX 解析失败: {err_msg}", orient=Qt.Orientation.Vertical, duration=3000,
                     parent=self.window(), position=InfoBarPosition.TOP_RIGHT)

    def cleanup_parse_thread(self):
        """退出前等待后台解析线程结束，避免 QThread 仍在运行时被回收崩溃。"""
        if self._parse_thread is not None:
            try:
                self._parse_thread.quit()
                self._parse_thread.wait(3000)
            except Exception:
                logger.exception("等待解析线程结束失败")
            self._parse_thread = None
            self._parse_worker = None
