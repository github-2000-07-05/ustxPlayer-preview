# settings_manager.py — 配置管理器
"""Settings.ini 配置读写 + .uplr 工程文件导入/导出。

通过 Qt Signal 通知 UI 所有配置变更。

支持两种导出模式：
- 普通导出：兼容原项目 TS player，导入需重新解析
- 精简导出（BETA）：预解析数据直接存储，导入时零解析、零缓存文件
"""

import os
import sys
import json
import copy
import configparser
from typing import Optional

from PySide6.QtCore import QObject, Signal

from core.log import logger


class ProjectFileMissingError(Exception):
    """工程文件中引用的文件路径不存在。

    import_uplr 加载配置后校验 ustx/lrc/audio/custom_font_paths 路径，
    收集全部缺失项后一次性抛出。此时配置已加载到内存，仅文件不可用，
    调用方捕获后仍应同步 UI 供用户重新选择文件。
    """

    def __init__(self, missing: list[tuple[str, str]]):
        self.missing = missing
        lines = "\n".join(f"  - {label}: {path}" for label, path in missing)
        super().__init__(f"以下文件路径不存在:\n{lines}")


class SettingsManager(QObject):
    """应用配置管理器，集中管理所有设置项。

    每个配置项对应一个属性，修改时发出对应的 Signal。
    UI 层通过 connect/setValue 模式绑定。
    """

    # ===================== 信号定义 =====================
    # 字符串信号
    ustx_path_changed = Signal(str)
    project_name_changed = Signal(str)
    song_name_changed = Signal(str)
    song_author_changed = Signal(str)
    ust_author_changed = Signal(str)
    bg_color_changed = Signal(str)
    note_color_changed = Signal(str)
    lyric_color_changed = Signal(str)
    pitch_curve_color_changed = Signal(str)
    lyric_pos_changed = Signal(str)
    lrc_path_changed = Signal(str)
    audio_path_changed = Signal(str)
    silent_display_changed = Signal(str)
    silent_custom_text_changed = Signal(str)
    end_display_changed = Signal(str)
    end_custom_text_changed = Signal(str)
    pitch_placeholder_changed = Signal(str)
    pitch_custom_text_changed = Signal(str)

    # 字体（逐字歌词字体 / 歌词及信息字体 分开控制）
    word_lyric_font_family_changed = Signal(str)
    info_font_family_changed = Signal(str)
    custom_font_paths_changed = Signal(list)

    # 歌词及信息颜色（独立于样式）
    info_text_color_changed = Signal(str)

    # 样式系统信号
    active_style_index_changed = Signal(int)
    styles_changed = Signal()
    global_bg_color_changed = Signal(str)
    global_bg_enabled_changed = Signal(bool)

    # 音符数据信号（供歌词编辑页使用）
    ustx_notes_changed = Signal(list)
    note_styles_changed = Signal(object)

    # 布尔信号
    show_bpm_changed = Signal(bool)
    show_play_time_changed = Signal(bool)
    show_song_name_changed = Signal(bool)
    show_song_author_changed = Signal(bool)
    show_ust_author_changed = Signal(bool)
    show_phoneme_changed = Signal(bool)
    show_midinote_changed = Signal(bool)
    show_waveform_changed = Signal(bool)
    fullscreen_changed = Signal(bool)
    show_copyright_changed = Signal(bool)
    curve_show_changed = Signal(bool)
    theme_mode_changed = Signal(str)
    accent_color_mode_changed = Signal(str)
    custom_accent_color_changed = Signal(str)

    # ===================== .uplr 工程文件字段注册表 =====================
    PROJECT_SCHEMA = [
        # 基础元信息
        ("project_name", "str"),
        ("ustx_path", "str"),
        ("song_name", "str"),
        ("song_author", "str"),
        ("ust_author", "str"),
        # 颜色（与活动样式联动，build_ust_info 中作为 fallback）
        ("bg_color", "str"),
        ("note_color", "str"),
        ("lyric_color", "str"),
        ("pitch_curve_color", "str"),
        ("info_text_color", "str"),
        # 全局背景
        ("global_bg_color", "str"),
        ("global_bg_enabled", "bool"),
        # 路径与位置
        ("lyric_pos", "str"),
        ("lrc_path", "str"),
        ("audio_path", "str"),
        # 静默/结尾/音高占位显示
        ("silent_display", "str"),
        ("silent_custom_text", "str"),
        ("end_display", "str"),
        ("end_custom_text", "str"),
        ("pitch_placeholder", "str"),
        ("pitch_custom_text", "str"),
        # 字体（逐字歌词字体 / 歌词及信息字体）
        ("word_lyric_font_family", "str"),
        ("info_font_family", "str"),
        # 自定义字体文件路径（打开工程时按路径重新加载恢复）
        ("custom_font_paths", "json"),
        # 布尔显示开关
        ("show_bpm", "bool"),
        ("show_play_time", "bool"),
        ("show_song_name", "bool"),
        ("show_song_author", "bool"),
        ("show_ust_author", "bool"),
        ("show_copyright", "bool"),
        ("show_phoneme", "bool"),
        ("show_midinote", "bool"),
        ("show_waveform", "bool"),
        ("fullscreen", "bool"),
        ("show_lyric", "bool"),
        ("show_lyric_autohide", "bool"),
        ("lyric_autohide_threshold", "float"),
        ("curve_show", "bool"),
        # 样式系统（结构化数据，导入顺序敏感）
        ("styles", "json"),
        ("active_style_index", "int"),
        ("note_styles", "json"),
    ]

    # 导入时需延后/特殊处理的字段名集合
    _DEFERRED_FIELDS = {"ustx_path", "styles", "active_style_index", "note_styles"}

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)

        # 程序根目录
        self.program_root = os.path.dirname(os.path.abspath(sys.argv[0]))
        self.settings_path = os.path.join(self.program_root, "Settings.ini")

        # 文本文件路径
        self.terms_file_path = os.path.join(self.program_root, "Terms.txt")
        self.license_file_path = os.path.join(self.program_root, "LICENSE")

        # 默认路径
        default_desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        self.last_open_dir = default_desktop
        self.last_export_dir = default_desktop

        # ===== 工程字段（PROJECT_SCHEMA）默认值 =====
        self._reset_project_to_defaults()

        # ===== 用户级 UI 偏好（不参与 .uplr 工程导入/导出，仅写入 Settings.ini）=====
        self._theme_mode = "auto"
        self._accent_color_mode = "auto"
        self._custom_accent_color = "#8245aa"

        # 初始化配置
        self._config = configparser.ConfigParser()
        self.read_settings()

    # ===================== 字符串属性（getter/setter + signal） =====================

    @property
    def ustx_path(self) -> str:
        return self._ustx_path

    @ustx_path.setter
    def ustx_path(self, v: str):
        if self._ustx_path != v:
            self._ustx_path = v
            # 路径变化时清除缓存，下次使用时会重新解析
            self._cached_ust_info = None
            self._ustx_notes = []
            self.ustx_path_changed.emit(v)

    @property
    def project_name(self) -> str:
        return self._project_name

    @project_name.setter
    def project_name(self, v: str):
        if self._project_name != v:
            self._project_name = v
            self.project_name_changed.emit(v)

    @property
    def song_name(self) -> str:
        return self._song_name

    @song_name.setter
    def song_name(self, v: str):
        if self._song_name != v:
            self._song_name = v
            self.song_name_changed.emit(v)

    @property
    def song_author(self) -> str:
        return self._song_author

    @song_author.setter
    def song_author(self, v: str):
        if self._song_author != v:
            self._song_author = v
            self.song_author_changed.emit(v)

    @property
    def ust_author(self) -> str:
        return self._ust_author

    @ust_author.setter
    def ust_author(self, v: str):
        if self._ust_author != v:
            self._ust_author = v
            self.ust_author_changed.emit(v)

    @property
    def bg_color(self) -> str:
        return self._bg_color

    @bg_color.setter
    def bg_color(self, v: str):
        if self._bg_color != v:
            self._bg_color = v
            self.bg_color_changed.emit(v)

    @property
    def note_color(self) -> str:
        return self._note_color

    @note_color.setter
    def note_color(self, v: str):
        if self._note_color != v:
            self._note_color = v
            self.note_color_changed.emit(v)

    @property
    def lyric_color(self) -> str:
        return self._lyric_color

    @lyric_color.setter
    def lyric_color(self, v: str):
        if self._lyric_color != v:
            self._lyric_color = v
            self.lyric_color_changed.emit(v)

    @property
    def pitch_curve_color(self) -> str:
        return self._pitch_curve_color

    @pitch_curve_color.setter
    def pitch_curve_color(self, v: str):
        if self._pitch_curve_color != v:
            self._pitch_curve_color = v
            self.pitch_curve_color_changed.emit(v)

    @property
    def lyric_pos(self) -> str:
        return self._lyric_pos

    @lyric_pos.setter
    def lyric_pos(self, v: str):
        if self._lyric_pos != v:
            self._lyric_pos = v
            self.lyric_pos_changed.emit(v)

    @property
    def lrc_path(self) -> str:
        return self._lrc_path

    @lrc_path.setter
    def lrc_path(self, v: str):
        if self._lrc_path != v:
            self._lrc_path = v
            self.lrc_path_changed.emit(v)

    @property
    def audio_path(self) -> str:
        return self._audio_path

    @audio_path.setter
    def audio_path(self, v: str):
        if self._audio_path != v:
            self._audio_path = v
            self.audio_path_changed.emit(v)

    @property
    def silent_display(self) -> str:
        return self._silent_display

    @silent_display.setter
    def silent_display(self, v: str):
        if self._silent_display != v:
            self._silent_display = v
            self.silent_display_changed.emit(v)

    @property
    def silent_custom_text(self) -> str:
        return self._silent_custom_text

    @silent_custom_text.setter
    def silent_custom_text(self, v: str):
        if self._silent_custom_text != v:
            self._silent_custom_text = v
            self.silent_custom_text_changed.emit(v)

    @property
    def end_display(self) -> str:
        return self._end_display

    @end_display.setter
    def end_display(self, v: str):
        if self._end_display != v:
            self._end_display = v
            self.end_display_changed.emit(v)

    @property
    def end_custom_text(self) -> str:
        return self._end_custom_text

    @end_custom_text.setter
    def end_custom_text(self, v: str):
        if self._end_custom_text != v:
            self._end_custom_text = v
            self.end_custom_text_changed.emit(v)

    @property
    def pitch_placeholder(self) -> str:
        return self._pitch_placeholder

    @pitch_placeholder.setter
    def pitch_placeholder(self, v: str):
        if self._pitch_placeholder != v:
            self._pitch_placeholder = v
            self.pitch_placeholder_changed.emit(v)

    @property
    def pitch_custom_text(self) -> str:
        return self._pitch_custom_text

    @pitch_custom_text.setter
    def pitch_custom_text(self, v: str):
        if self._pitch_custom_text != v:
            self._pitch_custom_text = v
            self.pitch_custom_text_changed.emit(v)

    # ===================== 字体 =====================

    @property
    def word_lyric_font_family(self) -> str:
        return self._word_lyric_font_family

    @word_lyric_font_family.setter
    def word_lyric_font_family(self, v: str):
        if self._word_lyric_font_family != v:
            self._word_lyric_font_family = v
            self.word_lyric_font_family_changed.emit(v)

    @property
    def info_font_family(self) -> str:
        return self._info_font_family

    @info_font_family.setter
    def info_font_family(self, v: str):
        if self._info_font_family != v:
            self._info_font_family = v
            self.info_font_family_changed.emit(v)

    @property
    def custom_font_paths(self) -> list:
        return self._custom_font_paths

    @custom_font_paths.setter
    def custom_font_paths(self, v: list):
        if self._custom_font_paths != v:
            self._custom_font_paths = v
            self.custom_font_paths_changed.emit(v)

    @property
    def info_text_color(self) -> str:
        return self._info_text_color

    @info_text_color.setter
    def info_text_color(self, v: str):
        if self._info_text_color != v:
            self._info_text_color = v
            self.info_text_color_changed.emit(v)

    # ===================== 歌词内容（内存缓存，精简导入时使用）=====================

    @property
    def lyric_content(self) -> str:
        """精简导入时从工程文件加载的歌词完整内容（内存中，不写磁盘）。"""
        return self._lyric_content

    @lyric_content.setter
    def lyric_content(self, v: str):
        self._lyric_content = v

    # ===================== 样式系统属性 =====================

    @property
    def styles(self) -> list:
        return self._styles

    @property
    def active_style_index(self) -> int:
        return self._active_style_index

    @active_style_index.setter
    def active_style_index(self, v: int):
        if 0 <= v < len(self._styles) and self._active_style_index != v:
            self._active_style_index = v
            self.active_style_index_changed.emit(v)
            p = self._styles[v]
            self._bg_color = p.get("bg_color", "#000000")
            self._note_color = p.get("note_color", "#6c6c6c")
            self._lyric_color = p.get("lyric_color", "#ffffff")
            self._pitch_curve_color = p.get("pitch_curve_color", "#ffffff")

    @property
    def active_style(self) -> dict:
        return self._styles[self._active_style_index] if self._styles else {}

    def set_style_color(self, style_index: int, key: str, value: str):
        if 0 <= style_index < len(self._styles):
            if self._styles[style_index].get(key) != value:
                self._styles[style_index][key] = value
                if style_index == self._active_style_index:
                    if key == "bg_color":
                        self._bg_color = value
                    elif key == "note_color":
                        self._note_color = value
                    elif key == "lyric_color":
                        self._lyric_color = value
                    elif key == "pitch_curve_color":
                        self._pitch_curve_color = value
                self.styles_changed.emit()

    @property
    def style_count(self) -> int:
        return len(self._styles)

    def add_style(self):
        new_idx = len(self._styles)
        self._styles.append(dict(self._styles[0]))
        logger.info(f"样式系统: 新建样式{new_idx + 1}（共{len(self._styles)}个）")
        self.styles_changed.emit()
        return new_idx

    def remove_style(self, index: int) -> bool:
        if len(self._styles) <= 3 or index < 0 or index >= len(self._styles):
            return False
        del self._styles[index]
        logger.info(f"样式系统: 删除样式{index + 1}（剩余{len(self._styles)}个）")
        if self._active_style_index >= len(self._styles):
            self._active_style_index = len(self._styles) - 1
        elif self._active_style_index > index:
            self._active_style_index -= 1
        new_styles = {}
        for row, si in self._note_styles.items():
            if si == index:
                new_styles[row] = 0
            elif si > index:
                new_styles[row] = si - 1
            else:
                new_styles[row] = si
        self.note_styles = new_styles
        p = self._styles[self._active_style_index]
        self._bg_color = p.get("bg_color", "#000000")
        self._note_color = p.get("note_color", "#6c6c6c")
        self._lyric_color = p.get("lyric_color", "#ffffff")
        self._pitch_curve_color = p.get("pitch_curve_color", "#ffffff")
        self.styles_changed.emit()
        self.active_style_index_changed.emit(self._active_style_index)
        return True

    def get_style_name(self, index: int) -> str:
        return f"样式{index + 1}"

    # ===================== 音符数据（歌词编辑用） =====================

    @property
    def ustx_notes(self) -> list:
        return self._ustx_notes

    @ustx_notes.setter
    def ustx_notes(self, v: list):
        self._ustx_notes = v
        self._note_styles = {}
        self.ustx_notes_changed.emit(v)

    @property
    def cached_ust_info(self) -> Optional[dict]:
        return self._cached_ust_info

    @cached_ust_info.setter
    def cached_ust_info(self, v: Optional[dict]):
        self._cached_ust_info = v

    def clear_cached_data(self):
        """释放内存中的缓存数据（音符列表、解析结果、歌词内容）。

        切换工程或不再需要内存数据时调用，避免持续占用内存。
        下次使用时（如 _on_play）若缓存为空会从文件重新解析。
        """
        self._ustx_notes = []
        self._cached_ust_info = None
        self._lyric_content = ""
        self._note_styles = {}
        logger.debug("已清除内存缓存数据")

    def maybe_fill_project_name_from_ustx(self) -> bool:
        if not self._project_name.strip() and self._ustx_path:
            base = os.path.splitext(os.path.basename(self._ustx_path))[0]
            if base:
                self.project_name = base
                return True
        return False

    @property
    def note_styles(self) -> dict:
        return self._note_styles

    @note_styles.setter
    def note_styles(self, v: dict):
        self._note_styles = v
        self.note_styles_changed.emit(v)

    @property
    def global_bg_color(self) -> str:
        return self._global_bg_color

    @global_bg_color.setter
    def global_bg_color(self, v: str):
        if self._global_bg_color != v:
            self._global_bg_color = v
            self.global_bg_color_changed.emit(v)

    @property
    def global_bg_enabled(self) -> bool:
        return self._global_bg_enabled

    @global_bg_enabled.setter
    def global_bg_enabled(self, v: bool):
        if self._global_bg_enabled != v:
            self._global_bg_enabled = v
            self.global_bg_enabled_changed.emit(v)

    def get_effective_bg_color(self) -> str:
        if self._global_bg_enabled:
            return self._global_bg_color
        return self.active_style.get("bg_color", "#000000")

    # ===================== 布尔属性（getter/setter + signal） =====================

    @property
    def show_bpm(self) -> bool:
        return self._show_bpm

    @show_bpm.setter
    def show_bpm(self, v: bool):
        if self._show_bpm != v:
            self._show_bpm = v
            self.show_bpm_changed.emit(v)

    @property
    def show_play_time(self) -> bool:
        return self._show_play_time

    @show_play_time.setter
    def show_play_time(self, v: bool):
        if self._show_play_time != v:
            self._show_play_time = v
            self.show_play_time_changed.emit(v)

    @property
    def show_song_name(self) -> bool:
        return self._show_song_name

    @show_song_name.setter
    def show_song_name(self, v: bool):
        if self._show_song_name != v:
            self._show_song_name = v
            self.show_song_name_changed.emit(v)

    @property
    def show_song_author(self) -> bool:
        return self._show_song_author

    @show_song_author.setter
    def show_song_author(self, v: bool):
        if self._show_song_author != v:
            self._show_song_author = v
            self.show_song_author_changed.emit(v)

    @property
    def show_ust_author(self) -> bool:
        return self._show_ust_author

    @show_ust_author.setter
    def show_ust_author(self, v: bool):
        if self._show_ust_author != v:
            self._show_ust_author = v
            self.show_ust_author_changed.emit(v)

    @property
    def show_copyright(self) -> bool:
        return self._show_copyright

    @show_copyright.setter
    def show_copyright(self, v: bool):
        if self._show_copyright != v:
            self._show_copyright = v
            self.show_copyright_changed.emit(v)

    @property
    def show_phoneme(self) -> bool:
        return self._show_phoneme

    @show_phoneme.setter
    def show_phoneme(self, v: bool):
        if self._show_phoneme != v:
            self._show_phoneme = v
            self.show_phoneme_changed.emit(v)

    @property
    def show_midinote(self) -> bool:
        return self._show_midinote

    @show_midinote.setter
    def show_midinote(self, v: bool):
        if self._show_midinote != v:
            self._show_midinote = v
            self.show_midinote_changed.emit(v)

    @property
    def show_waveform(self) -> bool:
        return self._show_waveform

    @show_waveform.setter
    def show_waveform(self, v: bool):
        if self._show_waveform != v:
            self._show_waveform = v
            self.show_waveform_changed.emit(v)

    @property
    def fullscreen(self) -> bool:
        return self._fullscreen

    @fullscreen.setter
    def fullscreen(self, v: bool):
        if self._fullscreen != v:
            self._fullscreen = v
            self.fullscreen_changed.emit(v)

    @property
    def show_lyric(self) -> bool:
        return self._show_lyric

    @show_lyric.setter
    def show_lyric(self, v: bool):
        if self._show_lyric != v:
            self._show_lyric = v

    @property
    def show_lyric_autohide(self) -> bool:
        return self._show_lyric_autohide

    @show_lyric_autohide.setter
    def show_lyric_autohide(self, v: bool):
        if self._show_lyric_autohide != v:
            self._show_lyric_autohide = v

    @property
    def lyric_autohide_threshold(self) -> float:
        return self._lyric_autohide_threshold

    @lyric_autohide_threshold.setter
    def lyric_autohide_threshold(self, v: float):
        if self._lyric_autohide_threshold != v:
            self._lyric_autohide_threshold = v

    @property
    def curve_show(self) -> bool:
        return self._curve_show

    @curve_show.setter
    def curve_show(self, v: bool):
        if self._curve_show != v:
            self._curve_show = v
            self.curve_show_changed.emit(v)

    # ===================== 主题模式属性 =====================

    @property
    def theme_mode(self) -> str:
        return self._theme_mode

    @theme_mode.setter
    def theme_mode(self, v: str):
        if v not in ("auto", "light", "dark"):
            v = "auto"
        if self._theme_mode != v:
            self._theme_mode = v
            self.theme_mode_changed.emit(v)

    # ===================== 强调色属性 =====================

    @property
    def accent_color_mode(self) -> str:
        return self._accent_color_mode

    @accent_color_mode.setter
    def accent_color_mode(self, v: str):
        if v not in ("auto", "custom"):
            v = "auto"
        if self._accent_color_mode != v:
            self._accent_color_mode = v
            self.accent_color_mode_changed.emit(v)

    @property
    def custom_accent_color(self) -> str:
        return self._custom_accent_color

    @custom_accent_color.setter
    def custom_accent_color(self, v: str):
        if self._custom_accent_color != v:
            self._custom_accent_color = v
            self.custom_accent_color_changed.emit(v)

    # ===================== Settings.ini 读写 =====================

    def read_settings(self):
        default_desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        try:
            if os.path.exists(self.settings_path):
                self._config.read(self.settings_path, encoding="utf-8")
                if "PathSettings" in self._config:
                    self.last_open_dir = self._config["PathSettings"].get(
                        "last_open_dir", default_desktop
                    )
                    self.last_export_dir = self._config["PathSettings"].get(
                        "last_export_dir", default_desktop
                    )
                    if not os.path.isdir(self.last_open_dir):
                        self.last_open_dir = default_desktop
                    if not os.path.isdir(self.last_export_dir):
                        self.last_export_dir = default_desktop
                if "ThemeSettings" in self._config:
                    mode = self._config["ThemeSettings"].get("theme_mode", "auto")
                    self._theme_mode = mode if mode in ("auto", "light", "dark") else "auto"
                    amode = self._config["ThemeSettings"].get("accent_color_mode", "auto")
                    self._accent_color_mode = amode if amode in ("auto", "custom") else "auto"
                    self._custom_accent_color = self._config["ThemeSettings"].get(
                        "custom_accent_color", "#8245aa"
                    )
            else:
                self.last_open_dir = default_desktop
                self.last_export_dir = default_desktop
        except Exception:
            self.last_open_dir = default_desktop
            self.last_export_dir = default_desktop
            logger.exception("读取配置文件失败")

    def write_settings(self):
        try:
            self._config = configparser.ConfigParser()
            self._config["PathSettings"] = {
                "last_open_dir": self.last_open_dir,
                "last_export_dir": self.last_export_dir,
            }
            self._config["ThemeSettings"] = {
                "theme_mode": self._theme_mode,
                "accent_color_mode": self._accent_color_mode,
                "custom_accent_color": self._custom_accent_color,
            }
            tmp = self.settings_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                self._config.write(f)
            os.replace(tmp, self.settings_path)
        except Exception:
            logger.exception("写入配置文件失败")

    # ===================== 工程字段默认值与重置 =====================

    def _get_project_defaults(self) -> dict:
        return {
            # 基础元信息
            "project_name": "",
            "ustx_path": "",
            "song_name": "",
            "song_author": "",
            "ust_author": "",
            # 颜色
            "bg_color": "#000000",
            "note_color": "#6c6c6c",
            "lyric_color": "#ffffff",
            "pitch_curve_color": "#ffffff",
            "info_text_color": "#ffffff",
            # 全局背景
            "global_bg_color": "#00ff00",
            "global_bg_enabled": False,
            # 路径与位置
            "lyric_pos": "上",
            "lrc_path": "",
            "audio_path": "",
            # 静默/结尾/音高占位显示
            "silent_display": "♪",
            "silent_custom_text": "",
            "end_display": "END",
            "end_custom_text": "",
            "pitch_placeholder": "无",
            "pitch_custom_text": "",
            # 字体
            "word_lyric_font_family": "等线",
            "info_font_family": "微软雅黑",
            # 自定义字体文件路径
            "custom_font_paths": [],
            # 布尔显示开关
            "show_bpm": True,
            "show_play_time": True,
            "show_song_name": True,
            "show_song_author": True,
            "show_ust_author": True,
            "show_copyright": True,
            "show_phoneme": False,
            "show_midinote": False,
            "show_waveform": False,
            "fullscreen": True,
            "show_lyric": True,
            "show_lyric_autohide": True,
            "lyric_autohide_threshold": 3.0,
            "curve_show": False,
            # 样式系统
            "styles": [
                {"bg_color": "#000000", "note_color": "#6c6c6c", "lyric_color": "#ffffff", "pitch_curve_color": "#ffffff"},
                {"bg_color": "#000000", "note_color": "#ff8a80", "lyric_color": "#ff0c0c", "pitch_curve_color": "#ff0c0c"},
                {"bg_color": "#000000", "note_color": "#a1887f", "lyric_color": "#795548", "pitch_curve_color": "#795548"},
            ],
            "active_style_index": 0,
            "note_styles": {},
            # 歌词内容（内存缓存，精简导入时使用）
            "lyric_content": "",
        }

    def _reset_project_to_defaults(self):
        defaults = self._get_project_defaults()
        for name, _ftype in self.PROJECT_SCHEMA:
            setattr(self, f"_{name}", copy.deepcopy(defaults[name]))
        # 额外非 schema 字段
        self._lyric_content = ""
        self._ustx_notes = []
        self._cached_ust_info = None

    # ===================== .uplr 工程文件导出 =====================

    def export_uplr(self, output_file: str):
        """普通导出：全量记录所有注册字段 + 内嵌完整 USTX 文件内容。

        兼容原项目 TS player，导入时需重新解析 USTX。
        """
        ustx_content = ""
        if self._ustx_path and os.path.isfile(self._ustx_path):
            try:
                with open(self._ustx_path, "r", encoding="utf-8") as f:
                    ustx_content = f.read()
            except Exception:
                logger.exception(f"读取 ustx 文件失败: {self._ustx_path}")

        settings_data = {}
        for name, _ftype in self.PROJECT_SCHEMA:
            settings_data[name] = getattr(self, name)

        if not settings_data["project_name"].strip():
            settings_data["project_name"] = "未命名"

        payload = {
            "format": "ustxPlayer-preview.uplr",
            "version": 2,
            "ustx_content": ustx_content,
            "settings": settings_data,
        }
        tmp = output_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, output_file)

    def export_uplr_compact(self, output_file: str):
        """精简导出（BETA）：存储预解析数据，导入时无需重新解析。

        与普通导出的区别：
        - 不存储原始 ustx_content
        - 存储 parsed_notes（预解析音符列表，直接加载到内存）
        - 存储 parsed_ust_info（tempo/version/tracks 等元信息）
        - 歌词存完整内容（lyric_content）
        - 音频仅存路径（与普通导出一致）
        - 使用专属 format 标识符，其他软件无法解析
        """
        # 读取歌词文件内容
        lyric_content = ""
        if self._lrc_path and os.path.isfile(self._lrc_path):
            try:
                with open(self._lrc_path, "r", encoding="utf-8") as f:
                    lyric_content = f.read()
            except Exception:
                logger.exception(f"读取歌词文件失败: {self._lrc_path}")

        # 获取预解析的 ust_info（优先缓存，否则实时解析）
        parsed_notes = []
        parsed_ust_info = {}
        cached = self._cached_ust_info
        if cached and cached.get("info"):
            info = cached["info"]
            parsed_notes = info.get("notes", [])
            parsed_ust_info = {
                "version": info.get("version", "unknown"),
                "tempo": info.get("tempo", 120.0),
                "tracks": info.get("tracks", 1),
                "track_name": info.get("track_name", "全部音轨"),
            }
        elif self._ustx_path and os.path.isfile(self._ustx_path):
            try:
                from core.ustxreader import get_ustx_info
                info = get_ustx_info(self._ustx_path)
                parsed_notes = info.get("notes", [])
                parsed_ust_info = {
                    "version": info.get("version", "unknown"),
                    "tempo": info.get("tempo", 120.0),
                    "tracks": info.get("tracks", 1),
                    "track_name": info.get("track_name", "全部音轨"),
                }
            except Exception:
                logger.exception("精简导出前解析 USTX 失败")

        # 序列化所有注册字段
        settings_data = {}
        for name, _ftype in self.PROJECT_SCHEMA:
            settings_data[name] = getattr(self, name)

        if not settings_data["project_name"].strip():
            settings_data["project_name"] = "未命名"

        payload = {
            "format": "ustxPlayer-preview-compact.uplr",
            "version": 2,
            "settings": settings_data,
            # 精简导出专属数据
            "parsed_notes": parsed_notes,
            "parsed_ust_info": parsed_ust_info,
            "lyric_content": lyric_content,
        }
        tmp = output_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, output_file)

    # ===================== .uplr 工程文件导入 =====================

    def import_uplr(self, input_file: str, parse_ustx: bool = True):
        """从 .uplr 工程文件导入全部配置。

        自动识别格式：
        - 普通格式（ustxPlayer-preview.uplr）：从 ustx_content 直接解析到内存
        - 精简格式（ustxPlayer-preview-compact.uplr）：加载预解析数据到内存

        两种格式均不再创建缓存文件，解析结果直接写入内存变量。
        """
        with open(input_file, "r", encoding="utf-8") as f:
            payload = json.load(f)

        fmt = payload.get("format", "")

        # 精简格式 → 走专属导入逻辑
        if fmt == "ustxPlayer-preview-compact.uplr":
            self._import_uplr_compact(payload)
            return

        # 普通格式校验
        if fmt != "ustxPlayer-preview.uplr" or payload.get("version") != 2:
            raise ValueError("不是有效的 ustxPlayer-preview 工程文件（format/version 不匹配）")

        data = payload.get("settings", {})
        ustx_content = payload.get("ustx_content", "") or ""

        # ---- 阶段0: 重置所有工程字段为默认值 ----
        self._reset_project_to_defaults()

        # ---- 阶段1: 普通字段（跳过顺序敏感字段）----
        for name, ftype in self.PROJECT_SCHEMA:
            if name in self._DEFERRED_FIELDS:
                continue
            if name not in data:
                continue
            raw = data[name]
            try:
                if ftype == "bool":
                    setattr(self, name, bool(raw))
                elif ftype == "int":
                    setattr(self, name, int(raw))
                elif ftype == "float":
                    setattr(self, name, float(raw))
                else:
                    setattr(self, name, raw)
            except (ValueError, TypeError):
                logger.warning(f"工程文件字段 {name} 值非法，已跳过: {raw!r}")

        # ---- 阶段2: styles（须先于 active_style_index）----
        styles = data.get("styles")
        if isinstance(styles, list) and styles:
            _style_defaults = {
                "bg_color": "#000000",
                "note_color": "#6c6c6c",
                "lyric_color": "#ffffff",
                "pitch_curve_color": "#ffffff",
            }
            self._styles = [
                {**_style_defaults, **s} if isinstance(s, dict) else dict(_style_defaults)
                for s in styles
            ]
            self.styles_changed.emit()

        # ---- 阶段3: active_style_index ----
        if "active_style_index" in data:
            try:
                idx = int(data["active_style_index"])
            except (ValueError, TypeError):
                logger.warning(f"active_style_index 值非法，已忽略: {data['active_style_index']!r}")
                idx = 0
            idx = max(0, min(idx, len(self._styles) - 1)) if self._styles else 0
            self.active_style_index = idx

        # ---- 阶段4: USTX 直接解析到内存（不再写缓存文件）----
        if ustx_content:
            from core.ustxreader import get_ustx_info_from_content
            try:
                ust_info = get_ustx_info_from_content(ustx_content)
                _notes = ust_info.get("notes", [])
                self.ustx_notes = _notes if isinstance(_notes, list) else []
                self.cached_ust_info = {
                    "path": data.get("ustx_path", ""),
                    "info": ust_info,
                }
            except Exception:
                logger.exception("内存解析 USTX 失败，音符数据置空")
                self.ustx_notes = []
        else:
            self.ustx_path = data.get("ustx_path", "")
            self.ustx_notes = []

        # 保留 ustx_path 为原始路径（用户参考用，可能不存在）
        self._ustx_path = data.get("ustx_path", "")

        # ---- 阶段5: note_styles ----
        ns = data.get("note_styles")
        if isinstance(ns, dict):
            self.note_styles = {int(k): int(v) for k, v in ns.items()}
        else:
            self.note_styles = {}

        # ---- 阶段6: 校验文件路径（ustx 不校验，已从内存解析）----
        missing: list[tuple[str, str]] = []
        if self._lrc_path and not os.path.exists(self._lrc_path):
            missing.append(("歌词文件", self._lrc_path))
        if self._audio_path and not os.path.exists(self._audio_path):
            missing.append(("音频文件", self._audio_path))
        for font_path in self._custom_font_paths:
            if font_path and not os.path.exists(font_path):
                missing.append(("字体文件", font_path))
        if missing:
            raise ProjectFileMissingError(missing)

    def _import_uplr_compact(self, payload: dict):
        """精简工程导入：预解析数据直接加载到内存，零缓存文件写入。

        与普通导入的区别：
        - 不读取 ustx_content
        - 直接从 parsed_notes 恢复音符数据到内存
        - 从 parsed_ust_info 恢复元信息
        - 歌词内容存到 lyric_content 内存变量
        - ustx_path 保留原路径（供参考，可能不存在）
        """
        if payload.get("version") != 2:
            raise ValueError("精简工程文件版本不匹配")

        data = payload.get("settings", {})

        # 重置所有工程字段
        self._reset_project_to_defaults()

        # 普通字段恢复（跳过延迟字段）
        for name, ftype in self.PROJECT_SCHEMA:
            if name in self._DEFERRED_FIELDS:
                continue
            if name not in data:
                continue
            raw = data[name]
            try:
                if ftype == "bool":
                    setattr(self, name, bool(raw))
                elif ftype == "int":
                    setattr(self, name, int(raw))
                elif ftype == "float":
                    setattr(self, name, float(raw))
                else:
                    setattr(self, name, raw)
            except (ValueError, TypeError):
                logger.warning(f"工程文件字段 {name} 值非法，已跳过: {raw!r}")

        # styles
        styles = data.get("styles")
        if isinstance(styles, list) and styles:
            _style_defaults = {
                "bg_color": "#000000", "note_color": "#6c6c6c",
                "lyric_color": "#ffffff", "pitch_curve_color": "#ffffff",
            }
            self._styles = [
                {**_style_defaults, **s} if isinstance(s, dict) else dict(_style_defaults)
                for s in styles
            ]
            self.styles_changed.emit()

        # active_style_index
        if "active_style_index" in data:
            try:
                idx = int(data["active_style_index"])
            except (ValueError, TypeError):
                idx = 0
            idx = max(0, min(idx, len(self._styles) - 1)) if self._styles else 0
            self.active_style_index = idx

        # ---- 精简专属：预解析数据直接加载到内存 ----
        parsed_notes = payload.get("parsed_notes", [])
        parsed_ust_info = payload.get("parsed_ust_info", {})
        if parsed_notes:
            cached_info = {
                "version": parsed_ust_info.get("version", "unknown"),
                "tempo": parsed_ust_info.get("tempo", 120.0),
                "tracks": parsed_ust_info.get("tracks", 1),
                "track_name": parsed_ust_info.get("track_name", "全部音轨"),
                "notes": parsed_notes,
            }
            self.ustx_notes = parsed_notes  # 直接加载到内存，零解析
            self.cached_ust_info = {
                "path": data.get("ustx_path", ""),
                "info": cached_info,
            }

        # 歌词内容存到内存
        lyric_content = payload.get("lyric_content", "")
        if lyric_content:
            self._lyric_content = lyric_content
            self._lrc_path = data.get("lrc_path", "")

        # ustx_path 保留原始路径
        self._ustx_path = data.get("ustx_path", "")

        # note_styles
        ns = data.get("note_styles")
        if isinstance(ns, dict):
            self.note_styles = {int(k): int(v) for k, v in ns.items()}
        else:
            self.note_styles = {}

        # 文件路径校验（仅校验音频和字体，USTX 和歌词已内嵌）
        missing: list[tuple[str, str]] = []
        if self._audio_path and not os.path.exists(self._audio_path):
            missing.append(("音频文件", self._audio_path))
        for font_path in self._custom_font_paths:
            if font_path and not os.path.exists(font_path):
                missing.append(("字体文件", font_path))
        if missing:
            raise ProjectFileMissingError(missing)

    # ===================== 构建播放器需要的 ust_info 字典 =====================

    def build_ust_info(self, core_ust_info: dict) -> dict:
        ap = self.active_style
        return {
            "version": core_ust_info.get("version", "未知版本"),
            "tempo": core_ust_info.get("tempo", 120.0),
            "tracks": core_ust_info.get("tracks", 1),
            "notes": core_ust_info.get("notes", []),
            "show_config": {
                "bpm": self.show_bpm,
                "play_time": self.show_play_time,
                "song_name": self.show_song_name,
                "song_author": self.show_song_author,
                "ust_author": self.show_ust_author,
                "copyright": self.show_copyright,
                "lyric": self.show_lyric,
                "lyric_autohide": self.show_lyric_autohide,
                "lyric_autohide_threshold": self.lyric_autohide_threshold,
                "curve_show": self.curve_show,
            },
            "project_info": {
                "project_name": self.project_name,
                "song_name": self.song_name,
                "song_author": self.song_author,
                "ust_author": self.ust_author,
            },
            "player_style": {
                "bg_color": ap.get("bg_color", self._bg_color),
                "global_bg_color": self._global_bg_color,
                "global_bg_enabled": self._global_bg_enabled,
                "note_color": ap.get("note_color", self._note_color),
                "lyric_color": ap.get("lyric_color", self._lyric_color),
                "info_text_color": self._info_text_color,
                "lyric_pos": self.lyric_pos,
                "show_phoneme": self.show_phoneme,
                "show_midinote": self.show_midinote,
                "show_waveform": self.show_waveform,
                "fullscreen": self.fullscreen,
                "lrc_path": self.lrc_path,
                "audio_path": self.audio_path,
                "silent_display": self.silent_display,
                "silent_custom_text": self.silent_custom_text,
                "end_display": self.end_display,
                "end_custom_text": self.end_custom_text,
                "pitch_placeholder": self.pitch_placeholder,
                "pitch_custom_text": self.pitch_custom_text,
                "pitch_curve_color": ap.get("pitch_curve_color", self._pitch_curve_color),
                "word_lyric_font_family": self.word_lyric_font_family,
                "info_font_family": self.info_font_family,
                "styles": list(self._styles),
                "note_styles": dict(self._note_styles),
            },
        }