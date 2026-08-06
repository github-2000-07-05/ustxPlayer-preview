# license_dialog.py — 协议查看对话框
"""在 GUI 内展示使用协议 / 开源许可全文的只读对话框。

用法：
    dlg = LicenseDialog(parent, title="...", path="...", encodings=["utf-8", "gbk"])
    dlg.exec()

背景随应用主题（亮/暗）自动切换，并监听 qconfig.themeChanged 实时刷新，
保证文本与背景的对比度；文件编码按 encodings 依次尝试，全部失败后回退默认提示。
"""

import os
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QWidget

from qfluentwidgets import (
    ScrollArea, BodyLabel, StrongBodyLabel, PrimaryPushButton,
    Theme, qconfig,
)

from core.log import logger


class LicenseDialog(QDialog):
    """只读协议查看对话框。

    参数：
        parent: 父窗口。
        title: 对话框标题（窗口标题 + 顶部标题）。
        path: 要展示的协议文件路径。
        encodings: 尝试读取文件的编码列表（按顺序）。
        default_text: 文件缺失或读取失败时显示的提示文本。
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        title: str = "使用协议",
        path: str = "",
        encodings: Optional[list[str]] = None,
        default_text: str = "（协议文件不存在）",
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(640, 560)
        self.setMinimumSize(480, 360)

        # 背景随主题（亮/暗）刷新，保证文本对比度
        qconfig.themeChanged.connect(self._on_app_theme_changed)
        qconfig.themeChangedFinished.connect(self._on_app_theme_changed)
        self._apply_theme_background()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)

        layout.addWidget(StrongBodyLabel(title))

        # 滚动只读文本区（可选中复制）
        content = self._read_text(path, encodings or ["utf-8", "gbk"], default_text)
        scroll = ScrollArea(self)
        scroll.setWidgetResizable(True)
        label = BodyLabel(content)
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        label.setMargin(12)
        scroll.setWidget(label)
        scroll.setMinimumHeight(360)
        layout.addWidget(scroll, 1)

        # 底部关闭按钮
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = PrimaryPushButton("关闭")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    # ===================== 主题适配 =====================

    def _apply_theme_background(self):
        """根据当前实际主题设置对话框 / 滚动区背景。

        qfluentwidgets 的应用级 QSS 不覆盖原生 QDialog 与 QScrollArea，
        暗色主题下二者默认为系统浅色背景，与暗色文字样式冲突，
        故在此统一设置背景并监听主题变化实时刷新。
        """
        is_dark = qconfig.theme == Theme.DARK
        bg = "#1a1a1a" if is_dark else "#f5f5f5"
        self.setStyleSheet(
            f"QDialog {{ background: {bg}; }}"
            f"QScrollArea {{ background: {bg}; }}"
            f"QScrollArea > QWidget > QWidget {{ background: {bg}; }}"
        )

    def _on_app_theme_changed(self, *_):
        """应用主题切换时刷新背景（themeChanged 可能携带未解析的 Theme.AUTO，直接重读）。"""
        self._apply_theme_background()

    # ===================== 文本读取 =====================

    @staticmethod
    def _read_text(path: str, encodings: list[str], default_text: str) -> str:
        """按 encodings 依次尝试读取文件；全部失败返回 default_text（静默回退，不打逐条告警）。"""
        if not os.path.exists(path):
            logger.warning(f"协议文件不存在: {path}")
            return default_text
        for enc in encodings:
            try:
                with open(path, "r", encoding=enc) as f:
                    return f.read()
            except (UnicodeDecodeError, OSError):
                continue
        logger.error(f"协议文件读取失败（已尝试编码：{'、'.join(encodings)}）: {path}")
        return default_text
