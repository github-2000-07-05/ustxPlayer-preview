# export_mode_dialog.py — 导出模式选择对话框（BETA 功能）
"""导出工程文件时弹出，让用户选择「普通导出」或「精简导出（BETA）」。

精简导出可缩小文件体积、加快导入速度，但属于 BETA 测试功能，可能不稳定。
"""

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QWidget

from qfluentwidgets import (
    BodyLabel, StrongBodyLabel, PrimaryPushButton, PushButton,
    Theme, qconfig, InfoBarIcon,
)


class ExportModeDialog(QDialog):
    """普通导出 / 精简导出（BETA）选择对话框。

    exec() 后通过 chosen_mode 读取结果：
        "normal"   — 普通导出（兼容 TS player，导入需重新解析）
        "compact"  — 精简导出（仅本项目可用，预解析数据，导入更快）
        None       — 取消
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._mode: Optional[str] = None
        self.setWindowTitle("选择导出模式 — BETA 功能")
        self.setModal(True)
        self.resize(500, 310)
        self.setMinimumWidth(420)

        is_dark = qconfig.theme == Theme.DARK
        bg = "#1a1a1a" if is_dark else "#f5f5f5"
        text_color = "#e0e0e0" if is_dark else "#333333"
        warn_bg = "#3d2e1a" if is_dark else "#fff3cd"
        warn_border = "#8a6d14" if is_dark else "#ffc107"
        self.setStyleSheet(f"QDialog {{ background: {bg}; }}")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(10)

        layout.addWidget(StrongBodyLabel("选择导出模式"))

        # 普通导出说明
        normal_desc = BodyLabel(
            "普通导出：完整保留原始 USTX 文件内容，兼容原项目 TS player。\n"
            "导入时需要重新解析文件，速度较慢。"
        )
        normal_desc.setWordWrap(True)
        normal_desc.setStyleSheet(f"color: {text_color};")
        layout.addWidget(normal_desc)

        layout.addSpacing(6)

        # 精简导出说明
        compact_title = StrongBodyLabel("精简导出（BETA）")
        compact_title.setStyleSheet("color: #e6a817;")
        layout.addWidget(compact_title)

        compact_desc = BodyLabel(
            "将预解析后的音符数据直接存入工程文件，导入时跳过解析步骤，\n"
            "大幅提升加载速度并缩小文件体积。\n"
            "仅限本项目使用，其他软件无法解析此格式。"
        )
        compact_desc.setWordWrap(True)
        compact_desc.setStyleSheet(f"color: {text_color};")
        layout.addWidget(compact_desc)

        # BETA 警告
        warn_label = BodyLabel(
            "⚠ BETA 功能警告：此为测试功能，可能存在不稳定情况。\n"
            "如遇问题请及时提交 Issues 反馈。"
        )
        warn_label.setWordWrap(True)
        warn_label.setStyleSheet(
            f"background: {warn_bg}; border: 1px solid {warn_border}; "
            f"border-radius: 6px; padding: 8px 12px; color: {text_color};"
        )
        layout.addWidget(warn_label)

        layout.addStretch()

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        normal_btn = PrimaryPushButton("普通导出")
        normal_btn.clicked.connect(lambda: self._choose("normal"))
        btn_row.addWidget(normal_btn)

        compact_btn = PushButton("精简导出 (BETA)")
        compact_btn.setStyleSheet(
            "PushButton { border: 1px solid #e6a817; color: #e6a817; }"
            "PushButton:hover { background: rgba(230, 168, 23, 0.1); }"
        )
        compact_btn.clicked.connect(lambda: self._choose("compact"))
        btn_row.addWidget(compact_btn)

        cancel_btn = PushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        layout.addLayout(btn_row)

    def _choose(self, mode: str):
        self._mode = mode
        self.accept()

    @property
    def chosen_mode(self) -> Optional[str]:
        """返回用户选择的模式："normal" / "compact" / None（取消）。"""
        return self._mode