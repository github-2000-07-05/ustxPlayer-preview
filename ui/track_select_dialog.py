# track_select_dialog.py — 原生风格音轨选择对话框
"""USTX 文件存在多条可解析音轨时，弹出选择窗口供用户明确指定加载的音轨。

使用 Qt 标准组件（QDialog + QListWidget + QDialogButtonBox）模拟原生对话框
外观，按钮文案由 Qt 中文翻译（qtbase_zh_CN）自动本地化。
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QLabel, QListWidget,
    QListWidgetItem, QVBoxLayout,
)


class TrackSelectDialog(QDialog):
    """原生风格音轨选择窗口。

    列出 USTX 文件中的所有可解析音轨（名称 + 音符数），用户选中后返回
    对应音轨下标；取消时返回 None。
    """

    def __init__(self, tracks: list[dict], parent=None):
        super().__init__(parent)
        self.setWindowTitle("选择音轨")
        self.setModal(True)
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        layout.addWidget(QLabel("检测到多条可解析音轨，请选择需要加载的音轨："))

        self._list = QListWidget()
        for t in tracks:
            name = t.get("name") or f"音轨 {t.get('index', 0) + 1}"
            count = t.get("note_count", 0)
            item = QListWidgetItem(f"{name}　（{count} 个音符）")
            item.setData(Qt.ItemDataRole.UserRole, t.get("index"))
            self._list.addItem(item)
        if self._list.count() > 0:
            self._list.setCurrentRow(0)
        layout.addWidget(self._list)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_index(self):
        """返回用户选中的音轨下标；未选择时返回 None。"""
        item = self._list.currentItem()
        if item is None:
            return None
        return item.data(Qt.ItemDataRole.UserRole)
