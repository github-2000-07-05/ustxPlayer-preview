# player_action_dialog.py — 播放前动作选择对话框
"""点击「播放」后弹出的确认框：启动播放器 / 取消。

使用 qfluentwidgets 组件模拟轻量对话框外观，背景随应用主题适配。
"""

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QWidget

from qfluentwidgets import (
    BodyLabel, StrongBodyLabel, PrimaryPushButton, PushButton,
    Theme, qconfig,
)


class PlayerActionDialog(QDialog):
    """播放 / 导出视频 / 取消 三选一对话框。

    exec() 后通过 chosen_action 读取结果：
        "play"    — 启动播放器
        "export"  — 导出为视频
        None      — 取消
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        project_name: str = "",
        audio_path: str = "",
    ):
        super().__init__(parent)
        self._action: Optional[str] = None
        self.setWindowTitle("ustxPlayer")
        self.setModal(True)
        self.resize(420, 240)
        self.setMinimumWidth(360)

        # 背景随主题（亮/暗）适配
        is_dark = qconfig.theme == Theme.DARK
        bg = "#1a1a1a" if is_dark else "#f5f5f5"
        self.setStyleSheet(f"QDialog {{ background: {bg}; }}")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)

        layout.addWidget(StrongBodyLabel("选择操作"))

        info_lines = [f"工程：{project_name or '（未命名）'}"]
        if audio_path:
            info_lines.append("音频：已关联")
        else:
            info_lines.append("音频：未关联")
        info = BodyLabel("\n".join(info_lines))
        info.setWordWrap(True)
        info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(info)

        layout.addStretch()

        # 主操作行：启动播放器 / 导出为视频
        main_row = QHBoxLayout()
        main_row.setSpacing(8)
        play_btn = PrimaryPushButton("启动播放器")
        play_btn.clicked.connect(lambda: self._choose("play"))
        main_row.addWidget(play_btn)

        export_btn = PrimaryPushButton("导出为视频")
        export_btn.clicked.connect(lambda: self._choose("export"))
        main_row.addWidget(export_btn)
        layout.addLayout(main_row)

        cancel_btn = PushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        layout.addWidget(cancel_btn)

    def _choose(self, action: str):
        self._action = action
        self.accept()

    @property
    def chosen_action(self) -> Optional[str]:
        """返回用户选择的动作："play" / None（取消）。"""
        return self._action
