# card_mixin.py — 共享卡片组件 Mixin
"""提供统一的卡片容器创建和主题刷新，消除 5 个页面间的重复代码。"""

from PySide6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout
from qfluentwidgets import StrongBodyLabel, Theme, qconfig, themeColor


class CardPageMixin:
    """卡片页面混入类 — 提供 _create_section_card 和 _apply_card_theme。

    混入到 QWidget 子类中，通过 self.findChildren 查找卡片并更新样式。
    子类 _setup_ui 中调用 self._create_section_card(title) 创建卡片，
    并在构造函数末尾调用 self._apply_card_theme() 应用初始主题。
    """

    _card_border_radius = 10  # 可被子类覆写

    def _create_section_card(self, title_text: str) -> tuple[QFrame, QVBoxLayout]:
        """创建带圆角边框和标题的卡片容器。

        Returns:
            (card_frame, card_layout) — 调用方通过 card_layout.addWidget/addLayout 填充内容。
        """
        card = QFrame()
        card.setObjectName("sectionCard")
        card.setFrameShape(QFrame.Shape.StyledPanel)
        card.setFrameShadow(QFrame.Shadow.Raised)

        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(20, 18, 20, 18)
        card_layout.setSpacing(12)

        # 标题行（左竖线装饰 + 文字）
        title_row = QHBoxLayout()
        title_row.setSpacing(10)
        accent_bar = QFrame()
        accent_bar.setObjectName("sectionTitleAccent")
        accent_bar.setFixedWidth(3)
        accent_bar.setFixedHeight(18)
        title_row.addWidget(accent_bar)
        title_lbl = StrongBodyLabel(title_text)
        title_row.addWidget(title_lbl)
        title_row.addStretch()
        card_layout.addLayout(title_row)

        return card, card_layout

    def _apply_card_theme(self):
        """根据当前主题刷新所有卡片和标题装饰条的样式。

        亮色模式：白色卡片 + 柔和阴影；
        暗色模式：深色玻璃质感卡片 + 微妙内发光。
        标题装饰条跟随当前强调色。
        """
        is_dark = qconfig.theme == Theme.DARK
        radius = self._card_border_radius

        if is_dark:
            card_bg = "#2a2a2a"
            card_border = "#3a3a3a"
            # 暗色阴影（透明边框模拟内发光）
            shadow = "0px 0px 0px rgba(0,0,0,0)"
        else:
            card_bg = "#ffffff"
            card_border = "#e8e8e8"
            # 亮色阴影（QSS 用透明边框 + 伪阴影效果）
            shadow = "0px 0px 0px rgba(0,0,0,0)"

        accent_color = themeColor().name()

        qss = (
            f"QFrame#sectionCard {{"
            f"  background: {card_bg};"
            f"  border: 1px solid {card_border};"
            f"  border-radius: {radius}px;"
            f"}}"
            f"QFrame#sectionCard:hover {{"
            f"  border-color: {accent_color}40;"
            f"}}"
            f"QFrame#sectionTitleAccent {{"
            f"  background: {accent_color};"
            f"  border-radius: 1.5px;"
            f"  min-width: 3px;"
            f"}}"
        )
        self.setStyleSheet(qss)