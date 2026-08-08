# splash_screen.py — 启动动画窗口
"""软件启动时的现代化启动动画（Splash Screen）。

无边框圆角卡片，依次展示：
    圆形应用图标 → 应用名称 → 版本号 → 加载状态文字 + 不定进度条。

特性：
    - 跟随系统亮/暗主题（主窗口创建前 qfluentwidgets 主题尚未初始化，系统主题是唯一可靠信号源）
    - 启动时淡入，主窗口就绪后短暂停留并淡出自动关闭
    - 阴影圆角卡片 + 现代排版

用法：
    splash = SplashScreen(icon_path)
    splash.show()
    splash.fade_in()
    splash.set_message("正在加载...")
    app.processEvents()          # 让动画帧与文字立即渲染
    ...
    splash.finish()              # 淡出后自动关闭
"""

import os
import time
import ctypes

from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve, QTimer
from PySide6.QtGui import QColor, QGuiApplication, QPainter, QPainterPath, QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel,
    QGraphicsDropShadowEffect,
)

from qfluentwidgets import CaptionLabel, BodyLabel, IndeterminateProgressBar, themeColor

from core.log import logger

# 从 main.py 导入版本号
try:
    from main import APP_VERSION
except ImportError:
    APP_VERSION = "v26h8"

# 卡片与图标尺寸
_CARD_W, _CARD_H = 360, 250
_ICON_SIZE = 84


class _SplashCard(QWidget):
    """圆角卡片：绘制启动画面的圆角背景。"""

    def __init__(self, bg_color: str, parent=None):
        super().__init__(parent)
        self._bg_color = QColor(bg_color)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(self.rect(), 14, 14)
        painter.fillPath(path, self._bg_color)
        painter.end()


class _CircularIconLabel(QLabel):
    """将方形图标裁剪为圆形显示（抗锯齿）；图标缺失时回退为强调色圆形。"""

    def __init__(self, pixmap: QPixmap, diameter: int, fallback_color: QColor, parent=None):
        super().__init__(parent)
        self.setFixedSize(diameter, diameter)
        self._pixmap = pixmap
        self._fallback = fallback_color

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        path = QPainterPath()
        path.addEllipse(0, 0, w, h)
        if self._pixmap.isNull():
            painter.fillPath(path, self._fallback)
        else:
            painter.setClipPath(path)
            scaled = self._pixmap.scaled(
                w, h,
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )
            painter.drawPixmap(
                (w - scaled.width()) // 2, (h - scaled.height()) // 2, scaled
            )
        painter.end()


class SplashScreen(QWidget):
    """启动动画窗口。

    参数：
        icon_path: 应用图标路径（PNG/ICO），不存在时回退为强调色圆形。
        app_name: 应用显示名称。
        version: 版本号文案。
    """

    _FADE_IN_MS = 220
    _FADE_OUT_MS = 420
    _HOLD_MS = 320  # finish() 后停留时长，让用户看到“启动完成”
    _MIN_VISIBLE_MS = 1500  # 启动动画最短展示时长（毫秒）

    def __init__(self, icon_path: str = "", app_name: str = "ustxPlayer-preview",
                 version: str = APP_VERSION):
        # WindowStaysOnTopHint：保证动画显示在已打开的旧实例/其他窗口之上。
        # Tool 窗口显示时不激活、不抢焦点，Windows 会将其沉到激活窗口之下，
        # 若无置顶标志，启动动画可能被其他窗口完全遮挡而“看不见”。
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowStaysOnTopHint,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setFixedSize(_CARD_W + 48, _CARD_H + 44)

        self._is_closing = False
        self._on_fade_out = None  # 动画开始淡出时的回调（用于显示主窗口）
        self._shown_at = time.monotonic()  # 显示时刻，finish() 依此保证最短展示时长

        # 主题配色：以系统主题为准（见模块 docstring 说明）
        self._is_dark = self._is_system_dark()
        self._build_ui(icon_path, app_name, version)

        self._center_on_screen()

        # 窗口透明度动画：用 windowOpacity（窗口级属性），不依赖 QGraphicsEffect。
        # QGraphicsOpacityEffect 在主窗口同步阻塞构造期间无法渲染，会导致窗口不显示。
        self.setWindowOpacity(1.0)
        self._fade = QPropertyAnimation(self, b"windowOpacity", self)
        self._fade.setEasingCurve(QEasingCurve.Type.OutCubic)

        logger.info(f"启动动画已创建（{'暗色' if self._is_dark else '亮色'}主题）")

    # ===================== UI 构建 =====================

    def _build_ui(self, icon_path: str, app_name: str, version: str):
        if self._is_dark:
            bg, fg, sub = "#1f1f1f", "#ffffff", "rgba(255, 255, 255, 0.62)"
        else:
            bg, fg, sub = "#ffffff", "#1a1a1a", "rgba(0, 0, 0, 0.55)"

        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 22, 24, 22)

        # 阴影卡片
        self._card = _SplashCard(bg, self)
        self._card.setFixedSize(_CARD_W, _CARD_H)
        shadow = QGraphicsDropShadowEffect(self._card)
        shadow.setBlurRadius(36)
        shadow.setOffset(0, 8)
        shadow.setColor(QColor(0, 0, 0, 90 if self._is_dark else 70))
        self._card.setGraphicsEffect(shadow)
        outer.addWidget(self._card, 0, Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout(self._card)
        layout.setContentsMargins(44, 30, 44, 26)
        layout.setSpacing(4)

        # 圆形图标
        pixmap = QPixmap(icon_path) if icon_path and os.path.exists(icon_path) else QPixmap()
        icon_label = _CircularIconLabel(
            pixmap, _ICON_SIZE, QColor(themeColor().name()), self._card
        )
        layout.addWidget(icon_label, 0, Qt.AlignmentFlag.AlignHCenter)

        # 应用名称
        name_label = QLabel(app_name, self._card)
        name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name_label.setStyleSheet(
            f"font-size: 20px; font-weight: 600; color: {fg};"
            "background: transparent; border: none;"
        )
        layout.addWidget(name_label)

        # 版本号
        ver_label = CaptionLabel(version, self._card)
        ver_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ver_label.setStyleSheet(f"color: {sub}; background: transparent; border: none;")
        layout.addWidget(ver_label)

        layout.addStretch(1)

        # 加载状态文字
        self._status_label = BodyLabel("正在启动...", self._card)
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_label.setStyleSheet(
            f"color: {sub}; background: transparent; border: none;"
        )
        layout.addWidget(self._status_label)

        # 不定进度条（构造后自动开始动画）
        self._progress = IndeterminateProgressBar(self._card)
        self._progress.setFixedHeight(4)
        layout.addWidget(self._progress)

        layout.addSpacing(2)

    # ===================== 对外接口 =====================

    def set_message(self, message: str):
        """更新状态文字；配合 app.processEvents() 即时渲染。"""
        self._status_label.setText(message)

    def fade_in(self):
        """立即可见并强制置顶（不依赖事件循环，show 后第一帧即可见）。"""
        self.setWindowOpacity(1.0)
        self._force_topmost()

    def finish(self, on_fade_out=None):
        """主窗口就绪：保证动画至少展示 _MIN_VISIBLE_MS 后淡出并自动关闭。

        参数：
            on_fade_out: 动画开始淡出时的回调（例如显示主窗口），
                         实现「动画期间加载 GUI、动画结束后启动 GUI」的节奏。

        主窗口创建通常远快于最短展示时长，故按已展示时间动态计算剩余等待，
        而不是固定延迟，避免在慢速机器上额外拖长启动时间。
        """
        if self._is_closing:
            return
        self._on_fade_out = on_fade_out
        self.set_message("启动完成")
        elapsed_ms = (time.monotonic() - self._shown_at) * 1000
        remaining_ms = max(0, self._MIN_VISIBLE_MS - elapsed_ms)
        QTimer.singleShot(int(remaining_ms + self._HOLD_MS), self._fade_out)

    # ===================== 内部实现 =====================

    def showEvent(self, event):
        """窗口显示时记录时刻并强制置顶。"""
        super().showEvent(event)
        self._shown_at = time.monotonic()
        self._force_topmost()

    def _force_topmost(self):
        """用 Win32 API 强制将窗口置顶到所有窗口之上。

        Qt 的 WindowStaysOnTopHint 对 Tool 窗口在某些场景下不可靠
        （Tool 窗口不激活、易被激活窗口压制），直接调用 SetWindowPos
        设置 HWND_TOPMOST 是 Windows 上最可靠的置顶方式。
        """
        try:
            hwnd = int(self.winId())
            if not hwnd:
                return
            # HWND_TOPMOST = -1; SWP_NOMOVE|SWP_NOSIZE = 0x0001|0x0002
            ctypes.windll.user32.SetWindowPos(
                hwnd, -1, 0, 0, 0, 0, 0x0001 | 0x0002
            )
        except Exception:
            pass

    def _fade_out(self):
        if self._is_closing:
            return
        self._is_closing = True
        # 动画开始淡出时先触发回调（显示主窗口），与淡出无缝衔接
        if self._on_fade_out is not None:
            callback, self._on_fade_out = self._on_fade_out, None
            callback()
        logger.info("启动动画已关闭")
        self._fade.stop()
        self._fade.setDuration(self._FADE_OUT_MS)
        self._fade.setStartValue(self.windowOpacity())
        self._fade.setEndValue(0.0)
        self._fade.finished.connect(self.close)
        self._fade.start()

    def _center_on_screen(self):
        """居中于主屏幕可用区域。"""
        screen = self.screen() or QGuiApplication.primaryScreen()
        if screen is None:
            return
        geo = screen.availableGeometry()
        self.move(
            geo.center().x() - self.width() // 2,
            geo.center().y() - self.height() // 2,
        )

    @staticmethod
    def _is_system_dark() -> bool:
        """判断系统当前是否为暗色模式。"""
        app = QGuiApplication.instance()
        if app is None:
            return False
        try:
            return QGuiApplication.styleHints().colorScheme() == Qt.ColorScheme.Dark
        except Exception:
            return False
