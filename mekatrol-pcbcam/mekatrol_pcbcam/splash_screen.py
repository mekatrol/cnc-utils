from __future__ import annotations

from PySide6.QtCore import QElapsedTimer, QEvent, QEventLoop, QRect, QTimer, Qt, QUrl
from PySide6.QtGui import (
    QCloseEvent,
    QColor,
    QDesktopServices,
    QFontDatabase,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import QSplashScreen

from . import PROJECT_URL, __version__
from .theme import AppTheme


class SplashScreen(QSplashScreen):
    """Application splash screen with explicit startup-state handling.

    This widget is no longer "debug-only". It is now the normal splash screen used
    during startup, while still supporting the special debug hold mode controlled
    by ``DEBUG_SPLASH_ENV`` in ``app.py``.

    The behavior is easiest to understand as a small state machine:

    1. Visible / starting up
       The splash is shown while the application constructs the main window and
       restores any startup state. During this phase the splash must remain visible
       for at least the configured minimum duration.

    2. Visible / startup complete / auto-close pending
       Startup work has finished, but the minimum visible timer may still be
       running. ``wait_until_ready()`` blocks until that minimum time has elapsed.

    3. Visible / manually held open
       A normal left click on the splash body sets ``_hold_open_requested``. That
       means the user wants to inspect the splash and explicitly close it later
       using the close action. The main window must not appear underneath in this
       state.

    4. Visible / debug hold
       When debug splash hold is enabled, ``wait_for_click()`` blocks until a user
       click arrives. This is intentionally separate from the normal hold-open path
       because the debug mode exists for layout inspection and should not depend on
       startup timing.

    5. Closed / dismissed
       Once the splash closes, ``_dismiss_requested`` becomes true and any nested
       event loops are released. ``app.py`` can then continue and show the main
       window.

    There are three independent nested event loops because they represent three
    separate reasons to pause control flow:

    - ``_minimum_visible_loop`` waits for the configured minimum visible time.
    - ``_debug_click_loop`` waits for the explicit debug click-to-continue action.
    - ``_close_loop`` waits for the splash window to actually close before the main
      window is allowed to appear.

    Keeping those loops separate makes the intent of each wait clearer and avoids
    overloading one loop with unrelated conditions.
    """

    def __init__(self, pixmap: QPixmap, theme: AppTheme) -> None:
        super().__init__(pixmap)
        self._theme = theme
        # Debug mode pauses startup until the splash is clicked. This loop is only
        # active in that special environment-controlled path.
        self._debug_click_loop: QEventLoop | None = None
        # The main startup path waits here only to satisfy the configured minimum
        # splash duration. It is unrelated to explicit user dismissal.
        self._minimum_visible_loop: QEventLoop | None = None
        # After startup work completes, app.py can wait here until the splash is
        # actually closed, guaranteeing the main window stays hidden until then.
        self._close_loop: QEventLoop | None = None

        self._border_width = 2
        self._source_pixmap = pixmap
        self._minimum_visible_ms = 0
        self._startup_complete = False
        self._dismiss_requested = False
        self._hold_open_requested = False
        self._visible_timer = QElapsedTimer()

        # Mouse tracking is required so the cursor can change over the clickable
        # URL and the always-visible Close action without requiring a button press.
        self.setMouseTracking(True)
        self.setPixmap(self._composited_pixmap())
        self.setFont(self._metadata_font())

    def begin_startup_timing(self, minimum_visible_ms: int) -> None:
        """Reset the splash state at the beginning of a startup sequence.

        This transitions the splash into the initial "visible / starting up" state.
        The elapsed timer is the source of truth for how much longer the splash must
        remain visible before automatic dismissal is allowed.
        """

        self._minimum_visible_ms = max(0, minimum_visible_ms)
        self._startup_complete = False
        self._dismiss_requested = False
        self._hold_open_requested = False
        self._visible_timer.start()

    def mark_startup_complete(self) -> None:
        """Record that the application finished startup work.

        This does not close the splash by itself. It only transitions the state
        machine from "starting up" to "startup complete", after which
        ``wait_until_ready()`` may either finish immediately or wait for the
        remaining minimum-visible duration.
        """

        self._startup_complete = True

    def wait_until_ready(self) -> None:
        """Wait until automatic splash dismissal is allowed.

        Normal startup uses this after ``mark_startup_complete()``. The method does
        nothing if startup is not complete yet or if dismissal already happened.

        The important distinction is:
        - This method enforces the minimum visible duration.
        - It does not wait for user dismissal.
        - It does not close the splash by itself.

        Closing and manual hold behavior are handled separately by ``app.py`` and
        ``wait_until_closed()``.
        """

        if not self._startup_complete:
            return

        if self._dismiss_requested:
            return

        remaining_ms = self._remaining_visible_ms()
        if remaining_ms > 0:
            loop = QEventLoop()
            self._minimum_visible_loop = loop
            QTimer.singleShot(remaining_ms, loop.quit)
            loop.exec()
            self._minimum_visible_loop = None

    def wait_for_click(self) -> None:
        """Block until the splash is clicked in debug hold mode.

        This is intentionally isolated from the normal startup state machine. Debug
        hold exists to keep the splash on screen for inspection and should not be
        coupled to the standard auto-close timing rules.
        """

        loop = QEventLoop()
        self._debug_click_loop = loop
        loop.exec()
        self._debug_click_loop = None

    def hold_open_requested(self) -> bool:
        """Return whether the user has requested manual dismissal.

        Once true, automatic close should no longer occur. The splash remains open
        until the user explicitly activates the Close action or the window is closed
        via native window controls.
        """

        return self._hold_open_requested and not self._dismiss_requested

    def enable_manual_close(self) -> None:
        """Bring the splash to the front when it enters the held-open state.

        The window is already created with close affordances in ``app.py``. This
        method exists to make the state transition explicit at the call site and to
        keep any future "manual close mode" adjustments in one place.
        """

        self.raise_()

    def wait_until_closed(self) -> None:
        """Block until the splash window is actually closed.

        ``app.py`` uses this after startup has completed when the splash is still
        visible. This ensures the main window does not appear until the splash has
        been dismissed.
        """

        if self._dismiss_requested or not self.isVisible():
            return

        loop = QEventLoop()
        self._close_loop = loop
        loop.exec()
        self._close_loop = None

    def _remaining_visible_ms(self) -> int:
        """Return how much of the minimum splash duration remains."""

        if not self._visible_timer.isValid():
            return 0
        elapsed_ms = self._visible_timer.elapsed()
        return max(0, self._minimum_visible_ms - elapsed_ms)

    def show_status_message(self, message: str) -> None:
        """Render the current startup instruction near the bottom of the splash."""

        self.showMessage(
            message,
            alignment=Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
            color=self.message_color(),
        )

    def _border_color(self) -> QColor:
        return self._theme.named_color("splash_border")

    def _background_color(self) -> QColor:
        return self._theme.named_color("splash_background")

    def message_color(self) -> QColor:
        return self._theme.named_color("splash_message_text")

    def _metadata_font(self):
        # A fixed-width font keeps the aligned "Version" and "Website" labels
        # stable across platforms and avoids layout drift in the bottom-right block.
        fixed_font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        fixed_font.setPointSize(max(16, fixed_font.pointSize()))
        return fixed_font

    def _composited_pixmap(self) -> QPixmap:
        """Build the final splash pixmap on top of a theme-aware background.

        The source artwork does not occupy the entire splash window. We expand it
        into a larger composited pixmap so the metadata block and close action have
        consistent room around the artwork.
        """

        composed = QPixmap(
            (self._source_pixmap.width() * 3) // 2,
            (self._source_pixmap.height() * 3) // 2,
        )
        composed.fill(self._background_color())

        painter = QPainter(composed)
        painter.drawPixmap(0, 0, self._source_pixmap)
        painter.end()
        return composed

    def drawContents(self, painter: QPainter) -> None:
        """Draw interactive overlay content on top of the base splash image."""

        super().drawContents(painter)
        painter.save()
        self._draw_close_action(painter)
        self._draw_metadata(painter)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.setPen(QPen(self._border_color(), self._border_width))
        inset = self._border_width // 2
        painter.drawRect(self.rect().adjusted(inset, inset, -inset - 1, -inset - 1))
        painter.restore()

    def _draw_close_action(self, painter: QPainter) -> None:
        """Draw the always-available textual close affordance.

        This action exists even when native titlebar buttons are unavailable or
        suppressed by platform-specific window-manager behavior.
        """

        close_rect = self._close_action_rect()
        painter.save()
        painter.setPen(self.message_color())
        painter.drawText(close_rect, Qt.AlignmentFlag.AlignCenter, "Close")
        painter.restore()

    def _draw_metadata(self, painter: QPainter) -> None:
        """Draw version metadata and the clickable project URL."""

        painter.setFont(self._metadata_font())
        metrics = painter.fontMetrics()
        version_text, website_label_text, website_url_text = self._metadata_text()
        version_rect, website_label_rect, website_url_rect = self._metadata_rects(
            metrics
        )

        painter.setPen(self.message_color())
        painter.drawText(version_rect, Qt.AlignmentFlag.AlignLeft, version_text)
        painter.drawText(
            website_label_rect, Qt.AlignmentFlag.AlignLeft, website_label_text
        )

        painter.save()
        painter.setPen(self._theme.named_color("splash_link"))
        painter.drawText(
            website_url_rect, Qt.AlignmentFlag.AlignLeft, website_url_text
        )
        underline_y = website_url_rect.bottom() - max(1, metrics.descent() // 2)
        painter.drawLine(
            website_url_rect.left(),
            underline_y,
            website_url_rect.right(),
            underline_y,
        )
        painter.restore()

    def _metadata_text(self) -> tuple[str, str, str]:
        version_text = f"{'Version':>8}: {__version__}"
        website_label_text = f"{'Website':>8}: "
        return version_text, website_label_text, PROJECT_URL

    def _metadata_rects(self, metrics) -> tuple[QRect, QRect, QRect]:
        """Return layout rectangles for the aligned metadata block."""

        version_text, website_label_text, website_url_text = self._metadata_text()
        line_height = metrics.height()
        block_width = max(
            metrics.horizontalAdvance(version_text),
            metrics.horizontalAdvance(website_label_text)
            + metrics.horizontalAdvance(website_url_text),
        )
        block_height = line_height * 2
        x_margin = 150
        y_margin = 50
        x = self.rect().right() - block_width - x_margin
        y = self.rect().bottom() - block_height - y_margin - 28

        version_rect = QRect(x, y, block_width, line_height)
        website_label_width = metrics.horizontalAdvance(website_label_text)
        website_url_width = metrics.horizontalAdvance(website_url_text)
        website_top = y + line_height
        website_label_rect = QRect(x, website_top, website_label_width, line_height)
        website_url_rect = QRect(
            x + website_label_width, website_top, website_url_width, line_height
        )
        return version_rect, website_label_rect, website_url_rect

    def _website_rect(self) -> QRect:
        """Return the clickable hit area for the website URL text."""

        return self._metadata_rects(self.fontMetrics())[2]

    def _close_action_rect(self) -> QRect:
        """Return the clickable hit area for the textual Close action."""

        metrics = self.fontMetrics()
        padding_x = 12
        padding_y = 8
        width = metrics.horizontalAdvance("Close") + (padding_x * 2)
        height = metrics.height() + (padding_y * 2)
        margin = 20
        return QRect(self.rect().right() - width - margin, margin, width, height)

    def _open_project_url(self) -> None:
        """Open the project URL in the platform-default browser."""

        QDesktopServices.openUrl(QUrl(PROJECT_URL))

    def _set_pointer_cursor(self, is_pointer: bool) -> None:
        """Swap between the normal arrow and a pointer over interactive regions."""

        desired = (
            Qt.CursorShape.PointingHandCursor
            if is_pointer
            else Qt.CursorShape.ArrowCursor
        )
        if self.cursor().shape() != desired:
            self.setCursor(desired)

    def event(self, event: QEvent) -> bool:
        """Update hover feedback for the URL and Close action."""

        if event.type() in {
            QEvent.Type.MouseMove,
            QEvent.Type.HoverMove,
            QEvent.Type.Enter,
        }:
            position = (
                event.position().toPoint() if hasattr(event, "position") else None
            )
            self._set_pointer_cursor(
                position is not None
                and (
                    self._website_rect().contains(position)
                    or self._close_action_rect().contains(position)
                )
            )
        elif event.type() == QEvent.Type.Leave:
            self._set_pointer_cursor(False)
        return super().event(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle the splash's three left-click behaviors.

        The click priority is:

        1. Project URL click opens the browser.
        2. Close action click dismisses the splash.
        3. Debug hold click releases the debug wait loop.
        4. Any other body click marks the splash as manually held open.

        That last behavior is why a generic click does not close the splash. It is
        intentionally treated as a request to keep the splash around until the user
        explicitly closes it.
        """

        if event.button() != Qt.MouseButton.LeftButton:
            event.ignore()
            return

        if self._website_rect().contains(event.position().toPoint()):
            self._open_project_url()

        if self._close_action_rect().contains(event.position().toPoint()):
            self.close()
            event.accept()
            return

        if self._debug_click_loop is not None:
            self._debug_click_loop.quit()
            event.accept()
            return

        self._hold_open_requested = True
        self.enable_manual_close()
        event.accept()

    def closeEvent(self, event: QCloseEvent) -> None:
        """Mark the splash as dismissed and release any active wait loops.

        This is the terminal state transition for the splash state machine.
        Everything waiting on the splash should be able to proceed once this runs.
        """

        self._dismiss_requested = True
        if self._minimum_visible_loop is not None:
            self._minimum_visible_loop.quit()
        if self._debug_click_loop is not None:
            self._debug_click_loop.quit()
        if self._close_loop is not None:
            self._close_loop.quit()
        event.accept()
