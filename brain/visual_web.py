"""
visual_web.py — the brain's eyes on the web: a headless browser it can navigate,
screenshot, and SCROLL, so it "reads" pages as pixels (the visual sensory channel)
rather than as pre-extracted HTML. This is how it learns by scrolling the news or
a page by itself.

Screenshots come back as PIL images; senses.encode_image turns them into the same
byte-level stream as every other sense. Visible text is also available (a second,
language channel) so the brain gets both what it SEES and what it can READ.
"""
from __future__ import annotations
import io
from PIL import Image


class VisualBrowser:
    def __init__(self, headless=True, width=1024, height=768):
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=headless)
        self.page = self._browser.new_page(viewport={"width": width, "height": height})
        self.width, self.height = width, height

    def open(self, url, timeout=20000):
        try:
            self.page.goto(url, timeout=timeout, wait_until="domcontentloaded")
            self.page.wait_for_timeout(500)
            return True
        except Exception:
            return False

    def screenshot(self) -> Image.Image:
        return Image.open(io.BytesIO(self.page.screenshot())).convert("RGB")

    def scroll(self, dy=1200):
        self.page.mouse.wheel(0, dy)
        self.page.wait_for_timeout(400)

    def visible_text(self, max_chars=4000):
        try:
            return self.page.inner_text("body")[:max_chars]
        except Exception:
            return ""

    def read_by_scrolling(self, url, n_scrolls=4, dy=1000):
        """Navigate then scroll down, yielding (screenshot, visible_text) at each step —
        the brain 'reads' the page by scrolling through it, as a human would."""
        if not self.open(url):
            return
        for _ in range(n_scrolls + 1):
            yield self.screenshot(), self.visible_text()
            self.scroll(dy)

    def close(self):
        try:
            self._browser.close(); self._pw.stop()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
