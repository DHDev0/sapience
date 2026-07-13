"""Render an HTML paper (with MathJax LaTeX) to PDF via headless Chromium — the browser runs
MathJax exactly as on screen, so every equation is typeset correctly in the PDF.

    python paper/render_pdf.py [paper/field-guide.html] [paper/field-guide.pdf]
"""
import sys, os
from playwright.sync_api import sync_playwright

SRC = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "field-guide.html"))
OUT = os.path.abspath(sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(SRC)[0] + ".pdf")

with sync_playwright() as p:
    browser = p.chromium.launch(args=["--no-sandbox"])
    page = browser.new_page()
    page.goto("file://" + SRC, wait_until="networkidle", timeout=180000)
    # let MathJax v3 finish typesetting (its startup promise resolves when the page is done)
    try:
        page.evaluate("""async () => {
            if (window.MathJax && window.MathJax.startup && window.MathJax.startup.promise) {
                await window.MathJax.startup.promise;
            }
        }""")
    except Exception as e:
        print("MathJax wait note:", str(e)[:80])
    page.wait_for_timeout(2500)                       # settle fonts/layout
    page.pdf(path=OUT, format="A4", print_background=True,
             margin={"top": "16mm", "bottom": "16mm", "left": "14mm", "right": "14mm"})
    browser.close()

print("PDF:", OUT, "(%.2f MB)" % (os.path.getsize(OUT) / 1e6))
