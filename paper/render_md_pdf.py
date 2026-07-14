"""Render a Markdown paper → styled HTML → PDF (headless Chromium via Playwright; MathJax typesets any
LaTeX exactly as on screen). pandoc/LaTeX are not needed.

    python paper/render_md_pdf.py paper/THE_LIVING_BRAIN.md   # → paper/THE_LIVING_BRAIN.{html,pdf}
"""
import sys, os, markdown

SRC = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "THE_LIVING_BRAIN.md"))
BASE = os.path.splitext(SRC)[0]
HTML, OUT = BASE + ".html", BASE + ".pdf"

body = markdown.markdown(open(SRC).read(),
                         extensions=["tables", "fenced_code", "toc", "sane_lists", "attr_list"])

TEMPLATE = """<!doctype html><html><head><meta charset="utf-8">
<script>window.MathJax={tex:{inlineMath:[['$','$'],['\\\\(','\\\\)']]}};</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
 body{font:15px/1.6 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#111;
      max-width:820px;margin:32px auto;padding:0 20px}
 h1{font-size:26px;border-bottom:2px solid #222;padding-bottom:6px}
 h2{font-size:20px;margin-top:28px;border-bottom:1px solid #ccc;padding-bottom:3px}
 h3{font-size:16px;margin-top:20px;color:#333}
 code{background:#f4f4f4;padding:1px 4px;border-radius:3px;font-size:90%}
 pre{background:#f6f8fa;padding:10px 12px;border-radius:6px;overflow-x:auto;font-size:13px}
 pre code{background:none;padding:0}
 table{border-collapse:collapse;margin:12px 0;font-size:14px;width:100%}
 th,td{border:1px solid #ccc;padding:5px 9px;text-align:left}
 th{background:#f0f0f0}
 blockquote{border-left:3px solid #bbb;margin:0;padding-left:14px;color:#555}
 a{color:#0645ad}
 @media print{body{max-width:none}h2{break-after:avoid}table,pre{break-inside:avoid}}
</style></head><body>
""" + body + "</body></html>"

with open(HTML, "w") as f:
    f.write(TEMPLATE)
print("HTML:", HTML)

from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    browser = p.chromium.launch(args=["--no-sandbox"])
    page = browser.new_page()
    page.goto("file://" + HTML, wait_until="networkidle", timeout=180000)
    try:
        page.evaluate("""async () => { if (window.MathJax?.startup?.promise) await window.MathJax.startup.promise; }""")
    except Exception as e:
        print("MathJax wait note:", str(e)[:80])
    page.wait_for_timeout(2000)
    page.pdf(path=OUT, format="A4", print_background=True,
             margin={"top": "16mm", "bottom": "16mm", "left": "14mm", "right": "14mm"})
    browser.close()
print("PDF:", OUT, "(%.2f MB)" % (os.path.getsize(OUT) / 1e6))
