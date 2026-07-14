"""
partner.py — the world the brain learns from: Claude Sonnet 5 (a stronger LLM it
converses with) and the open web (pages it navigates). Both return plain TEXT,
which the byte brain distils by next-byte prediction.

  claude_say(prompt) -> Sonnet 5's reply (via the `claude` CLI, no tools, budget-capped)
  web_text(url)      -> readable text of a web page
  web_topic(topic)   -> a Wikipedia article's text (simple "navigate the web")

The design: the brain emits a (rough) message or picks a topic; Claude answers with
rich, correct text; the brain learns from that text. Over many turns it "discovers
the world by talking," and web pages add real-world text beyond the conversation.
"""
from __future__ import annotations
import subprocess
import threading
import shutil
import re
import urllib.parse

_CLAUDE = shutil.which("claude") or "claude"
_UA = "Mozilla/5.0 (X11; Linux x86_64) sapience-learner/1.0"
_TEACH_SYSTEM = (
    "You are a patient teacher talking to a small language model that is learning "
    "to understand the world by reading your words. Reply with a rich, correct, "
    "self-contained explanation of about 8-12 sentences of plain prose. Use simple, "
    "common words and clear sentences (the student learns letter by letter). "
    "No markdown, no lists, no headings, no code fences, no tool use — just teach in a paragraph."
)


# in-flight teacher subprocesses, so a stop/kill can interrupt a blocking Sonnet call immediately
# instead of waiting up to `timeout` seconds for it to return.
_ACTIVE = set()
_ACTIVE_LOCK = threading.Lock()


def kill_active_calls():
    """Terminate every in-flight teacher subprocess — makes stop/kill instant even mid-lesson."""
    with _ACTIVE_LOCK:
        procs = list(_ACTIVE)
    for p in procs:
        try: p.kill()
        except Exception: pass


def claude_say(prompt, system=_TEACH_SYSTEM, model="sonnet", budget=0.30, timeout=120):
    """One-shot Sonnet 5 reply as plain text. Returns '' on failure. Runs as a tracked subprocess
    so a stop/kill can terminate it immediately (see kill_active_calls)."""
    cmd = [
        _CLAUDE, "-p", prompt,
        "--model", model,
        "--output-format", "text",
        "--disallowedTools", "Bash,Read,Write,Edit,WebFetch,WebSearch,Task,Glob,Grep",
        "--max-budget-usd", str(budget),
    ]
    if system:
        cmd += ["--append-system-prompt", system]
    p = None
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        with _ACTIVE_LOCK: _ACTIVE.add(p)
        out, _ = p.communicate(timeout=timeout)
        return (out or "").strip()
    except subprocess.TimeoutExpired:
        try: p.kill(); p.communicate(timeout=5)
        except Exception: pass
        return ""
    except Exception:
        try:
            if p: p.kill()
        except Exception: pass
        return ""
    finally:
        if p is not None:
            with _ACTIVE_LOCK: _ACTIVE.discard(p)


def _html_to_text(html, max_chars=20000):
    html = re.sub(r"(?is)<(script|style|table|sup|ref)[^>]*>.*?</\1>", " ", html)
    html = re.sub(r"(?is)<[^>]+>", " ", html)
    txt = re.sub(r"&#?\w+;", " ", html)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt[:max_chars]


def web_text(url, max_chars=20000, timeout=15):
    import requests
    try:
        r = requests.get(url, headers={"User-Agent": _UA}, timeout=timeout)
        if r.status_code != 200:
            return ""
        return _html_to_text(r.text, max_chars)
    except Exception:
        return ""


def web_topic(topic, max_chars=20000):
    """Fetch a Wikipedia article's CLEAN prose for a topic. Uses the extracts API (plain text,
    no navigation chrome) so the brain learns real article language — not 'Jump to content Main
    menu move to sidebar hide' UI boilerplate, which polluted both learning and the probe when
    the raw HTML page was scraped. Falls back to the HTML scrape if the API is unreachable."""
    import requests
    try:
        r = requests.get("https://en.wikipedia.org/w/api.php",
                         params={"action": "query", "prop": "extracts", "explaintext": 1,
                                 "redirects": 1, "format": "json", "titles": topic.strip()},
                         headers={"User-Agent": _UA}, timeout=15)
        pages = r.json().get("query", {}).get("pages", {})
        for _, pg in pages.items():
            txt = pg.get("extract", "")
            if txt and len(txt) > 200:
                return re.sub(r"\s+", " ", txt).strip()[:max_chars]
    except Exception:
        pass
    slug = urllib.parse.quote(topic.strip().replace(" ", "_"))    # fallback: scrape the page
    return web_text(f"https://en.wikipedia.org/wiki/{slug}", max_chars=max_chars)


def check_claude():
    """Quick availability probe; returns (ok, sample_reply)."""
    reply = claude_say("Reply with exactly: ready", system="Reply with exactly one word.", timeout=60)
    return (bool(reply), reply[:60])
