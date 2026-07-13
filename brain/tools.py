"""
tools.py — a live, API-driven tool/plugin registry.

Register ANY CLI tool or other AI (opencode + a free model, a local LLM, a TTS/ASR command,
an image generator, a shell utility…), optionally install it, and let the brain interact with
it. A tool is just: a command template with a {input} placeholder, and the MODALITY of its
output. BrainLife folds that output into its ONE byte stream (senses.py) — text is learned as
language; audio/image/bytes are encoded (cochlea/retina) into the same "electricity" — so any
modality, or a combination of tools, becomes something the brain learns from.

Specs persist to <run>/tools.json, so registered tools survive checkpoint/resume.
Security: tools run real commands on this machine — only register tools you trust; this is
your own local brain, driven by you.
"""
from __future__ import annotations
import os, json, shlex, subprocess


class ToolRegistry:
    def __init__(self, path):
        self.path = path
        self.tools = {}
        self._load()

    def _load(self):
        try:
            if os.path.exists(self.path):
                self.tools = json.load(open(self.path))
        except Exception:
            self.tools = {}

    def _save(self):
        try:
            json.dump(self.tools, open(self.path, "w"), indent=2)
        except Exception:
            pass

    def add(self, spec):
        name = (spec.get("name") or "").strip()
        cmd = spec.get("cmd")
        if not name or not cmd:
            return {"ok": False, "err": "need name + cmd (put {input} where the brain's message goes)"}
        self.tools[name] = {
            "name": name, "cmd": cmd, "kind": spec.get("kind", "text"),
            "install": spec.get("install", ""), "shell": bool(spec.get("shell", False)),
            "enabled": bool(spec.get("enabled", True)), "autonomous": bool(spec.get("autonomous", False)),
            "timeout": int(spec.get("timeout", 120)),
        }
        self._save()
        return {"ok": True, "tool": self.tools[name]}

    def remove(self, name):
        self.tools.pop(name, None); self._save(); return {"ok": True}

    def toggle(self, name, enabled=None, autonomous=None):
        t = self.tools.get(name)
        if not t:
            return {"ok": False, "err": "no such tool"}
        if enabled is not None:
            t["enabled"] = bool(enabled)
        if autonomous is not None:
            t["autonomous"] = bool(autonomous)
        self._save()
        return {"ok": True, "tool": t}

    def list(self):
        return list(self.tools.values())

    def get(self, name):
        return self.tools.get(name)

    def autonomous(self):
        """Names of tools the brain may converse with on its own (enabled + autonomous)."""
        return [n for n, t in self.tools.items() if t.get("enabled") and t.get("autonomous")]

    def install(self, name):
        t = self.tools.get(name)
        if not t or not t.get("install"):
            return {"ok": False, "err": "no install command set for this tool"}
        try:
            r = subprocess.run(t["install"], shell=True, capture_output=True, text=True, timeout=900)
            return {"ok": r.returncode == 0, "stdout": (r.stdout or "")[-1500:],
                    "stderr": (r.stderr or "")[-1500:]}
        except Exception as e:
            return {"ok": False, "err": str(e)[:200]}

    def run(self, name, input_text, timeout=None):
        """Invoke the tool with {input} substituted; capture stdout. Returns {ok, output, kind}."""
        t = self.tools.get(name)
        if not t:
            return {"ok": False, "err": "no such tool"}
        if not t.get("enabled"):
            return {"ok": False, "err": "tool disabled"}
        cmd = t["cmd"]; inp = (input_text or "").strip()
        to = timeout or t.get("timeout", 120)
        try:
            if t.get("shell"):
                r = subprocess.run(cmd.replace("{input}", shlex.quote(inp)), shell=True,
                                   capture_output=True, text=True, timeout=to)
            else:                                        # no shell → safer; {input} is one arg
                args = [inp if a == "{input}" else a.replace("{input}", inp) for a in shlex.split(cmd)]
                r = subprocess.run(args, capture_output=True, text=True, timeout=to)
            out = (r.stdout or "").strip()
            return {"ok": bool(out), "output": out, "kind": t.get("kind", "text"),
                    "stderr": "" if out else (r.stderr or "")[-400:]}
        except subprocess.TimeoutExpired:
            return {"ok": False, "err": "timeout"}
        except FileNotFoundError:
            return {"ok": False, "err": "command not found — install it first (set an install cmd)"}
        except Exception as e:
            return {"ok": False, "err": str(e)[:200]}
