#!/usr/bin/env python
"""
tui.py — watch the brain live, and talk to it while it learns.

An opencode-style terminal interface over BrainLife (brain/life.py):

  ┌ STATE ── AWAKE / SLEEPING, age, phase, granules, η, clock, sleep-pressure ┐
  ┌ VISION ── what it SEES on the web, rendered as ASCII                       ┐
  ┌ PERCEPTION ── what it is sensing now (Claude's lesson / the page text)     ┐
  ┌ THOUGHTS ── what it is thinking / saying (its own generation + babble)     ┐
  ┌ LIFE LOG ── the continuous stream of its life                              ┐
  └ chat box ── talk to it, or command it: 'browse <url>', '?time'             ┘

The brain lives in a background thread (talking to Claude Sonnet 5, browsing the
web, sleeping, developing); the UI updates as it happens. Type to teach/chat with
it live. Ctrl-C or 'q' quits.

    python tui.py                      # full life (Qwen kick-in + Claude + visual web)
    python tui.py --no-teacher         # skip the Qwen birth
    python tui.py --min-awake 120 --max-awake 600   # awake windows in SECONDS
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Static, RichLog, Input, Header, Footer
from textual import work
from rich.text import Text
from rich.panel import Panel

from brain.life import BrainLife


def _bar(frac, width=22, fill="█", empty="░"):
    frac = max(0.0, min(1.0, frac))
    k = int(frac * width)
    return fill * k + empty * (width - k)


class BrainTUI(App):
    CSS = """
    Screen { layout: vertical; }
    #top { height: 18; }
    #state { width: 40; border: round $accent; padding: 0 1; }
    #vision { width: 1fr; border: round $secondary; padding: 0 1; }
    #mid { height: 1fr; }
    #perception { width: 1fr; border: round $primary; padding: 0 1; }
    #thoughts { width: 1fr; border: round $success; padding: 0 1; }
    #log { width: 46; border: round $warning; }
    #chat { dock: bottom; }
    """
    BINDINGS = [("ctrl+c", "quit", "Quit"), ("q", "quit", "Quit")]

    def __init__(self, life: BrainLife):
        super().__init__()
        self.life = life
        self._stop = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="top"):
            yield Static("", id="state")
            yield Static("", id="vision")
        with Horizontal(id="mid"):
            with Vertical():
                yield Static("", id="perception")
                yield Static("", id="thoughts")
            yield RichLog(id="log", wrap=True, markup=True, max_lines=500)
        yield Input(placeholder="talk to the brain…  (or 'browse <url>', '?time')  — Enter to send", id="chat")
        yield Footer()

    def on_mount(self):
        self.query_one("#vision", Static).border_title = "👁 VISION (what it sees)"
        self.query_one("#state", Static).border_title = "🧠 STATE"
        self.query_one("#perception", Static).border_title = "👂 PERCEPTION (what it senses)"
        self.query_one("#thoughts", Static).border_title = "💭 THOUGHTS (what it thinks / says)"
        self.query_one("#log", RichLog).border_title = "📜 LIFE LOG"
        self.life.log_cb = lambda line: self.call_from_thread(self._log_line, line)
        self.run_life()

    # ---- background life -------------------------------------------- #
    @work(thread=True, exclusive=True)
    def run_life(self):
        try:
            self.life.birth(on_update=self._push)
            self.life.run(on_update=self._push, stop_flag=lambda: self._stop)
        except Exception as e:
            self.call_from_thread(self._log_line, f"[red]life stopped: {e}[/red]")

    def _push(self, st):
        self.call_from_thread(self._apply, st)

    # ---- rendering --------------------------------------------------- #
    def _apply(self, st):
        if "ascii" in st and st["ascii"]:
            self.query_one("#vision", Static).update(Text(st["ascii"], style="cyan"))
        if "status" in st:
            self.query_one("#thoughts", Static).border_subtitle = st["status"]
        if "perceived" in st and st["perceived"]:
            self.query_one("#perception", Static).update(st["perceived"][:1600])
        # streaming thought fragment (arrives many times/sec)
        if "mind" in st and "state" not in st:
            self._render_mind(st.get("thought", ""), st["mind"])
        # full state tick
        if "state" in st:
            self._render_state(st)
            self._render_mind(st.get("thought", ""), st.get("mind", ""))

    def _render_mind(self, thought, mind):
        th = Text()
        th.append("stream of thought (it is always thinking):\n\n", style="bold green")
        th.append((mind or "")[-900:], style="white")
        if thought:
            th.append("\n\n▸ now: ", style="bold yellow"); th.append(thought[:120], style="italic yellow")
        self.query_one("#thoughts", Static).update(th)

    def _render_state(self, st):
        awake = st["awake"]
        head = Text()
        if awake:
            head.append("  ☀  AWAKE  \n", style="bold black on green")
            frac = min(1.0, st["awake_seconds"] / max(1.0, st["max_awake"]))
            head.append(f"awake {int(st['awake_seconds'])}s / max {int(st['max_awake'])}s\n", style="green")
            head.append(_bar(frac) + "\n", style="green")
            head.append(f"sleep pressure {int(st['debt'])}/{int(st['debt_threshold'])}\n", style="yellow")
            head.append(_bar(st["debt"] / max(1, st["debt_threshold"])) + "\n", style="yellow")
        else:
            head.append("  🌙  SLEEPING  \n", style="bold white on blue")
            head.append(f"consolidating… {st['sleep_remaining']} left\n", style="blue")
        head.append(f"\nnights slept: {st['nights']}   thoughts: {st['cycle']}\n")
        head.append(f"age {st['age']}  ·  {st['phase']}\n", style="magenta")
        head.append(f"granules {st['granules']}   η {st['eta']:.3f}\n")
        head.append(f"model {st.get('model_gb',0):.3f} GB   disk {st.get('disk_gb',0):.2f} GB\n", style="dim")
        head.append(f"{st.get('device','cpu')} · {st.get('dtype','float32')}\n", style="dim")
        head.append(f"understanding {st['understanding']:.3f}\n", style="bold cyan")
        head.append(f"bits/byte     {st.get('bpb',0):.3f}\n", style="cyan")
        head.append(f"time-sense    {st['time_sense']:.3f}\n", style="cyan")
        head.append(f"novelty {st.get('novelty',0):.2f}  DA {st.get('da_tone',0):.2f}\n", style="dim")
        head.append("── speed ──\n", style="dim")
        head.append(f"⚡ teacher {st.get('teacher_name','—')} {st.get('teacher_cps',0):.0f} c/s\n", style="green")
        head.append(f"⚡ think {st.get('think_bps',0):.0f} b/s · learn {st.get('learn_bps',0):.0f} b/s\n", style="green")
        head.append(f"⏱ {st['clock']}\n", style="dim")
        self.query_one("#state", Static).update(head)

    def _log_line(self, line):
        self.query_one("#log", RichLog).write(line)

    def on_input_submitted(self, event: Input.Submitted):
        txt = event.value.strip()
        event.input.clear()
        if not txt:
            return
        if txt.lower() in ("q", "quit", "exit"):
            self.action_quit(); return
        self.life.inject(txt)
        self._log_line(f"[bold]you → brain:[/bold] {txt}")

    def action_quit(self):
        self._stop = True
        try: self.life.close()
        except Exception: pass
        self.exit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", default="spiking", help="spiking (faithful, default) | rnn (fluent reference)")
    ap.add_argument("--no-teacher", action="store_true")
    ap.add_argument("--no-visual", action="store_true")
    ap.add_argument("--min-awake", type=float, default=90.0, help="min seconds awake")
    ap.add_argument("--max-awake", type=float, default=300.0, help="max seconds awake before it MUST sleep")
    ap.add_argument("--granule", type=int, default=4000)
    ap.add_argument("--budget", type=float, default=0.30)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--dtype", default="auto", help="auto | fp32 | bf16")
    ap.add_argument("--max-model-gb", type=float, default=14.0, help="growth cap on model size")
    ap.add_argument("--checkpoint", default=None,
                    help="continue from this run folder; omit to start a FRESH run (previous runs kept)")
    ap.add_argument("--no-tb", action="store_true", help="disable tensorboard logging")
    args = ap.parse_args()
    from run_life import resolve_run_dir
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
    resdir, resume = resolve_run_dir(here, args.checkpoint)
    print("=" * 78)
    print(f"[run folder]  {resdir}   ({'RESUMING' if resume else 'FRESH RUN'})")
    print(f"[watch log]   tail -f {os.path.join(resdir, 'life.log')}")
    print(f"[tensorboard] tensorboard --logdir {os.path.join(resdir, 'tb')}   (http://localhost:6006)")
    print(f"[continue later] python tui.py --checkpoint {resdir}")
    print("=" * 78)
    life = BrainLife(resdir, core=args.core, granule=args.granule,
                     use_teacher=not args.no_teacher, use_visual=not args.no_visual,
                     min_awake=args.min_awake, max_awake=args.max_awake, budget=args.budget,
                     device=args.device, dtype=args.dtype, max_model_gb=args.max_model_gb,
                     resume=resume, use_tb=not args.no_tb)
    BrainTUI(life).run()


if __name__ == "__main__":
    main()
