#!/usr/bin/env python
"""
run_life.py — the brain's continuous life, HEADLESS (no TUI).

Same BrainLife the TUI drives. It thinks continuously, senses Claude + the visual web
in the background, sleeps when it decides, develops each night, tells time, and
checkpoints so you can stop and --resume. For the live interface use `python tui.py`.

  python run_life.py                         # think + Claude + visual web, forever
  python run_life.py --no-teacher --no-visual
  python run_life.py --resume                # continue the saved life
  python run_life.py --device cuda --dtype bf16 --max-model-gb 14   # when the GPU is back
"""
import os, sys, argparse, signal, time, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brain.life import BrainLife

_STOP = {"f": False}
signal.signal(signal.SIGINT, lambda *_: _STOP.update(f=True))


def resolve_run_dir(here, checkpoint):
    """No --checkpoint → a fresh timestamped run folder (previous runs are NOT touched).
    --checkpoint DIR → continue that exact run in place."""
    if checkpoint:
        d = os.path.abspath(checkpoint)
        if not os.path.isdir(d):
            print(f"[error] checkpoint folder not found: {d}"); sys.exit(1)
        return d, True
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    d = os.path.join(here, "runs", f"run_{ts}")
    os.makedirs(d, exist_ok=True)
    return d, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", default="spiking", help="spiking (faithful, default) | rnn (fluent reference)")
    ap.add_argument("--no-teacher", action="store_true")
    ap.add_argument("--no-visual", action="store_true")
    ap.add_argument("--min-awake", type=float, default=90.0)
    ap.add_argument("--max-awake", type=float, default=300.0)
    ap.add_argument("--granule", type=int, default=4000)
    ap.add_argument("--budget", type=float, default=0.30)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--max-model-gb", type=float, default=14.0)
    ap.add_argument("--checkpoint", default=None,
                    help="continue from this run folder; omit to start a FRESH run (previous runs kept)")
    ap.add_argument("--no-tb", action="store_true", help="disable tensorboard logging")
    ap.add_argument("--stop", nargs="?", const="__latest__", default=None,
                    help="graceful-stop a running brain (folder, or newest) and exit — no pkill needed")
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    if args.stop is not None:                        # graceful stop, then exit
        from brain.life import request_stop, latest_run
        base = os.path.join(here, "runs")
        d = latest_run(base) if args.stop == "__latest__" else (args.stop if os.path.isabs(args.stop) else os.path.join(base, args.stop))
        if not d:
            print("no run to stop"); return
        request_stop(d); print(f"graceful stop requested → {d} (it will checkpoint and exit)"); return
    resdir, resume = resolve_run_dir(here, args.checkpoint)

    life = BrainLife(resdir, core=args.core, granule=args.granule,
                     use_teacher=not args.no_teacher, use_visual=not args.no_visual,
                     min_awake=args.min_awake, max_awake=args.max_awake, budget=args.budget,
                     device=args.device, dtype=args.dtype, max_model_gb=args.max_model_gb,
                     resume=resume, use_tb=not args.no_tb)

    last = [0.0]
    def on_update(st):
        if "state" not in st:
            return
        if time.time() - last[0] < 2.0:            # print a status line every ~2s
            return
        last[0] = time.time()
        icon = "☀ AWAKE " if st["awake"] else "🌙 SLEEP"
        print(f"{icon} | thoughts {st['cycle']} night {st['nights']} {st['phase']} age {st['age']} | "
              f"gran {st['granules']} model {st['model_gb']:.3f}GB {st['device']}/{st['dtype']} | "
              f"und {st['understanding']:.3f} bpb {st.get('bpb',0):.2f} time {st['time_sense']:.3f} | "
              f"⚡{st.get('teacher_name','—')} {st.get('teacher_cps',0):.0f}c/s think {st.get('think_bps',0):.0f}b/s "
              f"learn {st.get('learn_bps',0):.0f}b/s | ⏱ {st['clock']}")
        print(f"        thinks> {st['thought'][:70]!r}")

    logf = os.path.join(resdir, "life.log")
    tbf = os.path.join(resdir, "tb")
    print("=" * 78)
    print(f"[run folder] {resdir}")
    print(f"[device]     {life.dev} / {life.dtype}  |  model cap {args.max_model_gb} GB  |  "
          f"{'RESUMING' if resume else 'FRESH RUN'}")
    print(f"[watch log]  tail -f {logf}")
    print(f"[tensorboard] tensorboard --logdir {tbf}   (then open http://localhost:6006)")
    print(f"[continue later] python run_life.py --checkpoint {resdir}")
    print("=" * 78, flush=True)
    life.birth(on_update=on_update)
    life.run(on_update=on_update, stop_flag=lambda: _STOP["f"])
    print(f"\nstopped; lived {life.cycle} thoughts, {life.slept_count} nights; "
          f"continue with: python run_life.py --checkpoint {resdir}")


if __name__ == "__main__":
    main()
