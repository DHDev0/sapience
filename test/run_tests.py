#!/usr/bin/env python
"""
run_tests.py — run the whole test suite with NO dependencies (no pytest needed).

    python test/run_tests.py            # run everything
    python test/run_tests.py cortex     # only test_cortex.py
    pytest test/                        # also works if you have pytest

Discovers every `test_*` function in every `test/test_*.py`, runs it, and reports
pass/fail with a one-line summary. Tests use tiny CPU configs and mock the network, so
the whole suite runs in well under a minute.
"""
import os, sys, time, traceback, importlib.util

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
TESTDIR = os.path.dirname(os.path.abspath(__file__))


def load(path):
    spec = importlib.util.spec_from_file_location(os.path.basename(path)[:-3], path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    files = sorted(f for f in os.listdir(TESTDIR)
                   if f.startswith("test_") and f.endswith(".py") and (only is None or only in f))
    passed = failed = 0; fails = []
    t0 = time.time()
    for f in files:
        print(f"\n\033[1m{f}\033[0m")
        try:
            mod = load(os.path.join(TESTDIR, f))
        except Exception as e:
            print(f"  \033[31mIMPORT FAIL\033[0m {e}"); failed += 1; fails.append(f + " (import)"); continue
        for name in sorted(d for d in dir(mod) if d.startswith("test_")):
            fn = getattr(mod, name)
            if not callable(fn):
                continue
            try:
                fn(); print(f"  \033[32m✓\033[0m {name}"); passed += 1
            except Exception as e:
                print(f"  \033[31m✗ {name}\033[0m — {e}")
                traceback.print_exc()
                failed += 1; fails.append(f"{f}::{name}")
    dt = time.time() - t0
    print(f"\n{'='*60}\n{passed} passed, {failed} failed in {dt:.1f}s")
    if fails:
        print("FAILURES:"); [print("  -", x) for x in fails]
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
