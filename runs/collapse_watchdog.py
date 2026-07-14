"""COLLAPSE-PROTECTION WATCHDOG (overnight safety net, runs independently of the Claude session).

The prior run collapsed UNATTENDED (bpb 4->17, representation-magnitude runaway). This watchdog polls
the live brain and, on a SUSTAINED runaway, auto-rescues it: halve eprop_lr_scale (floor 800) and, if
severe, freeze synaptogenesis. It ONLY acts in emergencies — normal optimization is left to the Claude
30-min tuner; this just guarantees the user doesn't wake to a dead brain.

Run: nohup python runs/collapse_watchdog.py &   (writes runs/watchdog.log; also notes actions in tuning_log.md)
"""
import json, time, urllib.request, urllib.parse, os, datetime

BASE = "http://127.0.0.1:8199"
LOG = "/home/dander/workspace/zk/sapience/runs/watchdog.log"
TUNE = "/home/dander/workspace/zk/sapience/runs/tuning_log.md"

# thresholds — healthy is bpb~4.4 / spike~0.08; a runaway climbs to bpb 8->17, spike >0.2
BPB_WARN, BPB_SEVERE = 7.5, 11.0
SPIKE_WARN = 0.22
SUSTAIN = 4                 # consecutive bad polls (~6 min) before acting — avoids transients
POLL = 90                   # seconds
LR_FLOOR = 800.0
COOLDOWN = 900              # 15 min between interventions (let a change settle)


def _get(path):
    with urllib.request.urlopen(BASE + path, timeout=8) as r:
        return json.load(r)


def _post(path, obj):
    data = json.dumps(obj).encode()
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.load(r)


def log(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    with open(LOG, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def note_tuning(msg):
    ts = datetime.datetime.now().strftime("%H:%M")
    try:
        with open(TUNE, "a") as f:
            f.write(f"\n- **{ts} WATCHDOG (auto):** {msg}\n")
    except Exception:
        pass


def read_lr():
    try:
        np = _get("/api/state").get("netparams", {}).get("cortex", {})
        return float(np.get("eprop_lr_scale", 2000.0))
    except Exception:
        return 2000.0


def main():
    log(f"watchdog START (poll {POLL}s, act after {SUSTAIN} sustained bad polls; bpb_warn={BPB_WARN} severe={BPB_SEVERE})")
    bad = 0
    last_action = 0.0
    while True:
        try:
            d = _get("/api/state")
            st = d.get("state", {})
            status = d.get("status")
            bpb = st.get("bpb")
            spike = st.get("spike_rate")
            # only judge while awake & measured
            if status in ("awake", "sleeping") and isinstance(bpb, (int, float)):
                runaway = (bpb > BPB_WARN) or (isinstance(spike, (int, float)) and spike > SPIKE_WARN and bpb > 6.0)
                if runaway:
                    bad += 1
                    log(f"WARN bad poll {bad}/{SUSTAIN}: bpb={bpb:.2f} spike={spike} status={status}")
                else:
                    if bad:
                        log(f"recovered: bpb={bpb:.2f} spike={spike} (reset counter)")
                    bad = 0
                now = time.time()
                if bad >= SUSTAIN and (now - last_action) > COOLDOWN:
                    lr = read_lr()
                    new_lr = max(LR_FLOOR, lr * 0.5)
                    try:
                        _post("/api/net", {"target": "cortex", "eprop_lr_scale": new_lr})
                        msg = f"RUNAWAY (bpb={bpb:.2f} sustained) -> eprop_lr_scale {lr:.0f}->{new_lr:.0f}"
                        if bpb > BPB_SEVERE:
                            _post("/api/set", {"freeze_growth": True})
                            msg += " + FROZE growth (severe)"
                        log(msg)
                        note_tuning(msg + " — auto-rescue; Claude tuner should verify + re-tune when it next runs.")
                        last_action = now
                        bad = 0
                    except Exception as e:
                        log(f"intervention FAILED: {str(e)[:80]}")
            else:
                bad = 0   # being born / not measured yet
        except Exception as e:
            log(f"poll error: {str(e)[:80]}")
        time.sleep(POLL)


if __name__ == "__main__":
    main()
