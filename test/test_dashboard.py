"""The dashboard control-plane: Controller methods + API surface (brain built directly to skip birth)."""
import os, shutil, json
from brain import partner
partner.claude_say = lambda *a, **k: "A clear simple lesson about the world in plain words."
partner.web_topic = lambda *a, **k: "readable text about the brain and how it learns."
from brain.life import BrainLife
from interface import dashboard as D

BASE = "/tmp/_dash_test"


def _controller_with_brain(sub):
    d = os.path.join(BASE, sub); shutil.rmtree(d, ignore_errors=True)
    c = D.Controller()
    c.life = BrainLife(d, core="spiking", use_teacher=False, use_visual=False,
                       emb=16, hidden=48, layers=1, device="cpu", seed=0)
    c.run_dir = d
    return c


def test_api_help_lists_all_endpoints():
    eps = [k for k in D.API_HELP if k.startswith(("GET", "POST"))]
    assert len(eps) >= 22                                        # the full documented API
    assert "POST /api/teach" in D.API_HELP and "POST /api/net" in D.API_HELP
    for ep in ("GET /api/arch", "GET /api/diag", "POST /api/arch", "GET /api/logs?n=N", "GET /api/history?key=K"):
        assert ep in D.API_HELP                                  # new observability + arch-control API
    assert any(k == "neurons" for k, *_ in D.CHART_KEYS) and any(k == "synapses" for k, *_ in D.CHART_KEYS)


def test_page_renders_self_contained():
    assert "__CHARTS__" not in D.PAGE                            # template fully substituted
    for fn in ("buildCharts", "hoverChart", "renderTools", "renderObs", "renderNet"):
        assert fn in D.PAGE
    assert "<audio" in D.PAGE and "resize:both" in D.PAGE        # replay + resizable charts


def test_controller_live_tuning():
    c = _controller_with_brain("set")
    r = c.set_params({"budget": 0.2, "grow_add": 96, "freeze_sleep": True, "resonate_k": 7})
    assert r["ok"] and c.life.budget == 0.2 and c.life.grow_add == 96 and c.life.freeze_sleep


def test_controller_net_teach_focus():
    c = _controller_with_brain("ntf")
    assert c.set_net({"target": "hippocampus", "beta": 11})["applied"]["beta"] == 11.0
    assert c.teach({"text": "a lesson about rivers and the sea.", "label": "geo"})["ok"]
    assert c.focus({"topics": ["music"], "mode": "topics"})["mode"] == "topics"


def test_controller_tools_and_save():
    c = _controller_with_brain("tools")
    assert c.tools_add({"name": "echo", "cmd": "echo {input}", "kind": "text"})["ok"]
    assert c.tools_list()["tools"][0]["name"] == "echo"
    assert "hi" in c.tools_run({"name": "echo", "input": "hi"})["output"]
    assert c.save()["ok"] and os.path.exists(c.life.ckpt)        # forced checkpoint written


def test_snapshot_shape():
    c = _controller_with_brain("snap")
    c.life._learn_text("the sun is bright and the sky is blue. " * 15, steps=4)
    c._on_state(c.life.state or {})
    c.life._emit(c._on_state)                                    # populate a full state tick
    snap = c.snapshot()
    for key in ("status", "state", "hp", "net", "netparams", "arch", "tools", "observations", "history", "feed", "logs", "compute"):
        assert key in snap, f"snapshot missing {key}"
    assert set(snap["netparams"]) == {"cortex", "hippocampus", "bg", "neuromod", "cerebellum"}


def test_arch_diag_reports_per_part_neurons_and_synapses():
    c = _controller_with_brain("archdiag")
    a = c.life._arch_diag()
    assert a["total_neurons"] > 0 and a["total_synapses"] > 0
    for part in ("cortex", "cerebellum", "bg", "hippocampus", "neuromod"):
        assert part in a["parts"] and "neurons" in a["parts"][part] and "synapses" in a["parts"][part]
    assert a["parts"]["cortex"]["layer_widths"]                 # per-layer visibility


def test_edit_arch_neurons_fixed_synapses_grow_and_prune():
    c = _controller_with_brain("archedit")
    c.life._learn_text("learning changes the connectome over time. " * 12, steps=4)
    n0 = c.life.brain.neuron_count(); s0 = c.life.brain.active_synapse_count()
    g = c.edit_arch({"target": "cortex", "op": "grow_synapses", "amount": 0.3})
    assert g["ok"] and c.life.brain.neuron_count() == n0                 # NEURONS fixed
    assert c.life.brain.active_synapse_count() > s0                      # synapses GREW
    p = c.edit_arch({"target": "cortex", "op": "prune_synapses", "amount": 0.1})
    assert p["ok"] and c.life.brain.neuron_count() == n0                 # still fixed
    gn = c.edit_arch({"target": "cortex", "op": "grow_neurons", "amount": 32})
    assert gn["ok"] and c.life.brain.neuron_count() == n0 + 32           # deliberate neuron growth
    assert c.edit_arch({"target": "bg", "op": "grow_neurons", "amount": 16})["ok"]


def test_arch_reports_params_growrate_device_per_region_and_global():
    c = _controller_with_brain("archfull")
    a = c.life._arch_diag()
    assert a["total_parameters"] > 0 and a["total_neurons"] > 0 and a["total_synapses"] > 0
    for part in ("cortex", "cerebellum", "bg", "hippocampus"):
        p = a["parts"][part]
        assert "parameters" in p and "grow_syn_frac" in p and "device" in p     # per-region full census


def test_set_neurons_and_synapses_to_target_and_global():
    c = _controller_with_brain("settarget")
    # set synapses to a target density on the cortex
    r = c.edit_arch({"target": "cortex", "op": "set_synapses", "density": 0.8})
    assert r["ok"] and abs(c.life.brain.active_synapse_count() / c.life.brain.synapse_capacity() - 0.8) < 0.05
    # set neurons to a target on a module (grows to reach it)
    n = c.life.bg.M.shape[0]
    r = c.edit_arch({"target": "bg", "op": "set_neurons", "amount": n + 40})
    assert r["ok"] and c.life.bg.M.shape[0] == n + 40
    # global fan-out across all regions
    r = c.edit_arch({"target": "all", "op": "grow_synapses", "amount": 0.1})
    assert r["ok"] and set(r["applied"]) >= {"cortex", "cerebellum", "bg", "hippocampus"}


def test_per_region_grow_rate_and_device_live():
    c = _controller_with_brain("growdev")
    assert c.set_net({"target": "hippocampus", "grow_syn_frac": 0.33})["applied"]["grow_syn_frac"] == 0.33
    assert c.life.hippo.grow_syn_frac == 0.33
    assert c.set_net({"target": "all", "prune_frac": 0.02})["ok"]           # global knob
    # per-part device (cpu no-op here; the path + boundary conversions must work)
    assert c.set_device({"target": "bg", "device": "cpu"})["ok"] and str(c.life.bg.device) == "cpu"
    assert c.set_device({"target": "all", "device": "cpu"})["ok"]


def test_arch_and_diag_endpoints_over_http():
    import threading, urllib.request, json as J
    from http.server import ThreadingHTTPServer
    c = _controller_with_brain("archhttp"); D.CTRL.life = c.life; D.CTRL.run_dir = c.run_dir
    c.life._learn_text("a mind that keeps learning. " * 10, steps=3); c.life._emit(D.CTRL._on_state)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), D.H); port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True); th.start()
    B = f"http://127.0.0.1:{port}"
    def get(p): return J.loads(urllib.request.urlopen(B + p, timeout=5).read())
    def post(p, d): return J.loads(urllib.request.urlopen(urllib.request.Request(B + p, data=J.dumps(d).encode(), method="POST"), timeout=5).read())
    try:
        arch = get("/api/arch")
        assert arch.get("total_neurons") and "cortex" in arch.get("parts", {})
        diag = get("/api/diag")
        assert "warnings" in diag and "systems" in diag and "trends_recent" in diag
        assert set(("cortex (§3)", "basal_ganglia (§2)", "hippocampus (§4)")) <= set(diag["systems"])
        r = post("/api/arch", {"target": "cortex", "op": "grow_synapses", "amount": 0.2})
        assert r["ok"] and "added_synapses" in r["applied"]
    finally:
        srv.shutdown(); srv.server_close()


def test_set_net_exposes_new_spiking_params_live():
    # the spiking BG + hippocampus + cortex prune fraction must ALL be tunable live (no restart)
    c = _controller_with_brain("netparams")
    assert c.set_net({"target": "bg", "beta": 0.8, "thr": 0.7})["applied"] == {"beta": 0.8, "thr": 0.7}
    assert c.life.bg.beta == 0.8 and c.life.bg.thr == 0.7
    assert c.set_net({"target": "hippocampus", "thr": 0.6, "g_inh": 1.2})["applied"] == {"thr": 0.6, "g_inh": 1.2}
    assert c.life.hippo.thr == 0.6 and c.life.hippo.g_inh == 1.2
    assert c.set_net({"target": "cortex", "prune_frac": 0.12})["applied"]["prune_frac"] == 0.12
    assert c.life.brain.prune_frac == 0.12


def test_snapshot_carries_per_part_diagnostics():
    # every module's live metric must be queryable via /api/state.state (I diagnose by API, not by eye)
    c = _controller_with_brain("diag")
    c.life._learn_text("rivers run to the sea and the sea is wide and blue. " * 12, steps=4)
    c.life._emit(c._on_state)
    st = c.snapshot()["state"]
    for k in ("spike_rate", "bg_spike_rate", "hippo_spike_rate", "cerebellum_mse",
              "perplexity_train", "gen_entropy", "replay_total",
              "neurons", "synapses", "synapse_density"):
        assert k in st, f"state missing per-part metric {k}"
    assert st["neurons"] and st["synapses"]                    # actually populated, not None


def test_history_accumulates_module_timeseries():
    c = _controller_with_brain("hist")
    for _ in range(3):
        c.life._learn_text("the mind grows by learning every day. " * 10, steps=3)
        c.life._emit(c._on_state)
    for k in ("bg_spike_rate", "hippo_spike_rate", "cerebellum_mse"):
        assert k in c.history and len(c.history[k]) >= 1           # time-series recorded for /api/history


def test_diagnostic_get_endpoints_over_http():
    import threading, urllib.request, json as J
    from http.server import ThreadingHTTPServer
    c = _controller_with_brain("diaghttp"); D.CTRL.life = c.life; D.CTRL.run_dir = c.run_dir
    # the HTTP handler reads the GLOBAL CTRL, so drive it (not the local c) for logs + history
    c.life._learn_text("a lesson repeated makes memory. " * 8, steps=3); c.life._emit(D.CTRL._on_state)
    D.CTRL._on_log("test-log-line")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), D.H); port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True); th.start()
    B = f"http://127.0.0.1:{port}"
    def get(p): return J.loads(urllib.request.urlopen(B + p, timeout=5).read())
    try:
        lg = get("/api/logs?n=50")
        assert "logs" in lg and lg["capacity"] == D.CTRL.logs.maxlen and any("test-log-line" in x for x in lg["logs"])
        allh = get("/api/history")
        assert "bg_spike_rate" in allh and "hippo_spike_rate" in allh          # every series
        one = get("/api/history?key=bg_spike_rate")
        assert "bg_spike_rate" in one and isinstance(one["bg_spike_rate"], list)
        assert "err" in get("/api/history?key=bogus")                          # unknown key handled
    finally:
        srv.shutdown(); srv.server_close()


def test_kill_flag_skips_checkpoint():
    c = _controller_with_brain("kill")
    c.kill()
    assert getattr(c, "life", None) is None                     # dropped without a graceful save


def test_kill_actually_stops_a_running_worker():
    # REGRESSION: kill() used to null the thread WITHOUT waiting, so the worker (and the sense
    # thread stuck in a teacher call) kept running while the board showed 'stopped'. kill() must
    # now JOIN the worker and confirm it exited.
    import threading, time
    c = _controller_with_brain("killrun")
    c.thread = threading.Thread(target=lambda: c.life.run(on_update=c._on_state), daemon=True)
    c.thread.start()
    time.sleep(2.0)                                             # let it enter the living loop
    assert c.running()                                          # worker genuinely alive
    worker = c.thread
    r = c.kill()
    assert r.get("stopped") is True                            # kill waited and confirmed exit
    assert c.thread is None and c.life is None
    assert not worker.is_alive()                               # the thread really stopped


def test_teacher_call_is_interruptible():
    # the Sonnet subprocess must be tracked so a stop/kill can terminate it (kill_active_calls);
    # calling it with no active calls is a safe no-op.
    from brain import partner
    assert hasattr(partner, "kill_active_calls")
    partner.kill_active_calls()                                 # no-op, no error


def test_controller_boundary_methods():
    c = _controller_with_brain("bound")
    assert c.chat("hello brain")["ok"]                          # chat/inject
    c.tools_add({"name": "t", "cmd": "echo {input}"})
    assert c.tools_toggle({"name": "t", "autonomous": True})["tool"]["autonomous"]
    assert c.tools_remove({"name": "t"})["ok"] and c.tools_list()["tools"] == []
    obs = c.observations()
    assert "observations" in obs and isinstance(obs["observations"], list)


def test_real_http_route_dispatch_and_prefix_order():
    # a REAL server + real HTTP requests — exercises do_GET/do_POST, JSON body, 404, and the
    # prefix-ordering-sensitive routes (/api/tools/add must beat /api/tools; observations vs observe)
    import threading, urllib.request, json as J
    from http.server import ThreadingHTTPServer
    c = _controller_with_brain("http"); D.CTRL.life = c.life; D.CTRL.run_dir = c.run_dir
    srv = ThreadingHTTPServer(("127.0.0.1", 0), D.H); port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True); th.start()
    B = f"http://127.0.0.1:{port}"
    def get(p): return J.loads(urllib.request.urlopen(B + p, timeout=5).read())
    def post(p, d): return J.loads(urllib.request.urlopen(urllib.request.Request(B + p, data=J.dumps(d).encode(), method="POST"), timeout=5).read())
    try:
        assert "/api/state" in "".join(get("/api/help").keys()) or get("/api/help")          # help serves
        assert isinstance(get("/api/tools"), dict)                                            # GET /api/tools
        assert post("/api/tools/add", {"name": "z", "cmd": "echo {input}"})["ok"]             # /api/tools/add ≠ /api/tools
        assert any(t["name"] == "z" for t in get("/api/tools")["tools"])
        assert post("/api/set", {"budget": 0.15})["applied"]["budget"] == 0.15               # POST body parsed
        assert isinstance(get("/api/observations")["observations"], list)                    # observations ≠ observe
        # 404 on unknown route
        try:
            urllib.request.urlopen(B + "/api/nonexistent", timeout=5); assert False
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        srv.shutdown(); srv.server_close()
