"""§17 — the SpikingGlia slow astrocytic activation FIELD (per-neuron metaplastic modulation)."""
import torch
from brain.glia import SpikingGlia


def _rate(hids, val):
    """Per-layer per-neuron rate vector r_l = (Σz)/(B·T), constant `val` across neurons."""
    return [torch.full((h,), float(val)) for h in hids]


def test_default_off_and_inert():
    g = SpikingGlia()
    assert g.on is False
    # OFF ⇒ every output is neutral no-op even after sensing
    g.sense(_rate([8, 8], 0.5))
    assert g.pgain_per_layer() is None
    assert g.global_gain() == 1.0 and g.metab_mult() == 1.0
    assert g.a == []                              # a disabled glia never builds the field


def test_calm_is_neutral():
    """One-sided brake: firing AT/BELOW target (a ≤ 1) leaves every gain untouched (never taxes calm)."""
    g = SpikingGlia(); g.on = True
    hids = [16, 16]
    # target_rate=0.08 ⇒ rate=0.08 → ratio 1.0 (at target); rate=0.02 → ratio 0.25 (below target)
    for _ in range(2000):
        g.sense(_rate(hids, 0.08))
    assert abs(g._mean_a() - 1.0) < 0.02          # converges to baseline 1.0
    assert g.global_gain() > 0.999                # calm ⇒ no region brake
    assert abs(g.metab_mult() - 1.0) < 1e-3       # calm ⇒ no metabolic scarcity
    pg = g.pgain_per_layer()
    assert all(float(p.min()) > 0.99 for p in pg) # calm ⇒ full per-neuron plasticity


def test_overactivity_brakes_and_detects_runaway():
    """Sustained OVER-firing (rate ≫ target) drives a>1 and engages all three one-sided brakes."""
    g = SpikingGlia(); g.on = True; g.tau_a = 50.0; g.k_p = 2.0
    hids = [32, 32]
    for _ in range(3000):
        g.sense(_rate(hids, 0.32))                # 4× target → ratio 4.0
    st = g.state()
    assert st["astro_activation"] > 1.0           # the field DETECTS the runaway
    assert st["astro_overactive_frac"] > 0.0      # localized-runaway coverage > 0
    assert g.global_gain() < 1.0                  # region-wide brake engages
    assert g.metab_mult() > 1.0                   # metabolic scarcity raised
    pg = g.pgain_per_layer()
    assert all(float(p.max()) < 1.0 for p in pg)  # per-neuron plasticity DOWN-gated (never > 1)
    assert all(float(p.min()) > 0.0 for p in pg)  # but strictly positive (stabilize, never zero-out)


def test_localized_runaway_is_per_neuron():
    """The FIELD is per-neuron: only the over-firing neurons get braked; calm neighbours keep full gain —
    exactly the localized damping the global attention/cortisol scalars are blind to."""
    g = SpikingGlia(); g.on = True; g.tau_a = 30.0
    h = 10
    rate = torch.zeros(h); rate[:3] = 0.40        # first 3 neurons run hot, rest at ~0
    for _ in range(2000):
        g.sense([rate])
    pg = g.pgain_per_layer()[0]
    assert float(pg[:3].max()) < 0.8              # hot neurons throttled
    assert float(pg[3:].min()) > 0.99             # calm neurons untouched (per-neuron, not scalar)


def test_sleep_clears_field():
    g = SpikingGlia(); g.on = True; g.tau_a = 20.0; g.rho_clear = 0.2
    for _ in range(2000):
        g.sense(_rate([16], 0.40))
    hot = g._mean_a(); assert hot > 1.0
    for _ in range(50):
        g.sleep_tick()                            # glymphatic clearance
    assert g._mean_a() < hot                      # field relaxes toward baseline
    assert g._mean_a() < 1.0                      # metabolic debt cleared → no residual brake
    assert g.global_gain() > 0.999


def test_device_dtype_propagation():
    """Every field tensor must inherit the module's device+dtype — no hard-coded cpu/float32."""
    for dt in (torch.float32, torch.float64, torch.bfloat16):
        g = SpikingGlia(device=torch.device("cpu"), dtype=dt); g.on = True
        g.sense(_rate([8, 8], 0.16))
        assert all(a.dtype == dt for a in g.a)
        assert all(a.device == torch.device("cpu") for a in g.a)
        pg = g.pgain_per_layer()
        assert all(p.dtype == dt for p in pg)     # the applied-gain tensors also carry the model dtype
        # scalar controllers are plain python floats (replicated, FSDP-safe)
        assert isinstance(g.global_gain(), float) and isinstance(g.metab_mult(), float)


def test_toggle_on_off_live_no_stale():
    g = SpikingGlia(); g.on = True
    for _ in range(500): g.sense(_rate([16], 0.40))
    assert g.pgain_per_layer() is not None
    ap = g.set_params(on="off")                   # string 'off' must DISABLE (coercion)
    assert ap["on"] is False and g.on is False
    assert g.pgain_per_layer() is None            # OFF ⇒ instant no-op, nothing stale applied
    assert g.global_gain() == 1.0 and g.metab_mult() == 1.0
    g.set_params(on="true")                       # and back on, live
    assert g.on is True and g.pgain_per_layer() is not None


def test_width_scalable_and_grow_invariant():
    """O(neurons) field only (no dense O(N²)); survives a mid-life grow() that widens a layer."""
    g = SpikingGlia(); g.on = True
    g.ensure_width([100, 100])
    assert [a.numel() for a in g.a] == [100, 100] # strictly O(H) per layer
    for _ in range(1000): g.sense(_rate([100, 100], 0.32))
    old_mean = float(g.a[0].mean())
    g.ensure_width([160, 100])                    # top layer GREW by 60 neurons
    assert g.a[0].numel() == 160
    assert float(g.a[0][:100].mean()) == old_mean # existing neurons' state preserved
    assert float(g.a[0][100:].min()) == 1.0       # new neurons start at neutral baseline (identity-safe)
    pg = g.pgain_per_layer([160, 100])            # pgain matches the grown widths (rows = post neurons)
    assert [p.numel() for p in pg] == [160, 100]


def test_load_field_growth_guard():
    """Persistence restores a matching field, else re-inits to baseline (grown brain still restores)."""
    g = SpikingGlia(); g.on = True
    saved = [torch.full((100,), 1.3), torch.full((100,), 1.1)]
    g.load_field(saved, [100, 100])               # exact match → restored
    assert abs(float(g.a[0].mean()) - 1.3) < 1e-4
    g2 = SpikingGlia(); g2.on = True
    g2.load_field(saved, [160, 100])              # layer-0 grew → guard falls back to baseline for it
    assert g2.a[0].numel() == 160 and float(g2.a[0].mean()) == 1.0
    assert abs(float(g2.a[1].mean()) - 1.1) < 1e-4  # unchanged layer still restores


def test_state_surfaces_all_metrics():
    g = SpikingGlia(); g.on = True
    g.sense(_rate([16], 0.32))
    st = g.state()
    for k in ("on", "astro_activation", "astro_overactive_frac", "astro_pgain",
              "astro_metab_mult", "glia_global_gain"):
        assert k in st
