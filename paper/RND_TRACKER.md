# Brainiac R&D campaign tracker (autonomous, 2026-07-13)

Order: keep doing R&D + fixing until nothing left to improve in speed & performance. CPU only.
Every change A/B-tested on wikitext-2; keep only measured wins. Serialize heavy runs.

## DONE (verified)
- [x] Claude conversation verified end-to-end + `_usable_lesson` filter hardened (life.py).
- [x] **Membrane readout** = default cortex win (bpb 4.48→3.39, acc 0.173→0.344). `readout="mem"`.
- [x] ALIF built (`ALIFCell`) but REJECTED as default (loses seq48/128). Off-by-default lever.
- [x] `resonate()` batched parallel reasoning + wired into `life._resonate`. 6 streams = 1-stream cost.

## QUEUE
- [x] 1. Cortex readout sweep: `mem` CONFIRMED best (mix/spike worse, memtanh tied). Depth=3 hurts. seq↑ helps (48→96: bpb 3.34→3.23). Default seq 48→64.
- [x] 2. Hippocampus REBUILT as spiking modern-Hopfield: recall fidelity 1.0→0.96 for K=10→400 (was DG-space-only, unusable); grow() added (§10 violation fixed), recall fidelity 0.996 after grow.

### MAJOR FINDING: the 4 non-cortex modules are ORPHANED — life.py only drives the cortex.
Integration pass needed (after per-module fixes): wire neuromod→cortex LR/temp, BG→curiosity topic pick, hippocampus→CLS sleep replay/novelty. §6 gradient cut = activation/reward coupling only. Must not destabilize the working cortex (A/B before/after).
- [x] 3. Basal ganglia frozen-actor bug FIXED (policy gradient + γV′ TD): old stays chance 0.32; new learns 0.40→0.98 on contextual bandit.
- [ ] 4. Neuromod → cortex plasticity/temperature wiring
- [x] 5. Cerebellum Golgi feedback inhibition: more scale/growth-robust sparsity (4× scale: old→0.42, golgi→0.36; stable across growth, no recalibration) at equal MSE (0.186 vs 0.182). Faithful + robust, accuracy-neutral. KEEP.
- [ ] 6. Two-compartment apical (§3.7) wiring
- [x] 7. SPEED: `run_seq` (layer-outer, vectorized Win+head) — BIT-IDENTICAL, ~7× faster (0.24s/call vs ~1.6-3.4s). Recurrence Wrec stays sequential (inherent).
- [ ] 8. K-stream consensus decode
- [ ] 9. Update paper + README

- [x] INTEGRATION: 5 modules wired into life (neuromod→wake/sleep tone, BG→curiosity topic pick w/ FREE learning-progress reward, hippo→byte-hist episode index + novelty-gated replay, hippo grows w/ cortex). Zero-overhead (1.00× throughput). Cortex metric within run-to-run noise on homogeneous MOCK feed; real benefit expected on heterogeneous real input. Primary value = faithfulness (real 5-system operation).
- [x] BUG FIXED: SpikingBrain.load() assumed uniform layer width — broke resume after ANY growth (grow widens only top layer). Now infers exact per-layer widths from saved shapes. Verified byte-identical bpb after grow+resume.
- [~] Two-compartment apical (§3.7) + K-stream consensus: DOCUMENTED as deliberate decisions, not built. Two-comp true top-down needs the slow time-outer path (sacrifices the 7× speedup); membrane readout already delivers the graded-potential benefit; TwoCompartmentLIF substrate stays for the §3.7 exposition. K-stream consensus for a SINGLE model = the self-likelihood pick already in life._resonate. Both = available/documented levers, off by default (like ALIF).

- [x] 9. Paper THE_LIVING_BRAIN.md (§15.1 module table, §15.2 substrate+readout+ALIF+speed, §15.4 numbers, §15.9 resonate, NEW §15.14 five-systems-coupled, §15.15 ledger) + README updated. Honest: A/B losers (ALIF, mix, depth) documented, not hidden.
- [x] EXTRA: sleep replay now consolidates on longer context (seq=96) — waking thought stays fast; sleep is the deep phase w/ the 7× headroom.

## STATUS: main levers exhausted. Remaining = documented FUTURE WORK (need architectural tradeoffs):
- Two-compartment §3.7 top-down apical → needs slow time-outer schedule (forfeits 7× speedup). Substrate kept for exposition.
- ALIF adaptive threshold → only pays off at long temporal context (loses at seq≤128 byte). Kept off-by-default.
- Deeper temporal spiking trunk / multi-layer readout → the real spiking-vs-rate gap (3.1 vs 0.5 bpb). Next chapter.
- Real-Claude heterogeneous run to measure curiosity/novelty benefit (mock can't; costs API budget + wall-time).

## LIVE END-TO-END RUN (real run_life.py, real Claude Sonnet 5) — VERIFIED 2026-07-13
- Ran the ACTUAL entry point `run_life.py --no-teacher --no-visual`. It births, lives (~10k thoughts), thinks in word-like fragments, autonomously sleeps/wakes, grows 512→576.
- **REAL Claude conversation works**: 14 live teacher exchanges in results_life/life.log, topics chosen by §2 curiosity ("how languages change", "the moon", "the history of writing", "the ocean") — learning real Sonnet prose. resonate× fires live.
- NOTE: life.log() writes to results_life/life.log (file), NOT stdout — only the TUI mirrors to screen. (Was briefly confused by empty stdout.)
- POST-RUN FIXES:
  - Birth was 60 epochs / ~220s (target_bpb=1.6 unreachable for spiking). Added PLATEAU early-stop → 43 ep / 148s (adaptive, stops when learning flattens).
  - web_topic scraped wikipedia NAV CHROME ("Jump to content Main menu...") → brain learned UI boilerplate + probe metric polluted (born bpb looked 4.9 on chrome vs ~3.3 on clean text). Fixed: web_topic now uses the extracts API → clean article prose. Verified chrome gone.

## OBSERVABILITY + LIFECYCLE (2026-07-13, user request)
- TUI STATE panel now shows teacher gen speed (Qwen@birth / Sonnet@life, chars/s) + main-model think & learn speed (bytes/s) + bits/byte + novelty + dopamine. (#top height 18, #state width 40.)
- TENSORBOARD: 40 metrics across 8 groups logged over time (step = seconds of life): eval/ (understanding, bpb probe+heldout, time-sense, word-likeness, gen-distinct-char), model/ (neurons, size_gb, capacity_frac, age, phase, eta, nights), life/ (thoughts, thoughts/s, awake, awake_frac, sleep_debt, perceptions, clock_events), speed/ (teacher/think/learn), memory/ (disk, hot, lived_chars, segments, compression, hippo episodes+DG), neuromod/ (novelty, DA/ACh/NE/5HT), curiosity/ (policy entropy, top pref, critic value), health/ (weight mean+max abs). Expensive ones (gen sample, weight sweep) on a 30s slow tick; rest every 5s.
- CHECKPOINT ALIGNMENT: save/load persists lived_seconds (TB step continues, not reset), perc_count, awake_frac, tps, teacher_cps/name, think/learn bps + (already) bg policy, hippo store, cycle/nights/age. Verified resume continues TB step 52→71.
- FOLDER/RESUME: no --checkpoint → fresh results_life/run_<timestamp>/ (previous runs untouched). --checkpoint DIR → continue in place. Startup prints run folder + `tail -f life.log` + `tensorboard --logdir tb` + continue cmd. Both run_life.py and tui.py.
- BIRTH SPEEDUP: plateau early-stop (was 60ep/220s → ~43ep/148s). web_topic uses extracts API (clean prose, no nav chrome).

## UNIFIED CONTROL PLANE — dashboard.py (2026-07-13, user request) — DONE + LIVE
One web board replaces tail+tensorboard+tui: live charts (12 key metrics, hand-rolled canvas, no deps), stream-of-thought, life log, vision, PERCEPTION, live CHAT box, and controls: Launch/Relaunch (config form: device auto/cpu/cuda/multi, threads, resonate_k, checkpoint dropdown, teacher, visual, budget, awake windows), Stop (graceful+checkpoint), Kill (force). In-process Controller manages ONE BrainLife; can stop+relaunch (e.g. resume same checkpoint on a different device) without dropping the board.
- Graceful stop = STOP control file (BrainLife.run checks it, checkpoints, exits). No more pkill. CLI: `dashboard.py --stop [dir]` or `run_life.py --stop`. Force: `dashboard.py --kill` (SIGKILL via results_life/dashboard.pid) + web Kill button.
- resolve_compute(device,threads): 'auto' maximizes — all GPUs (FSDP note) / 1 GPU bf16 / all CPU cores. BrainLife gained threads + resonate_k params. Checkpoint is device-agnostic (map_location) → start CPU, stop, resume GPU works.
- RESUME SKIPS BIRTH (self.resumed flag → birth() early-returns): resume reaches awake in ~6s not 150s, continues cycle/age/neurons/bpb. Verified.
- Folder rule unchanged: no --checkpoint = fresh run_<ts>/ (keeps old); --checkpoint = continue.
- VERIFIED end-to-end: launch→chat(ok)→history(17pts)→graceful stop(checkpoint 12MB)→CLI kill→resume-with-new-config(device/resonate_k, skips birth, cycle continues). LIVE now: http://localhost:8181 (resumed run_20260713_160759, awake, learning from Sonnet+web, 32 CPU threads). pid file registered.
- HOW TO RUN: `python dashboard.py` (fresh) or `--checkpoint <run>` (resume) → open the printed URL. GPU/multi-GPU device paths built (this box CPU-only, GPU wedged) — validated on CPU, consistent w/ honest ledger.

## DRIVE-BY-API + UI (2026-07-13 pm) — DONE, live on :8181
- GROWTH CONFIRMED: +64 neurons/night in child phase (512→576→640, log shows "(+64)"). Data-gated, 14GB cap, then prune (adolescent)/stabilize (adult).
- UI: fullscreen toggle on EVERY card (⛶/✕, addFsButtons injects btn per .card). Thought feed = timestamped, line-coalesced entries (life.thought_log; ↻ marks parallel reflections). status/config card shows all live hp incl focus/feed_mode/teach_queue.
- LIVE-TUNE: POST /api/set changes soft params w/o restart (budget,min/max_awake,debt_threshold,perceive_gap,think_chunk,resonate_k,threads,learn_steps,max_model_gb,visual,teacher) — read live each loop iter. Structural (device/core/checkpoint) via /api/start relaunch. Board: "✓ apply live" vs "▶ relaunch".
- DRIVE-BY-API (life.teach/focus): POST /api/teach {text|topic|url|path,label} → resolves to bytes, chunks, PRIORITY-learned (verified: taught it its own senses.py, learned[own-code] byte-by-byte). POST /api/focus {topics,urls,mode:random|topics|urls|mixed,label} → redirects the learning feed live (topics/browse are instance lists now; feed_mode steers sense_worker). Verified focus→coding.
- ALL board actions = API endpoints (10); GET /api/help lists them + curl examples. SSH: banner auto-prints `ssh -L <port>:localhost:<port> user@host` (from SSH_CONNECTION).
- KEY FILES: dashboard.py (Controller in-process: start/stop/kill/set/teach/focus + HTTP + HTML board), brain/life.py (topics/browse/feed_mode/learn_steps/_teach_q, focus(), teach(), _read_path(), thought_log, request_stop/resolve_compute, resume-skips-birth).
- View from laptop: `ssh -L 8181:localhost:8181 dander@192.168.1.19` then http://localhost:8181. Stop: dashboard.py --stop / board ⏸. It costs real Sonnet $ until stopped.

## TOOL/PLUGIN REGISTRY + PAPER (2026-07-13 pm) — DONE, live on :8181
- brain/tools.py ToolRegistry: register any CLI tool / other AI (opencode+free model, local LLM, TTS, image gen) as {name, cmd with {input}, kind:text|audio|image|bytes, install?, shell?, autonomous?}. Persists to <run>/tools.json (survives resume). run() subprocesses w/ timeout.
- BrainLife.use_tool + _encode_tool_output: folds tool OUTPUT into the ONE byte stream via senses.py (text→language BPTT; audio→cochlea samples; image→retina; bytes→self frame). learn_text accepts raw byte-level LISTS → any modality learned uniformly (the unified-code thesis, now runtime-extensible). Autonomous tools: sense_worker converses w/ them every 3rd step (generalised teacher). _teach_q generalized to (src, str|list).
- Dashboard: /api/tools (GET) + /api/tools/add|remove|toggle|install|run; 🔧 tools card (list + run/on/auto/install/✕ + register form); snapshot.tools; API_HELP + curl examples. 16 endpoints total, ALL board actions API-callable.
- VERIFIED live: registered echo-ai via API, ran it (returned output), brain LEARNED it + conversed autonomously (sent own babble, learned response). Removed after test (clean list).
- PAPER updated: NEW §15.15 "The cockpit — driving and extending the brain by API" (dashboard, full API, lifecycle/device control, live-tune, teach/focus, tools-as-unified-code); ledger→§15.16 w/ growth+cockpit. README front-door rewritten to lead with dashboard.py.
- GROWTH re-confirmed: run_20260713_160759 now 1024 neurons / 9 nights (was 512; +64/child-night).
- Restarted per user: live on http://localhost:8181 (resumed, tools code). View: ssh -L 8181:localhost:8181 dander@192.168.1.19.

## CHARTS UPGRADE (2026-07-13 pm) — DONE, live on :8181
- Each metric = a draggable/resizable TILE (native resize:both corner + HTML5 drag by header to reorder). Layout (order + per-tile size) persists to browser localStorage 'brainCharts', restored on reload.
- Real X/Y axes per chart: Y = auto-scaled value (fmtNum k/M/G + per-metric unit from CHART_KEYS 4th field), X = time-since-birth (fmtTime s/m/h). 3-4 gridline ticks each.
- Per-CHART fullscreen: each tile has its own ⛶/✕ (independent of the card-level one).
- Plotly-style HOVER: mousemove on a canvas → nearest point → dashed crosshair + dot + tooltip box "time · value+unit". tick() skips redrawing a hovered tile so the crosshair persists.
- CHART_KEYS now 4-tuples (key,label,lower,unit); unpack sites use `*_`. Served JS verified (buildCharts/hoverChart/fmtTime/saveLayout/resize:both).

## DEEP METRICS + FULL LIVE CONTROL + OBSERVATION REPLAY (2026-07-13 late) — DONE, live :8181
- NET DIAGNOSTICS (SpikingBrain.train_perplexity/generate_diag/spike_rate/weight_stats; life._net_diag cached 6s): train+gen perplexity, gen entropy(bits), spike firing rate (dead-neuron tripwire), per-layer weight mean/std, hippo recall fidelity. In state + TB (net/*) + 6 new charts. Board "🔬 net diagnostics" panel.
- PER-MODULE LIVE TUNING: POST /api/net {target:cortex|hippocampus|bg|neuromod,...} → life.set_net (cortex lr/read_alpha/seq/think_temp, hippo beta/sparsity/capacity, bg alpha_v/alpha_pi, neuromod da/ach/ne/ht). Board "🎚 tune a module" + netparams display.
- OBSERVATION REPLAY: non-text sensory frames recorded (life.last_observations); GET /api/observe?i=N decodes audio→wav / image→png (inverse of senses encoders — low-fi = exactly what it sensed). Board "👁👂 observations" plays <audio>/<img>. Feed/perception tagged w/ modality (src "tool:x (audio)").
- BOUNDED ARTIFACTS (live caps, evict earliest): max_log_mb (life.log tail), max_tb_mb (oldest tb events), hard_disk_gb (replay segments), checkpoint = single overwritten .pt bounded by max_model_gb. _bound_logs every 60s keeps tail under 0.9×cap.
- GROWTH/CYCLE CONTROL: settable initial neurons+layers at launch (start passes hidden/layers/emb); grow_add (neurons+synapses/night), grow_until/prune_until, freeze_growth/freeze_sleep(stay awake)/freeze_learning(observe-only) — all live. Clarified neurons=units (set initial/rate/freeze), synapses grow quadratically (=§10). Verified: initial 200, grow_add 128→+128, freezes work.
- EVERYTHING LIVE: /api/set broadened to ~20 params; only initial-neurons/layers/device/core need /api/start relaunch. 19 endpoints total. hp card shows all.
- CHARTS: fixed snapshot to expose net/netparams/observations (were in latest, not returned). 18 chart metrics, history 81+ pts each.
- PAPER §15.15 extended (everything-live-tunable + per-module + deep diagnostics + bounded-everything + observation replay); verified live values (train ppl 11.58, spike 0.032, hippo fidelity 0.994).

## CONSOLIDATION + TESTS + RENAME (2026-07-13 final)
- RENAMED brainiac→**sapience**. Layout: run_life.py + interface/{dashboard,tui}.py + brain/(16 mods) + paper/ + runs/ + test/. Legacy 12 modules + 4 entry points deleted.
- FUNDAMENTAL FIXES (were faked/missing): §1 cerebellum now instantiated as supervised forward model (cerebellum_mse metric); §10 pruning now real SYNAPTIC (mask-persistent, neurons kept) — was pruned=0.
- AUDIT FIXES: config persists across resume (all knobs + module params + focus); resume-wakes (no spurious develop); freeze_learning on sensory+sleep; kill skips ckpt; multi-GPU honest note; think_temp used; fresh-launch config applied; /api/save.
- TESTS: test/ = 54 tests (run_tests.py runner, no pytest needed). 41 unit/integration + 13 REGRESSION (test_regressions.py, one per fixed bug so behavior can't revert). All pass ~20s. `python test/run_tests.py`.
- Dashboard VERIFIED working (launch→alive→serves 20 endpoints); sandbox can't keep detached servers alive but user's `python interface/dashboard.py` works. No leaked processes.
- Paper §15 fully updated (synaptic pruning, birth policy, lesson filter, chat cmds, tool lifecycle, honest multi-GPU, interface/ paths). COVERAGE.md maps all requests.

## FINAL SWEEP (2026-07-13, workflow wz1rlt3oe) — honesty audit paper↔code
- HEADLINE over-claim fixed: NOT "five spiking growable" — only 3 spike+grow (cortex/cerebellum/hippocampus); BG=rate actor-critic, neuromod=scalar tone. Paper reworded (abstract, §15.1 table, §15.4, §15.16).
- IMPLEMENTED (make it real, not paper-patch): (a) §5 neuromod ACh tone now GATES cortex plasticity — scales lr in _learn_text (was display-only, eligibility was dead code); (b) §4 hippocampus write now NOVELTY-GATED (novelty_gate=0.15, live-tunable) — was storing every episode. Both tested.
- PAPER honesty edits: BG "rate actor-critic" (not LIF medium-spiny); hippocampus "soft-WTA/softmax attractor" (not spiking); SHY = open-loop constant decay (not closed-loop balance restore); REM never entered (roadmap); §6 router absent (cortex alone drives generation, cerebellum monitored not fed back); cited brain/partner.py (Sonnet) + brain/llm_teacher.py (frozen Qwen birth teacher); counts 40→54 tests, 40→46 metrics; fixed hippocampus module-header docstring (self-contradicting); API_HELP /api/net +cerebellum, /api/start +hidden/layers/caps.
- TESTS: 58 total (was 54) — added neuromod-gate, novelty-gate, REAL-HTTP route-dispatch (prefix ordering), Controller boundary. All pass ~30s.

## MADE MODULES FAITHFUL TO THE PAPER (2026-07-13) — 4 of 5 now spiking + growable
- Why we'd diverged: BG built as rate actor-critic + hippo recall as softmax attractor (simpler/higher-capacity) — diverged from field-guide spiking medium-spiny (§2) + spiking CA3 (§4).
- §2 BG REBUILT: LIF MEDIUM-SPINY neurons (msn() spike-rate code) → critic+actor; dopamine RPE trains both; grow() adds MSNs (identity-preserving); grows each night in life. VERIFIED: learns policy 0.33→1.0, spike_rate 0.06, grows 64→80.
- §4 hippocampus recall now SPIKING: LIF memory neurons + lateral WTA inhibition emit real Heaviside spikes (was softmax attractor). VERIFIED fidelity 0.99/0.96/0.92/0.85 @ K=30/100/200/400 (small honest capacity trade vs softmax).
- So NOW: cortex/cerebellum/hippocampus/BG = spiking + growable (4); neuromod = scalar modulatory glue (honest exception, gates cortex plasticity). Paper reworded "four spiking growable + glue" (abstract/§15.1 table/§15.4/§15.14/§15.16/docstring).
- METRICS readapted: bg_spike_rate + hippo_spike_rate added to net_diag + TB (net/) + board netdiag panel. Save/load handles new BG shape (guard for old ckpts).
- LIVE RUN verified: 5-system loop runs, BG spikes+grows, hippo recall spikes @0.99 fid, cerebellum/cortex fire, ACh gates plasticity. 58 tests pass.

## MEASUREMENTS LOG
| change | metric | before | after | keep? |
|---|---|---|---|---|
| membrane readout | bpb / acc | 4.484 / 0.173 | 3.393 / 0.344 | YES |
| alif (seq48) | bpb | 3.393 (lif/mem) | 3.971 | NO |
| alif mild (seq128) | bpb | 3.371 (lif/mem) | 3.917 | NO |
| readout mix/memtanh | bpb | 3.340 (mem) | 3.69/3.36 | NO (mem wins) |
| seq 48→96 | bpb | 3.340 | 3.232 | seq→64 default, 96 in sleep |
| run_seq vectorize | s/call | ~1.6-3.4 | 0.24 | YES (bit-identical) |
| hippocampus modern-Hopfield | recall@K=400 | ~unusable | 0.962 | YES |
| basal-ganglia actor-critic | bandit reward | 0.32 (frozen) | 0.98 | YES |
| cerebellum Golgi | MSE / scale-robust | 0.182 / drifts | 0.186 / stable | YES |
| module integration | throughput | 1.0× | 1.00× (zero-cost) | YES (faithful) |
