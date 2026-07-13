# Coverage — every request mapped to what shipped

A checklist so nothing from the recent asks is missed. ✅ done · 🔧 fixed · 📄 documented.

| # | Request | Status |
|---|---|---|
| 1 | Per-module architecture R&D; verify Claude conversation learns after distill; batching / resonate in parallel | ✅ membrane readout (2× A/B win), ALIF tested+rejected; Claude loop verified; `resonate(k)` batched + wired |
| 2 | Keep improving speed + performance autonomously | ✅ `run_seq` 7× (bit-identical); hippocampus modern-Hopfield; BG actor-critic fix; cerebellum Golgi; seq tuning |
| 3 | Actually run it to confirm it works | ✅ ran `run_life.py` with **real Claude Sonnet 5** — verified 14 live teacher exchanges |
| 4 | Teacher + main-model speeds; wikitext→sonnet; live log cmd; metrics; TensorBoard; fresh vs continue checkpoint folder | ✅ speeds, ~45 TB metrics, folder logic, TB step continuity |
| 5 | Add every metric worth tracking | ✅ 40+ TB series across 8 groups |
| 6 | Align checkpointing of those values (TB step continues) | ✅ verified step 52→71 across resume |
| 7 | Graceful stop (not pkill); unify into one board with chat | ✅ `dashboard.py` control-plane + STOP control file |
| 8 | Kill (CLI+web); launch; CPU→GPU→multi via checkpoint; parallelism; auto-maximise compute | ✅ full lifecycle, `resolve_compute`, threads/resonate_k |
| 9 | Every board action API-callable; every param live + auto-propagate | ✅ 20 endpoints; `/api/set` live-tunes; verified |
| 10 | Drive by API: teach visual/music/coding; change training cycle/tooling live; redirect feed | ✅ `/api/teach` (text/topic/url/path incl. code), `/api/focus` |
| 11 | Add tools / other AIs (opencode + free model), install, via API + dashboard; any encoder/modality | ✅ `brain/tools.py` registry + folding via senses |
| 12 | Restart + update paper | ✅ each session |
| 13 | MSE/entropy per part + global; replay-buffer state; per-part live params; train/gen perplexity; modality tags in thoughts; replay audio/image; confirm growth | ✅ net-diag, per-module `/api/net`, observation replay, growth confirmed (+64/night) |
| 14 | Bounded logging (evict like replay) | ✅ `max_log_mb`/`max_tb_mb`, live |
| 15 | Set initial neurons; synapse growth per cycle; freeze/modify cycle; checkpoint mem cap live; ALL values live | ✅ initial `hidden`/`layers`; `grow_add`; `freeze_growth/sleep/learning`; caps live |
| 16 | Chart X/Y axes + auto units; per-chart fullscreen; Plotly hover; drag-reorder+resize; remembered | ✅ hand-rolled canvas charts + localStorage layout |
| 17 | Add a metric with every fundamental implementation | ✅ (e.g. `cerebellum_mse`) |
| 18 | Analyse/add missing; add forgotten from paper; reorganise; delete legacy; paper+images in `paper/`; subfolders; requirements; full README; delete dead code | ✅ audit workflow; config persistence + save; `interface/`, `runs/`, `paper/`; `requirements.txt`; README |
| 19 | Unify run files to one; drop result folders (use data); brain_learning in brain; tui+dashboard in interface | ✅ `run_life.py` sole runner; `runs/`; legacy deleted; `interface/` |
| 20 | Fix issues found by R&D understanding + implementation | ✅ audit fixes: cerebellum **wired** (was never instantiated), resume-awake, freeze_learning sensory, kill-no-save, multi-GPU honest note, fresh-launch config, config persistence |
| 21 | Metric with every fundamental implementation (reminder) | ✅ |
| 22 | Read paper for un-integrated; add fundamental features; `test/` folder tests everything + fix; rename brainiac→sapience; README repro-tests; final sweep | ✅ §10 **synaptic pruning** implemented (was faked); paper gaps documented; `test/` suite (41 tests); renamed to **sapience**; README tests section; sweep |

**Fundamental gaps found + fixed this pass:** the §1 cerebellum was only imported, never instantiated (now a wired supervised forward model with a tracked MSE); §10 adolescent pruning was faked (`pruned=0`, now real magnitude-based pruning of the weakest **synapses** with a persistent mask, neurons unchanged).
