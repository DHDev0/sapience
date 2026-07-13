# Living Brain — build tracker

Everything you've asked for, and where it stands. ✅ done+validated · 🔨 in progress · ⏳ queued.

## The vision
A standalone, local-learning brain (from the field guide) that **lives**: born from a
frozen teacher, then always thinking, learning by talking to Claude + visually browsing,
on a self-run wake/sleep rhythm, growing over a lifetime, with a sense of time — watchable
and interactive in a TUI, checkpointable, resource-bounded, CPU/GPU/multi-GPU.

## Status

| # | requirement | status | where |
|---|---|---|---|
| 1 | Standalone byte-brain (cerebellum §1 + hippocampal replay §4/8), learns + generates | ✅ | `brain/byte_brain.py` |
| 2 | Unified sensorimotor code — image/audio/text/time/self → ONE byte stream; output→action | ✅ | `brain/senses.py`, `brain/motor.py` |
| 3 | Internal clock — sense of time from event/wall oscillator banks + accumulators | ✅ | `senses.Clock` |
| 4 | §10 development — grow (child) / prune (adolescent) / stabilize (adult), η(t) envelope, per night | ✅ | `byte_brain.develop`, `cerebellum.grow/prune` |
| 5 | Autonomous wake/sleep — sleep-debt (§7.6), min/max awake windows, it decides; visible | ✅ | `brain/life.py` |
| 6 | Continuous distill: frozen Qwen-0.8 → Claude Sonnet 5 (ongoing) + visual web browsing | ✅ | `brain/life.py`, `partner.py`, `visual_web.py` |
| 7 | **Continuous thinking** — always-on inner monologue, learns from own thoughts + world, non-blocking feedback, idle→autonomous | ✅ | `life.run()` / `think()` / bg sense-thread |
| 8 | **Memory hierarchy** — model params (VRAM/RAM), hot buffer (RAM), whole-life episodes (SSD) | ✅ | `byte_brain.DiskReplay`, tiered replay |
| 9 | **Resource caps as params** — model grows to `max_model_gb` (14 GB max), disk cap, RAM threads | ✅ | `ByteBrainLM(max_model_gb=…)`, growth headroom |
| 10 | **Device switch** CPU↔GPU (auto by availability) + **precision** fp32 (CPU) / bf16 (GPU) | ✅ (built) | `life.pick_device`; GPU run blocked (wedged) |
| 11 | **Checkpoint restart** — reload cycle/age/nights/clock/wake-state/mind | ✅ | `life.save_life/load_life`, `--resume` |
| 12 | opencode-style **TUI** — awake/sleep, ASCII vision, perception, live thought-stream, log, chat | 🔨 wiring to continuous loop | `tui.py` |
| 13 | **Fix the caveat** — deep LEARNED core with real capability | ✅ | `brain/rnn_brain.py` — byte GRU cortex = PC at β→0 (§3.5); **bpb 0.53, coherent English** |
| 14 | **Grow to 14 GB max** (data-gated, honest) | ✅ (cap) / GPU for scale | `ByteRNNBrain.grow/develop`, `max_model_gb` |
| 15 | **Continuous thinking** — always-on inner monologue, learn from own thoughts, non-blocking feedback | ✅ | `life.run/think`, bg sense-thread; **9.3 thoughts/s** |
| 16 | **Memory eviction/compression** — RAM→zstd-19 SSD segments→delete oldest at hard cap | ✅ | `brain/memory.py` (EpisodicMemory) |
| 17 | **Multi-GPU / FSDP2** + bf16 | ✅ (built) | `nn.Module` core; run needs GPU reset |
| 18 | **SHY sleep-downscale** (real §8.4 bug-fix: sleep no longer only potentiates) | ✅ | `life._sleep_tick` |
| 19 | **Completeness** — de-scoped per critique to load-bearing (SHY ✓); §2/§5 optional | 🔨 | `life` |
| 20 | **Paper** — audit ✓ + new §15 chapter documenting all additions | ✅ | `paper/THE_LIVING_BRAIN.md` |

## Honest status
- **Caveat solved.** The recurrent cortex (`rnn_brain.py`) writes coherent English at **0.53 bits/byte**
  standalone; inside the full living loop it **converges stably** — understanding **0.14→0.34 in ~2.5 min, no
  regression across nights**, generation moving toward real sentences. **Sleep drives the learning**: heavy BPTT
  consolidation on the whole-life memory during sleep (as in biology) more than doubled the learning rate while
  waking thought stays fast (~9 thoughts/s). Full fluency needs more wall-time/data (the architecture reaches 0.53).
- **Growth caveat (honest):** widening a GRU mid-life is not identity-preserving and destabilised learning, so
  structural growth is disabled by default — development acts via the η-envelope + memory + pruning; `max_model_gb`
  sets the size the trunk is *instantiated* at. Filling 14 GB needs real data + the GPU (not a trickle of babble).
- **The second GPU is driver-wedged** (`hsa_init → OUT_OF_RESOURCES`). Everything here runs on **CPU/fp32**;
  **bf16, FSDP2, and a large (→14 GB) trunk are built but need the GPU reset**
  (`sudo modprobe -r amdgpu && sudo modprobe amdgpu`).
- **Capability ceiling (not oversold):** a strong *local-context* language model — real words/morphology/local
  fluency — **not** long-range coherence, in-context learning, or reasoning. Next lever = a deeper recurrent trunk.

## Run
- `python tui.py` — watch it live + chat (continuous). `--resume` to continue a saved life.
- `python run_life.py --forever` — headless. `--device cuda --dtype bf16 --max-model-gb 14` when GPU is back.
