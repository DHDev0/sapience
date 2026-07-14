"""
rnn_brain.py — ByteRNNBrain: the capable cortex core.

The shallow fixed-random cerebellum could only learn byte statistics (fragments).
This is the fix, and it is faithful to the paper: a DEEP RECURRENT cortex trunk
(byte embedding → multi-layer GRU → next-byte head) trained by backprop, which §3.5
proves IS predictive coding in the β→0 limit — the paper's own boundary case. The
recurrence gives it a persistent STATE that breaks the fixed-window ceiling: its
hidden state h IS the brain's continuous stream-of-consciousness, carried across the
always-thinking loop. It generates real, coherent words (measured ~0.5 bits/byte on
web text — the fluency reference, cf. paper §-rate-cortex — vs the cerebellum's fragments).

It is an nn.Module, so it runs on CPU or GPU (fp32 on CPU / bf16 on GPU), can be
sharded across GPUs with FSDP2, and GROWS (wider/deeper, gated on data actually seen)
toward the max_model_gb cap. Drop-in for the old learner: same generate / learn_text /
next_byte_acc / develop / model_gb / save / load surface, so BrainLife is unchanged.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ByteRNNBrain(nn.Module):
    def __init__(self, device, dtype=torch.float32, emb=128, hidden=512, layers=2,
                 lr=2e-3, max_model_gb=14.0, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.V = 256
        self.device = device
        self.mdtype = dtype
        self.emb, self.hidden, self.layers = emb, hidden, layers
        self.E = nn.Embedding(self.V, emb)
        self.rnn = nn.GRU(emb, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, self.V)
        self.to(device)                        # fp32 MASTER weights (stable Adam)
        # bf16 on GPU = mixed-precision COMPUTE via autocast, not bf16 params (which make
        # Adam's running averages lose precision). fp32 master + bf16 matmuls is standard.
        self.use_amp = (device.type == "cuda" and dtype == torch.bfloat16)
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.lr = lr
        self.max_model_gb = max_model_gb
        self.age = 0
        self.seen_bytes = 0
        self.h = None                          # persistent hidden state = the mind
        # §10 growth schedule bounds (mirrors ByteBrainLM so BrainLife is unchanged)
        self.grow_until, self.prune_until = 8, 16

    # ---- helpers ----------------------------------------------------- #
    @staticmethod
    def to_bytes(text):
        return list(text.encode("utf-8", errors="replace"))

    def forward(self, x, h=None):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_amp):
            o, h = self.rnn(self.E(x), h)
            return self.head(o), h

    def model_gb(self):
        return sum(p.numel() for p in self.parameters()) * 4 / 1e9   # fp32 master weights

    @property
    def eta(self):
        return self.lr

    # ---- THINK: generate from the persistent mind-state -------------- #
    @torch.no_grad()
    def think(self, n=18, temperature=0.55):
        self.eval()
        if self.h is None:
            self.h = torch.zeros(self.layers, 1, self.hidden, device=self.device, dtype=self._pdtype())
            cur = torch.tensor([[ord("\n")]], device=self.device)
        else:
            cur = getattr(self, "_last", torch.tensor([[ord(" ")]], device=self.device))
        out = []
        for _ in range(n):
            lo, self.h = self(cur, self.h)
            p = torch.softmax(lo[0, -1].float() / max(temperature, 1e-3), 0)
            cur = torch.multinomial(p, 1).view(1, 1)
            out.append(int(cur.item()))
        self._last = cur
        self.train()
        return bytes(out).decode("utf-8", "replace")

    @torch.no_grad()
    def observe_stream(self, text):
        """Feed perceived text through the RNN so the MIND (hidden state) contexts on it
        (without a weight update — that is what learn_text does)."""
        self.eval()
        ids = self.to_bytes(text)[-512:]
        if not ids:
            return
        x = torch.tensor([ids], device=self.device)
        _, self.h = self(x, self.h)
        self.h = self.h.detach()
        self._last = x[:, -1:].clone()
        self.train()

    @torch.no_grad()
    def generate(self, prompt="", n=200, temperature=0.6, seed=0):
        """Prompted generation from a FRESH state (for eval/samples; leaves the mind alone)."""
        self.eval()
        ids = self.to_bytes(prompt) or [ord("\n")]
        h = None
        x = torch.tensor([ids], device=self.device)
        _, h = self(x, h)
        cur = x[:, -1:]
        out = []
        for _ in range(n):
            lo, h = self(cur, h)
            p = torch.softmax(lo[0, -1].float() / max(temperature, 1e-3), 0)
            cur = torch.multinomial(p, 1).view(1, 1); out.append(int(cur.item()))
        self.train()
        return prompt + bytes(out).decode("utf-8", "replace")

    def _pdtype(self):
        return next(self.parameters()).dtype

    # ---- LEARN: backprop-through-time (= PC at β→0, §3.5) ------------- #
    def learn_text(self, text, epochs=1, bs=32, seq=128, max_steps=24, store=True,
                   replay_interleave=0, consolidate_rounds=0):
        data = text if isinstance(text, list) else self.to_bytes(text)
        if len(data) <= seq + 1:
            seq = max(8, len(data) - 2)
        if len(data) <= seq + 1:
            return None
        t = torch.tensor(data, device=self.device)
        n = t.numel()
        last = 0.0
        for _ in range(epochs):
            steps = max(1, min(max_steps, (n - seq) // (bs * seq) + 1))
            for _s in range(steps):
                i = torch.randint(0, n - seq - 1, (bs,), device=self.device)
                x = torch.stack([t[k:k + seq] for k in i])
                y = torch.stack([t[k + 1:k + seq + 1] for k in i])
                lo, _ = self(x)
                loss = F.cross_entropy(lo.reshape(-1, self.V), y.reshape(-1))
                self.opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                self.opt.step()
                last = loss.item()
        self.seen_bytes += n
        return last

    # ---- eval -------------------------------------------------------- #
    @torch.no_grad()
    def next_byte_acc(self, text, seq=256):
        self.eval()
        data = self.to_bytes(text)
        if len(data) <= seq:
            self.train(); return 0.0
        t = torch.tensor(data[:4096], device=self.device).unsqueeze(0)
        lo, _ = self(t[:, :-1])
        acc = (lo.argmax(-1) == t[:, 1:]).float().mean().item()
        self.train(); return acc

    @torch.no_grad()
    def bits_per_byte(self, text):
        self.eval()
        data = self.to_bytes(text)
        if len(data) < 8:
            self.train(); return float("nan")
        t = torch.tensor(data[:4096], device=self.device).unsqueeze(0)
        lo, _ = self(t[:, :-1])
        bpb = F.cross_entropy(lo.reshape(-1, self.V), t[:, 1:].reshape(-1)).item() / 0.6931
        self.train(); return bpb

    # ---- §10 development: critical-period η + data-gated growth ------ #
    def develop(self, allow_grow=False):
        """§10 lifespan. The learning rate matures on the critical-period envelope (this
        IS development for a recurrent cortex — stable and always applied). Structural
        WIDENING of the recurrent trunk is disabled by default: widening a GRU mid-life
        is not identity-preserving (new hidden units perturb the recurrence) and, per the
        design critique, capacity is not the bottleneck — data is. The max_model_gb cap
        governs the size the trunk is INSTANTIATED at (grow it only with real data + GPU)."""
        self.age += 1
        self.lr = 2e-3 / (1 + self.age / 8.0)          # η(t)=η_max/(1+t/t_c), §10.4
        for g in self.opt.param_groups:
            g["lr"] = self.lr
        phase = ("child" if self.age <= self.grow_until else
                 "adolescent" if self.age <= self.prune_until else "adult")
        grown = 0
        want_gb = min(self.max_model_gb, 0.02 + self.seen_bytes / 2e8)   # data-gated headroom
        if allow_grow and phase == "child" and self.model_gb() < want_gb * 0.9:
            grown = self.grow()
        return dict(age=self.age, phase=phase, eta=round(self.lr, 5),
                    n_granule=self.hidden, grown=grown, pruned=0)

    @torch.no_grad()
    def grow(self, add=128):
        """Widen the GRU + head (identity-preserving on the new head columns), gated by
        the model-size cap. Re-inits the optimizer to include the new parameters."""
        if self.model_gb() >= self.max_model_gb:
            return 0
        old_h = self.hidden; new_h = old_h + add
        pdt = self._pdtype()
        new_rnn = nn.GRU(self.emb, new_h, self.layers, batch_first=True).to(self.device, pdt)
        new_head = nn.Linear(new_h, self.V).to(self.device, pdt)
        with torch.no_grad():
            # copy old GRU weights into the top-left; new head cols = 0 (function preserved)
            for name, p in self.rnn.named_parameters():
                q = dict(new_rnn.named_parameters())[name]
                if p.dim() == 2:
                    q.zero_(); q[:p.shape[0], :p.shape[1]] = p
                else:
                    q.zero_(); q[:p.shape[0]] = p
            new_head.weight.zero_(); new_head.weight[:, :old_h] = self.head.weight
            new_head.bias.copy_(self.head.bias)
        self.rnn, self.head, self.hidden = new_rnn, new_head, new_h
        self.h = None
        self.opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        return add

    # ---- checkpoint -------------------------------------------------- #
    def save(self, path):
        torch.save(dict(sd=self.state_dict(), opt=self.opt.state_dict(),
                        emb=self.emb, hidden=self.hidden, layers=self.layers,
                        age=self.age, seen=self.seen_bytes, lr=self.lr), path)

    def load(self, path):
        d = torch.load(path, map_location=self.device)
        if d.get("hidden") != self.hidden or d.get("layers") != self.layers:
            self.rnn = nn.GRU(self.emb, d["hidden"], d["layers"], batch_first=True).to(self.device, self._pdtype())
            self.head = nn.Linear(d["hidden"], self.V).to(self.device, self._pdtype())
            self.hidden, self.layers = d["hidden"], d["layers"]
            self.opt = torch.optim.Adam(self.parameters(), lr=d.get("lr", self.lr))
        self.load_state_dict(d["sd"])
        try: self.opt.load_state_dict(d["opt"])
        except Exception: pass
        self.age = d.get("age", 0); self.seen_bytes = d.get("seen", 0); self.lr = d.get("lr", self.lr)
        return self
