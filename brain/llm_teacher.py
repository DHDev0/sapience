"""
llm_teacher.py — the frozen small VLM backbone (Qwen3.5-0.8B, vision+language)
as the init/teacher of the evolving brain.

Qwen3.5-0.8B is a vision-language model; keeping vision is the point — the
"Learning Brain" is about perception + prediction, and the visual cortex is the
canonical predictive-coding system. So the frozen backbone here is the FULL VLM:
images and text go in, fused hidden states come out, and the brain head rides on
those fused representations (it is modality-agnostic — vision enters through the
backbone).

This is the §6 "frozen backbone": activations cross into the brain head, no
gradient crosses back. The teacher supplies three per-position signals:

  mid   hidden state at an intermediate layer  -> the head's INPUT feature
  tgt   pre-lm_head representation             -> the head's distillation TARGET
  prob  next-token distribution                -> for measuring distribution match

Because the output embedding E is tied and the readout is exactly
`final_norm(h) @ Eᵀ`, distilling the 1024-d `tgt` reproduces the full 248320-way
next-token distribution — and replay only needs to store the tiny (mid, tgt)
pairs, never the 248320-d distribution.
"""
from __future__ import annotations
import torch
from .ops import sg


def _install_mistral_shim():
    """The env's mistral_common is version-broken and poisons AutoProcessor import.
    Inject the missing symbol at runtime (non-invasive; no env changes)."""
    try:
        import mistral_common.protocol.instruct.request as _r
        if not hasattr(_r, "ReasoningEffort"):
            _r.ReasoningEffort = type("ReasoningEffort", (), {})
    except Exception:
        pass


class QwenVLTeacher:
    LOCAL_ID = "Qwen/Qwen3.5-0.8B"

    def __init__(self, device, dtype=None, mid_layer=None, model_id=None):
        _install_mistral_shim()
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.device = device
        if dtype is None:
            dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        self.dtype = dtype
        mid = model_id or self.LOCAL_ID
        self.proc = AutoProcessor.from_pretrained(mid)
        self.tok = self.proc.tokenizer
        self.model = AutoModelForImageTextToText.from_pretrained(mid, dtype=dtype).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        tcfg = self.model.config.text_config
        self.hidden = tcfg.hidden_size
        self.n_layers = tcfg.num_hidden_layers
        self.vocab = tcfg.vocab_size
        # default: read a NEAR-TOP fused feature — the head rides almost the whole
        # frozen backbone and learns only a thin, adaptable top transform (strong
        # kick-in; a mid feature would ask the head to replicate too many layers).
        self.mid_layer = max(1, self.n_layers - 4) if mid_layer is None else mid_layer
        # frozen readout = the actual lm_head module. We capture its INPUT via a
        # forward hook and distil to that, so readout(tgt) reproduces the teacher's
        # logits EXACTLY regardless of final-norm placement, bias, or logit scaling.
        self.lm_head = self.model.get_output_embeddings()
        self.E = self.lm_head.weight.detach()
        self._lm_in = None
        self.lm_head.register_forward_pre_hook(self._capture_lm_input)

    def _capture_lm_input(self, module, args):
        self._lm_in = args[0].detach()      # [B,T,H] fed into lm_head

    # ------------------------------------------------------------------ #
    #  build inputs — raw text (wikitext) or full image+chat (VQA)
    # ------------------------------------------------------------------ #
    def _to_device(self, inputs):
        return {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

    def build_text_inputs(self, texts):
        """Raw text, no image, no chat scaffolding — every position is a real
        next-token prediction (best distillation signal for language)."""
        enc = self.proc.tokenizer(texts, return_tensors="pt", padding=True,
                                  truncation=True, max_length=256)
        return self._to_device(enc)

    def build_chat_inputs(self, samples):
        """samples: list of dict(image=PIL|None, messages=[{role,content},...]).
        Full conversation is teacher-forced, so every question+answer position is a
        distillation target. Images enter through the frozen vision encoder."""
        prompts, imgs = [], []
        for s in samples:
            msgs = s["messages"]
            if s.get("image") is not None:
                # ensure the first user turn carries the image placeholder
                m0 = msgs[0]
                if isinstance(m0["content"], str):
                    msgs = [{"role": m0["role"],
                             "content": [{"type": "image"}, {"type": "text", "text": m0["content"]}]}] + msgs[1:]
                imgs.append(s["image"])
            prompts.append(self.proc.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False))
        kw = dict(text=prompts, return_tensors="pt", padding=True)
        if imgs:
            kw["images"] = imgs
        return self._to_device(self.proc(**kw))

    @torch.no_grad()
    def features(self, inputs, temperature=1.0, want_prob=False, want_logits=True):
        """Run the frozen VLM once. Returns per-position, flattened over valid tokens:
           mid  [N,H] head input | tgt [N,H] distill target
        and, when requested, `logits [N,V]` / `prob [N,V]` (skip during training to
        avoid materialising the 248320-d copies). Also returns the valid-token `mask`."""
        self._lm_in = None
        out = self.model(**{k: v for k, v in inputs.items() if torch.is_tensor(v)},
                         output_hidden_states=True)
        hs = out.hidden_states
        mid = hs[self.mid_layer]
        tgt = self._lm_in if self._lm_in is not None else hs[-1]   # exact lm_head input (hook)
        B, T, H = mid.shape
        mask = inputs.get("attention_mask")
        if mask is None:
            mask = torch.ones(B, T, device=self.device, dtype=torch.bool)
        mask = mask.reshape(B * T).bool()
        res = dict(mid=sg(mid.reshape(B * T, H)).float()[mask],
                   tgt=sg(tgt.reshape(B * T, H)).float()[mask], mask=mask)
        if want_logits or want_prob:
            logits = out.logits.reshape(B * T, -1).float()[mask]
            res["logits"] = logits
            if want_prob:
                res["prob"] = torch.softmax(logits / temperature, dim=-1)
        return res

    @torch.no_grad()
    def readout(self, z):
        """Frozen readout of a pre-lm_head representation z -> next-token logits.
        Uses the real lm_head module so it reproduces the teacher exactly."""
        return self.lm_head(z.to(self.E.dtype))

    # ------------------------------------------------------------------ #
    #  generation — the teacher's OUTPUTS (what our model distils)
    # ------------------------------------------------------------------ #
    def build_gen_inputs(self, prompt, image=None):
        content = ([{"type": "image"}] if image is not None else []) + [{"type": "text", "text": prompt}]
        text = self.proc.apply_chat_template([{"role": "user", "content": content}],
                                             tokenize=False, add_generation_prompt=True)
        kw = dict(text=[text], return_tensors="pt", padding=True)
        if image is not None:
            kw["images"] = [image]
        return self._to_device(self.proc(**kw))

    @torch.no_grad()
    def generate(self, prompt, image=None, max_new_tokens=128, temperature=0.8):
        """Frozen teacher generates text (optionally grounded on an image -> vision)."""
        inp = self.build_gen_inputs(prompt, image)
        gen = self.model.generate(**{k: v for k, v in inp.items() if torch.is_tensor(v)},
                                  max_new_tokens=max_new_tokens,
                                  do_sample=temperature > 0, temperature=max(temperature, 0.01))
        new = gen[0][inp["input_ids"].shape[1]:]
        return self.tok.decode(new, skip_special_tokens=True)
