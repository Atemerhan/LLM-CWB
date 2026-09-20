"""Local open-weight HuggingFace adapter for the U2 probe experiment.

White-box access to an open model while **read-only reusing the frozen pipeline**:
the v0 prompt (`agents.build_prompt`), the frozen parser (`scoring.parse_response`),
and the frozen dataset (`run.load_dataset`). It adds, per sample:
  - the parsed move (greedy / temperature-0 decode),
  - the per-layer hidden state at the LAST PROMPT TOKEN (pre-generation) — the
    standard probing position for "the model's state just before it answers",
  - a next-token logprob (sanity signal; full move-logprob comes in M-C2).

NO benchmark/scoring/prompt/oracle changes. Products write to new paths
(`results/<slug>_u2*.json`); the frozen v0 baseline is never touched.

Heavy deps (torch, transformers) are imported lazily inside `LocalHFAgent` so the
rest of the repo and its tests never require them.

M-C1 smoke usage:
  python local_model.py --model Qwen/Qwen2.5-3B-Instruct --slug qwen2_5_3b --limit 10
  python local_model.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
      --slug dsr1_qwen1_5b --reasoning --limit 10
"""
from __future__ import annotations

import argparse
import json
import os

from agents import build_prompt
from run import load_dataset
from scoring import parse_response


class LocalHFAgent:
    """Loads an open-weight causal LM and exposes its internals for probing."""

    def __init__(self, model_id: str, reasoning: bool = False,
                 max_new_tokens: int | None = None, device: str = "cpu") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.reasoning = reasoning
        # Reasoning models emit a long <think> trace before the JSON answer, so
        # they need a larger generation budget to reach the parseable object.
        self.max_new_tokens = max_new_tokens or (512 if reasoning else 64)
        self.device = device
        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        # bfloat16 + low_cpu_mem_usage to fit small CI runners; activations are
        # cast to float32 when extracted.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, output_hidden_states=True,
        )
        self.model.to(device).eval()

    def _format(self, prompt_text: str) -> str:
        msgs = [{"role": "user", "content": prompt_text}]
        try:
            return self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
        except Exception:  # tokenizer without a chat template
            return prompt_text

    def analyze(self, sample) -> dict:
        """Return move + per-layer last-token activations + a logprob sanity signal."""
        torch = self._torch
        text = self._format(build_prompt(sample))
        inputs = self.tok(text, return_tensors="pt").to(self.device)
        n_prompt = inputs["input_ids"].shape[1]
        with torch.no_grad():
            # (1) activations at the last PROMPT token (before any generation)
            out = self.model(**inputs, output_hidden_states=True)
            hidden_last = [h[0, -1, :].float().cpu() for h in out.hidden_states]
            logprobs = torch.log_softmax(out.logits[0, -1, :].float(), dim=-1)
            top_lp, _top_id = torch.max(logprobs, dim=-1)
            # (2) greedy (temperature-0) decode for the move text
            gen = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                pad_token_id=(self.tok.pad_token_id or self.tok.eos_token_id),
            )
        gen_text = self.tok.decode(gen[0, n_prompt:], skip_special_tokens=True)
        parsed = parse_response(gen_text)
        return {
            "move": parsed.move,
            "malformed": bool(parsed.malformed),
            "n_layers": len(hidden_last),
            "hidden_dim": int(hidden_last[0].shape[0]),
            "next_token_logprob": float(top_lp),
            "gen_preview": gen_text[:200],
            "hidden_last": hidden_last,  # kept in memory; not serialized in smoke
        }


def _smoke() -> None:
    ap = argparse.ArgumentParser(description="U2 M-C1 smoke for LocalHFAgent.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--reasoning", action="store_true")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--dataset", default="data/eval_dataset.json")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    samples = load_dataset(a.dataset)[: a.limit]
    print(f"loading {a.model} (reasoning={a.reasoning}) ...", flush=True)
    agent = LocalHFAgent(a.model, reasoning=a.reasoning)
    rows = []
    for i, s in enumerate(samples, 1):
        r = agent.analyze(s)
        r.pop("hidden_last")  # smoke records shapes only, not the vectors
        rows.append({"i": i, **r})
        print(f"[{i}/{len(samples)}] move={r['move']} malformed={r['malformed']} "
              f"layers={r['n_layers']} H={r['hidden_dim']} "
              f"lp={r['next_token_logprob']:.3f} | {r['gen_preview']!r}", flush=True)

    summary = {
        "model": a.model, "slug": a.slug, "reasoning": a.reasoning,
        "n": len(rows),
        "malformed": sum(x["malformed"] for x in rows),
        "n_layers": rows[0]["n_layers"] if rows else None,
        "hidden_dim": rows[0]["hidden_dim"] if rows else None,
        "rows": rows,
    }
    out = a.out or f"results/{a.slug}_u2_smoke.json"
    os.makedirs("results", exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSMOKE OK: n={summary['n']} malformed={summary['malformed']} "
          f"layers={summary['n_layers']} H={summary['hidden_dim']} -> {out}", flush=True)


if __name__ == "__main__":
    _smoke()
