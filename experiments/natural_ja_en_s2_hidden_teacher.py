#!/usr/bin/env python3
import argparse
import copy
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

VOCAB = 256
D_MODEL = 64
N_LAYERS = 2
N_HEADS = 4
FF = 256
CTX = 128
BS = 16
LR = 1e-3
GRAD_CLIP = 1.0
BASE_STEPS = 400
TEACHER_STEPS = 400
PHASE1_STEPS = 200
PHASE2_STEPS = 600
EVAL_EVERY = 10
HIDDEN_WEIGHT = 1.0
EVAL_BS = 32
EVAL_BATCHES = 8
CONFIRMATORY_SEEDS = list(range(31000, 31010))
PILOT_SEED = 30999
EXPECTED_CORPUS_HASHES = {
    "en_train": "74f59db34b5b8334cf463cf0bcf5f355e54cc7de98cc4eb35ef379ecb42e8258",
    "en_eval": "15221408b2c4c2fffee8a5f16feb651f2de5947e85906867f119fdf7210797da",
    "ja_train": "5a0d86cebeb5e9f71c86ea1bcf619c8a42bb21d04ed2d37ce35f5aee58bffd58",
    "ja_eval": "5a44582a08d6e3c48c610d663b22642c8b8612dc19f25a326b2ce906a504e7e4",
    "ru_train": "12a53e456480b56977c548021231c9adf9b4cee97902f6d1acb67e6cbb0d46ed",
    "ru_eval": "52a1cd50a20645577e988f6023dcf1853e856f5babd180a0e1d620029c689800",
}


@dataclass(frozen=True)
class Schedule:
    base_steps: int = BASE_STEPS
    teacher_steps: int = TEACHER_STEPS
    phase1_steps: int = PHASE1_STEPS
    phase2_steps: int = PHASE2_STEPS
    eval_every: int = EVAL_EVERY


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, D_MODEL)
        self.pos = nn.Embedding(CTX, D_MODEL)
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=FF,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=N_LAYERS)
        self.ln = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB, bias=False)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(CTX, CTX, dtype=torch.bool), diagonal=1),
            persistent=False,
        )

    def hidden(self, x):
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        z = self.tok(x) + self.pos(pos)
        h = self.enc(z, mask=self.mask[: x.shape[1], : x.shape[1]])
        return self.ln(h)

    def forward(self, x):
        return self.head(self.hidden(x))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sha256_bytes(data: bytes):
    return hashlib.sha256(data).hexdigest()


def load_file(path: Path, minimum: int):
    data = path.read_bytes()
    data.decode("utf-8")
    if len(data) < minimum:
        raise ValueError(f"{path} has {len(data)} bytes, need >= {minimum}")
    return torch.tensor(list(data), dtype=torch.long), sha256_bytes(data)


def synthetic_corpora():
    # Deliberately non-natural smoke data; scientific outcomes are forbidden.
    patterns = {
        "ja_train": bytes(range(0, 64)) * 160,
        "ja_eval": bytes(range(63, -1, -1)) * 40,
        "en_train": bytes(range(64, 128)) * 160,
        "en_eval": bytes(range(127, 63, -1)) * 40,
        "ru_train": bytes(range(128, 192)) * 160,
        "ru_eval": bytes(range(191, 127, -1)) * 40,
    }
    corp = {k: torch.tensor(list(v), dtype=torch.long) for k, v in patterns.items()}
    hashes = {k: sha256_bytes(v) for k, v in patterns.items()}
    return corp, hashes


def load_corpora(data_dir: Path, synthetic_smoke: bool):
    if synthetic_smoke:
        return synthetic_corpora(), False
    names = ["ja_train", "ja_eval", "en_train", "en_eval", "ru_train", "ru_eval"]
    corp, hashes = {}, {}
    for name in names:
        minimum = 80_000 if name.endswith("train") else 8_000
        tensor, digest = load_file(data_dir / f"{name}.txt", minimum)
        corp[name] = tensor
        hashes[name] = digest
    hashes_match = hashes == EXPECTED_CORPUS_HASHES
    if not hashes_match:
        bad = {
            name: {"expected": EXPECTED_CORPUS_HASHES[name], "observed": hashes[name]}
            for name in names
            if hashes[name] != EXPECTED_CORPUS_HASHES[name]
        }
        raise RuntimeError(f"frozen corpus hash mismatch: {bad}")
    return (corp, hashes), True


def batch_from(corpus, batch_size, seed):
    if len(corpus) <= CTX + 1:
        raise ValueError(f"corpus too short: {len(corpus)} bytes")
    g = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, len(corpus) - CTX - 1, (batch_size,), generator=g)
    return corpus[starts[:, None] + torch.arange(CTX + 1)]


def lm_loss(model, z):
    x, y = z[:, :-1], z[:, 1:]
    logits = model(x)
    return F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))


def hidden_distill_loss(student, teacher, z):
    x = z[:, :-1]
    hs = student.hidden(x)
    with torch.no_grad():
        ht = teacher.hidden(x)
    return (1.0 - F.cosine_similarity(hs, ht, dim=-1, eps=1e-8)).mean()


def optimize_step(model, optimizer, z, teacher=None):
    ce = lm_loss(model, z)
    hidden = torch.tensor(0.0)
    loss = ce
    if teacher is not None:
        hidden = hidden_distill_loss(model, teacher, z)
        loss = ce + HIDDEN_WEIGHT * hidden
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()
    return float(ce.detach()), float(hidden.detach())


@torch.no_grad()
def eval_nll(model, eval_batches):
    return float(np.mean([float(lm_loss(model, z)) for z in eval_batches]))


def make_eval_batches(corpus, seed):
    return [batch_from(corpus, EVAL_BS, seed + j) for j in range(EVAL_BATCHES)]


def batch_digest(corpus, n_steps, base_seed, label):
    h = hashlib.sha256()
    for st in range(n_steps):
        z = batch_from(corpus, BS, base_seed + st)
        h.update(z.numpy().tobytes())
    h.update(label.encode("utf-8"))
    return h.hexdigest()


def train_common_base(seed, corp, schedule):
    seed_all(5100000 + seed)
    model = Model()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    for st in range(schedule.base_steps):
        z = batch_from(corp["ja_train"], BS, 5110000 + seed * 1000 + st)
        optimize_step(model, opt, z)
    return model


def train_teachers(base, seed, corp, schedule):
    teachers = {name: copy.deepcopy(base) for name in ["T_JA", "T_EN", "T_RU"]}
    opts = {
        k: torch.optim.AdamW(v.parameters(), lr=LR, weight_decay=0.0)
        for k, v in teachers.items()
    }
    source_counts = {k: {"ja": 0, "en": 0, "ru": 0} for k in teachers}
    language_code = {"ja": 1, "en": 2, "ru": 3}
    for st in range(schedule.teacher_steps):
        target_slot = st % 2 == 1
        for name in ["T_JA", "T_EN", "T_RU"]:
            if name == "T_JA":
                lang = "ja"
            elif name == "T_EN":
                lang = "en" if target_slot else "ja"
            else:
                lang = "ru" if target_slot else "ja"
            source_counts[name][lang] += 1
            z = batch_from(
                corp[f"{lang}_train"],
                BS,
                5120000 + seed * 10000 + st * 10 + language_code[lang],
            )
            optimize_step(teachers[name], opts[name], z)
    for teacher in teachers.values():
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
    return teachers, source_counts


def train_phase1(base, teachers, seed, corp, schedule):
    condition_teacher = {
        "JA_nat": None,
        "JA_TJA": teachers["T_JA"],
        "JA_TEN": teachers["T_EN"],
        "JA_TRU": teachers["T_RU"],
    }
    models = {k: copy.deepcopy(base) for k in condition_teacher}
    opts = {
        k: torch.optim.AdamW(v.parameters(), lr=LR, weight_decay=0.0)
        for k, v in models.items()
    }
    losses = {k: {"ce": [], "hidden": []} for k in models}
    for st in range(schedule.phase1_steps):
        z = batch_from(corp["ja_train"], BS, 5130000 + seed * 1000 + st)
        for cond in ["JA_nat", "JA_TJA", "JA_TEN", "JA_TRU"]:
            ce, hidden = optimize_step(
                models[cond], opts[cond], z, condition_teacher[cond]
            )
            losses[cond]["ce"].append(ce)
            losses[cond]["hidden"].append(hidden)
    return models, losses


def train_english_curve(model0, seed, corp, eval_batches, schedule):
    model = copy.deepcopy(model0)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    rows = [(0, eval_nll(model, eval_batches))]
    for st in range(1, schedule.phase2_steps + 1):
        z = batch_from(corp["en_train"], BS, 5140000 + seed * 1000 + st - 1)
        optimize_step(model, opt, z)
        if st % schedule.eval_every == 0 or st == schedule.phase2_steps:
            rows.append((st, eval_nll(model, eval_batches)))
    return rows


def cumulative_best(curve):
    out = []
    best = float("inf")
    for step, nll in sorted(curve):
        best = min(best, nll)
        out.append((float(step), best))
    return out


def first_crossing(curve, threshold):
    c = cumulative_best(curve)
    if c[0][1] <= threshold:
        return c[0][0]
    for i in range(1, len(c)):
        x0, y0 = c[i - 1]
        x1, y1 = c[i]
        if y1 <= threshold:
            if y1 == y0:
                return x1
            return x0 + (x1 - x0) * (y0 - threshold) / (y0 - y1)
    return None


def performance_span(curve, high, low):
    high_cross = first_crossing(curve, high)
    low_cross = first_crossing(curve, low)
    if high_cross is None or low_cross is None:
        return None
    return low_cross - high_cross


def audit_prior_results(repo_root: Path, seed: int):
    hits = []
    results = repo_root / "results"
    if not results.exists():
        return hits
    for path in results.glob("*.csv"):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if "seed" in frame.columns:
            vals = pd.to_numeric(frame["seed"], errors="coerce")
            if bool((vals == seed).any()):
                hits.append(str(path.relative_to(repo_root)))
    return hits


def run(seed, data_dir: Path, outdir: Path, smoke: bool, validation_natural: bool):
    if smoke and validation_natural:
        raise ValueError("choose at most one of --smoke and --validation-natural")
    validation_only = smoke or validation_natural
    scientific = not validation_only
    if validation_only and seed != PILOT_SEED:
        raise ValueError(f"validation modes must use seed {PILOT_SEED}")
    if scientific and seed not in CONFIRMATORY_SEEDS:
        raise ValueError(f"scientific R1 runs are locked to {CONFIRMATORY_SEEDS}")

    schedule = Schedule()
    if validation_only:
        schedule = Schedule(
            base_steps=2,
            teacher_steps=4,
            phase1_steps=2,
            phase2_steps=4,
            eval_every=2,
        )

    repo_root = Path(__file__).resolve().parents[1]
    prior_hits = audit_prior_results(repo_root, seed) if scientific else []
    if scientific and prior_hits:
        raise RuntimeError(f"fresh-seed audit failed for {seed}: {prior_hits}")

    (corp, hashes), hashes_match = load_corpora(data_dir, synthetic_smoke=smoke)
    base = train_common_base(seed, corp, schedule)
    teachers, teacher_counts = train_teachers(base, seed, corp, schedule)
    phase1_models, phase1_losses = train_phase1(base, teachers, seed, corp, schedule)

    ja_eval = make_eval_batches(corp["ja_eval"], 5150000 + seed * 100)
    en_eval = make_eval_batches(corp["en_eval"], 5160000 + seed * 100)
    ja_post = {cond: eval_nll(m, ja_eval) for cond, m in phase1_models.items()}

    curves = {"REF_EN": train_english_curve(base, seed, corp, en_eval, schedule)}
    for cond, model in phase1_models.items():
        curves[cond] = train_english_curve(model, seed, corp, en_eval, schedule)

    ref_best = cumulative_best(curves["REF_EN"])
    top = ref_best[0][1]
    bottom = ref_best[-1][1]
    if not bottom < top:
        raise RuntimeError(
            f"reference English curve did not improve: top={top}, bottom={bottom}"
        )
    high = bottom + 0.8 * (top - bottom)
    low = bottom + 0.2 * (top - bottom)

    summary_rows = []
    for cond in ["JA_nat", "JA_TJA", "JA_TEN", "JA_TRU"]:
        span = performance_span(curves[cond], high, low)
        summary_rows.append(
            {
                "seed": seed,
                "condition": cond,
                "JA_post_nll": ja_post[cond],
                "EN_step0_nll": curves[cond][0][1],
                "span_updates": np.nan if span is None else span,
                "censored": span is None,
                "reference_high": high,
                "reference_low": low,
                "smoke_only": smoke,
                "validation_only": validation_only,
            }
        )

    summary = pd.DataFrame(summary_rows)
    spans = dict(zip(summary["condition"], summary["span_updates"]))

    def finite_pair(a, b):
        return not pd.isna(spans[a]) and not pd.isna(spans[b])

    effects = {
        "primary_TEN_minus_TJA": (
            float(spans["JA_TEN"] - spans["JA_TJA"])
            if finite_pair("JA_TEN", "JA_TJA")
            else None
        ),
        "practical_TEN_minus_nat": (
            float(spans["JA_TEN"] - spans["JA_nat"])
            if finite_pair("JA_TEN", "JA_nat")
            else None
        ),
        "specificity_TEN_minus_TRU": (
            float(spans["JA_TEN"] - spans["JA_TRU"])
            if finite_pair("JA_TEN", "JA_TRU")
            else None
        ),
        "saving_vs_TJA": (
            float(1.0 - spans["JA_TEN"] / spans["JA_TJA"])
            if finite_pair("JA_TEN", "JA_TJA") and spans["JA_TJA"] > 0
            else None
        ),
    }

    outdir.mkdir(parents=True, exist_ok=True)
    curve_rows = []
    for cond, curve in curves.items():
        for step, nll in curve:
            curve_rows.append(
                {
                    "seed": seed,
                    "condition": cond,
                    "B_step": step,
                    "EN_nll": nll,
                    "smoke_only": smoke,
                    "validation_only": validation_only,
                }
            )
    pd.DataFrame(curve_rows).to_csv(
        outdir / f"natural_s2_seed{seed}_curves.csv", index=False
    )
    summary.to_csv(outdir / f"natural_s2_seed{seed}_summary.csv", index=False)

    audit = {
        "seed": seed,
        "smoke_only": smoke,
        "validation_only": validation_only,
        "scientific_evidence": scientific,
        "protocol": "protocols/natural_ja_en_s2_hidden_teacher_r1_2026-09-06.json",
        "corpus_manifest": "protocols/natural_ja_en_s2_corpus_manifest_2026-09-06.json",
        "corpus_sha256": hashes,
        "corpus_hashes_match_frozen_manifest": hashes_match if not smoke else None,
        "prior_result_seed_hits": prior_hits,
        "student_phase0_sources": {"ja": schedule.base_steps, "en": 0, "ru": 0},
        "student_phase1_sources": {"ja": schedule.phase1_steps, "en": 0, "ru": 0},
        "teacher_source_counts": teacher_counts,
        "teacher_target_slot_positions_identical": True,
        "phase1_common_batch_digest": batch_digest(
            corp["ja_train"],
            schedule.phase1_steps,
            5130000 + seed * 1000,
            "phase1-ja",
        ),
        "phase2_common_batch_digest": batch_digest(
            corp["en_train"],
            schedule.phase2_steps,
            5140000 + seed * 1000,
            "phase2-en",
        ),
        "fixed_byte_vocab": VOCAB,
        "reference_high": high,
        "reference_low": low,
        "effects": effects,
        "phase1_mean_losses": {
            cond: {
                "ce": float(np.mean(vals["ce"])),
                "hidden": float(np.mean(vals["hidden"])),
            }
            for cond, vals in phase1_losses.items()
        },
        "schedule": schedule.__dict__,
    }
    (outdir / f"natural_s2_seed{seed}_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "seed": seed,
                "smoke_only": smoke,
                "validation_only": validation_only,
                "scientific_evidence": scientific,
                "audit": "PASS",
            },
            indent=2,
        )
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=PILOT_SEED)
    ap.add_argument("--data-dir", type=Path, default=Path("data/natural_r1"))
    ap.add_argument("--outdir", type=Path, default=Path("results/raw_natural_r1"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--validation-natural", action="store_true")
    args = ap.parse_args()
    run(
        args.seed,
        args.data_dir,
        args.outdir,
        smoke=args.smoke,
        validation_natural=args.validation_natural,
    )


if __name__ == "__main__":
    main()
