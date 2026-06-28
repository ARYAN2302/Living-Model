# Living-Model: Self-Improving LFM2.5-350M with SLAO + MVA

## What this is

A self-improving on-device language model that combines:
- **SLAO** (Single LoRA Adaptive Orthogonal) — continual learning without forgetting
- **MVA** (Model-Validated Adaptation) — self-improvement via reduced-error filtering

Built on Liquid LFM2.5-350M (hybrid conv+attention, 350M params).

## The framing that survives reading the code

**Self-improvement via reduced-error filtering, not self-improvement via correct self-generated data.**

MVA's certainty gate (INTUITOR KL(U||p)) filters out 60-88% of wrong answers
(precision 79-94% across seeds), reducing wrong-answer contamination from ~30%
(naive) to ~6-21% (MVA). The pass^5 improvement reflects the model training on
fewer mistakes, not on verified-correct self-generated data.

## Version lineage

| Version | What it does | Result |
|---------|--------------|--------|
| **v13** | SLAO continual learning baseline (3 domains, r=32) | FF(A)=1.09, 10% forgetting vs naive's 55% |
| **v14** | SLAO + 1 MVA round (Architecture A) | COMPOSES: +0.070 pass^5, no PPL spike |
| **v15** | SLAO + 3 MVA rounds (threshold tracking) | *running* — tests if signal compounds |

## Key results

### SLAO (v13)
- 5.5x better retention than naive sequential fine-tuning
- 10% residual forgetting (the gap MVA aims to recover)
- Cross-term noise ratio: 1.91 (vs 1.96 naive)

### MVA validation (v5, standalone)
- Certainty signal r=0.341 with correctness (p<0.001) on 350M
- 2-seed confirmation: MVA beats naive by +11-12pp on pass^5
- Validation precision 79-94% (known flaw — reduced-error filtering, not correct data)

### SLAO + MVA integration (v14)
- MVA composes with SLAO: +0.070 pass^5, domain PPL stable
- Adaptive threshold necessary (SLAO shifts certainty distribution: 16.68 -> 11.58)
- 1 round = directional but not statistically significant (chi-sq=1.8 vs 3.84 needed)

## Directory structure

```
v13/  — SLAO baseline (continual learning without forgetting)
v14/  — SLAO + 1 MVA round (integration test)
v15/  — SLAO + 3 MVA rounds (compounding test, with threshold tracking)
```

## Model

LiquidAI/LFM2.5-350M
- Hybrid architecture: 6 attention + 10 conv layers (16 total)
- 350M parameters, vocab 65536
- Native transformers support (>=5.0)

## LoRA config

- v13 minimal: `["in_proj", "out_proj"]` — 26 modules, rank 32
- MVA validation (v5): full 8-module config — used to validate certainty signal
- Integration (v14/v15): v13 minimal config for consistency with proven SLAO baseline
