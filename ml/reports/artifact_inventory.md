# Local ML artifact inventory & cleanup record

> **HISTORICAL INVENTORY — NOT A DELETION LIST.**
> Everything below is still present and intentionally retained, including the
> non-canonical training arms: they are the evidence behind the release decision
> in [`ml/tune/FINAL_MODEL_ASSESSMENT.md`](../tune/FINAL_MODEL_ASSESSMENT.md).
> The canonical model is v2 only
> (see [`ml/tune/FINAL_STATUS.md`](../tune/FINAL_STATUS.md)). Where an inventory
> entry conflicts with that status, the status document wins. Adapters are
> gitignored and are not part of the Git repository regardless of this listing.

Date: after the v3 control-arm negative result and before the full-corpus v3
treatment run.

Disk base: `~/.cache/huggingface/hub` holds only `mlx-community/Qwen3-8B-4bit`
(the training/fusion base, needed) plus an unrelated embedding repo (`G37A`,
44K, left untouched). Nothing else in caches is project-attributable and
disposable; the `uv`/`torch`/`outlines` caches belong to other tooling and were
left alone.

## Adapters + fused artifacts (`ml/data/tune/adapters/`, gitignored)

All six fused directories are 4.3 GB each; `config.json`, `tokenizer.json`,
`tokenizer_config.json`, `chat_template.jinja` were byte-identical across all of
them. `model.safetensors` shas:

| path | size | sha256(model) | verdict |
|---|---|---|---|
| `qwen3-8b-lora-v2-attempt10-fused/` | 4.3G | `84c85195…` | **KEEP** — canonical v2, equals `Fenomen-Alex/ukrainian-municipal-accountability-qwen3-8b` |
| `qwen3-8b-lora-v2-fused/` | 4.3G | `84c85195…` | DELETE copy; replaced by symlink to attempt10 (byte-identical alias, documented in V2_SERVING.md) |
| `qwen3-8b-lora-v2-attempt5-fused/` | 4.3G | `3ad4d73d…` | DELETE — obsolete v2 attempt, superseded by attempt10 |
| `qwen3-8b-lora-v2-attempt6-fused/` | 4.3G | `94f8fc96…` | DELETE — obsolete v2 attempt, superseded by attempt10 |
| `qwen3-8b-lora-v3-control-fused/` | 4.3G | `1d9569c8…` | DELETE — control negative result; re-fusable from kept `qwen3-8b-lora-v3-control/` adapter (fuse_v3), eval evidence committed as JSON |
| `qwen3-8b-lora-fused/` | 4.3G | `6f9bccf2…` | DELETE — v1 fusion. Lossless: kept `qwen3-8b-lora/` adapter re-fuses to sha `6f9bccf28bb489c5`, verified in this session |

Adapters:

| path | size | verdict |
|---|---|---|
| `qwen3-8b-lora-v2/` | 333M | KEEP — canonical v2 adapter (recipe source, test-referenced) |
| `qwen3-8b-lora-v3-control/` | 333M | KEEP — control adapter, cheap repro of the failed validation |
| `qwen3-8b-lora/` | 185M | KEEP — v1 adapter, regenerates v1 fused losslessly |
| `qwen3-8b-lora-v2-attempt3/` | 333M | DELETE — obsolete attempt, no references |

## Temporary evaluation copies (`$TMPDIR/opencode/`)

| path | size | verdict |
|---|---|---|
| `fresh/` (model + HF cache) | 4.3G | DELETE — disposable fresh-download copy, re-downloadable from HF |
| `tiny/` (runA/B/C reproducibility runs) | 111M | DELETE — temp reproducibility runs (audit conclusions committed) |
| `v2_baseline_backup/`, `*.log`, `*_attempt*.json`, `multitok_candidates.txt` | ~2M | DELETE — temp evidence, committed results supersede |
| `v1_refuse/` | 4.3G | DELETE — the lossless-re-fusion proof; sha recorded above |

## Kept elsewhere (no action)

- `ml/data/tune/v2/`, `v3/`, smoke `results/` incl. `v1_finetuned_stash/`
  (the surviving real v1 baseline), eval evidence, source, tests, reports.
- HF cache `mlx-community/Qwen3-8B-4bit` (base, needed for training/fusion).

## tally

Scheduled deletions: ~21.6 GB of adapters/fused + ~4.4 GB temp = ~26 GB.