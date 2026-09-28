"""Fuse a v3 LoRA adapter into a standalone model, reproducing the v2 artifact.

Why this exists
---------------
Every arm of the experiment has to be scored the *same* way, and the v2 baseline
was measured on a genuinely fused model rather than on a base model with an
adapter applied at load time. Fusing a LoRA into the 4-bit weights and applying
it on the fly are not guaranteed to be numerically identical, so a comparison
that mixes the two measures a conversion artifact instead of a training
difference.

The trap this script closes
---------------------------
``mlx_lm.fuse`` saves the tokenizer and chat template it copies from the *base
Hugging Face repo*, not from the artifact that was actually served. That is how
v2's fused directory ended up with a template that did not honour
``enable_thinking``: the template had to be patched afterwards, by hand, and
nothing recorded that it had been. A fresh v3 arm fused the naive way would
silently ship the unpatched template, and the v2 serving contract
(``enable_thinking=False``) would be broken for v3 only.

So this fuses the weights with ``mlx_lm`` but then takes the tokenizer files and
chat template from the verified v2 artifact, and writes a manifest recording
exactly what was inherited. The result is that a v3 arm differs from the v2
baseline in the weights and nothing else.

Usage::

    .venv-mlx/bin/python -m ml.tune.fuse_v3 \\
        --adapter ml/data/tune/adapters/qwen3-8b-lora-v3-control \\
        --out ml/data/tune/adapters/qwen3-8b-lora-v3-control-fused
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "ml/data/tune/adapters"

#: The v2 artifact that ``ml/tune/V2_SERVING.md`` and ``ml/tune/serve_v2.py``
#: treat as canonical. Its tokenizer and chat template are the reference.
V2_FUSED = DATA / "qwen3-8b-lora-v2-attempt10-fused"

#: Files that define the tokenization/serving contract rather than the weights.
#: Copied from V2_FUSED so that every arm is served identically.
INHERITED = ("chat_template.jinja", "tokenizer.json", "tokenizer_config.json")

BASE_MODEL = "mlx-community/Qwen3-8B-4bit"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def fuse(adapter: Path, out: Path, *, base: str = BASE_MODEL,
         template_from: Path = V2_FUSED) -> dict:
    """Fuse ``adapter`` into ``out`` and return the provenance manifest."""
    if not (adapter / "adapter_config.json").exists():
        raise SystemExit(f"{adapter} is not a LoRA adapter directory "
                         "(no adapter_config.json)")
    for name in INHERITED:
        if not (template_from / name).exists():
            raise SystemExit(
                f"{template_from}/{name} is missing; refusing to fuse without the "
                "verified serving contract. See ml/tune/V2_SERVING.md.")

    # Imported after the cheap checks so a bad path fails without loading MLX.
    from mlx.utils import tree_unflatten
    from mlx_lm.utils import load, save

    print(f"loading {base} + {adapter} ...", flush=True)
    model, tokenizer, config = load(base, adapter_path=str(adapter), return_config=True)

    fused = [(n, m.fuse(dequantize=False))
             for n, m in model.named_modules() if hasattr(m, "fuse")]
    if not fused:
        raise SystemExit("no fusable LoRA modules found; adapter did not attach")
    model.update_modules(tree_unflatten(fused))
    print(f"fused {len(fused)} modules", flush=True)

    save(out, base, model, tokenizer, config, donate_model=False)

    # The serving contract, inherited rather than re-derived.
    for name in INHERITED:
        shutil.copyfile(template_from / name, out / name)

    # config.json must already match v2's: same base, same quantisation, same
    # fuse mode. If it does not, the arms are not comparable and we should know
    # now rather than after a 20-minute evaluation.
    v2_config = template_from / "config.json"
    config_match = (not v2_config.exists()
                    or sha256(out / "config.json") == sha256(v2_config))

    manifest = {
        "base_model": base,
        "adapter_path": str(adapter.relative_to(ROOT)),
        "adapter_sha256": {
            p.name: sha256(p) for p in sorted(adapter.glob("*.safetensors"))},
        "fused_path": str(out.relative_to(ROOT)),
        "inherited_from": str(template_from.relative_to(ROOT)),
        "inherited_sha256": {name: sha256(out / name) for name in INHERITED},
        "fused_modules": len(fused),
        "dequantize": False,
        "config_matches_v2": config_match,
        "git_head": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
    }
    (out / "fuse_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not config_match:
        print("WARNING: config.json differs from the v2 artifact; the arms are not "
              "directly comparable. See fuse_manifest.json.", flush=True)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--adapter", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--template-from", default=V2_FUSED, type=Path)
    a = ap.parse_args()
    m = fuse(a.adapter, a.out, base=a.base, template_from=a.template_from)
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
