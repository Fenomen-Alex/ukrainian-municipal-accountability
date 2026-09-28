"""Decide whether an MLX training run has actually finished.

Why this needs to exist
-----------------------
``mlx_lm``'s trainer writes ``adapters.safetensors`` at **every** checkpoint, not
once at the end:

    if it % args.steps_per_save == 0 and rank == 0:
        mx.save_safetensors(str(args.adapter_file), adapter_weights)
        checkpoint = ... / f"{it:07d}_adapters.safetensors"
        mx.save_safetensors(str(checkpoint), adapter_weights)
    ...
    # Save final weights        <- only after the loop
    mx.save_safetensors(str(args.adapter_file), adapter_weights)

So ``adapters.safetensors`` is a copy of the newest checkpoint and its presence
means nothing about completion. An unattended pipeline that waited on that file
would have fused a 100-iteration adapter and reported it as the finished arm --
silently, because the file is exactly the right size and shape.

Two independent signals are used instead, and both must agree:

  1. the final numbered checkpoint exists, and the numbered checkpoints run
     continuously from ``save_every`` to ``iters`` with none missing;
  2. the training log contains ``"Saved final weights to"``, which the trainer
     only prints after the loop.

Either alone would be defensible; requiring both means a truncated run, a
crashed process and a half-written final save are all caught.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

CHECKPOINT_RE = re.compile(r"^(\d+)_adapters\.safetensors$")
FINAL_MARKER = "Saved final weights to"


def checkpoints(adapter_dir: Path) -> list[int]:
    """Iteration numbers of the numbered checkpoints, ascending."""
    out = []
    for p in Path(adapter_dir).glob("*_adapters.safetensors"):
        m = CHECKPOINT_RE.match(p.name)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def describe(adapter_dir: Path, iters: int, save_every: int) -> dict:
    got = checkpoints(adapter_dir)
    expected = list(range(save_every, iters + 1, save_every))
    missing = [i for i in expected if i not in got]
    return {
        "dir": str(adapter_dir),
        "iters": iters,
        "save_every": save_every,
        "checkpoints": got,
        "n_checkpoints": len(got),
        "expected_checkpoints": expected,
        "missing": missing,
        # ``adapters.safetensors`` is rewritten at each checkpoint, so its
        # presence is deliberately not treated as progress.
        "final_checkpoint_present": not missing and bool(got),
    }


def is_complete(adapter_dir: Path, iters: int, save_every: int,
                log: Path | None = None) -> bool:
    d = describe(adapter_dir, iters, save_every)
    if not d["final_checkpoint_present"]:
        return False
    if log is not None:
        if not log.exists() or FINAL_MARKER not in log.read_text(errors="replace"):
            return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", required=True, type=Path)
    ap.add_argument("--iters", type=int, required=True)
    ap.add_argument("--save-every", type=int, required=True)
    ap.add_argument("--log", type=Path, default=None)
    ap.add_argument("--describe", action="store_true",
                    help="print state as JSON; always exits 0")
    a = ap.parse_args()

    d = describe(a.dir, a.iters, a.save_every)
    d["log_has_final_marker"] = bool(
        a.log and a.log.exists()
        and FINAL_MARKER in a.log.read_text(errors="replace"))
    d["complete"] = is_complete(a.dir, a.iters, a.save_every, a.log)
    if a.describe:
        print(json.dumps(d, indent=2))
        return
    # Poll from a shell loop: exit 0 means "done, go ahead".
    raise SystemExit(0 if d["complete"] else 1)


if __name__ == "__main__":
    main()
