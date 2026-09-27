# v2 training set

Assembled by `ml/tune/build_v2.py` (final candidate = attempt-10 recipe):

- `train.jsonl` — **8,519** chat examples = frozen v1 single-topic train
  (5,384, unchanged) + multi-topic augmentation (3,135: 22 real decompositions
  ×12 real copies + 420 generic/160 same-street/140 compact/160 registry/160
  smoke-clone synthetic ×3). Multi-topic stream share ≈ **0.368**; single-topic
  stays the majority (~0.63).
- `validation.jsonl` / `test.jsonl` — **byte-identical copies of the frozen v1
  benchmark**. v1↔v2 comparison is therefore exact; nothing about the benchmark
  changed.
- `valid.jsonl` — relative symlink to `validation.jsonl` (mirrors v1 layout).
- `meta.json` — build metadata (components, upsampling factors, per-kind
  exposure counts) consumed by `ml/tests/test_v2_augmentation.py`.

Multi-topic ability is measured on the separate targeted suite
`ml/data/tune/multitopic/eval.jsonl` (85 examples, never trained on) — it is not
part of the frozen benchmark.