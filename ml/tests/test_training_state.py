

def test_is_complete_accepts_legacy_final_marker(tmp_path):
    """Runs launched before the marker rename print "Adapters saved to ...";
    the driver must still recognise them as finished rather than looping."""
    from ml.tune.training_state import is_complete
    names = [f"{i:07d}_adapters.safetensors" for i in range(500, 4725, 500)]
    for n in names:
        (tmp_path / n).write_bytes(b"x")
    log = tmp_path / "train.log"
    log.write_text("Iter 4724: ...\nAdapters saved to /x/adapters.safetensors\n")
    assert is_complete(tmp_path, iters=4724, save_every=500, log=log)


def test_is_complete_requires_marker_substring(tmp_path):
    """A run whose checkpoints are present but whose log only has an
    intermediate save message must not count as finished."""
    from ml.tune.training_state import is_complete
    names = [f"{i:07d}_adapters.safetensors" for i in range(500, 4725, 500)]
    for n in names:
        (tmp_path / n).write_bytes(b"x")
    log = tmp_path / "train.log"
    log.write_text("Iter 4500: Saved adapter weights to /x/04500_adapters.safetensors.\n")
    assert not is_complete(tmp_path, iters=4724, save_every=500, log=log)
