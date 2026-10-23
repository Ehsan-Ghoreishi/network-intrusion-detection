"""End-to-end smoke test: train both models for one epoch on the tiny fixture."""

import json

import pytest

from nids.robustness import summarise


@pytest.fixture
def dirs(raw_dir, tmp_path, monkeypatch):
    from nids import anomaly, data, supervised
    out = tmp_path / "out"
    for mod in (supervised, anomaly):
        monkeypatch.setattr(mod, "MODELS_DIR", out / "models")
        monkeypatch.setattr(mod, "REPORTS_DIR", out / "reports")
    monkeypatch.setattr(data, "DATA_DIR", raw_dir)
    return out


def test_supervised_end_to_end(dirs):
    from nids import supervised
    summary = supervised.run(epochs=1, batch_size=16)
    assert (dirs / "models" / "classifier.keras").exists()
    saved = json.loads((dirs / "reports" / "supervised_metrics.json").read_text())
    assert saved == summary
    assert 0 <= summary["test_binary"]["detection_rate"] <= 1


def test_autoencoder_end_to_end(dirs):
    from nids import anomaly
    summary = anomaly.run(epochs=1, batch_size=16)
    assert (dirs / "models" / "autoencoder_scores.npz").exists()
    assert summary["threshold"] > 0


def test_run_without_save_writes_nothing(dirs):
    from nids import anomaly
    anomaly.run(epochs=1, batch_size=16, save=False)
    assert not dirs.exists()


def test_summarise_mean_and_std():
    runs = [{"s": {"m": v}} for v in (0.1, 0.2, 0.3)]
    out = summarise(runs, [("s", "m")])["m"]
    assert out["mean"] == 0.2 and out["std"] == 0.1
    assert (out["min"], out["max"]) == (0.1, 0.3)
