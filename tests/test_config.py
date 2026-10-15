from nbft import config


def test_path_formats_and_resolves(monkeypatch, tmp_path):
    monkeypatch.setenv("NBFT_DATA_ROOT", str(tmp_path))
    p = config.path("statics.npr_ref", am_pm="am")
    assert p == tmp_path / "proc/statics/NPR_am_thaw_ref.nc"
