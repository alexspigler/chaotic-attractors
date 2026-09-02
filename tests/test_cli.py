import sys

import pytest

from chaotic_attractors import __main__ as cli


def test_default_generate_uses_viable_tinkerbell_start(monkeypatch, tmp_path):
    captured = {}

    def fake_prepare(**kwargs):
        captured.update(kwargs)
        return {
            "x": [0.0],
            "y": [0.0],
            "density": [1.0],
            "params": kwargs["params"],
            "equation_id": kwargs["equation_id"],
            "lyapunov_exponent": 0.2,
            "seed": 12,
        }

    monkeypatch.setattr(cli, "prepare_generate_data", fake_prepare)
    monkeypatch.setattr(cli, "save_attractor", lambda **_: [])
    monkeypatch.setattr(
        sys,
        "argv",
        ["chaotic-attractors", "--output-dir", str(tmp_path)],
    )

    cli.main()

    assert captured["params"] == cli.DEFAULT_PARAMS
    assert captured["x_start"] == -0.72
    assert captured["y_start"] == -0.64


def test_search_forwards_output_directory_and_seed(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(
        cli, "search_attractors", lambda **kwargs: captured.update(kwargs)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "chaotic-attractors",
            "--mode",
            "search",
            "--output-dir",
            str(tmp_path),
            "--seed",
            "91",
        ],
    )

    cli.main()

    assert captured["output_dir"] == str(tmp_path)
    assert captured["seed"] == 91
    assert (captured["x_start"], captured["y_start"]) == (0.5, 0.5)


def test_custom_equation_requires_all_parameters(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["chaotic-attractors", "--equation", "Custom3", "--a", "-2.17"],
    )

    with pytest.raises(SystemExit) as error:
        cli.main()

    assert error.value.code == 2


def test_invalid_search_count_exits_cleanly(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["chaotic-attractors", "--mode", "search", "--max-attempts", "0"],
    )

    with pytest.raises(SystemExit) as error:
        cli.main()

    assert error.value.code == 2
