"""Tests for the deliverables checklist."""
from src.check import find_missing, readme_image_paths


def test_find_missing(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    assert find_missing(["a.txt", "b.txt"], str(tmp_path)) == ["b.txt"]


def test_readme_image_paths():
    text = "![Equity curves](results/equity_curves.png)\ntext ![x](results/robustness.png)"
    assert readme_image_paths(text) == ["results/equity_curves.png", "results/robustness.png"]
