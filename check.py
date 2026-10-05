"""Pre-push checklist: are all deliverables of the challenge present in the repository folder?"""
from __future__ import annotations

import re
from pathlib import Path

from src.data_loader import load_config


def find_missing(paths: list[str], root: str = ".") -> list[str]:
    return [p for p in paths if not (Path(root) / p).exists()]


def readme_image_paths(readme: str) -> list[str]:
    return re.findall(r"!\[[^\]]*\]\(([^)\s]+)\)", readme)


def check_deliverables(cfg_path: str = "config.yaml", root: str = ".") -> bool:
    cfg = load_config(cfg_path)
    required = [
        "README.md", "requirements.txt", "config.yaml",
        cfg["data"]["clean_path"],                        # dataset used for the reported results
        cfg["simulation"]["trades_path"], cfg["simulation"]["summary_path"],
        cfg["evaluation"]["scores_path"], cfg["evaluation"]["metrics_path"],
        cfg["backtest"]["results_path"], cfg["backtest"]["results_md_path"],   # comparison table
        cfg["backtest"]["equity_plot_path"],                                   # equity curve
        cfg["backtest"]["trades_path"], cfg["backtest"]["decisions_path"],
        cfg["robustness"]["results_path"], cfg["robustness"]["results_md_path"],
        cfg["robustness"]["plot_path"],
    ]
    missing = find_missing(required, root)

    readme = Path(root) / "README.md"
    if readme.exists():
        missing += [p for p in find_missing(readme_image_paths(readme.read_text(encoding="utf-8")), root)
                    if p not in missing]

    if missing:
        print("MISSING deliverables (run the matching --step, then check again):")
        for p in missing:
            print("  -", p)
        return False
    print("All deliverables are present. Commit data/processed/ and results/ together with the code.")
    return True
