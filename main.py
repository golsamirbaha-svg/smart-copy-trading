import argparse
import logging

from src.backtest import build_backtest
from src.copy_engine import build_decisions
from src.data_loader import build_dataset
from src.diagnostics import build_diagnostics
from src.evaluation import build_evaluation
from src.risk import build_risk_replay
from src.traders import build_trader_dataset


def main():
    parser = argparse.ArgumentParser(description="Smart Copy Trading prototype")
    parser.add_argument("--step", default="data", choices=["data", "traders", "evaluate", "engine", "risk", "backtest", "diagnostics"],
                        help="which step to run (more steps will be added)")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.step == "data":
        build_dataset(args.config)
    elif args.step == "traders":
        build_trader_dataset(args.config)
    elif args.step == "evaluate":
        build_evaluation(args.config)
    elif args.step == "engine":
        build_decisions(args.config)
    elif args.step == "risk":
        build_risk_replay(args.config)
    elif args.step == "backtest":
        build_backtest(args.config)
    elif args.step == "diagnostics":
        build_diagnostics(args.config)


if __name__ == "__main__":
    main()
