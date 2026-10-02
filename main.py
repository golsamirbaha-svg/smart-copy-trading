import argparse
import logging

from src.data_loader import build_dataset
from src.traders import build_trader_dataset


def main():
    parser = argparse.ArgumentParser(description="Smart Copy Trading prototype")
    parser.add_argument("--step", default="data", choices=["data", "traders"],
                        help="which step to run (more steps will be added)")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.step == "data":
        build_dataset(args.config)
    elif args.step == "traders":
        build_trader_dataset(args.config)


if __name__ == "__main__":
    main()
