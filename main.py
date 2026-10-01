import argparse
import logging

from src.data_loader import build_dataset


def main():
    parser = argparse.ArgumentParser(description="Smart Copy Trading prototype")
    parser.add_argument("--step", default="data", choices=["data"],
                        help="کدام مرحله اجرا شود (مراحل بعدی اضافه می‌شوند)")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.step == "data":
        build_dataset(args.config)


if __name__ == "__main__":
    main()
