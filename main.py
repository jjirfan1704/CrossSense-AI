import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate CrossSense AI models")
    commands = parser.add_subparsers(dest="machine", required=True)
    fan = commands.add_parser("fan")
    fan.add_argument("--data", type=Path, default=Path("dataset/fan"))
    bearing = commands.add_parser("bearing")
    bearing.add_argument("--data", type=Path, required=True)
    bearing.add_argument("--channel", type=int, required=True)
    bearing.add_argument("--rate", type=float, default=20000)
    cnc = commands.add_parser("cnc")
    cnc.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.machine == "fan":
            from src.fan import train
            train(args.data)
        elif args.machine == "bearing":
            from src.bearing import train
            train(args.data, args.channel, args.rate)
        else:
            from src.cnc import train
            train(args.data)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
