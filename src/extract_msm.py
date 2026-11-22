"""Extract the documented CNC sensor CSVs and labels from the Kaggle archive."""
import argparse
from pathlib import Path, PurePosixPath
import shutil
from zipfile import ZipFile


def extract(source, destination, machine=None):
    destination.mkdir(parents=True, exist_ok=True)
    with ZipFile(source) as archive:
        selected = [
            entry for entry in archive.infolist()
            if not entry.is_dir()
            and (machine is None or entry.filename.startswith(f"{machine}/")
                 or entry.filename == "dataset_summary.xlsx")
            and PurePosixPath(entry.filename).name.endswith(
                ("label.csv", "_acc.csv", "_ct.csv", "_mtc.csv", "dataset_summary.xlsx")
            )
        ]
        if not selected or (machine and not any(entry.filename.startswith(f"{machine}/") for entry in selected)):
            raise ValueError(f"No matching CNC sensor and label files for {machine or source}.")
        for entry in selected:
            target = (destination / entry.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError(f"Unsafe ZIP entry: {entry.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(entry) as input_file, target.open("wb") as output:
                shutil.copyfileobj(input_file, output, 4 * 1024 * 1024)
        print(f"Extracted {len(selected)} MSM sensor, label and summary files to {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=Path("dataset/MSM-Milling.zip"))
    parser.add_argument("--output", type=Path, default=Path("dataset/cnc"))
    parser.add_argument("--machine", help="Extract one machine folder from the full archive")
    args = parser.parse_args()
    extract(args.archive, args.output, args.machine)

