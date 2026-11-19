from argparse import ArgumentParser
from pathlib import Path
from tempfile import NamedTemporaryFile

import yaml


ROOT = Path(__file__).resolve().parents[1]


def default_dataset_config():
    dataset = ROOT / "data" / "D-Fire-datasets" / "fire"
    if not dataset.is_dir():
        raise FileNotFoundError(f"Extract D-Fire-datasets.zip into {ROOT / 'data'} or pass --data.")

    config = {
        "path": str(dataset),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": ["fire", "default", "smoke"],
    }
    data_file = dataset.parent.parent / "fireguard-local.yaml"
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=data_file.parent, delete=False) as output:
        yaml.safe_dump(config, output, sort_keys=False)
    Path(output.name).replace(data_file)
    return data_file


def main():
    parser = ArgumentParser(description="Train FireGuardML with YOLO26s.")
    parser.add_argument("--data", type=Path, help="Path to a custom Ultralytics dataset YAML.")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs.")
    parser.add_argument("--device", help="Training device, such as cpu or 0.")
    args = parser.parse_args()

    try:
        data = args.data.resolve() if args.data else default_dataset_config()
    except FileNotFoundError as error:
        parser.error(str(error))
    if not data.is_file():
        parser.error(f"Dataset YAML not found at {data}")

    from ultralytics import YOLO

    model = YOLO("yolo26s.pt")
    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=416,
        plots=True,
        device=args.device,
        project=str(ROOT / "runs" / "detect"),
        name="fireguard-yolo26s",
    )
    print(f"Best checkpoint: {model.trainer.best}")


if __name__ == "__main__":
    main()
