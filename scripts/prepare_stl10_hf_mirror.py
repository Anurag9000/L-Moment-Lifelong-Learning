#!/usr/bin/env python3
"""Convert the official-split Hugging Face STL-10 parquet mirror to torchvision bins."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image


SOURCES = {
    "train": "https://huggingface.co/datasets/tanganke/stl10/resolve/main/data/train-00000-of-00001.parquet",
    "test": "https://huggingface.co/datasets/tanganke/stl10/resolve/main/data/test-00000-of-00001.parquet",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def convert(partition: str, source: Path, destination: Path, expected: int) -> dict:
    parquet = pq.ParquetFile(source)
    if parquet.metadata.num_rows != expected:
        raise ValueError(f"{partition}: expected {expected} rows, got {parquet.metadata.num_rows}")
    counts = Counter()
    image_path = destination / f"{partition}_X.bin"
    label_path = destination / f"{partition}_y.bin"
    with image_path.open("wb") as images_out, label_path.open("wb") as labels_out:
        for batch in parquet.iter_batches(batch_size=128, columns=["image", "label"]):
            for row in batch.to_pylist():
                label = int(row["label"])
                if not 0 <= label < 10:
                    raise ValueError(f"Unexpected {partition} label {label}")
                encoded = row["image"].get("bytes")
                if not encoded:
                    raise ValueError(f"Missing image bytes for {partition} row")
                with Image.open(io.BytesIO(encoded)) as image:
                    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
                if rgb.shape != (96, 96, 3):
                    raise ValueError(f"Unexpected STL-10 image shape {rgb.shape}")
                images_out.write(np.ascontiguousarray(rgb.transpose(2, 0, 1)).tobytes())
                # torchvision's STL10 reader converts the source's 1-based bytes to 0-based labels.
                labels_out.write(bytes((label + 1,)))
                counts[label] += 1
    expected_per_class = expected // 10
    if counts != Counter({label: expected_per_class for label in range(10)}):
        raise ValueError(f"Unexpected class balance for {partition}: {dict(counts)}")
    return {"rows": expected, "class_counts": dict(sorted(counts.items())),
            "parquet_sha256": sha256(source),
            "images_sha256": sha256(image_path), "labels_sha256": sha256(label_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if not args.parquet_root.is_dir():
        parser.error(f"Missing parquet directory: {args.parquet_root}")
    output = args.output_root / "stl10_binary"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing dataset files: {output}")
    output.mkdir(parents=True, exist_ok=True)
    partitions = {
        part: convert(part, args.parquet_root / f"{part}.parquet", output, rows)
        for part, rows in (("train", 5000), ("test", 8000))
    }
    metadata = {"dataset": "STL-10", "source_dataset": "tanganke/stl10",
                "source_urls": SOURCES, "format": "torchvision STL10 binary layout",
                "image_shape": [96, 96, 3], "label_mapping": "mirror labels 0-9 stored as 1-10",
                "partitions": partitions}
    (output / "mirror_provenance.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
