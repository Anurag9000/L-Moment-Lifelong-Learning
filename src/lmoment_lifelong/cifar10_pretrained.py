"""Shared frozen DINOv2 features for CIFAR-10 and STL-10 experiment suites."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

MODEL_ID = "facebook/dinov2-base"
# Immutable Hub commit containing the published safetensors checkpoint.
MODEL_REVISION = "fed69336d31be0ed1883be4cd5ea0456fa3fdbac"
FEATURE_SCHEMA = 1


def quality_filename(feature_dim, whiten, normalize):
    return f"quality_k{feature_dim}_whiten{int(whiten)}_l2{int(normalize)}.json"


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _torchvision(dataset="cifar10"):
    try:
        import torch
        from torchvision.datasets import CIFAR10, STL10
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("DINOv2 image features require the vision dependencies") from exc
    return torch, (CIFAR10 if dataset == "cifar10" else STL10), Image


def _dataset_digest(train, test) -> str:
    digest = hashlib.sha256()
    for dataset in (train, test):
        digest.update(np.asarray(dataset.data, dtype=np.uint8).tobytes())
        labels = getattr(dataset, "targets", getattr(dataset, "labels", None))
        digest.update(np.asarray(labels, dtype=np.int64).tobytes())
    return digest.hexdigest()


def _extract(images, model, processor, *, torch, Image, device, batch_size, label):
    values = np.lib.format.open_memmap(
        label, mode="w+", dtype=np.float32, shape=(len(images), int(model.config.hidden_size)))
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            pil_images = [Image.fromarray(np.asarray(x, dtype=np.uint8), mode="RGB")
                          for x in images[start:start + batch_size]]
            batch = processor(images=pil_images, return_tensors="pt")
            output = model(pixel_values=batch["pixel_values"].to(device))
            cls = output.last_hidden_state[:, 0].float().cpu().numpy()
            if not np.isfinite(cls).all():
                raise ValueError("DINOv2 produced non-finite CLS embeddings")
            values[start:start + len(cls)] = cls
            if start == 0 or (start // batch_size) % 100 == 0:
                print(f"[DINOv2] {start + len(cls)}/{len(images)} {Path(label).name}",
                      flush=True)
    values.flush()
    del values


def prepare_cifar10_cache(data_root, cache_root, *, device="cuda", batch_size=16,
                          download=False, corruption_sigma=0.35, seed=0,
                          val_per_class=500, dataset="cifar10"):
    """Extract official labeled partition embeddings once, plus paired corruptions."""
    if batch_size < 1:
        raise ValueError("Extraction microbatch must be positive")
    torch, Dataset, Image = _torchvision(dataset)
    try:
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as exc:
        raise RuntimeError("Install the vision extra, including transformers, for DINOv2") from exc
    if dataset == "cifar10":
        raw_train = Dataset(data_root, train=True, download=download, transform=None)
        raw_test = Dataset(data_root, train=False, download=download, transform=None)
        train_images, test_images = raw_train.data, raw_test.data
    elif dataset == "stl10":
        raw_train = Dataset(data_root, split="train", download=download, transform=None)
        raw_test = Dataset(data_root, split="test", download=download, transform=None)
        train_images = np.transpose(raw_train.data, (0, 2, 3, 1))
        test_images = np.transpose(raw_test.data, (0, 2, 3, 1))
    else:
        raise ValueError("dataset must be cifar10 or stl10")
    train_labels = getattr(raw_train, "targets", getattr(raw_train, "labels", None))
    test_labels = getattr(raw_test, "targets", getattr(raw_test, "labels", None))
    data_digest = _dataset_digest(raw_train, raw_test)
    root = Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    metadata_path = root / "metadata.json"
    train_per_class, test_capacity = ((5000, 1000) if dataset == "cifar10" else (500, 800))
    expected = {"schema": FEATURE_SCHEMA, "dataset": dataset, "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION, "dataset_sha256": data_digest,
                "train_images": len(train_images), "test_images": len(test_images),
                "val_per_class": val_per_class,
                "feature_type": "last_hidden_state_cls", "feature_normalization": "none",
                "processor_resize_crop": "processor_defaults", "corruption_sigma": corruption_sigma,
                "corruption_seed": seed}
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(f"DINOv2 feature cache mismatch for {key}")
        for name in ("train_features.npy", "test_features.npy", "corrupted_test_features.npy",
                     "corrupted_validation_features.npy", "train_labels.npy", "test_labels.npy"):
            if not (root / name).is_file():
                raise ValueError(f"DINOv2 feature cache is incomplete: {name}")
        return metadata
    if any(root.iterdir()):
        raise FileExistsError(f"Refusing to reuse incomplete feature cache: {root}")

    processor = AutoImageProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
                                      use_safetensors=True).to(device).eval()
    from huggingface_hub import hf_hub_download
    checkpoint_path = hf_hub_download(MODEL_ID, "model.safetensors", revision=MODEL_REVISION)
    checkpoint_sha256 = _file_sha256(checkpoint_path)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    width = int(model.config.hidden_size)
    # Extract to temporary filenames and publish only after every array is complete.
    train_tmp = root / ".train_features.tmp.npy"
    test_tmp = root / ".test_features.tmp.npy"
    corrupt_tmp = root / ".corrupted_test_features.tmp.npy"
    corrupt_val_tmp = root / ".corrupted_validation_features.tmp.npy"
    _extract(train_images, model, processor, torch=torch, Image=Image, device=device,
             batch_size=batch_size, label=train_tmp)
    _extract(test_images, model, processor, torch=torch, Image=Image, device=device,
             batch_size=batch_size, label=test_tmp)

    rng = np.random.default_rng(seed)
    corrupted = np.empty_like(test_images)
    for i, image in enumerate(test_images):
        noise = rng.normal(0.0, corruption_sigma, size=image.shape)
        corrupted[i] = np.clip(
            np.rint((image.astype(np.float32) / 255.0 + noise) * 255),
            0, 255).astype(np.uint8)
    _extract(corrupted, model, processor, torch=torch, Image=Image, device=device,
             batch_size=batch_size, label=corrupt_tmp)
    _, validation_ids, _ = split_indices(train_labels, test_labels,
        seed=seed, train_per_class=train_per_class-val_per_class, val_per_class=val_per_class,
        test_per_class=test_capacity, dataset=dataset)
    corrupted_validation = np.empty((len(validation_ids), *train_images.shape[1:]), dtype=np.uint8)
    val_rng = np.random.default_rng(seed + 98317)
    for out_index, image_id in enumerate(validation_ids):
        image = train_images[image_id]
        noise = val_rng.normal(0.0, corruption_sigma, size=image.shape)
        corrupted_validation[out_index] = np.clip(
            np.rint((image.astype(np.float32) / 255.0 + noise) * 255),
            0, 255).astype(np.uint8)
    _extract(corrupted_validation, model, processor, torch=torch, Image=Image,
             device=device, batch_size=batch_size, label=corrupt_val_tmp)
    np.save(root / "train_labels.npy", np.asarray(train_labels, dtype=np.int64))
    np.save(root / "test_labels.npy", np.asarray(test_labels, dtype=np.int64))
    train_tmp.replace(root / "train_features.npy")
    test_tmp.replace(root / "test_features.npy")
    corrupt_tmp.replace(root / "corrupted_test_features.npy")
    corrupt_val_tmp.replace(root / "corrupted_validation_features.npy")
    metadata = {**expected, "feature_dim": width,
                "resolved_model_commit": getattr(model.config, "_commit_hash", MODEL_REVISION),
                "checkpoint_filename": Path(checkpoint_path).name,
                "checkpoint_sha256": checkpoint_sha256,
                "processor": processor.to_dict(), "device": device,
                "extraction_microbatch": batch_size,
                "files": ["train_features.npy", "test_features.npy",
                          "corrupted_test_features.npy", "corrupted_validation_features.npy",
                          "train_labels.npy", "test_labels.npy"]}
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metadata


def split_indices(train_labels, test_labels, *, seed=0, train_per_class=4500,
                   val_per_class=500, test_per_class=1000, dataset="cifar10"):
    """Deterministic stratified split within the official labeled partitions."""
    train_capacity, test_capacity = ((5000, 1000) if dataset == "cifar10" else (500, 800))
    if train_per_class + val_per_class > train_capacity or not 1 <= test_per_class <= test_capacity:
        raise ValueError(f"Requested split exceeds canonical {dataset} sizes")
    rng = np.random.default_rng(seed)
    train_ids, val_ids = [], []
    train_labels = np.asarray(train_labels)
    for cls in range(10):
        indices = np.flatnonzero(train_labels == cls)
        rng.shuffle(indices)
        val_ids.extend(indices[:val_per_class])
        train_ids.extend(indices[val_per_class:val_per_class + train_per_class])
    rng = np.random.default_rng(seed + 1001)
    test_ids = []
    test_labels = np.asarray(test_labels)
    for cls in range(10):
        indices = np.flatnonzero(test_labels == cls)
        rng.shuffle(indices)
        test_ids.extend(indices[:test_per_class])
    return np.asarray(train_ids), np.asarray(val_ids), np.asarray(test_ids)


def _project_splits(root, x_train, x_test, train_ids, val_ids, test_ids, *,
                    seed, feature_dim, whiten):
    if feature_dim >= x_train.shape[1]:
        if feature_dim != x_train.shape[1] or whiten:
            raise ValueError("PCA dimension must be below raw feature dimension")
        return x_train[train_ids], x_train[val_ids], x_test[test_ids]
    projection_root = Path(root) / "projections"
    projection_root.mkdir(exist_ok=True)
    name = f"seed{seed}_k{feature_dim}_whiten{int(whiten)}"
    folder = projection_root / name
    paths = [folder / f"{part}.npy" for part in
             ("train", "validation", "test", "corrupted_validation", "corrupted_test")]
    marker = folder / "metadata.json"
    if not marker.exists():
        if folder.exists() and any(folder.iterdir()):
            raise FileExistsError(f"Refusing incomplete PCA cache: {folder}")
        folder.mkdir(exist_ok=True)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=feature_dim, whiten=whiten, svd_solver="randomized",
                  random_state=seed)
        pca.fit(np.asarray(x_train[train_ids], dtype=np.float32))
        inputs = [x_train[train_ids], x_train[val_ids], x_test[test_ids],
                  np.load(Path(root) / "corrupted_validation_features.npy", mmap_mode="r"),
                  np.load(Path(root) / "corrupted_test_features.npy", mmap_mode="r")[test_ids]]
        for source, path in zip(inputs, paths):
            target = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                               shape=(len(source), feature_dim))
            for start in range(0, len(source), 4096):
                target[start:start+4096] = pca.transform(np.asarray(source[start:start+4096]))
            target.flush()
            del target
        marker.write_text(json.dumps({"seed": seed, "feature_dim": feature_dim,
            "whiten": whiten, "fit_indices_sha256": hashlib.sha256(
                np.asarray(train_ids, dtype=np.int64).tobytes()).hexdigest(),
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist()}, indent=2))
    else:
        config = json.loads(marker.read_text())
        expected_ids = hashlib.sha256(np.asarray(train_ids, dtype=np.int64).tobytes()).hexdigest()
        if (config["seed"] != seed or config["feature_dim"] != feature_dim
                or config["whiten"] != whiten or config["fit_indices_sha256"] != expected_ids):
            raise ValueError("PCA feature cache configuration mismatch")
        if any(not path.is_file() for path in paths):
            raise ValueError(f"Incomplete PCA cache: {folder}")
    return tuple(np.load(path, mmap_mode="r") for path in paths[:3])


def load_cifar10_features(cache_root, *, seed=0, train_per_class=4500,
                          val_per_class=500, test_per_class=1000,
                          feature_dim=768, whiten=False, normalize=True, dataset="cifar10"):
    root = Path(cache_root)
    metadata = json.loads((root / "metadata.json").read_text())
    if metadata.get("schema") != FEATURE_SCHEMA or metadata.get("model_revision") != MODEL_REVISION:
        raise ValueError("Unsupported or mismatched DINOv2 feature cache")
    x_train = np.load(root / "train_features.npy", mmap_mode="r")
    y_train = np.load(root / "train_labels.npy", mmap_mode="r")
    x_test = np.load(root / "test_features.npy", mmap_mode="r")
    y_test = np.load(root / "test_labels.npy", mmap_mode="r")
    if metadata.get("dataset", "cifar10") != dataset:
        raise ValueError("Feature cache belongs to a different dataset")
    train_ids, val_ids, test_ids = split_indices(y_train, y_test, seed=seed,
        train_per_class=train_per_class, val_per_class=val_per_class,
        test_per_class=test_per_class, dataset=dataset)
    split = {"seed": seed, "train_indices": train_ids.tolist(), "val_indices": val_ids.tolist(),
             "validation_indices": val_ids.tolist(),
             "test_indices": test_ids.tolist(), "dataset_version": f"torchvision.{dataset.upper()}",
             "feature_cache": str(root), "backbone": MODEL_ID,
             "backbone_revision": metadata["resolved_model_commit"]}
    (train_features, val_features, test_features) = _project_splits(
        root, x_train, x_test, train_ids, val_ids, test_ids, seed=seed,
        feature_dim=feature_dim, whiten=whiten)
    def normalized(values):
        values = np.asarray(values, dtype=np.float32).copy()
        if normalize:
            norms = np.linalg.norm(values, axis=1, keepdims=True)
            values = values / np.maximum(norms, 1e-12)
        return values
    return ((normalized(train_features), y_train[train_ids]),
            (normalized(val_features), y_train[val_ids]),
            (normalized(test_features), y_test[test_ids]), split, metadata)


def validation_halves(labels, seed=0):
    """Deterministically split validation into disjoint calibration and scoring halves."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed + 71237)
    development, scoring = [], []
    for cls in np.unique(labels):
        indices = np.flatnonzero(labels == cls)
        rng.shuffle(indices)
        midpoint = len(indices) // 2
        development.extend(indices[:midpoint])
        scoring.extend(indices[midpoint:])
    return np.asarray(development), np.asarray(scoring)


def load_corrupted_features(cache_root, split, *, partition="test", feature_dim=768,
                             whiten=False, normalize=True):
    root = Path(cache_root)
    if partition not in ("test", "validation"):
        raise ValueError("Corrupted feature partition must be test or validation")
    if feature_dim == 768:
        name = "corrupted_test_features.npy" if partition == "test" else "corrupted_validation_features.npy"
        values = np.load(root / name, mmap_mode="r")
        if partition == "test":
            values = values[split["test_indices"]]
    else:
        filename = "corrupted_test.npy" if partition == "test" else "corrupted_validation.npy"
        path = root / "projections" / f"seed{split['seed']}_k{feature_dim}_whiten{int(whiten)}" / filename
        values = np.load(path, mmap_mode="r")
    values = np.asarray(values, dtype=np.float32).copy()
    if normalize:
        values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    return values


def quality_report(cache_root, *, seed=0, train_per_class=4500,
                   val_per_class=500, test_per_class=1000, feature_dim=768,
                   whiten=False, normalize=True, include_test=False, dataset="cifar10"):
    """Train-only ridge probe and class/covariance diagnostics for frozen features."""
    from sklearn.metrics import accuracy_score, confusion_matrix
    (train_x, train_y), (val_x, val_y), (test_x, test_y), split, meta = load_cifar10_features(
        cache_root, seed=seed, train_per_class=train_per_class,
        val_per_class=val_per_class, test_per_class=test_per_class,
        feature_dim=feature_dim, whiten=whiten, normalize=normalize, dataset=dataset)
    mean = np.asarray(train_x).mean(axis=0)
    scale = np.asarray(train_x).std(axis=0)
    scale = np.maximum(scale, 1e-6)
    x = (np.asarray(train_x, dtype=np.float64) - mean) / scale
    target = np.eye(10, dtype=np.float64)[np.asarray(train_y, dtype=np.int64)]
    gram = x.T @ x
    alpha = 1.0
    coefficients = np.linalg.solve(gram + alpha * np.eye(gram.shape[0]), x.T @ target)
    def predict(part):
        return ((np.asarray(part, dtype=np.float64) - mean) / scale @ coefficients).argmax(axis=1)
    val_pred = predict(val_x)
    class_means = np.stack([np.asarray(train_x)[np.asarray(train_y) == cls].mean(0)
                            for cls in range(10)])
    global_mean = np.asarray(train_x).mean(0)
    between = np.square(class_means - global_mean).mean()
    within = np.mean([np.square(np.asarray(train_x)[np.asarray(train_y) == cls] - class_means[cls]).mean()
                      for cls in range(10)])
    covariance = np.cov(np.asarray(train_x, dtype=np.float64), rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance)
    cutoff = max(float(eigenvalues[-1]) * 1e-6, 1e-12)
    report = {"backbone": MODEL_ID, "backbone_revision": meta["resolved_model_commit"],
              "feature_dim": int(train_x.shape[1]), "pca_whiten": whiten,
              "l2_normalize": normalize, "train_examples": len(train_x),
              "validation_ridge_accuracy": float(accuracy_score(val_y, val_pred)),
              "validation_confusion_matrix": confusion_matrix(
                  val_y, val_pred, labels=np.arange(10)).tolist(),
              "between_within_variance_ratio": float(between / max(within, 1e-12)),
              "covariance_trace": float(eigenvalues.clip(min=0).sum()),
              "covariance_effective_rank": int(np.sum(eigenvalues > cutoff)),
              "feature_variance_min": float(np.min(np.asarray(train_x).var(0))),
              "feature_variance_max": float(np.max(np.asarray(train_x).var(0))),
              "validation_probe_predictions": {
                  "indices": split["val_indices"],
                  "labels": np.asarray(val_y).astype(int).tolist(),
                  "predictions": val_pred.astype(int).tolist()}}
    if include_test:
        test_pred = predict(test_x)
        report["test_ridge_accuracy"] = float(accuracy_score(test_y, test_pred))
        report["test_confusion_matrix"] = confusion_matrix(
            test_y, test_pred, labels=np.arange(10)).tolist()
        report["probe_predictions"] = {"test_indices": split["test_indices"],
            "labels": np.asarray(test_y).astype(int).tolist(),
            "predictions": test_pred.astype(int).tolist()}
    return report
