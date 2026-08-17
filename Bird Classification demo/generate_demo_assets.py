"""Generate all static assets for the bird-classifier portfolio demo.

The browser never loads PyTorch or model checkpoints. Run this script once after
the CUB dataset and the two linear-classifier checkpoints are available.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset


SEED = 597
SAMPLE_COUNT = 12
NEIGHBOR_COUNT = 5
TOP_CLASS_COUNT = 5
IMAGE_SIZE = 224
NORMALIZE = T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))


@dataclass(frozen=True)
class ImageRecord:
    image_id: int
    relative_path: str
    label: int
    is_train: bool


class RecordDataset(Dataset):
    def __init__(self, records: list[ImageRecord], image_root: Path):
        self.records = records
        self.image_root = image_root

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(self.image_root / record.relative_path) as image:
            tensor = image_to_tensor(image.convert("RGB"))
        return tensor, record.label, record.image_id


class DemoClassifier(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.encoder = models.resnet50(weights=None)
        if embedding_dim == 2048:
            self.encoder.fc = nn.Identity()
        else:
            self.encoder.fc = nn.Linear(self.encoder.fc.in_features, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, 200)

    def embeddings(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images).flatten(1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embeddings(images))


def image_to_crop(image: Image.Image) -> Image.Image:
    return T.Compose([T.Resize(256), T.CenterCrop(IMAGE_SIZE)])(image)


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    return NORMALIZE(T.ToTensor()(image_to_crop(image)))


def read_metadata(dataset_root: Path) -> tuple[list[ImageRecord], list[str]]:
    cub_root = dataset_root / "CUB_200_2011"
    required = ["images.txt", "image_class_labels.txt", "train_test_split.txt", "classes.txt"]
    missing = [name for name in required if not (cub_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"CUB metadata missing from {cub_root}: {', '.join(missing)}")

    def pairs(name: str) -> dict[int, str]:
        result = {}
        for line in (cub_root / name).read_text(encoding="utf-8").splitlines():
            key, value = line.split(maxsplit=1)
            result[int(key)] = value
        return result

    images = pairs("images.txt")
    labels = {key: int(value) - 1 for key, value in pairs("image_class_labels.txt").items()}
    split = {key: value == "1" for key, value in pairs("train_test_split.txt").items()}
    records = [ImageRecord(key, images[key], labels[key], split[key]) for key in sorted(images)]

    class_names = []
    for line in (cub_root / "classes.txt").read_text(encoding="utf-8").splitlines():
        _, raw_name = line.split(maxsplit=1)
        name = raw_name.split(".", 1)[-1].replace("_", " ")
        class_names.append(name)
    return records, class_names


def load_model(checkpoint: Path, embedding_dim: int, device: torch.device) -> DemoClassifier:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Classifier checkpoint not found: {checkpoint}")
    model = DemoClassifier(embedding_dim).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def predict_records(model: DemoClassifier, loader: DataLoader, device: torch.device) -> dict[int, dict]:
    predictions = {}
    with torch.inference_mode():
        for images, labels, image_ids in loader:
            logits = model(images.to(device))
            probabilities = logits.softmax(dim=1)
            values, indices = probabilities.topk(TOP_CLASS_COUNT, dim=1)
            for row, image_id in enumerate(image_ids.tolist()):
                predictions[image_id] = {
                    "label": int(labels[row]),
                    "top_indices": indices[row].cpu().tolist(),
                    "top_values": values[row].cpu().tolist(),
                }
    return predictions


def curate_samples(
    test_records: list[ImageRecord], predictions: dict[str, dict[int, dict]]
) -> list[ImageRecord]:
    rng = random.Random(SEED)
    categories: dict[str, list[ImageRecord]] = {
        "both_correct": [], "proxy_only": [], "simclr_only": [], "both_wrong": []
    }
    for record in test_records:
        sim_ok = predictions["simclr"][record.image_id]["top_indices"][0] == record.label
        proxy_ok = predictions["proxynca"][record.image_id]["top_indices"][0] == record.label
        key = "both_correct" if sim_ok and proxy_ok else "proxy_only" if proxy_ok else "simclr_only" if sim_ok else "both_wrong"
        categories[key].append(record)
    for records in categories.values():
        rng.shuffle(records)

    selected: list[ImageRecord] = []
    used_labels: set[int] = set()
    quotas = {"both_correct": 4, "proxy_only": 3, "simclr_only": 2, "both_wrong": 3}
    for key, quota in quotas.items():
        for record in categories[key]:
            if record.label not in used_labels:
                selected.append(record)
                used_labels.add(record.label)
                if sum(item in categories[key] for item in selected) >= quota:
                    break
    if len(selected) < SAMPLE_COUNT:
        remainder = list(test_records)
        rng.shuffle(remainder)
        for record in remainder:
            if record.image_id not in {item.image_id for item in selected} and record.label not in used_labels:
                selected.append(record)
                used_labels.add(record.label)
                if len(selected) == SAMPLE_COUNT:
                    break
    return sorted(selected[:SAMPLE_COUNT], key=lambda item: item.image_id)


def extract_embeddings(
    model: DemoClassifier, loader: DataLoader, device: torch.device
) -> tuple[torch.Tensor, list[int]]:
    chunks, image_ids = [], []
    with torch.inference_mode():
        for images, _, ids in loader:
            embeddings = F.normalize(model.embeddings(images.to(device)), dim=1)
            chunks.append(embeddings.cpu())
            image_ids.extend(ids.tolist())
    return torch.cat(chunks), image_ids


def gradcam(model: DemoClassifier, tensor: torch.Tensor, target: int, device: torch.device) -> np.ndarray:
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []
    layer = model.encoder.layer4[-1]
    forward_hook = layer.register_forward_hook(lambda _m, _i, output: activations.append(output))
    backward_hook = layer.register_full_backward_hook(lambda _m, _gi, go: gradients.append(go[0]))
    try:
        model.zero_grad(set_to_none=True)
        logits = model(tensor.unsqueeze(0).to(device))
        logits[0, target].backward()
        weights = gradients[0].mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations[0]).sum(dim=1).relu()
        cam = F.interpolate(cam.unsqueeze(1), size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)[0, 0]
        cam -= cam.min()
        cam /= cam.max().clamp_min(1e-8)
        return cam.detach().cpu().numpy()
    finally:
        forward_hook.remove()
        backward_hook.remove()


def heatmap_overlay(image: Image.Image, cam: np.ndarray) -> Image.Image:
    x = np.clip(cam, 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    green = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    blue = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    color = np.stack([red, green, blue], axis=-1)
    base = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    alpha = (0.18 + 0.42 * x)[..., None]
    overlay = np.clip(base * (1 - alpha) + color * alpha, 0, 1)
    return Image.fromarray((overlay * 255).astype(np.uint8))


def save_webp(image: Image.Image, path: Path, size: tuple[int, int], quality: int = 84) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    copy = image.copy()
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    copy.save(path, "WEBP", quality=quality, method=6)


def validate_manifest(manifest_path: Path) -> None:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert len(data["samples"]) == SAMPLE_COUNT
    sample_ids = {sample["id"] for sample in data["samples"]}
    assert len(sample_ids) == SAMPLE_COUNT
    for sample in data["samples"]:
        assert sample["split"] == "test"
        assert (manifest_path.parent / sample["image"]).is_file()
        for model_key in ("simclr", "proxynca"):
            result = sample["results"][model_key]
            assert len(result["top_classes"]) == TOP_CLASS_COUNT
            assert len(result["neighbors"]) == NEIGHBOR_COUNT
            assert result["predicted_class"] == result["top_classes"][0]["class_name"]
            probabilities = [item["probability"] for item in result["top_classes"]]
            assert all(0 <= value <= 1 for value in probabilities)
            assert probabilities == sorted(probabilities, reverse=True)
            assert (manifest_path.parent / result["heatmap"]).is_file()
            for neighbor in result["neighbors"]:
                assert neighbor["split"] == "train"
                assert (manifest_path.parent / neighbor["image"]).is_file()


def generate(args: argparse.Namespace) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = (repo_root / args.dataset_root).resolve()
    output = (repo_root / args.output).resolve()
    if repo_root not in output.parents:
        raise ValueError(f"Output must remain inside the repository: {output}")
    assets = output / "assets"
    if assets.exists():
        shutil.rmtree(assets)
    assets.mkdir(parents=True)

    records, class_names = read_metadata(dataset_root)
    image_root = dataset_root / "CUB_200_2011" / "images"
    train_records = [record for record in records if record.is_train]
    test_records = [record for record in records if not record.is_train]
    record_by_id = {record.image_id: record for record in records}
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Generating demo assets on {device} ({len(train_records)} train, {len(test_records)} test)")

    configs = {
        "simclr": {"checkpoint": repo_root / args.simclr_checkpoint, "embedding_dim": 2048},
        "proxynca": {"checkpoint": repo_root / args.proxynca_checkpoint, "embedding_dim": 64},
    }
    test_loader = DataLoader(RecordDataset(test_records, image_root), batch_size=args.batch_size, shuffle=False)
    predictions, loaded_models = {}, {}
    for key, config in configs.items():
        model = load_model(config["checkpoint"], config["embedding_dim"], device)
        predictions[key] = predict_records(model, test_loader, device)
        loaded_models[key] = model
        print(f"Predicted test split with {key}")

    selected = curate_samples(test_records, predictions)
    selected_ids = {record.image_id for record in selected}
    print(f"Selected test image IDs: {sorted(selected_ids)}")

    train_loader = DataLoader(RecordDataset(train_records, image_root), batch_size=args.batch_size, shuffle=False)
    retrievals: dict[str, dict[int, list[tuple[int, float]]]] = {key: {} for key in configs}
    for key, model in loaded_models.items():
        train_embeddings, train_ids = extract_embeddings(model, train_loader, device)
        selected_loader = DataLoader(RecordDataset(selected, image_root), batch_size=args.batch_size, shuffle=False)
        query_embeddings, query_ids = extract_embeddings(model, selected_loader, device)
        similarities = query_embeddings @ train_embeddings.T
        values, indices = similarities.topk(NEIGHBOR_COUNT, dim=1)
        for row, query_id in enumerate(query_ids):
            retrievals[key][query_id] = [
                (train_ids[index], float(values[row, rank]))
                for rank, index in enumerate(indices[row].tolist())
            ]
        print(f"Retrieved nearest training images with {key}")

    copied_neighbors: set[int] = set()
    samples_json = []
    for sample_number, record in enumerate(selected, start=1):
        source_path = image_root / record.relative_path
        with Image.open(source_path) as source:
            query_image = image_to_crop(source.convert("RGB"))
        query_relative = f"assets/queries/sample-{sample_number:02d}.webp"
        save_webp(query_image, output / query_relative, (640, 640), 88)
        tensor = image_to_tensor(query_image)
        results_json = {}
        for key, model in loaded_models.items():
            prediction = predictions[key][record.image_id]
            predicted_label = prediction["top_indices"][0]
            cam = gradcam(model, tensor, predicted_label, device)
            heatmap_relative = f"assets/heatmaps/{key}-sample-{sample_number:02d}.webp"
            save_webp(heatmap_overlay(query_image, cam), output / heatmap_relative, (640, 640), 88)
            neighbors_json = []
            for rank, (neighbor_id, similarity) in enumerate(retrievals[key][record.image_id], start=1):
                neighbor = record_by_id[neighbor_id]
                neighbor_relative = f"assets/neighbors/train-{neighbor_id:05d}.webp"
                if neighbor_id not in copied_neighbors:
                    with Image.open(image_root / neighbor.relative_path) as image:
                        save_webp(image.convert("RGB"), output / neighbor_relative, (360, 260), 82)
                    copied_neighbors.add(neighbor_id)
                neighbors_json.append({
                    "rank": rank,
                    "image_id": neighbor_id,
                    "class_id": neighbor.label,
                    "class_name": class_names[neighbor.label],
                    "similarity": round(similarity, 6),
                    "image": neighbor_relative,
                    "split": "train",
                })
            top_classes = [
                {"rank": rank, "class_id": label, "class_name": class_names[label], "probability": round(float(probability), 7)}
                for rank, (label, probability) in enumerate(zip(prediction["top_indices"], prediction["top_values"]), start=1)
            ]
            results_json[key] = {
                "predicted_class": class_names[predicted_label],
                "predicted_class_id": predicted_label,
                "confidence": top_classes[0]["probability"],
                "correct": predicted_label == record.label,
                "top_classes": top_classes,
                "heatmap": heatmap_relative,
                "neighbors": neighbors_json,
            }
        samples_json.append({
            "id": f"sample-{sample_number:02d}",
            "image_id": record.image_id,
            "source_path": record.relative_path,
            "split": "test",
            "image": query_relative,
            "true_class_id": record.label,
            "true_class": class_names[record.label],
            "results": results_json,
        })

    manifest = {
        "schema_version": 1,
        "title": "Learning to See Birds",
        "dataset": "Caltech-UCSD Birds-200-2011",
        "default_model": "proxynca",
        "default_sample": samples_json[0]["id"],
        "models": {
            "proxynca": {"name": "ProxyNCA", "description": "Supervised metric learning that pulls images toward learned class proxies.", "embedding_dimensions": 64, "top1_accuracy": 0.7974, "top5_accuracy": 0.9254},
            "simclr": {"name": "SimCLR", "description": "Self-supervised contrastive learning from paired augmentations of each bird image.", "embedding_dimensions": 2048, "top1_accuracy": 0.6049, "top5_accuracy": 0.8749},
        },
        "samples": samples_json,
    }
    manifest_path = output / "demo-data.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    validate_manifest(manifest_path)
    print(f"Wrote {manifest_path} with {len(copied_neighbors)} unique neighbor thumbnails")
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="cub2011")
    parser.add_argument("--simclr-checkpoint", default="simclr_linear_classifier_resnet50.pth")
    parser.add_argument("--proxynca-checkpoint", default="proxynca_linear_classifier_resnet50.pth")
    parser.add_argument("--output", default="demo/site")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.validate_only:
        root = Path(__file__).resolve().parents[1]
        validate_manifest(root / cli_args.output / "demo-data.json")
        print("Manifest and generated assets are valid")
    else:
        generate(cli_args)
