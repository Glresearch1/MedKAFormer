import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from tensorboardX import SummaryWriter
from tqdm import trange

from kvasir.kvasir_dataset import create_kvasir_dataloaders
from kvasir.kvasir_utils import evaluate
from medkanformer import medkaformer_base, medkaformer_small, medkaformer_tiny


MODEL_BUILDERS = {
    "tiny": medkaformer_tiny,
    "small": medkaformer_small,
    "base": medkaformer_base,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train MedKAFormer on Kvasir-style image folders.")
    parser.add_argument("--train-dir", required=True, help="Path to the training image folder.")
    parser.add_argument("--test-dir", required=True, help="Path to the test image folder.")
    parser.add_argument("--val-dir", default=None, help="Optional validation image folder. Defaults to test-dir.")
    parser.add_argument("--output-dir", default="kvasir_output", help="Directory for checkpoints and logs.")
    parser.add_argument("--model", default="tiny", choices=sorted(MODEL_BUILDERS), help="MedKAFormer model size.")
    parser.add_argument(
        "--num-classes",
        default=None,
        type=int,
        help="Number of classes. Defaults to train folder count.",
    )
    parser.add_argument("--num-epochs", default=15, type=int, help="Number of training epochs.")
    parser.add_argument("--batch-size", default=32, type=int, help="Batch size.")
    parser.add_argument("--num-workers", default=4, type=int, help="DataLoader worker count.")
    parser.add_argument("--lr", default=1e-3, type=float, help="Initial learning rate.")
    parser.add_argument("--weight-decay", default=1e-4, type=float, help="Optimizer weight decay.")
    parser.add_argument("--resize-size", default=256, type=int, help="Resize image edge before center crop.")
    parser.add_argument("--image-size", default=224, type=int, help="Final image size after crop.")
    parser.add_argument("--drop-last", action="store_true", help="Drop the last incomplete training batch.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--device", default="auto", help="Device to use: auto, cuda, cpu, or mps.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint for evaluation or fine-tuning.")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate the checkpoint/model.")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested_device):
    if requested_device != "auto":
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(model_name, num_classes):
    return MODEL_BUILDERS[model_name](pretrained=False, num_classes=num_classes)


def clone_state_dict(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("net", checkpoint))
    model.load_state_dict(state_dict)
    return checkpoint


def validate_class_mappings(train_loader, val_loader, test_loader):
    expected = train_loader.dataset.class_to_idx
    for split_name, loader in [("val", val_loader), ("test", test_loader)]:
        if loader.dataset.class_to_idx != expected:
            raise ValueError(
                f"{split_name} class folders do not match train folders. "
                f"Train mapping: {expected}; {split_name} mapping: {loader.dataset.class_to_idx}"
            )


def specificity_score(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    total = cm.sum()
    scores = []

    for class_idx in range(num_classes):
        true_positive = cm[class_idx, class_idx]
        false_positive = cm[:, class_idx].sum() - true_positive
        false_negative = cm[class_idx, :].sum() - true_positive
        true_negative = total - true_positive - false_positive - false_negative
        denom = true_negative + false_positive
        scores.append(true_negative / denom if denom else 0.0)

    return float(np.mean(scores))


def train_one_epoch(model, train_loader, criterion, optimizer, device, writer, global_step):
    model.train()
    total_loss = []

    for inputs, targets in train_loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).long()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())
        writer.add_scalar("train/loss_step", loss.item(), global_step)
        global_step += 1

    return float(np.mean(total_loss)), global_step


def evaluate_model(model, data_loader, criterion, device, num_classes):
    model.eval()
    total_loss = []
    y_score_chunks = []
    y_true_chunks = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).long()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss.append(loss.item())
            y_score_chunks.append(torch.softmax(outputs, dim=1).detach().cpu())
            y_true_chunks.append(targets.detach().cpu())

    y_score = torch.cat(y_score_chunks, dim=0).numpy()
    y_true = torch.cat(y_true_chunks, dim=0).numpy()
    auc, acc = evaluate(y_score, y_true)
    y_pred = np.argmax(y_score, axis=1)

    return {
        "loss": float(np.mean(total_loss)),
        "auc": float(auc),
        "acc": float(acc),
        "precision": float(
            precision_score(y_true, y_pred, labels=list(range(num_classes)), average="macro", zero_division=0)
        ),
        "sensitivity": float(
            recall_score(y_true, y_pred, labels=list(range(num_classes)), average="macro", zero_division=0)
        ),
        "specificity": specificity_score(y_true, y_pred, num_classes),
        "f1": float(f1_score(y_true, y_pred, labels=list(range(num_classes)), average="macro", zero_division=0)),
    }


def metric_for_selection(metrics):
    return metrics["auc"] if not math.isnan(metrics["auc"]) else metrics["acc"]


def write_metrics(writer, split, metrics, epoch):
    for key, value in metrics.items():
        writer.add_scalar(f"{split}/{key}", value, epoch)


def format_metrics(split, metrics):
    return (
        f"{split} "
        f"loss: {metrics['loss']:.5f} "
        f"auc: {metrics['auc']:.5f} "
        f"acc: {metrics['acc']:.5f} "
        f"precision: {metrics['precision']:.5f} "
        f"sensitivity: {metrics['sensitivity']:.5f} "
        f"specificity: {metrics['specificity']:.5f} "
        f"f1: {metrics['f1']:.5f}"
    )


def save_run_config(output_root, args, class_to_idx, device):
    config = vars(args).copy()
    config["class_to_idx"] = class_to_idx
    config["device"] = str(device)

    with open(output_root / "config.json", "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, sort_keys=True)


def main(args):
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_root = Path(args.output_dir) / time.strftime("%y%m%d_%H%M%S")
    output_root.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = create_kvasir_dataloaders(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        test_dir=args.test_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        resize_size=args.resize_size,
        image_size=args.image_size,
        pin_memory=device.type == "cuda",
        drop_last=args.drop_last,
    )
    validate_class_mappings(train_loader, val_loader, test_loader)

    inferred_classes = len(train_loader.dataset.classes)
    num_classes = args.num_classes or inferred_classes
    if num_classes != inferred_classes:
        print(f"Warning: --num-classes={num_classes} but train folder contains {inferred_classes} classes.")

    save_run_config(output_root, args, train_loader.dataset.class_to_idx, device)
    print(f"Using device: {device}")
    print(f"Classes: {train_loader.dataset.class_to_idx}")
    print(f"Outputs: {output_root}")

    model = build_model(args.model, num_classes=num_classes).to(device)
    loaded_checkpoint = None
    if args.checkpoint:
        loaded_checkpoint = load_checkpoint(model, args.checkpoint, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    warmup_epochs = 2
    cosine_scale = 0.5

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return 0.9 * epoch / warmup_epochs + 0.1
        cosine_value = cosine_scale * (
            1 + math.cos(math.pi * (epoch - warmup_epochs) / max(args.num_epochs - warmup_epochs, 1))
        )
        return max(cosine_value, 0.1)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    writer = SummaryWriter(log_dir=str(output_root / "Tensorboard_Results"))
    best_state = clone_state_dict(model)
    best_epoch = int(loaded_checkpoint.get("epoch", 0)) if isinstance(loaded_checkpoint, dict) else 0
    best_score = (
        float(loaded_checkpoint.get("best_score", float("-inf")))
        if isinstance(loaded_checkpoint, dict)
        else float("-inf")
    )
    global_step = 0

    if not args.eval_only and args.num_epochs > 0:
        for epoch in trange(args.num_epochs, desc="Training"):
            train_loss, global_step = train_one_epoch(
                model, train_loader, criterion, optimizer, device, writer, global_step
            )
            train_metrics = evaluate_model(model, train_loader, criterion, device, num_classes)
            val_metrics = evaluate_model(model, val_loader, criterion, device, num_classes)
            test_metrics = evaluate_model(model, test_loader, criterion, device, num_classes)
            scheduler.step()

            train_metrics["loss"] = train_loss
            write_metrics(writer, "train", train_metrics, epoch)
            write_metrics(writer, "val", val_metrics, epoch)
            write_metrics(writer, "test", test_metrics, epoch)

            score = metric_for_selection(val_metrics)
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_state = clone_state_dict(model)
                print(f"New best validation score: {best_score:.5f} at epoch {best_epoch}")

    model.load_state_dict({key: value.to(device) for key, value in best_state.items()})
    final_train_metrics = evaluate_model(model, train_loader, criterion, device, num_classes)
    final_val_metrics = evaluate_model(model, val_loader, criterion, device, num_classes)
    final_test_metrics = evaluate_model(model, test_loader, criterion, device, num_classes)
    if best_score == float("-inf"):
        best_score = metric_for_selection(final_val_metrics)

    checkpoint = {
        "epoch": best_epoch,
        "best_score": best_score,
        "model_name": args.model,
        "model_state_dict": best_state,
        "class_to_idx": train_loader.dataset.class_to_idx,
        "args": vars(args),
    }
    torch.save(checkpoint, output_root / "best_model_kvasir.pth")

    lines = [
        f"best_epoch: {best_epoch}",
        f"best_validation_score: {best_score:.5f}",
        format_metrics("train", final_train_metrics),
        format_metrics("val", final_val_metrics),
        format_metrics("test", final_test_metrics),
    ]
    metrics_text = "\n".join(lines)
    print(metrics_text)

    with open(output_root / "metrics.txt", "w", encoding="utf-8") as file:
        file.write(metrics_text + "\n")
    with open(output_root / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "best_epoch": best_epoch,
                "best_validation_score": best_score,
                "train": final_train_metrics,
                "val": final_val_metrics,
                "test": final_test_metrics,
            },
            file,
            indent=2,
            sort_keys=True,
        )

    writer.close()
    return output_root


if __name__ == "__main__":
    main(parse_args())
