import torch
from torch.utils.data import DataLoader, random_split
import sys
import os
import pickle
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from Dataset import Dataset, train_transforms, test_transforms
from ResNet18 import get_model
from train import train, test


def _save_stats(stats, log_dir):
    filename = os.path.join(log_dir, "centralized_stats.pkl")
    with open(filename, "ab") as f:
        pickle.dump(stats, f)


def run_centralized_baseline(run_id,epochs):
    print(f"--- Starting Centralized Baseline (Run {run_id}) ---")

    # Create logs directory: Partie1/logs/centralized/{id}/
    log_dir = os.path.join("Partie1", "logs", "centralized", run_id)
    os.makedirs(log_dir, exist_ok=True)

    # Clean up previous file
    stats_path = os.path.join(log_dir, "centralized_stats.pkl")
    if os.path.exists(stats_path): os.remove(stats_path)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_set = Dataset(
        root_dir="Dataset",
        split="Train",
        transform=train_transforms
    )

    test_set = Dataset(
        root_dir="Dataset",
        split="Test",
        transform=test_transforms
    )

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    model = get_model()

    # Start training
    history = train(model, train_loader, epochs=epochs, device=device)

    print("Saving training stats..." + log_dir)
    for stat in history:
        _save_stats(stat, log_dir)

    # Evaluate model
    loss, accuracy = test(model, test_loader, device=device)
    print(f"CENTRALIZED RESULT -> Accuracy: {accuracy * 100:.2f}%")
    _save_stats({"type": "final_test", "loss": loss, "accuracy": accuracy}, log_dir)

    torch.save(model.state_dict(), os.path.join(log_dir, "baseline_centralized.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, default="0", help="Run ID")
    parser.add_argument("--epochs", type=int, default=20, help="Number of local epochs per round")
    args = parser.parse_args()

    run_centralized_baseline(args.run_id,args.epochs)