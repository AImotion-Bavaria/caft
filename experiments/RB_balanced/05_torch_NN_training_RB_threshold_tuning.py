import json
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

from src.torch_utils import torch_NN_CS_loss, torch_NN_datasets, torch_NN_models
from src.torch_utils import torch_NN_training, torch_NN_evaluation
from src.torch_utils.torch_NN_metrics import get_average_cost_metric
from src.utils.Utils import SaveOutput
from src.torch_utils.torch_device import get_device
import config.paths as paths

# --------------------------------------------------
# General Configuration
# --------------------------------------------------

NUM_CLASSES = 3  # 0=small, 1=large, 2=not repairable
EXPERIMENT_NAME = "Exp_1"

# --------------------------------------------------
# Random Blobs Training Configuration
# --------------------------------------------------
COST_MATRIX_RAW = torch.tensor([[0, 3, 13],
                               [2, 0, 12],
                               [6, 3, 0]], dtype=torch.float32) #CM1

LEARNING_RATE = 5e-5
TRAIN_VAL_TEST_SIZE = [0.6, 0.2, 0.2]
DATA_SPLIT_STATE = 0
BATCH_SIZE = 64
NUM_EPOCHS = 250

EARLY_STOPPING = True
PATIENCE = 25
MONITOR = 'val_loss'
MODE = 'min'

USE_CLASS_WEIGHTS = True

# --------------------------------------------------
# Phase 2: Threshold-Tuning Configuration
# --------------------------------------------------
MAX_FP_RATE = 0.005            # Max allowable false-class-2 prediction rate
THRESHOLD_STEPS = 200          # Number of threshold values to search

config_dict = {
    "COST_MATRIX_RAW": str(COST_MATRIX_RAW.tolist()),
    "LEARNING_RATE": LEARNING_RATE,
    "TRAIN_VAL_TEST_SIZE": TRAIN_VAL_TEST_SIZE,
    "DATA_SPLIT_STATE": DATA_SPLIT_STATE,
    "BATCH_SIZE": BATCH_SIZE,
    "NUM_EPOCHS": NUM_EPOCHS,
    "EARLY_STOPPING": EARLY_STOPPING,
    "PATIENCE": PATIENCE,
    "MONITOR": MONITOR,
    "MODE": MODE,
    "USE_CLASS_WEIGHTS": USE_CLASS_WEIGHTS,
    "MAX_FP_RATE": MAX_FP_RATE,
    "THRESHOLD_STEPS": THRESHOLD_STEPS,
    "METHOD": "Threshold-Tuning (post-hoc, no retraining)",
}


# ============================================================
# Helper Functions
# ============================================================

def false_class2_rate(model, dataloader, device="cpu"):
    """
    Compute the rate at which true class-0 and class-1 samples
    are incorrectly predicted as class 2.
    """
    false_as_2 = 0
    total_01 = 0
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            preds = torch.argmax(model(x), dim=1)
            mask = (y != 2)
            false_as_2 += ((preds == 2) & mask).sum().item()
            total_01 += mask.sum().item()
    return false_as_2 / max(total_01, 1)


def predict_with_threshold(model, dataloader, tau, device="cpu"):
    """
    Predict with a rejection threshold on class 2.
    Only predict class 2 if P(class=2) > tau, otherwise fall back
    to the best among class 0 and class 1.

    Returns:
        all_preds: tensor of predictions
        all_labels: tensor of true labels
    """
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            probs = F.softmax(model(x), dim=1)

            # Default: argmax over all classes
            preds = torch.argmax(probs, dim=1)

            # Override: where argmax = 2 but P(2) <= tau, pick best of {0, 1}
            is_class2 = (preds == 2)
            below_tau = (probs[:, 2] <= tau)
            override_mask = is_class2 & below_tau

            if override_mask.any():
                # Pick best among class 0 and 1 for overridden samples
                probs_01 = probs[override_mask, :2]
                preds[override_mask] = torch.argmax(probs_01, dim=1)

            all_preds.append(preds)
            all_labels.append(y)

    return torch.cat(all_preds), torch.cat(all_labels)


def compute_metrics_at_threshold(model, dataloader, tau, eval_cost_matrix, device="cpu"):
    """
    Compute FC2-Rate, accuracy, average cost, classification report
    and confusion matrix at a given threshold tau.
    """
    preds, labels = predict_with_threshold(model, dataloader, tau, device)
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()

    # FC2-Rate
    mask_01 = (labels != 2)
    false_as_2 = ((preds == 2) & mask_01).sum().item()
    total_01 = mask_01.sum().item()
    fc2_rate = false_as_2 / max(total_01, 1)

    # Accuracy
    accuracy = (preds == labels).float().mean().item() * 100

    # Confusion matrix & classification report (sklearn, same as evaluate_model)
    cm = confusion_matrix(labels_np, preds_np)
    report = classification_report(labels_np, preds_np, zero_division=0)

    # Average cost
    avg_cost = get_average_cost_metric(eval_cost_matrix, cm)

    # Class-2 recall: how many true class-2 are found
    true_2 = (labels == 2).sum().item()
    pred_2_correct = ((preds == 2) & (labels == 2)).sum().item()
    recall_class2 = pred_2_correct / max(true_2, 1)

    return {
        "tau": tau,
        "fc2_rate": fc2_rate,
        "accuracy": accuracy,
        "avg_cost": avg_cost.item() if torch.is_tensor(avg_cost) else avg_cost,
        "recall_class2": recall_class2,
        "cm": cm,
        "report": report,
    }


def print_threshold_metrics_summary(model, dataloader, tau, eval_cost_matrix, dataset_name, device="cpu"):
    """
    Print metrics summary in the same format as torch_NN_evaluation.print_metrics_summary,
    but using threshold-adjusted predictions for CM and classification report.
    AEC and RWWCE are computed on raw probabilities (independent of threshold).
    """
    from src.torch_utils.torch_NN_metrics import get_AEC_metric, get_RWWCE_metric

    metrics = compute_metrics_at_threshold(model, dataloader, tau, eval_cost_matrix, device)
    aec = get_AEC_metric(model, dataloader, eval_cost_matrix)
    rwwce = get_RWWCE_metric(model, dataloader, eval_cost_matrix)

    print("-" * 15 + f"{dataset_name} Metrics Summary" + "-" * 15)
    print(f"{dataset_name} Accuracy: {metrics['accuracy']:.3f}")
    print(f"Average cost: {metrics['avg_cost']:.3f}")
    print(f"AEC: {aec:.3f}")
    print(f"RWWCE: {rwwce:.3f}")
    print(f"FC2-Rate: {metrics['fc2_rate']:.4f}")
    print(f"Recall Class 2: {metrics['recall_class2']:.3f}")
    print(f"{dataset_name} Confusion Matrix: \n", metrics['cm'])
    print(f"{dataset_name} Classification Report: \n", metrics['report'])
    print("\n")


def find_optimal_threshold(model, val_loader, eval_cost_matrix, device="cpu"):
    """
    Grid search over tau in [0, 1] to find the threshold that:
    1. Satisfies FC2-Rate <= MAX_FP_RATE
    2. Minimizes average cost among feasible thresholds

    Returns:
        best_tau: optimal threshold
        search_df: DataFrame with metrics at each tau
    """
    taus = np.linspace(0.0, 1.0, THRESHOLD_STEPS + 1)
    results = []

    for tau in taus:
        metrics = compute_metrics_at_threshold(model, val_loader, tau, eval_cost_matrix, device)
        results.append({
            "tau": tau,
            "fc2_rate": metrics["fc2_rate"],
            "accuracy": metrics["accuracy"],
            "avg_cost": metrics["avg_cost"],
            "recall_class2": metrics["recall_class2"],
        })

    search_df = pd.DataFrame(results)

    # Find feasible thresholds (FC2-Rate constraint satisfied)
    feasible = search_df[search_df["fc2_rate"] <= MAX_FP_RATE]

    if len(feasible) > 0:
        # Among feasible: pick lowest avg_cost
        best_idx = feasible["avg_cost"].idxmin()
        best_tau = feasible.loc[best_idx, "tau"]
        print(f"Found {len(feasible)} feasible thresholds. "
              f"Best tau={best_tau:.4f} (AvgCost={feasible.loc[best_idx, 'avg_cost']:.3f}, "
              f"FC2={feasible.loc[best_idx, 'fc2_rate']:.4f}, "
              f"Recall2={feasible.loc[best_idx, 'recall_class2']:.3f})")
    else:
        # No feasible threshold → pick lowest FC2-rate
        best_idx = search_df["fc2_rate"].idxmin()
        best_tau = search_df.loc[best_idx, "tau"]
        print(f"WARNING: No threshold satisfies FC2-Rate <= {MAX_FP_RATE}. "
              f"Best tau={best_tau:.4f} (FC2={search_df.loc[best_idx, 'fc2_rate']:.4f})")

    return best_tau, search_df


def plot_threshold_search(search_df, best_tau, save_path=None):
    """
    Plot FC2-Rate, Average Cost, Accuracy, and Recall-Class-2 as function of tau.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Threshold Tuning — Class-2 Prediction Threshold Search (best tau={best_tau:.4f})", fontsize=14)

    # FC2-Rate
    ax = axes[0, 0]
    ax.plot(search_df["tau"], search_df["fc2_rate"], color="tab:red")
    ax.axhline(y=MAX_FP_RATE, color="gray", linestyle="--", label=f"max FC2={MAX_FP_RATE}")
    ax.axvline(x=best_tau, color="tab:green", linestyle="--", alpha=0.7, label=f"tau*={best_tau:.4f}")
    ax.set_xlabel("Threshold tau")
    ax.set_ylabel("FC2-Rate")
    ax.set_title("FC2-Rate vs. Threshold")
    ax.legend()
    ax.grid(True)

    # Average Cost
    ax = axes[0, 1]
    ax.plot(search_df["tau"], search_df["avg_cost"], color="tab:olive")
    ax.axvline(x=best_tau, color="tab:green", linestyle="--", alpha=0.7, label=f"tau*={best_tau:.4f}")
    ax.set_xlabel("Threshold tau")
    ax.set_ylabel("Average Cost")
    ax.set_title("Average Cost vs. Threshold")
    ax.legend()
    ax.grid(True)

    # Accuracy
    ax = axes[1, 0]
    ax.plot(search_df["tau"], search_df["accuracy"], color="tab:blue")
    ax.axvline(x=best_tau, color="tab:green", linestyle="--", alpha=0.7, label=f"tau*={best_tau:.4f}")
    ax.set_xlabel("Threshold tau")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs. Threshold")
    ax.legend()
    ax.grid(True)

    # Recall Class 2
    ax = axes[1, 1]
    ax.plot(search_df["tau"], search_df["recall_class2"], color="tab:purple")
    ax.axvline(x=best_tau, color="tab:green", linestyle="--", alpha=0.7, label=f"tau*={best_tau:.4f}")
    ax.set_xlabel("Threshold tau")
    ax.set_ylabel("Recall Class 2")
    ax.set_title("Recall Class 2 vs. Threshold")
    ax.legend()
    ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    DEVICE = get_device()

    for run in range(1, 8):
        torch.manual_seed(run)
        np.random.seed(run)

        DATA_PATH = paths.BASE_DIR / "artifacts" / "data" / "datasets" / "random_blobs_data_balanced.csv"
        RESULTS_PATH = paths.BASE_DIR / "artifacts" / "results" / "_random_blobs_balanced" / EXPERIMENT_NAME / f"DATA_SPLIT_STATE_{DATA_SPLIT_STATE}" / "test_runs_threshold_tuning"
        MODEL_PATH = paths.BASE_DIR / "artifacts" / "models" / "model_weights" / "_random_blobs_balanced" / EXPERIMENT_NAME / f"DATA_SPLIT_STATE_{DATA_SPLIT_STATE}"

        # Save config on first run
        if run == 1:
            RESULTS_PATH.mkdir(parents=True, exist_ok=True)
            with open(RESULTS_PATH / "experiment_config.json", 'w') as f:
                json.dump(config_dict, f, indent=4)

        # Redirect stdout
        sys.stdout = SaveOutput(RESULTS_PATH / f"RB_results_threshold_tuning_{run}.txt")

        # ---- Data ----
        dataset = torch_NN_datasets.CSVDataset(DATA_PATH, 'label')
        train_data, val_data, test_data = torch_NN_datasets.stratified_split(
            dataset, TRAIN_VAL_TEST_SIZE, random_state=DATA_SPLIT_STATE
        )

        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=False)
        test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)

        print(f"Length train data: {len(train_data)}, Num batches: {len(train_loader)}")

        # ---- Model ----
        x_sample, _ = dataset[0]
        input_dim = x_sample.shape[0]
        model = torch_NN_models.FeedForwardNNRandomBlobs(input_dim, NUM_CLASSES).to(DEVICE)

        # ============================================================
        # PHASE 1: Training with RWWCE + Early Stopping (best known setup)
        # ============================================================
        print("=" * 60)
        print(f"Run {run} -- PHASE 1: Training (Weighted CE)")
        print("=" * 60)
        print(f"Device: {DEVICE}")

        if USE_CLASS_WEIGHTS:
            class_weights = torch_NN_datasets.compute_class_weights(train_data, NUM_CLASSES)
            print("Class weights:", class_weights)
            ce_criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
        else:
            ce_criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

        if EARLY_STOPPING:
            early_stopping = torch_NN_training.EarlyStopping(
                patience=PATIENCE, monitor=MONITOR,
                mode=MODE, restore_best_weights=True
            )
        else:
            early_stopping = None

        _, _, _, _, phase1_history_df, _ = torch_NN_training.train_model(
            model, train_loader, val_loader,
            optimizer, ce_criterion,
            eval_cost_matrix=COST_MATRIX_RAW.to(DEVICE),
            training_cost_matrix=None,
            num_epochs=NUM_EPOCHS,
            early_stopping=early_stopping,
            device=DEVICE
        )

        # Evaluate Phase 1 (baseline with argmax)
        print("\n--- Phase 1 Evaluation (argmax, no threshold) ---")
        cost_matrix = COST_MATRIX_RAW.to(DEVICE)
        torch_NN_evaluation.print_metrics_summary(model, train_loader, ce_criterion, None, cost_matrix, "Train")
        torch_NN_evaluation.print_metrics_summary(model, val_loader, ce_criterion, None, cost_matrix, "Validation")
        torch_NN_evaluation.print_metrics_summary(model, test_loader, ce_criterion, None, cost_matrix, "Test")

        fc2_baseline = false_class2_rate(model, val_loader, DEVICE)
        print(f"Baseline FC2-Rate (val): {fc2_baseline:.4f}")

        # ============================================================
        # PHASE 2: Threshold Tuning (no retraining)
        # ============================================================
        print("\n" + "=" * 60)
        print(f"Run {run} -- PHASE 2: Threshold Tuning")
        print("=" * 60)
        print(f"Constraint: FC2-Rate <= {MAX_FP_RATE}")
        print(f"Search: {THRESHOLD_STEPS} threshold values in [0, 1]")
        print()

        # Find optimal threshold on validation set
        best_tau, search_df = find_optimal_threshold(model, val_loader, cost_matrix, DEVICE)

        # Evaluate with optimal threshold on all sets (same format as other experiments)
        print(f"\n--- Phase 2 Evaluation (tau={best_tau:.4f}) ---")
        print("--------------------------------------------------")
        print("Evaluation cost matrix (raw):\n", COST_MATRIX_RAW)
        print("--------------------------------------------------")

        print_threshold_metrics_summary(model, train_loader, best_tau, cost_matrix, "Train", DEVICE)
        print_threshold_metrics_summary(model, val_loader, best_tau, cost_matrix, "Validation", DEVICE)
        print_threshold_metrics_summary(model, test_loader, best_tau, cost_matrix, "Test", DEVICE)

        fc2_test = compute_metrics_at_threshold(model, test_loader, best_tau, cost_matrix, DEVICE)["fc2_rate"]
        print(f"\nTest FC2-Rate: {fc2_test:.4f} (constraint: <= {MAX_FP_RATE})")

        # ---- Restore stdout ----
        sys.stdout.file.close()
        sys.stdout = sys.stdout.terminal

        # ---- Save artifacts ----
        RESULTS_PATH.mkdir(parents=True, exist_ok=True)

        # Phase 1 history

        phase1_history_df.to_csv(
            RESULTS_PATH / f"RB_history_phase1_{run}.csv", index=False)
        torch_NN_evaluation.plot_training_metrics(
            phase1_history_df,
            save_path=RESULTS_PATH / f"RB_figures_phase1_{run}.png",
            Title="Phase 1: Training (Weighted CE)")

        # Threshold search results
        search_df.to_csv(
            RESULTS_PATH / f"RB_threshold_search_{run}.csv", index=False)
        plot_threshold_search(
            search_df, best_tau,
            save_path=RESULTS_PATH / f"RB_figures_threshold_search_{run}.png")

        # Model weights
        weights_dir = MODEL_PATH / "threshold_tuning"
        weights_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), weights_dir / f"RB_weights_{run}.pth")
        # Save optimal threshold alongside weights
        torch.save({"tau": best_tau}, weights_dir / f"RB_threshold_{run}.pth")

        print(f"Run {run} completed. tau={best_tau:.4f}, FC2-Rate(test)={fc2_test:.4f}")
