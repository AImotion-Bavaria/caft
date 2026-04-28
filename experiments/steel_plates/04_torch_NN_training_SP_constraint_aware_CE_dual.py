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
EXPERIMENT_NAME = "Exp_3"

# --------------------------------------------------
# Steel Plates Training Configuration
# --------------------------------------------------
COST_MATRIX_RAW = torch.tensor([[0, 3, 13],
                               [2, 0, 12],
                               [6, 3, 0]], dtype=torch.float32) #CM1

# COST_MATRIX_RAW = torch.tensor([[0, 1, 10],
#                                [1, 0, 10],
#                                [1, 1, 0]], dtype=torch.float32) #CM2

# COST_MATRIX_RAW = torch.tensor([[0, 1, 5],
#                                [1, 0, 3],
#                                [1, 1, 0]], dtype=torch.float32) #CM3

LEARNING_RATE = 5e-5
TRAIN_VAL_TEST_SIZE = [0.6, 0.2, 0.2]
DATA_SPLIT_STATE = 0
BATCH_SIZE = 64
NUM_EPOCHS = 500

EARLY_STOPPING = True
PATIENCE = 25
MONITOR = 'val_loss'
MODE = 'min'

USE_CLASS_WEIGHTS = False

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
    "USE_CLASS_WEIGHTS": USE_CLASS_WEIGHTS
}

# --------------------------------------------------
# Phase 2: Constraint-Aware Fine-Tuning Configuration
# --------------------------------------------------

MAX_FC2_RATE = 0.005           # Max allowable false-class-2 prediction rate
MAX_FP_0AS1_RATE = 0.02       # Max allowable rate of true-0 predicted as 1
FINETUNE_LR = 5e-6 #1e-6       # Learning rate for fine-tuning (must be very low to avoid collapse)
COST_INCREMENT = 2          # How much to increase lambda per cost step
INNER_PATIENCE = 20            # Epochs without improvement before raising cost
MAX_FINETUNE_EPOCHS = 1500      # Total epoch budget for constraint-aware fine-tuning

# Start from a zero cost matrix — costs are built up purely via Lagrangian multipliers
BASE_COST_MATRIX = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.float32)

constraint_config = {
    "MAX_FC2_RATE": MAX_FC2_RATE,
    "MAX_FP_0AS1_RATE": MAX_FP_0AS1_RATE,
    "FINETUNE_LR": FINETUNE_LR,
    "COST_INCREMENT": COST_INCREMENT,
    "INNER_PATIENCE": INNER_PATIENCE,
    "MAX_FINETUNE_EPOCHS": MAX_FINETUNE_EPOCHS,
    "BASE_COST_MATRIX": str(BASE_COST_MATRIX.tolist()),
    "CLASS_WEIGHTS": "eye (CE anchor + lambda-scaled cost penalty)",
    "CONSTRAINT": "FC2-Rate <= 0.005 AND FP_0as1-Rate <= 0.02",
}


# ============================================================
# Helper Functions
# ============================================================

def false_class2_rate(model, dataloader, device="cpu"):
    """
    Compute the rate at which true class-0 and class-1 samples
    are incorrectly predicted as class 2.

    FC2-Rate = |{x : y(x) in {0,1} AND y_hat(x) = 2}| / |{x : y(x) in {0,1}}|
    """
    false_as_2 = 0
    total_01 = 0
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            preds = torch.argmax(model(x), dim=1)
            mask = (y != 2)  # samples with true class 0 or 1
            false_as_2 += ((preds == 2) & mask).sum().item()
            total_01 += mask.sum().item()
    return false_as_2 / max(total_01, 1)


def false_0_as_1_rate(model, dataloader, device="cpu"):
    """
    Compute the rate at which true class-0 samples
    are incorrectly predicted as class 1.

    FP_0as1-Rate = |{x : y(x) = 0 AND y_hat(x) = 1}| / |{x : y(x) = 0}|
    """
    false_0_as_1 = 0
    total_0 = 0
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            preds = torch.argmax(model(x), dim=1)
            mask = (y == 0)
            false_0_as_1 += ((preds == 1) & mask).sum().item()
            total_0 += mask.sum().item()
    return false_0_as_1 / max(total_0, 1)


def build_cost_matrix(num_classes, lambda_fc2, lambda_0as1):
    """
    Build a cost matrix with two independent Lagrangian multipliers:
        C[0, 2] = lambda_fc2    (penalize class 0 predicted as 2)
        C[1, 2] = lambda_fc2    (penalize class 1 predicted as 2)
        C[0, 1] = lambda_0as1   (penalize class 0 predicted as 1)
    """
    C = torch.zeros((num_classes, num_classes), dtype=torch.float32)
    C[0, 2] = lambda_fc2
    C[1, 2] = lambda_fc2
    C[0, 1] = lambda_0as1
    return C


# ============================================================
# Phase 2: Constraint-Aware Fine-Tuning Loop
# ============================================================

def train_constraint_aware(model, train_loader, val_loader, optimizer, criterion,
                           eval_cost_matrix, num_classes=NUM_CLASSES, device="cpu"):
    """
    Stepwise cost-escalation fine-tuning with TWO independent constraints.
    Each constraint has its own lambda that is increased independently
    when the respective violation rate stagnates.

    Stopping:
      - Both constraints satisfied → stop
      - MAX_FINETUNE_EPOCHS reached → stop
    """
    lambda_fc2 = COST_INCREMENT
    lambda_0as1 = COST_INCREMENT

    # Best model tracking
    best_constrained_state = None
    best_constrained_total_violation = float('inf')
    best_constrained_cost = float('inf')

    best_fallback_state = None
    best_fallback_total_violation = float('inf')

    # Inner patience tracking per constraint
    best_fc2_at_level = float('inf')
    epochs_without_fc2_improvement = 0

    best_0as1_at_level = float('inf')
    epochs_without_0as1_improvement = 0

    # Per-epoch history
    history = {
        "epoch": [], "cost_step_fc2": [], "cost_step_0as1": [],
        "lambda_fc2": [], "lambda_0as1": [],
        "train_loss": [], "train_accuracy": [],
        "val_loss": [], "val_accuracy": [],
        "val_average_cost": [], "fc2_rate": [], "fp_0as1_rate": [],
    }

    total_epoch = 0
    cost_step_fc2 = 0
    cost_step_0as1 = 0
    constraint_met = False

    while total_epoch < MAX_FINETUNE_EPOCHS:
        training_cost_matrix = build_cost_matrix(num_classes, lambda_fc2, lambda_0as1).to(device)

        # --- Train one epoch ---
        model.train()
        total_loss = 0.0
        total_correct = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y, training_cost_matrix)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total_correct += (predicted == y).sum().item()

        total_epoch += 1
        train_loss = total_loss / len(train_loader)
        train_acc = total_correct / len(train_loader.dataset) * 100

        # --- Evaluate on validation set ---
        val_loss, val_acc, val_cm, _ = torch_NN_training.evaluate_model(
            model, val_loader, criterion, cost_matrix=training_cost_matrix
        )
        val_avg_cost = get_average_cost_metric(eval_cost_matrix, val_cm)
        fc2_rate = false_class2_rate(model, val_loader, device)
        fp_0as1_rate = false_0_as_1_rate(model, val_loader, device)

        fc2_satisfied = fc2_rate <= MAX_FC2_RATE
        fp0as1_satisfied = fp_0as1_rate <= MAX_FP_0AS1_RATE
        total_violation = fc2_rate + fp_0as1_rate

        # --- Log metrics ---
        history["epoch"].append(total_epoch)
        history["cost_step_fc2"].append(cost_step_fc2)
        history["cost_step_0as1"].append(cost_step_0as1)
        history["lambda_fc2"].append(lambda_fc2)
        history["lambda_0as1"].append(lambda_0as1)
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_average_cost"].append(val_avg_cost.item())
        history["fc2_rate"].append(fc2_rate)
        history["fp_0as1_rate"].append(fp_0as1_rate)

        print(
            f"[Epoch {total_epoch:3d}/{MAX_FINETUNE_EPOCHS}] "
            f"Loss={train_loss:.4f} | Acc={train_acc:.1f}% | "
            f"ValLoss={val_loss:.4f} | ValAcc={val_acc:.1f}% | "
            f"AvgCost={val_avg_cost:.3f} | "
            f"FC2={fc2_rate:.4f}{'*' if fc2_satisfied else ''} (max={MAX_FC2_RATE}) | "
            f"FP_0as1={fp_0as1_rate:.4f}{'*' if fp0as1_satisfied else ''} (max={MAX_FP_0AS1_RATE}) | "
            f"lam_fc2={lambda_fc2:.1f} lam_0as1={lambda_0as1:.1f}"
        )

        # --- Track best model ---
        both_satisfied = fc2_satisfied and fp0as1_satisfied
        if both_satisfied:
            if (total_violation < best_constrained_total_violation or
                    (total_violation == best_constrained_total_violation and val_avg_cost.item() < best_constrained_cost)):
                best_constrained_total_violation = total_violation
                best_constrained_cost = val_avg_cost.item()
                best_constrained_state = {k: v.clone() for k, v in model.state_dict().items()}
        elif total_violation < best_fallback_total_violation:
            best_fallback_total_violation = total_violation
            best_fallback_state = {k: v.clone() for k, v in model.state_dict().items()}

        # --- Check if both constraints satisfied ---
        if both_satisfied:
            print(f"  -> Both constraints satisfied! FC2={fc2_rate:.4f}, FP_0as1={fp_0as1_rate:.4f}")
            print(f"  Val CM:\n{val_cm}")
            constraint_met = True
            break

        # --- Inner patience for FC2 (only escalate if NOT yet satisfied) ---
        if not fc2_satisfied:
            if fc2_rate < best_fc2_at_level:
                best_fc2_at_level = fc2_rate
                epochs_without_fc2_improvement = 0
            else:
                epochs_without_fc2_improvement += 1

            if epochs_without_fc2_improvement >= INNER_PATIENCE:
                print(f"  -> FC2 stagnation (best={best_fc2_at_level:.4f}). "
                      f"lambda_fc2 {lambda_fc2:.1f} -> {lambda_fc2 + COST_INCREMENT:.1f}")
                lambda_fc2 += COST_INCREMENT
                cost_step_fc2 += 1
                best_fc2_at_level = float('inf')
                epochs_without_fc2_improvement = 0
        else:
            best_fc2_at_level = fc2_rate
            epochs_without_fc2_improvement = 0

        # --- Inner patience for FP_0as1 (only escalate if NOT yet satisfied) ---
        if not fp0as1_satisfied:
            if fp_0as1_rate < best_0as1_at_level:
                best_0as1_at_level = fp_0as1_rate
                epochs_without_0as1_improvement = 0
            else:
                epochs_without_0as1_improvement += 1

            if epochs_without_0as1_improvement >= INNER_PATIENCE:
                print(f"  -> FP_0as1 stagnation (best={best_0as1_at_level:.4f}). "
                      f"lambda_0as1 {lambda_0as1:.1f} -> {lambda_0as1 + COST_INCREMENT:.1f}")
                lambda_0as1 += COST_INCREMENT
                cost_step_0as1 += 1
                best_0as1_at_level = float('inf')
                epochs_without_0as1_improvement = 0
        else:
            best_0as1_at_level = fp_0as1_rate
            epochs_without_0as1_improvement = 0

    else:
        print(f"\n  -> Max fine-tuning epochs ({MAX_FINETUNE_EPOCHS}) reached. "
              f"FC2={fc2_rate:.4f}, FP_0as1={fp_0as1_rate:.4f}")

    # Restore best model
    if best_constrained_state is not None:
        model.load_state_dict(best_constrained_state)
        print(f"\nRestored best constrained model (total_violation={best_constrained_total_violation:.4f}, AvgCost={best_constrained_cost:.3f})")
    elif best_fallback_state is not None:
        model.load_state_dict(best_fallback_state)
        print(f"\nRestored best fallback model (total_violation={best_fallback_total_violation:.4f}, constraint not met)")
    else:
        print("\nWarning: No best model state was saved.")

    final_cost_matrix = build_cost_matrix(num_classes, lambda_fc2, lambda_0as1)
    history_df = pd.DataFrame(history)

    return model, lambda_fc2, lambda_0as1, final_cost_matrix, history_df


# ============================================================
# Plotting: Combined Phase 1 + Phase 2 Learning Curves
# ============================================================

def plot_two_phase_training(phase1_df, phase2_df, save_path=None):
    """
    Plot learning curves from both training phases side-by-side.
    Left column: Phase 1 (pre-training), right column: Phase 2 (fine-tuning).
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex='col')
    fig.suptitle("Phase 1: Pre-Training (CE) | Phase 2: Dual-Constraint Fine-Tuning (RWWCE)", fontsize=14)

    # --- Phase 1 (left column) ---
    ax = axes[0, 0]
    ax.plot(phase1_df["epoch"], phase1_df["train_loss"], label="Train Loss", color="tab:red")
    ax.plot(phase1_df["epoch"], phase1_df["val_loss"], label="Val Loss", color="tab:orange")
    ax.set_ylabel("Loss")
    ax.set_title("Phase 1: Loss")
    ax.legend()
    ax.grid(True)

    ax = axes[1, 0]
    ax.plot(phase1_df["epoch"], phase1_df["train_accuracy"], label="Train Acc", color="tab:blue")
    ax.plot(phase1_df["epoch"], phase1_df["val_accuracy"], label="Val Acc", color="tab:cyan")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Phase 1: Accuracy")
    ax.legend()
    ax.grid(True)

    ax = axes[2, 0]
    if "train_average_cost" in phase1_df.columns:
        ax.plot(phase1_df["epoch"], phase1_df["train_average_cost"], label="Train AvgCost", color="tab:green")
        ax.plot(phase1_df["epoch"], phase1_df["val_average_cost"], label="Val AvgCost", color="tab:olive")
    ax.set_ylabel("Average Cost")
    ax.set_xlabel("Epoch")
    ax.set_title("Phase 1: Average Cost")
    ax.legend()
    ax.grid(True)

    # --- Phase 2 (right column) ---
    ax = axes[0, 1]
    ax.plot(phase2_df["epoch"], phase2_df["train_loss"], label="Train Loss", color="tab:red", marker='o', markersize=3)
    ax.plot(phase2_df["epoch"], phase2_df["val_loss"], label="Val Loss", color="tab:orange", marker='o', markersize=3)
    ax.set_ylabel("Loss")
    ax.set_title("Phase 2: Loss")
    ax.legend()
    ax.grid(True)

    ax = axes[1, 1]
    ax.plot(phase2_df["epoch"], phase2_df["val_accuracy"], label="Val Acc", color="tab:cyan", marker='o', markersize=3)
    ax2 = ax.twinx()
    ax2.plot(phase2_df["epoch"], phase2_df["fc2_rate"], label="FC2-Rate", color="tab:purple", marker='s', markersize=3)
    ax2.plot(phase2_df["epoch"], phase2_df["fp_0as1_rate"], label="FP_0as1", color="tab:red", marker='^', markersize=3)
    ax2.axhline(y=MAX_FC2_RATE, color="tab:purple", linestyle="--", alpha=0.5, label=f"max_fc2={MAX_FC2_RATE}")
    ax2.axhline(y=MAX_FP_0AS1_RATE, color="tab:red", linestyle="--", alpha=0.5, label=f"max_0as1={MAX_FP_0AS1_RATE}")
    ax.set_ylabel("Accuracy (%)")
    ax2.set_ylabel("Violation Rate")
    ax.set_title("Phase 2: Accuracy & Constraints")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="best")
    ax.grid(True)

    ax = axes[2, 1]
    ax.plot(phase2_df["epoch"], phase2_df["val_average_cost"], label="Val AvgCost", color="tab:olive", marker='o', markersize=3)
    ax2 = ax.twinx()
    ax2.plot(phase2_df["epoch"], phase2_df["lambda_fc2"], label="lambda_fc2", color="tab:gray", marker='x', markersize=3)
    ax2.plot(phase2_df["epoch"], phase2_df["lambda_0as1"], label="lambda_0as1", color="tab:brown", marker='+', markersize=3)
    ax.set_ylabel("Average Cost")
    ax2.set_ylabel("lambda")
    ax.set_xlabel("Total Epoch (fine-tuning)")
    ax.set_title("Phase 2: Average Cost & Lambdas")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="best")
    ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()


# ============================================================
# Main: Two-Phase Training Pipeline
# ============================================================

if __name__ == "__main__":
    DEVICE = get_device()

    for run in range(1, 11):
        torch.manual_seed(run)
        np.random.seed(run)

        DATA_PATH = paths.BASE_DIR / "artifacts" / "data" / "datasets" / "steel_plates_3cls.csv"
        RESULTS_PATH = paths.BASE_DIR / "artifacts" / "results" / "_steel_plates" / EXPERIMENT_NAME / f"DATA_SPLIT_STATE_{DATA_SPLIT_STATE}" / "test_runs_constraint_aware_stepwise_CE"
        MODEL_PATH = paths.BASE_DIR / "artifacts" / "models" / "model_weights" / "_steel_plates" / EXPERIMENT_NAME / f"DATA_SPLIT_STATE_{DATA_SPLIT_STATE}"

        # Save config on first run & create folder
        if run == 1:
            RESULTS_PATH.mkdir(parents=True, exist_ok=True)
            full_config = {**config_dict, **constraint_config}
            with open(RESULTS_PATH / "experiment_config.json", 'w') as f:
                json.dump(full_config, f, indent=4)
            with open(RESULTS_PATH / "constraint_aware_config.json", 'w') as f:
                json.dump(constraint_config, f, indent=4)

        # Redirect stdout to file + console
        sys.stdout = SaveOutput(RESULTS_PATH / f"_steel_plates_results_constraint_aware_stepwise_{run}.txt")

        # get train, test and validation data (stratified to preserve class distribution)
        dataset = torch_NN_datasets.CSVDatasetScaled(DATA_PATH, 'label')
        train_data, val_data, test_data = torch_NN_datasets.stratified_split(
            dataset, TRAIN_VAL_TEST_SIZE, random_state=DATA_SPLIT_STATE
        )

        # Scaler fitten - MUSS vor DataLoader-Erstellung passieren
        dataset.fit_scaler(train_data.indices)

        # Kurzer Check ob es geklappt hat:
        print("Nach Skalierung:")
        print(f"  Mean: {dataset.X.mean(dim=0)[:5]}")
        print(f"  Std:  {dataset.X.std(dim=0)[:5]}")

        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=False)
        test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)

        print(f"Length train data: {len(train_data)}, Num batches: {len(train_loader)}")

        # ---- Model ----
        x_sample, _ = dataset[0]
        input_dim = x_sample.shape[0]
        model = torch_NN_models.FeedForwardNNSP(input_dim, NUM_CLASSES).to(DEVICE)

        # ============================================================
        # PHASE 1: Pre-Training with Cross-Entropy + Early Stopping
        # ============================================================
        print("=" * 60)
        print(f"Run {run} -- PHASE 1: Pre-Training (CrossEntropy)")
        print("=" * 60)
        print(f"Device: {DEVICE}")

        if USE_CLASS_WEIGHTS:
            class_weights = torch_NN_datasets.compute_class_weights(train_data, NUM_CLASSES)
            print("Class weights:", class_weights)
            ce_criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
        else:
            ce_criterion = nn.CrossEntropyLoss()

        optimizer_phase1 = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

        if EARLY_STOPPING:
            early_stopping = torch_NN_training.EarlyStopping(
                patience=PATIENCE, monitor=MONITOR,
                mode=MODE, restore_best_weights=True
            )
        else:
            early_stopping = None

        _, _, _, _, phase1_history_df, _ = torch_NN_training.train_model(
            model, train_loader, val_loader,
            optimizer_phase1, ce_criterion,
            eval_cost_matrix=COST_MATRIX_RAW.to(DEVICE),
            training_cost_matrix=None,
            num_epochs=NUM_EPOCHS,
            early_stopping=early_stopping,
            device=DEVICE
        )

        # Evaluate Phase 1 model
        print("\n--- Phase 1 Evaluation ---")
        cost_matrix = COST_MATRIX_RAW.to(DEVICE)
        torch_NN_evaluation.print_metrics_summary(model, train_loader, ce_criterion, None, cost_matrix, "Train")
        torch_NN_evaluation.print_metrics_summary(model, val_loader, ce_criterion, None, cost_matrix, "Validation")
        torch_NN_evaluation.print_metrics_summary(model, test_loader, ce_criterion, None, cost_matrix, "Test")

        fc2_phase1 = false_class2_rate(model, val_loader, DEVICE)
        fp_0as1_phase1 = false_0_as_1_rate(model, val_loader, DEVICE)
        print(f"Phase 1 FC2-Rate (val): {fc2_phase1:.4f}")
        print(f"Phase 1 FP_0as1-Rate (val): {fp_0as1_phase1:.4f}")

        # Save Phase 1 weights
        pretrain_weights_dir = MODEL_PATH / "pretrained"
        pretrain_weights_dir.mkdir(parents=True, exist_ok=True)
        pretrain_weights_path = pretrain_weights_dir / f"SP_pretrained_weights_{run}.pth"
        torch.save(model.state_dict(), pretrain_weights_path)
        print(f"Phase 1 weights saved to: {pretrain_weights_path}")

        # ============================================================
        # PHASE 2: Constraint-Aware Fine-Tuning (RWWCE, class_weights=0)
        # ============================================================
        print("\n" + "=" * 60)
        print(f"Run {run} -- PHASE 2: Dual-Constraint Fine-Tuning (RWWCE)")
        print("=" * 60)
        print(f"Constraint 1: FC2-Rate <= {MAX_FC2_RATE}")
        print(f"Constraint 2: FP_0as1-Rate <= {MAX_FP_0AS1_RATE}")
        print(f"Initial cost matrix (zeros): costs are built via Lagrangian multipliers")
        print(f"Fine-tuning LR: {FINETUNE_LR}")
        print()

        # Reload Phase 1 weights to ensure clean starting point
        model.load_state_dict(torch.load(pretrain_weights_path, weights_only=True, map_location=DEVICE))

        # --- DEBUG: Evaluate loaded weights BEFORE fine-tuning ---
        print("=" * 60)
        print("DEBUG: Evaluation with loaded Phase 1 weights BEFORE fine-tuning")
        print("=" * 60)
        _, debug_acc_val, debug_cm_val, _ = torch_NN_training.evaluate_model(
            model, val_loader, ce_criterion, cost_matrix=None)
        print(f"Val Accuracy: {debug_acc_val:.3f}")
        print(f"Val Confusion Matrix:\n{debug_cm_val}")
        debug_avg_cost_val = get_average_cost_metric(COST_MATRIX_RAW, debug_cm_val)
        print(f"Val Average Cost: {debug_avg_cost_val:.3f}")
        _, debug_acc_test, debug_cm_test, _ = torch_NN_training.evaluate_model(
            model, test_loader, ce_criterion, cost_matrix=None)
        print(f"Test Accuracy: {debug_acc_test:.3f}")
        print(f"Test Confusion Matrix:\n{debug_cm_test}")
        debug_avg_cost_test = get_average_cost_metric(COST_MATRIX_RAW, debug_cm_test)
        print(f"Test Average Cost: {debug_avg_cost_test:.3f}")
        debug_fc2 = false_class2_rate(model, val_loader, DEVICE)
        debug_0as1 = false_0_as_1_rate(model, val_loader, DEVICE)
        print(f"Val FC2-Rate: {debug_fc2:.4f}")
        print(f"Val FP_0as1-Rate: {debug_0as1:.4f}")
        print("=" * 60)

        # RWWCE with eye class weights: CE anchor + lambda-scaled cost penalty
        eye_class_weights = torch.eye(NUM_CLASSES, dtype=torch.float32).to(DEVICE)
        rwwce_criterion = torch_NN_CS_loss.RWWCE(
            class_weights=eye_class_weights,
            cost_matrix_normalized=False
        )

        # Fresh optimizer with very low LR for fine-tuning
        optimizer_phase2 = torch.optim.AdamW(model.parameters(), lr=FINETUNE_LR)

        model, final_lambda_fc2, final_lambda_0as1, final_cost_matrix, phase2_history_df = train_constraint_aware(
            model, train_loader, val_loader, optimizer_phase2, rwwce_criterion,
            eval_cost_matrix=COST_MATRIX_RAW,
            num_classes=NUM_CLASSES,
            device=DEVICE
        )

        print(f"\nFinal lambda_fc2 = {final_lambda_fc2:.2f}")
        print(f"Final lambda_0as1 = {final_lambda_0as1:.2f}")
        print(f"Final cost matrix:\n{final_cost_matrix}")

        # ---- Final Evaluation (Phase 2 model) ----
        print("\n--- Phase 2 Final Evaluation ---")
        print("--------------------------------------------------")
        print("Evaluation cost matrix (raw):\n", COST_MATRIX_RAW)
        print("Evaluation cost matrix (normalized):\n", COST_MATRIX_RAW / COST_MATRIX_RAW.sum())
        print("--------------------------------------------------")

        torch_NN_evaluation.print_metrics_summary(
            model, train_loader, rwwce_criterion, final_cost_matrix, cost_matrix, "Train")
        torch_NN_evaluation.print_metrics_summary(
            model, val_loader, rwwce_criterion, final_cost_matrix, cost_matrix, "Validation")
        torch_NN_evaluation.print_metrics_summary(
            model, test_loader, rwwce_criterion, final_cost_matrix, cost_matrix, "Test")

        fc2_test = false_class2_rate(model, test_loader, DEVICE)
        fp_0as1_test = false_0_as_1_rate(model, test_loader, DEVICE)
        print(f"Test FC2-Rate: {fc2_test:.4f} (constraint: <= {MAX_FC2_RATE})")
        print(f"Test FP_0as1-Rate: {fp_0as1_test:.4f} (constraint: <= {MAX_FP_0AS1_RATE})")

        # ---- Restore stdout ----
        sys.stdout.file.close()
        sys.stdout = sys.stdout.terminal

        # ---- Save all artifacts ----
        RESULTS_PATH.mkdir(parents=True, exist_ok=True)

        # Phase 1 history
        phase1_history_df.to_csv(
            RESULTS_PATH / f"_steel_plates_history_phase1_CE_{run}.csv", index=False)
        torch_NN_evaluation.plot_training_metrics(
            phase1_history_df,
            save_path=RESULTS_PATH / f"_steel_plates_figures_phase1_CE_{run}.png",
            Title="Phase 1: Pre-Training (CE)")

        # Phase 2 history
        phase2_history_df.to_csv(
            RESULTS_PATH / f"_steel_plates_history_phase2_constraint_{run}.csv", index=False)

        # Combined two-phase plot
        plot_two_phase_training(
            phase1_history_df, phase2_history_df,
            save_path=RESULTS_PATH / f"_steel_plates_figures_combined_{run}.png"
        )

        # Phase 2 model weights
        finetune_weights_dir = MODEL_PATH / "constraint_aware"
        finetune_weights_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(),
                    finetune_weights_dir / f"_steel_plates_constraint_aware_weights_{run}.pth")

        print(f"Run {run} completed. FC2(test)={fc2_test:.4f}, FP_0as1(test)={fp_0as1_test:.4f}, "
              f"lambda_fc2={final_lambda_fc2:.2f}, lambda_0as1={final_lambda_0as1:.2f}")