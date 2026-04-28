import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from tqdm import tqdm
import pandas as pd

from src.torch_utils import torch_NN_CS_loss, torch_NN_datasets, torch_NN_metrics, torch_NN_models
from src.utils.Utils import SaveOutput
from src.torch_utils import torch_NN_training
from src.torch_utils import torch_NN_evaluation
from src.torch_utils.torch_device import get_device

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
from pathlib import Path
import config.paths as paths

# --------------------------------------------------
# Experiment Configuration
# --------------------------------------------------

# Cost matrices (selectable via EXP_COST_MATRIX env var)
COST_MATRICES = {
    "CM1": torch.tensor([[0, 3, 13], [2, 0, 12], [6, 3, 0]], dtype=torch.float32),
    "CM2": torch.tensor([[0, 1, 10], [1, 0, 10], [1, 1, 0]], dtype=torch.float32),
    "CM3": torch.tensor([[0, 1, 5], [1, 0, 3], [1, 1, 0]], dtype=torch.float32),
}
COST_MATRIX_NAME = os.environ.get("EXP_COST_MATRIX", "CM1")
COST_MATRIX_RAW = COST_MATRICES[COST_MATRIX_NAME]

# model training parameters
LEARNING_RATE = 5e-5
TRAIN_VAL_TEST_SIZE = [0.6, 0.2, 0.2]
DATA_SPLIT_STATE = 0
BATCH_SIZE = 64
NUM_EPOCHS = 500

EARLY_STOPPING = True
PATIENCE = 25
MONITOR = 'val_loss'
MODE = 'min'

config_dict = {
    "COST_MATRIX_RAW": str(COST_MATRIX_RAW.tolist()),
    "LEARNING_RATE": LEARNING_RATE,
    "TRAIN_VAL_TEST_SIZE": TRAIN_VAL_TEST_SIZE,
    "DATA_SPLIT_STATE": DATA_SPLIT_STATE,
    "COST_MATRIX_NAME": COST_MATRIX_NAME,
    "BATCH_SIZE": BATCH_SIZE,
    "NUM_EPOCHS": NUM_EPOCHS,
    "EARLY_STOPPING": EARLY_STOPPING,
    "PATIENCE": PATIENCE,
    "MONITOR": MONITOR,
    "MODE": MODE
}


if __name__ == "__main__":

    device = get_device()
    print(f"Using device: {device}")

    for run in range(1, 11):
        torch.manual_seed(run)
        np.random.seed(run)
        EXPERIMENT_NAME = "Exp_1"
        DATA_PATH = paths.BASE_DIR / "artifacts" / "data" / "datasets" / "steel_plates_3cls.csv"
        RESULTS_PATH = paths.BASE_DIR / "artifacts" / "results" / "_steel_plates" / EXPERIMENT_NAME / f"DATA_SPLIT_STATE_{DATA_SPLIT_STATE}" / f"test_runs_RWWCE3_{COST_MATRIX_NAME}" 
        MODEL_PATH = paths.BASE_DIR / "artifacts" / "models" / "model_weights" / "_steel_plates" / EXPERIMENT_NAME / f"DATA_SPLIT_STATE_{DATA_SPLIT_STATE}" / f"RWWCE3_{COST_MATRIX_NAME}" 

        # save config dict to json file & create folder
        if os.path.exists(RESULTS_PATH):
            print(f"Warning: {RESULTS_PATH} already exists. No new directory will be created.")
        else:
            RESULTS_PATH.mkdir(parents=True, exist_ok=True)
            with open(RESULTS_PATH / f"experiment_config.json", 'w') as f:
                json.dump(config_dict, f, indent=4)

        # Initialize SaveOutput
        sys.stdout = SaveOutput(RESULTS_PATH / f"SP_results_RWWCE_{run}.txt")  # Redirect stdout
        
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
        test_loader = DataLoader(test_data)

        print("Length train data:", len(train_data), "Num batches:", len(train_loader))

        # create Neural Network
        test_net = torch_NN_models.FeedForwardNNSP(27, 3)
        test_net.to(device)
        #test_net.load_state_dict(torch.load("methods/model_weights/blobs_test_weights.pth"))

        weight_matrix = torch.tensor([[1, 0, 0],
                                       [0, 1, 0],
                                       [0, 0, 1]], dtype=torch.float32).to(device)
        
        criterion = torch_NN_CS_loss.RWWCE(class_weights=weight_matrix, cost_matrix_normalized=True)
        optimizer = torch.optim.AdamW(test_net.parameters(), lr=LEARNING_RATE)

        if EARLY_STOPPING:
            early_stopping = torch_NN_training.EarlyStopping(patience=PATIENCE, monitor=MONITOR, 
                                           mode=MODE, restore_best_weights=True)
        else:
            early_stopping = None

        _, _, _, _, history_df, df_probs = \
            torch_NN_training.train_model(test_net, train_loader, val_loader, 
                                          optimizer, criterion=criterion,
                                          eval_cost_matrix=COST_MATRIX_RAW.to(device),
                                          training_cost_matrix=COST_MATRIX_RAW.to(device),
                                          num_epochs=NUM_EPOCHS, 
                                          early_stopping=early_stopping,
                                          device=device)

        # print used cost matrix
        print("--------------------------------------------------")
        print("Used cost matrix -- real cost: \n", COST_MATRIX_RAW)
        print("Used cost matrix -- normalized cost: \n", COST_MATRIX_RAW / COST_MATRIX_RAW.sum())
        print("--------------------------------------------------")
        
        train_cost_matrix = COST_MATRIX_RAW.to(device)
        eval_cost_matrix = torch.tensor([[0, 3, 13], [2, 0, 12], [6, 3, 0]], dtype=torch.float32).to(device)
        # evaluate model on training data
        torch_NN_evaluation.print_metrics_summary(test_net, train_loader, criterion, train_cost_matrix, train_cost_matrix, "Train")
        torch_NN_evaluation.print_metrics_summary(test_net, train_loader, criterion, train_cost_matrix, eval_cost_matrix, "Train (eval cost)")
        # evaluate model on validation data
        torch_NN_evaluation.print_metrics_summary(test_net, val_loader, criterion, train_cost_matrix, train_cost_matrix, "Validation")
        torch_NN_evaluation.print_metrics_summary(test_net, val_loader, criterion, eval_cost_matrix, eval_cost_matrix, "Validation (eval cost)")
        # evaluate model on test data
        torch_NN_evaluation.print_metrics_summary(test_net, test_loader, criterion, train_cost_matrix, train_cost_matrix, "Test")
        torch_NN_evaluation.print_metrics_summary(test_net, test_loader, criterion, eval_cost_matrix, eval_cost_matrix, "Test (eval cost)")
        
        # Restore stdout to normal after execution
        sys.stdout.file.close()
        sys.stdout = sys.stdout.terminal

        #save history_df to csv
        history_df.to_csv(RESULTS_PATH / f"_steel_plates_history_RWWCE_{run}.csv", index=False) 
        #df_probs.to_csv(RESULTS_PATH / f"_steel_plates_probabilities_RWWCE_{run}.csv", index=False)   

        torch_NN_evaluation.plot_training_metrics(history_df, 
                                    save_path=RESULTS_PATH / f"_steel_plates_figures_RWWCE_{run}.png")