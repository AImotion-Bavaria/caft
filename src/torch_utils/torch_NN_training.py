import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
from src.torch_utils import torch_NN_metrics
from src.torch_utils.torch_NN_CS_loss import AEC
from src.torch_utils.torch_NN_metrics import get_average_cost_metric


def train_model(model, train_loader, val_loader, optimizer, criterion, eval_cost_matrix=None, 
                training_cost_matrix=None, num_epochs=32, early_stopping=None, device=None):

    if device is None:
        device = next(model.parameters()).device

    # define cost matrices for evaluation
    # only when evaluation cost_matrix is not None, the evaluation cost matrix is used
    if eval_cost_matrix is not None:
        eval_cost_matrix_raw = eval_cost_matrix
        eval_cost_matrix_norm = eval_cost_matrix / eval_cost_matrix.sum()
    
    # define empty lists to store training and validation metrics
    # these will be used to create a DataFrame at the end of training
    train_loss_history, val_loss_history = [], []
    train_accuracy_history, val_accuracy_history = [], []
    
    train_average_cost_history, val_average_cost_history = [], []
    
    all_epoch_probs = []
    val_labels = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # use criterion to calculate loss
            if training_cost_matrix is not None:
                loss = criterion(outputs, labels, training_cost_matrix)
            else:
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # get accuracy of the model
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()

        # Evaluate on training set
        average_loss = total_loss / len(train_loader)
        accuracy = total_correct / len(train_loader.dataset) * 100
        
        _, _, train_cm, _ = evaluate_model(model, train_loader, criterion, 
                                           cost_matrix=training_cost_matrix, device=device)
        train_loss_history.append(average_loss)
        train_accuracy_history.append(accuracy)
        if eval_cost_matrix is not None:
            train_average_cost = get_average_cost_metric(eval_cost_matrix_raw, train_cm)
        else:
            train_average_cost = "N/A"
        train_average_cost_history.append(train_average_cost.item())
            

        # Evaluate on validation set
        val_loss, val_accuracy, val_cm, _ = evaluate_model(model, val_loader, criterion, 
                                                           cost_matrix=training_cost_matrix, device=device)
        val_loss_history.append(val_loss)
        val_accuracy_history.append(val_accuracy)
        if eval_cost_matrix is not None:
            val_average_cost = torch_NN_metrics.get_average_cost_metric(eval_cost_matrix_raw, val_cm)
        else:
            val_average_cost = "N/A"
        val_average_cost_history.append(val_average_cost.item())
        
        # Get prediction probabilities for validation set
        if eval_cost_matrix is not None:
            val_labels, val_probs, _ = torch_NN_metrics.get_labels_probs_costs(
                model, val_loader, eval_cost_matrix_norm, device=device)
        else:
            num_classes = len(torch.unique(val_loader.dataset.targets))
            zero_cost_matrix = torch.zeros((num_classes, num_classes), dtype=torch.float32, device=device)
            val_labels, val_probs, _ = torch_NN_metrics.get_labels_probs_costs(
                model, val_loader, zero_cost_matrix, device=device)
        
        all_epoch_probs.append(val_probs)

        # Check early stopping
        if early_stopping is not None:
            # Get the metric to monitor
            if early_stopping.monitor == 'val_loss':
                current_metric = val_loss
            elif early_stopping.monitor == 'val_accuracy':
                current_metric = val_accuracy
            elif early_stopping.monitor == 'val_average_cost':
                current_metric = val_average_cost.item()
            else:
                raise ValueError(f"Unknown monitor metric: {early_stopping.monitor}")
            
            if early_stopping(current_metric, model, epoch):
                print(f"Training stopped early at epoch {epoch + 1}")
                break
       
        # Print progress every 5 epochs and last epoch
        if epoch % 5 == 4 or epoch == num_epochs - 1:
            print(f"Epoch {epoch + 1} finished! -- Loss: {average_loss:.3f}, Accuracy: {accuracy:.3f}, "
                  f"Val Loss: {val_loss:.3f}, Val Accuracy: {val_accuracy:.3f}")
            print(f"Average cost -- validation set: {val_average_cost:.3f}")
            print("Validation Confusion Matrix: \n", val_cm)

            # Print early stopping info if available
            if early_stopping is not None:
                print(f"Early stopping: {early_stopping.counter}/{early_stopping.patience} "
                      f"(Best {early_stopping.monitor}: {early_stopping.best_score:.4f} at epoch {early_stopping.best_epoch + 1})")

    # Create history DataFrame
    actual_epochs = len(train_loss_history)
    history_df = pd.DataFrame({
        'epoch': range(1, actual_epochs + 1),
        'train_loss': train_loss_history,
        'val_loss': val_loss_history,
        'train_accuracy': train_accuracy_history,
        'val_accuracy': val_accuracy_history,
        'train_average_cost': train_average_cost_history,
        'val_average_cost': val_average_cost_history            
    })
    
    # Create probability DataFrame using helper function
    df_probs = torch_NN_metrics.create_probability_dataframe(all_epoch_probs, val_labels)
        
    return train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history, history_df, df_probs


def evaluate_model(model, data_loader, criterion, cost_matrix=None, device=None):

    if device is None:
        device = next(model.parameters()).device

    if cost_matrix is not None:
        cost_matrix = cost_matrix.to(device)

    model.eval()
    total_loss = 0.0
    total_correct = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            if cost_matrix is not None:
                loss = criterion(outputs, labels, cost_matrix)
            else:
                loss = criterion(outputs, labels)

            total_loss += loss.item()

            # get accuracy of the model
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu())
            all_labels.extend(labels.cpu())

        average_loss = total_loss / len(data_loader)
        accuracy = total_correct / len(data_loader.dataset) * 100

        # get confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        report = classification_report(all_labels, all_preds, zero_division=0)

    model.train()

    return average_loss, accuracy, cm, report


class EarlyStopping:
    """Early stopping utility to stop training when validation metric stops improving."""
    
    def __init__(self, patience=7, min_delta=0, monitor='val_loss', mode='min', restore_best_weights=True):
        """
        Args:
            patience (int): Number of epochs to wait after last improvement
            min_delta (float): Minimum change to qualify as an improvement
            monitor (str): Metric to monitor ('val_loss', 'val_accuracy', 'val_average_cost')
            mode (str): 'min' for metrics to minimize, 'max' for metrics to maximize
            restore_best_weights (bool): Whether to restore model weights from best epoch
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.counter = 0
        self.best_epoch = 0
        self.best_weights = None
        
        self.is_better = self._get_comparison_function()
    
    def _get_comparison_function(self):
        if self.mode == 'min':
            return lambda current, best: current < best - self.min_delta
        else:  # mode == 'max'
            return lambda current, best: current > best + self.min_delta
    
    def __call__(self, current_score, model, epoch):
        """
        Check if early stopping should be triggered.
        
        Returns:
            bool: True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = current_score
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            return False
        
        if self.is_better(current_score, self.best_score):
            self.best_score = current_score
            self.best_epoch = epoch
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                print(f"Early stopping triggered! Restoring weights from epoch {self.best_epoch + 1}")
            else:
                print(f"Early stopping triggered at epoch {epoch + 1}")
            return True
        
        return False


