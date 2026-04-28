from matplotlib import pyplot as plt
import pandas as pd

from src.torch_utils import torch_NN_training
from src.torch_utils.torch_NN_metrics import get_average_cost_metric, get_AEC_metric, get_RWWCE_metric


def print_metrics_summary(model, data_loader, criterion, training_cost_matrix, eval_cost_matrix, dataset_name):
    """
    Print a summary of metrics for the given model and data loader, including loss, accuracy, expected cost, AECWCE and AEC.
    
    Parameters:
    - model: The PyTorch model to evaluate
    - data_loader: The DataLoader for the dataset to evaluate on
    - criterion: The loss function used for evaluation
    - training_cost_matrix: The cost matrix used during training for calculating expected cost and AEC metrics
    - eval_cost_matrix: The cost matrix used during evaluation for calculating expected cost and AEC metrics

    Rerturns:
    - None: This function prints the metrics summary to the console
    """

    print("-"*15 + f"{dataset_name} Metrics Summary" + "-"*15)

    loss, accuracy, cm, report = torch_NN_training.evaluate_model(model, data_loader, criterion, training_cost_matrix)
    average_cost = get_average_cost_metric(eval_cost_matrix, cm)
    average_expected_cost = get_AEC_metric(model, data_loader, eval_cost_matrix)
    real_world_weighted_cross_entropy = get_RWWCE_metric(model, data_loader, eval_cost_matrix)

    print(f"{dataset_name} Loss: {loss:.3f}, {dataset_name} Accuracy: {accuracy:.3f}")
    print(f"Average cost: {average_cost:.3f}")
    print(f"AEC: {average_expected_cost:.3f}")
    print(f"RWWCE: {real_world_weighted_cross_entropy:.3f}")
    print(f"{dataset_name} Confusion Matrix: \n", cm)
    print(f"{dataset_name} Classification Report: \n", report)
    print("\n")

#######################################################################################
# Plotting                                                                            #
#######################################################################################

def plot_history(history, loss_only=False, acc_only=False, figsize=(12, 7)):
    """
    Plots history (learning curve) of a keras model

    Args:
        history: history of trained keras model
        loss_only: Boolean - True if only loss curve should be plottet
        acc_only: Boolean - True if only accuracy curve should be plotted
        figsize: size of the resulting figure - as Tuple with (size_x, size_y)
    """

    history = history.history_

    plt.rc('font', size=15)
    df = pd.DataFrame(data=[history['loss'], history['accuracy'], history['val_loss'], history['val_accuracy']],
                      index=['loss', 'accuracy', 'val_loss', 'val_accuracy']).T

    if loss_only:
        df_loss = df.drop(df.columns[1], axis=1)
        df = df_loss.drop(df.columns[3], axis=1)

    elif acc_only:
        df_acc = df.drop(df.columns[0], axis=1)
        df = df_acc.drop(df.columns[2], axis=1)

    df.plot(figsize=figsize)
    plt.grid(True)
    plt.gca().set_ylim(df.min().min() - df.min().min() * 0.01,
                       df.max().max())
    plt.show()
    

def plot_training_metrics(df, save_path=False, Title="Training & Validation Metrics over Epochs"):
    """
    Plots training and validation metrics (loss, accuracy, and average cost if available).

    Parameters:
    -----------
    df : pandas.DataFrame
        Can contain any of the following columns:
        'epoch', 'train_loss', 'val_loss',
        'train_accuracy', 'val_accuracy',
        'train_average_cost', 'val_average_cost'.
    save_path : str
        Path to save the generated plot (e.g., 'results_folder/metrics.png').
    """

    # Prepare list of plots to create
    plot_specs = []

    if {'train_loss', 'val_loss'}.issubset(df.columns):
        plot_specs.append(('Loss', 'train_loss', 'val_loss', 'tab:red', 'tab:orange'))

    if {'train_accuracy', 'val_accuracy'}.issubset(df.columns):
        plot_specs.append(('Accuracy (%)', 'train_accuracy', 'val_accuracy', 'tab:blue', 'tab:cyan'))

    if {'train_average_cost', 'val_average_cost'}.issubset(df.columns):
        plot_specs.append(('Average Cost', 'train_average_cost', 'val_average_cost', 'tab:green', 'tab:olive'))

    # Create subplots dynamically
    fig, axes = plt.subplots(len(plot_specs), 1, figsize=(8, 4 * len(plot_specs)), sharex=True)

    if len(plot_specs) == 1:
        axes = [axes]  # Ensure axes is iterable even if one plot

    for ax, (ylabel, train_col, val_col, train_color, val_color) in zip(axes, plot_specs):
        ax.plot(df["epoch"], df[train_col], label=f"Train {ylabel}", color=train_color)
        ax.plot(df["epoch"], df[val_col], label=f"Val {ylabel}", color=val_color)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)

    axes[-1].set_xlabel("Epoch")
    fig.suptitle("Training & Validation Metrics over Epochs", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save the figure
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()