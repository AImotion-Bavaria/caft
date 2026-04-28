import torch
import torch.nn.functional as F
import pandas as pd

def get_labels_probs_costs(model, data_loader, cost_matrix, device=None):
    """
    Get labels, probabilities and costs for the given model and data loader.
    :param model: PyTorch model
    :param data_loader: DataLoader for the dataset
    :param cost_matrix: Cost matrix
    :param device: torch.device (auto-detected from model if None)
    :return: labels, probabilities, costs
    """
    
    if device is None:
        device = next(model.parameters()).device

    cost_matrix = cost_matrix.to(device)
    model.eval()

    labels = []
    probs = []
    costs = []
    
    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            inputs, label = data
            inputs, label = inputs.to(device), label.to(device)
            cost = cost_matrix[label]

            outputs = model(inputs)
            ## get softmax probabilities
            softmax_output = F.softmax(outputs, dim=1)

            labels.append(label)
            probs.append(softmax_output)
            costs.append(cost)

    labels = torch.cat(labels, dim=0)
    probs = torch.cat(probs, dim=0)
    costs = torch.cat(costs, dim=0)

    return labels.cpu(), probs.cpu(), costs.cpu()


def get_AEC_metric(model, data_loader, cost_matrix): # works to reach similar results as "real" AEC metric
    """
    Calculate the Average Expected Cost (AEC) for the given model and data loader.
    :param model: PyTorch model
    :param data_loader: DataLoader for the dataset
    :param cost_matrix: Cost matrix
    :return: Average Expected Cost (AEC)
    """ 

    _, probs, costs = get_labels_probs_costs(model, data_loader, cost_matrix)
    aec = torch.sum(costs * probs, dim=1).mean()

    return aec


# def get_AECWCE_metric(model, data_loader, cost_matrix):
#     """
#     Calculate the Average Expected Cost Weighted Cross Entropy (AECWCE) for the given model and data loader.
#     :param model: PyTorch model
#     :param data_loader: DataLoader for the dataset
#     :param cost_matrix: Cost matrix
#     :return: Average Expected Cost Weighted Cross Entropy (AECWCE)
#     """

#     _, probs, costs = get_labels_probs_costs(model, data_loader, cost_matrix)
    
#     log_probs = torch.log((1-probs) + 1e-10)  # add small value to avoid log(0)
    
#     # calculate weighted cross entropy loss
#     aecwce = torch.sum(costs * -log_probs, dim=1).mean()

#     return aecwce


def get_RWWCE_metric(model, data_loader, cost_matrix):
    # TODO: Implement the RWWCE metric
    """    Calculate the Relative Weighted Cross Entropy (RWWCE) for the given model and data loader.
    :param model: PyTorch model 
    :param data_loader: DataLoader for the dataset
    :param cost_matrix: Cost matrix
    :return: Real-World Weighted Cross Entropy (RWWCE)
    """

    labels, probs, costs = get_labels_probs_costs(model, data_loader, cost_matrix)

    n_classes = cost_matrix.shape[0]
    #cost_matrix = cost_matrix.clone().float() / cost_matrix.sum()

    # Create false-negative weights: identity matrix per true class
    fn_weights = torch.eye(n_classes, dtype=torch.float32)[labels]

    # Compute logs for correct and incorrect predictions
    logs = torch.log(probs + 1e-10)
    logs_1_sub = torch.log((1 - probs) + 1e-10)

    # Match loss logic:
    m_full_fn_weights = torch.sum(fn_weights * logs, dim=1, keepdim=True)
    m_full_fp_weights = torch.sum(costs * logs_1_sub, dim=1, keepdim=True)

    rwwce = -torch.sum(m_full_fn_weights + m_full_fp_weights) / probs.size(0)

    return rwwce.item()


def get_average_cost_metric(cost_matrix, cm):
    """
    Calculate the average cost of misclassfication per part, 
    based on the confusion matrix and cost matrix.
    :param cost_matrix: Cost matrix
    :param cm: Confusion matrix
    :return: Average cost
    """

    cm_copy = cm.copy()
    # convert confusion matrix to tensor
    cm_copy = torch.tensor(cm_copy, dtype=torch.float32)
    # clone cost matrix to avoid modifying the original cost matrix
    cost_matrix = cost_matrix.detach().cpu().clone()

    # check if cm and cost matrix have the same shape
    if cm_copy.shape != cost_matrix.shape:
        raise ValueError("Confusion matrix and cost matrix must have the same shape")
    
    # calculate the expected cost for each class
    average_cost = torch.sum(cm_copy * cost_matrix, dim=0)  # sum over all classes
    average_cost = torch.sum(average_cost)  # sum over all classes
    # mean over all instances
    average_cost = average_cost / cm_copy.sum()  # mean over all instances
    return average_cost


def create_probability_dataframe(all_epoch_probs, labels):
    """
    Create a DataFrame with probabilities for each sample across all epochs.
    
    Args:
        all_epoch_probs: List of probability arrays for each epoch
        labels: True labels for the samples
    
    Returns:
        pd.DataFrame: DataFrame with columns ['Label', 'Epoch_1', 'Epoch_2', ...]
    """
    if not all_epoch_probs:
        return pd.DataFrame()
    
    # Convert each epoch's probabilities to list of tuples
    prob_columns = []
    for epoch_probs in all_epoch_probs:
        prob_tuples = [tuple(row.tolist()) for row in epoch_probs]
        prob_columns.append(prob_tuples)
    
    # Transpose to shape [n_samples x n_epochs]
    prob_columns = list(zip(*prob_columns))
    
    # Create column names for each epoch
    #epoch_columns = [f'Epoch_{i+1}' for i in range(len(all_epoch_probs))]
    
    # Create DataFrame
    df_probs = pd.DataFrame(prob_columns)
    df_probs.insert(0, "Label", labels)

    if all_epoch_probs:
        final_epoch_probs = all_epoch_probs[-1]  # Get last epoch probabilities
        final_predictions = torch.argmax(final_epoch_probs, dim=1).tolist()
        df_probs.insert(1, "Predictions", final_predictions)
    
    return df_probs

