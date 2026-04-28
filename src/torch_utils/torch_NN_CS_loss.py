import torch
import torch.nn as nn
import torch.nn.functional as F

class AEC(nn.Module):
    def __init__(self, cost_matrix_normalized=True):
        super(AEC, self).__init__()

        self.cost_matrix_normalized = cost_matrix_normalized

    def forward(self, outputs, labels, training_cost_matrix):
        if self.cost_matrix_normalized:
            cost_matrix = training_cost_matrix.clone() / training_cost_matrix.sum()
        else: 
            cost_matrix = training_cost_matrix.clone()

        probs = F.softmax(outputs, dim=1)
        costs = cost_matrix[labels]
        return torch.sum(costs * probs, dim=1).mean()

    

class RWWCE(nn.Module):
    def __init__(self, class_weights=None, cost_matrix_normalized=True):
        super(RWWCE, self).__init__()

        if class_weights is not None:
            self.class_weights = class_weights
        else:
            self.class_weights = None

        self.cost_matrix_normalized = cost_matrix_normalized

    def forward(self, outputs, labels, training_cost_matrix):        
        
        # get weights in tensor shape of (num_classes, num_classes) with 1 on the diagonal
        n_classes = training_cost_matrix.shape[0]

        if self.class_weights is not None:
            class_weights = self.class_weights
        else:
            class_weights = torch.eye(n_classes, dtype=torch.float32, device=outputs.device)

    
        if self.cost_matrix_normalized:
            cost_matrix = training_cost_matrix.clone() / training_cost_matrix.sum()
        else:
            cost_matrix = training_cost_matrix.clone()

        costs = cost_matrix[labels]
        class_weights = class_weights[labels]
        output_probs = F.softmax(outputs, dim=1)

        logs = torch.log(output_probs + 1e-10)  # logs for true class: add small value to avoid log(0)
        logs_1_sub = torch.log((1 - output_probs) + 1e-10)  # logs for false class: log(1 - p)

        full_true_prediction_weights = torch.sum(class_weights * logs, dim=1, keepdim=True)
        full_misclassification_cost_weights = torch.sum(costs * logs_1_sub, dim=1, keepdim=True)
    
        return - torch.sum(full_true_prediction_weights + full_misclassification_cost_weights) / outputs.size(0)
