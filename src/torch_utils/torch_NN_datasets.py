import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split
from src.utils import DataHandler
from sklearn.preprocessing import StandardScaler
from config import paths


class CSVDataset(Dataset):

    """
    Dataset class for loading data from a CSV file.
    """

    def __init__(self, path, label_column):
        self.data = DataHandler.load_data(path)
        #self.data = self.data.apply(pd.to_numeric, errors='coerce')
        self.X = self.data.drop(columns=label_column).to_numpy().astype(np.float32)
        self.y = self.data[label_column].to_numpy().astype(np.float32)

        # convert to tensors
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    
def compute_class_weights(dataset_subset, num_classes):
    labels = []

    # dataset_subset is a Subset → need original dataset
    for idx in dataset_subset.indices:
        _, y = dataset_subset.dataset[idx]   # assumes __getitem__ returns (x, y)
        labels.append(int(y))

    labels = np.array(labels)
    class_counts = np.bincount(labels, minlength=num_classes)

    total_samples = class_counts.sum()
    weights = total_samples / (num_classes * class_counts)

    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.mean()  # normalize (stability)

    return weights


class CSVDatasetScaled(Dataset):
    """
    Dataset class for loading data from a CSV file with StandardScaler.
    Scaler wird auf den gesamten Datensatz gefittet. Falls du den Scaler
    nur auf Trainingsdaten fitten willst, nutze fit_scaler() und transform()
    nach dem Split.
    """
    def __init__(self, path, label_column):
        self.data = DataHandler.load_data(path)
        self.X = self.data.drop(columns=label_column).to_numpy().astype(np.float32)
        self.y = self.data[label_column].to_numpy().astype(np.float32)
        
        self.feature_names = [c for c in self.data.columns if c != label_column]
        
        # Scaler initialisieren, aber noch NICHT fitten
        self.scaler = StandardScaler()
        self._is_fitted = False
        
        # convert to tensors (noch unskaliert)
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)
    
    def fit_scaler(self, indices):
        """Fittet den Scaler NUR auf den angegebenen Indices (z.B. Train-Indices).
        Transformiert anschließend ALLE Daten."""
        X_numpy = self.X.numpy()
        self.scaler.fit(X_numpy[indices])
        X_scaled = self.scaler.transform(X_numpy)
        self.X = torch.tensor(X_scaled, dtype=torch.float32)
        self._is_fitted = True
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



def stratified_split(dataset, split_ratios, random_state=42):
    """
    Stratified train/val/test split that preserves class distribution in each subset.

    Unlike torch.utils.data.random_split, this guarantees that every split has
    (approximately) the same class proportions **and** the same number of samples
    per class, regardless of the random_state.

    Parameters
    ----------
    dataset : CSVDataset
        Must expose a `.y` attribute (tensor of integer labels).
    split_ratios : list[float]
        Three floats that sum to 1.0, e.g. [0.6, 0.2, 0.2] for 60/20/20.
    random_state : int
        Seed for reproducibility.  Different seeds yield different samples
        but identical per-class counts in each split.

    Returns
    -------
    train_subset, val_subset, test_subset : torch.utils.data.Subset
        Drop-in replacements for the subsets returned by random_split.
    """
    assert len(split_ratios) == 3, "Exactly three split ratios required (train, val, test)."
    assert abs(sum(split_ratios) - 1.0) < 1e-6, "Split ratios must sum to 1.0."

    labels = dataset.y.numpy()
    indices = np.arange(len(dataset))

    # First split: train vs. (val + test)
    val_test_ratio = split_ratios[1] + split_ratios[2]
    train_idx, val_test_idx = train_test_split(
        indices, test_size=val_test_ratio, stratify=labels[indices],
        random_state=random_state
    )

    # Second split: val vs. test (relative ratio within val+test portion)
    relative_test_ratio = split_ratios[2] / val_test_ratio
    val_idx, test_idx = train_test_split(
        val_test_idx, test_size=relative_test_ratio, stratify=labels[val_test_idx],
        random_state=random_state
    )

    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)

    
if __name__ == "__main__":
    data = CSVDataset(paths.BASE_DIR / "artifacts" / "data" / "datasets" / "steel_plates_3cls.csv", 'label')
    print(data)

