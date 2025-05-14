import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from typing import List, Union, Tuple
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from ..embeddings.embed import Embeddings

class BayesianMSELoss(torch.nn.Module):
    """ Custom MSE Loss with Bayesian Regularization. """
    def __init__(self, device=None):
        super(BayesianMSELoss, self).__init__()
        self.device = device

    def forward(self, x, y, w, lam):
        # Standard MSE Loss
        mse_loss = torch.nn.functional.mse_loss(x, y)
        # Bayesian regularization (penalizing weights' deviation from identity matrix)
        identity = torch.eye(w.size()[1], device=self.device)
        reg_loss = torch.sum((w - identity) ** 2) / x.nelement()
        return mse_loss + lam * reg_loss

class FewShotLinearRegression(torch.nn.Module):
    """ Linear Regression model for Few-Shot Learning with custom Bayesian MSE Loss. """
    def __init__(self, input_dim, output_dim, device=None):
        super(FewShotLinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=False)
        torch.nn.init.zeros_(self.linear.weight)

        self.device = device

    def forward(self, x):
        return self.linear(x)


class FewShotLearner:
    """
    Few-Shot Learning class using Bayesian MSE Loss and Linear Regression.

    Parameters:
    - support_set (pd.DataFrame): DataFrame with support samples.
    - text_col (str): Name of the text column.
    - label_col (str): Name of the category column.
    - embedding_model (Embeddings): Instance of the Embeddings class for text embedding.
    """
    def __init__(self, support_set, text_col, label_col, embedding_model: Embeddings):
        self.support_set = support_set
        self.text_col = text_col
        self.label_col = label_col
        self.embedding_model = embedding_model
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_embeddings = None
        self.label_dict = None

        self._validate_inputs()

    def _validate_inputs(self):
        """ 
        Ensure the support set has both text and label columns, 
        and the query set only requires the text column. 
        """
        # Check support set
        if self.text_col not in self.support_set.columns or self.label_col not in self.support_set.columns:
            raise ValueError(f"Support set must contain both '{self.text_col}' and '{self.label_col}' columns.")

    def _prepare_training_data(self, val_split):
        """
        Embed the support set texts and labels, split into training and validation, 
        and convert them to PyTorch tensors.

        Args:
            val_split (float): Proportion of data to be used for validation.
        """
        # Embed support set texts
        support_emb_df = self.embedding_model.embed_df(self.support_set, self.text_col)
        X_support = support_emb_df.filter(like='emb_').values

        # Embed labels and create mapping
        labels = self.support_set[self.label_col].unique().tolist()
        label_embeddings, _ = self.embedding_model.embed_text(labels)
        self.label_dict = {i: label for i, label in enumerate(labels)}
        self.label_embeddings = torch.tensor(label_embeddings, dtype=torch.float32, device=self.device)

        # Map labels to their embeddings
        label_to_index = {label: idx for idx, label in enumerate(labels)}
        y_support = self.support_set[self.label_col].apply(lambda lbl: label_to_index.get(lbl, -1)).values

        # Ensure no invalid labels (-1) are present
        if -1 in y_support:
            raise ValueError("Some labels in the support set are not found in the label dictionary.")

        # Map to embeddings
        y_support = np.stack([label_embeddings[idx] for idx in y_support])

        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_support, y_support, test_size=val_split, random_state=42)

        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y_train, dtype=torch.float32, device=self.device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32, device=self.device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32, device=self.device)

        # Determine input and output dimensions
        input_dim = X_support.shape[1]
        output_dim = y_support.shape[1]

        return (X_tensor, y_tensor), (X_val_tensor, y_val_tensor), input_dim, output_dim

    def _train_model(self, train_data, val_data, input_dim, output_dim, lam, lr, epochs, early_stop, verbose=True):
        """
        Train the Few-Shot model with regularization, early stopping, and Bayesian MSE loss.

        Args:
            train_data (tuple): Training data as (X_tensor, y_tensor).
            val_data (tuple): Validation data as (X_val_tensor, y_val_tensor).
            input_dim (int): Input dimensionality.
            output_dim (int): Output dimensionality.
            lam (float): Regularization parameter.
            lr (float): Learning rate.
            epochs (int): Number of training epochs.
            early_stop (int): patience for early stopping.
            verbose (bool): Whether to print training progress.
        """
        
        # Unpack training data
        X_tensor, y_tensor = train_data
        support_size = len(X_tensor)
        batch_size = min(32, max(8, support_size // 10))

        # Prepare DataLoader
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Unpack validation data
        X_val_tensor, y_val_tensor = val_data

        # Initialize model with dropout
        self.model = FewShotLinearRegression(input_dim, output_dim, self.device)
        loss_fcn = BayesianMSELoss(device=self.device)

        # Optimizer with weight decay
        optimizer = Adam(self.model.parameters(), lr=lr)

        # Early Stopping setup
        best_loss = float('inf')
        early_stop_counter = 0

        # Training loop
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0

            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                preds = self.model(X_batch)
                loss = loss_fcn(preds, y_batch, self.model.linear.weight, lam)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_preds = self.model(X_val_tensor)
                val_loss = loss_fcn(val_preds, y_val_tensor, self.model.linear.weight, lam).item()

            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - Training Loss: {epoch_loss:.4f} - Validation Loss: {val_loss:.4f}")

            # Early stopping check
            if val_loss < best_loss:
                best_loss = val_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter >= early_stop:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break

    def fit(self, val_split: float = 0.2, lam=0.1, lr=0.01, epochs=50, early_stop=5, verbose=False):
        """
        Train the Few-Shot model using support set data, embedding both texts and labels.
        It uses Bayesian MSE Loss and Linear Regression with l2 regularization, and early stopping.

        Parameters:
        - val_split (float): Proportion of labeled data to be used for validation.
        - lam (float): Regularization parameter.
        - lr (float): Learning rate.
        - epochs (int): Number of epochs.
        - early_stop (int): early_stop for early stopping.
        - verbose (bool): Whether to print training progress.
        """
        
        # Prepare training and validation data
        train_data, val_data, input_dim, output_dim = self._prepare_training_data(val_split=val_split)

        # Train the model
        self._train_model(
            train_data=train_data,
            val_data=val_data,
            input_dim=input_dim,
            output_dim=output_dim,
            lam=lam,
            lr=lr,
            epochs=epochs,
            early_stop=early_stop,
            verbose=verbose
        )

    def predict(self, query_set: pd.DataFrame, k: int = 3, return_accuracy: bool = False) -> Union[pd.DataFrame, Tuple[pd.DataFrame, float]]:
        """
        Predict categories for the query set based on the trained model.

        Parameters:
        - query_set (pd.DataFrame): DataFrame with text data for prediction.
        - k (int): Number of top predictions to consider.
        - return_accuracy (bool): If True, also return accuracy score.

        Returns:
        - pd.DataFrame: DataFrame with 'pred' (label index) and 'pred_label' (label name) columns.
        - Optional[float]: Accuracy score, only returned if `return_accuracy=True`.
        """
        # Ensure necessary columns and model are ready
        if self.text_col not in query_set.columns:
            raise ValueError(f"Query set must contain the '{self.text_col}' column for predictions.")
        if self.model is None or self.label_embeddings is None:
            raise ValueError("Model is not trained. Call 'fit' first.")

        # Embed query set
        query_emb_df = self.embedding_model.embed_df(query_set, self.text_col)
        X_query = query_emb_df.filter(like='emb_').values
        X_tensor = torch.tensor(X_query, dtype=torch.float32, device=self.device)

        # Determine the number of labels
        num_labels = self.label_embeddings.size(0)

        # Adjust k if it exceeds the number of labels
        k = min(k, num_labels)

        # Apply model and normalize embeddings
        with torch.no_grad():
            query_embeddings = self.model(X_tensor)
            query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
            label_embeddings = F.normalize(self.label_embeddings, p=2, dim=1)

            # Compute similarities
            similarities = torch.matmul(query_embeddings, label_embeddings.T)
            top_k_vals, top_k_indices = similarities.topk(k, dim=1, largest=True, sorted=True)

        # Construct predictions DataFrame
        query_set['pred'] = top_k_indices[:, 0].tolist()
        query_set['pred_label'] = query_set['pred'].apply(lambda idx: self.label_dict.get(idx, "Unknown"))

        # If return_accuracy is enabled and true labels are provided
        if return_accuracy and self.label_col in query_set.columns:
            # Map true labels to indices
            label_to_index = {label: idx for idx, label in self.label_dict.items()}
            query_set['true_label_idx'] = query_set[self.label_col].apply(lambda lbl: label_to_index.get(lbl, -1))

            # Compute accuracy
            correct_preds = (query_set['pred'] == query_set['true_label_idx']).sum()
            accuracy = correct_preds / len(query_set)
            return query_set, accuracy

        return query_set