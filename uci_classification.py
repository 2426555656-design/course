import os
import urllib.request
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ADULT_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
ADULT_TEST_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
IRIS_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

DATA_DIR = "data_cache"


def download_if_needed(url: str, filename: str) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"Downloading {url} -> {path}")
        urllib.request.urlretrieve(url, path)
    return path


def load_adult_dataset(test_size: float = 0.2, random_state: int = 42) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    column_names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    ]

    train_path = download_if_needed(ADULT_DATA_URL, "adult.data")
    test_path = download_if_needed(ADULT_TEST_URL, "adult.test")

    train_df = pd.read_csv(train_path, names=column_names, na_values="?", skipinitialspace=True)
    test_df = pd.read_csv(test_path, names=column_names, na_values="?", skipinitialspace=True, skiprows=1)

    df = pd.concat([train_df, test_df], ignore_index=True)
    df.dropna(inplace=True)

    y = df["income"].apply(lambda x: 1 if x.strip().strip('.') == ">50K" else 0)
    X = df.drop(columns=["income"])

    categorical_cols = X.select_dtypes(include=["object"]).columns
    X = pd.get_dummies(X, columns=categorical_cols)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y.values, test_size=test_size, random_state=random_state, stratify=y.values
    )

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor


def load_iris_dataset(test_size: float = 0.2, random_state: int = 42) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
    iris_path = download_if_needed(IRIS_DATA_URL, "iris.data")
    df = pd.read_csv(iris_path, names=columns)
    df.dropna(inplace=True)

    species_map = {label: idx for idx, label in enumerate(sorted(df["species"].unique()))}
    y = df["species"].map(species_map).values
    X = df.drop(columns=["species"]).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor


class AdultNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class IrisNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 20,
    lr: float = 1e-3,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total

        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()

        test_acc = test_correct / test_total if test_total else 0.0
        print(f"Epoch {epoch:02d}: loss={train_loss:.4f}, train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")


def main() -> None:
    print("Preparing Adult dataset...")
    X_train_a, X_test_a, y_train_a, y_test_a = load_adult_dataset()
    adult_train_loader = DataLoader(TensorDataset(X_train_a, y_train_a), batch_size=256, shuffle=True)
    adult_test_loader = DataLoader(TensorDataset(X_test_a, y_test_a), batch_size=256)

    adult_input_dim = X_train_a.shape[1]
    adult_model = AdultNet(adult_input_dim)
    print("Training AdultNet...")
    train_model(adult_model, adult_train_loader, adult_test_loader, epochs=15, lr=1e-3)

    print("Preparing Iris dataset...")
    X_train_i, X_test_i, y_train_i, y_test_i = load_iris_dataset()
    iris_train_loader = DataLoader(TensorDataset(X_train_i, y_train_i), batch_size=16, shuffle=True)
    iris_test_loader = DataLoader(TensorDataset(X_test_i, y_test_i), batch_size=16)

    iris_input_dim = X_train_i.shape[1]
    iris_model = IrisNet(iris_input_dim)
    print("Training IrisNet...")
    train_model(iris_model, iris_train_loader, iris_test_loader, epochs=50, lr=5e-3)


if __name__ == "__main__":
    main()
