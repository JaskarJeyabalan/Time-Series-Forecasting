# ===============================
# 1. IMPORT LIBRARIES
# ===============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ===============================
# 2. LOAD DATASET
# ===============================
df = pd.read_csv("electricity_load.csv")  # change path if needed
df['load'] = df['load'].astype(float)

values = df['load'].values.reshape(-1, 1)

# ===============================
# 3. NORMALIZATION
# ===============================
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(values)


# ===============================
# 4. CREATE SEQUENCES
# ===============================
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

SEQ_LEN = 24
X, y = create_sequences(scaled_data, SEQ_LEN)

# ===============================
# 5. TRAIN / VAL / TEST SPLIT
# ===============================
train_size = int(0.7 * len(X))
val_size = int(0.85 * len(X))

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:val_size], y[train_size:val_size]
X_test, y_test = X[val_size:], y[val_size:]

# ===============================
# 6. DATASET CLASS
# ===============================
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=32, shuffle=False)
val_loader = DataLoader(TimeSeriesDataset(X_val, y_val), batch_size=32, shuffle=False)
test_loader = DataLoader(TimeSeriesDataset(X_test, y_test), batch_size=32, shuffle=False)


# ===============================
# 7. BASELINE LSTM MODEL
# ===============================
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


# ===============================
# 8. ATTENTION MECHANISM
# ===============================
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        weights = torch.softmax(self.attn(lstm_output), dim=1)
        context = torch.sum(weights * lstm_output, dim=1)
        return context, weights


# ===============================
# 9. LSTM WITH ATTENTION
# ===============================
class LSTMAttention(nn.Module):
    def __init__(self, input_size=1, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context, attn_weights = self.attention(lstm_out)
        out = self.fc(context)
        return out, attn_weights


# ===============================
# 10. TRAINING FUNCTION
# ===============================
def train_model(model, train_loader, val_loader, epochs=20):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            if isinstance(output, tuple):
                output = output[0]
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                output = model(X_batch)
                if isinstance(output, tuple):
                    output = output[0]
                val_loss += criterion(output, y_batch).item()

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")


# ===============================
# 11. TRAIN MODELS
# ===============================
baseline_model = LSTMModel()
train_model(baseline_model, train_loader, val_loader)

attention_model = LSTMAttention()
train_model(attention_model, train_loader, val_loader)


# ===============================
# 12. EVALUATION
# ===============================
def evaluate(model, loader):
    preds, actuals = [], []
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in loader:
            output = model(X_batch)
            if isinstance(output, tuple):
                output = output[0]
            preds.append(output.numpy())
            actuals.append(y_batch.numpy())

    preds = scaler.inverse_transform(np.vstack(preds))
    actuals = scaler.inverse_transform(np.vstack(actuals))

    mae = mean_absolute_error(actuals, preds)
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    return mae, rmse, preds, actuals


baseline_mae, baseline_rmse, _, _ = evaluate(baseline_model, test_loader)
attn_mae, attn_rmse, preds, actuals = evaluate(attention_model, test_loader)

print("\nRESULTS")
print(f"Baseline LSTM → MAE: {baseline_mae:.2f}, RMSE: {baseline_rmse:.2f}")
print(f"LSTM + Attention → MAE: {attn_mae:.2f}, RMSE: {attn_rmse:.2f}")


# ===============================
# 13. ATTENTION VISUALIZATION
# ===============================
X_sample, _ = next(iter(test_loader))
_, attention_weights = attention_model(X_sample)

plt.figure(figsize=(10,4))
plt.imshow(attention_weights[0].detach().numpy(), cmap='hot')
plt.colorbar()
plt.title("Attention Weights over Time Steps")
plt.xlabel("Hidden Units")
plt.ylabel("Time Steps")
plt.show()
