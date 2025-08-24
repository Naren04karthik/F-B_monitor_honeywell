import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import lightgbm as lgb
import joblib

# Load data and preprocess
data = pd.read_csv('milk_timeseries_dataset.csv', parse_dates=['Timestamp'])
data.columns = data.columns.str.strip()  # Remove leading/trailing spaces

# Rename typo column
if 'Temprature' in data.columns:
    data.rename(columns={'Temprature': 'Temperature'}, inplace=True)

# Confirm columns
print("Columns:", data.columns.tolist())

# Features and target
features = ['pH', 'Temperature', 'Taste', 'Odor', 'Fat', 'Turbidity', 'Colour', 'Grade']
target = 'Anomaly_Status'

# Encode categorical features
categorical_encoders = {}
for col in ['Taste', 'Odor', 'Colour', 'Grade']:
    if col in data.columns and data[col].dtype == 'object':
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        categorical_encoders[col] = le

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data[features])

# Encode target
if data[target].dtype == 'object':
    le_target = LabelEncoder()
    y = le_target.fit_transform(data[target])
else:
    y = data[target].values

# Train-test split (time-wise)
split_idx = int(len(data) * 0.8)
X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# TCN components
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                                   stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                                   stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, 1, dilation_size,
                                     padding=(kernel_size-1)*dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)

# Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, window_size):
        self.X = X
        self.y = y
        self.window_size = window_size
    def __len__(self):
        return len(self.X) - self.window_size + 1
    def __getitem__(self, idx):
        return (self.X[idx:idx+self.window_size].T.astype(np.float32), self.y[idx+self.window_size-1])

window_size = 10
batch_size = 32
num_epochs = 30
learning_rate = 0.001
num_channels = [64, 64]

train_dataset = TimeSeriesDataset(X_train, y_train, window_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TimeSeriesDataset(X_test, y_test, window_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tcn = TCN(num_inputs=len(features), num_channels=num_channels).to(device)
classifier = nn.Linear(num_channels[-1], 2).to(device)  # Binary classification

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(tcn.parameters()) + list(classifier.parameters()), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    tcn.train()
    classifier.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        features_out = tcn(batch_x)
        out = classifier(features_out[:, :, -1])
        loss = criterion(out, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

# Extract TCN features
def extract_tcn_features(tcn_model, data_array, window_size, batch_size=64):
    tcn_model.eval()
    features_list = []
    dataset = TimeSeriesDataset(data_array, np.zeros(len(data_array)), window_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            feat = tcn_model(batch_x)[:, :, -1]
            features_list.append(feat.cpu().numpy())
    return np.vstack(features_list)

X_train_tcn_features = extract_tcn_features(tcn, X_train, window_size)
X_test_tcn_features = extract_tcn_features(tcn, X_test, window_size)

y_train_lgb = y_train[window_size-1:]
y_test_lgb = y_test[window_size-1:]

# Train LightGBM
lgb_train = lgb.Dataset(X_train_tcn_features, label=y_train_lgb)
lgb_eval = lgb.Dataset(X_test_tcn_features, label=y_test_lgb, reference=lgb_train)

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.03,
    'num_leaves': 31,
    'verbose': -1,
    'seed': 42
}

print("Training LightGBM...")
gbm = lgb.train(params,
                lgb_train,
                valid_sets=[lgb_train, lgb_eval],
                num_boost_round=100,
                callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(10)])

# Predict and evaluate
y_pred_prob = gbm.predict(X_test_tcn_features, num_iteration=gbm.best_iteration)
y_pred = (y_pred_prob > 0.5).astype(int)

print("LightGBM Classification Report:")
print(classification_report(y_test_lgb, y_pred))
print(f"Accuracy: {accuracy_score(y_test_lgb, y_pred):.4f}")

# Save models and scaler
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(gbm, 'lightgbm_model.pkl')
torch.save(tcn.state_dict(), 'tcn_model.pth')
print("Models and scaler saved successfully!")

# If using Google Colab, download files
try:
    from google.colab import files
    files.download('scaler.pkl')
    files.download('lightgbm_model.pkl')
    files.download('tcn_model.pth')
except:
    print("Files saved locally. Download manually if needed.")

# classify_sample function (unchanged, you can use it as before)
def classify_sample(input_dict, scaler, tcn_model, gbm_model, window_size, device, feature_order, categorical_encoders=None):
    missing = [f for f in feature_order if f not in input_dict]
    if missing:
        raise ValueError(f"Missing input features: {missing}")

    input_values = []
    for f in feature_order:
        val = input_dict[f]
        if categorical_encoders and f in categorical_encoders and isinstance(val, str):
            try:
                val = categorical_encoders[f].transform([val])[0]
            except ValueError:
                print(f"Warning: Unseen label '{val}' for feature '{f}'. Using 0.")
                val = 0
        input_values.append(val)

    input_array = np.array(input_values).reshape(1, -1).astype(float)
    input_scaled = scaler.transform(input_array)
    input_sequence = np.tile(input_scaled.T, (1, window_size))
    input_tensor = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(0).to(device)

    tcn_model.eval()
    with torch.no_grad():
        tcn_features = tcn_model(input_tensor)[:, :, -1]

    tcn_features_np = tcn_features.cpu().numpy()
    pred_prob = gbm_model.predict(tcn_features_np)
    pred_class = (pred_prob > 0.5).astype(int)[0]

    return "Anomaly" if pred_class == 1 else "Normal"

# Example usage
feature_order_list = ['pH', 'Temperature', 'Taste', 'Odor', 'Fat', 'Turbidity', 'Colour', 'Grade']
sample_input_encoded = {'pH':6.7, 'Temperature':24.5, 'Taste':0, 'Odor':1, 'Fat':3.8, 'Turbidity':1.1, 'Colour':0, 'Grade':1}

try:
    result = classify_sample(sample_input_encoded, scaler, tcn, gbm, window_size, device, feature_order_list, categorical_encoders=None)
    print("Prediction:", result)
except ValueError as e:
    print(f"Error during classification: {e}")
