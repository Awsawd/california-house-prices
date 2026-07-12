#探索数据
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.nanops import F
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#读取数据
data = pd.read_csv(PROJECT_ROOT / 'data' / 'train.csv')

# 定义特征列和目标变量
NUMERIC_FEATURES = [
    'Year built',
    'Bathrooms',
    'Full bathrooms',
    'Total interior livable area',
    'Garage spaces',
    'Elementary School Score',
    'High School Score',
    'Tax assessed value',
    'Annual tax amount',
    'Bedrooms',
    'Lot',
    'Middle School Score'
]
CAT_FEATURES = ['Type','City']

TARGET = 'Sold Price'

# 从 data 里取出特征和目标变量
df = data[NUMERIC_FEATURES + CAT_FEATURES + [TARGET]]

df['Bedrooms'] = pd.to_numeric(df['Bedrooms'], errors='coerce')

#划分训练集和验证集
X = df[NUMERIC_FEATURES + CAT_FEATURES]
y = df[TARGET]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

top_cities = X_train['City'].value_counts().nlargest(20).index
X_train['City'] = X_train['City'].where(X_train['City'].isin(top_cities), 'Other')
X_val['City']   = X_val['City'].where(X_val['City'].isin(top_cities), 'Other')

#中位值填充（统计量仅用训练集）
median = X_train[NUMERIC_FEATURES].median()
X_train[NUMERIC_FEATURES] = X_train[NUMERIC_FEATURES].fillna(median)
X_val[NUMERIC_FEATURES] = X_val[NUMERIC_FEATURES].fillna(median)

# 清除异常值：所有数值列统一用训练集 1%/99% 分位数 clip
clip_lower = X_train[NUMERIC_FEATURES].quantile(0.01)
clip_upper = X_train[NUMERIC_FEATURES].quantile(0.99)
X_train[NUMERIC_FEATURES] = X_train[NUMERIC_FEATURES].clip(lower=clip_lower, upper=clip_upper, axis=1)
X_val[NUMERIC_FEATURES] = X_val[NUMERIC_FEATURES].clip(lower=clip_lower, upper=clip_upper, axis=1)


# 目标 log1p 变换：缓解房价右偏，训练在 log 空间进行，评估时再 expm1 转回美元
y_train_log = np.log1p(y_train.values)

#标准化
scaler = StandardScaler()
scaler.fit(X_train[NUMERIC_FEATURES])
X_train_num = scaler.transform(X_train[NUMERIC_FEATURES])
X_val_num = scaler.transform(X_val[NUMERIC_FEATURES])

X_train_cat = pd.get_dummies(X_train[['Type','City']], prefix=['Type','City']).astype(np.float32)
X_val_cat = pd.get_dummies(X_val[['Type','City']], prefix=['Type','City']).astype(np.float32)

X_val_cat = X_val_cat.reindex(columns=X_train_cat.columns, fill_value=0)

X_train_scaled = np.hstack([X_train_num, X_train_cat.values]).astype(np.float32)
X_val_scaled = np.hstack([X_val_num, X_val_cat.values]).astype(np.float32)

input_size = X_train_scaled.shape[1]
print(f"Input size: {input_size}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 转换为 Tensor（先放 CPU，训练时按 batch 送到 GPU，节省显存）
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train_log, dtype=torch.float32).reshape(-1, 1)
X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)

batch_size = 256
train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=batch_size,
    shuffle=True,
)

# print(X_train_t.shape)    # 期望 torch.Size([37951, 9])
# print(X_train_t.device)   # 期望 cuda:0
# print(y_train_t.shape)    # 期望 torch.Size([37951, 1])

#定义模型
class HousePriceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

model = HousePriceModel().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

best_val_mae = float('inf')
patience = 20
patience_counter = 0
best_model_state = None

for epoch in range(150):
    epoch_loss = 0.0
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Loss (log space): {avg_loss:.4f}")


    model.eval()
    with torch.no_grad():
        X_val_gpu = X_val_t.to(device)
        y_val_pred_log = model(X_val_gpu)
        y_pred = np.expm1(y_val_pred_log.cpu().numpy().flatten())
        y_true = y_val.values.flatten()
        val_mae = np.mean(np.abs(y_true - y_pred))
    
    if val_mae < best_val_mae:
        best_val_mae = val_mae
        patience_counter = 0
        best_model_state = model.state_dict()
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

model.load_state_dict(best_model_state)

model.eval()
with torch.no_grad():
    X_val_gpu = X_val_t.to(device)
    y_val_pred_log = model(X_val_gpu)
    y_pred = np.expm1(y_val_pred_log.cpu().numpy().flatten())
    y_true = y_val.values.flatten()

mae = np.mean(np.abs(y_true - y_pred))
rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

print(f"Validation MAE:  ${mae:,.0f}")
print(f"Validation RMSE: ${rmse:,.0f}")

(PROJECT_ROOT / 'outputs').mkdir(exist_ok=True)
torch.save(best_model_state, PROJECT_ROOT / 'outputs' / 'best_model.pt')


#预测
test_data = pd.read_csv(PROJECT_ROOT / 'data' / 'test.csv')
test_ids = test_data['Id']
X_test = test_data[NUMERIC_FEATURES + CAT_FEATURES]
X_test['Bedrooms'] = pd.to_numeric(X_test['Bedrooms'], errors='coerce')
X_test['City'] = X_test['City'].where(X_test['City'].isin(top_cities), 'Other')
X_test[NUMERIC_FEATURES] = X_test[NUMERIC_FEATURES].fillna(median)
X_test[NUMERIC_FEATURES] = X_test[NUMERIC_FEATURES].clip(lower=clip_lower, upper=clip_upper, axis=1)
X_test_num = scaler.transform(X_test[NUMERIC_FEATURES])
X_test_cat = pd.get_dummies(X_test[['Type','City']], prefix=['Type','City']).astype(np.float32)
X_test_cat = X_test_cat.reindex(columns=X_train_cat.columns, fill_value=0)
X_test_scaled = np.hstack([X_test_num, X_test_cat.values]).astype(np.float32)
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)


model.eval()
with torch.no_grad():
    y_test_pred_log = model(X_test_t)
    y_test_pred = np.expm1(y_test_pred_log.cpu().numpy().flatten())

submission = pd.DataFrame({
    'Id': test_ids,
    'Sold Price': y_test_pred
})
(PROJECT_ROOT / 'outputs').mkdir(exist_ok=True)
submission.to_csv(PROJECT_ROOT / 'outputs' / 'submission.csv', index=False)
print(f"Saved {len(submission)} predictions to outputs/submission.csv")