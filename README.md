# California House Prices

个人学习项目：加州房价预测（EDA → Random Forest 基线 → PyTorch MLP）。

## 环境

- Python 3.11
- Conda 环境：`california-dl`（PyTorch GPU + pandas + scikit-learn + joblib）
- 解释器路径示例：`H:\conda_envs\california-dl\python.exe`

## 项目结构

```
data/                  原始 CSV（train / test / sample_submission）
scripts/
  01_eda.py              数据探索
  02_baseline.py         Random Forest 基线
  03_deep_learning.py    PyTorch MLP + 早停 + test 预测
outputs/                 模型、submission 等（本地生成，不入库）
```

## 特征与预处理

两套脚本共用相似的特征工程思路：

- **数值特征（14）**：Year built、Bathrooms、面积、Lot、学校评分、税评、Bedrooms、house_age、Listed Price 等
- **类别特征（4）**：Type、City、Zip、Region（Top 20 + Other）
- **预处理**：split 后仅用训练集统计 — median 填充 → 1%/99% clip → One-Hot
- **目标**：训练时对 `Sold Price` 做 `log1p`（仅 MLP）；评估与提交用美元

MLP 额外使用 `StandardScaler`；Random Forest 不需要标准化。

## 模型与验证结果（80/20, random_state=42）

| 脚本 | 模型 | Val MAE | Val RMSE |
|------|------|---------|----------|
| `02_baseline.py` | Random Forest (100 trees) | ~$181,000 | ~$1,122,000 |
| `03_deep_learning.py` | MLP 64→32 + Dropout + 早停 | ~$186,000 | ~$1,108,000 |

`Listed Price` 对两者都是强特征；表格数据上 Random Forest 略优。

## 运行

```bash
conda activate california-dl

# Random Forest 基线
python scripts/02_baseline.py

# MLP（调参模式：80/20 + 早停 + 保存模型 + 预测 test）
python scripts/03_deep_learning.py
```

### MLP 两种模式（`03_deep_learning.py` 顶部配置）

| `FULL_TRAIN` | 说明 |
|--------------|------|
| `False` | 80/20 划分，早停，打印 Val MAE，保存 `best_model.pt` 与 `preprocess.joblib` |
| `True` | 100% 训练集重训，固定 `FULL_TRAIN_EPOCHS` 轮，生成最终 `submission.csv` |

全量训练前，先在 `FULL_TRAIN=False` 下跑一遍，将早停 epoch 填入 `FULL_TRAIN_EPOCHS`。

## 输出文件

```
outputs/
  best_model.pt          MLP 权重
  preprocess.joblib      预处理统计量、scaler、列名等
  submission.csv         test 集预测（Id, Sold Price）
```

## 数据

Kaggle 风格加州房价数据集：`train.csv`（~47k）、`test.csv`（~31k）、`sample_submission.csv`。
