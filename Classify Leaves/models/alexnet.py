import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

PROJECT_PATH = Path(__file__).resolve().parent.parent
if str(PROJECT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_PATH))
from scripts import dataset_common as dc


class AlexNet(nn.Module):
    def __init__(self, num_classes=176):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )
        self.classifier.add_module('fc', nn.Linear(4096, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x




if __name__ == "__main__":
    # True: 训练并保存权重；False: 加载已有权重，只做测试集推理
    DO_TRAIN = True
    NUM_EPOCHS = 100
    # 验证准确率连续 PATIENCE 个 epoch 没有提升就停止
    PATIENCE = 10
    CKPT_PATH = PROJECT_PATH / "outputs" / "alexnet.pth"

    data_root = PROJECT_PATH / "data" / "classify-leaves"
    # 验证/测试：固定预处理，不做随机增强
    val_tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    # 训练：随机翻转 + 小角度旋转，减轻过拟合
    train_tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
        ]
    )
    # 先建一份只为拿 label 映射与划分索引（transform 用 val 即可）
    ds = dc.LeavesDataset(data_root / "train.csv", data_root, transform=val_tfm)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (PROJECT_PATH / "outputs").mkdir(parents=True, exist_ok=True)

    if DO_TRAIN:
        indices = np.arange(len(ds))
        train_idx, val_idx = train_test_split(
            indices,
            test_size=0.2,
            random_state=42,
            stratify=ds.targets,
        )
        train_base = dc.LeavesDataset(
            data_root / "train.csv",
            data_root,
            transform=train_tfm,
            label_to_idx=ds.label_to_idx,
        )
        val_base = dc.LeavesDataset(
            data_root / "train.csv",
            data_root,
            transform=val_tfm,
            label_to_idx=ds.label_to_idx,
        )
        train_loader = DataLoader(
            Subset(train_base, train_idx), batch_size=32, shuffle=True
        )
        val_loader = DataLoader(
            Subset(val_base, val_idx), batch_size=32, shuffle=False
        )

        model = AlexNet(num_classes=len(ds.label_to_idx)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.01, momentum=0.9
        )

        best_val_acc = -1.0
        epochs_no_improve = 0

        for epoch in range(NUM_EPOCHS):
            model.train()
            train_loss = 0.0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = model(imgs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            avg_train_loss = train_loss / len(train_loader)

            # 每个 epoch 验证一次
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    pred = model(imgs).argmax(dim=1)
                    correct += (pred == labels).sum().item()
                    total += labels.size(0)
            val_acc = correct / total
            print(
                f"Epoch {epoch + 1}, train loss: {avg_train_loss:.4f}, "
                f"val acc: {val_acc:.4f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "label_to_idx": ds.label_to_idx,
                        "num_classes": len(ds.label_to_idx),
                        "best_val_acc": best_val_acc,
                        "best_epoch": epoch + 1,
                    },
                    CKPT_PATH,
                )
                print(f"  ↑ new best, saved -> {CKPT_PATH}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= PATIENCE:
                    print(
                        f"early stop at epoch {epoch + 1} "
                        f"(best val acc {best_val_acc:.4f})"
                    )
                    break

        # 测试前加载验证集上最好的权重（不是发散后的最后一轮）
        ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(
            f"loaded best weights (epoch {ckpt.get('best_epoch', '?')}, "
            f"val acc {ckpt.get('best_val_acc', best_val_acc):.4f})"
        )
    else:
        if not CKPT_PATH.exists():
            raise FileNotFoundError(
                f"找不到权重文件: {CKPT_PATH}，请先把 DO_TRAIN=True 训练一次"
            )
        ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
        model = AlexNet(num_classes=ckpt["num_classes"]).to(device)
        model.load_state_dict(ckpt["model"])
        ds.label_to_idx = ckpt["label_to_idx"]
        print(f"loaded weights <- {CKPT_PATH}")

    # --- 测试集推理并写出提交文件 ---
    idx_to_label = {idx: name for name, idx in ds.label_to_idx.items()}
    test_loader = DataLoader(
        dc.LeavesTestDataset(
            data_root / "test.csv", data_root, transform=val_tfm
        ),
        batch_size=32,
        shuffle=False,
    )

    model.eval()
    image_paths = []
    pred_labels = []
    with torch.no_grad():
        for imgs, paths in test_loader:
            imgs = imgs.to(device)
            pred = model(imgs).argmax(dim=1).cpu().tolist()
            image_paths.extend(paths)
            pred_labels.extend(idx_to_label[i] for i in pred)

    submission = pd.DataFrame({"image": image_paths, "label": pred_labels})
    out_path = PROJECT_PATH / "outputs" / "submission.csv"
    submission.to_csv(out_path, index=False)
    print(f"wrote {len(submission)} rows -> {out_path}")
