import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

PROJECT_PATH = Path(__file__).resolve().parent.parent
if str(PROJECT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_PATH))
from scripts import dataset_common as dc


class VGG(nn.Module):
    def __init__(self, num_classes=176):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # True: 训练并保存最优权重；False: 只加载权重做测试集推理
    DO_TRAIN = True
    NUM_EPOCHS = 100
    PATIENCE = 15
    BATCH_SIZE = 16  # VGG 显存占用大，不够再改成 8
    CKPT_PATH = PROJECT_PATH / "outputs" / "vgg.pth"
    data_root = PROJECT_PATH / "data" / "classify-leaves"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (PROJECT_PATH / "outputs").mkdir(parents=True, exist_ok=True)

    val_tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    train_tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
        ]
    )

    ds = dc.LeavesDataset(data_root / "train.csv", data_root, transform=val_tfm)

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
            Subset(train_base, train_idx),
            batch_size=BATCH_SIZE,
            shuffle=True,
        )
        val_loader = DataLoader(
            Subset(val_base, val_idx),
            batch_size=BATCH_SIZE,
            shuffle=False,
        )

        model = VGG(num_classes=len(ds.label_to_idx)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        # Adam 用稍小 lr 更稳；需要衰减时可再打开 scheduler
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        best_val_acc = -1.0
        epochs_no_improve = 0

        for epoch in range(NUM_EPOCHS):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_correct += (outputs.argmax(dim=1) == labels).sum().item()
                train_total += labels.size(0)

            avg_train_loss = train_loss / len(train_loader)
            train_acc = train_correct / train_total

            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                    val_total += labels.size(0)

            avg_val_loss = val_loss / len(val_loader)
            val_acc = val_correct / val_total
            print(
                f"Epoch {epoch + 1}/{NUM_EPOCHS}, "
                f"train loss: {avg_train_loss:.4f}, train acc: {train_acc:.4f}, "
                f"val loss: {avg_val_loss:.4f}, val acc: {val_acc:.4f}"
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

        ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(
            f"loaded best weights (epoch {ckpt['best_epoch']}, "
            f"val acc {ckpt['best_val_acc']:.4f})"
        )
    else:
        if not CKPT_PATH.exists():
            raise FileNotFoundError(
                f"找不到权重文件: {CKPT_PATH}，请先把 DO_TRAIN=True 训练一次"
            )
        ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
        model = VGG(num_classes=ckpt["num_classes"]).to(device)
        model.load_state_dict(ckpt["model"])
        ds.label_to_idx = ckpt["label_to_idx"]
        print(f"loaded weights <- {CKPT_PATH}")

    # --- 测试集推理并写出提交文件 ---
    idx_to_label = {idx: name for name, idx in ds.label_to_idx.items()}
    test_loader = DataLoader(
        dc.LeavesTestDataset(
            data_root / "test.csv", data_root, transform=val_tfm
        ),
        batch_size=BATCH_SIZE,
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
    out_path = PROJECT_PATH / "outputs" / "submission_vgg.csv"
    submission.to_csv(out_path, index=False)
    print(f"wrote {len(submission)} rows -> {out_path}")
