import os 
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import tempfile
import numpy as np
from collections import Counter

# ---------------- 配置（可调） ----------------
use_class_weight_loss = True     # ✅ 改成加权 CrossEntropyLoss
use_focal_loss = False           # ✅ 不再使用 Focal Loss
use_weighted_sampler = True      
batch_size = 4
decision_threshold = 0.3         # ✅ 调整阈值，降低 FNR

# ---------------- 路径设置 ----------------
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "output")
os.makedirs(output_dir, exist_ok=True)
print(f"All outputs will be saved to: {output_dir}")

# ---------------- 数据集拆分 ----------------
def split_dataset(dataset_dir, train_ratio=0.7, valid_ratio=0.15, seed=42):
    random.seed(seed)
    classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    class_to_imgs = {cls: [os.path.join(dataset_dir, cls, f) 
                           for f in os.listdir(os.path.join(dataset_dir, cls))
                           if os.path.isfile(os.path.join(dataset_dir, cls, f))] 
                     for cls in classes}

    # 整理成列表 [(path, cls)]
    all_files = [(f, cls) for cls, files in class_to_imgs.items() for f in files]
    labels = [cls for _, cls in all_files]

    # 先拆 test（保证 stratify）
    test_size = 1 - (train_ratio + valid_ratio)
    train_val_files, test_files = train_test_split(
        all_files, test_size=test_size, stratify=labels, random_state=seed
    )

    # 再从 train_val 拆 valid
    train_size_adjusted = train_ratio / (train_ratio + valid_ratio)  # 比例调整
    train_files, valid_files = train_test_split(
        train_val_files, test_size=1-train_size_adjusted,
        stratify=[cls for _, cls in train_val_files], random_state=seed
    )

    print(f"Train: {len(train_files)}, Valid: {len(valid_files)}, Test: {len(test_files)}")
    def count_classes(file_list):
        labels = [cls for _, cls in file_list]
        return Counter(labels)

    print(f"Train: {len(train_files)}, distribution: {count_classes(train_files)}")
    print(f"Valid: {len(valid_files)}, distribution: {count_classes(valid_files)}")
    print(f"Test:  {len(test_files)}, distribution: {count_classes(test_files)}")

    return train_files, valid_files, test_files

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, file_list, transform=None, to_gray=False):
        self.file_list = file_list
        self.transform = transform
        self.to_gray = to_gray
        self.class_to_idx = {'good':0, 'bad':1}
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, idx):
        path, cls = self.file_list[idx]
        img = Image.open(path).convert("RGB")
        if self.to_gray:
            img = img.convert("L").convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.class_to_idx[cls]
        return img, label

# ---------------- transforms ----------------
train_transform_gray = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=3),
    # ✅ 调整增强策略：只做轻微增强，避免破坏坏品特征
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485]*3, [0.229]*3)
])
eval_transform_gray = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485]*3, [0.229]*3)
])

# ---------------- 准备数据 ----------------
dataset_dir = "dataset"
train_files, valid_files, test_files = split_dataset(dataset_dir)

train_dataset_gray = CustomImageDataset(train_files, transform=train_transform_gray, to_gray=True)
valid_dataset_gray = CustomImageDataset(valid_files, transform=eval_transform_gray, to_gray=True)
test_dataset_gray  = CustomImageDataset(test_files,  transform=eval_transform_gray, to_gray=True)

# 计算训练集类别分布
def get_label_list(file_list):
    label_map = {'good':0, 'bad':1}
    return [label_map[cls] for (_, cls) in file_list]

train_labels = get_label_list(train_files)
good_count = sum(1 for l in train_labels if l == 0)
bad_count = sum(1 for l in train_labels if l == 1)
print(f"Train distribution -> good: {good_count}, bad: {bad_count}")

# 创建 sampler
if use_weighted_sampler:
    weight_per_class = {0: 1.0, 1: (good_count / (bad_count + 1e-8))}
    sample_weights = [weight_per_class[l] for l in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader_gray = DataLoader(train_dataset_gray, batch_size=batch_size, sampler=sampler, num_workers=2)
else:
    train_loader_gray = DataLoader(train_dataset_gray, batch_size=batch_size, shuffle=True, num_workers=2)

valid_loader_gray = DataLoader(valid_dataset_gray, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader_gray  = DataLoader(test_dataset_gray, batch_size=batch_size, shuffle=False, num_workers=2)

# ---------------- 设备 ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 模型开关（True 表示跑，False 表示跳过）
model_flags = {
    "resnet50": True,
    "resnet18": True,
    "mobilenet_v2": True,
    "mobilenet_v3_large": True,
    "squeezenet1_0": True,
    "convnextv2_tiny": True,
    "swin_v2_t": True,
    "maxvit_t": True,
    "resnext101_64x4d": True,
}

def get_model(name, num_classes=2):
    if name == "resnet50":
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif name == "resnet18":
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(pretrained=True)
        try:
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        except Exception:
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif name == "squeezenet1_0":
        model = models.squeezenet1_0(pretrained=True)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1))

    elif name == "convnextv2_tiny": 
        from torchvision.models import convnext_large
        model = convnext_large(pretrained=True)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)

    elif name == "swin_v2_t":
        from torchvision.models import swin_v2_b
        model = swin_v2_b(pretrained=True)
        model.head = nn.Linear(model.head.in_features, num_classes)

    elif name == "maxvit_t":
        from torchvision.models import maxvit_t
        model = maxvit_t(pretrained=True)
        model.classifier[5] = nn.Linear(model.classifier[5].in_features, num_classes)
    elif name == "resnext101_64x4d":
        model = models.resnext101_64x4d(weights="IMAGENET1K_V1")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unknown model name {name}")

    return model.to(device)

# ---------------- 损失函数（改成加权 CE） ----------------
class_weights = None
if use_class_weight_loss:
    w_good = 1.0
    w_bad = float(good_count / (bad_count + 1e-8)) * 2.0  # ✅ 再提高一点坏品权重
    class_weights = torch.tensor([w_good, w_bad], dtype=torch.float).to(device)
    print(f"Using class weights for loss: good={w_good}, bad={w_bad:.3f}")
 
def get_criterion():
    if use_class_weight_loss:
        return nn.CrossEntropyLoss(weight=class_weights)
    else:
        return nn.CrossEntropyLoss()

# ---------------- 验证 FNR + FPR ----------------
def calculate_fnr_fpr(model, valid_loader):
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for imgs, labels in valid_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)[:,1]
            preds = (probs > decision_threshold).long()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    cm = confusion_matrix(all_labels, all_preds, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    fnr = fn / (fn + tp + 1e-8)
    fpr = fp / (fp + tn + 1e-8)
    return fnr, fpr

# ---------------- 复合指标（优先 FNR，次要 FPR） ----------------
def combined_score(fnr, fpr, alpha=0.6):
    """
    alpha 控制权重：越大越优先 FNR
    """
    return alpha * fnr + (1 - alpha) * fpr

# ---------------- 训练 + Early Stopping (优先 FNR, 次要 FPR) ----------------
def train_model_early_stopping(model, optimizer, train_loader, valid_loader, criterion,
                               max_epochs=20, patience=3, alpha=0.8):
    best_score = float('inf')  # 最小化复合指标
    trigger_times = 0
    best_model_state = None

    for epoch in range(max_epochs):
        # ---- 训练 ----
        model.train()
        running_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs} Training"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)

        # ---- 验证 (复合指标) ----
        fnr, fpr = calculate_fnr_fpr(model, valid_loader)
        score = combined_score(fnr, fpr, alpha)
        print(f"Epoch {epoch+1} - Train Loss: {epoch_train_loss:.4f} - FNR: {fnr:.4f} - FPR: {fpr:.4f} - Score: {score:.4f}")

        if score < best_score - 1e-4:
            best_score = score
            trigger_times = 0
            best_model_state = model.state_dict()
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model

# ---------------- 测试及指标 ----------------
def evaluate_model(model, test_loader):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    total_infer_time = 0.0
    total_samples = 0

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc=f"Evaluating"):
            imgs, labels = imgs.to(device), labels.to(device)
            start = time.time()
            outputs = model(imgs)
            end = time.time()
            infer_time = end - start
            total_infer_time += infer_time
            total_samples += imgs.size(0)

            probs = torch.softmax(outputs, dim=1)[:,1]
            preds = (probs > decision_threshold).long()  # ✅ 使用调整后的阈值

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    fnr = fn / (fn + tp + 1e-8)
    fpr = fp / (fp + tn + 1e-8)

    # 模型大小
    param_size = sum(p.numel()*p.element_size() for p in model.parameters()) / (1024**2)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmpfile:
        torch.save(model.state_dict(), tmpfile.name)
        state_dict_size = os.path.getsize(tmpfile.name)/(1024**2)
    os.remove(tmpfile.name)
    total_model_size = param_size + state_dict_size

    avg_infer_time = total_infer_time / (total_samples + 1e-8)

    return all_labels, all_preds, all_probs, total_model_size, avg_infer_time, fnr, fpr

# ---------------- 运行 benchmark ----------------
if __name__ == '__main__':
    summary = []
    for name, flag in model_flags.items():
        if not flag:
            print(f"⏩ Skipping {name} ...")
            continue
        print(f"\n==== Training {name} with Early Stopping ====")
        model = get_model(name)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion = get_criterion()  
        model = train_model_early_stopping(
            model, optimizer,
            train_loader_gray, valid_loader_gray,
            criterion, max_epochs=20, patience=3
        )

        print(f"==== Evaluating {name} ====")
        labels, preds, probs, size, avg_time, fnr, fpr = evaluate_model(model, test_loader_gray)
        summary.append((name, fnr, fpr, size, avg_time))

    # 排序 FNR > FPR > Size > AvgTime
    summary = sorted(summary, key=lambda x: (x[1], x[2], x[3], x[4]))

    # 输出 benchmark 表格
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axis('off')
    columns = ["Model", "FNR (↓)", "FPR (↓)", "Size (MB, ↓)", "Avg Time (s/img, ↓)"]
    table_data = [[name, f"{fnr:.4f}", f"{fpr:.4f}", f"{size:.2f}", f"{avg_time*1000:.2f} ms"] 
                  for name, fnr, fpr, size, avg_time in summary]
    table = ax.table(cellText=table_data, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    plt.title("Benchmark Comparison (Early Stopping on FNR)", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "benchmark_comparison.png"))
    plt.close()

    print("✅ Benchmark comparison saved to output/benchmark_comparison.png")
