import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import platform
import psutil
import tempfile
import numpy as np

# ---------------- 路径设置 ----------------
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "output")
os.makedirs(output_dir, exist_ok=True)
print(f"All outputs will be saved to: {output_dir}")

# ---------------- 数据集 ----------------
def split_dataset(dataset_dir, train_ratio=0.7, valid_ratio=0.15, seed=42):
    random.seed(seed)
    classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    class_to_imgs = {cls: [os.path.join(dataset_dir, cls, f) 
                           for f in os.listdir(os.path.join(dataset_dir, cls))
                           if os.path.isfile(os.path.join(dataset_dir, cls, f))] 
                     for cls in classes}
    train_files, valid_files, test_files = [], [], []
    for cls, files in class_to_imgs.items():
        random.shuffle(files)
        n = len(files)
        n_train = int(n * train_ratio)
        n_valid = int(n * valid_ratio)
        train_files += [(f, cls) for f in files[:n_train]]
        valid_files += [(f, cls) for f in files[n_train:n_train+n_valid]]
        test_files  += [(f, cls) for f in files[n_train+n_valid:]]
    print(f"Train: {len(train_files)}, Valid: {len(valid_files)}, Test: {len(test_files)}")
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

train_transform_gray = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize([0.485]*3, [0.229]*3)
])
eval_transform_gray = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485]*3, [0.229]*3)
])

dataset_dir = "dataset"
train_files, valid_files, test_files = split_dataset(dataset_dir)

train_dataset_gray = CustomImageDataset(train_files, transform=train_transform_gray, to_gray=True)
valid_dataset_gray = CustomImageDataset(valid_files, transform=eval_transform_gray, to_gray=True)
test_dataset_gray  = CustomImageDataset(test_files,  transform=eval_transform_gray, to_gray=True)

batch_size = 4
train_loader_gray = DataLoader(train_dataset_gray, batch_size=batch_size, shuffle=True)
valid_loader_gray = DataLoader(valid_dataset_gray, batch_size=batch_size, shuffle=False)
test_loader_gray  = DataLoader(test_dataset_gray,  batch_size=batch_size, shuffle=False)

# ---------------- 设备 ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- 模型列表 ----------------
model_names = ["resnet50", "resnet18", "mobilenet_v2", "mobilenet_v3_large", "squeezenet1_0"]

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
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif name == "squeezenet1_0":
        model = models.squeezenet1_0(pretrained=True)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1))
    else:
        raise ValueError(f"Unknown model name {name}")
    return model.to(device)

criterion = nn.CrossEntropyLoss()

# ---------------- 训练 ----------------
def train_model(model, optimizer, train_loader, valid_loader, epochs=5):
    for epoch in range(epochs):
        model.train()
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model

# ---------------- 测试及指标记录 ----------------
def evaluate_model(model, test_loader):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    process = psutil.Process(os.getpid())
    times = {"cpu":0.0, "cuda":0.0}

    ram_list_cpu = []
    vram_list_gpu = []

    for device_type in ["cpu", "cuda"]:
        if device_type=="cuda" and not torch.cuda.is_available():
            continue
        device_bench = torch.device(device_type)
        model.to(device_bench)

        total_time, total_samples = 0.0, 0
        prev_ram = process.memory_info().rss / (1024**2)
        prev_vram = torch.cuda.memory_allocated(device_bench)/(1024**2) if device_type=="cuda" else 0
        ram_list_dev, vram_list_dev = [], []

        with torch.no_grad():
            for imgs, labels in tqdm(test_loader, desc=f"Evaluating on {device_type}"):
                imgs, labels = imgs.to(device_bench), labels.to(device_bench)
                start = time.time()
                outputs = model(imgs)
                end = time.time()
                total_time += (end - start)
                total_samples += imgs.size(0)

                probs = torch.softmax(outputs, dim=1)[:,1].detach()
                preds = torch.argmax(outputs, dim=1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

                # CPU RAM 增量
                curr_ram = process.memory_info().rss / (1024**2)
                delta_ram = curr_ram - prev_ram
                ram_list_dev.append(delta_ram)
                prev_ram = curr_ram

                # GPU VRAM 增量
                if device_type=="cuda":
                    curr_vram = torch.cuda.memory_allocated(device_bench)/(1024**2)
                    delta_vram = curr_vram - prev_vram
                    vram_list_dev.append(delta_vram)
                    prev_vram = curr_vram

        times[device_type] = total_time/total_samples
        if device_type=="cpu":
            ram_list_cpu = ram_list_dev
        if device_type=="cuda":
            vram_list_gpu = vram_list_dev

    # 模型大小
    param_size = sum(p.numel()*p.element_size() for p in model.parameters()) / (1024**2)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmpfile:
        torch.save(model.state_dict(), tmpfile.name)
        state_dict_size = os.path.getsize(tmpfile.name)/(1024**2)
    os.remove(tmpfile.name)
    total_model_size = param_size + state_dict_size

    return times, ram_list_cpu, vram_list_gpu if torch.cuda.is_available() else None, param_size, state_dict_size, total_model_size, all_labels, all_preds, all_probs

# ---------------- 运行 ----------------
results = {}
for name in model_names:
    print(f"\n==== Training {name} ====")
    model = get_model(name)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    model = train_model(model, optimizer, train_loader_gray, valid_loader_gray, epochs=3)
    print(f"==== Evaluating {name} ====")
    times, ram_list_cpu, ram_list_gpu, param_size, state_dict_size, total_model_size, labels, preds, probs = evaluate_model(model, test_loader_gray)
    results[name] = {
        "times": times,
        "ram_list": ram_list_cpu,
        "vram_list": ram_list_gpu,
        "param_size": param_size,
        "state_dict_size": state_dict_size,
        "total_model_size": total_model_size,
        "labels": labels,
        "preds": preds,
        "probs": probs
    }

# ---------------- 可视化 ----------------
for model_name, res in results.items():
    # 1. CPU RAM 增量
    plt.figure(figsize=(10,6))
    plt.plot(res["ram_list"], label=f"CPU RAM Δ (MB)")
    plt.xlabel("Inference Step")
    plt.ylabel("CPU RAM Increment (MB)")
    plt.title(f"CPU RAM Increment - {model_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"cpu_ram_increment_{model_name}.png"))
    plt.close()

    # 2. GPU VRAM 增量
    if res["vram_list"]:
        plt.figure(figsize=(10,6))
        plt.plot(res["vram_list"], label=f"GPU VRAM Δ (MB)")
        plt.xlabel("Inference Step")
        plt.ylabel("GPU VRAM Increment (MB)")
        plt.title(f"GPU VRAM Increment - {model_name}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"gpu_vram_increment_{model_name}.png"))
        plt.close()

    # 3. 模型大小
    plt.figure(figsize=(6,6))
    sizes = [res["param_size"], res["state_dict_size"], res["total_model_size"]]
    labels_size = ["Params","State_dict","Total"]
    plt.bar(labels_size, sizes, color=['skyblue','orange','green'])
    plt.ylabel("Size (MB)")
    plt.title(f"Model Size - {model_name}")
    for i,v in enumerate(sizes):
        plt.text(i,v,f"{v:.2f} MB",ha='center',va='bottom')
    plt.savefig(os.path.join(output_dir, f"model_size_{model_name}.png"))
    plt.close()

    # 4. 推理速度
    plt.figure(figsize=(6,6))
    plt.bar(res["times"].keys(), res["times"].values(), color=['blue','orange'][:len(res["times"])])
    plt.ylabel("Avg Time per Image (s)")
    plt.title(f"Inference Speed - {model_name}")
    for i,(k,v) in enumerate(res["times"].items()):
        plt.text(i,v,f"{v:.6f}s",ha='center',va='bottom')
    plt.savefig(os.path.join(output_dir, f"inference_speed_{model_name}.png"))
    plt.close()

# ---------------- ROC & Confusion Matrix ----------------
# ROC 对比图
plt.figure()
for model_name, res in results.items():
    fpr, tpr, _ = roc_curve(res["labels"], res["probs"])
    roc_auc_val = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{model_name} (AUC={roc_auc_val:.4f})")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "roc_comparison.png"))
plt.close()

# 混淆矩阵
for model_name, res in results.items():
    cm = confusion_matrix(res["labels"], res["preds"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix - {model_name}")
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_{model_name}.png"))
    plt.close()

print("All evaluation and plots saved in output directory.")
