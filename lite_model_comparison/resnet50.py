import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from sklearn.metrics import roc_auc_score
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import platform
import psutil

current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "output")
os.makedirs(output_dir, exist_ok=True)

print(f"All outputs will be saved to: {output_dir}")

report_path = os.path.join(output_dir, "report.txt")  # 汇总报告文件

def split_dataset(dataset_dir, train_ratio=0.7, valid_ratio=0.15, seed=42):
    random.seed(seed)
    classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    print(f"Classes found: {classes}")
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def get_model(num_classes=2):
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)

model_gray = get_model()
criterion = nn.CrossEntropyLoss()
optimizer_gray = optim.Adam(model_gray.parameters(), lr=1e-4)

def log_hardware_info():
    info_lines = []

    cpu_name = platform.processor() or "Unknown CPU"
    cpu_count = psutil.cpu_count(logical=True)
    cpu_freq = psutil.cpu_freq()
    mem = psutil.virtual_memory()

    info_lines.append(f"CPU: {cpu_name}")
    info_lines.append(f"CPU Cores (logical): {cpu_count}")
    if cpu_freq:
        info_lines.append(f"CPU Frequency: {cpu_freq.max:.2f} MHz")
    info_lines.append(f"RAM: {mem.total / (1024**3):.2f} GB")

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_props = torch.cuda.get_device_properties(i)
            info_lines.append(f"GPU {i}: {gpu_name}")
            info_lines.append(f"  - Total Memory: {gpu_props.total_memory / (1024**3):.2f} GB")
            info_lines.append(f"  - CUDA Capability: {gpu_props.major}.{gpu_props.minor}")
    else:
        info_lines.append("No CUDA GPU available.")

    print("\n===== Hardware Info =====")
    for line in info_lines:
        print(line)

    with open(report_path, "w") as f:
        f.write("===== Hardware Info =====\n")
        for line in info_lines:
            f.write(line + "\n")
        f.write("\n")
    print(f"Hardware info saved to {report_path}\n")

def train_model(model, optimizer, train_loader, valid_loader, epochs=5):
    for epoch in range(epochs):
        model.train()
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        all_labels, all_probs = [], []
        with torch.no_grad():
            for imgs, labels in valid_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                probs = torch.softmax(outputs, dim=1)[:,1]
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        auc_score = roc_auc_score(all_labels, all_probs)
        print(f"Epoch {epoch+1}/{epochs}, Val AUC: {auc_score:.4f}")
    return model

def benchmark_inference(model, test_loader, model_name="gray"):
    model.eval()
    times = {}

    for device_type in ["cuda", "cpu"]:
        if device_type == "cuda" and not torch.cuda.is_available():
            continue

        device_bench = torch.device(device_type)
        model_bench = model.to(device_bench)

        total_time, total_samples = 0.0, 0
        with torch.no_grad():
            for imgs, _ in tqdm(test_loader, desc=f"Benchmark on {device_type.upper()}"):
                imgs = imgs.to(device_bench)
                start = time.time()
                _ = model_bench(imgs)
                end = time.time()
                total_time += (end - start)
                total_samples += imgs.size(0)

        avg_time = total_time / total_samples
        times[device_type] = avg_time
        print(f"{device_type.upper()} avg time: {avg_time:.6f} s/image")

    with open(report_path, "a") as f:
        f.write("===== Inference Speed =====\n")
        for k,v in times.items():
            f.write(f"{k.upper()} avg inference time: {v:.6f} s/image\n")

    plt.figure()
    plt.bar(times.keys(), times.values(), color=['blue','orange'][:len(times)])
    plt.ylabel("Avg Time per Image (s)")
    plt.title("Grayscale Inference Speed (CPU vs GPU)")
    for i, (k,v) in enumerate(times.items()):
        plt.text(i, v, f"{v:.6f}s", ha="center", va="bottom")
    save_path = os.path.join(output_dir, f"inference_speed_comparison_{model_name}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Comparison plot saved: {save_path}")


if __name__ == "__main__":
    log_hardware_info()

    print("Training Grayscale model...")
    model_gray = train_model(model_gray, optimizer_gray, train_loader_gray, valid_loader_gray, epochs=5)

    print("Benchmarking inference speed (Grayscale model)...")
    benchmark_inference(model_gray, test_loader_gray, model_name="gray")

    print(f"\nFinal report saved at: {report_path}")
