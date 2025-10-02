import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from sklearn.metrics import roc_auc_score
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import datetime
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "output")
img_log_dir = os.path.join(output_dir, "img_log")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(img_log_dir, exist_ok=True)

log_file_path = os.path.join(output_dir, "detection_log.txt")

dataset_dir = "dataset"

def split_dataset(dataset_dir, train_ratio=0.7, valid_ratio=0.15, seed=42):
    random.seed(seed)
    classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    class_to_imgs = {cls: [os.path.join(dataset_dir, cls, f) 
                           for f in os.listdir(os.path.join(dataset_dir, cls))
                           if os.path.isfile(os.path.join(dataset_dir, cls, f))] for cls in classes}
    train_files, valid_files, test_files = [], [], []
    for cls, files in class_to_imgs.items():
        random.shuffle(files)
        n = len(files)
        n_train = int(n * train_ratio)
        n_valid = int(n * valid_ratio)
        train_files += [(f, cls) for f in files[:n_train]]
        valid_files += [(f, cls) for f in files[n_train:n_train+n_valid]]
        test_files  += [(f, cls) for f in files[n_train+n_valid:]]
    return train_files, valid_files, test_files

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, file_list, transform=None, to_gray=True):
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
        return img, label, os.path.basename(path)

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

train_files, valid_files, test_files = split_dataset(dataset_dir)
train_dataset_gray = CustomImageDataset(train_files, transform=train_transform_gray, to_gray=True)
valid_dataset_gray = CustomImageDataset(valid_files, transform=eval_transform_gray, to_gray=True)
test_dataset_gray  = CustomImageDataset(test_files,  transform=eval_transform_gray, to_gray=True)

batch_size = 4
train_loader_gray = DataLoader(train_dataset_gray, batch_size=batch_size, shuffle=True)
valid_loader_gray = DataLoader(valid_dataset_gray, batch_size=batch_size, shuffle=False)
test_loader_gray  = DataLoader(test_dataset_gray,  batch_size=1, shuffle=False)

device = torch.device("cpu")

def get_model(num_classes=2):
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)

model_gray = get_model()
criterion = nn.CrossEntropyLoss()
optimizer_gray = optim.Adam(model_gray.parameters(), lr=1e-4)

def train_model(model, optimizer, train_loader, valid_loader, epochs=5):
    for epoch in range(epochs):
        model.train()
        for imgs, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        model.eval()
        all_labels, all_probs = [], []
        with torch.no_grad():
            for imgs, labels, _ in valid_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                probs = torch.softmax(outputs, dim=1)[:,1]
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        auc_score = roc_auc_score(all_labels, all_probs)
        print(f"Epoch {epoch+1}/{epochs}, Val AUC: {auc_score:.4f}")
    return model

def save_gradcam_with_text(img_tensor, cam, filename, info_text):
    # 将输入tensor还原到0-1范围
    rgb_img = img_tensor.squeeze().permute(1,2,0).cpu().numpy()
    rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())

    # 确保Grad-CAM在[0,1]范围
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    # 叠加Grad-CAM
    visualization = show_cam_on_image(rgb_img, cam, use_rgb=True)

    # 用matplotlib绘制
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(visualization)
    ax.axis("off")

    # 在图像上加文字（自动换行，白底黄字）
    ax.text(
        0.02, 0.02, info_text,
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        ha="left",
        color="black",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="yellow", lw=1, alpha=0.8)
    )

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def simulate_production(model, test_loader):
    model.eval()
    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
    with open(log_file_path, "w") as log_file:
        for imgs, labels, filenames in tqdm(test_loader, desc="Simulating Production"):
            start_time = time.time()
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)[:,1].detach().cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            end_time = time.time()
            total_time = end_time - start_time
            seq_no = filenames[0]
            detect_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            pred_label = "good" if preds[0]==0 else "bad"
            pred_prob = float(probs[0])
            log_line = f"{seq_no}, {detect_time}, {pred_label}, {pred_prob:.4f}, {total_time:.4f}s"
            log_file.write(log_line+"\n")
            log_file.flush()
            info_text = f"Seq: {seq_no}\nTime: {detect_time}\nPred: {pred_label}\nProb: {pred_prob:.4f}\nDuration: {total_time:.4f}s"
            grayscale_cam = cam(input_tensor=imgs, targets=[ClassifierOutputTarget(preds[0])])[0]
            save_path = os.path.join(img_log_dir, f"{seq_no}_gradcam.png")
            save_gradcam_with_text(imgs[0], grayscale_cam, save_path, info_text)

if __name__ == "__main__":
    model_gray = train_model(model_gray, optimizer_gray, train_loader_gray, valid_loader_gray, epochs=5)
    simulate_production(model_gray, test_loader_gray)
    print(f"Simulation finished. Logs in {log_file_path}, Grad-CAM images in {img_log_dir}")
