import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time


current_dir = os.path.dirname(os.path.abspath(__file__))

output_dir = os.path.join(current_dir, "output")
os.makedirs(output_dir, exist_ok=True)

print(f"All outputs will be saved to: {output_dir}")

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
    print(f"Train samples: {len(train_files)}, Valid samples: {len(valid_files)}, Test samples: {len(test_files)}")
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

train_transform_rgb = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
eval_transform_rgb = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

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

train_dataset_rgb = CustomImageDataset(train_files, transform=train_transform_rgb)
valid_dataset_rgb = CustomImageDataset(valid_files, transform=eval_transform_rgb)
test_dataset_rgb  = CustomImageDataset(test_files,  transform=eval_transform_rgb)

train_dataset_gray = CustomImageDataset(train_files, transform=train_transform_gray, to_gray=True)
valid_dataset_gray = CustomImageDataset(valid_files, transform=eval_transform_gray, to_gray=True)
test_dataset_gray  = CustomImageDataset(test_files,  transform=eval_transform_gray, to_gray=True)

batch_size = 4
train_loader_rgb = DataLoader(train_dataset_rgb, batch_size=batch_size, shuffle=True)
valid_loader_rgb = DataLoader(valid_dataset_rgb, batch_size=batch_size, shuffle=False)
test_loader_rgb  = DataLoader(test_dataset_rgb,  batch_size=batch_size, shuffle=False)

train_loader_gray = DataLoader(train_dataset_gray, batch_size=batch_size, shuffle=True)
valid_loader_gray = DataLoader(valid_dataset_gray, batch_size=batch_size, shuffle=False)
test_loader_gray  = DataLoader(test_dataset_gray,  batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def get_model(num_classes=2):
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)

model_rgb = get_model()
model_gray = get_model()

criterion = nn.CrossEntropyLoss()
optimizer_rgb = optim.Adam(model_rgb.parameters(), lr=1e-4)
optimizer_gray = optim.Adam(model_gray.parameters(), lr=1e-4)

def train_model(model, optimizer, train_loader, valid_loader, model_name="model", epochs=5):
    best_auc = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (imgs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training")):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i+1) % 10 == 0:
                print(f"Batch {i+1}/{len(train_loader)}, Loss: {running_loss/10:.4f}")
                running_loss = 0.0
        torch.cuda.empty_cache()

        model.eval()
        all_labels, all_probs = [], []
        with torch.no_grad():
            for imgs, labels in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                probs = torch.softmax(outputs, dim=1)[:,1]
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        auc_score = roc_auc_score(all_labels, all_probs)
        print(f"Epoch {epoch+1}/{epochs}, Val AUC: {auc_score:.4f}")
        if auc_score > best_auc:
            best_auc = auc_score
            model_path = os.path.join(output_dir, f"{model_name}_best_model.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Model saved as {model_path}")
    return model

def evaluate_model(model, test_loader, model_name="model"):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    total_time = 0.0
    total_samples = 0

    with torch.no_grad():
        for i, (imgs, labels) in enumerate(tqdm(test_loader, desc=f"Evaluating {model_name}")):
            batch_size = imgs.size(0)
            start_time = time.time()
            
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)[:,1]
            preds = torch.argmax(outputs, dim=1)
            
            end_time = time.time()
            batch_time = end_time - start_time
            total_time += batch_time
            total_samples += batch_size

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            if (i+1) % 5 == 0:
                print(f"Processed {i+1}/{len(test_loader)} batches, batch time: {batch_time:.4f}s")

    avg_time_per_image = total_time / total_samples
    print(f"Average inference time per image for {model_name}: {avg_time_per_image:.4f}s")

    time_file = os.path.join(output_dir, f"{model_name}_avg_inference_time.txt")
    with open(time_file, "w") as f:
        f.write(f"Average inference time per image: {avg_time_per_image:.4f} seconds\n")
    print(f"Saved average inference time to {time_file}")

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    cm_file = os.path.join(output_dir, f"confusion_matrix_{model_name}.png")
    plt.savefig(cm_file)
    plt.close()
    print(f"Confusion matrix saved: {cm_file}")

    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc_score_val = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score_val:.4f}')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right")
    roc_file = os.path.join(output_dir, f"roc_curve_{model_name}.png")
    plt.savefig(roc_file)
    plt.close()
    print(f"ROC curve saved: {roc_file}")

def visualize_gradcam(model, img_tensor, target_class=1, filename="gradcam.png"):
    model.eval()
    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
    targets = [ClassifierOutputTarget(target_class)]
    
    rgb_img = img_tensor.squeeze().permute(1,2,0).cpu().numpy()
    rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
    
    grayscale_cam = cam(input_tensor=img_tensor.unsqueeze(0).to(img_tensor.device), targets=targets)[0]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    save_path = os.path.join(output_dir, filename)
    plt.imsave(save_path, visualization)
    plt.close()
    print(f"Grad-CAM saved: {save_path}")

if __name__ == "__main__":
    print("Starting RGB training...")
    model_rgb = train_model(model_rgb, optimizer_rgb, train_loader_rgb, valid_loader_rgb, model_name="rgb", epochs=5)
    print("Evaluating RGB model...")
    evaluate_model(model_rgb, test_loader_rgb, model_name="rgb")

    print("Starting Grayscale training...")
    model_gray = train_model(model_gray, optimizer_gray, train_loader_gray, valid_loader_gray, model_name="gray", epochs=5)
    print("Evaluating Grayscale model...")
    evaluate_model(model_gray, test_loader_gray, model_name="gray")

    sample_img, sample_label = test_dataset_rgb[0]
    visualize_gradcam(model_rgb, sample_img, target_class=sample_label, filename="gradcam_rgb_sample0.png")

    sample_img_gray, sample_label_gray = test_dataset_gray[0]
    visualize_gradcam(model_gray, sample_img_gray, target_class=sample_label_gray, filename="gradcam_gray_sample0.png")
