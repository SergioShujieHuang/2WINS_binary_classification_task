import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import tempfile
import numpy as np
from collections import Counter

use_class_weight_loss = True   
use_focal_loss = False         
use_weighted_sampler = True
batch_size = 4
decision_threshold = 0.3     
n_splits = 5                   
seed = 42

current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "output")
os.makedirs(output_dir, exist_ok=True)
print(f"All outputs will be saved to: {output_dir}")

def split_dataset(dataset_dir, train_ratio=0.7, valid_ratio=0.15, seed=42):
    random.seed(seed)
    classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    class_to_imgs = {cls: [os.path.join(dataset_dir, cls, f)
                           for f in os.listdir(os.path.join(dataset_dir, cls))
                           if os.path.isfile(os.path.join(dataset_dir, cls, f))]
                     for cls in classes}

    all_files = [(f, cls) for cls, files in class_to_imgs.items() for f in files]
    labels = [cls for _, cls in all_files]

    test_size = 1 - (train_ratio + valid_ratio)
    train_val_files, test_files = train_test_split(
        all_files, test_size=test_size, stratify=labels, random_state=seed
    )

    train_size_adjusted = train_ratio / (train_ratio + valid_ratio)
    train_files, valid_files = train_test_split(
        train_val_files, test_size=1-train_size_adjusted,
        stratify=[cls for _, cls in train_val_files], random_state=seed
    )

    print(f"Train: {len(train_files)}, Valid: {len(valid_files)}, Test: {len(test_files)}")
    def count_classes(file_list):
        labels_ = [cls for _, cls in file_list]
        return Counter(labels_)

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

train_transform_gray = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=3),
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

dataset_dir = "dataset"
train_files, valid_files, test_files = split_dataset(dataset_dir)

trainval_files = train_files + valid_files

test_dataset_gray  = CustomImageDataset(test_files,  transform=eval_transform_gray, to_gray=True)
test_loader_gray  = DataLoader(test_dataset_gray, batch_size=batch_size, shuffle=False, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model_flags = {
    "resnext101_64x4d": True,
}

def get_model(name, num_classes=2):
    if name == "resnext101_64x4d":
        model = models.resnext101_64x4d(weights="IMAGENET1K_V1")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unknown model name {name}")
    return model.to(device)

def make_criterion(train_files_fold):
    # train_files_fold: list of (path, cls)
    label_map = {'good':0, 'bad':1}
    labels = [label_map[cls] for _, cls in train_files_fold]
    good_count = sum(1 for l in labels if l==0)
    bad_count = sum(1 for l in labels if l==1)
    if use_class_weight_loss:
        w_good = 1.0
        w_bad = float(good_count / (bad_count + 1e-8)) * 2.0
        class_weights = torch.tensor([w_good, w_bad], dtype=torch.float).to(device)
        return nn.CrossEntropyLoss(weight=class_weights)
    else:
        return nn.CrossEntropyLoss()

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

def combined_score(fnr, fpr, alpha=0.6):
    return alpha * fnr + (1 - alpha) * fpr

def train_model_early_stopping(model, optimizer, train_loader, valid_loader, criterion,
                               max_epochs=20, patience=3, alpha=0.6):
    best_score = float('inf')  
    trigger_times = 0
    best_model_state = None

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs} Training", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)

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

def evaluate_model(model, test_loader):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    total_infer_time = 0.0
    total_samples = 0

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc=f"Evaluating", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            start = time.time()
            outputs = model(imgs)
            end = time.time()
            infer_time = end - start
            total_infer_time += infer_time
            total_samples += imgs.size(0)

            probs = torch.softmax(outputs, dim=1)[:,1]
            preds = (probs > decision_threshold).long()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    fnr = fn / (fn + tp + 1e-8)
    fpr = fp / (fp + tn + 1e-8)

    param_size = sum(p.numel()*p.element_size() for p in model.parameters()) / (1024**2)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmpfile:
        torch.save(model.state_dict(), tmpfile.name)
        state_dict_size = os.path.getsize(tmpfile.name)/(1024**2)
    os.remove(tmpfile.name)
    total_model_size = param_size + state_dict_size

    avg_infer_time = total_infer_time / (total_samples + 1e-8)

    return all_labels, all_preds, all_probs, total_model_size, avg_infer_time, fnr, fpr

def create_folds_from_files(file_list, n_splits=5, seed=42):
    label_map = {'good':0, 'bad':1}
    labels = [label_map[cls] for _, cls in file_list]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    file_array = np.array(file_list)
    for train_idx, valid_idx in skf.split(file_array, labels):
        train_files = [tuple(x) for x in file_array[train_idx].tolist()]
        valid_files = [tuple(x) for x in file_array[valid_idx].tolist()]
        folds.append((train_files, valid_files))
    return folds


def ensemble_evaluate(models, test_loader):
    all_labels, all_preds = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Ensemble Evaluating", leave=False):
            imgs = imgs.to(device)
            avg_probs = torch.zeros((imgs.size(0), 2), device=device)
            for model in models:
                outputs = model(imgs)
                probs = torch.softmax(outputs, dim=1)
                avg_probs += probs
            avg_probs /= len(models)
            preds = (avg_probs[:,1] > decision_threshold).long()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    cm = confusion_matrix(all_labels, all_preds, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    fnr = fn / (fn + tp + 1e-8)
    fpr = fp / (fp + tn + 1e-8)
    return fnr, fpr

if __name__ == '__main__':
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    folds = create_folds_from_files(trainval_files, n_splits=n_splits, seed=seed)
    print(f"Created {len(folds)} folds. Each fold sizes:")
    for i,(tr,va) in enumerate(folds):
        cnt_tr = Counter([cls for _,cls in tr])
        cnt_va = Counter([cls for _,cls in va])
        print(f" Fold {i+1}: train {len(tr)} {dict(cnt_tr)}, valid {len(va)} {dict(cnt_va)}")

    trained_models = []
    fold_summaries = []  # store (name, fnr, fpr, size, avg_time)

    for fold_id, (train_files_fold, valid_files_fold) in enumerate(folds):
        print(f"\n===== TRAINING fold {fold_id+1}/{len(folds)} =====")
        # dataset & loaders for this fold
        train_dataset = CustomImageDataset(train_files_fold, transform=train_transform_gray, to_gray=True)
        valid_dataset = CustomImageDataset(valid_files_fold, transform=eval_transform_gray, to_gray=True)

        # recompute class counts for sampler & loss
        label_map = {'good':0, 'bad':1}
        labels_fold = [label_map[cls] for _, cls in train_files_fold]
        good_count = sum(1 for l in labels_fold if l==0)
        bad_count = sum(1 for l in labels_fold if l==1)

        if use_weighted_sampler:
            weight_per_class = {0: 1.0, 1: (good_count / (bad_count + 1e-8))}
            sample_weights = [weight_per_class[label_map[cls]] for _, cls in train_files_fold]
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=2)
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        # model / criterion / optimizer
        model = get_model("resnext101_64x4d")
        criterion = make_criterion(train_files_fold)  # per-fold class weight
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        # train with early stopping (on combined score)
        model = train_model_early_stopping(model, optimizer, train_loader, valid_loader, criterion,
                                           max_epochs=20, patience=3, alpha=0.6)

        # save model for this fold
        fold_model_path = os.path.join(output_dir, f"resnext101_64x4d_fold{fold_id+1}.pt")
        torch.save(model.state_dict(), fold_model_path)
        print(f"Saved fold model: {fold_model_path}")

        # evaluate this fold model on holdout test set
        labels, preds, probs, size, avg_time, fnr, fpr = evaluate_model(model, test_loader_gray)
        fold_name = f"resnext101_64x4d_fold{fold_id+1}"
        fold_summaries.append((fold_name, fnr, fpr, size, avg_time))
        trained_models.append(model)

        print(f"Fold {fold_id+1} -> Test FNR: {fnr:.4f}, FPR: {fpr:.4f}, Size: {size:.2f}MB, AvgTime: {avg_time*1000:.2f} ms")

    print("\n===== ENSEMBLE evaluation on holdout test =====")
    ensemble_fnr, ensemble_fpr = ensemble_evaluate(trained_models, test_loader_gray)
    # compute ensemble "size" and "time" summary: size = sum? or mean? We'll use mean param size and mean time for table
    sizes = [s for _,_,_,s,_ in fold_summaries]
    times = [t for _,_,_,_,t in fold_summaries]
    ensemble_size = float(np.mean(sizes))
    ensemble_avg_time = float(np.mean(times))
    fold_summaries.append(("Ensemble", ensemble_fnr, ensemble_fpr, ensemble_size, ensemble_avg_time))
    print(f"Ensemble -> FNR: {ensemble_fnr:.4f}, FPR: {ensemble_fpr:.4f}")

    summary = sorted(fold_summaries, key=lambda x: (x[1], x[2], x[3], x[4]))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axis('off')
    columns = ["Model/Fold", "FNR (↓)", "FPR (↓)", "Size (MB, ↓)", "Avg Time (s/img, ↓)"]
    table_data = [[name, f"{fnr:.4f}", f"{fpr:.4f}", f"{size:.2f}", f"{avg_time*1000:.2f} ms"]
                  for name, fnr, fpr, size, avg_time in summary]
    table = ax.table(cellText=table_data, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    plt.title("5-Fold Benchmark + Ensemble (resnext101_64x4d)", fontsize=14, pad=20)
    plt.tight_layout()
    out_png = os.path.join(output_dir, "benchmark_5fold_ensemble_resnext101_64x4d.png")
    plt.savefig(out_png)
    plt.close()
    print(f"✅ Benchmark comparison saved to {out_png}")

    print("\nFinal benchmark (sorted):")
    for row in summary:
        print(f"{row[0]:25s} | FNR: {row[1]:.4f} | FPR: {row[2]:.4f} | Size(MB): {row[3]:.2f} | AvgTime(ms): {row[4]*1000:.2f}")
