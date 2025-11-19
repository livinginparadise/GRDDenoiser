import os
import time
import random
import argparse
from glob import glob
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# ==========================================
# 1. Configuration & Hyperparameters
# ==========================================
class Config:
    # Data paths
    CLEAN_DIR = "C:/Users/admin/Desktop/denoiser/dat/train/good"
    POISONED_DIR = "C:/Users/admin/Desktop/denoiser/dat/train/bad"
    
    
    # Training settings
    IMG_SIZE = 256
    BATCH_SIZE = 20
    EPOCHS = 50
    LR = 1e-4
    WEIGHT_DECAY = 1e-5
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Loss Weights
    W_NOISE = 10.0    # Pixel-wise noise match
    W_RESTORE = 5.0   # Pixel-wise restoration match
    W_CLASS = 1.0     # Binary classification
    W_SEMANTIC = 0.5  # Semantic Anchor (Perceptual Loss)
    
    # Saving
    SAVE_DIR = "./checkpoints"
    VIS_DIR = "./visualizations"

os.makedirs(Config.SAVE_DIR, exist_ok=True)
os.makedirs(Config.VIS_DIR, exist_ok=True)

# ==========================================
# 2. Advanced Architecture Components
# ==========================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block: Channel Attention"""
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    """ResNet Block with SE Attention"""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        
        self.se = SEBlock(out_c)
        
        # Shortcut connection to match dimensions
        self.shortcut = nn.Sequential()
        if in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out) 
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling: Multi-scale context"""
    def __init__(self, in_dims, out_dims, rate=[1, 6, 12, 18]):
        super(ASPP, self).__init__()
        self.aspp_blocks = nn.ModuleList()
        
        for r in rate:
            self.aspp_blocks.append(nn.Sequential(
                nn.Conv2d(in_dims, out_dims, 3, stride=1, padding=r, dilation=r),
                nn.BatchNorm2d(out_dims),
                nn.ReLU(inplace=True)
            ))
            
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_dims, out_dims, 1, stride=1),
            nn.BatchNorm2d(out_dims),
            nn.ReLU(inplace=True)
        )
        
        self.output_conv = nn.Sequential(
            nn.Conv2d((len(rate) + 1) * out_dims, out_dims, 1, stride=1),
            nn.BatchNorm2d(out_dims),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = []
        for block in self.aspp_blocks:
            out.append(block(x))
        
        global_feat = self.global_avg_pool(x)
        global_feat = F.interpolate(global_feat, size=x.shape[2:], mode='bilinear', align_corners=True)
        out.append(global_feat)
        
        out = torch.cat(out, dim=1)
        return self.output_conv(out)

class AttentionGate(nn.Module):
    """Focuses on relevant features from the encoder during skip connection"""
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class GRDNet(nn.Module):
    """
    GRD-Net V2: Residual Attention U-Net with ASPP
    """
    def __init__(self, n_channels=3):
        super(GRDNet, self).__init__()
        
        filters = [64, 128, 256, 512, 1024]
        self.enc1 = ResidualBlock(n_channels, filters[0])
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ResidualBlock(filters[0], filters[1])
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ResidualBlock(filters[1], filters[2])
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ResidualBlock(filters[2], filters[3])
        self.pool4 = nn.MaxPool2d(2)
        self.aspp = ASPP(filters[3], filters[4])
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att1 = AttentionGate(F_g=filters[4], F_l=filters[3], F_int=filters[3] // 2)
        self.dec1 = ResidualBlock(filters[4] + filters[3], filters[3])
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att2 = AttentionGate(F_g=filters[3], F_l=filters[2], F_int=filters[2] // 2)
        self.dec2 = ResidualBlock(filters[3] + filters[2], filters[2])
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att3 = AttentionGate(F_g=filters[2], F_l=filters[1], F_int=filters[1] // 2)
        self.dec3 = ResidualBlock(filters[2] + filters[1], filters[1])
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att4 = AttentionGate(F_g=filters[1], F_l=filters[0], F_int=filters[0] // 2)
        self.dec4 = ResidualBlock(filters[1] + filters[0], filters[0])
        self.out_conv = nn.Conv2d(filters[0], n_channels, 1)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(filters[4], 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        
        # Bottleneck
        b = self.aspp(self.pool4(e4))
        
        # Decoder
        d1 = self.up1(b)
        e4_att = self.att1(g=d1, x=e4) 
        d1 = torch.cat((e4_att, d1), dim=1)
        d1 = self.dec1(d1)
        
        d2 = self.up2(d1)
        e3_att = self.att2(g=d2, x=e3)
        d2 = torch.cat((e3_att, d2), dim=1)
        d2 = self.dec2(d2)
        
        d3 = self.up3(d2)
        e2_att = self.att3(g=d3, x=e2)
        d3 = torch.cat((e2_att, d3), dim=1)
        d3 = self.dec3(d3)
        
        d4 = self.up4(d3)
        e1_att = self.att4(g=d4, x=e1)
        d4 = torch.cat((e1_att, d4), dim=1)
        d4 = self.dec4(d4)
        
        # Outputs
        mask = self.out_conv(d4)
        logits = self.classifier(b)
        
        return mask, logits

# ==========================================
# 3. Paired Dataset
# ==========================================

class PairedPoisonDataset(Dataset):
    def __init__(self, clean_dir, poisoned_dir, transform=None):
        self.clean_dir = clean_dir
        self.poisoned_dir = poisoned_dir
        self.transform = transform
        self.image_names = sorted(os.listdir(clean_dir))
        self.valid_names = []
        for name in self.image_names:
            if os.path.exists(os.path.join(poisoned_dir, name)):
                self.valid_names.append(name)
        print(f"Found {len(self.valid_names)} valid pairs.")

    def __len__(self):
        return len(self.valid_names)

    def __getitem__(self, idx):
        img_name = self.valid_names[idx]
        clean_path = os.path.join(self.clean_dir, img_name)
        poison_path = os.path.join(self.poisoned_dir, img_name)
        
        clean_img = Image.open(clean_path).convert("RGB")
        poison_img = Image.open(poison_path).convert("RGB")
        
        if self.transform:
            clean_tensor = self.transform(clean_img)
            poison_tensor = self.transform(poison_img)
            
        is_poisoned_sample = random.random() > 0.5
        if is_poisoned_sample:
            input_img = poison_tensor
            target_clean = clean_tensor
            gt_noise = poison_tensor - clean_tensor
            label = torch.tensor([1.0], dtype=torch.float32)
        else:
            input_img = clean_tensor
            target_clean = clean_tensor
            gt_noise = torch.zeros_like(clean_tensor)
            label = torch.tensor([0.0], dtype=torch.float32)

        return input_img, target_clean, gt_noise, label, img_name

# ==========================================
# 4. Semantic Anchor & Loss Function
# ==========================================

class SemanticAnchor(nn.Module):
    """Frozen feature extractor to prevent hallucinations"""
    def __init__(self, device):
        super().__init__()
        # Use VGG16 features (commonly used for perceptual loss)
        print("Loading Semantic Anchor (VGG16)...")
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        
        # Slice VGG to get meaningful features (e.g., relu3_3)
        # Layers: 0..15 is up to ReLU 3_3 roughly
        self.slice = nn.Sequential()
        for x in range(16): 
            self.slice.add_module(str(x), vgg[x])
            
        self.slice.eval()
        for param in self.slice.parameters():
            param.requires_grad = False
        self.slice.to(device)
        
        # Normalization for VGG
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def forward(self, x):
        # Normalize x (assumed 0-1) to VGG stats
        x = (x - self.mean) / self.std
        return self.slice(x)

class GhostLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.anchor = SemanticAnchor(device)

    def forward(self, pred_mask, pred_logits, input_img, gt_noise, gt_clean, label):
        # 1. Ghost Loss (Pixel)
        loss_noise = self.l1_loss(pred_mask, gt_noise)
        
        # 2. Restoration (Pixel)
        restored_img = input_img - pred_mask
        loss_restore = self.mse_loss(restored_img, gt_clean)
        
        # 3. Classification (Binary)
        loss_class = self.bce(pred_logits, label)
        
        # 4. Semantic Anchor Loss (Perceptual)
        # Ensure the restored image matches the semantic content of the CLEAN image
        with torch.no_grad():
            clean_feats = self.anchor(gt_clean)
        restored_feats = self.anchor(restored_img)
        loss_semantic = self.mse_loss(restored_feats, clean_feats)
        
        total_loss = (Config.W_NOISE * loss_noise) + \
                     (Config.W_RESTORE * loss_restore) + \
                     (Config.W_CLASS * loss_class) + \
                     (Config.W_SEMANTIC * loss_semantic)
                     
        return total_loss, {
            "noise": loss_noise.item(),
            "restore": loss_restore.item(),
            "class": loss_class.item(),
            "semantic": loss_semantic.item()
        }

# ==========================================
# 5. Visualization (Updated for logits)
# ==========================================

def visualize_batch(model, loader, epoch, device):
    model.eval()
    try:
        inputs, clean_gt, noise_gt, labels, _ = next(iter(loader))
    except StopIteration:
        return

    inputs = inputs.to(device)
    with torch.no_grad():
        pred_mask, pred_logits = model(inputs)
        # Apply sigmoid manually for visualization probability
        pred_prob = torch.sigmoid(pred_logits)
        restored = inputs - pred_mask

    inputs = inputs.cpu()
    pred_mask = pred_mask.cpu()
    restored = restored.cpu()
    noise_gt = noise_gt.cpu()
    
    viz_mask = pred_mask * 5.0 
    viz_gt_noise = noise_gt * 5.0

    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    plt.suptitle(f"Epoch {epoch} Visualization (GRD-Net V2)", fontsize=16)
    
    cols = ["Input", "GT Noise (x5)", "Pred Noise (x5)", "Restored", "Score"]
    for ax, col in zip(axes[0], cols):
        ax.set_title(col)

    for i in range(min(4, inputs.size(0))): 
        axes[i, 0].imshow(inputs[i].permute(1, 2, 0).clamp(0, 1))
        axes[i, 0].axis('off')
        axes[i, 1].imshow((viz_gt_noise[i].permute(1, 2, 0) + 0.5).clamp(0, 1))
        axes[i, 1].axis('off')
        axes[i, 2].imshow((viz_mask[i].permute(1, 2, 0) + 0.5).clamp(0, 1))
        axes[i, 2].axis('off')
        axes[i, 3].imshow(restored[i].permute(1, 2, 0).clamp(0, 1))
        axes[i, 3].axis('off')
        axes[i, 4].text(0.1, 0.5, 
                        f"GT: {int(labels[i].item())}\nPred: {pred_prob[i].item():.4f}", 
                        fontsize=12)
        axes[i, 4].axis('off')

    plt.tight_layout()
    plt.savefig(f"{Config.VIS_DIR}/epoch_{epoch}.png")
    plt.close()
    model.train()

# ==========================================
# 6. Main Training Loop
# ==========================================
def load_checkpoint(path, model, optimizer):
    """
    Loads model and optimizer state from a checkpoint file.
    """
    if not os.path.isfile(path):
        print(f"[-] Checkpoint not found at: {path}")
        return 1, 0.0  # Return start_epoch=1, best_acc=0.0 if no file found

    print(f"[+] Loading checkpoint: {path}")
    checkpoint = torch.load(path, map_location=Config.DEVICE)
    
    # Load Model Weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load Optimizer State (Essential for resuming training momentum)
    if 'optimizer_state_dict' in checkpoint and optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load Metadata
    start_epoch = checkpoint.get('epoch', 0) + 1
    best_val_acc = checkpoint.get('val_acc', 0.0)
    
    print(f"[+] Resuming from Epoch {start_epoch}. Previous Best Acc: {best_val_acc:.4f}")
    return start_epoch, best_val_acc
def train(resume_path=None):
    print(f"--- Initializing GRD-Net V2 (Res-AttU-Net + ASPP + Semantic Anchor) on {Config.DEVICE} ---")
    
    transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
    ])
    
    if not os.path.exists(Config.CLEAN_DIR):
        print("WARNING: Dataset paths not found. Please set CLEAN_DIR and POISONED_DIR.")
        return

    dataset = PairedPoisonDataset(Config.CLEAN_DIR, Config.POISONED_DIR, transform=transform)
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, 
                              num_workers=Config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, 
                            num_workers=Config.NUM_WORKERS)
    
    model = GRDNet().to(Config.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    print("model have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    
    criterion = GhostLoss(Config.DEVICE).to(Config.DEVICE)
    scaler = GradScaler()
    
    # --- RESUME LOGIC START ---
    start_epoch = 1
    best_val_acc = 0.0
    
    if resume_path:
        start_epoch, best_val_acc = load_checkpoint(resume_path, model, optimizer)
    # --- RESUME LOGIC END ---

    print(f"Starting training from epoch {start_epoch} to {Config.EPOCHS}...")

    # Update loop range to use start_epoch
    for epoch in range(start_epoch, Config.EPOCHS + 1):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{Config.EPOCHS}", leave=True)
        
        for inputs, gt_clean, gt_noise, labels, _ in loop:
            inputs = inputs.to(Config.DEVICE)
            gt_clean = gt_clean.to(Config.DEVICE)
            gt_noise = gt_noise.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)
            
            optimizer.zero_grad()
            
            with autocast():
                pred_mask, pred_logits = model(inputs)
                loss, loss_components = criterion(pred_mask, pred_logits, inputs, gt_noise, gt_clean, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            # IMPORTANT: Apply sigmoid to logits for accuracy calculation
            predicted_probs = torch.sigmoid(pred_logits)
            predicted_labels = (predicted_probs > 0.5).float()
            correct += (predicted_labels == labels).sum().item()
            total += labels.size(0)
            
            loop.set_postfix(
                loss=loss.item(), 
                nse=loss_components['noise'], 
                sem=loss_components['semantic']
            )

        avg_train_loss = epoch_loss / len(train_loader)
        train_acc = correct / total
        
        # --- Validation ---
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, gt_clean, gt_noise, labels, _ in val_loader:
                inputs = inputs.to(Config.DEVICE)
                _, pred_logits = model(inputs)
                predicted = (torch.sigmoid(pred_logits) > 0.5).float()
                val_correct += (predicted.to(Config.DEVICE) == labels.to(Config.DEVICE)).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total if val_total > 0 else 0
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Acc: {val_acc:.4f}")
        
        if epoch % 5 == 0 or epoch == 1:
            visualize_batch(model, val_loader, epoch, Config.DEVICE)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(Config.SAVE_DIR, "best_grdnet_v2.pth"))
            print(f"-> New Best Model Saved! ({val_acc:.4f})")
        
        # Always save the last state to allow resuming exactly where we left off
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
        }, os.path.join(Config.SAVE_DIR, "last.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or Resume GRD-Net")
    parser.add_argument('--resume', type=str, default=None, 
                        help='Path to the checkpoint file (e.g., ./checkpoints/last.pth)')
    
    args = parser.parse_args()
    
    # Example usage in terminal: 
    # python main.py --resume ./checkpoints/last.pth
    train(resume_path=args.resume)