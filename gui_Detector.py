import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import glob

try:
    from safetensors.torch import load_file as load_safetensors
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

# --- NEURAL NETWORK BLOCKS ---

class SEBlock(nn.Module):
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
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.se = SEBlock(out_c)
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
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        
        b = self.aspp(self.pool4(e4))
        
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
        
        mask = self.out_conv(d4)
        logits = self.classifier(b)
        
        return mask, logits

# --- HELPER UTILS ---

class PatchManager:
    def __init__(self, img_pil, patch_size=256):
        self.img = img_pil
        self.patch_size = patch_size
        self.w, self.h = img_pil.size
        self.pad_w = (patch_size - (self.w % patch_size)) % patch_size
        self.pad_h = (patch_size - (self.h % patch_size)) % patch_size
        
        self.padded_img = ImageOps.expand(img_pil, (0, 0, self.pad_w, self.pad_h))
        self.p_w, self.p_h = self.padded_img.size
        
        self.cols = self.p_w // patch_size
        self.rows = self.p_h // patch_size
        
    def get_patches(self):
        transform = transforms.Compose([transforms.ToTensor()])
        patches_list = []
        coords = [] # (row, col, x, y)
        
        for r in range(self.rows):
            for c in range(self.cols):
                x = c * self.patch_size
                y = r * self.patch_size
                
                crop = self.padded_img.crop((x, y, x+self.patch_size, y+self.patch_size))
                patches_list.append(transform(crop))
                coords.append((r, c, x, y))
                
        return torch.stack(patches_list), coords

    def stitch_heatmap(self, mask_tensors):
        full_map = torch.zeros((self.p_h, self.p_w))
        idx = 0
        for r in range(self.rows):
            for c in range(self.cols):
                x = c * self.patch_size
                y = r * self.patch_size
                
                mask_patch = mask_tensors[idx]
                heatmap_patch = torch.mean(torch.abs(mask_patch), dim=0)
                
                full_map[y:y+self.patch_size, x:x+self.patch_size] = heatmap_patch
                idx += 1
        return full_map[:self.h, :self.w].numpy()

# --- GUI APP ---

class PoisonDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Patch Poison Detector (Batch Support)")
        self.root.geometry("1300x850")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.patch_size = 256
        
        self.setup_layout()
        
    def setup_layout(self):
        # Sidebar
        sidebar = tk.Frame(self.root, width=300, bg="#e0e0e0", padx=10, pady=10)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)
        
        # Header
        tk.Label(sidebar, text="Detection System", font=("Helvetica", 16, "bold"), bg="#e0e0e0").pack(pady=(0, 20))
        
        # Model Loading
        tk.Button(sidebar, text="Load Model Checkpoint", command=self.load_model_file, 
                  bg="white", relief="groove").pack(fill=tk.X, pady=5)
        self.lbl_model = tk.Label(sidebar, text="Model: None", bg="#e0e0e0", fg="red", font=("Arial", 9))
        self.lbl_model.pack()
        
        ttk.Separator(sidebar, orient='horizontal').pack(fill='x', pady=15)
        
        # Single Image Section
        tk.Label(sidebar, text="Single Image", font=("Helvetica", 11, "bold"), bg="#e0e0e0", anchor="w").pack(fill=tk.X)
        tk.Button(sidebar, text="Analyze One Image", command=self.process_single_ui, 
                  bg="#4CAF50", fg="white", font=("Arial", 12, "bold"), height=2).pack(fill=tk.X, pady=5)
        
        ttk.Separator(sidebar, orient='horizontal').pack(fill='x', pady=15)
        
        # Batch Section
        tk.Label(sidebar, text="Batch Processing", font=("Helvetica", 11, "bold"), bg="#e0e0e0", anchor="w").pack(fill=tk.X)
        
        btn_frame = tk.Frame(sidebar, bg="#e0e0e0")
        btn_frame.pack(fill=tk.X)
        
        tk.Button(btn_frame, text="Batch Files (List)", command=self.handle_batch_files, 
                  bg="#2196F3", fg="white").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        tk.Button(btn_frame, text="Batch Folder", command=self.handle_batch_folder, 
                  bg="#2196F3", fg="white").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        # Progress Bar for Batch
        self.progress = ttk.Progressbar(sidebar, orient="horizontal", length=100, mode="determinate")
        self.progress.pack(fill=tk.X, pady=(10, 0))
        
        self.lbl_batch_status = tk.Label(sidebar, text="Idle", bg="#e0e0e0", font=("Arial", 9))
        self.lbl_batch_status.pack()

        ttk.Separator(sidebar, orient='horizontal').pack(fill='x', pady=15)

        # Results Section
        tk.Label(sidebar, text="Single Result", font=("Helvetica", 12, "bold"), bg="#e0e0e0").pack(pady=(5, 5))
        
        self.lbl_verdict = tk.Label(sidebar, text="Verdict: WAITING", font=("Helvetica", 14), bg="#e0e0e0", fg="grey")
        self.lbl_verdict.pack(pady=5)
        
        self.lbl_stats = tk.Label(sidebar, text="", bg="#e0e0e0", justify=tk.LEFT)
        self.lbl_stats.pack(pady=5, anchor="w")

        # Main Content (Canvas)
        content = tk.Frame(self.root, bg="white")
        content.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.fig = plt.figure(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=content)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def load_model_file(self):
        ftypes = [("PyTorch Model", "*.pth"), ("Safetensors", "*.safetensors"), ("All Files", "*.*")]
        path = filedialog.askopenfilename(filetypes=ftypes)
        if not path: return
        
        try:
            self.model = GRDNet().to(self.device)
            
            if path.lower().endswith(".safetensors"):
                if not SAFETENSORS_AVAILABLE:
                    raise ImportError("The 'safetensors' library is not installed.\nPlease run: pip install safetensors")
                
                # Load using safetensors
                state_dict = load_safetensors(path, device=self.device)
                self.model.load_state_dict(state_dict)
            else:
                # Load using standard PyTorch
                checkpoint = torch.load(path, map_location=self.device)
                
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                
            self.model.eval()
            self.lbl_model.config(text=f"Loaded: {os.path.basename(path)}", fg="green")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{e}")

    # --- CORE LOGIC ---
    
    def predict_single_image(self, file_path):
        """
        Core logic: Reads image -> Patches -> Model -> Verdict.
        Returns: (verdict_string, is_poisoned, stats_dict, visualization_data)
        """
        try:
            pil_img = Image.open(file_path).convert("RGB")
            manager = PatchManager(pil_img, self.patch_size)
            patches_tensor, coords = manager.get_patches()
            
            batch_size = 4
            all_probs = []
            all_masks = []
            
            num_patches = patches_tensor.size(0)
            
            with torch.no_grad():
                for i in range(0, num_patches, batch_size):
                    batch = patches_tensor[i : i + batch_size].to(self.device)
                    masks, logits = self.model(batch)
                    
                    probs = torch.sigmoid(logits).cpu()
                    all_probs.append(probs)
                    all_masks.append(masks.cpu())
            
            all_probs = torch.cat(all_probs).view(-1)
            all_masks = torch.cat(all_masks)
            
            poison_indices = (all_probs > 0.5).nonzero(as_tuple=True)[0]
            detected_count = len(poison_indices)
            
            is_poisoned = detected_count >= 2
            verdict = "POISONED" if is_poisoned else "CLEAN"
            
            stats = {
                "num_patches": num_patches,
                "detected_count": detected_count,
                "max_conf": all_probs.max().item() if len(all_probs) > 0 else 0
            }
            
            vis_data = {
                "pil_img": pil_img,
                "manager": manager,
                "coords": coords,
                "poison_indices": poison_indices,
                "all_masks": all_masks
            }
            
            return verdict, is_poisoned, stats, vis_data
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return "ERROR", False, None, None

    # --- SINGLE IMAGE UI ---

    def process_single_ui(self):
        if self.model is None:
            messagebox.showwarning("Warning", "Please load a model first.")
            return
            
        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg *.bmp")])
        if not file_path: return
        
        verdict, is_poisoned, stats, vis_data = self.predict_single_image(file_path)
        
        if verdict == "ERROR":
            messagebox.showerror("Error", "Could not process image.")
            return

        # Update UI Labels
        if is_poisoned:
            self.lbl_verdict.config(text="POISONED", fg="red")
        else:
            self.lbl_verdict.config(text="CLEAN", fg="green")
            
        stats_text = (f"Total Patches: {stats['num_patches']}\n"
                      f"Noisy Patches Found: {stats['detected_count']}\n"
                      f"Rule: >= 2 Patches needed\n"
                      f"Max Conf: {stats['max_conf']:.2%}")
        self.lbl_stats.config(text=stats_text)
        
        # Draw
        self.visualize(**vis_data)

    def visualize(self, pil_img, manager, coords, poison_indices, all_masks):
        self.fig.clear()
        ax1 = self.fig.add_subplot(1, 2, 1)
        ax2 = self.fig.add_subplot(1, 2, 2)
        
        ax1.imshow(pil_img)
        ax1.set_title("Detection Results (Red = Noise Detected)")
        ax1.axis('off')
        
        for idx in poison_indices:
            idx = idx.item()
            r, c, x, y = coords[idx]
            rect = matplotlib.patches.Rectangle((x, y), self.patch_size, self.patch_size, 
                                            linewidth=2, edgecolor='red', facecolor='none')
            ax1.add_patch(rect)
            
            ax1.text(x, y, "NOISE", color='red', fontsize=8, fontweight='bold', 
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        heatmap = manager.stitch_heatmap(all_masks)
        im = ax2.imshow(heatmap, cmap='inferno')
        ax2.set_title("Reconstructed Noise Heatmap")
        ax2.axis('off')
        self.fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        
        self.canvas.draw()

    # --- BATCH PROCESSING LOGIC ---

    def handle_batch_files(self):
        if self.model is None:
            messagebox.showwarning("Warning", "Please load a model first.")
            return
            
        files = filedialog.askopenfilenames(title="Select Images", 
                                            filetypes=[("Images", "*.jpg *.png *.jpeg *.bmp")])
        if files:
            self.run_batch_processing(list(files))

    def handle_batch_folder(self):
        if self.model is None:
            messagebox.showwarning("Warning", "Please load a model first.")
            return
            
        folder = filedialog.askdirectory(title="Select Folder")
        if folder:
            # Find images in folder
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.PNG']
            files = []
            for ext in extensions:
                files.extend(glob.glob(os.path.join(folder, ext)))
            
            if not files:
                messagebox.showinfo("Info", "No images found in that folder.")
                return
                
            self.run_batch_processing(files)

    def run_batch_processing(self, file_list):
        save_path = filedialog.asksaveasfilename(defaultextension=".txt", 
                                                 filetypes=[("Text File", "*.txt")],
                                                 title="Save Report As")
        if not save_path:
            return

        total = len(file_list)
        self.progress["maximum"] = total
        self.progress["value"] = 0
        
        processed_count = 0
        poison_count = 0
        
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(f"Batch Report - Total Images: {total}\n")
                f.write("-" * 60 + "\n")
                
                for i, img_path in enumerate(file_list):
                    # UI Update
                    self.lbl_batch_status.config(text=f"Processing {i+1}/{total}...")
                    self.root.update() # Keep GUI responsive
                    
                    # Logic
                    full_path = os.path.abspath(img_path)
                    verdict, is_poisoned, _, _ = self.predict_single_image(full_path)
                    
                    # Write to file
                    line = f"{full_path} | {verdict}\n"
                    f.write(line)
                    print(line.strip())
                    
                    if is_poisoned:
                        poison_count += 1
                    processed_count += 1
                    
                    # Update bar
                    self.progress["value"] = i + 1
            
            self.lbl_batch_status.config(text="Complete.")
            messagebox.showinfo("Batch Complete", 
                                f"Processed: {processed_count}\n"
                                f"Poisoned: {poison_count}\n"
                                f"Clean: {processed_count - poison_count}\n\n"
                                f"Report saved to:\n{save_path}")
                                
        except Exception as e:
            messagebox.showerror("Batch Error", f"An error occurred during batch processing:\n{e}")
        finally:
            self.lbl_batch_status.config(text="Idle")
            self.progress["value"] = 0

if __name__ == "__main__":
    root = tk.Tk()
    app = PoisonDetectorApp(root)
    root.mainloop()