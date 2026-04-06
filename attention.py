import torch
import cv2
import numpy as np

# 1. Define Preprocessing Constants (matching Cell 6 of the notebook)
NUM_FRAMES = 16
CROP_SIZE  = 224
FRAME_SIZE = 256
MEAN = [0.45, 0.45, 0.45]
STD  = [0.225, 0.225, 0.225]
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    def __init__(self, embed_dim=192, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_out, weights = self.attn(x, x, x)
        x = self.norm(attn_out + x)
        return x.mean(dim=1), weights

class AbandonedObjectClassifier(nn.Module):
    def __init__(self, freeze_backbone=True):
        super().__init__()
        # Loads X3D-M from PyTorchVideo hub
        backbone = torch.hub.load(
            'facebookresearch/pytorchvideo', 'x3d_m',
            pretrained=True, verbose=False
        )
        self.feature_blocks = nn.Sequential(*list(backbone.blocks[:-1]))

        if freeze_backbone:
            for p in self.feature_blocks.parameters():
                p.requires_grad = False
            for p in self.feature_blocks[-1].parameters():
                p.requires_grad = True

        self.attention  = TemporalAttention(embed_dim=192, num_heads=4)
        self.classifier = nn.Sequential(
            nn.Linear(192, 512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1),   nn.Sigmoid()
        )

    def forward(self, x, return_attention=False):
        feats = self.feature_blocks(x)          # (B,192,T,H,W)
        feats = feats.mean(dim=[-2,-1])          # (B,192,T)
        feats = feats.permute(0,2,1)             # (B,T,192)
        ctx, attn = self.attention(feats)        # (B,192)
        out = self.classifier(ctx)               # (B,1)
        return (out, attn) if return_attention else out

def get_prediction(video_path, model_weights_path="attemton_model.pt"):
    # 2. Load the Video and Sample 16 frames
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, max(total_frames - 1, 0), NUM_FRAMES, dtype=int)
    
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize and Center Crop
            h, w = frame.shape[:2]
            scale = FRAME_SIZE / min(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            y0, x0 = (frame.shape[0] - CROP_SIZE) // 2, (frame.shape[1] - CROP_SIZE) // 2
            frames.append(frame[y0:y0+CROP_SIZE, x0:x0+CROP_SIZE])
    cap.release()

    # 3. Normalize and convert to Tensor (Shape: 1, C, T, H, W)
    input_tensor = np.stack(frames).astype(np.float32) / 255.0
    input_tensor = torch.from_numpy(input_tensor).permute(3, 0, 1, 2) # (C, T, H, W)
    for c, (m, s) in enumerate(zip(MEAN, STD)):
        input_tensor[c] = (input_tensor[c] - m) / s
    input_tensor = input_tensor.unsqueeze(0) # Add batch dimension

    # 4. Load Model and Run Inference
    # Note: AbandonedObjectClassifier class must be defined in your script (from Cell 5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AbandonedObjectClassifier().to(device)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.eval()

    with torch.no_grad():
        prediction = model(input_tensor.to(device))
        # Return 1 if score >= 0.5, else 0
        return 1 if prediction.item() >= 0.5 else 0

# Usage:

if __name__ == "__main__":
  result = get_prediction("test3.mp4", "attemton_model.pt")
  print(result)