import torch, glob, numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn, torch.optim as optim
from model import UNet
from config import *

class PatchDataset(Dataset):
    def __init__(self, files): self.files = files
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        arr = np.load(self.files[idx])
        x, y = arr["x"], arr["y"]
        return torch.from_numpy(x.transpose(2,0,1)).float(), torch.from_numpy(y).float()

def train():
    files = glob.glob(os.path.join(PATCH_NPY_DIR, "*.npz"))
    split = int(len(files)*SPLIT_BY_REGION)
    train_ds = PatchDataset(files[:split])
    val_ds   = PatchDataset(files[split:])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = UNet(in_ch=train_ds[0][0].shape[0], out_ch=1)
    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    crit = nn.BCELoss()

    best_loss = float("inf")
    for epoch in range(MAX_EPOCHS):
        model.train()
        tot, tot_bce = 0, 0
        for xb, yb in train_loader:
            opt.zero_grad()
            pred = model(xb)
            loss = crit(pred.squeeze(1), yb)
            loss.backward()
            opt.step()
            tot += 1; tot_bce += loss.item()
        val_loss = np.mean([
            crit(model(x).squeeze(1), y).item()
            for x,y in val_loader
        ])
        print(f"Epoch {epoch} train_loss={tot_bce/tot:.4f} val_loss={val_loss:.4f}")
        if val_loss < best_loss:
            torch.save(model.state_dict(), MODEL_WEIGHTS)
            best_loss = val_loss
    print("Training complete. Best val_loss = %.4f" % best_loss)

if __name__ == "__main__":
    train()
