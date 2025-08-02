import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.fc(x)

class UNet(nn.Module):
    def __init__(self, in_ch=8, out_ch=1, features=[64,128,256,512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        for f in features:
            self.downs.append(DoubleConv(in_ch, f))
            in_ch = f
        self.pool = nn.MaxPool2d(2)
        for f in reversed(features[:-1]):
            self.ups.append(nn.ConvTranspose2d(in_ch, f, 2, stride=2))
            self.ups.append(DoubleConv(in_ch, f))
            in_ch = f
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final = nn.Conv2d(features[0], out_ch, 1)

    def forward(self, x):
        skip_connections = []
        for d in self.downs:
            x = d(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            up = self.ups[idx]
            conv = self.ups[idx+1]
            x = up(x)
            skip = skip_connections[idx//2]
            if x.shape != skip.shape:
                x = torch.nn.functional.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = conv(x)
        return torch.sigmoid(self.final(x))
