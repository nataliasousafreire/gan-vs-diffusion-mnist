import os, imageio
from typing import List
import torchvision
import torch
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torch.nn as nn

def save_grid(tensor, path, nrow=8):
    grid = make_grid(tensor, nrow=nrow, normalize=True, value_range=(-1, 1))
    save_image(grid, path)

def make_gif_from_folder(folder: str, out_path: str, pattern: str = "epoch_", duration: float = 0.8):
    frames = []
    for name in sorted(os.listdir(folder)):
        if pattern in name and (name.endswith(".png") or name.endswith(".jpg")):
            frames.append(imageio.v2.imread(os.path.join(folder, name)))
    if not frames:
        raise RuntimeError(f"No frames found in folder")
    imageio.mimsave(out_path, frames, duration=duration)

def write_csv(path: str, header: List[str], rows: List[List]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        f.write(",".join(map(str, header)) + "\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")

def get_loader(bs=128, image_size=64, workers=2):
    tfm = T.Compose([T.Resize(image_size), T.ToTensor(), T.Normalize([0.5],[0.5])])
    ds = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=tfm)
    return DataLoader(ds, batch_size=bs, shuffle=True, num_workers=workers, pin_memory=True)

class EMA:
    def __init__(self, model, decay=0.999):
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if torch.is_floating_point(v):
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()

    @torch.no_grad()
    def copy_to(self, dest_model):
        dest_model.load_state_dict(self.shadow, strict=True)

def gan_losses(logits_real=None, logits_fake=None, kind="bce"):
    lossD, lossG = None, None
    if kind == "hinge":
        if (logits_real is not None) and (logits_fake is not None):
            lossD = (torch.relu(1 - logits_real).mean() +
                     torch.relu(1 + logits_fake).mean())
        if logits_fake is not None:
            lossG = (-logits_fake).mean()
    else: 
        bce = nn.BCEWithLogitsLoss()
        if (logits_real is not None) and (logits_fake is not None):
            lossD = 0.5 * (
                bce(logits_real, torch.ones_like(logits_real)) +
                bce(logits_fake, torch.zeros_like(logits_fake))
            )
        if logits_fake is not None:
            lossG = bce(logits_fake, torch.ones_like(logits_fake))

    return lossD, lossG