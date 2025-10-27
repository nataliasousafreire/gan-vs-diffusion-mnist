import os
from glob import glob
from typing import Tuple

import numpy as np
from PIL import Image
from scipy import linalg

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import inception_v3, Inception_V3_Weights
from torch.utils.data import DataLoader, Dataset


class ImageFolderDataset(Dataset):
    def __init__(self, folder: str, image_size: int = 64):
        self.paths = sorted(
            glob(os.path.join(folder, "*.png"))
            + glob(os.path.join(folder, "*.jpg"))
            + glob(os.path.join(folder, "*.jpeg"))
            + glob(os.path.join(folder, "*.bmp"))
            + glob(os.path.join(folder, "*.webp"))
        )
        self.tfm = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.tfm(img)


class _FeatureHook:
  
    def __init__(self, module):
        self.features = None
        module.register_forward_hook(self.hook)

    def hook(self, module, input, output):
        feat = F.adaptive_avg_pool2d(output, output_size=(1, 1)).squeeze(-1).squeeze(-1)
        self.features = feat


def _get_inception_with_hook(device="cpu"):

    weights = Inception_V3_Weights.IMAGENET1K_V1
    net = inception_v3(weights=weights, transform_input=False, aux_logits=True)
    target = net._modules["Mixed_7c"]
    hook = _FeatureHook(target)
    net.eval().to(device)
    return net, hook


@torch.no_grad()
def inception_feats_and_probs(dataloader: DataLoader, device="cpu", show_progress: bool = False):
    net, hook = _get_inception_with_hook(device)

    weights = Inception_V3_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()

    iterator = dataloader if not show_progress else __import__("tqdm").tqdm(dataloader, desc="Inception")

    feats = []
    probs = []

    for x in iterator:
        x = x.to(device)
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = preprocess(x)  

        logits = net(x)
        feats.append(hook.features.detach().cpu().numpy())
        probs.append(F.softmax(logits, dim=1).detach().cpu().numpy())

    feats = np.concatenate(feats, axis=0)
    probs = np.concatenate(probs, axis=0)

    return feats, probs

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):

    diff = mu1 - mu2
    offset = np.eye(sigma1.shape[0]) * eps #evitar singularidade
    covmean, _ = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset), disp=False)

    if not np.isfinite(covmean).all():
        covmean = linalg.sqrtm((sigma1 + np.eye(sigma1.shape[0]) * eps).dot(sigma2 + np.eye(sigma2.shape[0]) * eps))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def fid_from_folders(folder_real: str, folder_fake: str, batch_size: int = 64,
                     image_size: int = 64, device="cpu", quiet: bool = True) -> Tuple[float, float, float]:
    ds_r = ImageFolderDataset(folder_real, image_size=image_size)
    ds_f = ImageFolderDataset(folder_fake, image_size=image_size)

    if len(ds_r) == 0 or len(ds_f) == 0:
        raise ValueError("Empty folder for FID/IS")

    dl_r = DataLoader(ds_r, batch_size=batch_size, shuffle=False, num_workers=2)
    dl_f = DataLoader(ds_f, batch_size=batch_size, shuffle=False, num_workers=2)

    feats_r, _ = inception_feats_and_probs(dl_r, device=device, show_progress=not quiet)
    feats_f, probs_f = inception_feats_and_probs(dl_f, device=device, show_progress=not quiet)

    mu_r, sigma_r = np.mean(feats_r, axis=0), np.cov(feats_r, rowvar=False)
    mu_f, sigma_f = np.mean(feats_f, axis=0), np.cov(feats_f, rowvar=False)
    fid = calculate_frechet_distance(mu_r, sigma_r, mu_f, sigma_f)

    py = np.mean(probs_f, axis=0)
    probs_f = np.clip(probs_f, 1e-8, 1.0)
    py = np.clip(py, 1e-8, 1.0)
    kl = probs_f * (np.log(probs_f) - np.log(py))
    kl_sum = np.sum(kl, axis=1)
    kl_sum = np.nan_to_num(kl_sum, nan=0.0, posinf=0.0, neginf=0.0)
    is_mean = float(np.exp(np.mean(kl_sum)))
    is_std = float(np.exp(np.std(kl_sum)))

    return fid, is_mean, is_std
