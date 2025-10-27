import argparse, os, math, torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from models import Generator, UNet,linear_beta_schedule,extract,q_sample,p_sample,p_sample_loop


def _load_ckpt_or_best(path, default_path, name):
    ckpt = path if path else default_path
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"{name}: checkpoint não encontrado: {ckpt}")
    return ckpt

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gan-ckpt", type=str, default=None)
    p.add_argument("--diff-ckpt", type=str, default=None)
    p.add_argument("--n", type=int, default=64,)
    p.add_argument("--nz", type=int, default=100)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--timesteps", type=int, default=200)
    p.add_argument("--out", type=str, default="samples/infer_side_by_side.png")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs("samples", exist_ok=True)

    if int(args.n ** 0.5) ** 2 != args.n:
        raise SystemExit("--n deve ser quadrado (ex.: 16, 25, 36, 49, 64).")

    # usa o melhor modelo 
    gan_ckpt  = _load_ckpt_or_best(args.gan_ckpt,  "checkpoints/best_GAN_G.pt", "GAN")
    diff_ckpt = _load_ckpt_or_best(args.diff_ckpt, "checkpoints/best_DIFF.pt",  "Diffusion")

    # gan
    G = Generator(nz=args.nz, ngf=64, nc=1).to(device)
    G.load_state_dict(torch.load(gan_ckpt, map_location=device))
    G.eval()
    with torch.no_grad():
        z = torch.randn(args.n, args.nz, 1, 1, device=device)
        gan_samples = G(z).cpu()                       # [-1,1]
        gan_vis = (gan_samples * 0.5 + 0.5).clamp(0,1)
    grid_gan = make_grid(gan_vis, nrow=int(args.n**0.5))
    save_image(grid_gan, "samples/infer_gan.png")

    #difusao
    net = UNet(base=64, in_ch=1).to(device)
    net.load_state_dict(torch.load(diff_ckpt, map_location=device))
    net.eval()

    betas = linear_beta_schedule(args.timesteps).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.ones(1, device=device), alphas_cumprod[:-1]], dim=0)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    posterior_variance = torch.clamp(betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod), min=1e-20)

    with torch.no_grad():
        diffs = p_sample_loop(net, (args.n,1,args.image_size,args.image_size),
                              args.timesteps, betas, sqrt_one_minus_alphas_cumprod,
                              sqrt_recip_alphas, posterior_variance, device).cpu()
        diff_vis = (diffs * 0.5 + 0.5).clamp(0,1)
    grid_diff = make_grid(diff_vis, nrow=int(args.n**0.5))
    save_image(grid_diff, "samples/infer_diffusion.png")


    side_by_side = torch.cat([grid_gan, grid_diff], dim=2)
    save_image(side_by_side, args.out)

    print(f"[OK] GAN grid: samples/infer_gan.png")
    print(f"[OK] Diffusion grid: samples/infer_diffusion.png")
    print(f"[OK] Side-by-side: {args.out}")

if __name__ == "__main__":
    main()
