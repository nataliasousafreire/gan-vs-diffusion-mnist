import os, argparse, json, torch
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm import tqdm

from callbacks import EarlyStopper, ReduceLROnFID
from utils import save_grid, make_gif_from_folder, write_csv, get_loader
from metrics import fid_from_folders
from models import UNet, linear_beta_schedule, q_sample, p_sample_loop

class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.decay = decay

    @torch.no_grad()
    def update(self):
        for k, v in self.model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def copy_to(self, dest):
        dest.load_state_dict(self.shadow, strict=True)

def count_imgs(path):
    return len([f for f in os.listdir(path)
                if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.webp'))])

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    torch.backends.cudnn.benchmark = True

    dl = get_loader(args.batch_size, args.image_size, args.workers)

    T = args.timesteps
    betas = linear_beta_schedule(T).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.ones(1, device=device), alphas_cumprod[:-1]], dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    posterior_variance = torch.clamp(betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod), min=1e-20)

    net = UNet(base=args.base, in_ch=1).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    use_ema = args.ema
    if use_ema:
        net_ema = UNet(base=args.base, in_ch=1).to(device)
        ema = EMA(net, decay=args.ema_decay)
    else:
        net_ema = net  

    os.makedirs('samples/diffusion', exist_ok=True)
    os.makedirs('samples/_real_cache', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('reports', exist_ok=True)

    log_rows = [['epoch','mse_loss','FID','IS_mean','IS_std']]
    best_fid = float('inf'); best_epoch = -1
    BEST_PATH = 'checkpoints/best_DIFF.pt'

    # callbacks
    stopper = EarlyStopper(patience=args.patience, min_delta=args.min_delta) if args.early_stop else None
    lr_scheduler = ReduceLROnFID(opt, factor=0.5, patience=args.lr_patience, min_lr=1e-6)

    real_cache = 'samples/_real_cache'
    if count_imgs(real_cache) < args.n_real_fid:
        saved = count_imgs(real_cache)
        for rb, _ in dl:
            imgs = (rb * 0.5 + 0.5).clamp(0,1)    
            if imgs.size(1) == 1:                  
                imgs = imgs.repeat(1,3,1,1)
            for i in range(imgs.size(0)):
                save_image(imgs[i], os.path.join(real_cache, f'real_{saved:06d}.png'))
                saved += 1
                if saved >= args.n_real_fid: break
            if saved >= args.n_real_fid: break

    for epoch in range(1, args.epochs+1):
        net.train()
        mse_acc = 0.0; steps = 0

        for real, _ in tqdm(dl, desc=f'Diffusion Epoch {epoch}/{args.epochs}'):
            real = real.to(device)
            b = real.size(0)
            t = torch.randint(0, T, (b,), device=device).long()
            noise = torch.randn_like(real)
            x_noisy = q_sample(real, t, noise, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
            pred_noise = net(x_noisy, t)
            loss = F.mse_loss(pred_noise, noise)

            opt.zero_grad(); loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            opt.step()

            if use_ema: ema.update()

            mse_acc += float(loss.item()); steps += 1

        net.eval()
        with torch.no_grad():
            if use_ema: ema.copy_to(net_ema)
            samples = p_sample_loop(net_ema, (64,1,args.image_size,args.image_size),
                                    T, betas, sqrt_one_minus_alphas_cumprod,
                                    sqrt_recip_alphas, posterior_variance, device)
        save_grid(samples.cpu(), f'samples/diffusion/epoch_{epoch:03d}.png', nrow=8)
        torch.save(net.state_dict(), f'checkpoints/DIFF_epoch_{epoch}.pt')

        fake_folder = 'samples/diff_fake_tmp'
        os.makedirs(fake_folder, exist_ok=True)
        for fpath in os.listdir(fake_folder):
            if fpath.lower().endswith(('.png','.jpg','.jpeg','.bmp','.webp')):
                try: os.remove(os.path.join(fake_folder, fpath))
                except: pass

        f = (samples * 0.5 + 0.5).clamp(0,1).cpu()
        if f.size(1) == 1: f = f.repeat(1,3,1,1)

        counter = 0
        for i in range(f.size(0)):
            save_image(f[i], os.path.join(fake_folder, f'fake_{counter:05d}.png')); counter += 1

        bsize = 64
        while counter < args.n_fake_fid:
            with torch.no_grad():
                gen = p_sample_loop(net_ema, (bsize,1,args.image_size,args.image_size),
                                    T, betas, sqrt_one_minus_alphas_cumprod,
                                    sqrt_recip_alphas, posterior_variance, device).cpu()
                gen = (gen * 0.5 + 0.5).clamp(0,1)
                if gen.size(1) == 1: gen = gen.repeat(1,3,1,1)
            for j in range(gen.size(0)):
                save_image(gen[j], os.path.join(fake_folder, f'fake_{counter:05d}.png'))
                counter += 1
                if counter >= args.n_fake_fid: break
        try:
            fid, is_mean, is_std = fid_from_folders(
                real_cache, fake_folder,
                batch_size=64, image_size=args.image_size, device=str(device)
            )
        except Exception as e:
            print('[Metrics DIFF] Warning:', repr(e))
            fid, is_mean, is_std = float('nan'), float('nan'), float('nan')

        lr_scheduler.step(fid)
        log_rows.append([epoch, round(mse_acc/steps,4), round(fid,3), round(is_mean,3), round(is_std,3)])

        if not (fid != fid):  # NaN
            if fid < best_fid - args.min_delta:
                best_fid = fid
                best_epoch = epoch
                torch.save(net_ema.state_dict(), BEST_PATH)
                save_grid(samples.cpu(), f'samples/diffusion/best_epoch_{epoch:03d}.png', nrow=8)
                with open('reports/best_diffusion.json', 'w') as f:
                    json.dump({"epoch": best_epoch, "best_fid": float(best_fid)}, f, indent=2)
                print(f'[Checkpoint] New best FID: {best_fid:.3f} @ epoch {best_epoch}')

        if stopper is not None and not (fid != fid):
            if stopper.step(fid, epoch):
                print(f'[EarlyStop] Stopping at epoch {epoch} (best FID={stopper.best:.3f}).')
                break

    write_csv('reports/diffusion_training_log.csv', log_rows[0], log_rows[1:])
    make_gif_from_folder('samples/diffusion', 'reports/diffusion_progress.gif',
                         pattern='epoch_', duration=0.7)

    if best_epoch != -1:
        print(f'Diffusion training done. Best FID={best_fid:.3f} @ epoch {best_epoch}')
        print(f'Best checkpoint: {BEST_PATH}')
    else:
        print('Diffusion training done. (no valid FID for best checkpoint)')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--image-size', type=int, default=64)
    p.add_argument('--timesteps', type=int, default=1000)
    p.add_argument('--base', type=int, default=64)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--ema', action='store_true')
    p.add_argument('--ema-decay', type=float, default=0.999)
    p.add_argument('--early-stop', action='store_true')
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--min-delta', type=float, default=0.0)
    p.add_argument('--lr-patience', type=int, default=5)
    p.add_argument('--n-real-fid', type=int, default=10)
    p.add_argument('--n-fake-fid', type=int, default=10)

    args = p.parse_args()
    main(args)
