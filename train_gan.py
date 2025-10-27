# train_gan.py
import os, argparse, json, torch, torchvision
import torch.optim as optim
from torchvision.utils import save_image
from tqdm import tqdm
from callbacks import EarlyStopper, ReduceLROnFID
from metrics import fid_from_folders
from utils import save_grid, make_gif_from_folder, write_csv, get_loader,EMA,gan_losses
from models import Generator, Discriminator

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    torch.backends.cudnn.benchmark = True

    dl = get_loader(args.batch_size, args.image_size, args.workers)

    G = Generator(args.nz, args.ngf, 1).to(device)
    D = Discriminator(args.ndf, 1).to(device)

    lrG = args.lr
    lrD = args.lr * (2.0 if args.ttur else 1.0)
    optG = optim.Adam(G.parameters(), lr=lrG, betas=(0.5, 0.999))
    optD = optim.Adam(D.parameters(), lr=lrD, betas=(0.5, 0.999))

    use_ema = args.ema
    if use_ema:
        G_ema = Generator(args.nz, args.ngf, 1).to(device)
        ema = EMA(G, decay=args.ema_decay)
        ema.copy_to(G_ema)  
    else:
        G_ema = G

    fixed = torch.randn(64, args.nz, 1, 1, device=device)

    os.makedirs('samples/gan', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    os.makedirs('samples/_real_cache', exist_ok=True)

    log_rows = [['epoch','lossG','lossD','FID','IS_mean','IS_std']]

    best_fid = float('inf')
    best_epoch = -1
    BEST_G_PATH = 'checkpoints/best_GAN_G.pt'
    BEST_D_PATH = 'checkpoints/best_GAN_D.pt'

    stopper = EarlyStopper(patience=args.patience, min_delta=args.min_delta) if args.early_stop else None
    lrG_sched = ReduceLROnFID(optG, factor=0.5, patience=args.lr_patience, min_lr=1e-6)
    lrD_sched = ReduceLROnFID(optD, factor=0.5, patience=args.lr_patience, min_lr=1e-6) if args.lr_both else None

    real_cache = 'samples/_real_cache'
    def _count_imgs(path):
        return len([f for f in os.listdir(path) if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.webp'))])

    if _count_imgs(real_cache) < args.n_real_fid:
        saved = _count_imgs(real_cache)
        for rb, _ in dl:
            imgs = (rb * 0.5 + 0.5).clamp(0,1)  # [0,1]
            if imgs.size(1) == 1: imgs = imgs.repeat(1,3,1,1)  # RGB p/ Inception
            for i in range(imgs.size(0)):
                save_image(imgs[i], os.path.join(real_cache, f'real_{saved:06d}.png'))
                saved += 1
                if saved >= args.n_real_fid: break
            if saved >= args.n_real_fid: break

    for epoch in range(1, args.epochs+1):
        G.train(); D.train()
        lossG_acc = 0.0; lossD_acc = 0.0; steps = 0

        for real, _ in tqdm(dl, desc=f'GAN Epoch {epoch}/{args.epochs}'):
            real = real.to(device)

            z = torch.randn(real.size(0), args.nz, 1, 1, device=device)
            fake = G(z).detach()
            logits_real = D(real); logits_fake = D(fake)
            lossD, _ = gan_losses(logits_real, logits_fake, kind=('hinge' if args.hinge else 'bce'))
            optD.zero_grad(); lossD.backward(); optD.step()

            z = torch.randn(real.size(0), args.nz, 1, 1, device=device)
            fake = G(z)
            logits_fake = D(fake)
            _, lossG = gan_losses(None, logits_fake, kind=('hinge' if args.hinge else 'bce'))
            optG.zero_grad(); lossG.backward(); optG.step()

            if use_ema:
                ema.update(G)

            lossG_acc += float(lossG.item()); lossD_acc += float(lossD.item()); steps += 1

        with torch.no_grad():
            if use_ema: ema.copy_to(G_ema)
            samples = G_ema(fixed).cpu()
        save_grid(samples, f'samples/gan/epoch_{epoch:03d}.png', nrow=8)
        torch.save(G.state_dict(), f'checkpoints/GAN_G_epoch_{epoch}.pt')
        torch.save(D.state_dict(), f'checkpoints/GAN_D_epoch_{epoch}.pt')

        fake_folder = 'samples/gan_fake_tmp'
        os.makedirs(fake_folder, exist_ok=True)
        for fpath in os.listdir(fake_folder):
            if fpath.lower().endswith(('.png','.jpg','.jpeg','.bmp','.webp')):
                try: os.remove(os.path.join(fake_folder, fpath))
                except: pass

        counter = 0
        bsize = 64
        iters = (args.n_fake_fid + bsize - 1) // bsize
        for _ in range(iters):
            with torch.no_grad():
                z = torch.randn(bsize, args.nz, 1, 1, device=device)
                f = G_ema(z).cpu()
            f = (f * 0.5 + 0.5).clamp(0,1)
            if f.size(1) == 1: f = f.repeat(1,3,1,1)
            for j in range(f.size(0)):
                save_image(f[j], os.path.join(fake_folder, f'fake_{counter:05d}.png'))
                counter += 1
                if counter >= args.n_fake_fid: break
            if counter >= args.n_fake_fid: break

        try:
            fid, is_mean, is_std = fid_from_folders(
                real_cache, fake_folder,
                batch_size=64, image_size=args.image_size, device=str(device)
            )
        except Exception as e:
            print('[Metrics GAN] Warning:', repr(e))
            fid, is_mean, is_std = float('nan'), float('nan'), float('nan')

        lrG_sched.step(fid)
        if lrD_sched is not None: lrD_sched.step(fid)

        log_rows.append([
            epoch,
            round(lossG_acc/steps, 4),
            round(lossD_acc/steps, 4),
            round(fid, 3),
            round(is_mean, 3),
            round(is_std, 3)
        ])

        if not (fid != fid):  # NaN
            if fid < best_fid - args.min_delta:
                best_fid = fid
                best_epoch = epoch
                torch.save(G_ema.state_dict(), BEST_G_PATH)
                torch.save(D.state_dict(), BEST_D_PATH)
                save_grid(samples, f'samples/gan/best_epoch_{epoch:03d}.png', nrow=8)
                with open('reports/best_checkpoint.json', 'w') as f:
                    json.dump({"epoch": best_epoch, "best_fid": float(best_fid)}, f, indent=2)
                print(f'[Checkpoint] New best FID: {best_fid:.3f} @ epoch {best_epoch}')

        if stopper is not None and not (fid != fid):
            if stopper.step(fid, epoch):
                print(f'[EarlyStop] Stopping at epoch {epoch} (best FID={stopper.best:.3f}).')
                break

    write_csv('reports/gan_training_log.csv', log_rows[0], log_rows[1:])
    make_gif_from_folder('samples/gan', 'reports/gan_progress.gif', pattern='epoch_', duration=0.7)

    if best_epoch != -1:
        print(f'Done. Best FID={best_fid:.3f} @ epoch {best_epoch}')
        print(f'Checkpoints: {BEST_G_PATH} | {BEST_D_PATH}')
    else:
        print('Done. (no valid FID for best checkpoint)')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--nz', type=int, default=100)
    p.add_argument('--ngf', type=int, default=64)
    p.add_argument('--ndf', type=int, default=64)
    p.add_argument('--image-size', type=int, default=64)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--ema', action='store_true', help='Use EMA for G (evaluate/save EMA)')
    p.add_argument('--ema-decay', type=float, default=0.999)
    p.add_argument('--ttur', action='store_true', help='Use TTUR (higher LR for D)')
    p.add_argument('--hinge', action='store_true', help='Use Hinge loss instead of BCE')
    p.add_argument('--early-stop', action='store_true')
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--min-delta', type=float, default=0.0)
    p.add_argument('--n-real-fid', type=int, default=10000, help='#real imgs cached for FID')
    p.add_argument('--n-fake-fid', type=int, default=5000, help='#fake imgs generated per epoch for FID')
    p.add_argument('--lr-patience', type=int, default=5, help='epochs without FID improvement to reduce LR')
    p.add_argument('--lr-both', action='store_true', help='Apply ReduceLROnFID to D as well')

    args = p.parse_args()
    main(args)
