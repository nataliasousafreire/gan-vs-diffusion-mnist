import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# gan
class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, z): return self.main(z)

class Discriminator(nn.Module):
    def __init__(self, ndf=64, nc=1):
        super().__init__()
        def block(in_c, out_c, k=4, s=2, p=1, bn=True):
            layers = [nn.utils.spectral_norm(nn.Conv2d(in_c, out_c, k, s, p, bias=not bn))]
            if bn: layers += [nn.BatchNorm2d(out_c)]
            layers += [nn.LeakyReLU(0.2, inplace=True)]
            return nn.Sequential(*layers)
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            block(ndf, ndf*2, bn=True),
            block(ndf*2, ndf*4, bn=True),
            block(ndf*4, ndf*8, bn=True),
            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False)
        )
    def forward(self, x): return self.main(x).view(-1, 1).squeeze(1)

# difusao
class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, time_c):
        super().__init__()
        self.time = nn.Sequential(nn.SiLU(), nn.Linear(time_c, out_c))
        self.block1 = nn.Sequential(nn.GroupNorm(8, in_c), nn.SiLU(), nn.Conv2d(in_c, out_c, 3, 1, 1))
        self.block2 = nn.Sequential(nn.GroupNorm(8, out_c), nn.SiLU(), nn.Conv2d(out_c, out_c, 3, 1, 1))
        self.skip = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
    def forward(self, x, t_emb):
        h = self.block1(x)
        h = h + self.time(t_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.skip(x)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim): super().__init__(); self.dim = dim
    def forward(self, t):
        device = t.device; half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=1)


class UNet(nn.Module):
    def __init__(self, base=64, time_dim=128, in_ch=1):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim*4), nn.SiLU(),
            nn.Linear(time_dim*4, time_dim)
        )
        self.in_conv = nn.Conv2d(in_ch, base, 3, 1, 1)
        self.down1 = ResBlock(base, base*2, time_dim)
        self.down2 = ResBlock(base*2, base*4, time_dim)
        self.down3 = ResBlock(base*4, base*4, time_dim)
        self.pool = nn.AvgPool2d(2)
        self.mid  = ResBlock(base*4, base*4, time_dim)
        self.up1  = ResBlock(base*4, base*4, time_dim)
        self.up2  = ResBlock(base*4, base*4, time_dim)
        self.up3  = ResBlock(base*4, base*2, time_dim)
        self.out_conv = nn.Sequential(nn.GroupNorm(8, base*2), nn.SiLU(), nn.Conv2d(base*2, in_ch, 3, 1, 1))
    def forward(self, x, t):
        t = self.time_mlp(t)
        x1 = self.in_conv(x)
        d1 = self.down1(x1, t); p1 = self.pool(d1)
        d2 = self.down2(p1, t); p2 = self.pool(d2)
        d3 = self.down3(p2, t); p3 = self.pool(d3)
        m = self.mid(p3, t)
        u1 = self.up1(m, t);  u1 = F.interpolate(u1, scale_factor=2, mode='nearest'); u1 = u1 + d3
        u2 = self.up2(u1, t); u2 = F.interpolate(u2, scale_factor=2, mode='nearest'); u2 = u2 + d2
        u3 = self.up3(u2, t); u3 = F.interpolate(u3, scale_factor=2, mode='nearest'); u3 = u3 + d1
        out = self.out_conv(u3)
        return out
        
def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

def extract(a, t, x_shape):
    b = t.shape[0]
    out = a.gather(-1, t).float()
    return out.view(b, *((1,) * (len(x_shape) - 1)))

def q_sample(x_start, t, noise, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    return extract(sqrt_alphas_cumprod, t, x_start.shape) * x_start + \
           extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

@torch.no_grad()
def p_sample(model, x, t, betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance, device):
    b = x.shape[0]
    t_tensor = torch.full((b,), t, device=device, dtype=torch.long)
    model_mean = extract(sqrt_recip_alphas, t_tensor, x.shape) * \
                 (x - extract(betas / sqrt_one_minus_alphas_cumprod, t_tensor, x.shape) * model(x, t_tensor))
    if t == 0:
        return model_mean
    noise = torch.randn_like(x)
    return model_mean + torch.sqrt(extract(posterior_variance, t_tensor, x.shape)) * noise

@torch.no_grad()
def p_sample_loop(model, shape, timesteps, betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance, device):
    img = torch.randn(shape, device=device)
    for t in reversed(range(timesteps)):
        img = p_sample(model, img, t, betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance, device)
    return img


