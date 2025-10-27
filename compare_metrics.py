import argparse, pandas as pd, matplotlib.pyplot as plt

def main(args):
    gan = pd.read_csv('reports/gan_training_log.csv')
    dif = pd.read_csv('reports/diffusion_training_log.csv')

    fig, axs = plt.subplots(1, 3, figsize=(12,4))
    axs[0].plot(gan['epoch'], gan['lossG'], label='GAN lossG')
    axs[0].plot(gan['epoch'], gan['lossD'], label='GAN lossD')
    axs[0].plot(dif['epoch'], dif['mse_loss'], label='Diffusion MSE')
    axs[0].set_title('Loss per epoch'); axs[0].legend(); axs[0].grid(True, alpha=0.3)

    axs[1].plot(gan['epoch'], gan['FID'], label='GAN FID')
    axs[1].plot(dif['epoch'], dif['FID'], label='Diffusion FID')
    axs[1].set_title('FID ↓'); axs[1].legend(); axs[1].grid(True, alpha=0.3)

    axs[2].plot(gan['epoch'], gan['IS_mean'], label='GAN IS')
    axs[2].plot(dif['epoch'], dif['IS_mean'], label='Diffusion IS')
    axs[2].set_title('Inception Score ↑'); axs[2].legend(); axs[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig('reports/compare_gan_vs_diffusion.png', dpi=160)
    print('Saved: reports/compare_gan_vs_diffusion.png')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--noop', action='store_true')
    args = p.parse_args()
    main(args)
