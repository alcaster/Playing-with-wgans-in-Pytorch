import argparse

from torchvision.transforms import transforms
from torchvision.utils import save_image

import torch.nn as nn
import torch

from dataset import FacesDataset
from utils import CheckpointSaver

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=5000, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=128, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.00005, help='learning rate')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the hidden space')
parser.add_argument('--img_size', type=int, default=120, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--n_critic', type=int, default=5, help='number of training steps for discriminator per iter')
parser.add_argument('--clip_value', type=float, default=0.01, help='lower and upper clip value for disc. weights')
parser.add_argument('--sample_interval', type=int, default=2000, help='interval betwen image samples')
parser.add_argument('--save_interval', type=int, default=30, help='interval betwen checkpoint saving')
parser.add_argument('--save_dir_name', type=str, default="images_faces_wgan",
                    help='save path for images and checkpoints')
opt = parser.parse_args()
print(opt)

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, opt.img_size ** 2),
            nn.Tanh()
        )

    def forward(self, noise):
        img = self.model(noise)
        img = img.view(img.shape[0], opt.channels, opt.img_size, opt.img_size)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.img_size ** 2, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

dataloader = torch.utils.data.DataLoader(FacesDataset(root_dir="data/images",
                                                      extention="ppm",
                                                      transform=transforms.Compose([
                                                          transforms.Grayscale(),
                                                          transforms.Resize((opt.img_size, opt.img_size)),
                                                          transforms.ToTensor(),
                                                          # TODO normalize according to dataset stats.
                                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                      ])), batch_size=opt.batch_size,
                                         shuffle=True, num_workers=opt.n_cpu)

# Optimizers
optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lr)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr)

discriminator_saver = CheckpointSaver(opt.save_dir_name, max_checkpoints=3)
generator_saver = CheckpointSaver(opt.save_dir_name, max_checkpoints=3)
# ----------
#  Training
# ----------
batches_done = 0
for epoch in range(opt.n_epochs):

    # Batch iterator
    data_iter = iter(dataloader)

    for i in range(len(data_iter) // opt.n_critic):
        # Train discriminator for n_critic times
        for _ in range(opt.n_critic):
            imgs = data_iter.next()

            # Adversarial ground truths
            valid = Tensor(imgs.shape[0], 1).fill_(-1.0)
            fake = Tensor(imgs.shape[0], 1).fill_(1.0)

            # Configure input
            real_imgs = Tensor(imgs.type(Tensor))

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = torch.randn(imgs.shape[0], opt.latent_dim, device=device)

            # Generate a batch of images
            fake_imgs = generator(z)

            # Train on real images
            real_validity = discriminator(real_imgs)
            real_validity.backward(valid)
            # Train on fake images
            fake_validity = discriminator(fake_imgs)
            fake_validity.backward(fake)

            d_loss = real_validity - fake_validity

            optimizer_D.step()

            # Clip weights of discriminator
            for p in discriminator.parameters():
                p.data.clamp_(-opt.clip_value, opt.clip_value)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = torch.randn(imgs.shape[0], opt.latent_dim, device=device)

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        # Train on fake images
        gen_validity = discriminator(gen_imgs)
        gen_validity.backward(valid)

        optimizer_G.step()

        print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, opt.n_epochs,
                                                                         opt.n_critic * (i + 1),
                                                                         len(dataloader),
                                                                         d_loss.data[0], gen_validity.data[0]))

        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], f'{opt.save_dir_name}/{batches_done}.png', nrow=5, normalize=True)
        if batches_done % opt.save_interval == 0:
            discriminator_saver.save(discriminator, batches_done, optimizer_D, epoch)
            generator_saver.save(generator, batches_done, optimizer_G, epoch)
        batches_done += opt.n_critic
