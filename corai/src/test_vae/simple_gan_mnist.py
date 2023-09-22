# Script to train a first GAN example. We train a GAN to learn the MNIST dataset.

import sys
import torch
import torch.nn as nn
import torchvision.datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.utils as vutils
from matplotlib import pyplot as plt
from torchvision import datasets, transforms

batch_size = 64


class Discriminator(torch.nn.Module):
    def __init__(self, inp_dim=784):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(inp_dim, 128)
        self.nonlin1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.view(x.size(0), 784)  # flatten (bs x 1 x 28 x 28) -> (bs x 784)
        h = self.nonlin1(self.fc1(x))
        out = self.fc2(h)
        out = torch.sigmoid(out)
        return out


class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(z_dim, 128)
        self.nonlin1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(128, 784)

    def forward(self, x):
        h = self.nonlin1(self.fc1(x))
        out = self.fc2(h)
        out = torch.tanh(out)  # range [-1, 1]
        # convert to image
        out = out.view(out.size(0), 1, 28, 28)
        return out


train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True)

discriminator = Discriminator()
generator = Generator()

optimizerD = torch.optim.SGD(discriminator.parameters(), lr=0.01)
optimizerG = torch.optim.SGD(generator.parameters(), lr=0.01)
criterion = nn.BCELoss()

#########################
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('Device: ', device)
# Re-initialize D, G:
discriminator.to(device)
generator.to(device)
# Now let's set up the optimizers (Adam, better than SGD for this)
optimizerD = torch.optim.SGD(discriminator.parameters(), lr=0.03)
optimizerG = torch.optim.SGD(generator.parameters(), lr=0.03)
# optimizerD = torch.optim.Adam(D.parameters(), lr=0.0002)
# optimizerG = torch.optim.Adam(G.parameters(), lr=0.0002)
lab_real = torch.ones(64, 1, device=device)
lab_fake = torch.zeros(64, 1, device=device)

# for logging:
collect_x_gen = []
fixed_noise = torch.randn(64, 100, device=device)
fig = plt.figure()  # keep updating this one
plt.ion()


def show_imgs(x, new_fig=True):
    grid = vutils.make_grid(x.detach().cpu(), nrow=8, normalize=True, pad_value=0.3)
    grid = grid.transpose(0, 2).transpose(0, 1)  # channels as last dimension
    if new_fig:
        plt.figure()
    plt.imshow(grid.numpy())
    plt.axis('off')


for epoch in range(10):  # 10 epochs
    for i, data in enumerate(train_loader, 0):
        # STEP 1: Discriminator optimization step
        x_real, y_real = data[0].to(device), data[1].to(device)
        # reset accumulated gradients from previous iteration
        optimizerD.zero_grad()

        D_x = discriminator(x_real)
        lossD_real = criterion(D_x, lab_real[:D_x.size(0)])

        z = torch.randn(64, 100, device=device)  # random noise, 64 samples, z_dim=100
        x_gen = generator(z).detach()
        D_G_z = discriminator(x_gen)
        lossD_fake = criterion(D_G_z, lab_fake)

        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizerD.step()

        # STEP 2: Generator optimization step
        # reset accumulated gradients from previous iteration
        optimizerG.zero_grad()

        z = torch.randn(64, 100, device=device)  # random noise, 64 samples, z_dim=100
        x_gen = generator(z)
        D_G_z = discriminator(x_gen)
        lossG = criterion(D_G_z, lab_real)  # -log D(G(z))

        lossG.backward()
        optimizerG.step()
        if i % 200 == 0:
            x_gen = generator(fixed_noise)
            show_imgs(x_gen, new_fig=False)
            fig.canvas.draw()
            print('e{}.i{}/{} last mb D(x)={:.4f} D(G(z))={:.4f}'.format(
                epoch, i, len(data), D_x.mean().item(), D_G_z.mean().item()))

    # End of epoch
    x_gen = generator(fixed_noise)
    collect_x_gen.append(x_gen.detach().clone())

    for x_gen in collect_x_gen:
        show_imgs(x_gen)

plt.show()
