# CelebA image generation with DCGAN

# Попробуем RMSprop оптимайзер c параметрами  weight_decay=1e-8 вместо всеми любимого Adam 
# и возьмем LeakyReLU вместо обычной ReLU, добавим Dropout(0.4)
# добавила class CSP(nn.Module) и добавила его во второй слой деконволюции блоком CSP.

import torch
from torch.autograd import Variable
import torch.nn as nn



import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio

# Parameters
image_size = 64
G_input_dim = 100
G_output_dim = 3
D_input_dim = 3
D_output_dim = 1
num_filters = [1024, 512, 256, 128]

learning_rate = 0.0002
betas = (0.5, 0.999)
batch_size = 128
num_epochs = 50
data_dir = '../Data/celebA_data/resized_celebA'
save_dir = 'CelebA_DCGAN_results/'

# CelebA dataset
transform = transforms.Compose([transforms.Resize(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

celebA_data = dsets.ImageFolder(data_dir, transform=transform)

data_loader = torch.utils.data.DataLoader(dataset=celebA_data,
                                          batch_size=batch_size,
                                          shuffle=True)


# De-normalization
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)




#######       CSP block:      ####### 
class CSP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CSP, self).__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim // 2
        self.block1 = nn.Sequential(
            nn.Conv2d(self.out_channels, out_channels=self.out_channels, kernel_size= 1, stride=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(self.out_channels, self.out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(self.out_channels)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(self.out_channels, out_channels=self.out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(self.out_channels, out_channels=self.out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(self.out_channels)
        )
        self.skip_upscale = nn.ConvTranspose2d(self.out_channels, self.out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        
    def forward(self, x):
        x_1, x_2 = x.split(self.out_channels, dim=1)
        x_2 = self.block1(x_2)
        x_2 = self.block2(x_2)
        x_1 = self.skip_upscale(x_1)
        out = x_1 + x_2
        return out
#######       how it looks over the Internet:      ####### 


###############      CSP INSIDE MY GENERATOR:  ##############
#          CSP block is added as the second hidden layer   #
# Generator 
class Generator(torch.nn.Module):
    def __init__(self, input_dim, num_filters, output_dim):
        super(Generator, self).__init__()

        # Hidden layers
        self.hidden_layer = torch.nn.Sequential()
        for i in range(len(num_filters)):
            # Deconvolutional layer
            if i == 0:
                deconv = torch.nn.ConvTranspose2d(input_dim, num_filters[i], kernel_size=4, stride=1, padding=0)
            elif i == 1:  # Add CSP block as the second hidden layer
                csp_block = CSP(num_filters[i-1], num_filters[i])
                self.hidden_layer.add_module('csp', csp_block)
                deconv = torch.nn.ConvTranspose2d(num_filters[i], num_filters[i], kernel_size=4, stride=2, padding=1)
            else:
                deconv = torch.nn.ConvTranspose2d(num_filters[i-1], num_filters[i], kernel_size=4, stride=2, padding=1)

            deconv_name = 'deconv' + str(i + 1)
            self.hidden_layer.add_module(deconv_name, deconv)

            # Initializer
            torch.nn.init.normal(deconv.weight, mean=0.0, std=0.02)
            torch.nn.init.constant(deconv.bias, 0.0)

            # Batch normalization
            bn_name = 'bn' + str(i + 1)
            self.hidden_layer.add_module(bn_name, torch.nn.BatchNorm2d(num_filters[i]))

            # Activation
            act_name = 'act' + str(i + 1)
            self.hidden_layer.add_module(act_name, torch.nn.LeakyReLU())

        # Output layer
        self.output_layer = torch.nn.Sequential()
        # Deconvolutional layer
        out = torch.nn.ConvTranspose2d(num_filters[i], output_dim, kernel_size=4, stride=2, padding=1)
        self.output_layer.add_module('out', out)
        # Initializer
        torch.nn.init.normal(out.weight, mean=0.0, std=0.02)
        torch.nn.init.constant(out.bias, 0.0)
        # Activation
        self.output_layer.add_module('act', torch.nn.Tanh())

    def forward(self, x):
        h = self.hidden_layer(x)
        out = self.output_layer(h)
        return out


# Discriminator 
class Discriminator(torch.nn.Module):
    def __init__(self, input_dim, num_filters, output_dim):
        super(Discriminator, self).__init__()

        self.hidden_layer = torch.nn.Sequential()
        for i in range(len(num_filters)):
            if i == 0:
                conv = torch.nn.Conv2d(input_dim, num_filters[i], kernel_size=4, stride=2, padding=1)
            else:
                conv = torch.nn.Conv2d(num_filters[i-1], num_filters[i], kernel_size=4, stride=2, padding=1)

            conv_name = 'conv' + str(i + 1)
            self.hidden_layer.add_module(conv_name, conv)

            torch.nn.init.normal(conv.weight, mean=0.0, std=0.02)
            torch.nn.init.constant(conv.bias, 0.0)

            if i != 0:
                bn_name = 'bn' + str(i + 1)
                self.hidden_layer.add_module(bn_name, torch.nn.BatchNorm2d(num_filters[i]))

            act_name = 'act' + str(i + 1)
            self.hidden_layer.add_module(act_name, torch.nn.LeakyReLU(0.2))

        # Output layer
        self.output_layer = torch.nn.Sequential()
        # Convolutional layer
        out = torch.nn.Conv2d(num_filters[i], output_dim, kernel_size=4, stride=1, padding=0)
        self.output_layer.add_module('out', out)
        drop=torch.nn.Dropout(p=0.3, inplace=False)
        self.output_layer.add_module('out', out)


        # Initializer
        torch.nn.init.normal(out.weight, mean=0.0, std=0.02)
        torch.nn.init.constant(out.bias, 0.0)

        # Activation
        self.output_layer.add_module('act', torch.nn.Sigmoid())

    def forward(self, x):
        h = self.hidden_layer(x)
        out = self.output_layer(h)
        return out


def plot_loss(d_losses, g_losses, num_epoch, save=False, save_dir='CelebA_DCGAN_results/', show=False):
    fig, ax = plt.subplots()
    ax.set_xlim(0, num_epochs)
    ax.set_ylim(0, max(np.max(g_losses), np.max(d_losses))*1.1)
    plt.xlabel('Epoch {0}'.format(num_epoch + 1))
    plt.ylabel('Loss values')
    plt.plot(d_losses, label='Discriminator')
    plt.plot(g_losses, label='Generator')
    plt.legend()

    # save 
    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_fn = save_dir + 'CelebA_DCGAN_losses_epoch_{:d}'.format(num_epoch + 1) + '.png'
        plt.savefig(save_fn)

    if show:
        plt.show()
    else:
        plt.close()


def plot_result(generator, noise, num_epoch, save=False, save_dir='CelebA_DCGAN_results/', show=False, fig_size=(5, 5)):
    generator.eval()

    noise = Variable(noise.cuda(), volatile=True)
    gen_image = generator(noise)
    gen_image = denorm(gen_image)

    generator.train()

    n_rows = np.sqrt(noise.size()[0]).astype(np.int32)
    n_cols = np.sqrt(noise.size()[0]).astype(np.int32)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)
    for ax, img in zip(axes.flatten(), gen_image):
        ax.axis('off')
        ax.set_adjustable('box-forced')
        # Scale to 0-255
        img = (((img - img.min()) * 255) / (img.max() - img.min())).cpu().data.numpy().transpose(1, 2, 0).astype(np.uint8)
        ax.imshow(img, cmap=None, aspect='equal')
    plt.subplots_adjust(wspace=0, hspace=0)
    title = 'Epoch {0}'.format(num_epoch+1)
    fig.text(0.5, 0.04, title, ha='center')

    # save 
    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_fn = save_dir + 'CelebA_DCGAN_epoch_{:d}'.format(num_epoch+1) + '.png'
        plt.savefig(save_fn)

    if show:
        plt.show()
    else:
        plt.close()


# Models
G = Generator(G_input_dim, num_filters, G_output_dim)
D = Discriminator(D_input_dim, num_filters[::-1], D_output_dim)
G.cuda()
D.cuda()

# Loss 
criterion = torch.nn.BCELoss()

# Optimizers
G_optimizer = torch.optim.RMSprop(G.parameters(), lr=learning_rate, betas=betas,  weight_decay=1e-8)
D_optimizer = torch.optim.RMSprop(D.parameters(), lr=learning_rate, betas=betas,  weight_decay=1e-8)

# Training GAN
D_avg_losses = []
G_avg_losses = []

#  noise for test
num_test_samples = 5*5
fixed_noise = torch.randn(num_test_samples, G_input_dim).view(-1, G_input_dim, 1, 1)

for epoch in range(num_epochs):
    D_losses = []
    G_losses = []


    for i, (images, _) in enumerate(data_loader):

        # image 
        mini_batch = images.size()[0]
        x_ = Variable(images.cuda())

        # labels
        y_real_ = Variable(torch.ones(mini_batch).cuda())
        y_fake_ = Variable(torch.zeros(mini_batch).cuda())

        #  discriminator with real data
        D_real_decision = D(x_).squeeze()
        # print(D_real_decision, y_real_)
        D_real_loss = criterion(D_real_decision, y_real_)

        #  discriminator with fake data
        z_ = torch.randn(mini_batch, G_input_dim).view(-1, G_input_dim, 1, 1)
        z_ = Variable(z_.cuda())
        gen_image = G(z_)

        D_fake_decision = D(gen_image).squeeze()
        D_fake_loss = criterion(D_fake_decision, y_fake_)

        D_loss = D_real_loss + D_fake_loss
        D.zero_grad()
        D_loss.backward()
        D_optimizer.step()

        # Train generator
        z_ = torch.randn(mini_batch, G_input_dim).view(-1, G_input_dim, 1, 1)
        z_ = Variable(z_.cuda())
        gen_image = G(z_)

        D_fake_decision = D(gen_image).squeeze()
        G_loss = criterion(D_fake_decision, y_real_)

        D.zero_grad()
        G.zero_grad()
        G_loss.backward()
        G_optimizer.step()

        # loss 
        D_losses.append(D_loss.data[0])
        G_losses.append(G_loss.data[0])

        print('Epoch [%d/%d], Step [%d/%d], D_loss: %.4f, G_loss: %.4f'
              % (epoch+1, num_epochs, i+1, len(data_loader), D_loss.data[0], G_loss.data[0]))

    D_avg_loss = torch.mean(torch.FloatTensor(D_losses))
    G_avg_loss = torch.mean(torch.FloatTensor(G_losses))

    D_avg_losses.append(D_avg_loss)
    G_avg_losses.append(G_avg_loss)

    plot_loss(D_avg_losses, G_avg_losses, epoch, save=True)

    plot_result(G, fixed_noise, epoch, save=True, fig_size=(5, 5))




ax1 = plt.subplot(1, 1, 1)
ax1.plot(range(len(D_avg_losses)), D_avg_losses, label='Generator loss')
ax1.plot(range(len(G_avg_losses)), G_avg_losses, label='Discriminator loss')
ax1.set_xlabel('Iterations')
ax1.set_ylabel('Loss')
ax1.legend()

###################
# Set scond x-axis
ax2 = ax1.twiny()
newlabel = list(range(num_epochs+1))
iter_per_epoch = len(data_loader)
newpos = [e*iter_per_epoch for e in newlabel]

ax2.set_xticklabels(newlabel[::10])
ax2.set_xticks(newpos[::10])

ax2.xaxis.set_ticks_position('bottom')
ax2.xaxis.set_label_position('bottom')
ax2.spines['bottom'].set_position(('outward', 45))
ax2.set_xlabel('Epochs')
ax2.set_xlim(ax1.get_xlim())
###################

plt.show()















loss_plots = []
gen_image_plots = []
for epoch in range(num_epochs):
    # plot for generating gif
    save_fn1 = save_dir + 'CelebA_DCGAN_losses_epoch_{:d}'.format(epoch + 1) + '.png'
    loss_plots.append(imageio.imread(save_fn1))

    save_fn2 = save_dir + 'CelebA_DCGAN_epoch_{:d}'.format(epoch + 1) + '.png'
    gen_image_plots.append(imageio.imread(save_fn2))

imageio.mimsave(save_dir + 'CelebA_DCGAN_losses_epochs_{:d}'.format(num_epochs) + '.gif', loss_plots, fps=5)
imageio.mimsave(save_dir + 'CelebA_DCGAN_epochs_{:d}'.format(num_epochs) + '.gif', gen_image_plots, fps=5)
