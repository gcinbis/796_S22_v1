from pkg_resources import require
from sklearn import discriminant_analysis
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim

from utils import (adjust_gradient, discriminator_loss, 
                   generator_loss, gradient_penalty)

from models.discriminator import Discriminator
from models.generator import Generator


# Sample data from the loader
def get_sample(loader):
    while True:
        for batch in loader:
            yield batch


# Train Generator and Discriminator
def train(config, loader, generator, discriminator, g_optim, d_optim, device):
    '''
    Training function of Generator and Discriminator. 
    
    Arguments:
        args:          Contains process information 
        loader:        Data Loader
        generator:     Style-Swin Transformer Generator
        discriminator: Conv-Based discriminator
        g_optim:       Generator Optimizer
        d_optim:       Discriminator Optimizer
        device:        Training device
        
    '''
    # Yield a batch of data
    loader = get_sample(loader)

    # Set the configuration of training
    losses = {}
    # Initialize gradient penalty
    grad_pen_loss = torch.tensor(0.0, device=device)
    # L2 loss 
    l2 = nn.MSELoss()
    # Gradient clipping
    gradient_clip = nn.utils.clip_grad_

    for epoch in range(config['num_epochs']):
        # ------------------ Train Discriminator -------------------- #
        generator.train()

        # Get batch of images and put them to device
        real_img = next(loader).to(device)
        # Avoid Generator to be updated
        adjust_gradient(generator, False)
        # Permit only discriminator to be updated
        adjust_gradient(discriminator, True)
        
        # Sample random noise from normal distribution
        noise_dim = (config['batch'], config['initial_channel']) # ~ initial channel 512
        noise = torch.randn(noise_dim).to(device) # ~ maybe .cuda()?

        # Generate Fake image from random noise
        fake_img = generator(noise)
        # Get discriminator performance on generated images
        fake_pred = discriminator(fake_img)
        # Get discriminator performance on real images
        real_pred = discriminator(real_img)

        # Calculate Discriminator Loss
        d_loss = discriminator_loss(real_pred, fake_pred)

        # Update discriminator
        discriminator.zero_grad()
        d_loss.backward()
        gradient_clip(discriminator.parameters(), 5.0)
        d_optim.step()

        # Employ Gradient Penalty
        if epoch % config['d_reg_every'] == 0:
            real_img.requires_grad = True
            # Get the prediction on updated discriminator
            real_pred = discriminator(real_img)
            # Calculate the R1 loss: Gradient penalty
            grad_pen_loss = gradient_penalty(real_pred, real_img)
            
            # Update Discriminator
            discriminator.zero_grad()
            grad_pen_loss.backward() # ~ Ideally add some weighting... to this loss
            d_optim.step()

        # Save the losses
        losses['discriminator'] = d_loss        
        losses['gradient_penalty'] = grad_pen_loss

        
        # ------------------ Train Generator -------------------- #
        
        # Avoid Discriminator to be updated
        adjust_gradient(discriminator, False)
        # Permit only generator to be updated
        adjust_gradient(generator, True)

        # Get the next batch of real images
        real_img = next(loader).to(device)

        # Sample random noise from normal distribution
        noise_dim = (config['batch'], config['initial_channel']) # ~ initial channel 512
        noise = torch.randn(noise_dim).to(device) # ~ maybe .cuda()?

        # Generate Fake image from random noise
        fake_img = generator(noise)
        # Get discriminator performance on generated images
        fake_pred = discriminator(fake_img)
        
        # Calculate the Generator loss
        g_loss = generator_loss(fake_pred) # Ideally, add weight

        # Save the loss
        losses['generator'] = g_loss

        # Update Generator
        generator.zero_grad()
        g_loss.backward()
        g_optim.step()


if __name__ == '__main__':
    
    # Configuration of training
    config = {
        'batch':  128,
        'initial_channel':  512,
        'd_reg_every':  50,
        'num_epochs': ...  
    }

    # Generator 
    generator = Generator(...)
    generator_learning_rate = ...
    generator_betas = (... , ....)
    g_optim = optim.Adam(generator.parameters(), 
                        lr=generator_learning_rate, 
                        betas=generator_betas)

    # Discriminator
    discriminator = Discriminator(...)
    discriminator_learning_rate = ...
    discriminator_betas = (... , ....)
    d_optim = optim.Adam(discriminator.parameters(), 
                        lr=generator_learning_rate, 
                        betas=generator_betas)
    
    # Set device
    ...
