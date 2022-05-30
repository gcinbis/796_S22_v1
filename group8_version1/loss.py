# implement loss functions
from this import d
import utils
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
# See Eq. 5
# g: generated image, d_score: scalar output of discriminator
def non_sat_generator_loss(g, d_score, hist_t):
    #c_loss = hellinger_dist_loss(g, hist_t)
    #alpha = 2.0  # See Sec. 5.2 Training details
    g_loss = torch.mean(torch.log(torch.sigmoid(d_score))) #- alpha*c_loss
    return -g_loss 

def classics_disc_loss(g_scores, r_scores):
    return -torch.mean(torch.log(torch.sigmoid(r_scores))) - torch.mean(torch.log(1-torch.sigmoid(g_scores)))

def wgan_gp_gen_loss(disc_score):
    return -torch.mean(disc_score)

def wgan_gp_disc_loss(real_scores, fake_scores, gradient_penalty, coeff_penalty):
    return -torch.mean(real_scores) + torch.mean(fake_scores) + coeff_penalty*gradient_penalty

def r1_reg(batch_data, discriminator, r1_factor):
    # for autograd.grad to work input should also have requires_grad = True
    batch_data_grad = batch_data.clone().detach().requires_grad_(True)
    real_score_for_r1 = discriminator(batch_data_grad)
    gradients1 = torch.autograd.grad(outputs=real_score_for_r1, inputs=batch_data_grad, grad_outputs=torch.ones(real_score_for_r1.size()).to(device))[0]
    r1_reg = torch.mean(torch.sum(torch.square(gradients1.view(gradients1.size(0), -1)), dim=1))
    return r1_factor*r1_reg  

def pl_reg(fake_data, w, target_scale, plr_factor, ema_decay_coeff):
    gradients2 = torch.autograd.grad(outputs=fake_data*torch.randn_like(fake_data).to(device), inputs=w, grad_outputs=torch.ones(fake_data.size()).to(device), retain_graph=True)[0]
    j_norm  = torch.sqrt(torch.sum(torch.square(gradients2.view(gradients2.size(0), -1)),dim=1))
    plr = torch.mean(torch.square(j_norm - target_scale))
    pl_reg = plr * plr_factor
    target_scale = (1-ema_decay_coeff)* target_scale + ema_decay_coeff * j_norm
    return pl_reg, target_scale

# This is color matching loss, see Eq. 4
# It takes histogram of generated and target
def hellinger_dist_loss(g, hist_t):
    hist_g = utils.histogram_feature_v2(g, device=device)  # Compute histogram feature of generated img
    t_sqred = torch.sqrt(hist_t)
    g_sqred = torch.sqrt(hist_g)
    diff = t_sqred - g_sqred
    h = torch.sum(torch.square(diff), dim=(1,2,3))
    # print(hist_t.min(), hist_g.min())
    h_norm = torch.sqrt(h)
    h_norm = h_norm * (torch.sqrt(torch.ones((g.shape[0]))/2))
    
    # Used mean reduction, other option is sum reduction
    h_norm = torch.mean(h_norm)

    return h_norm

def compute_gradient_penalty(fake_data, real_data, discriminator):
    a = torch.rand((fake_data.size(0), 1, 1, 1)).to(device)
    comb_data = a* fake_data + (1-a)*real_data
    comb_data = comb_data.requires_grad_(True)
    comb_score = discriminator(comb_data)
    gradients = torch.autograd.grad(outputs=comb_score, inputs=comb_data, grad_outputs=torch.ones(comb_score.size()).to(device), create_graph=True, retain_graph=True)[0]
    gradient_norm = torch.sqrt(1e-8+torch.sum(torch.square(gradients.view(gradients.size(0), -1)), dim=1))
    gradient_penalty = torch.mean(torch.square(gradient_norm-1))
    return gradient_penalty
