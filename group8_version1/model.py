# implement histogan model
from re import S
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

class ResidualBlock(torch.nn.Module):
    def __init__(self, inp_dim, out_dim):
        super().__init__()
        ## residual block architecture is taken from the paper Figure S1
        self.conv1 = torch.nn.Conv2d(inp_dim, out_dim, kernel_size=3, padding=1, stride=1)
        self.conv2 = torch.nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, stride=1)
        self.conv1x1 = torch.nn.Conv2d(inp_dim, out_dim, kernel_size=1, padding=0, stride=1)
        self.conv3 = torch.nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, stride=2)
        self.lrelu = torch.nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.lrelu(self.conv1(x))
        out = self.lrelu(self.conv2(out))
        out += self.conv1x1(x)
        out = self.conv3(out)
        return out

class Discriminator(torch.nn.Module):
    def __init__(self, num_res_block=7, network_capacity=16):
        super().__init__()
        ## Discriminator architecture taken from Section 5.1. Details of Our Networks
        residual_block_layers = []
        in_dim = 3
        output_channels = network_capacity
        for _ in range(num_res_block):
            residual_block_layers.append(ResidualBlock(in_dim, output_channels))
            in_dim = output_channels
            output_channels = 2*output_channels 
        self.residual_layers = torch.nn.Sequential(*residual_block_layers)

        ## test tensor is used to calculate the input dimension to the fc layer
        test =  torch.zeros((3,256,256))
        with torch.no_grad():
            out = self.residual_layers(test)
        linear_inp = out.size(0) * out.size(1) * out.size(2)

        self.fc = torch.nn.Linear(linear_inp, 1)
    
    def forward(self, x):
        out = self.residual_layers(x)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

class ModDemodConv3x3(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # StyleGAN papers mention each weight is initalized with N(0, I)
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, 3, 3))
    
    def convolution(self, inp, weight):
        # Implementaion of convolution operation using torch.unfold 
        # Taken from https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html
        # Changed such that it can work with batch of styles
        # After the operation the output feature maps' spatial size should be the same as the input, therefore padding=1
        inp_unf = torch.nn.functional.unfold(inp, (3, 3), padding=(1,1)) 
        # print(weight.view(weight.size(0), weight.size(1), -1).transpose(1, 2).size())
        # print(inp_unf.transpose(1, 2).size())
        out_unf = inp_unf.transpose(1, 2).matmul(weight.view(weight.size(0), weight.size(1), -1).transpose(1, 2)).transpose(1, 2)
        out = torch.nn.functional.fold(out_unf, (inp.size(2), inp.size(3)), (1, 1))
        return out    
    
    def forward(self, x, style):
        style = style.view(style.size(0),1,-1,1,1)
        modulated_weight = style * self.weight
        sigma = torch.sqrt(modulated_weight.square().sum(dim=(2,3,4), keepdim=True) +1e-3)
        # print(sigma.size())
        demodulated_weight = modulated_weight / sigma
        out = self.convolution(x, demodulated_weight)
        return out     

class ModConv1x1(torch.nn.Module):
    def __init__(self, in_channels, out_channels=3):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, 1, 1))

    def convolution(self, inp, weight):
        # Implementaion of convolution operation using torch.unfold 
        # Taken from https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html
        # Changed such that it can work with batch of styles
        # After the operation the output feature maps' spatial size should be the same as the input, therefore padding=1
        inp_unf = torch.nn.functional.unfold(inp, (1, 1)) 
        out_unf = inp_unf.transpose(1, 2).matmul(weight.view(weight.size(0), weight.size(1), -1).transpose(1, 2)).transpose(1, 2)
        out = torch.nn.functional.fold(out_unf, (inp.size(2), inp.size(3)), (1, 1))
        return out    

    def forward(self, x, style):
        style = style.view(style.size(0),1,-1,1,1)
        modulated_weight = style * self.weight
        out = self.convolution(x, modulated_weight)
        return out 

class StyleGAN2Block(torch.nn.Module):
    # StyleGan2 Block depicted in Figure 2/A of HistoGAN paper
    def __init__(self, input_channel_size, output_channel_size): # these channel numbers should be decided soon
        super().__init__()
        self.affine_1 = torch.nn.Linear(512, input_channel_size)
        self.affine_2 = torch.nn.Linear(512, output_channel_size)
        self.affine_3 = torch.nn.Linear(512, output_channel_size)
        # In stylegan2 paper, section B. Implementation Details/Generator redesign  
        # authors say that they used only single shared scaling factor for each feature map 
        # and they initiazlize the factor by 0
        self.noise_scaling_factor_1 = torch.nn.Parameter(torch.zeros((1)))
        self.noise_scaling_factor_2 = torch.nn.Parameter(torch.zeros((1))) 
        self.md3x3_1 = ModDemodConv3x3(input_channel_size, output_channel_size)
        self.md3x3_2 = ModDemodConv3x3(output_channel_size, output_channel_size)
        self.m1x1 = ModConv1x1(output_channel_size, 3)
        self.bias_1 = torch.nn.Parameter(torch.zeros(output_channel_size))
        self.bias_2 = torch.nn.Parameter(torch.zeros(output_channel_size))
        self.lrelu = torch.nn.LeakyReLU(0.2)


    def forward(self, fm, w):
        # fm: input feature map (B, C, H, W), C = channel1
        # latent vector w, w = f(z), (B, 512)
        # in figure noise vectors are outside the block
        # but it does no matter since they are independent and random
        # in style gan2 paper each noise is tought as a N(0,I) image 
        # with same height and with as the feature map
        batch, channel, height, width = fm.size()  
        noise_1 = torch.randn((batch, 1, height, width)).to(device)
        noise_2 = torch.randn((batch, 1, height, width)).to(device)
        style_1 = self.affine_1(w)
        style_2 = self.affine_2(w)
        style_3 = self.affine_3(w)
        # if bias is not reshaped gives error, this version can broadcast to batch
        out = self.md3x3_1(fm, style_1) + self.noise_scaling_factor_1 * noise_1 + self.bias_1.view(-1,1,1) 
        out = self.lrelu(out)
        out = self.md3x3_2(out, style_2) + self.noise_scaling_factor_2 * noise_2 + self.bias_2.view(-1,1,1)
        out = self.lrelu(out)
        rgb_out = self.m1x1(out, style_3) # rgb_out should then be upsampled, for now left it out of this block
        return out, rgb_out

class HistoGAN(torch.nn.Module):
    def __init__(self, network_capacity=16, h=64, channel_sizes = [1024, 512, 512, 512, 256, 128, 64]):
        super().__init__()
        latent_mapping_list = []
        latent_mapping_list.extend([torch.nn.Linear(512, 1024), 
                                   torch.nn.LeakyReLU(0.2),
                                   torch.nn.Linear(1024, 512),
                                   torch.nn.LeakyReLU(0.2)])
        hist_projection_list = []
        hist_projection_list.extend([torch.nn.Linear(h*h*3, 1024), 
                                    torch.nn.LeakyReLU(0.2),
                                    torch.nn.Linear(1024, 512),
                                    torch.nn.LeakyReLU(0.2)]) 
        for i in range(6):
            if i == 5:
                latent_mapping_list.append(torch.nn.Linear(512, 512))
                hist_projection_list.append(torch.nn.Linear(512, 512))
            else:
                latent_mapping_list.extend([torch.nn.Linear(512, 512),
                                           torch.nn.LeakyReLU(0.2)])
                hist_projection_list.extend([torch.nn.Linear(512, 512),
                                            torch.nn.LeakyReLU(0.2)])

        self.latent_mapping = torch.nn.Sequential(*latent_mapping_list)
        self.hist_projection = torch.nn.Sequential(*hist_projection_list)
        self.learned_const_inp = torch.nn.Parameter(torch.randn(4*network_capacity, 4, 4))
        self.upsample = torch.nn.Upsample(scale_factor=2, mode="bilinear")
        stylegan2_block_list = [] 
        inp_size = 4 * network_capacity
        for channel_size in channel_sizes:
            stylegan2_block_list.append(StyleGAN2Block(inp_size, channel_size))
            inp_size = channel_size
        self.stylegan2_blocks = torch.nn.ModuleList(stylegan2_block_list)

    def forward(self, z, target_hist):
        # noise input z, size: B, 512
        B = z.size(0)
        w = self.latent_mapping(z)
        fm = self.learned_const_inp.unsqueeze(0).repeat(B, 1, 1, 1)
        for i, stylegan2_block in enumerate(self.stylegan2_blocks[:-1]):
            fm, rgb = stylegan2_block(fm, w)
            fm = self.upsample(fm)
            if i == 0:
                rgb = self.upsample(rgb)
                rgb_sum = rgb
            else:
                rgb_sum += rgb
                rgb_sum = self.upsample(rgb_sum)
        hist_w  = self.hist_projection(target_hist.flatten(1))
        _, rgb = self.stylegan2_blocks[-1](fm, hist_w)
        rgb_sum += rgb
        # w is returned to compute path length regularization
        return rgb_sum, w 
    