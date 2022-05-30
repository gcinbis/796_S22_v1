import torch
import torch.nn as nn
from models.sinusoidal_positional_embedding import SinusoidalPositionalEmbedding

# Basic building layers

class AdaIN(nn.Module):
    '''
    Adaptive Instance Normalization Layer. 
    
    Each Swin Transformer Block will contain 2 AdaIN Layer: 
        1. On the top that takes Constant Input and Style vector
        2. After Double Attention Layer which takes attended feature map and Style vector.
                        |----------------------------------------------------|
                        |                      AdaIN                         |
                        |----------------------------------------------------|
                        |                 Double Attention                   |
                        |----------------------------------------------------|
                        |                      AdaIN                         |   
                        |----------------------------------------------------|
                        |                       MLP                          |
                        |----------------------------------------------------|

    Equation:
                     <------------- Scale-----------> <--Shift-->
        AdaIN(x, y) = std(y) x [x - mean(x) / std(x)] +  mean(y)
                                
    Args:
        n_channels: Number of channels of the feature map -> [C]                      
        n_style:    Dimension of the style vector
    '''
    def __init__(self, n_channels, n_style):
        super().__init__()

        self.instance_norm = nn.InstanceNorm1d(n_channels)
        # To get the mean and std of the Style vector. Linear Transformation is applied
        # to align the dimensions of mean & std of style vector with channel norm of input feature map.
        self.style_linear = nn.Linear(n_style, 2 * n_channels) ### scaledLinear + nonlinearity => optional 

    def forward(self, input, style): 
        ''' 
        Dimensions:
            
            Input:  [B x C x D x D]
            Style:  [B x n_style]
            Output: [B x C x D * D] 
        
        Example:
        >> Input: [32 x 512 x 4 x 4]
        >> Style: [32, 10]
        >> Output: [32, 512, 16]
        '''
        # Get mean and std of style vector for every batch instance                  
        style = self.style_linear(style).unsqueeze(-1)
        # separate 2 x n_channels => [mean, std]
        mean, std = style.chunk(2,1) 
        # adaptive normalize along channels
        norm = self.instance_norm(input.view(input.size(0), input.size(1), -1))
        # shift + scale
        norm = mean + std * norm
        # # [ B x C x L ] => [ B x L x C ]
        return norm.transpose(-1,-2)



class PixelNorm(nn.Module):
    '''
    Pixel Norm is a common choice in most of the GAN implementations.
    
    When the signal magnitudes gets out control (meaning that, it gets smaller and 
    smaller or bigger and bigger) PixelNorm prevents the signal escalating.

    Equation:
                    PixelNorm(a) = a / mean_square_root(a)
    
    For Numerical issues, an epsilon term is added to denominator in general.
    '''

    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        # Epsilon term for numerical stability
        eps = 1e-8
        # rsqrt is just the reverse square root.
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + eps)


class ConstInput(nn.Module):
    '''
    Generator takes a constant [N x N x C] dimensional feature map and gradually 
    upsamples this feature map through cascade of transformer blocks. 
    
    Here, the dimension [N x N x C] corresponds to [H, W, C] ordering.
    Important Note: Every batch will get the same constant input. 
    
    Args:
        n_spatial = Spatial dimension of the constant input: [N x N]
        n_channel = Depth   dimension of the constant input: [C]
    Returns:
        const: Generator input of size [C x N x N] [Channel, Height, Width] 
    Paper Details:
        In the paper, authors propose to sample from Standard Normal Distribution & start with spatial dimensionality of 4.
    '''

    def __init__(self, n_channel, n_spatial=4):
        super().__init__()
        # Create the constant input that will be repeated across batch dimension
        self.const = nn.Parameter(torch.randn(1, n_channel, n_spatial, n_spatial))

    def forward(self, input):
        # Get the number of batches
        n_batch = input.size(0)
        # Repeat the const along batch axis
        constant_input = self.const.repeat(n_batch, 1, 1, 1) # Size: [B x C x N x N]

        return constant_input

class MLP(nn.Module):
    '''
    Multilayer Perceptron Layer. 
    In each transformer block in Generator, MLP block is used right after second AdaIN.
    It takes the output of the AdaIN layer [Instance Normalized feature map] and pass through Linear + Activation Layers.    
    '''
    def __init__(self, input_dim, 
                       hidden_dim, 
                       out_dim, 
                       activation=nn.GELU, 
                       drop_rate=0.0):
        
        # Cascade the layers to form MLP
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation,
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(drop_rate))

    def forward(self, input):
        return self.net(input)

class tRGB(nn.Module):
    '''
    tRGB module is used to convert input feature map to output activation map with channel dimension of 3 [R, G, B].

    Generator Transformer Block uses tRGB module in the following way:

                      Transformer Block L                                 Transformer Block L+1
                ----------------------------------                  ----------------------------------
                |             ...                |                  |             ...                |
                |---------------------------------  [N x N]         |---------------------------------  [(2 x N) x (2 x N)]
                |             MLP                |                  |             MLP                |
                ----------------------------------                  ----------------------------------           
                            |                                                   |                          
                            |          ----------                               |         ----------
                            -------->  |  tRGB  |                               ------->  |  tRGB  |
                                        ----------                                         ----------   
                                            |                                                   ^
                                            |                       --------                    |             
                                            |-----------------------|  UP  |---------------------  
                                                                    --------  
                                Fig: tRGB module between 2 consecutive Generator Transformer Block
    Explanation of Fig:

        After the last layer of the Generator Transformer Block L, which is the MLP layer, the input of the tRGB layer is 
        the embedding map, 

    '''                           
    def __init__(self, input_channels, kernel_size):
        # reduce input_channels => 3 channels (r,g,b)
        self.conv_layer = nn.Conv2d(input_channels, 3, kernel_size=kernel_size)
    
    def forward(self, x):
        return self.conv_layer(x)
    
class Upsample(nn.Module):
    '''
    Expects [ N x D x D x C ] input => converts to [ N x C x D x D ] 
        => interpolates to [ N x C x 2.D x 2.D ] => returns [ N x 2.D x 2.D x C ] 
    ## Expects [ N x C x D x D ] input => Returns [ N x C x 2.D x 2.D ]
    N: batch size
    C: channel size
    D: input dimension D x D

    Corresponds to Up block in generator diagram figure 2. in the paper.
    '''
    def __init__(self, isLearnt=False): 
        # isLearnt whether upsampling parameters are learnable or not
        # TransposedConv => learnable params, torch.upsample => interpolation no learned params
        # if isLearnt:
        #     self.upsamle = nn.ConvTranspose2d(in_channels=input_channels, out_channels=2*input_channels)
        # else:
        # conv has to be updated, we cannot upsample 2 channels at the same time
        self.upsamle = torch.nn.Upsample(scale_factor=2, mode="nearest") # no learnable parameters, interpolation
    def forward(self, x):
        # x shape [B H W C]
        x = x.permute(0,3,1,2) # [B C H W]
        x = self.upsample(x) # [B C 2.H 2.W]
        x = x.permute(0,2,3,1) # [B 2.H 2.W C] 
        return x



def window_partition(x, window_size):
    """
    Taken from Original SWIN implementation
    Creates windows from the given input/img x of size [window_size x window_size]
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Taken from Original SWIN implementation
    Creates img/output from the given window partitions of size [window_size x window_size]
    num_windows => number of patches per img
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class BasicAttentionBlock(nn.Module):
    ''''
    Taken from Original SWIN implementation
    Main Swin Block

    Taken from swin existing implementation => ask for validity?

    window_dim: window size assuming [window_dim x window_dim]
    '''
    def __init__(self, input_channels, heads, window_dim, attn_dropout=0):
        super().__init__()
        self.input_channels = input_channels
        # partition channels across different heads: how many channels per head
        self.head_size = input_channels // heads
        self.heads = heads
        self.window_dim = window_dim

        self.qkv = nn.Linear(input_channels, input_channels * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_dropout)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = attn_dropout

        # define a parameter table of relative position bias
        # bu kismi tam anlayamadim swin de de boyle yapmis paperdaki kismi: 5.sayfa sol alt, swin paper
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_dim - 1) * (2 * window_dim - 1), heads))  # 2*Wh-1 * 2*Ww-1, nH
        # initialize with truncated gaussian values
        torch.nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_dim)
        coords_w = torch.arange(window_dim)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_dim - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_dim - 1
        relative_coords[:, :, 0] *= 2 * window_dim - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)


    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: queries with shape of (num_windows*B, N, C)
            k: keys with shape of (num_windows*B, N, C)
            v: values with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = q.shape
        q = q.reshape(B_, N, self.heads, C // self.heads).permute(0, 2, 1, 3)
        k = k.reshape(B_, N, self.heads, C // self.heads).permute(0, 2, 1, 3)
        v = v.reshape(B_, N, self.heads, C // self.heads).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_dim * self.window_dim, self.window_dim * self.window_dim, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)  # add rpe on top

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        
        return x


class DoubleAttentionBlock(nn.Module):
    '''
    Double Attention block depicted in the generator diagram.
    W-MSA => SW-MSA
    Takes 2 inputs: 
        1. signal from RPE (relative positional embedding)
        2. signal from the output of ADAIN (also added skip connection)

    input_channels: number of channels
    input_reso: input resolution eg. [256 x 256] = [h,w]
    window_dim: window dimension to apply window based attention on
    style_dim: dimension of the style vector that is fed at each style-swin block
    heads: number of attention heads

    '''
    def __init__(self, input_channels, input_reso, window_dim, style_dim, heads, attn_dropout):
        super().__init__()
        self.window_dim = window_dim
        # partition heads and channels among 2 attn blocks
        self.attn1 = BasicAttentionBlock(input_channels=input_channels//2, heads=heads//2, window_dim=window_dim, attn_dropout=attn_dropout)
        self.attn2 = BasicAttentionBlock(input_channels=input_channels//2, heads=heads//2, window_dim=window_dim, attn_dropout=attn_dropout)
        # linear embedding that is applied after attention block
        # this procedure was normally a part of swin attention block, however we need to apply
        # attention blocks sequentially and then linear projection afterwards
        self.attn_proj = nn.Linear(window_dim, window_dim)
    
    def forward(self, qkvm1, qkvm2):
        '''
        qkvm1: list of [query, key, value, mask] matrices for first attention block (W-MSA)
        qkvm2: list of [query, key, value, mask] matrices for second attention block (SW-MSA)
        '''
        # extract q k v m for each qkvm
        q1, k1, v1, m1 = qkvm1
        q2, k2, v2, m2 = qkvm2
        
        # W-MSA propagation
        x_w = self.attn1(q1, k1, v1, m1)
        # SW-MSA propagation
        x_sw = self.attn2(q2, k2, v2, m2)

        return x_w, x_sw




class StyleSwinBlock(nn.Module):
    def __init__(self, n_channels, n_style, hid_dim, window_dim, input_reso, heads, attn_dropout):
        super().__init__()
        '''
        Fundamental block in style swin architecture
        starts with const_input 4x4x512 => 8x8x512 => ...
        
        Architecture:
        1. AdaIn
        2. DoubleAttn
        3. AdaIn
        4. MLP

        Inputs:
        n_channels: input channel dim
        hid_dim: hidden dimension for the mlp
        window_dim: window dim
        input_reso: input resolution eg. [256x256]
        heads: number of attention heads
        n_style: Dimension of the style vector
        attn_dropout: dropout in attention block
        '''
        
        # save some parameters required for the forward
        self.window_dim = window_dim
        self.shift = window_dim//2 # shift the window by half of the window_dim
        self.heads = heads
        self.n_channels = n_channels
        self.hid_dim = hid_dim
        self.n_style = n_style

        # extract input resolution
        height, width = input_reso
        self.height = height
        self.width = width
        self.L = height*width

        # layers to pass before passing through BasicAttentionBlock
        self.qkv_proj = nn.Linear(n_channels, 3*n_channels) # in order to obtain q,k,v
        self.attn_proj = nn.Linear(n_channels, n_channels) # applied after passed the attention block
        # main blocks
        # Adaptive Instance Norm Block - 1
        self.adaIN_1 = AdaIN(self.n_channels, self.n_style)
        # Double Attention Block
        self.attn = DoubleAttentionBlock(n_channels, input_reso, window_dim, n_style, heads, attn_dropout)
        # Adaptive Instance Norm Block - 2
        self.adaIN_2 = AdaIN(self.n_channels, self.n_style)
        # MLP layer: n_channels => hid_dim => n_channels
        self.mlp = MLP(self.n_channels, self.hid_dim, self.n_channels)

        # Base Case
        # in case window_dim > input_reso[height] or input_reso[width] => windows not possible
        if window_dim>self.height or window_dim>self.width:
            # take window based on smallest resolution dimension
            self.window_dim =  min(self.height, self.width) 
            self.shift = 0  # shift not possible
        
        # W-MSA Mask Calculation
        # Taken From Swin implementation
        if self.shift > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_dim),
                        slice(-self.window_dim, -self.shift),
                        slice(-self.shift, None))
            w_slices = (slice(0, -self.window_dim),
                        slice(-self.window_dim, -self.shift),
                        slice(-self.shift, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            # padding overflown window
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        
    def forward(self, x, style_vec):
        '''
        Inputs:
        x:  signal coming from the previous styleswin block or the starting input (const 4x4x512)
            shape:  [ B x L x C ]
            
        style_vec: style vector that is computed from style block (noise massaging) 
                that is fed to each AdaIN blocks (2 per styleswinblock) for each styleswinblock
                Note that style vectors are linearly projected (depicted in diagram as A blocks) 
                in AdaIN block

        [ B x D x D x C ]
        [ B x L x C ]  where L = D x D
        [ B x C x L ]
        '''
        
        B,L,C = x.shape
        # added between double attention and adaIN-2 layer
        skip_con_1 = x
        
        # adaIN norm - 1 
            # [B x L x C]
        x = self.adaIN_1(x, style_vec)
        
        # double attention
            # create [query,key,value] matrices
            # [B x L x 3C] => [B x L x 3 x C]
        x = self.qkv_proj(x)
            # objective: obtain [3 x B x L x C]=[q,k,v] => divide for heads
            # [3 x B x L x C/2]=[q1,k1,v1], [3 x B x L x C/2]=[q2,k2,v2]
        x = x.reshape(B, L, 3, C) # [B L 3 C]
        x = x.permute(2,0,1,3) # [3 B L C]: bring 3-B near otherwise reshape meaningless
        x = x.reshape(3*B, L, C) # [3*B x L x C]
        x = x.reshape(3*B, self.height, self.width, C) # split L=> height,width: [3*B, H, W, C]
        # split among heads
        qkv1 = x[:,:,:,:C//2] # take first half channels for qkv1
        qkv1 = qkv1.reshape(3, B, self.height, self.width, -1) # [3 B H W C//2]
        qkv2 = x[:,:,:,C//2:] # take second half channels for qkv2
        qkv2 = qkv2.reshape(3, B, self.height, self.width, -1) # [3 B H W C//2]

        # split qkv into [q k v] since it is of shape [3 ....]
            # W-MSA qkv
        q1, k1, v1 = qkv1

        # mask for the second attn depends on the shift
            # case wout shift/mask => W-MSA type
        
        # Based on Swin Imp
        # cyclic shift
        if self.shift > 0:
            qkv2 = torch.roll(qkv2, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        q2, k2, v2 = qkv2
        qkvm2 = [q2, k2, v2, None]

        # partition windows
            # x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
            # x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        qkv1 = [window_partition(x, self.window_dim).view(-1, self.window_dim * self.window_dim, C//2)\
                for x in qkv1] # calculate windows for [q k v]-1
        qkv2 = [window_partition(x, self.window_dim).view(-1, self.window_dim * self.window_dim, C//2)\
                for x in qkv2] # calculate windows for [q k v]-2
        
        # create qkvm then pass through double attn
        qkvm1 = qkv1 + [None] # no attention mask for W-MSA
        qkvm2 = qkv2 + [self.attn_mask] # no attention mask for SW-MSA

        # W-MSA/SW-MSA
        # attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        # [q k v m] calculated for each W-MSA and SW-MSA
        # pass through double attention
        x_w, x_sw = self.attn(qkvm1, qkvm2)

        # merge windows
        # attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        # shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        x_w = window_reverse(x_w.view(-1, self.window_dim * self.window_dim, C // 2), self.window_dim, self.height, self.width)
        x_sw = window_reverse(x_sw.view(-1, self.window_dim * self.window_dim, C // 2), self.window_dim, self.height, self.width)


        # reverse cyclic shift
        if self.shift > 0:
            x_sw = torch.roll(x_sw, shifts=(self.shift, self.shift), dims=(1, 2))
        
        # x_w, x_sw of shape [B,H,W,C//2]
        # out of shape [B,H,W,C]
        out = torch.cat(x_w, x_sw, dim=3)
        # attn output projection
        out = self.attn_proj(out)
        skip_con_2 = out
        # attn processes have finalized => adaIN-2 + MLP
        # since adaIN accepts shape of [B L C]
        out = out.reshape(B, self.L, C)
        out = self.adaIN_2(out + skip_con_1)
        out = self.mlp(out) + skip_con_2

        return out


class StyleBlock(nn.Module):
    def __init__(self, n_channels, n_style, hid_dim, window_dim, input_reso, heads, attn_dropout) -> None:
        '''
        B x H x W x C
        - takes input of resolution input_shape
        - adds SPE (sin pos embed) on input
        - passes through StyleSwinBlock
        - outputs 2 signals: 1 upsampled and 1 trgb-ed

        '''
        super().__init__()
        # sin-pos-embed
        self.height, self.width = input_reso
        self.spe = SinusoidalPositionalEmbedding(embedding_dim=n_channels//2, padding_idx=0, init_size=n_channels//2) # missing, how to set, how initz
        self.block = StyleSwinBlock(n_channels, n_style, hid_dim, window_dim, input_reso, heads, attn_dropout)
        self.up = Upsample()
        self.trgb = tRGB(n_channels, kernel_size=1)
    
    def forward(self, x, style_vec):
        B,H,W,C = x.shape
        # add sin pos embed (spe)
        x = x + x.spe
        x = self.block(x, style_vec)
        # signal for side network: img generation & up pipeline
        x_trgb = self.trgb(x)
        # signal for the next styleblock
        x_up = self.up(x)

        return x_up, x_trgb


class NoiseMassage(nn.Module):
    def __init__(self, n_style, n_layers, activation=nn.ReLU) -> None:
        '''
        Network that creates the massaged noise from initial gaussian noise vector.
        It is like dough knealding.

        n_style: dimensionality of the style vector which is also equal to noise dimension
        n_layers: number of fcc layers stacked
        '''
        super().__init__()

        self.n_layers = n_layers
        self.n_style = n_style

        self.massage_mlp = nn.ModuleList()

        for i in range(n_layers):
            self.massage_mlp.append(nn.Linear(n_style, n_style))
            self.massage_mlp.append(activation)
        
        self.massage_mlp = nn.Sequential(self.massage_mlp)
        self.pixel_norm = PixelNorm()

    def forward(self, noise):
        style = self.pixel_norm(noise)
        style = self.massage_mlp(style)
        return style

class Generator(nn.Module):
    def __init__(self, n_style, n_layers_style_net, n_layers_styleblock, heads, attn_dropout, \
        activation_noise_net=nn.ReLU, n_channels=512, n_spatial=4, hid_dim=256, window_dim=4,
        upsample_isLearnt=False
        ) -> None:
        '''
        Generator block that is composed of blocks that are created above.
        '''
        self.n_layers_styleblock = n_layers_styleblock
        self.n_layers_style_net = n_layers_style_net

        self.n_channels = n_channels
        super().__init__()
        # noise massage network
        self.noise_net = NoiseMassage(n_style=n_style, n_layers=n_layers_style_net, activation=activation_noise_net)
        # styleblock network
        self.style_blocks = nn.ModuleList()
        self.const_input = ConstInput(n_channel=n_channels, n_spatial=n_spatial) # [4 x 4 x 512]
        # first style block
        self.style_blocks.append(StyleBlock(n_channels=n_channels, n_style=n_style, hid_dim=hid_dim,\
             window_dim=window_dim, input_reso=(n_spatial,n_spatial), heads=heads, attn_dropout=attn_dropout))
        # up network for image generation
        self.up_blocks = nn.ModuleList()

        for i in range(n_layers_styleblock-1):
            self.style_blocks.append(StyleBlock(n_channels=n_channels, n_style=n_style, hid_dim=hid_dim,\
                window_dim=window_dim, input_reso=(n_spatial*(2**(i+1)),n_spatial*(2**(i+1))),\
                heads=heads, attn_dropout=attn_dropout))
            self.up_blocks.append(Upsample())


    def forward(self, noise):
        '''
        Forward propagation starts with a noise vector
        First noise is massaged => style vector created => fed to many  styleblocks
        each styleblock outputs 2 signal: 1 signal for the next styleblock (x_up), other signal (x_trgb)
        is for image generation network which upsamples and is added to the x_trgb output of next styleblock
        '''
        style = self.noise_net(noise)
        
        x_up, x_trgb = styleblock(self.const_input, style)

        for styleblock, upblock in zip(self.style_blocks[1:], self.up_blocks):
            x_trgb_prev = x_trgb
            x_up, x_trgb = styleblock(x_up, style)
            x_trgb = upblock(x_trgb_prev) + x_trgb

        # returns generated upsampled tensor corresponding image
        return x_trgb