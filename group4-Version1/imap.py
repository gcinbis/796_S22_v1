import torch
from torch import nn
import math
import numpy as np
class IMAP(nn.Module):
	def __init__(self, input_channels):
		super(IMAP, self).__init__()
		self.input_channels = input_channels
		self.weight = torch.empty([self.input_channels, self.input_channels,1])
		nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
		#self.weight = torch.nn.Parameter(self.weight).cuda().type(torch.float64)
		self.bias = torch.empty([self.input_channels])
		fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
		bound = 1 / math.sqrt(fan_in)
		nn.init.uniform_(self.bias, -bound, bound)
		self.conv1d = nn.Conv1d(self.input_channels, self.input_channels, kernel_size=1, bias=True)
		#self.bias = torch.nn.Parameter(self.bias).cuda()
		self.register_parameter("s", nn.Parameter(torch.randn([1, self.input_channels, 1])))
		self.register_parameter("offset", nn.Parameter(torch.ones([1])*8))
		self.pool1 = torch.nn.AvgPool1d(self.input_channels)

	def forward(self, input: torch.Tensor, logdet=0, reverse=False, permute=False):
		if not reverse:
			self.num_channels = input.shape[-1]**2
			#checkerboard
			B, C, H, W = input.shape
			self.mask = torch.tensor(np.ones((B,C, H*W), dtype=np.float64)).cuda()

			deneme_1 = torch.tensor(np.ones((1, 1, H*W), dtype=np.float64))
			deneme_1.flatten()[::2] = 0
			deneme_0 = torch.tensor(np.zeros((1, 1, H*W), dtype=np.float64))
			deneme_0.flatten()[::2] = 1
			cat_deneme = torch.cat((deneme_1, deneme_0))
			cat_deneme_1 = torch.cat((deneme_1, deneme_0))

			for i in range(0, C//2 - 1):
				cat_deneme = torch.cat((cat_deneme, cat_deneme_1))
			cat_deneme = cat_deneme.view(1,C,H*W)

			cat_son = torch.cat((cat_deneme, cat_deneme))
			cat_son_1 = torch.cat((cat_deneme, cat_deneme))

			for i in range(1, B//2):
				cat_son = torch.cat((cat_son, cat_son_1))

			self.mask = cat_son.cuda()
			sig = torch.nn.Sigmoid()
			input_masked = input.view(B, C, H*W) * self.mask
			z = self.conv1d(input_masked.type(torch.FloatTensor).cuda())
			z_new = z.transpose(1, 2)
			pool_out = self.pool1(z_new)
			attn_out = (sig(pool_out.squeeze(-1) + self.offset) + 1e-5).unsqueeze(1)
			attn_mask = (1 - self.mask) * attn_out + self.mask * (sig(self.s) + 1e-5)
			out_new = input * attn_mask.view(B, C, H*W).view(B, C, H, W)
			logdet = logdet + torch.sum((self.input_channels//2) * (torch.log(sig(pool_out.squeeze(-1)+ self.offset)+1e-5)), dim=-1)
			logdet = logdet + torch.sum(torch.log(sig(self.s)+1e-5) * self.mask)
			return out_new, logdet
		else:
			out_new = input
			self.num_channels = input.shape[-1]**2
			B, C, H, W = out_new.shape
			self.mask = torch.tensor(np.ones((B, C, H * W), dtype=np.float64)).cuda()

			deneme_1 = torch.tensor(np.ones((1, 1, H * W), dtype=np.float64))
			deneme_1.flatten()[::2] = 0
			deneme_0 = torch.tensor(np.zeros((1, 1, H * W), dtype=np.float64))
			deneme_0.flatten()[::2] = 1
			cat_deneme = torch.cat((deneme_1, deneme_0))
			cat_deneme_1 = torch.cat((deneme_1, deneme_0))

			for i in range(0, C // 2 - 1):
				cat_deneme = torch.cat((cat_deneme, cat_deneme_1))
			cat_deneme = cat_deneme.view(1, C, H * W)

			cat_son = torch.cat((cat_deneme, cat_deneme))
			cat_son_1 = torch.cat((cat_deneme, cat_deneme))

			for i in range(1, B // 2):
				cat_son = torch.cat((cat_son, cat_son_1))

			self.mask = cat_son.cuda()
			sig = torch.nn.Sigmoid()
			s_sig = sig(self.s) + 1e-5
			s_sig_in = torch.ones_like(s_sig) / s_sig
			inp_masked = out_new.view(B, C, H*W) * self.mask * s_sig_in
			out_conv = self.conv1d(inp_masked.type(torch.FloatTensor).cuda())
			pool_out = self.pool1(out_conv.transpose(1, 2))
			attn_out = (sig(pool_out.squeeze(2) + self.offset) + 1e-5).unsqueeze(1)
			attn_mask = torch.ones_like(attn_out) / attn_out
			input_rev = out_new * (attn_mask.view(B, C, H*W).view(B, C, H, W))
			logdet = logdet - torch.sum((self.input_channels//2) * (torch.log(sig(pool_out.squeeze(-1) + self.offset)+1e-5)), dim=-1)
			logdet = logdet - torch.sum(torch.log(sig(self.s) + 1e-5) * self.mask)
			return input_rev, logdet