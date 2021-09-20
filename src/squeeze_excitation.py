import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class SqueezeExcitation(nn.Module):
	def __init__(self, in_dim, in_channels, reduced_dim):
		super(SqueezeExcitation, self).__init__()
		self.reduce = nn.MaxPool2d(kernel_size = (in_dim, 1))
		self.squeeze = nn.Conv2d(
				in_channels = in_channels,
				out_channels = reduced_dim,
				kernel_size = 1
			)
		self.excite = nn.Conv2d(
				in_channels = reduced_dim,
				out_channels = in_channels,
				kernel_size = 1
			)

	def forward(self, x):
		out = self.reduce(x)
		out = self.squeeze(x)
		out = self.excite(x)

		return x * out