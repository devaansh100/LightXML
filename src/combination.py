import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinationBlock(nn.Module):
	def __init__(self, n_features):
		super(CombinationBlock, self).__init__()
		self.alpha = Vector(n_features)
		self.beta  = Vector(n_features)
		self.gamma = Vector(n_features)

	def forward(self, x, bert_tokens, labels_cooccur):
		out = np.multiply(self.alpha, x) + np.multiply(self.beta, bert_tokens)
		if self.train():
			out += np.multiply(self.gamma, labels_cooccur)

		return out


class Vector(nn.Module):
	def __init__(self, size):
		super(Vector, self).__init__()
		self.vector = torch.tensor(torch.random(size, 1), requires_grad=True)
