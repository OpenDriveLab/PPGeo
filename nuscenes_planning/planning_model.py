import numpy as np
import torch 
from torch import nn
from resnet import *

class Planning_Model(nn.Module):
	def __init__(self):
		super().__init__()
		self.perception = resnet34(pretrained=False)
		self.perception.fc = nn.Sequential()

		self.join = nn.Sequential(
							nn.Linear(512, 256),
							nn.ReLU(inplace=True),
							nn.Linear(256, 256),
						)
		self.decoder = nn.GRUCell(input_size=2, hidden_size=256)
		self.output = nn.Linear(256, 2)

		self.pred_len = 6

	def forward(self, img):
		feature_emb = self.perception(img)
		j = self.join(feature_emb)
		z = j
		output_wp = list()

		# initial input variable to GRU
		x = torch.zeros(size=(j.shape[0], 2)).type_as(j)

		# autoregressive generation of output waypoints
		for _ in range(self.pred_len):
			x_in = x
			z = self.decoder(x_in, z)
			dx = self.output(z)
			x = dx + x
			output_wp.append(x)

		pred_wp = torch.stack(output_wp, dim=1)

		return pred_wp