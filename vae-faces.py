#!/bin/env python

import torch;
import torch.nn as nn;
import torchvision as tv;
from torch.utils.data import DataLoader;
import matplotlib.pyplot as pyplot;

b_size = 32;
# celeb = DataLoader(
# 	tv.datasets.CelebA(
# 		root = './dataset/',
# 		split = 'all',
# 		download = False,
# 		target_type = 'attr',
# 		transform = tv.transforms.PILToTensor(),
# 	),
# 	batch_size = 32,
# 	shuffle = True,
# );


img, label = torch.randn(32, 3, 128, 128), torch.randint(2, (32, 40,));
print(img.size(), label.size());

class BAD(nn.Module):
	def __init__(_, in_channels):
		super().__init__();
		_.bat = nn.BatchNorm2d(in_channels);
		_.act = nn.LeakyReLU();
		_.drp = nn.Dropout();

	def forward(_, x):
		x = _.bat(x);
		x = _.act(x);
		x = _.drp(x);
		return x;

class Gaussian(nn.Module):
	def __init__(_, in_feature, out_feature):
		super().__init__();
		_.out_feat = out_feature;
		_.mean = nn.Linear(in_feature, out_feature);
		_.stdv = nn.Linear(in_feature, out_feature);

	def forward(_, x):
		mu = _.mean(x);
		sd = _.stdv(x);
		eps = torch.randn(_.out_feat);
		return mu+eps*torch.exp(sd/2.);

class Encoder(nn.Module):
	def __init__(_, ls_dim):
		super().__init__();
		lyr = [];
		conv_in = [3, 32, 64, 64];
		conv_out = [32, 64, 64, 64];
		_.ls_dim = ls_dim;

		for i in range(4):
			lyr.append(nn.Conv2d(
				in_channels = conv_in[i],
				out_channels = conv_out[i],
				stride = 2,
				kernel_size = 3,
				padding = 1
			));
			lyr.append(BAD(
				in_channels = conv_out[i]
			));
		lyr.append(nn.Flatten());
		lyr.append(Gaussian(4096, _.ls_dim));
		_.layer = nn.ModuleList(lyr);

	def forward(_, x):
		format = "%-35s%s";
		print(format%(x.size(), "Input"));
		for L in _.layer:
			x = L(x);
			print(format%(x.size(), L));
		return x;

enc = Encoder(ls_dim=200)(img);

class Decoder(nn.Module):
	def __init__(_, ls_dim):
		super().__init__();
		lyr = [];

		lyr.append(nn.Linear(

