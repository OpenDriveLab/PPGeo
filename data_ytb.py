import os
from PIL import Image
from PIL import ImageFilter
import random

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as T

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def pil_loader(path):
	# open path as file to avoid ResourceWarning
	# (https://github.com/python-pillow/Pillow/issues/835)
	with open(path, 'rb') as f:
		with Image.open(f) as img:
			return img.convert('RGB')


class YTB_Data(Dataset):

	def __init__(self, root, meta_path, is_train):
		self.data_root = root
		self.is_train = is_train
		meta = np.load(meta_path, allow_pickle=True).item()
		self.prev_path = meta["prev_path"]
		self.cur_path = meta["cur_path"]
		self.next_path = meta["next_path"]
		self.video_idx = meta["video_idx"]

		self.num_scales = len([0, 1, 2, 3])
		self.height = 160
		self.width = 320
		self.interp = Image.ANTIALIAS
		self.frame_idxs = [0, -1, 1]

		self.loader = pil_loader
		self.to_tensor = T.ToTensor()
		self.brightness = (0.8, 1.2)
		self.contrast = (0.8, 1.2)
		self.saturation = (0.8, 1.2)
		self.hue = (-0.1, 0.1)

		self.resize = {}
		for i in range(self.num_scales):
			s = 2 ** i
			self.resize[i] = T.Resize((self.height // s, self.width // s),
											   interpolation=self.interp)

	def __len__(self):
		"""Returns the length of the dataset. """
		return len(self.cur_path)

	def preprocess(self, inputs, color_aug):
		"""Resize colour images to the required scales and augment if required
		We create the color_aug object in advance and apply the same augmentation to all
		images in this item. This ensures that all images input to the pose network receive the
		same augmentation.
		"""
		for k in list(inputs):
			frame = inputs[k]
			if "color" in k:
				n, im, i = k
				for i in range(self.num_scales):
					inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

		for k in list(inputs):
			f = inputs[k]
			if "color" in k:
				n, im, i = k
				inputs[(n, im, i)] = self.to_tensor(f)
				inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

	def __getitem__(self, index):
		"""Returns the item at index idx. """

		inputs = {}
		do_color_aug = self.is_train

		inputs[("color", 0, -1)] = self.loader(os.path.join(self.data_root, self.cur_path[index])).crop((0, 10, 320, 170))
		inputs[("color", -1, -1)] = self.loader(os.path.join(self.data_root, self.prev_path[index])).crop((0, 10, 320, 170))
		inputs[("color", 1, -1)] = self.loader(os.path.join(self.data_root, self.next_path[index])).crop((0, 10, 320, 170))

		if do_color_aug:
			color_aug = T.Compose([
            T.RandomApply([
                T.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)  # not strengthened
            ], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([GaussianBlur([.1, 2.])], p=0.5)])
		else:
			color_aug = (lambda x: x)

		self.preprocess(inputs, color_aug)

		for i in self.frame_idxs:
			del inputs[("color", i, -1)]
			del inputs[("color_aug", i, -1)]

		return inputs


