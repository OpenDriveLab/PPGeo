import numpy as np
import cv2
import torch
import torch.utils.data
import torch.nn.functional as F
import torchvision
from pyquaternion import Quaternion
import PIL
from PIL import Image
import os
from torch.utils.data import Dataset
import json

class FuturePredictionDataset(Dataset):
	def __init__(self, root='data', json_path_pattern='p3_%s.json', split='train'):

		self.original_height = 900 
		self.original_width = 1600
		self.final_dim = (224, 480)
		self.resize_scale = 0.3
		self.top_crop = 46


		# Image resizing and cropping
		self.augmentation_parameters = self.get_resizing_and_cropping_parameters()

		# Normalising input images
		self.normalise_image = torchvision.transforms.Compose(
			[torchvision.transforms.ToTensor(),
			 torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
			]
		)

		self.sensor_keys = ['brake_sensor', 'steering_sensor', 'throttle_sensor']

		samples = json.load(open(os.path.join(root, json_path_pattern % split)))
		self.samples = samples
		self.img_root = os.path.join(root, 'nuscenes')


	def get_resizing_and_cropping_parameters(self):
		original_height, original_width = self.original_height, self.original_width
		final_height, final_width = self.final_dim

		resize_scale = self.resize_scale
		resize_dims = (int(original_width * resize_scale), int(original_height * resize_scale))
		resized_width, resized_height = resize_dims

		crop_h = self.top_crop
		crop_w = int(max(0, (resized_width - final_width) / 2))
		# Left, top, right, bottom crops.
		crop = (crop_w, crop_h, crop_w + final_width, crop_h + final_height)

		if resized_width != final_width:
			print('Zero padding left and right parts of the image.')
		if crop_h + final_height != resized_height:
			print('Zero padding bottom part of the image.')

		return {'scale_width': resize_scale,
				'scale_height': resize_scale,
				'resize_dims': resize_dims,
				'crop': crop,
				}
	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):

		data = {}
		sample = self.samples[idx]
		imgs, future_poses = sample['imgs'], sample['future_poses']

		img_path = os.path.join(self.img_root, imgs[0])
		img = Image.open(img_path)
		# Resize and crop
		img = resize_and_crop_image(
			img, resize_dims=self.augmentation_parameters['resize_dims'], crop=self.augmentation_parameters['crop']
		)
		# Normalise image
		normalised_img = self.normalise_image(img)

		data['images'] = normalised_img
				
		# process future_poses
		future_poses = torch.tensor(future_poses)
		future_poses[:, 0] = future_poses[:, 0].clamp(1e-2, )  # the car will never go backward

		data['gt_trajectory'] = future_poses

		return data



def resize_and_crop_image(img, resize_dims, crop):
	# Bilinear resizing followed by cropping
	img = img.resize(resize_dims, resample=PIL.Image.Resampling.BILINEAR)
	img = img.crop(crop)
	return img