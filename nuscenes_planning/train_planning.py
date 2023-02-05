import argparse
import os
from tqdm import tqdm
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F


import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

from planning_model import Planning_Model
from data import FuturePredictionDataset



class PlannerEngine(pl.LightningModule):
	def __init__(self,config):
		super().__init__()
		self.lr = config.lr
		self.model = Planning_Model()

		ckpt = torch.load(config.pretrained_ckpt,map_location='cpu')['state_dict']

		new_state_dict = OrderedDict()
		for key, value in ckpt.items():
			new_key = key.replace("motionnet.visual_encoder.encoder.","")
			new_state_dict[new_key] = value
		self.model.perception.load_state_dict(new_state_dict, strict = False)
	
	def forward(self, batch):
		pass

	def training_step(self, batch, batch_idx):

		front_img = batch['images']
		gt_trajectory = batch['gt_trajectory']
		gt_trajectory = gt_trajectory[:, :, :2]

		pred_gt_trajectory = self.model(front_img)

		loss_l2 = torch.sqrt(((pred_gt_trajectory - gt_trajectory) ** 2).sum(dim=-1)).mean()

		loss_l1 = F.l1_loss(pred_gt_trajectory, gt_trajectory, reduction='none').mean()
		
		self.log('train_loss_l1', loss_l1.item())
		self.log('train_loss_l2', loss_l2.item())
		return loss_l2

	def configure_optimizers(self):
		optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr, weight_decay=1e-4)
		lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 15, 0.1)
		return [optimizer], [lr_scheduler]

	def validation_step(self, batch, batch_idx):
		front_img = batch['images']
		gt_trajectory = batch['gt_trajectory']
		gt_trajectory = gt_trajectory[:, :, :2]

		pred_gt_trajectory = self.model(front_img)

		loss_l2 = torch.sqrt(((pred_gt_trajectory - gt_trajectory) ** 2).sum(dim=-1)).mean()

		loss_l2_1s = torch.sqrt(((pred_gt_trajectory[:, :2, :] - gt_trajectory[:, :2, :]) ** 2).sum(dim=-1)).mean()
		loss_l2_2s = torch.sqrt(((pred_gt_trajectory[:, :4, :] - gt_trajectory[:, :4, :]) ** 2).sum(dim=-1)).mean()

		loss_l1 = F.l1_loss(pred_gt_trajectory, gt_trajectory, reduction='none').mean()
		
		self.log('val_loss_l1', loss_l1.item(), sync_dist=True)
		self.log('val_loss_l2_1s', loss_l2_1s.item(), sync_dist=True)
		self.log('val_loss_l2_2s', loss_l2_2s.item(), sync_dist=True)
		self.log('val_loss_l2', loss_l2.item(), sync_dist=True)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument('--id', type=str, default='ppgeo_pretrain', help='Unique experiment identifier.')
	parser.add_argument('--epochs', type=int, default=30, help='Number of train epochs.')
	parser.add_argument('--pretrained_ckpt', type=str, help='pretrained ckpt')
	parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
	parser.add_argument('--val_every', type=int, default=1, help='Validation frequency (epochs).')
	parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
	parser.add_argument('--logdir', type=str, default='log', help='Directory to log data to.')

	args = parser.parse_args()
	args.logdir = os.path.join(args.logdir, args.id)

	dataroot = 'data/nuscenes'
	train_set = FuturePredictionDataset('data', 'p3_6pts_can_bus_%s.json', 'train')
	val_set = FuturePredictionDataset('data', 'p3_6pts_can_bus_%s.json', 'val')
	print(len(train_set))
	print(len(val_set))

	dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8)
	dataloader_val = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=8)

	planner = PlannerEngine(args)

	checkpoint_callback = ModelCheckpoint(save_weights_only=False, mode="min", monitor="val_loss_l2", save_top_k=1, save_last=True,
											dirpath=args.logdir, filename="best_{epoch:02d}-{val_loss_l2:.3f}")
	checkpoint_callback.CHECKPOINT_NAME_LAST = "{epoch}-last"
	trainer = pl.Trainer.from_argparse_args(args,
											default_root_dir=args.logdir,
											gpus = 4,
											accelerator='ddp',
											sync_batchnorm=True,
											plugins=DDPPlugin(find_unused_parameters=False),
											profiler='simple',
											benchmark=True,
											log_every_n_steps=1,
											flush_logs_every_n_steps=5,
											callbacks=[checkpoint_callback,
														],
											check_val_every_n_epoch = args.val_every,
											max_epochs = args.epochs
											)

	trainer.fit(planner, dataloader_train, dataloader_val)




		




