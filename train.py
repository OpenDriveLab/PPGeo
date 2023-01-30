import argparse
import os
from collections import OrderedDict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

from model import Monodepth, MotionNet
from data_ytb import YTB_Data

class PPGeoEngine(pl.LightningModule):
	def __init__(self, config):
		super().__init__()
		self.stage = config.stage
		assert self.stage in [1,2]
		self.lr = config.lr
		self.config = config
		self.stage = self.stage
		self.model = Monodepth(stage = self.stage, batch_size=config.batch_size)
		if self.stage == 2:
			self.motionnet = MotionNet()
			path_to_ckpt_file = config.ckpt
			ckpt = torch.load(path_to_ckpt_file, map_location='cpu')
			ckpt = ckpt["state_dict"]
			new_state_dict = OrderedDict()
			for key, value in ckpt.items():
				new_key = key.replace("model.","")
				new_state_dict[new_key] = value
			self.model.load_state_dict(new_state_dict, strict = True)
			self.model.eval()
			for param in self.model.parameters():
				param.requires_grad = False
	
	def forward(self, batch):
		pass

	def training_step(self, batch, batch_idx):
		if self.stage == 1:
			outputs, losses = self.model(batch)
		else:
			self.model.eval()
			motion = self.motionnet(batch)
			outputs, losses = self.model(batch, *motion)
		for k,v in losses.items():
			self.log('train_{}'.format(k), v.item())

		return losses['loss']

	def configure_optimizers(self):
		if self.stage == 2:
			optimizer = optim.AdamW(self.motionnet.parameters(), lr=self.lr, weight_decay=1e-4)
		else:
			optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
		lr_scheduler = optim.lr_scheduler.CyclicLR(
			optimizer, base_lr=1e-6, max_lr=1e-4, step_size_up=2000, cycle_momentum=False)
		return [optimizer], [lr_scheduler]


	def validation_step(self, batch, batch_idx):
		if self.stage == 1:
			outputs, losses = self.model(batch)
		else:
			motion = self.motionnet(batch)
			outputs, losses = self.model(batch, *motion)

		for k,v in losses.items():
			self.log('val_{}'.format(k), v.item(), sync_dist=True)

		self.log("val_loss", losses['loss/0'].item(), sync_dist=True)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument('--id', type=str, default='ppgeo_stage1_log', help='Unique experiment identifier.')
	parser.add_argument('--stage', type=int, default=1, help='stage 1 for depth and pose networks, stage 2 for visual encoder')
	parser.add_argument('--ckpt', type=str, help='stage 1 ckpt')
	parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs.')
	parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
	parser.add_argument('--val_every', type=int, default=3, help='Validation frequency (epochs).')
	parser.add_argument('--batch_size', type=int, default=48, help='Batch size')
	parser.add_argument('--logdir', type=str, default='log', help='Directory to log data to.')

	args = parser.parse_args()
	args.logdir = os.path.join(args.logdir, args.id)

	train_set = YTB_Data(root="data", meta_path = "ytb_meta_train_trip.npy", is_train=True)
	val_set = YTB_Data(root="data", meta_path = "ytb_meta_val_trip.npy", is_train=False)
	print(len(train_set))
	print(len(val_set))

	dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
	dataloader_val = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=True)

	ppgeo = PPGeoEngine(args)

	checkpoint_callback = ModelCheckpoint(save_weights_only=False, mode="min", monitor="val_loss", save_top_k=1, save_last=True,
											dirpath=args.logdir, filename="best_{epoch:02d}-{val_loss:.3f}")
	checkpoint_callback.CHECKPOINT_NAME_LAST = "{epoch}-last"
	trainer = pl.Trainer.from_argparse_args(args,
											default_root_dir=args.logdir,
											gpus = 4,
											accelerator='ddp',
											sync_batchnorm=True,
											plugins=DDPPlugin(find_unused_parameters=True),
											profiler='simple',
											benchmark=True,
											log_every_n_steps=1,
											flush_logs_every_n_steps=5,
											callbacks=[checkpoint_callback,
														],
											check_val_every_n_epoch = 3,
											max_epochs = args.epochs
											)

	trainer.fit(ppgeo, dataloader_train, dataloader_val)




		




