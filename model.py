import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.resnet_encoder import ResnetEncoder
from networks.depth_decoder import DepthDecoder
from networks.pose_decoder import PoseDecoder
from layers import BackprojectDepth, Project3D, get_smooth_loss, transformation_from_parameters, disp_to_depth
from loss import SSIM


class MotionNet(nn.Module):
	def __init__(self):
		super(MotionNet, self).__init__()
		self.visual_encoder = ResnetEncoder(34, True, num_input_images=1)
		self.motion_decoder = PoseDecoder(self.visual_encoder.num_ch_enc, num_input_features=1, num_frames_to_predict_for=2)
	def forward(self, inputs):
		motion_inputs1 = inputs["color_aug", -1, 0]
		motion_inputs2 = inputs["color_aug", 0, 0]
		enc1 = self.visual_encoder(motion_inputs1, normalize=True)
		enc2 = self.visual_encoder(motion_inputs2, normalize=True)

		axisangle1, translation1 =  self.motion_decoder([enc1])
		axisangle2, translation2 =  self.motion_decoder([enc2])
		return axisangle1, translation1, axisangle2, translation2

class Monodepth(nn.Module):
	def __init__(self, stage = 1, batch_size=1):
		super(Monodepth, self).__init__()
		self.stage = stage
		self.num_scales = len([0, 1, 2, 3])
		self.scales = [0, 1, 2, 3]
		self.frame_ids = [0, -1, 1]
		self.height = 160
		self.width = 320
		self.num_input_frames = len([0, -1, 1])
		self.num_pose_frames = 2 

		self.min_depth = 0.1
		self.max_depth = 100.0

		self.depth_encoder = ResnetEncoder(18, True)
		self.depth_decoder = DepthDecoder(self.depth_encoder.num_ch_enc, self.scales)

		self.pose_encoder = ResnetEncoder(18, True, num_input_images=self.num_pose_frames)
		self.pose_decoder = PoseDecoder(self.pose_encoder.num_ch_enc, num_input_features=1, num_frames_to_predict_for=2)


		self.fl = nn.Sequential(nn.Linear(512, 256),
								nn.ReLU(True),
								nn.Linear(256, 2),
								nn.Softplus()
								)

		self.offset = nn.Sequential(nn.Linear(512, 256),
									nn.ReLU(True),
									nn.Linear(256, 2),
									nn.Sigmoid()
								)

		self.avg_pooling = nn.AdaptiveAvgPool2d((1,1))

		self.backproject_depth = {}
		self.project_3d = {}
		for scale in self.scales:
			h = self.height // (2 ** scale)
			w = self.width // (2 ** scale)

			self.backproject_depth[scale] = BackprojectDepth(batch_size, h, w)

			self.project_3d[scale] = Project3D(batch_size, h, w)

		self.ssim = SSIM()

		self.initialized = False
	
	def initialize(self):
		for scale in self.scales:
			self.backproject_depth[scale] = self.backproject_depth[scale].to(self.device)

			self.project_3d[scale] = self.project_3d[scale].to(self.device)

			self.initialized = True

	def forward_stage1(self, inputs):
		if not self.initialized:
			self.device = inputs["color_aug", 0, 0].device
			self.initialize()

		# depth prediction
		features = self.depth_encoder(inputs["color_aug", 0, 0], normalize=True)
		outputs = self.depth_decoder(features)

		pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.frame_ids}
		poses_inputs1 = torch.cat([pose_feats[-1], pose_feats[0]], 1)
		poses_inputs2 = torch.cat([pose_feats[0], pose_feats[1]], 1)
		pose_enc1 = self.pose_encoder(poses_inputs1, normalize=True)
		pose_enc2 = self.pose_encoder(poses_inputs2, normalize=True)

		# intrinsics prediction
		feature_pooled1 = torch.flatten(self.avg_pooling(pose_enc1[-1]), 1)
		fl1 = self.fl(feature_pooled1)
		offsets1 = self.offset(feature_pooled1)
		feature_pooled2 = torch.flatten(self.avg_pooling(pose_enc2[-1]), 1)
		fl2 = self.fl(feature_pooled2)
		offsets2 = self.offset(feature_pooled2)

		K1 = self.compute_K(fl1, offsets1)
		K2 = self.compute_K(fl2, offsets2)
		K = (K1+K2)/2
		inputs = self.add_K(K, inputs)

		# pose prediction
		axisangle1, translation1 =  self.pose_decoder([pose_enc1])
		axisangle2, translation2 =  self.pose_decoder([pose_enc2])

		outputs[("axisangle", 0, -1)] = axisangle1
		outputs[("translation", 0, -1)] = translation1
		outputs[("axisangle", 0, 1)] = axisangle2
		outputs[("translation", 0, 1)] = translation2

		outputs[("cam_T_cam", 0, -1)] = transformation_from_parameters(
						axisangle1[:, 0], translation1[:, 0], invert=True)
		outputs[("cam_T_cam", 0, 1)] = transformation_from_parameters(
						axisangle2[:, 0], translation2[:, 0], invert=False)

		for scale in self.scales:
			disp = outputs[("disp", scale)]
			disp = F.interpolate(
					disp, [self.height, self.width], mode="bilinear", align_corners=False)
			source_scale = 0

			_, depth = disp_to_depth(disp, self.min_depth, self.max_depth)

			outputs[("depth", 0, scale)] = depth

			for i, frame_id in enumerate(self.frame_ids[1:]):
				T = outputs[("cam_T_cam", 0, frame_id)]

				cam_points = self.backproject_depth[source_scale](
					depth, inputs[("inv_K", source_scale)])
				pix_coords = self.project_3d[source_scale](
					cam_points, inputs[("K", source_scale)], T)

				outputs[("sample", frame_id, scale)] = pix_coords

				outputs[("color", frame_id, scale)] = F.grid_sample(
					inputs[("color", frame_id, source_scale)],
					outputs[("sample", frame_id, scale)],
					padding_mode="border")

				outputs[("color_identity", frame_id, scale)] = \
						inputs[("color", frame_id, source_scale)]

		losses = {}
		total_loss = 0

		for scale in self.scales:
			loss = 0
			reprojection_losses = []

			source_scale = 0

			disp = outputs[("disp", scale)]
			color = inputs[("color", 0, scale)]
			target = inputs[("color", 0, source_scale)]


			for frame_id in self.frame_ids[1:]:
				pred = outputs[("color", frame_id, scale)]
				reprojection_losses.append(self.compute_reprojection_loss(pred, target))

			reprojection_losses = torch.cat(reprojection_losses, 1)


			identity_reprojection_losses = []
			for frame_id in self.frame_ids[1:]:
				pred = inputs[("color", frame_id, source_scale)]
				identity_reprojection_losses.append(
					self.compute_reprojection_loss(pred, target))

			identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

			
			identity_reprojection_loss = identity_reprojection_losses

			reprojection_loss = reprojection_losses


			identity_reprojection_loss += torch.randn(
					identity_reprojection_loss.shape, device=self.device) * 0.00001

			combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)

			if combined.shape[1] == 1:
				to_optimise = combined
			else:
				to_optimise, idxs = torch.min(combined, dim=1)

			outputs["identity_selection/{}".format(scale)] = (
					idxs > identity_reprojection_loss.shape[1] - 1).float()

			loss += to_optimise.mean()

			mean_disp = disp.mean(2, True).mean(3, True)
			norm_disp = disp / (mean_disp + 1e-7)
			smooth_loss = get_smooth_loss(norm_disp, color)

			loss += 1e-3 * smooth_loss / (2 ** scale)
			total_loss += loss
			losses["loss/{}".format(scale)] = loss

		total_loss /= self.num_scales
		losses["loss"] = total_loss
		return outputs, losses
		    

	def forward_stage2(self, inputs, axisangle1, translation1, axisangle2, translation2):
		if not self.initialized:
			self.device = inputs["color_aug", 0, 0].device
			self.initialize()
		with torch.no_grad():
			features = self.depth_encoder(inputs["color", 0, 0], normalize=True)
			outputs = self.depth_decoder(features)

			pose_feats = {f_i: inputs["color", f_i, 0] for f_i in self.frame_ids}
			poses_inputs1 = torch.cat([pose_feats[-1], pose_feats[0]], 1)
			poses_inputs2 = torch.cat([pose_feats[0], pose_feats[1]], 1)
			pose_enc1 = self.pose_encoder(poses_inputs1, normalize=True)
			pose_enc2 = self.pose_encoder(poses_inputs2, normalize=True)

			feature_pooled1 = torch.flatten(self.avg_pooling(pose_enc1[-1]), 1)
			fl1 = self.fl(feature_pooled1)
			offsets1 = self.offset(feature_pooled1)
			feature_pooled2 = torch.flatten(self.avg_pooling(pose_enc2[-1]), 1)
			fl2 = self.fl(feature_pooled2)
			offsets2 = self.offset(feature_pooled2)

			K1 = self.compute_K(fl1, offsets1)
			K2 = self.compute_K(fl2, offsets2)
			K = (K1+K2)/2
			inputs = self.add_K(K, inputs)

		outputs[("axisangle", 0, -1)] = axisangle1
		outputs[("translation", 0, -1)] = translation1
		outputs[("axisangle", 0, 1)] = axisangle2
		outputs[("translation", 0, 1)] = translation2

		outputs[("cam_T_cam", 0, -1)] = transformation_from_parameters(
						axisangle1[:, 0], translation1[:, 0], invert=True)
		outputs[("cam_T_cam", 0, 1)] = transformation_from_parameters(
						axisangle2[:, 0], translation2[:, 0], invert=False)

		for scale in [0]:
			with torch.no_grad():
				disp = outputs[("disp", scale)]
				disp = F.interpolate(
						disp, [self.height, self.width], mode="bilinear", align_corners=False)
				source_scale = 0

				_, depth = disp_to_depth(disp, self.min_depth, self.max_depth)

				outputs[("depth", 0, scale)] = depth

			for i, frame_id in enumerate(self.frame_ids[1:]):
				T = outputs[("cam_T_cam", 0, frame_id)]

				cam_points = self.backproject_depth[source_scale](
					depth, inputs[("inv_K", source_scale)])
				pix_coords = self.project_3d[source_scale](
					cam_points, inputs[("K", source_scale)], T)

				outputs[("sample", frame_id, scale)] = pix_coords

				outputs[("color", frame_id, scale)] = F.grid_sample(
					inputs[("color", frame_id, source_scale)],
					outputs[("sample", frame_id, scale)],
					padding_mode="border")

				outputs[("color_identity", frame_id, scale)] = \
						inputs[("color", frame_id, source_scale)]

		losses = {}
		total_loss = 0

		for scale in [0]:
			loss = 0
			reprojection_losses = []

			source_scale = 0

			disp = outputs[("disp", scale)]
			color = inputs[("color", 0, scale)]
			target = inputs[("color", 0, source_scale)]


			for frame_id in self.frame_ids[1:]:
				pred = outputs[("color", frame_id, scale)]
				reprojection_losses.append(self.compute_reprojection_loss(pred, target))

			reprojection_losses = torch.cat(reprojection_losses, 1)


			identity_reprojection_losses = []
			for frame_id in self.frame_ids[1:]:
				pred = inputs[("color", frame_id, source_scale)]
				identity_reprojection_losses.append(
					self.compute_reprojection_loss(pred, target))

			identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

			
			identity_reprojection_loss = identity_reprojection_losses

			reprojection_loss = reprojection_losses


			identity_reprojection_loss += torch.randn(
					identity_reprojection_loss.shape, device=self.device) * 0.00001

			combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)

			if combined.shape[1] == 1:
				to_optimise = combined
			else:
				to_optimise, idxs = torch.min(combined, dim=1)

			outputs["identity_selection/{}".format(scale)] = (
					idxs > identity_reprojection_loss.shape[1] - 1).float()

			loss += to_optimise.mean()

			mean_disp = disp.mean(2, True).mean(3, True)
			norm_disp = disp / (mean_disp + 1e-7)
			smooth_loss = get_smooth_loss(norm_disp, color)

			loss += 1e-3 * smooth_loss / (2 ** scale)
			total_loss += loss
			losses["loss/{}".format(scale)] = loss

		total_loss /= self.num_scales
		losses["loss"] = total_loss
		return outputs, losses

	def forward(self, inputs, axisangle1=None, translation1=None, axisangle2=None, translation2=None):
		if self.stage == 1:
			return self.forward_stage1(inputs)
		else:
			return self.forward_stage2(inputs, axisangle1, translation1, axisangle2, translation2)

	def compute_reprojection_loss(self, pred, target):
		"""Computes reprojection loss between a batch of predicted and target images
		"""
		abs_diff = torch.abs(target - pred)
		l1_loss = abs_diff.mean(1, True)

		ssim_loss = self.ssim(pred, target).mean(1, True)
		reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

		return reprojection_loss

	def compute_K(self, fl, offsets):
		B = fl.shape[0]

		fl = torch.diag_embed(fl) # B * 2 * 2

		K = torch.cat([fl, offsets.view(-1, 2, 1)], 2) # B * 2 * 3
		row = torch.tensor([[0, 0, 1], [0, 0, 0]]).view(1, 2, 3).repeat(B, 1, 1).type_as(K)
		K = torch.cat([K, row], 1) # B * 4 * 3
		col = torch.tensor([0, 0, 0, 1]).view(1, 4, 1).repeat(B, 1, 1).type_as(K)
		K = torch.cat([K, col], 2) # B * 4 * 4

		return K


	def add_K(self, K, inputs):
		for scale in self.scales:
			K_scale = K.clone()
			K_scale[:, 0] *= self.width // (2 ** scale)
			K_scale[:, 1] *= self.height // (2 ** scale)
			inv_K_scale = torch.linalg.pinv(K_scale)
			inputs[("K", scale)] = K_scale
			inputs[("inv_K", scale)] = inv_K_scale
			return inputs