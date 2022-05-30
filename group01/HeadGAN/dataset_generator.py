# coding: utf-8

__author__ = 'cleardusk'

import sys
import argparse
import textwrap
import cv2
import yaml
import numpy as np
from PIL import Image
import os
import datetime
import torch
import torchvision 

sys.path.append(os.path.join(os.path.dirname(
	os.path.realpath(__file__)), "TDDFA_V2/"))

sys.path.append(os.path.join(os.path.dirname(
	os.path.realpath(__file__)), "Retinaface/"))


from TDDFA_V2.FaceBoxes import FaceBoxes
from TDDFA_V2.TDDFA import TDDFA
from TDDFA_V2.utils.render import render
#from TDDFA_V2.utils.render_ctypes import render  # faster
from TDDFA_V2.utils.depth import depth
from TDDFA_V2.utils.pncc import pncc
from TDDFA_V2.utils.uv import uv_tex
from TDDFA_V2.utils.pose import viz_pose
from TDDFA_V2.utils.serialization import ser_to_ply, ser_to_obj
from TDDFA_V2.utils.functions import draw_landmarks, get_suffix
from TDDFA_V2.utils.tddfa_util import str2bool
import sys
import os

config = 'TDDFA_V2/configs/mb1_120x120.yml'
cfg = yaml.load(open(config), Loader=yaml.SafeLoader)
opt = 'pncc'
show_flag = False

def get_xt(img, boxes, param_lst, roi_box_lst, img2, opt, show_flag, tddfa, face_boxes):
	# Detect faces, get 3DMM params and roi boxes
	
	boxes2 = face_boxes(img2)
	
	n = len(boxes2)
	
	if n == 0:
		return None
	
	param_lst2, roi_box_lst2 = tddfa(img2, boxes2)

	new_params = np.concatenate((param_lst2[0][0:12], param_lst[0][12: 52], param_lst[0][52:62])).reshape(1, 62)

	wfp = None
	
	dense_flag = opt in ('2d_dense', '3d', 'depth', 'pncc', 'uv_tex', 'ply', 'obj')
	ver_lst = tddfa.recon_vers(new_params, roi_box_lst, dense_flag=dense_flag)

	# draw_landmarks(img2, ver_lst, show_flag=True, dense_flag=dense_flag, wfp=None)

	pncc_result = pncc(img, ver_lst, tddfa.tri, show_flag=show_flag, wfp=wfp, with_bg_flag=False)
	
	return pncc_result

def get_yt(img, boxes, param_lst, roi_box_lst, opt, show_flag, tddfa):
	wfp = None
	
	dense_flag = opt in ('2d_dense', '3d', 'depth', 'pncc', 'uv_tex', 'ply', 'obj')
	ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)
	pncc_result = pncc(img, ver_lst, tddfa.tri, show_flag=show_flag, wfp=wfp, with_bg_flag=False)
	
	return pncc_result


def process_pair(img, img2s, face_boxes, tddfa):
	# choices=['2d_sparse', '2d_dense', '3d', 'depth', 'pncc', 'uv_tex', 'pose', 'ply', 'obj'])
	
	boxes = face_boxes(img)
	n = len(boxes)
	
	if n == 0:
		return None, None
	
	param_lst, roi_box_lst = tddfa(img, boxes)
	
	yt = get_yt(img, boxes, param_lst, roi_box_lst, opt, show_flag, tddfa)
	xts = []	
	for img2 in img2s:
		if img2 is None:
			return None, None
		new_xt = get_xt(img, boxes, param_lst, roi_box_lst, img2, opt, show_flag, tddfa, face_boxes)
		if img2 is None:
			return None, None
		xts.append(new_xt)
	return yt, xts

# https://stackoverflow.com/questions/44231209/resize-rectangular-image-to-square-keeping-ratio-and-fill-background-with-black
def make_square(im, min_size=256, fill_color=(0, 0, 0, 0)):
	x, y = im.size
	size = max(min_size, x, y)
	new_im = Image.new('RGB', (size, size), fill_color)
	new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
	return new_im.resize((256, 256))

def get_data_batch(paths, frame_per_path, is_gpu, debug_output=False):
	image_array = []
	image_pncc_array = []
	transfer_pncc_array = []
	audio_array = []
	ground_truth_image_array = []

	norm_image_array = []
	norm_image_pncc_array = []
	norm_transfer_pncc_array = []
	norm_audio_array = []
	norm_ground_truth_image_array = []

	if True:
		import os
		os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
		os.environ['OMP_NUM_THREADS'] = '16'
		from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
		from TDDFA_ONNX import TDDFA_ONNX
		face_boxes = FaceBoxes_ONNX()
		tddfa = TDDFA_ONNX('TDDFA_V2/', gpu_mode=is_gpu, **cfg)
	else:
		tddfa = TDDFA(dir='TDDFA_V2/', gpu_mode=is_gpu, **cfg)
		face_boxes = FaceBoxes()

	v_index = 0
	for video_path in paths:
		videocap = cv2.VideoCapture(video_path)
		success,image = videocap.read()
		success = True
		count = 0

		images = []
		timestamps = []
		# transcription_indices = []
		# sound_feature_indices = []
		while success:
			success,image = videocap.read()

			if success:
				image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

				time_stamp = float(videocap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
				if time_stamp == 0.0:
					break
				image = Image.fromarray(image).convert("RGB")
				image = make_square(image)

				# landmarks = landmark_extractor.extract_landmarks(image)

				images.append(np.asarray(image))
				timestamps.append(time_stamp)
				transcription_index = 0
				#for i_d in range(len(decoded_offsets)):
				#	if decoded_offsets[i_d] > time_stamp:
				#		transcription_index = max(i_d - 1, 0)
				#		break
				# transcription_indices.append(transcription_index)

		# print(np.asarray(images).shape)
		# print(np.asarray(timestamps).shape)
		# print(transcription_indices)
		# print(sound_feature_indices)

		for i in range(frame_per_path):
			if len(images) < 8:
				continue
			i_static = np.random.randint(4, len(images) - 3) # L for sound data is 4
			i_dynamic = np.random.randint(4, len(images) - 3) # L for sound data is 4
			im_static = images[i_static]
			im_dynamics = [images[i_dynamic], images[i_dynamic - 1], images[i_dynamic - 2]]
			yt, xts = process_pair(im_static, im_dynamics, face_boxes, tddfa)

			# tr_index = transcription_indices[i_static]
			# sound_feature_index = int((decoded_offsets[tr_index] / decoded_offsets[-1])*audio_features.shape[0])

			# all_sound_features = []
			# print(decoded_output)
			# for i in range(-4, 4):
			# 	tr_move_index = min(max(tr_index + i, 0), decoded_offsets.shape[0])
			# 	# print(decoded_output[tr_move_index])
			# 	char_id = ord('Z') - ord(decoded_output[tr_move_index])
			# 	sf = [0]*27
			# 	sf[char_id] = 1

			# 	all_sound_features += sf
			# all_sound_features = np.concatenate((np.asarray(all_sound_features), audio_features[sound_feature_index]))
			
            
			if yt is not None and xts is not None: 
				transform = torchvision.transforms.Compose([
				torchvision.transforms.ToTensor(),
				torchvision.transforms.Normalize(
					mean=[0.485, 0.456, 0.406],
					std=[0.229, 0.224, 0.225],
				),
				torchvision.transforms.ToPILImage(),
				])
				nomalized_im_static = np.asarray(transform(im_static))
				normalized_ground_truth_image = np.asarray(transform(im_dynamics[0]))

				norm_image_array.append(nomalized_im_static)
				norm_ground_truth_image_array.append(normalized_ground_truth_image)
				norm_image_pncc_array.append(yt)
				norm_transfer_pncc_array.append(xts)
				norm_audio_array.append(np.random.normal(0.0, 1.0, 300))

				image_array.append(im_static)
				ground_truth_image_array.append(im_dynamics[0])
				image_pncc_array.append(yt)
				transfer_pncc_array.append(xts)
				audio_array.append(np.random.normal(0.0, 1.0, 300))

				if debug_output:
					# print(all_sound_features.shape, all_sound_features)	
					cv2.imwrite("test_outputs/" + str(v_index) + "_" + str(i) + "_" + "im_static.png", im_static)
					cv2.imwrite("test_outputs/" + str(v_index) + "_" + str(i) + "_" + "im_dynamic.png", im_dynamics[0])
					cv2.imwrite("test_outputs/" + str(v_index) + "_" + str(i) + "_" + "yt.png", yt)
					cv2.imwrite("test_outputs/" + str(v_index) + "_" + str(i) + "_" + "xt0.png", xts[0])
					cv2.imwrite("test_outputs/" + str(v_index) + "_" + str(i) + "_" + "xt1.png", xts[1])
					cv2.imwrite("test_outputs/" + str(v_index) + "_" + str(i) + "_" + "xt2.png", xts[2])

			v_index += 1
	if debug_output:
		print(len(image_pncc_array))
		print(len(image_array))
		print(len(image_pncc_array))
		print(len(transfer_pncc_array))
		print(len(audio_array))

	return norm_image_array, norm_image_pncc_array, norm_transfer_pncc_array, norm_audio_array, ground_truth_image_array, (image_array, image_pncc_array, transfer_pncc_array, audio_array, ground_truth_image_array)

if __name__ == '__main__':
	image_array, image_pncc_array, transfer_pncc_array, audio_array, ground_truth_image_array, unnormalized = get_data_batch(['TrainingData/vox2_mp4/mp4/id00017/_mjZ87sK6cA/00095.mp4'], 1, is_gpu=True, debug_output=True)

# image_array: statik yüzün resmi (1024 tane np array)
# image_pncc_array: statik yüzün geometri imajı (pncc) (1024 tane np array)
# transfer_pncc_array: dinamik yüzlerin geometri imajları (pncc) (1024 tane list, her bir listin içinde 3 tane np array var)
# audio_array: ses featureları, (1024 tane np array)
# ground_truth_image_array: dinamik yüzün ground truth resmi (1024 tane np array)

