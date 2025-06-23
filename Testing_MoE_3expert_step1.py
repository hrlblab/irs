import argparse
import os, sys
import pandas as pd

#sys.path.append("..")
sys.path.append("/Data4/IRS_github/EfficientSAM_Omni_Swin_Final_3expert_8")
import glob
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import pickle
import cv2
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from scipy.ndimage import morphology
from matplotlib import cm

import skimage

# from unet2D_Dodnet import UNet2D as UNet2D
# from unet2D_Dodnet_ms_scalecontrol import UNet2D as UNet2D_ms_scalecontrol
import os.path as osp
from MOTSDataset_2D_Patch_supervise_csv_512 import MOTSValDataSet as MOTSValDataSet

# from unet2D_Dodnet_ns import UNet2D as UNet2D_ns

import random
import timeit
from tensorboardX import SummaryWriter
import loss_functions.loss_2D as loss

from sklearn import metrics
from math import ceil

from engine import Engine
from apex import amp
from apex.parallel import convert_syncbn_model
#from focalloss import FocalLoss2dff
from sklearn.metrics import f1_score, confusion_matrix

start = timeit.default_timer()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from util_a.image_pool import ImagePool
from efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits

def one_hot_3D(targets,C = 2):
	targets_extend=targets.clone()
	targets_extend.unsqueeze_(1) # convert to Nx1xHxW
	one_hot = torch.cuda.FloatTensor(targets_extend.size(0), C, targets_extend.size(2), targets_extend.size(3)).zero_()
	one_hot.scatter_(1, targets_extend, 1)
	return one_hot


def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

def smooth_segmentation_with_blur(logits, lambda_term=50, sigma=10.0):
	smoothed_mask = cv2.GaussianBlur(logits, (3, 3), 0)
	smoothed_mask[smoothed_mask >= 0.5] = 1.
	smoothed_mask[smoothed_mask != 1.] = 0.

	return smoothed_mask.astype(np.uint8)

def get_arguments(epoch):

	parser = argparse.ArgumentParser(description="DeepLabV3")
	parser.add_argument("--trainset_dir", type=str, default='/Data2/KI_data_train_scale_aug_patch')


	parser.add_argument("--valset_dir", type=str, default='/Data4/IRS_github/data_step1/test/data_list_step1.csv')
	parser.add_argument("--snapshot_dir", type=str, default='snapshots_2D/MoE_3_step1')
	parser.add_argument("--reload_path", type=str, default='snapshots_2D/MoE_3_step1/MOTS_DynConv_MoE_3_step1_e%d.pth' % (epoch))

	parser.add_argument("--best_epoch", type=int, default=epoch)

	# parser.add_argument("--validsetname", type=str, default='scale')
	parser.add_argument("--validsetname", type=str, default='normal')
	#parser.add_argument("--valset_dir", type=str, default='/Data2/Demo_KI_data_train_patch_with_white')
	parser.add_argument("--train_list", type=str, default='list/MOTS/MOTS_train.txt')
	parser.add_argument("--val_list", type=str, default='list/MOTS/xx.txt')
	parser.add_argument("--edge_weight", type=float, default=1.2)
	# parser.add_argument("--snapshot_dir", type=str, default='1027results/fold1_with_white_Unet2D_scaleid3_fullydata_1027')
	parser.add_argument("--reload_from_checkpoint", type=str2bool, default=True)
	parser.add_argument("--input_size", type=str, default='512,512')
	parser.add_argument("--batch_size", type=int, default=4)
	parser.add_argument("--num_gpus", type=int, default=1)
	parser.add_argument('--local_rank', type=int, default=0)
	parser.add_argument("--FP16", type=str2bool, default=False)
	parser.add_argument("--num_epochs", type=int, default=200)
	parser.add_argument("--itrs_each_epoch", type=int, default=250)
	parser.add_argument("--patience", type=int, default=3)
	parser.add_argument("--start_epoch", type=int, default=0)
	parser.add_argument("--val_pred_every", type=int, default=10)
	parser.add_argument("--learning_rate", type=float, default=1e-3)
	parser.add_argument("--num_classes", type=int, default=2)
	parser.add_argument("--num_workers", type=int, default=4)
	parser.add_argument("--weight_std", type=str2bool, default=True)
	parser.add_argument("--momentum", type=float, default=0.9)
	parser.add_argument("--power", type=float, default=0.9)
	parser.add_argument("--weight_decay", type=float, default=0.0005)
	parser.add_argument("--ignore_label", type=int, default=255)
	parser.add_argument("--is_training", action="store_true")
	parser.add_argument("--random_mirror", type=str2bool, default=True)
	parser.add_argument("--random_scale", type=str2bool, default=True)
	parser.add_argument("--random_seed", type=int, default=1234)
	parser.add_argument("--gpu", type=str, default='None')
	return parser


def lr_poly(base_lr, iter, max_iter, power):
	return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter, lr, num_stemps, power):
	"""Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
	lr = lr_poly(lr, i_iter, num_stemps, power)
	optimizer.param_groups[0]['lr'] = lr
	return lr


def surfd(input1, input2, sampling=1, connectivity=1):
	# input_1 = np.atleast_1d(input1.astype(bool))
	# input_2 = np.atleast_1d(input2.astype(bool))

	conn = morphology.generate_binary_structure(input1.ndim, connectivity)

	S = input1 - morphology.binary_erosion(input1, conn)
	Sprime = input2 - morphology.binary_erosion(input2, conn)

	S = np.atleast_1d(S.astype(bool))
	Sprime = np.atleast_1d(Sprime.astype(bool))


	dta = morphology.distance_transform_edt(~S, sampling)
	dtb = morphology.distance_transform_edt(~Sprime, sampling)

	sds = np.concatenate([np.ravel(dta[Sprime != 0]), np.ravel(dtb[S != 0])])

	return np.max(sds), np.mean(sds)

def count_score(preds, labels, rmin, rmax, cmin, cmax):

	Val_F1 = 0
	Val_DICE = 0
	Val_HD = 0
	Val_MSD = 0
	cnt = 0

	for ki in range(len(preds)):
		cnt += 1

		pred = preds[ki,:,rmin:rmax,cmin:cmax]
		label = labels[ki,:,rmin:rmax,cmin:cmax]

		Val_DICE += dice_score(pred, label)
		preds1 = pred[1, ...].flatten().detach().cpu().numpy()
		labels1 = label[1, ...].detach().flatten().detach().cpu().numpy()

		if preds1.sum() == 0 and labels1.sum() == 0:
			Val_HD += 0
			Val_MSD += 0

		else:
			hausdorff, meansurfaceDistance = surfd(preds1, labels1)
			Val_HD += hausdorff
			Val_MSD += meansurfaceDistance

		Val_F1 += f1_score(preds1, labels1, average='macro')

	return Val_F1/cnt, Val_DICE/cnt, Val_HD/cnt, Val_MSD/cnt

def dice_score(preds, labels):  # on GPU
	assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
	predict = preds.contiguous().view(preds.shape[0], -1)
	target = labels.contiguous().view(labels.shape[0], -1)

	num = torch.sum(torch.mul(predict, target), dim=1)
	den = torch.sum(predict, dim=1) + torch.sum(target, dim=1) + 1

	dice = 2 * num / den

	return dice.mean()

def mask_to_box(tensor):
	tensor = tensor.permute([0,2,3,1]).cpu().numpy()
	rmin = np.zeros((4))
	rmax = np.zeros((4))
	cmin = np.zeros((4))
	cmax = np.zeros((4))
	for ki in range(len(tensor)):
		rows = np.any(tensor[ki], axis=1)
		cols = np.any(tensor[ki], axis=0)

		rmin[ki], rmax[ki] = np.where(rows)[0][[0, -1]]
		cmin[ki], cmax[ki] = np.where(cols)[0][[0, -1]]

	# plt.imshow(tensor[0,int(rmin[0]):int(rmax[0]),int(cmin[0]):int(cmax[0]),:])
	return rmin.astype(np.uint32), rmax.astype(np.uint32), cmin.astype(np.uint32), cmax.astype(np.uint32)


def main(epoch):
	"""Create the model and start the training."""
	parser = get_arguments(epoch)
	print(parser)

	with Engine(custom_parser=parser) as engine:
		args = parser.parse_args()
		if args.num_gpus > 1:
			torch.cuda.set_device(args.local_rank)

		writer = SummaryWriter(args.snapshot_dir)

		if not args.gpu == 'None':
			os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

		h, w = map(int, args.input_size.split(','))
		input_size = (h, w)

		cudnn.benchmark = True
		seed = args.random_seed
		if engine.distributed:
			seed = args.local_rank
		torch.manual_seed(seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed(seed)

		# Create model
		criterion = None
		model = build_efficient_sam_vits(task_num=8, scale_num=4)
		check_wo_gpu = 0

		print(f"total params: \t{sum([np.prod(p.size()) for p in model.parameters()])}")

		if not check_wo_gpu:
			device = torch.device('cuda:{}'.format(args.local_rank))
			model.to(device)

		optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)

		if not check_wo_gpu:
			if args.FP16:
				print("Note: Using FP16 during training************")
				model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

			if args.num_gpus > 1:
				model = engine.data_parallel(model)

		# load checkpoint...
		if args.reload_from_checkpoint:
			print('loading from checkpoint: {}'.format(args.reload_path))
			if os.path.exists(args.reload_path):
				if args.FP16:
					checkpoint = torch.load(args.reload_path, map_location=torch.device('cpu'))
					model.load_state_dict(checkpoint['model'])
					optimizer.load_state_dict(checkpoint['optimizer'])
					amp.load_state_dict(checkpoint['amp'])
				else:
					model.load_state_dict(torch.load(args.reload_path, map_location=torch.device('cpu')))
			else:
				print('File not exists in the reload path: {}'.format(args.reload_path))

		if not check_wo_gpu:
			weights = [1., 1.]
			class_weights = torch.FloatTensor(weights).cuda()
			loss_seg_DICE = loss.DiceLoss4MOTS(num_classes=args.num_classes).to(device)
			loss_seg_CE = loss.CELoss4MOTS(num_classes=args.num_classes, ignore_index=255).to(device)
			#criterion1 = nn.CrossEntropyLoss(weight=class_weights).to(device)
			#criterion2 = FocalLoss2d(weight=class_weights).to(device)

		else:
			weights = [1., 1.]
			class_weights = torch.FloatTensor(weights)
			loss_seg_DICE = loss.DiceLoss4MOTS(num_classes=args.num_classes)
			loss_seg_CE = loss.CELoss4MOTS(num_classes=args.num_classes, ignore_index=255)
			#criterion1 = nn.CrossEntropyLoss(weight=class_weights)
			#criterion2 = FocalLoss2d(weight=class_weights)

		if not os.path.exists(args.snapshot_dir):
			os.makedirs(args.snapshot_dir)

		edge_weight = args.edge_weight

		valloader = DataLoader(
			MOTSValDataSet(args.valset_dir, args.val_list, max_iters=args.itrs_each_epoch * args.batch_size,
						   crop_size=input_size, scale=args.random_scale, mirror=args.random_mirror,
						   edge_weight=edge_weight),batch_size=4,shuffle=False,num_workers=8)

		all_tr_loss = []
		all_va_loss = []
		train_loss_MA = None
		val_loss_MA = None

		val_best_loss = 999999
		batch_size = args.batch_size
		task_num = 8
		# for epoch in range(0,args.num_epochs):

		model.load_state_dict(torch.load(args.reload_path, map_location=torch.device('cpu')))

		model.eval()
		# semi_pool_image = ImagePool(8 * 6)
		task0_pool_image = ImagePool(8)
		task0_pool_mask = ImagePool(8)
		task0_scale = []
		task0_layer = []
		task0_name = []

		task1_pool_image = ImagePool(8)
		task1_pool_mask = ImagePool(8)
		task1_scale = []
		task1_layer = []
		task1_name = []

		task2_pool_image = ImagePool(8)
		task2_pool_mask = ImagePool(8)
		task2_scale = []
		task2_layer = []
		task2_name = []

		task3_pool_image = ImagePool(8)
		task3_pool_mask = ImagePool(8)
		task3_scale = []
		task3_layer = []
		task3_name = []

		task4_pool_image = ImagePool(8)
		task4_pool_mask = ImagePool(8)
		task4_scale = []
		task4_layer = []
		task4_name = []

		task5_pool_image = ImagePool(8)
		task5_pool_mask = ImagePool(8)
		task5_scale = []
		task5_layer = []
		task5_name = []

		task6_pool_image = ImagePool(8)
		task6_pool_mask = ImagePool(8)
		task6_scale = []
		task6_layer = []
		task6_name = []

		task7_pool_image = ImagePool(8)
		task7_pool_mask = ImagePool(8)
		task7_scale = []
		task7_layer = []
		task7_name = []

		val_loss = np.zeros((task_num))
		val_F1 = np.zeros((task_num))
		val_Dice = np.zeros((task_num))
		val_HD = np.zeros((task_num))
		val_MSD = np.zeros((task_num))
		cnt = np.zeros((task_num))

		single_df_0 = pd.DataFrame(columns=['name', 'F1', 'Dice', 'HD', 'MSD'])
		single_df_1 = pd.DataFrame(columns=['name', 'F1', 'Dice', 'HD', 'MSD'])
		single_df_2 = pd.DataFrame(columns=['name', 'F1', 'Dice', 'HD', 'MSD'])
		single_df_3 = pd.DataFrame(columns=['name', 'F1', 'Dice', 'HD', 'MSD'])
		single_df_4 = pd.DataFrame(columns=['name', 'F1', 'Dice', 'HD', 'MSD'])
		single_df_5 = pd.DataFrame(columns=['name', 'F1', 'Dice', 'HD', 'MSD'])
		single_df_6 = pd.DataFrame(columns=['name', 'F1', 'Dice', 'HD', 'MSD'])
		single_df_7 = pd.DataFrame(columns=['name', 'F1', 'Dice', 'HD', 'MSD'])

		layer_num = [0,2,6]

		# for iter, batch1, batch2 in enumerate(zip(valloaderloader, semi_valloaderloader)):
		with torch.no_grad():
			for iter, batch1 in enumerate(valloader):


				'dataloader'
				imgs = batch1[0].cuda()
				lbls = batch1[1].cuda()
				wt = batch1[2].cuda().float()
				volumeName = batch1[3]
				l_ids = batch1[4].cuda()
				t_ids = batch1[5].cuda()
				s_ids = batch1[6].cuda()

				# semi_img = batch2[0]

				for ki in range(len(imgs)):
					now_task = layer_num[l_ids[ki]] + t_ids[ki]

					if now_task == 0:
						task0_pool_image.add(imgs[ki].unsqueeze(0))
						task0_pool_mask.add(lbls[ki].unsqueeze(0))
						task0_scale.append((s_ids[ki]))
						task0_layer.append((l_ids[ki]))
						task0_name.append((volumeName[ki]))
					elif now_task == 1:
						task1_pool_image.add(imgs[ki].unsqueeze(0))
						task1_pool_mask.add(lbls[ki].unsqueeze(0))
						task1_scale.append((s_ids[ki]))
						task1_layer.append((l_ids[ki]))
						task1_name.append((volumeName[ki]))
					elif now_task == 2:
						task2_pool_image.add(imgs[ki].unsqueeze(0))
						task2_pool_mask.add(lbls[ki].unsqueeze(0))
						task2_scale.append((s_ids[ki]))
						task2_layer.append((l_ids[ki]))
						task2_name.append((volumeName[ki]))
					elif now_task == 3:
						task3_pool_image.add(imgs[ki].unsqueeze(0))
						task3_pool_mask.add(lbls[ki].unsqueeze(0))
						task3_scale.append((s_ids[ki]))
						task3_layer.append((l_ids[ki]))
						task3_name.append((volumeName[ki]))
					elif now_task == 4:
						task4_pool_image.add(imgs[ki].unsqueeze(0))
						task4_pool_mask.add(lbls[ki].unsqueeze(0))
						task4_scale.append((s_ids[ki]))
						task4_layer.append((l_ids[ki]))
						task4_name.append((volumeName[ki]))
					elif now_task == 5:
						task5_pool_image.add(imgs[ki].unsqueeze(0))
						task5_pool_mask.add(lbls[ki].unsqueeze(0))
						task5_scale.append((s_ids[ki]))
						task5_layer.append((l_ids[ki]))
						task5_name.append((volumeName[ki]))
					elif now_task == 6:
						task6_pool_image.add(imgs[ki].unsqueeze(0))
						task6_pool_mask.add(lbls[ki].unsqueeze(0))
						task6_scale.append((s_ids[ki]))
						task6_layer.append((l_ids[ki]))
						task6_name.append((volumeName[ki]))
					elif now_task == 7:
						task7_pool_image.add(imgs[ki].unsqueeze(0))
						task7_pool_mask.add(lbls[ki].unsqueeze(0))
						task7_scale.append((s_ids[ki]))
						task7_layer.append((l_ids[ki]))
						task7_name.append((volumeName[ki]))

				output_folder = os.path.join(args.snapshot_dir.replace('snapshots_2D/', '/Data4/IRS_github/Testing_'),
											 str(args.best_epoch))

				if not os.path.exists(output_folder):
					os.makedirs(output_folder)
				optimizer.zero_grad()

				'medulla'
				if task0_pool_image.num_imgs >= batch_size:
					images = task0_pool_image.query(batch_size)
					labels = task0_pool_mask.query(batch_size)
					now_task = torch.tensor(0)
					scales = torch.ones(batch_size).cuda()
					layers = torch.ones(batch_size).cuda()
					filename = []
					for bi in range(len(scales)):
						scales[bi] = task0_scale.pop(0)
						layers[bi] = task0_layer.pop(0)
						filename.append(task0_name.pop(0))

					preds = torch.zeros((batch_size, 2, 1024, 1024)).cuda()
					preds[:, :, :512, :512] = model(images[:, :, :512, :512], torch.ones(batch_size).cuda() * now_task,
													   scales)
					preds[:, :, :512, 512:] = model(images[:, :, :512, 512:], torch.ones(batch_size).cuda() * now_task,
													   scales)
					preds[:, :, 512:, 512:] = model(images[:, :, 512:, 512:], torch.ones(batch_size).cuda() * now_task,
													   scales)
					preds[:, :, 512:, :512] = model(images[:, :, 512:, :512], torch.ones(batch_size).cuda() * now_task,
													   scales)

					now_preds = torch.argmax(preds, 1) == 1
					now_preds_onehot = one_hot_3D(now_preds.long())

					rmin, rmax, cmin, cmax = 0, 1024, 0, 1024

					labels_onehot = one_hot_3D(labels.long())

					for pi in range(len(images)):
						prediction = smooth_segmentation_with_blur(preds[pi,1].detach().cpu().numpy(), lambda_term=50, sigma=5.0)
						out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
						img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
						plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
								   img)
						plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
								   labels[pi, ...].detach().cpu().numpy(), cmap = cm.gray)
						plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
								   prediction, cmap = cm.gray)

						F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0), labels_onehot[pi].unsqueeze(0), rmin, rmax, cmin, cmax)
						row = len(single_df_0)
						single_df_0.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]


						val_F1[now_task] += F1
						val_Dice[now_task] += DICE
						val_HD[now_task] += HD
						val_MSD[now_task] += MSD
						cnt[now_task] += 1

				'cortex'
				if task1_pool_image.num_imgs >= batch_size:
					images = task1_pool_image.query(batch_size)
					labels = task1_pool_mask.query(batch_size)
					now_task = torch.tensor(1)
					scales = torch.ones(batch_size).cuda()
					layers = torch.ones(batch_size).cuda()
					filename = []
					for bi in range(len(scales)):
						scales[bi] = task1_scale.pop(0)
						layers[bi] = task1_layer.pop(0)
						filename.append(task1_name.pop(0))

					preds = torch.zeros((batch_size, 2, 1024, 1024)).cuda()
					preds[:, :, :512, :512] = model(images[:, :, :512, :512],
													   torch.ones(batch_size).cuda() * now_task, scales)
					preds[:, :, :512, 512:] = model(images[:, :, :512, 512:],
													   torch.ones(batch_size).cuda() * now_task, scales)
					preds[:, :, 512:, 512:] = model(images[:, :, 512:, 512:],
													   torch.ones(batch_size).cuda() * now_task, scales)
					preds[:, :, 512:, :512] = model(images[:, :, 512:, :512],
													   torch.ones(batch_size).cuda() * now_task, scales)


					now_preds = torch.argmax(preds, 1) == 1
					now_preds_onehot = one_hot_3D(now_preds.long())

					labels_onehot = one_hot_3D(labels.long())
					rmin, rmax, cmin, cmax = 0, 1024, 0, 1024


					for pi in range(len(images)):
						prediction = smooth_segmentation_with_blur(preds[pi,1].detach().cpu().numpy(), lambda_term=50, sigma=5.0)
						num = len(glob.glob(os.path.join(output_folder, '*')))
						out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
						img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
						plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
								   img)
						plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
								   labels[pi, ...].detach().cpu().numpy(), cmap = cm.gray)
						plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
								   prediction, cmap = cm.gray)

						F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0), labels_onehot[pi].unsqueeze(0),
														 rmin, rmax, cmin, cmax)
						row = len(single_df_1)
						single_df_1.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]

						val_F1[now_task] += F1
						val_Dice[now_task] += DICE
						val_HD[now_task] += HD
						val_MSD[now_task] += MSD
						cnt[now_task] += 1

				'dt'
				if task2_pool_image.num_imgs >= batch_size:
					images = task2_pool_image.query(batch_size)
					labels = task2_pool_mask.query(batch_size)
					now_task = torch.tensor(2)
					scales = torch.ones(batch_size).cuda()
					layers = torch.ones(batch_size).cuda()
					filename = []
					for bi in range(len(scales)):
						scales[bi] = task2_scale.pop(0)
						layers[bi] = task2_layer.pop(0)
						filename.append(task2_name.pop(0))

					preds = model(images[:, :, 256:768, 256:768], torch.ones(batch_size).cuda() * now_task, scales)


					now_preds = torch.argmax(preds, 1) == 1
					now_preds_onehot = one_hot_3D(now_preds.long())

					labels_onehot = one_hot_3D(labels[:, 256:768, 256:768].long())
					rmin, rmax, cmin, cmax = 128, 384, 128, 384


					for pi in range(len(images)):
						prediction = smooth_segmentation_with_blur(preds[pi, 1, 128:384, 128:384].detach().cpu().numpy(), lambda_term=50,sigma=5.0)
						num = len(glob.glob(os.path.join(output_folder, '*')))
						out_image = images[pi, :, 384:640, 384:640].permute([1, 2, 0]).detach().cpu().numpy()
						img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
						plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
								   img)
						plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
								   labels[pi, 384:640, 384:640].detach().cpu().numpy(), cmap = cm.gray)
						plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
								   prediction, cmap = cm.gray)

						F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0),
														labels_onehot[pi].unsqueeze(0),
														rmin, rmax, cmin, cmax)

						row = len(single_df_2)
						single_df_2.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]

						val_F1[now_task] += F1
						val_Dice[now_task] += DICE
						val_HD[now_task] += HD
						val_MSD[now_task] += MSD
						cnt[now_task] += 1

				'pt'
				if task3_pool_image.num_imgs >= batch_size:
					images = task3_pool_image.query(batch_size)
					labels = task3_pool_mask.query(batch_size)
					now_task = torch.tensor(3)
					scales = torch.ones(batch_size).cuda()
					layers = torch.ones(batch_size).cuda()
					filename = []
					for bi in range(len(scales)):
						scales[bi] = task3_scale.pop(0)
						layers[bi] = task3_layer.pop(0)
						filename.append(task3_name.pop(0))

					preds = model(images[:, :, 256:768, 256:768], torch.ones(batch_size).cuda() * now_task, scales)


					now_preds = torch.argmax(preds, 1) == 1
					now_preds_onehot = one_hot_3D(now_preds.long())

					labels_onehot = one_hot_3D(labels[:, 256:768, 256:768].long())
					rmin, rmax, cmin, cmax = 128, 384, 128, 384


					for pi in range(len(images)):
						prediction = smooth_segmentation_with_blur(preds[pi, 1, 128:384, 128:384].detach().cpu().numpy(), lambda_term=50,sigma=5.0)
						num = len(glob.glob(os.path.join(output_folder, '*')))
						out_image = images[pi, :, 384:640, 384:640].permute([1, 2, 0]).detach().cpu().numpy()
						img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
						plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
								   img)
						plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
								   labels[pi, 384:640, 384:640].detach().cpu().numpy(), cmap = cm.gray)
						plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
								   prediction, cmap = cm.gray)

						F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0),
														labels_onehot[pi].unsqueeze(0),
														rmin, rmax, cmin, cmax)

						row = len(single_df_3)
						single_df_3.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]

						val_F1[now_task] += F1
						val_Dice[now_task] += DICE
						val_HD[now_task] += HD
						val_MSD[now_task] += MSD
						cnt[now_task] += 1

				'cap'
				if task4_pool_image.num_imgs >= batch_size:
					images = task4_pool_image.query(batch_size)
					labels = task4_pool_mask.query(batch_size)
					now_task = torch.tensor(4)
					scales = torch.ones(batch_size).cuda()
					layers = torch.ones(batch_size).cuda()
					filename = []
					for bi in range(len(scales)):
						scales[bi] = task4_scale.pop(0)
						layers[bi] = task4_layer.pop(0)
						filename.append(task4_name.pop(0))

					preds = model(images[:, :, 256:768, 256:768], torch.ones(batch_size).cuda() * now_task, scales)

					now_preds = torch.argmax(preds, 1) == 1
					now_preds_onehot = one_hot_3D(now_preds.long())

					labels_onehot = one_hot_3D(labels[:, 256:768, 256:768].long())
					rmin, rmax, cmin, cmax = 128, 384, 128, 384


					for pi in range(len(images)):
						prediction = smooth_segmentation_with_blur(preds[pi, 1, 128:384, 128:384].detach().cpu().numpy(), lambda_term=50,sigma=5.0)
						num = len(glob.glob(os.path.join(output_folder, '*')))
						out_image = images[pi, :, 384:640, 384:640].permute([1, 2, 0]).detach().cpu().numpy()
						img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
						plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
								   img)
						plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
								   labels[pi, 384:640, 384:640].detach().cpu().numpy(), cmap = cm.gray)
						plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())), prediction, cmap = cm.gray)

						F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0),
														labels_onehot[pi].unsqueeze(0),
														rmin, rmax, cmin, cmax)

						row = len(single_df_4)
						single_df_4.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]

						val_F1[now_task] += F1
						val_Dice[now_task] += DICE
						val_HD[now_task] += HD
						val_MSD[now_task] += MSD
						cnt[now_task] += 1

				'art'
				if task5_pool_image.num_imgs >= batch_size:
					images = task5_pool_image.query(batch_size)
					labels = task5_pool_mask.query(batch_size)
					now_task = torch.tensor(5)
					scales = torch.ones(batch_size).cuda()
					layers = torch.ones(batch_size).cuda()
					filename = []
					for bi in range(len(scales)):
						scales[bi] = task5_scale.pop(0)
						layers[bi] = task5_layer.pop(0)
						filename.append(task5_name.pop(0))

					preds = model(images[:, :, 256:768, 256:768], torch.ones(batch_size).cuda() * now_task, scales)


					now_preds = torch.argmax(preds, 1) == 1
					now_preds_onehot = one_hot_3D(now_preds.long())

					labels_onehot = one_hot_3D(labels[:, 256:768, 256:768].long())
					rmin, rmax, cmin, cmax = 128, 384, 128, 384


					for pi in range(len(images)):
						prediction = smooth_segmentation_with_blur(preds[pi,1,128:384, 128:384].detach().cpu().numpy(), lambda_term=50,sigma=5.0)
						num = len(glob.glob(os.path.join(output_folder, '*')))
						out_image = images[pi, :, 384:640, 384:640].permute([1, 2, 0]).detach().cpu().numpy()
						img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
						plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
								   img)
						plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
								   labels[pi, 384:640, 384:640].detach().cpu().numpy(), cmap = cm.gray)
						plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
								   prediction, cmap = cm.gray)

						F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0),
														labels_onehot[pi].unsqueeze(0),
														rmin, rmax, cmin, cmax)

						row = len(single_df_5)
						single_df_5.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]

						val_F1[now_task] += F1
						val_Dice[now_task] += DICE
						val_HD[now_task] += HD
						val_MSD[now_task] += MSD
						cnt[now_task] += 1

				'pod'
				if task6_pool_image.num_imgs >= batch_size:
					images = task6_pool_image.query(batch_size)
					labels = task6_pool_mask.query(batch_size)
					now_task = torch.tensor(6)
					scales = torch.ones(batch_size).cuda()
					layers = torch.ones(batch_size).cuda()
					filename = []
					for bi in range(len(scales)):
						scales[bi] = task6_scale.pop(0)
						layers[bi] = task6_layer.pop(0)
						filename.append(task6_name.pop(0))

					preds = model(images[:, :, 256:768, 256:768], torch.ones(batch_size).cuda() * now_task, scales)

					now_preds = torch.argmax(preds, 1) == 1
					now_preds_onehot = one_hot_3D(now_preds.long())

					labels_onehot = one_hot_3D(labels[:, 256:768, 256:768].long())
					rmin, rmax, cmin, cmax = 0, 512, 0, 512

					for pi in range(len(images)):
						prediction = smooth_segmentation_with_blur(preds[pi,1].detach().cpu().numpy(), lambda_term=50, sigma=5.0)
						num = len(glob.glob(os.path.join(output_folder, '*')))
						out_image = images[pi, :, 256:768, 256:768].permute([1, 2, 0]).detach().cpu().numpy()
						img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
						plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
								   img)
						plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
								   labels[pi, 256:768, 256:768].detach().cpu().numpy(), cmap = cm.gray)
						plt.imsave(
							os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
							prediction, cmap = cm.gray)


						F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0),
														labels_onehot[pi].unsqueeze(0),
														rmin, rmax, cmin, cmax)

						row = len(single_df_6)
						single_df_6.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]

						val_F1[now_task] += F1
						val_Dice[now_task] += DICE
						val_HD[now_task] += HD
						val_MSD[now_task] += MSD
						cnt[now_task] += 1

				'mes'
				if task7_pool_image.num_imgs >= batch_size:
					images = task7_pool_image.query(batch_size)
					labels = task7_pool_mask.query(batch_size)
					now_task = torch.tensor(7)
					scales = torch.ones(batch_size).cuda()
					layers = torch.ones(batch_size).cuda()
					filename = []
					for bi in range(len(scales)):
						scales[bi] = task7_scale.pop(0)
						layers[bi] = task7_layer.pop(0)
						filename.append(task7_name.pop(0))

					preds = model(images[:, :, 256:768, 256:768], torch.ones(batch_size).cuda() * now_task, scales)

					now_preds = torch.argmax(preds, 1) == 1
					now_preds_onehot = one_hot_3D(now_preds.long())

					labels_onehot = one_hot_3D(labels[:, 256:768, 256:768].long())
					rmin, rmax, cmin, cmax = 0, 512, 0, 512

					for pi in range(len(images)):
						prediction = smooth_segmentation_with_blur(preds[pi,1].detach().cpu().numpy(), lambda_term=50, sigma=5.0)
						num = len(glob.glob(os.path.join(output_folder, '*')))
						out_image = images[pi, :, 256:768, 256:768].permute([1, 2, 0]).detach().cpu().numpy()
						img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
						plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
								   img)
						plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
								   labels[pi, 256:768, 256:768].detach().cpu().numpy(), cmap=cm.gray)
						plt.imsave(
							os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
							prediction, cmap=cm.gray)

						F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0),
														labels_onehot[pi].unsqueeze(0),
														rmin, rmax, cmin, cmax)

						row = len(single_df_7)
						single_df_7.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]

						val_F1[now_task] += F1
						val_Dice[now_task] += DICE
						val_HD[now_task] += HD
						val_MSD[now_task] += MSD
						cnt[now_task] += 1

			'last round pop'

			'medulla'
			if task0_pool_image.num_imgs > 0:
				batch_size = task0_pool_image.num_imgs
				images = task0_pool_image.query(batch_size)
				labels = task0_pool_mask.query(batch_size)
				now_task = torch.tensor(0)
				scales = torch.ones(batch_size).cuda()
				layers = torch.ones(batch_size).cuda()
				for bi in range(len(scales)):
					scales[bi] = task0_scale.pop(0)
					layers[bi] = task0_layer.pop(0)

				preds = torch.zeros((batch_size, 2, 1024, 1024)).cuda()
				preds[:, :, :512, :512] = model(images[:, :, :512, :512], torch.ones(batch_size).cuda() * now_task,
												scales)
				preds[:, :, :512, 512:] = model(images[:, :, :512, 512:], torch.ones(batch_size).cuda() * now_task,
												scales)
				preds[:, :, 512:, 512:] = model(images[:, :, 512:, 512:], torch.ones(batch_size).cuda() * now_task,
												scales)
				preds[:, :, 512:, :512] = model(images[:, :, 512:, :512], torch.ones(batch_size).cuda() * now_task,
												scales)

				now_preds = torch.argmax(preds, 1) == 1
				now_preds_onehot = one_hot_3D(now_preds.long())

				labels_onehot = one_hot_3D(labels.long())

				rmin, rmax, cmin, cmax = 0, 1024, 0, 1024
				F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin, cmax)

				val_F1[now_task] += F1
				val_Dice[now_task] += DICE
				val_HD[now_task] += HD
				val_MSD[now_task] += MSD
				cnt[now_task] += 1

				for pi in range(len(images)):
					prediction = smooth_segmentation_with_blur(preds[pi,1].detach().cpu().numpy(), lambda_term=50, sigma=5.0)
					num = len(glob.glob(os.path.join(output_folder, '*')))
					out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
					img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
					plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
							   img)
					plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
							   labels[pi, ...].detach().cpu().numpy())
					plt.imsave(os.path.join(output_folder, str(num) + '_preds_%s.png' % (now_task.item())),
							   prediction)

			'cortex'
			if task1_pool_image.num_imgs > 0:
				batch_size = task1_pool_image.num_imgs
				images = task1_pool_image.query(batch_size)
				now_task = torch.tensor(1)
				labels = task1_pool_mask.query(batch_size)
				scales = torch.ones(batch_size).cuda()
				layers = torch.ones(batch_size).cuda()
				for bi in range(len(scales)):
					scales[bi] = task1_scale.pop(0)
					layers[bi] = task1_layer.pop(0)

				preds = torch.zeros((batch_size, 2, 1024, 1024)).cuda()
				preds[:, :, :512, :512] = model(images[:, :, :512, :512],
												torch.ones(batch_size).cuda() * now_task, scales)
				preds[:, :, :512, 512:] = model(images[:, :, :512, 512:],
												torch.ones(batch_size).cuda() * now_task, scales)
				preds[:, :, 512:, 512:] = model(images[:, :, 512:, 512:],
												torch.ones(batch_size).cuda() * now_task, scales)
				preds[:, :, 512:, :512] = model(images[:, :, 512:, :512],
												torch.ones(batch_size).cuda() * now_task, scales)

				now_preds = torch.argmax(preds, 1) == 1
				now_preds_onehot = one_hot_3D(now_preds.long())

				labels_onehot = one_hot_3D(labels.long())
				rmin, rmax, cmin, cmax = 0, 1024, 0, 1024
				F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin, cmax)

				val_F1[now_task] += F1
				val_Dice[now_task] += DICE
				val_HD[now_task] += HD
				val_MSD[now_task] += MSD
				cnt[now_task] += 1

				for pi in range(len(images)):
					prediction = smooth_segmentation_with_blur(preds[pi,1].detach().cpu().numpy(), lambda_term=50, sigma=5.0)
					num = len(glob.glob(os.path.join(output_folder, '*')))
					out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
					img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
					plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
							   img)
					plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
							   labels[pi, ...].detach().cpu().numpy())
					plt.imsave(os.path.join(output_folder, str(num) + '_preds_%s.png' % (now_task.item())),
							   prediction)

			'dt'
			if task2_pool_image.num_imgs > 0:
				batch_size = task2_pool_image.num_imgs
				images = task2_pool_image.query(batch_size)
				labels = task2_pool_mask.query(batch_size)
				scales = torch.ones(batch_size).cuda()
				layers = torch.ones(batch_size).cuda()
				for bi in range(len(scales)):
					scales[bi] = task2_scale.pop(0)
					layers[bi] = task2_layer.pop(0)

				now_task = torch.tensor(2)
				preds = model(images[:, :, 256:768, 256:768], torch.ones(batch_size).cuda() * now_task, scales)

				now_preds = torch.argmax(preds, 1) == 1
				now_preds_onehot = one_hot_3D(now_preds.long())

				labels_onehot = one_hot_3D(labels[:, 256:768, 256:768].long())
				rmin, rmax, cmin, cmax = 128, 384, 128, 384
				F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin, cmax)

				val_F1[now_task] += F1
				val_Dice[now_task] += DICE
				val_HD[now_task] += HD
				val_MSD[now_task] += MSD
				cnt[now_task] += 1

				for pi in range(len(images)):
					prediction = smooth_segmentation_with_blur(preds[pi, 1, 128:384, 128:384].detach().cpu().numpy(), lambda_term=50,sigma=5.0)
					num = len(glob.glob(os.path.join(output_folder, '*')))
					out_image = images[pi, :, 384:640, 384:640].permute([1, 2, 0]).detach().cpu().numpy()
					img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
					plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
							   img)
					plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
							   labels[pi, 384:640, 384:640].detach().cpu().numpy())
					plt.imsave(os.path.join(output_folder, str(num) + '_preds_%s.png' % (now_task.item())),
							   prediction)

			'pt'
			if task3_pool_image.num_imgs > 0:
				batch_size = task3_pool_image.num_imgs
				images = task3_pool_image.query(batch_size)
				labels = task3_pool_mask.query(batch_size)
				scales = torch.ones(batch_size).cuda()
				layers = torch.ones(batch_size).cuda()
				for bi in range(len(scales)):
					scales[bi] = task3_scale.pop(0)
					layers[bi] = task3_layer.pop(0)

				now_task = torch.tensor(3)
				preds = model(images[:, :, 256:768, 256:768], torch.ones(batch_size).cuda() * now_task, scales)

				now_preds = torch.argmax(preds, 1) == 1
				now_preds_onehot = one_hot_3D(now_preds.long())

				labels_onehot = one_hot_3D(labels[:, 256:768, 256:768].long())
				rmin, rmax, cmin, cmax = 128, 384, 128, 384
				F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin, cmax)

				val_F1[now_task] += F1
				val_Dice[now_task] += DICE
				val_HD[now_task] += HD
				val_MSD[now_task] += MSD
				cnt[now_task] += 1

				for pi in range(len(images)):
					prediction = smooth_segmentation_with_blur(preds[pi, 1, 128:384, 128:384].detach().cpu().numpy(), lambda_term=50,sigma=5.0)
					num = len(glob.glob(os.path.join(output_folder, '*')))
					out_image = images[pi, :, 384:640, 384:640].permute([1, 2, 0]).detach().cpu().numpy()
					img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
					plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
							   img)
					plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
							   labels[pi, 384:640, 384:640].detach().cpu().numpy())
					plt.imsave(os.path.join(output_folder, str(num) + '_preds_%s.png' % (now_task.item())),
							   prediction)

			'cap'
			if task4_pool_image.num_imgs > 0:
				batch_size = task4_pool_image.num_imgs
				images = task4_pool_image.query(batch_size)
				labels = task4_pool_mask.query(batch_size)
				scales = torch.ones(batch_size).cuda()
				layers = torch.ones(batch_size).cuda()
				for bi in range(len(scales)):
					scales[bi] = task4_scale.pop(0)
					layers[bi] = task4_layer.pop(0)

				now_task = torch.tensor(4)
				preds = model(images[:, :, 256:768, 256:768], torch.ones(batch_size).cuda() * now_task, scales)

				now_preds = torch.argmax(preds, 1) == 1
				now_preds_onehot = one_hot_3D(now_preds.long())

				labels_onehot = one_hot_3D(labels[:, 256:768, 256:768].long())
				rmin, rmax, cmin, cmax = 128, 384, 128, 384
				F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin, cmax)

				val_F1[now_task] += F1
				val_Dice[now_task] += DICE
				val_HD[now_task] += HD
				val_MSD[now_task] += MSD
				cnt[now_task] += 1

				for pi in range(len(images)):
					prediction = smooth_segmentation_with_blur(preds[pi, 1, 128:384, 128:384].detach().cpu().numpy(), lambda_term=50,sigma=5.0)
					num = len(glob.glob(os.path.join(output_folder, '*')))
					out_image = images[pi, :, 384:640, 384:640].permute([1, 2, 0]).detach().cpu().numpy()
					img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
					plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
							   img)
					plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
							   labels[pi, 384:640, 384:640].detach().cpu().numpy())
					plt.imsave(os.path.join(output_folder, str(num) + '_preds_%s.png' % (now_task.item())),
							   prediction)

			'art'
			if task5_pool_image.num_imgs > 0:
				batch_size = task5_pool_image.num_imgs
				images = task5_pool_image.query(batch_size)
				labels = task5_pool_mask.query(batch_size)
				scales = torch.ones(batch_size).cuda()
				layers = torch.ones(batch_size).cuda()
				for bi in range(len(scales)):
					scales[bi] = task5_scale.pop(0)
					layers[bi] = task5_layer.pop(0)

				now_task = torch.tensor(5)
				preds = model(images[:, :, 256:768, 256:768], torch.ones(batch_size).cuda() * now_task, scales)

				now_preds = torch.argmax(preds, 1) == 1
				now_preds_onehot = one_hot_3D(now_preds.long())

				labels_onehot = one_hot_3D(labels[:, 256:768, 256:768].long())
				rmin, rmax, cmin, cmax = 128, 384, 128, 384
				F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin, cmax)

				val_F1[now_task] += F1
				val_Dice[now_task] += DICE
				val_HD[now_task] += HD
				val_MSD[now_task] += MSD
				cnt[now_task] += 1

				for pi in range(len(images)):
					prediction = smooth_segmentation_with_blur(preds[pi, 1, 128:384, 128:384].detach().cpu().numpy(), lambda_term=50,sigma=5.0)
					num = len(glob.glob(os.path.join(output_folder, '*')))
					out_image = images[pi, :, 384:640, 384:640].permute([1, 2, 0]).detach().cpu().numpy()
					img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
					plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
							   img)
					plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
							   labels[pi, 384:640, 384:640].detach().cpu().numpy())
					plt.imsave(os.path.join(output_folder, str(num) + '_preds_%s.png' % (now_task.item())),
							   prediction)

			'pod'
			if task6_pool_image.num_imgs > 0:
				batch_size = task6_pool_image.num_imgs
				labels = task6_pool_mask.query(batch_size)
				scales = torch.ones(batch_size).cuda()
				layers = torch.ones(batch_size).cuda()
				for bi in range(len(scales)):
					scales[bi] = task6_scale.pop(0)
					layers[bi] = task6_layer.pop(0)

				now_task = torch.tensor(6)
				preds = model(images[:, :, 256:768, 256:768], torch.ones(batch_size).cuda() * now_task, scales)

				now_preds = torch.argmax(preds, 1) == 1
				now_preds_onehot = one_hot_3D(now_preds.long())

				labels_onehot = one_hot_3D(labels[:, 256:768, 256:768].long())
				rmin, rmax, cmin, cmax = 0, 512, 0, 512
				F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin,
												 cmax)

				val_F1[now_task] += F1
				val_Dice[now_task] += DICE
				val_HD[now_task] += HD
				val_MSD[now_task] += MSD
				cnt[now_task] += 1
				for pi in range(len(images)):
					prediction = smooth_segmentation_with_blur(preds[pi, 1].detach().cpu().numpy(), lambda_term=50,sigma=5.0)
					num = len(glob.glob(os.path.join(output_folder, '*')))
					out_image = images[pi, :, 256:768, 256:768].permute([1, 2, 0]).detach().cpu().numpy()
					img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
					plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
							   img)
					plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
							   labels[pi, 256:768, 256:768].detach().cpu().numpy())
					plt.imsave(
						os.path.join(output_folder, str(num) + '_preds_%s.png' % (now_task.item())),
						prediction)

			'mes'
			if task7_pool_image.num_imgs > 0:
				batch_size = task7_pool_image.num_imgs
				images = task7_pool_image.query(batch_size)
				labels = task7_pool_mask.query(batch_size)
				scales = torch.ones(batch_size).cuda()
				layers = torch.ones(batch_size).cuda()
				for bi in range(len(scales)):
					scales[bi] = task7_scale.pop(0)
					layers[bi] = task7_layer.pop(0)

				now_task = torch.tensor(7)
				preds = model(images[:, :, 256:768, 256:768], torch.ones(batch_size).cuda() * now_task, scales)

				now_preds = torch.argmax(preds, 1) == 1
				now_preds_onehot = one_hot_3D(now_preds.long())

				labels_onehot = one_hot_3D(labels[:, 256:768, 256:768].long())
				rmin, rmax, cmin, cmax = 0, 512, 0, 512
				F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin,
												 cmax)

				val_F1[now_task] += F1
				val_Dice[now_task] += DICE
				val_HD[now_task] += HD
				val_MSD[now_task] += MSD
				cnt[now_task] += 1

				for pi in range(len(images)):
					prediction = smooth_segmentation_with_blur(preds[pi, 1].detach().cpu().numpy(), lambda_term=50,sigma=5.0)
					num = len(glob.glob(os.path.join(output_folder, '*')))
					out_image = images[pi, :, 256:768, 256:768].permute([1, 2, 0]).detach().cpu().numpy()
					img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
					plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
							   img)
					plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
							   labels[pi, 256:768, 256:768].detach().cpu().numpy())
					plt.imsave(
						os.path.join(output_folder, str(num) + '_preds_%s.png' % (now_task.item())),
						prediction)


			avg_val_F1 = val_F1 / cnt
			avg_val_Dice = val_Dice / cnt
			avg_val_HD = val_HD / cnt
			avg_val_MSD = val_MSD / cnt

			print('Validate \n 0medulla_f1={:.4} dsc={:.4} hd={:.4} msd={:.4}'
				  ' \n 1cortex_f1={:.4} dsc={:.4} hd={:.4} msd={:.4}\n'
				  ' \n 2dt_f1={:.4} dsc={:.4} hd={:.4} msd={:.4}\n'
				  ' \n 3pt_f1={:.4} dsc={:.4} hd={:.4} msd={:.4}\n'
				  ' \n 4cap_f1={:.4} dsc={:.4} hd={:.4} msd={:.4}\n'
				  ' \n 5art_f1={:.4} dsc={:.4} hd={:.4} msd={:.4}\n'
				  ' \n 6pod_f1={:.4} dsc={:.4} hd={:.4} msd={:.4}\n'
				  ' \n 7mes_f1={:.4} dsc={:.4} hd={:.4} msd={:.4}\n'
				  .format(avg_val_F1[0].item(), avg_val_Dice[0].item(), avg_val_HD[0].item(), avg_val_MSD[0].item(),
						  avg_val_F1[1].item(), avg_val_Dice[1].item(), avg_val_HD[1].item(), avg_val_MSD[1].item(),
						  avg_val_F1[2].item(), avg_val_Dice[2].item(), avg_val_HD[2].item(), avg_val_MSD[2].item(),
						  avg_val_F1[3].item(), avg_val_Dice[3].item(), avg_val_HD[3].item(), avg_val_MSD[3].item(),
						  avg_val_F1[4].item(), avg_val_Dice[4].item(), avg_val_HD[4].item(), avg_val_MSD[4].item(),
						  avg_val_F1[5].item(), avg_val_Dice[5].item(), avg_val_HD[5].item(), avg_val_MSD[5].item(),
						  avg_val_F1[6].item(), avg_val_Dice[6].item(), avg_val_HD[6].item(), avg_val_MSD[6].item(),
						  avg_val_F1[7].item(), avg_val_Dice[7].item(), avg_val_HD[7].item(), avg_val_MSD[7].item()))

		df = pd.DataFrame(columns=['task', 'F1', 'Dice', 'HD', 'MSD'])
		df.loc[0] = ['0medulla', avg_val_F1[0].item(), avg_val_Dice[0].item(), avg_val_HD[0].item(), avg_val_MSD[0].item()]
		df.loc[1] = ['1cortex', avg_val_F1[1].item(), avg_val_Dice[1].item(), avg_val_HD[1].item(),avg_val_MSD[1].item()]
		df.loc[2] = ['2dt', avg_val_F1[2].item(), avg_val_Dice[2].item(), avg_val_HD[2].item(),avg_val_MSD[2].item()]
		df.loc[3] = ['3pt', avg_val_F1[3].item(), avg_val_Dice[3].item(), avg_val_HD[3].item(),avg_val_MSD[3].item()]
		df.loc[4] = ['4cap', avg_val_F1[4].item(), avg_val_Dice[4].item(), avg_val_HD[4].item(),avg_val_MSD[4].item()]
		df.loc[5] = ['5art', avg_val_F1[5].item(), avg_val_Dice[5].item(), avg_val_HD[5].item(),avg_val_MSD[5].item()]
		df.loc[6] = ['6pod', avg_val_F1[6].item(), avg_val_Dice[6].item(), avg_val_HD[6].item(),avg_val_MSD[6].item()]
		df.loc[7] = ['7mes', avg_val_F1[7].item(), avg_val_Dice[7].item(), avg_val_HD[7].item(),avg_val_MSD[7].item()]

		df.to_csv(os.path.join(output_folder, 'testing_result.csv'))
		single_df_0.to_csv(os.path.join(output_folder,'testing_result_0.csv'))
		single_df_1.to_csv(os.path.join(output_folder,'testing_result_1.csv'))
		single_df_2.to_csv(os.path.join(output_folder,'testing_result_2.csv'))
		single_df_3.to_csv(os.path.join(output_folder,'testing_result_3.csv'))
		single_df_4.to_csv(os.path.join(output_folder,'testing_result_4.csv'))
		single_df_5.to_csv(os.path.join(output_folder,'testing_result_5.csv'))
		single_df_6.to_csv(os.path.join(output_folder,'testing_result_6.csv'))
		single_df_7.to_csv(os.path.join(output_folder,'testing_result_7.csv'))

	end = timeit.default_timer()
	print(end - start, 'seconds')


if __name__ == '__main__':
	main(100)
