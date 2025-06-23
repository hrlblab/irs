import argparse
import os, sys
import pandas as pd

#sys.path.append("..")
sys.path.append("/Data4/IRS_github/EfficientSAM_Omni_Swin_Final_3expert_24_from8")
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

import os.path as osp
from MOTSDataset_2D_Patch_supervise_csv_512 import MOTSValDataSet as MOTSValDataSet


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


    parser.add_argument("--valset_dir", type=str, default='/Data4/IRS_github/data_step2/test/data_list_step2.csv')
    parser.add_argument("--snapshot_dir", type=str, default='snapshots_2D/MoE_3_step2/')
    parser.add_argument("--reload_path", type=str, default='snapshots_2D/MoE_3_step2/MOTS_DynConv_MoE_3_step2_e%d.pth' % (epoch))

    parser.add_argument("--best_epoch", type=int, default=epoch)

    # parser.add_argument("--validsetname", type=str, default='scale')
    parser.add_argument("--validsetname", type=str, default='normal')
    #parser.add_argument("--valset_dir", type=str, default='/Data2/Demo_KI_data_train_patch_with_white')
    parser.add_argument("--train_list", type=str, default='list/MOTS/MOTS_train.txt')
    parser.add_argument("--val_list", type=str, default='list/MOTS/xx.txt')
    parser.add_argument("--edge_weight", type=float, default=1.2)

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
        # if engine.distributed:
        #     seed = args.local_rank
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Create model
        criterion = None
        task_num = 24
        model = build_efficient_sam_vits(task_num=task_num, scale_num=4)
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

            # if args.num_gpus > 1:
            #     model = engine.data_parallel(model)

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

        model.load_state_dict(torch.load(args.reload_path, map_location=torch.device('cpu')))

        model.eval()
        # semi_pool_image = ImagePool(8 * 6)

        layer_num = [0, 2, 9, 15]
        for i in range(task_num):
            globals()[f'task{i}_pool_image'] = ImagePool(8)
            globals()[f'task{i}_pool_mask'] = ImagePool(8)
            globals()[f'task{i}_pool_weight'] = ImagePool(8)
            globals()[f'task{i}_scale'] = []
            globals()[f'task{i}_layer'] = []
            globals()[f'task{i}_filename'] = []
            globals()[f'task{i}_single_df'] = pd.DataFrame(columns=['name', 'F1', 'Dice', 'HD', 'MSD'])


        val_loss = np.zeros((task_num))
        val_F1 = np.zeros((task_num))
        val_Dice = np.zeros((task_num))
        val_TPR = np.zeros((task_num))
        val_PPV = np.zeros((task_num))
        cnt = np.zeros((task_num))

        size_1024 = [0, 1]
        size_256 = [2, 3, 4, 5, 6, 7]
        size_512 = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

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
                    # Dynamically access the corresponding pools and lists using `globals()`
                    globals()[f'task{now_task}_pool_image'].add(imgs[ki].unsqueeze(0))
                    globals()[f'task{now_task}_pool_mask'].add(lbls[ki].unsqueeze(0))
                    globals()[f'task{now_task}_pool_weight'].add(wt[ki].unsqueeze(0))
                    globals()[f'task{now_task}_scale'].append(s_ids[ki])
                    globals()[f'task{now_task}_layer'].append(l_ids[ki])
                    globals()[f'task{now_task}_filename'].append(volumeName[ki])

                output_folder = os.path.join(
                    args.snapshot_dir.replace('snapshots_2D/', '/Data4/IRS_github/testing_'),
                    str(epoch))

                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                optimizer.zero_grad()

                for now_task in range(task_num):  # Loop through tasks 0 to 22
                    task_pool_image = globals()[f'task{now_task}_pool_image']
                    task_pool_mask = globals()[f'task{now_task}_pool_mask']
                    task_pool_weight = globals()[f'task{now_task}_pool_weight']
                    task_scale = globals()[f'task{now_task}_scale']
                    task_layer = globals()[f'task{now_task}_layer']
                    task_filename = globals()[f'task{now_task}_filename']
                    task_single_df = globals()[f'task{now_task}_single_df']

                    if task_pool_image.num_imgs >= batch_size:
                        if now_task in size_1024:
                            images = task_pool_image.query(batch_size)
                            labels = task_pool_mask.query(batch_size)
                            now_task = torch.tensor(now_task)
                            scales = torch.ones(batch_size).cuda()
                            layers = torch.ones(batch_size).cuda()
                            filename = []
                            for bi in range(len(scales)):
                                scales[bi] = task_scale.pop(0)
                                layers[bi] = task_layer.pop(0)
                                filename.append(task_filename.pop(0))

                            preds = torch.zeros((batch_size, 2, 1024, 1024)).cuda()
                            preds[:, :, :512, :512], _, _, _ = model(images[:, :, :512, :512],
                                                                  torch.ones(batch_size).cuda() * now_task, scales)
                            preds[:, :, :512, 512:], _, _, _ = model(images[:, :, :512, 512:],
                                                                  torch.ones(batch_size).cuda() * now_task, scales)
                            preds[:, :, 512:, 512:], _, _, _ = model(images[:, :, 512:, 512:],
                                                                  torch.ones(batch_size).cuda() * now_task, scales)
                            preds[:, :, 512:, :512], _, _, _ = model(images[:, :, 512:, :512],
                                                                  torch.ones(batch_size).cuda() * now_task, scales)

                            now_preds = torch.argmax(preds, 1) == 1
                            now_preds_onehot = one_hot_3D(now_preds.long())

                            labels_onehot = one_hot_3D(labels.long())

                            rmin, rmax, cmin, cmax = 0, 1024, 0, 1024
                            F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin, cmax)

                            val_F1[now_task] += F1
                            val_Dice[now_task] += DICE
                            val_TPR[now_task] += TPR
                            val_PPV[now_task] += PPV
                            cnt[now_task] += 1

                            for pi in range(len(images)):
                                prediction = now_preds[pi]
                                num = len(glob.glob(os.path.join(output_folder, '*')))
                                out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
                                img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                                plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'), img)
                                plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                                           labels[pi, ...].detach().cpu().numpy(), cmap = cm.gray)
                                plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
                                           prediction.detach().cpu().numpy(), cmap = cm.gray)

                                F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0),labels_onehot[pi].unsqueeze(0), rmin, rmax, cmin, cmax)
                                row = len(task_single_df)
                                task_single_df.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]

                        elif now_task in size_256:
                            images = task_pool_image.query(batch_size)
                            labels = task_pool_mask.query(batch_size)
                            scales = torch.ones(batch_size).cuda()
                            layers = torch.ones(batch_size).cuda()
                            now_task = torch.tensor(now_task)
                            filename = []
                            for bi in range(len(scales)):
                                scales[bi] = task_scale.pop(0)
                                layers[bi] = task_layer.pop(0)
                                filename.append(task_filename.pop(0))

                            preds, _, _, _ = model(images[:, :, 256:768, 256:768],
                                                torch.ones(batch_size).cuda() * now_task, scales)

                            now_preds = torch.argmax(preds, 1) == 1
                            now_preds_onehot = one_hot_3D(now_preds.long())

                            labels_onehot = one_hot_3D(labels[:, 256:768, 256:768].long())
                            rmin, rmax, cmin, cmax = 128, 384, 128, 384
                            F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin, cmax)

                            val_F1[now_task] += F1
                            val_Dice[now_task] += DICE
                            val_TPR[now_task] += TPR
                            val_PPV[now_task] += PPV
                            cnt[now_task] += 1

                            for pi in range(len(images)):
                                prediction = now_preds[pi, 128:384, 128:384]
                                num = len(glob.glob(os.path.join(output_folder, '*')))
                                out_image = images[pi, :, 384:640, 384:640].permute([1, 2, 0]).detach().cpu().numpy()
                                img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                                plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'), img)
                                plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                                           labels[pi, 384:640, 384:640].detach().cpu().numpy(), cmap = cm.gray)
                                plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
                                           prediction.detach().cpu().numpy(), cmap = cm.gray)

                                F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0),labels_onehot[pi].unsqueeze(0), rmin, rmax, cmin, cmax)
                                row = len(task_single_df)
                                task_single_df.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]

                        elif now_task in size_512:
                            images = task_pool_image.query(batch_size)
                            labels = task_pool_mask.query(batch_size)
                            scales = torch.ones(batch_size).cuda()
                            layers = torch.ones(batch_size).cuda()
                            now_task = torch.tensor(now_task)
                            filename = []
                            for bi in range(len(scales)):
                                scales[bi] = task_scale.pop(0)
                                layers[bi] = task_layer.pop(0)
                                filename.append(task_filename.pop(0))
                            preds, _, _, _ = model(images[:, :, 256:768, 256:768],
                                                torch.ones(batch_size).cuda() * now_task, scales)

                            now_preds = torch.argmax(preds, 1) == 1
                            now_preds_onehot = one_hot_3D(now_preds.long())

                            labels_onehot = one_hot_3D(labels[:, 256:768, 256:768].long())
                            rmin, rmax, cmin, cmax = 0, 512, 0, 512
                            F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin, cmax)

                            val_F1[now_task] += F1
                            val_Dice[now_task] += DICE
                            val_TPR[now_task] += TPR
                            val_PPV[now_task] += PPV
                            cnt[now_task] += 1

                            for pi in range(len(images)):
                                prediction = now_preds[pi]
                                num = len(glob.glob(os.path.join(output_folder, '*')))
                                out_image = images[pi, :, 256:768, 256:768].permute([1, 2, 0]).detach().cpu().numpy()
                                img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                                plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'), img)
                                plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                                           labels[pi, 256:768, 256:768].detach().cpu().numpy(), cmap = cm.gray)
                                plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
                                           prediction.detach().cpu().numpy(), cmap = cm.gray)

                                F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0),labels_onehot[pi].unsqueeze(0), rmin, rmax, cmin, cmax)
                                row = len(task_single_df)
                                task_single_df.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]

            'last round pop'
            for now_task in range(task_num):  # Loop through tasks 0 to 22
                task_pool_image = globals()[f'task{now_task}_pool_image']
                task_pool_mask = globals()[f'task{now_task}_pool_mask']
                task_pool_weight = globals()[f'task{now_task}_pool_weight']
                task_scale = globals()[f'task{now_task}_scale']
                task_layer = globals()[f'task{now_task}_layer']
                task_filename = globals()[f'task{now_task}_filename']
                task_single_df = globals()[f'task{now_task}_single_df']

                if task_pool_image.num_imgs > 0:
                    batch_size = task_pool_image.num_imgs

                    if now_task in size_1024:
                        images = task_pool_image.query(batch_size)
                        labels = task_pool_mask.query(batch_size)
                        now_task = torch.tensor(now_task)
                        scales = torch.ones(batch_size).cuda()
                        layers = torch.ones(batch_size).cuda()
                        filename = []
                        for bi in range(len(scales)):
                            scales[bi] = task_scale.pop(0)
                            layers[bi] = task_layer.pop(0)
                            filename.append(task_filename.pop(0))

                        preds = torch.zeros((batch_size, 2, 1024, 1024)).cuda()
                        preds[:, :, :512, :512], _, _, _ = model(images[:, :, :512, :512],
                                                              torch.ones(batch_size).cuda() * now_task, scales)
                        preds[:, :, :512, 512:], _, _, _ = model(images[:, :, :512, 512:],
                                                              torch.ones(batch_size).cuda() * now_task, scales)
                        preds[:, :, 512:, 512:], _, _, _ = model(images[:, :, 512:, 512:],
                                                              torch.ones(batch_size).cuda() * now_task, scales)
                        preds[:, :, 512:, :512], _, _, _ = model(images[:, :, 512:, :512],
                                                              torch.ones(batch_size).cuda() * now_task, scales)

                        now_preds = torch.argmax(preds, 1) == 1
                        now_preds_onehot = one_hot_3D(now_preds.long())

                        labels_onehot = one_hot_3D(labels.long())

                        rmin, rmax, cmin, cmax = 0, 1024, 0, 1024
                        F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin,
                                                         cmax)

                        val_F1[now_task] += F1
                        val_Dice[now_task] += DICE
                        val_TPR[now_task] += TPR
                        val_PPV[now_task] += PPV
                        cnt[now_task] += 1

                        for pi in range(len(images)):
                            prediction = now_preds[pi]
                            num = len(glob.glob(os.path.join(output_folder, '*')))
                            out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
                            img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                            plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'), img)
                            plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'), labels[pi, ...].detach().cpu().numpy(),cmap = cm.gray)
                            plt.imsave(
                                os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
                                prediction.detach().cpu().numpy(), cmap = cm.gray)

                            F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0),
                                                            labels_onehot[pi].unsqueeze(0), rmin, rmax, cmin, cmax)
                            row = len(task_single_df)
                            task_single_df.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]

                    elif now_task in size_256:
                        images = task_pool_image.query(batch_size)
                        labels = task_pool_mask.query(batch_size)
                        scales = torch.ones(batch_size).cuda()
                        layers = torch.ones(batch_size).cuda()
                        now_task = torch.tensor(now_task)
                        filename = []
                        for bi in range(len(scales)):
                            scales[bi] = task_scale.pop(0)
                            layers[bi] = task_layer.pop(0)
                            filename.append(task_filename.pop(0))

                        preds, _, _, _ = model(images[:, :, 256:768, 256:768], torch.ones(batch_size).cuda() * now_task,
                                            scales)

                        now_preds = torch.argmax(preds, 1) == 1
                        now_preds_onehot = one_hot_3D(now_preds.long())

                        labels_onehot = one_hot_3D(labels[:, 256:768, 256:768].long())
                        rmin, rmax, cmin, cmax = 128, 384, 128, 384
                        F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin,
                                                         cmax)

                        val_F1[now_task] += F1
                        val_Dice[now_task] += DICE
                        val_TPR[now_task] += TPR
                        val_PPV[now_task] += PPV
                        cnt[now_task] += 1

                        for pi in range(len(images)):
                            prediction = now_preds[pi, 128:384, 128:384]
                            num = len(glob.glob(os.path.join(output_folder, '*')))
                            out_image = images[pi, :, 384:640, 384:640].permute(
                                [1, 2, 0]).detach().cpu().numpy()
                            img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                            plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'), img)
                            plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                                       labels[pi, 384:640, 384:640].detach().cpu().numpy(), cmap = cm.gray)
                            plt.imsave(
                                os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
                                prediction.detach().cpu().numpy(), cmap = cm.gray)

                            F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0),
                                                            labels_onehot[pi].unsqueeze(0), rmin, rmax, cmin, cmax)
                            row = len(task_single_df)
                            task_single_df.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]

                    elif now_task in size_512:
                        images = task_pool_image.query(batch_size)
                        labels = task_pool_mask.query(batch_size)
                        scales = torch.ones(batch_size).cuda()
                        layers = torch.ones(batch_size).cuda()
                        now_task = torch.tensor(now_task)
                        filename = []
                        for bi in range(len(scales)):
                            scales[bi] = task_scale.pop(0)
                            layers[bi] = task_layer.pop(0)
                            filename.append(task_filename.pop(0))

                        preds, _, _, _ = model(images[:, :, 256:768, 256:768], torch.ones(batch_size).cuda() * now_task,
                                            scales)

                        now_preds = torch.argmax(preds, 1) == 1
                        now_preds_onehot = one_hot_3D(now_preds.long())

                        labels_onehot = one_hot_3D(labels[:, 256:768, 256:768].long())
                        rmin, rmax, cmin, cmax = 0, 512, 0, 512
                        F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin,
                                                         cmax)

                        val_F1[now_task] += F1
                        val_Dice[now_task] += DICE
                        val_TPR[now_task] += TPR
                        val_PPV[now_task] += PPV
                        cnt[now_task] += 1

                        for pi in range(len(images)):
                            prediction = now_preds[pi]
                            num = len(glob.glob(os.path.join(output_folder, '*')))
                            out_image = images[pi, :, 256:768, 256:768].permute(
                                [1, 2, 0]).detach().cpu().numpy()
                            img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                            plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'), img)
                            plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                                       labels[pi, 256:768, 256:768].detach().cpu().numpy(), cmap = cm.gray)
                            plt.imsave(
                                os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
                                prediction.detach().cpu().numpy(), cmap = cm.gray)

                            F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0),
                                                            labels_onehot[pi].unsqueeze(0), rmin, rmax, cmin, cmax)
                            row = len(task_single_df)
                            task_single_df.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]

            avg_val_F1 = val_F1 / cnt
            avg_val_Dice = val_Dice / cnt
            avg_val_TPR = val_TPR / cnt
            avg_val_PPV = val_PPV / cnt

            class_name = ['0medulla', '1cortex', '2dt', '3pt', '4cap', '5tuft', '6art', '7ptc', '8mv', '9pod', '10mes',
                          '11gloendo', '12glopecs', '13smooth', '14vesendo',
                          '15Adhesion', '16CapsularDrop', '17GlobalSclerosis', '18Haylinosis', '19MesangialExpansion',
                          '20MesangialLysis', '21Microaneurysm', '22NodularSclerosis', '23SegmentalSclerosis']

            output_str = 'Validate\n'

            for i in range(task_num):
                output_str += ' {}_f1={{:.4}} dsc={{:.4}} tpr={{:.4}} ppv={{:.4}}\n'.format(class_name[i])
                task_single_df = globals()[f'task{i}_single_df']
                task_single_df.to_csv(os.path.join(output_folder, 'testing_result_%d.csv' % (i)))

            # Print the formatted string with the metrics for each task
            print(output_str.format(
                *[val.item() for task_metrics in zip(avg_val_F1, avg_val_Dice, avg_val_TPR, avg_val_PPV) for val
                  in task_metrics]
            ))

            # Create a DataFrame with the appropriate columns
            df = pd.DataFrame(columns=['task', 'F1', 'Dice', 'TPR', 'PPV'])

            # Populate the DataFrame for each task using a loop
            for i in range(task_num):
                df.loc[i] = [
                    class_name[i],
                    avg_val_F1[i].item(),
                    avg_val_Dice[i].item(),
                    avg_val_TPR[i].item(),
                    avg_val_PPV[i].item()
                ]

            # Save the DataFrame to a CSV file
            df.to_csv(os.path.join(output_folder, 'testing_result.csv'), index=False)

    end = timeit.default_timer()
    print(end - start, 'seconds')


if __name__ == '__main__':
    main(0)
