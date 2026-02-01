import glob

import cv2
import natsort
import numpy as np
import torch
import os, argparse

import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from datetime import datetime
from model.MVANet import MVANet
from utils.dataset_strategy_fpn import get_loader
from utils.misc import adjust_lr, AvgMeter
import torch.nn.functional as F
from torch.autograd import Variable
from torch.backends import cudnn
from torchvision import transforms
import torch.nn as nn
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
from utils.eval_metrics import *


writer = SummaryWriter()

cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=200, help='epoch number')
parser.add_argument('--lr_gen', type=float, default=1e-5, help='learning rate')
parser.add_argument('--batchsize', type=int, default=1, help='training batch size')
parser.add_argument('--trainsize', type=int, default=1024, help='training dataset size')
parser.add_argument('--decay_rate', type=float, default=0.9, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=30, help='every n epochs decay learning rate')
parser.add_argument('--resume', type=bool, default=True, help="Flag for resume Training")
parser.add_argument('--save_path', type=str, default='/media/volume/Team2/mva_saved_models/1024/', help="path to saved model checkpoint")
parser.add_argument('--image_root', type=str, default='/home/exouser/CapstoneProject/MVANet/training_dataset_4_class/val/images/', help="path to Original Images")
parser.add_argument('--gt_root', type=str, default='/home/exouser/CapstoneProject/MVANet/training_dataset_4_class/val/masks/', help="path to Masks")
parser.add_argument('--outside_bgs_paths', type=str,
                    default='/media/nithin/17b8fbe0-a0a3-46d6-bb0e-d8bebe7bcbd0/360booth/SEGMENTATION_TRAIN_DATA/CURRENT_OUTSIDE/back_grounds/*/*', help="path to backgrounds")


opt = parser.parse_args()
print('Generator Learning Rate: {}'.format(opt.lr_gen))
# build models
if hasattr(torch.cuda, 'empty_cache'):
    torch.cuda.empty_cache()

num_classes = len(os.listdir(glob.glob(opt.gt_root +  "*/")[0])) + 1

print("Num_classes: ", num_classes)

generator = MVANet(num_classes=num_classes, emb_dim=320)
generator.cuda()

generator_params = generator.parameters()
generator_optimizer = torch.optim.Adam(generator_params, opt.lr_gen)

# Resume training if checkpoint exists
start_epoch = 1  # Default start epoch
if opt.resume:
    last_epoch_path = natsort.natsorted(glob.glob(opt.save_path + "/*"))[-1]
    epoch_num = int(os.path.basename(last_epoch_path).split("Model_")[1].split(".pth")[0])
    print(f"=> Loading checkpoint: {last_epoch_path}")
    checkpoint = torch.load(last_epoch_path, map_location="cpu")

    try:
        generator.load_state_dict(checkpoint["model_state_dict"])
        # generator_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    except:
        print("===================================")
        print("ERROR: USING DEFAULT MVANET MODEL")
        print("===================================")

        generator.load_state_dict(checkpoint["model_state_dict"])

    del checkpoint
    start_epoch = epoch_num + 1  # Start from the next epoch
    print(f"=> Resumed training from epoch {start_epoch}")


image_root = opt.image_root
gt_root = opt.gt_root




outside_bgs_paths = glob.glob(opt.outside_bgs_paths)
val_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, shuffle=False,
                        trainsize=opt.trainsize, outside_bgs_paths=outside_bgs_paths, mode="val")


total_step = len(val_loader)
to_pil = transforms.ToPILImage()
## define loss

size_rates = [1]
use_fp16 = True



torch.cuda.empty_cache()
generator.eval()




def compute_classwise_auroc(preds, gts):


    preds_np = preds.squeeze(0).detach().cpu().numpy()  # (C, H, W)
    gts_np = gts.squeeze(0).detach().cpu().numpy()

    n_classes = preds.shape[1]
    aurocs = []
    f1_scores = []
    avg_precisions = []
    for c in range(n_classes):
        # if c == 0:
        #     continue

        pred_c = preds_np[c, :, :].flatten()
        gt_c = gts_np[c, :, :].flatten()
        gt_c = (gt_c >= 0.5).astype(np.uint8)

        if len(np.unique(gt_c)) < 2:
            # Only 0s or only 1s ? AUROC is undefined
            # print(f"[SKIPPED] Class {c}: Only one class present in GT: {np.unique(gt_c)}")
            aurocs.append(0)
        else:
            auroc = roc_auc_score(gt_c, pred_c)
            # f1_scores.append(f1_score(gt_c, pred_c))
            avg_precisions.append(average_precision_score(gt_c, pred_c))
            aurocs.append(auroc)

    # valid_aurocs = [a for a in aurocs if not np.isnan(a)]
    # mean_auroc = np.mean(valid_aurocs) if valid_aurocs else np.nan
    #
    # valid_f1 = [a for a in f1_scores if not np.isnan(a)]
    # valid_avg_precision = [a for a in avg_precisions if not np.isnan(a)]


    return aurocs, f1_scores, avg_precisions
print('Generator Learning Rate: {}'.format(generator_optimizer.param_groups[0]['lr']))


with torch.no_grad():

    per_class_aurocs = []
    per_class_f1_scores = []
    per_class_avg_precisions = []
    per_class_aupros = []
    for i, vpack in enumerate(tqdm.tqdm(val_loader), start=1):

        # if i > 10:
        #     break

        images, gts = vpack

        images = Variable(images)
        gts = Variable(gts)

        images = images.cuda()
        gts = gts.cuda()

        with amp.autocast(enabled=use_fp16):
            # (sideout5, sideout4, sideout3, sideout2, sideout1, final,
            #  glb5, glb4, glb3, glb2, glb1,
            #  tokenattmap4, tokenattmap3, tokenattmap2, tokenattmap1) = generator.forward(images)

            final = generator.forward(images)

            empty_mask = torch.zeros(1, 1, opt.trainsize, opt.trainsize, device=gts.device, dtype=gts.dtype)
            gts = torch.cat((empty_mask, gts), dim=1)

            final_p = final.sigmoid()


            # per_class_auroc_list, pre_class_f1_score_list, per_class_avg_precision_list = compute_classwise_auroc(final_p, gts)

            metrics = compute_segmentation_metrics(final_p, gts)

            per_class_aurocs.append(metrics['auroc'])
            per_class_f1_scores.append(metrics['f1'])
            per_class_avg_precisions.append(metrics['ap'])
            per_class_aupros.append(metrics['aupro'])

            # continue

            training_resultsPath = f"results/1024/val/{i}/"
            os.makedirs(training_resultsPath, exist_ok=True)

            # Iterate over channels
            for ch in range(final_p.shape[1]):
                channel_img = final_p[0, ch].detach().cpu()  # shape: [H, W]
                pil_img = to_pil(channel_img)
                gt_img = to_pil(gts[0, ch].detach().cpu())

                image_np = np.array(pil_img)
                gt_np = np.array(gt_img)

                hstacked = np.hstack((image_np, gt_np))
                cv2.imwrite(f"{training_resultsPath}/{i+1}.png", hstacked)



avg_aurocs, auroc_counts = average_per_class_metric(per_class_aurocs)
avg_f1s, f1_counts = average_per_class_metric(per_class_f1_scores)
avg_aps, ap_counts = average_per_class_metric(per_class_avg_precisions)
avg_aupros, aupro_counts = average_per_class_metric(per_class_aupros)



weighted_auroc = np.average(avg_aurocs, weights=auroc_counts)
weighted_f1 = np.average(avg_f1s, weights=f1_counts)
weighted_ap = np.average(avg_aps, weights=ap_counts)
weighted_aupro = np.average(avg_aupros, weights=aupro_counts)



# Print per-class AUROC
print("=== Per-Class AUROC ===")
for i, (score, count) in enumerate(zip(avg_aurocs, auroc_counts)):
    print(f"Class {i}: AUROC = {score:.4f} (Valid Samples = {count})")
print(f">>> Weighted AUROC: {weighted_auroc:.4f}")
print("=" * 50)

# Print per-class F1 Score
print("=== Per-Class F1 Score ===")
for i, (score, count) in enumerate(zip(avg_f1s, f1_counts)):
    print(f"Class {i}: F1 Score = {score:.4f} (Valid Samples = {count})")
print(f">>> Weighted F1 Score: {weighted_f1:.4f}")
print("=" * 50)

# Print per-class Average Precision (AP)
print("=== Per-Class Average Precision (AP) ===")
for i, (score, count) in enumerate(zip(avg_aps, ap_counts)):
    print(f"Class {i}: AP = {score:.4f} (Valid Samples = {count})")
print(f">>> Weighted Average Precision: {weighted_ap:.4f}")
print("=" * 50)

# Print per-class AUPRO
print("=== Per-Class AUPRO ===")
for i, (score, count) in enumerate(zip(avg_aupros, aupro_counts)):
    print(f"Class {i}: AUPRO = {score:.4f} (Valid Samples = {count})")
print(f">>> Weighted AUPRO: {weighted_aupro:.4f}")
print("=" * 50)



metrics_dict = {
    "AUROC": (avg_aurocs, auroc_counts, weighted_auroc),
    "F1": (avg_f1s, f1_counts, weighted_f1),
    "AP": (avg_aps, ap_counts, weighted_ap),
    "AUPRO": (avg_aupros, aupro_counts, weighted_aupro)
}

plot_all_metrics_per_class(metrics_dict, save_path="combined_metrics_plot.png")
