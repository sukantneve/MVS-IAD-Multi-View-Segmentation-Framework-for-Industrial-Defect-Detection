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

writer = SummaryWriter()

cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=200, help='epoch number')
parser.add_argument('--lr_gen', type=float, default=1e-5, help='learning rate')
parser.add_argument('--batchsize', type=int, default=1, help='training batch size')
parser.add_argument('--trainsize', type=int, default=1024, help='training dataset size')
parser.add_argument('--decay_rate', type=float, default=0.9, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=30, help='every n epochs decay learning rate')
parser.add_argument('--resume', type=bool, default=False, help="Flag for resume Training")
parser.add_argument('--save_path', type=str, default='/media/volume/Team2/mva_saved_models/1024_jun15/', help="path to saved model checkpoint")
parser.add_argument('--image_root', type=str, default='/home/exouser/CapstoneProject/MVANet/training_dataset_4_class/train/images/', help="path to Original Images")
parser.add_argument('--gt_root', type=str, default='/home/exouser/CapstoneProject/MVANet/training_dataset_4_class/train/masks/', help="path to Masks")
parser.add_argument('--outside_bgs_paths', type=str,
                    default='/media/nithin/17b8fbe0-a0a3-46d6-bb0e-d8bebe7bcbd0/360booth/SEGMENTATION_TRAIN_DATA/CURRENT_OUTSIDE/back_grounds/*/*', help="path to backgrounds")


opt = parser.parse_args()
print('Generator Learning Rate: {}'.format(opt.lr_gen))
# build models
if hasattr(torch.cuda, 'empty_cache'):
    torch.cuda.empty_cache()

num_classes = len(os.listdir(glob.glob(opt.gt_root +  "*/")[0]))

print("Num_classes: ", num_classes)

generator = MVANet(num_classes=num_classes, emb_dim=256)
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
        generator_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
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
train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize,
                          trainsize=opt.trainsize, outside_bgs_paths=outside_bgs_paths)

#val_loader = get_loader(image_root.replace("/train/", "/val/"), gt_root.replace("/train/", "/val/"), batchsize=opt.batchsize,
#                          trainsize=opt.trainsize, outside_bgs_paths=outside_bgs_paths)

total_step = len(train_loader)
to_pil = transforms.ToPILImage()
## define loss

CE = torch.nn.BCELoss()
mse_loss = torch.nn.MSELoss(size_average=True, reduce=True)
size_rates = [1]
criterion = nn.BCEWithLogitsLoss().cuda()
criterion_mae = nn.L1Loss().cuda()
criterion_mse = nn.MSELoss().cuda()
use_fp16 = True
scaler = amp.GradScaler(enabled=use_fp16)


def structure_loss_default(pred, mask):
    pred = pred.cuda()
    mask = mask.cuda()
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))

    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()



def compute_class_weighted_loss(preds, targets, structure_loss_fn):
    B, C, H, W = preds.shape
    losses = []
    for i in range(C):
        pred_i = preds[:, i:i+1]
        target_i = targets[:, i:i+1]
        weight = torch.clamp(target_i.sum() / (H * W), min=0.01).item()
        loss_i = structure_loss_fn(pred_i, target_i) * (1.0 / weight)
        losses.append(loss_i)
    return sum(losses) / C


class MultiClassSegLoss(nn.Module):
    """
    Focal-Tversky + BCE Loss for multi-class defect segmentation.
    Automatically skips loss for empty masks (non-defect images).
    """
    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        B, C, H, W = logits.shape
        targets = targets.float()
        probs = torch.sigmoid(logits)

        loss_total = 0.0
        valid_class_count = 0

        for c in range(C):
            pred_c = probs[:, c]
            target_c = targets[:, c]
            logit_c = logits[:, c]

            # Skip class if completely empty
            class_area = target_c.sum().item()
            if class_area < 1e-5:
                continue

            # BCE loss
            bce = F.binary_cross_entropy_with_logits(logit_c, target_c, reduction='mean')

            # Focal Tversky loss
            tp = (pred_c * target_c).sum(dim=(1, 2))
            fp = (pred_c * (1 - target_c)).sum(dim=(1, 2))
            fn = ((1 - pred_c) * target_c).sum(dim=(1, 2))
            tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
            ft_loss = ((1.0 - tversky) ** self.gamma).mean()

            loss_total += (bce + ft_loss)
            valid_class_count += 1

            print(f"class {c} loss: ", (bce + ft_loss).item(), end=" | ")
            
        if valid_class_count == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        
        print("total loss: ", loss_total.item(), end=" | ")
        print("Valid Loss: ", (loss_total / valid_class_count).item(), end="\n")
        
        return loss_total / valid_class_count



class MultiClassSegLossBalanced(nn.Module):
    """
    BCE + Focal-Tversky loss with:
    - Dynamic skipping of empty masks
    - Balanced weighting between BCE and Tversky
    - Tunable Tversky and focal parameters
    Suitable for imbalanced multi-label segmentation (e.g., defect detection).
    """
    def __init__(self, alpha=0.3, beta=0.7, gamma=0.5, smooth=1.0,
                 bce_weight=0.5, tversky_weight=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.bce_weight = bce_weight
        self.tversky_weight = tversky_weight

    def forward(self, logits, targets):
        B, C, H, W = logits.shape
        targets = targets.float()
        probs = torch.sigmoid(logits)

        total_loss = 0.0
        valid_class_count = 0

        for Dc in range(C):
            pred_c = probs[:, c]
            target_c = targets[:, c]
            logit_c = logits[:, c]

            class_area = target_c.sum().item()
            if class_area < 1e-5:
                continue  # skip empty masks

            # BCE Loss
            bce = F.binary_cross_entropy_with_logits(logit_c, target_c, reduction='mean')

            # Focal Tversky Loss
            tp = (pred_c * target_c).sum(dim=(1, 2))
            fp = (pred_c * (1 - target_c)).sum(dim=(1, 2))
            fn = ((1 - pred_c) * target_c).sum(dim=(1, 2))
            tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
            ft_loss = ((1.0 - tversky) ** self.gamma).mean()

            # Combined loss
            loss = self.bce_weight * bce + self.tversky_weight * ft_loss
            total_loss += loss
            valid_class_count += 1

            print(f"class {c} loss: ", loss, end=" | ")

        if valid_class_count == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)


        print("total loss: ", total_loss, end=" | ")
        print("Valid Loss: ", total_loss / valid_class_count, end=" | ")

        return total_loss / valid_class_count


structure_loss_multi_class = MultiClassSegLoss().cuda()

for epoch in range(start_epoch, opt.epoch + 1):
    train_loader.dataset.shuffle_samples()

    torch.cuda.empty_cache()
    generator.train()
    loss_record = AvgMeter()
    print('Generator Learning Rate: {}'.format(generator_optimizer.param_groups[0]['lr']))
    for i, pack in enumerate(train_loader, start=1):

        torch.cuda.empty_cache()
        for rate in size_rates:
            torch.cuda.empty_cache()
            generator_optimizer.zero_grad()
            images, gts = pack

            images = Variable(images)
            gts = Variable(gts)

            images = images.cuda()
            gts = gts.cuda()

            # empty_channel = torch.zeros((1, 1, opt.trainsize, opt.trainsize), device=gts.device, dtype=gts.dtype)
            # gts = torch.cat([empty_channel, gts], dim=1)

            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear',
                                    align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

            b, c, h, w = gts.size()
            target_1 = F.upsample(gts, size=h // 4, mode='nearest')
            target_2 = F.upsample(gts, size=h // 8, mode='nearest').cuda()
            target_3 = F.upsample(gts, size=h // 16, mode='nearest').cuda()
            target_4 = F.upsample(gts, size=h // 32, mode='nearest').cuda()
            target_5 = F.upsample(gts, size=h // 64, mode='nearest').cuda()

            with amp.autocast(enabled=use_fp16):
                (sideout5, sideout4, sideout3, sideout2, sideout1, final,
                 glb5, glb4, glb3, glb2, glb1,
                 tokenattmap4, tokenattmap3, tokenattmap2, tokenattmap1) = generator.forward(images)

                loss1 = structure_loss_default(sideout5, target_4)
                loss2 = structure_loss_default(sideout4, target_3)
                loss3 = structure_loss_default(sideout3, target_2)
                loss4 = structure_loss_default(sideout2, target_1)
                loss5 = structure_loss_default(sideout1, target_1)
                loss6 = structure_loss_multi_class(final, gts)
                # loss6 = compute_class_weighted_loss(final, gts, structure_loss_default)
                # loss6 = sum(structure_loss_default(final[:, i:i+1, :, :], gts[:, i:i+1, :, :]) for i in range(num_classes))

                loss7 = structure_loss_default(glb5, target_5)
                loss8 = structure_loss_default(glb4, target_4)
                loss9 = structure_loss_default(glb3, target_3)
                loss10 = structure_loss_default(glb2, target_2)
                loss11 = structure_loss_default(glb1, target_2)
                loss12 = structure_loss_default(tokenattmap4, target_3)
                loss13 = structure_loss_default(tokenattmap3, target_2)
                loss14 = structure_loss_default(tokenattmap2, target_1)
                loss15 = structure_loss_default(tokenattmap1, target_1)
                loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + 0.3 * (
                            loss7 + loss8 + loss9 + loss10 + loss11) + 0.3 * (loss12 + loss13 + loss14 + loss15)
                Loss_loc = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
                Loss_glb = loss7 + loss8 + loss9 + loss10 + loss11
                Loss_map = loss12 + loss13 + loss14 + loss15
                writer.add_scalar('loss', loss.item(), epoch * len(train_loader) + i)

            generator_optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(generator_optimizer)
            scaler.update()

            if rate == 1:
                loss_record.update(loss.data, opt.batchsize)

        if i % 10 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], gen Loss: {:.4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step, loss_record.show()))


        if i % 250 == 0 or i % 251 == 0  or i == total_step:
            final_p = final.sigmoid()
            training_resultsPath = f"results/1024_jun15/train/epoch_{epoch}_{i}/"
            os.makedirs(training_resultsPath, exist_ok=True)

            # Iterate over channels
            for ch in range(final_p.shape[1]):
                channel_img = final_p[0, ch].detach().cpu()  # shape: [H, W]
                pil_img = to_pil(channel_img)
                gt_img = to_pil(gts[0, ch].detach().cpu())

                image_np = np.array(pil_img)
                gt_np = np.array(gt_img)

                hstacked = np.hstack((image_np, gt_np))
                cv2.imwrite(f"{training_resultsPath}/epoch_{epoch}_ch{ch + 1}.png", hstacked)


    adjust_lr(generator_optimizer, opt.lr_gen, epoch, opt.decay_rate, opt.decay_epoch)
    # save checkpoints every 20 epochs
    if epoch % 1 == 0:
        save_path = opt.save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        # torch.save(generator.state_dict(), save_path + 'Model' + '_%d' % epoch + '.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': generator.state_dict(),
            # 'optimizer_state_dict': generator_optimizer.state_dict(),
        }, save_path + 'Model' + '_%d' % epoch + '.pth')

