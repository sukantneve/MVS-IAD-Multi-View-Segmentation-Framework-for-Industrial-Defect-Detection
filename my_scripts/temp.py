import glob
import random

import cv2
import natsort
import numpy as np
from PIL import Image
from scipy.ndimage import rotate as nd_rotate



def cv_random_flip(img, label):
    flip_flag = random.randint(0, 1)
    # flip_flag2= random.randint(0,1)
    # left right flip
    if flip_flag == 1:
        img = cv2.flip(img, 1)
        try:
            # label = label.transpose(Image.FLIP_LEFT_RIGHT)
            label = cv2.flip(label, 1)
        except:
            label = np.fliplr(label)

    # top bottom flip
    # if flip_flag2==1:
    #     img = img.transpose(Image.FLIP_TOP_BOTTOM)
    #     label = label.transpose(Image.FLIP_TOP_BOTTOM)
    #     depth = depth.transpose(Image.FLIP_TOP_BOTTOM)
    return img, label



def randomCrop(image, label):
    """
    Random center crop for image (PIL.Image) and label (NumPy array with shape HxWxC).
    """
    border = 60
    image_width, image_height = image.size

    # Random crop size
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)

    # Compute centered crop region
    left = (image_width - crop_win_width) >> 1
    upper = (image_height - crop_win_height) >> 1
    right = (image_width + crop_win_width) >> 1
    lower = (image_height + crop_win_height) >> 1

    # Crop image (PIL)
    cropped_image = image.crop((left, upper, right, lower))

    # Crop label (NumPy)
    cropped_label = label[upper:lower, left:right, :]  # maintain all channels

    return cropped_image, cropped_label



def randomRotation(image: Image.Image, label: np.ndarray):
    """
    Random rotation for image (PIL.Image) and label (NumPy array of shape HxWxC).
    """
    mode = Image.BICUBIC
    if random.random() > 0.5:
        random_angle = np.random.randint(-30, 30)
        image = image.rotate(random_angle, resample=mode)

        # Rotate each channel of the label independently
        rotated_channels = []
        for c in range(label.shape[2]):
            rotated = nd_rotate(label[:, :, c], random_angle, reshape=False, order=0)  # nearest-neighbor for masks
            rotated_channels.append(rotated)

        label = np.stack(rotated_channels, axis=-1)

    return image, label


def multi_binary_loader(pathdir):
    paths = natsort.natsorted(glob.glob(pathdir + "/*"))
    images_stacked = []
    for path in paths:
        img_np = np.array(Image.open(path).convert('L').resize((512, 512)))
        images_stacked.append(img_np)  # shape: (1, H, W)
        # images_stacked.append(torch.from_numpy(img_np).unsqueeze(0))  # shape: (1, H, W)
    # if len(images_stacked) == 0:
    #     return torch.empty(0)
    # dstacked = torch.cat(images_stacked, dim=0)  # shape: (num_masks, H, W)

    dstacked = np.dstack(images_stacked)

    # HWC -> CHW
    # dstacked = np.transpose(dstacked, (2, 0, 1))

    return dstacked



RGBImage = multi_binary_loader("/home/exouser/CapstoneProject/MVANet/training_dataset_4_class/val/masks/pcb_0001_NG_HS_C3_20231028093757")
_, RGBImage = randomRotation(Image.fromarray(RGBImage), RGBImage)


cv2.imshow("rgb", RGBImage)
cv2.waitKey(0)





