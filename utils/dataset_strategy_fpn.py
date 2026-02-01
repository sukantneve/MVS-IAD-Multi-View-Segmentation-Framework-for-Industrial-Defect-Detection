import glob
import os
import string

import cv2
import natsort
import torch
import tqdm
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance
from .augmentation_utils import do_mod_lines, do_mod_vertical_lines, do_mod_words, do_mod_shapes
from .augmentation_utils import paste_half_bg_and_objects, check_for_edges, apply_cars_behind_partial_images, apply_cars_behind_image
from scipy.ndimage import rotate as nd_rotate



# several data augumentation strategies
def cv_random_flip(img, label):
    flip_flag = random.randint(0, 1)
    # flip_flag2= random.randint(0,1)
    # left right flip
    if flip_flag == 1:
        img = Image.fromarray(cv2.flip(np.array(img), 1))
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


def pil_resize(img, size):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img = img.resize(size)
    return np.array(img)


def randomCrop(image, label):
    """
    Random center crop for image (PIL.Image) and label (NumPy array with shape HxWxC).
    """
    border = 30
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
    if random.random() >= 0.5:
        random_angle = np.random.randint(-30, 30)
        image = image.rotate(random_angle, resample=mode)

        # Rotate each channel of the label independently
        rotated_channels = []
        for c in range(label.shape[2]):
            rotated = nd_rotate(label[:, :, c], random_angle, reshape=False, order=0)  # nearest-neighbor for masks
            rotated_channels.append(rotated)

        label = np.stack(rotated_channels, axis=-1)

    return image, label


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:

            img[randX, randY] = 0

        else:

            img[randX, randY] = 255
    return Image.fromarray(img)





# dataset for training
# The current loader is not using the normalized depth maps for training and test. If you use the normalized depth maps
# (e.g., 0 represents background and 1 represents foreground.), the performance will be further improved.
class DISDataset(data.Dataset):
    def __init__(self, image_root, gt_root, trainsize, outside_bgs_paths, mode='train'):
        self.mode = mode
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('tif')]
        # self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
        #             or f.endswith('.png') or f.endswith('tif')]

        self.gts = [gt_root + f for f in os.listdir(gt_root)]
        self.images = natsort.natsorted(self.images)
        self.gts = natsort.natsorted(self.gts)
        # self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            # transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.outside_bgs_path = outside_bgs_paths

        if mode == "train":
            self.blank_samples = []
            self.non_blank_samples = []
            print("Seperating blank and non-blank samples...")
            for img_path, gt_path in tqdm.tqdm(zip(self.images, self.gts)):
                gts = self.multi_binary_loader(gt_path)
                if np.all(gts == 0):
                    self.blank_samples.append((img_path, gt_path))
                elif np.any(gts > 0):
                    self.non_blank_samples.append((img_path, gt_path))

            self.size = 2 * min(len(self.blank_samples), len(self.non_blank_samples))
            print("Number of blank samples: ", len(self.blank_samples))
            print("Number of non-blank samples: ", len(self.non_blank_samples))
            print("Total samples: ", len(self.blank_samples) + len(self.non_blank_samples))
            print("Balanced samples: ", self.size)



    def shuffle_samples(self):
        random.shuffle(self.blank_samples)
        random.shuffle(self.non_blank_samples)


    def __getitem__(self, index):
        # image = self.rgb_loader(self.images[index])
        # # gt = self.binary_loader(self.gts[index])
        # gts = self.multi_binary_loader(self.gts[index])
        # image_name = os.path.basename(self.images[index])
        # aug_random = random.random()

        if self.mode == 'val':
            image = self.rgb_loader(self.images[index])
            # gt = self.binary_loader(self.gts[index])
            gts = self.multi_binary_loader(self.gts[index])

            image = self.img_transform(image)
            mask_tensors = self.gt_transform(gts)

            return image, mask_tensors


        if index % 2 == 0:
            i = index // 2 % len(self.blank_samples)
            img_path, gt_path = self.blank_samples[i]
        else:
            i = index // 2 % len(self.non_blank_samples)
            img_path, gt_path = self.non_blank_samples[i]

        image = self.rgb_loader(img_path)
        gts = self.multi_binary_loader(gt_path)

        if random.random() < 0.5:
            image, gt = cv_random_flip(image, gts)
            image, gt = randomCrop(image, gts)
            image, gt = randomRotation(image, gts)
            image = colorEnhance(image)

        image = self.img_transform(image)
        mask_tensors = self.gt_transform(gts)


        return image, mask_tensors
    

    def filter_files(self):
        assert len(self.images) == len(self.gts) and len(self.gts) == len(self.images)
        images = []
        gts = []
        for img_path, gt_path in tqdm.tqdm(zip(self.images, self.gts)):
            img = cv2.imread(img_path)

            paths = natsort.natsorted(glob.glob(gt_path + "/*"))
            
            gt_shape = list(set(cv2.imread(path, cv2.IMREAD_GRAYSCALE).shape for path in paths))
            
            assert len(gt_shape) == 1

            img_h, img_w = img.shape[:2]
            gt_h, gt_w = gt_shape[0]

            # Replace PIL's size attribute check with shape check
            if (img_h, img_w) == (gt_h, gt_w):
                images.append(img_path)
                gts.append(gt_path)


        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def multi_binary_loader(self, pathdir):
        paths = natsort.natsorted(glob.glob(pathdir + "/*"))
        images_stacked = []
        for path in paths:
            img_np = pil_resize(cv2.imread(path, 0), (self.trainsize, self.trainsize))
            images_stacked.append(img_np)  # shape: (1, H, W)
            # images_stacked.append(torch.from_numpy(img_np).unsqueeze(0))  # shape: (1, H, W)
        # if len(images_stacked) == 0:
        #     return torch.empty(0)
        # dstacked = torch.cat(images_stacked, dim=0)  # shape: (num_masks, H, W)

        dstacked = np.dstack(images_stacked)

        # HWC -> CHW
        # dstacked = np.transpose(dstacked, (2, 0, 1))

        return dstacked

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


# dataloader for training
def get_loader(image_root, gt_root, batchsize, trainsize, shuffle=True, num_workers=12, pin_memory=False, outside_bgs_paths="", mode="train"):
    dataset = DISDataset(image_root, gt_root,  trainsize, outside_bgs_paths, mode)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


# test dataset and loader
class test_dataset:
    def __init__(self, image_root, depth_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.bmp')
                       or f.endswith('.png')]
        self.images = sorted(self.images)
        self.depths = sorted(self.depths)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        # self.gt_transform = transforms.Compose([
        #     transforms.Resize((self.trainsize, self.trainsize)),
        #     transforms.ToTensor()])
        self.depths_transform = transforms.Compose(
            [transforms.Resize((self.testsize, self.testsize)), transforms.ToTensor()])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        HH = image.size[0]
        WW = image.size[1]
        image = self.transform(image).unsqueeze(0)
        depth = self.rgb_loader(self.depths[self.index])
        depth = self.depths_transform(depth).unsqueeze(0)

        name = self.images[self.index].split('/')[-1]
        # image_for_post=self.rgb_loader(self.images[self.index])
        # image_for_post=image_for_post.resize(gt.size)
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        self.index = self.index % self.size
        return image, depth, HH, WW, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size

