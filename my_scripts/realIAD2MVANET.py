import os.path
import random
import shutil

import cv2
import glob

import natsort
import numpy as np
import tqdm
import concurrent.futures

GoodImagesDir = "/home/exouser/Downloads/pcb/OK/*/*"
basebadImagesDir = "/home/exouser/Downloads/pcb/NG/"
badImagesDir = "/home/exouser/Downloads/pcb/NG/*/*/*.jpg"

num_classes = len(os.listdir(basebadImagesDir))

class_names = natsort.natsorted(os.listdir(basebadImagesDir))

training_datasetsDir = "/home/exouser/CapstoneProject/MVANet/training_dataset/"

def prepare_good_data(imagePath, mode="train"):
    imageName_without_ext = os.path.splitext(os.path.basename(imagePath))[0]

    savedirImages =  os.path.join(training_datasetsDir, mode, "images")  # "/home/exouser/CapstoneProject/MVANet/training_dataset/train/images"
    savedirMasks =  os.path.join(training_datasetsDir, mode, "masks", imageName_without_ext)
    if not os.path.exists(savedirImages):
        os.makedirs(savedirImages, exist_ok=True)

    if not os.path.exists(savedirMasks):
        os.makedirs(savedirMasks, exist_ok=True)

    shutil.copy(imagePath, savedirImages)

    im = cv2.imread(imagePath)

    mask = np.zeros_like(im)
    for i in range(num_classes):
        cv2.imwrite(os.path.join(savedirMasks, f"{i + 1}.png"), mask)

    pass


def prepare_bad_data(imagePath, mode="train"):

    defect_type = os.path.basename(os.path.dirname(os.path.dirname(imagePath)))

    imageName_without_ext = os.path.splitext(os.path.basename(imagePath))[0]
    savedirImages = os.path.join(training_datasetsDir, mode, "images")
    savedirMasks = os.path.join(training_datasetsDir, mode,  "masks", imageName_without_ext)
    if not os.path.exists(savedirImages):
        os.makedirs(savedirImages, exist_ok=True)

    if not os.path.exists(savedirMasks):
        os.makedirs(savedirMasks, exist_ok=True)

    shutil.copy(imagePath, savedirImages)


    for i in range(num_classes):

        try:

            if class_names[i] in defect_type:
                shutil.copy(imagePath.replace(".jpg", ".png"), os.path.join(savedirMasks, f"{i + 1}.png"))
            else:
                im = cv2.imread(imagePath)
                mask = np.zeros_like(im)
                cv2.imwrite(os.path.join(savedirMasks, f"{i + 1}.png"), mask)

        except:
            im = cv2.imread(imagePath)
            mask = np.zeros_like(im)
            cv2.imwrite(os.path.join(savedirMasks, f"{i+1}.png"), mask)


if __name__ == '__main__':
    goodimagespaths = glob.glob(GoodImagesDir)
    badimagespaths = glob.glob(badImagesDir)


    random.shuffle(goodimagespaths)
    random.shuffle(badimagespaths)


    train_good_dataset_images = goodimagespaths[:int(len(goodimagespaths) * .80)]
    val_good_dataset_images = goodimagespaths[int(len(goodimagespaths) * .80):]


    train_bad_dataset_images = badimagespaths[:int(len(goodimagespaths) * .80)]
    val_bad_dataset_images = badimagespaths[int(len(goodimagespaths) * .80):]


    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:

        # futures_good = [executor.submit(prepare_good_data, goodimagespath) for goodimagespath in tqdm.tqdm(goodimagespaths[:])]
        # futures_bad = [executor.submit(prepare_bad_data, badimagespath) for badimagespath in tqdm.tqdm(badimagespaths[:])]
        #
        # [future.result() for future in tqdm.tqdm(futures_good + futures_bad)]


        for goodimagespath in tqdm.tqdm(train_good_dataset_images[:]):
            prepare_good_data(goodimagespath, mode="train")


        for goodimagespath in tqdm.tqdm(val_good_dataset_images[:]):
            prepare_good_data(goodimagespath, mode="val")


        for badimagespath in tqdm.tqdm(train_bad_dataset_images[:]):
            prepare_bad_data(badimagespath, mode="train")

        for badimagespath in tqdm.tqdm(val_bad_dataset_images[:]):
            prepare_bad_data(badimagespath, mode="val")