# import natsort
import glob
import os.path
import time

import cv2
import natsort
import numpy as np
# import os
# import time
import torch
# import tqdm
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from model.MVANet import inf_MVANet
import ttach as tta

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

depth_transform = transforms.ToTensor()
target_transform = transforms.ToTensor()
to_pil = transforms.ToPILImage()

transforms = tta.Compose(
    [
        tta.HorizontalFlip(),
        tta.Scale(scales=[0.75, 1, 1.25], interpolation='bilinear', align_corners=False),
        # tta.Scale(scales=[1], interpolation='bilinear', align_corners=False),
    ]
)


class MVASEG:
    # def __init__(self, modelPath, device = "cuda"):
    def __init__(self, modelPath, device = "cuda"):
        self.modelPath = modelPath
        self.device = device
        self.net = self.loadModel()


    def loadModel(self):
        net = inf_MVANet(num_classes=5, emb_dim=320).to(self.device)
        print(f"Model instantiated with num_classes = {net.num_classes}")

        pretrained_dict = torch.load(self.modelPath, map_location='cpu')
        model_dict = net.state_dict()

        if "model_state_dict" in pretrained_dict.keys():
            pretrained_dict = pretrained_dict["model_state_dict"]

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)
        # net.eval()

        return net


    @torch.inference_mode()
    def __call__(self, img):
        #### img : pillow

        w_, h_ = img.size
        img_resize = img.resize([512, 512], Image.BILINEAR)
        img_var = Variable(img_transform(img_resize).unsqueeze(0), volatile=True).to(self.device)
        # img_var = Variable(img_transform(img_resize).unsqueeze(0), volatile=True).cpu()
        mask = []

        for m_idx, transformer in enumerate(transforms):
            rgb_trans = transformer.augment_image(img_var)
            model_output = self.net(rgb_trans)
            deaug_mask = transformer.deaugment_mask(model_output)
            mask.append(deaug_mask)

        prediction = torch.mean(torch.stack(mask, dim=0), dim=0)
        prediction = prediction.sigmoid()
        prediction = prediction[:, 1:]
        prediction = to_pil(prediction.data.squeeze(0).detach().cpu())
        prediction = prediction.resize((w_, h_), Image.BILINEAR)

        return prediction


if __name__ == '__main__':
    modelPath = "/media/volume/Team2/mva_saved_models/512/Model_88.pth"
    mva_seg = MVASEG(modelPath, "cuda")

    imagePaths = natsort.natsorted(glob.glob("/home/exouser/CapstoneProject/for_vis/images/*"))[6:]
    maskPaths = natsort.natsorted(glob.glob("/home/exouser/CapstoneProject/for_vis/gts/*"))[6:]
    saveDir = "/home/exouser/CapstoneProject/for_vis/pred"
    os.makedirs(saveDir, exist_ok=True)



    HVSTACKED_IMAGE = None
    for idx, imagePath in enumerate(imagePaths):
        # img_pil = Image.open("/home/exouser/CapstoneProject/MVANet/training_dataset_4_class/val/images/pcb_0001_NG_HS_C3_20231028093757.jpg").convert("RGB")
        img_pil = Image.open(imagePath).convert("RGB")

        if "NG_HS" in imagePath:
            gt_class = 0
        elif "NG_QS" in imagePath:
            gt_class = 1
        elif "NG_YW" in imagePath:
            gt_class = 2
        else:
            gt_class = 3


        st = time.time()
        pred_mask = mva_seg(img_pil)
        print("end: ", time.time() - st)

        pred_mask_np = cv2.split(np.array(pred_mask))[gt_class]
        cv2.imwrite(os.path.join(saveDir, os.path.basename(maskPaths[idx])), pred_mask_np)
        pred_mask_np = np.dstack((pred_mask_np, pred_mask_np, pred_mask_np))


        image_np = cv2.imread(imagePath)
        gt_mask_np = cv2.imread(maskPaths[idx])
        hstacked_image = np.hstack([image_np, gt_mask_np, pred_mask_np])

        if HVSTACKED_IMAGE is None:
            HVSTACKED_IMAGE = hstacked_image.copy()
            continue


        HVSTACKED_IMAGE = np.vstack([HVSTACKED_IMAGE, hstacked_image])

        # for i, mask in enumerate(masks_list):
        #     cv2.imshow(str(i), mask)
        # cv2.waitKey(0)
        #
        # pred_mask.show()


    input_text_image = np.full((256, 512, 3), dtype=np.uint8)


    HVSTACKED_IMAGE = np.vstack([])

    cv2.imwrite("HVSTACKED_IMAGE.png", HVSTACKED_IMAGE)
