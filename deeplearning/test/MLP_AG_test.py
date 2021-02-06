# -*- coding:utf8 -*-
# @TIME     : 2020/12/21 21:39
# @Author   : SuHao
# @File     : MLP_AG_test.py

import os
import tqdm
import torch
import numpy as np
import cv2
import argparse
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torchvision.transforms as T
from skimage.metrics import structural_similarity as ssim
from deeplearning.models.MLP_AG import MLP
from deeplearning.dataload.dataload import load_dataset
from deeplearning.test.evaluate import Evaluate
from deeplearning.dataload.self_transforms import Fllip, AddGaussianNoise, AddSaltPepperNoise, Style


parser = argparse.ArgumentParser()
parser.add_argument("--experiment", default=r"../experiments/MLP_AG_test", help="path to save experiments results")
parser.add_argument("--dataset", default="mnist_siamese", help="mnist")
parser.add_argument('--dataroot', default=r"../data", help='path to dataset')
parser.add_argument("--batchSize", type=int, default=512, help="size of the batches")
parser.add_argument("--size", type=int, default=28, help="size of image after scaled")
parser.add_argument("--imageSize", type=int, default=28, help="size of each image dimension")
parser.add_argument("--mlp_pth", default=r"../experiments/MLP_AG_train/mlp.pth", help="pretrained model of mlp")
opt = parser.parse_args()
print(opt)
os.makedirs(opt.experiment, exist_ok=True)


## random seed
# opt.seed = 42
# torch.manual_seed(opt.seed)
# np.random.seed(opt.seed)

## cudnn
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

## device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def test():
    ## dataset
    trans = T.Compose([T.Resize(opt.imageSize),  Style(2), T.ToTensor(), T.Normalize(0.5, 0.5)])
    test_dataset = load_dataset(opt.dataroot, opt.dataset, opt.imageSize, trans=trans, train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=False)
    opt.dataSize = test_dataset.__len__()

    ## model
    mlp = MLP(2*opt.imageSize**2, 1).to(device)
    assert mlp.load_state_dict(torch.load(opt.mlp_pth))
    print("Pretrained models have been loaded.")
    print([torch.max(i).item() for i in mlp.state_dict().values()])
    print([torch.min(i).item() for i in mlp.state_dict().values()])

    # loss
    mlp.eval()
    mlp_loss = []
    labels = []
    predictions = []
    SSIM = []
    evaluation = Evaluate(opt.experiment)
    tqdm_loader = tqdm.tqdm(test_dataloader)
    for i, (test_inputs0, test_inputs1, targets) in enumerate(tqdm_loader):
        tqdm_loader.set_description(f"Test Sample {i+1} / {opt.dataSize}")
        ## inference
        batchsize = test_inputs0.size(0)
        test_inputs = torch.cat([test_inputs0.view(batchsize, -1), test_inputs1.view(batchsize, -1)], dim=1)
        test_inputs = test_inputs.to(device)
        targets = targets.to(device)
        with torch.no_grad():
            outputs = mlp(test_inputs)
        outputs = list(torch.abs(outputs).view(-1).detach().cpu().numpy())
        # outputs = [j if j<1 else 1 for j in outputs]
        predictions.extend(outputs)
        targets = list(torch.abs(targets).detach().cpu().numpy().flatten())
        labels.extend(targets)

        for k in range(batchsize):
            img1 = test_inputs0[k].detach().cpu().numpy().reshape([opt.imageSize, opt.imageSize])
            img2 = test_inputs1[k].detach().cpu().numpy().reshape([opt.imageSize, opt.imageSize])
            SSIM.append(ssim(img1, img2))


        vutils.save_image(torch.cat([test_inputs0[:, :, :, :], test_inputs1[:, :, :, :]], dim=2),
                          '{0}/{1}-0.png'.format(opt.experiment, i), pad_value=2)
        vutils.save_image(torch.cat([test_inputs0[0:1, 0:1, :, :], test_inputs1[0:1, 0:1, :, :]], dim=2),
                          '{0}/{1}-example{2}.png'.format(opt.experiment, i, targets[0].item()), pad_value=2)



    # print(np.array(labels).reshape((-1, 8)))
    # print(np.array(predictions).reshape((-1, 8)))
    ssim_diff = np.sum(np.array(labels) * np.array(SSIM)) / len(np.where(np.array(labels)==1)[0])
    ssim_iden = np.sum((1-np.array(labels)) * np.array(SSIM)) / len(np.where(np.array(labels)==0)[0])
    print("ssim_iden: ", ssim_iden)
    print("ssim_diff: ", ssim_diff)
    evaluation.labels = labels
    evaluation.scores = np.array(predictions)
    return evaluation.run()


auc = []
F1_score = []
thre = []
for i in range(10):
    a = test()
    auc.append(a["auc"])
    F1_score.append(a["best_F1_score"])
    thre.append(a["best_thre"])
print(np.mean(np.array(auc)), np.std(np.array(auc)))
print(np.mean(np.array(F1_score)), np.std(np.array(F1_score)))
print(np.mean(np.array(thre)), np.std(np.array(F1_score)))