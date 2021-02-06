# -*- coding:utf8 -*-
# @TIME     : 2020/12/24 17:08
# @Author   : SuHao
# @File     : siamese_MLP_test.py

import os
import tqdm
import torch
import numpy as np
import cv2
import argparse
import torch.nn as nn
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torchvision.transforms as T
from deeplearning.models.siamese_MLP import SiameseNetwork, ContrastiveLoss
from deeplearning.dataload.dataload import load_dataset
from deeplearning.test.evaluate import Evaluate
from deeplearning.dataload.self_transforms import Fllip, AddGaussianNoise, AddSaltPepperNoise, Style


parser = argparse.ArgumentParser()
parser.add_argument("--experiment", default=r"../experiments/siamese_MLP_test", help="path to save experiments results")
parser.add_argument("--dataset", default="mnist_siamese", help="mnist")
parser.add_argument('--dataroot', default=r"../data", help='path to dataset')
parser.add_argument("--batchSize", type=int, default=256, help="size of the batches")
parser.add_argument("--size", type=int, default=28, help="size of image after scaled")
parser.add_argument("--imageSize", type=int, default=28, help="size of each image dimension")
parser.add_argument("--nc", type=int, default=1, help="number of image channels")
parser.add_argument("--siamese_pth", default=r"../experiments/siamese_MLP_train/siamese.pth", help="pretrained model")
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
    trans = T.Compose([T.Resize(opt.imageSize), Style(2), T.ToTensor(), T.Normalize((0.5), (0.5))])
    test_dataset = load_dataset(opt.dataroot, opt.dataset, opt.imageSize, trans=trans, train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=False)
    opt.dataSize = test_dataset.__len__()

    ## model
    siamese = SiameseNetwork(opt.imageSize).to(device)
    assert siamese.load_state_dict(torch.load(opt.siamese_pth))
    print("Pretrained models have been loaded.")
    print([torch.max(i).item() for i in siamese.state_dict().values()])
    print([torch.min(i).item() for i in siamese.state_dict().values()])

    # loss
    siamese_criteria = ContrastiveLoss()
    siamese.eval()
    siamese_loss = []
    labels = []
    predictions = []
    evaluation = Evaluate(opt.experiment)
    tqdm_loader = tqdm.tqdm(test_dataloader)
    for i, (test_inputs0, test_inputs1, targets) in enumerate(tqdm_loader):
        tqdm_loader.set_description(f"Test Sample {i+1} / {opt.dataSize}")
        ## inference
        test_inputs0, test_inputs1, targets = test_inputs0.to(device), test_inputs1.to(device), targets.to(device)
        with torch.no_grad():
            outputs0, outputs1 = siamese(test_inputs0, test_inputs1)
        euclidean_distance = F.pairwise_distance(outputs0, outputs1)
        predictions.extend(list(euclidean_distance.detach().cpu().numpy()))
        labels.extend(list(torch.abs(targets).detach().cpu().numpy()))

        vutils.save_image(torch.cat([test_inputs0[:, :, :, :], test_inputs1[:, :, :, :]], dim=2),
                          '{0}/{1}-0.png'.format(opt.experiment, i), pad_value=2)
        # print(targets.detach().cpu().numpy().reshape([-1, 8]))
        # print(euclidean_distance.detach().cpu().numpy().reshape([-1, 8]))


    # print(np.array(labels).reshape((-1, 8)))
    # print(np.array(predictions).reshape((-1, 8)))

    evaluation.labels = labels
    predictions = np.array(predictions)
    predictions = (predictions - np.min(predictions)) / (np.max(predictions) - np.min(predictions))
    evaluation.scores = predictions
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