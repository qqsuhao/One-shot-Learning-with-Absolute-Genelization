# -*- coding:utf8 -*-
# @TIME     : 2021/1/4 9:36
# @Author   : SuHao
# @File     : siamese_shot_test.py

import numpy as np
import torch
import argparse
import os
import tqdm
import torch.nn as nn
from torch import optim
from tensorboardX import SummaryWriter
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from deeplearning.dataload.dataload import get_test_loader
from deeplearning.test.evaluate import Evaluate
from deeplearning.models.siamese_shot import Siamese
from deeplearning.dataload.self_transforms import Fllip, AddSaltPepperNoise, AddGaussianNoise, Style


parser = argparse.ArgumentParser()
parser.add_argument("--experiment", default=r"../experiments/siamese_shot_test", help="path to save experiments results")
parser.add_argument('--dataroot', default=r"../data/images_evaluation", help='path to dataset')
parser.add_argument("--way", type=int, default=5, help="how much way one-shot learning")
parser.add_argument("--trials", type=int, default=400, help="number of samples to test accuracy")
parser.add_argument("--workers", type=int, default=0, help="number of dataLoader workers")
parser.add_argument("--siamese_pth", default=r"../experiments/siamese_shot_train/siamese.pth", help="pretrained model")
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
    trans = T.Compose([Fllip(),  T.ToTensor(), ])
    test_dataloader = get_test_loader(opt.dataroot,
                                      way=opt.way,
                                      trials=opt.trials,
                                      seed=0,
                                      num_workers=opt.workers,
                                      trans=trans,)

    ## model
    siamese = Siamese().to(device)
    assert siamese.load_state_dict(torch.load(opt.siamese_pth))
    print("Pretrained models have been loaded.")
    print([torch.max(i).item() for i in siamese.state_dict().values()])
    print([torch.min(i).item() for i in siamese.state_dict().values()])

    ## loss
    siamese_criteria = nn.BCEWithLogitsLoss(reduction='mean')
    siamese.eval()
    right = 0
    error = 0
    tqdm_loader = tqdm.tqdm(test_dataloader)
    for i, (test_inputs0, test_inputs1) in enumerate(tqdm_loader):
        test_inputs0, test_inputs1 = test_inputs0.to(device), test_inputs1.to(device)
        outputs = siamese(test_inputs0, test_inputs1)
        outputs = outputs.detach().cpu().numpy()
        pred = np.argmax(outputs)
        if pred == 0:
            right += 1
        else:
            error += 1
        # print('[%d]\tTest set\tcorrect:\t%d\terror:\t%d\tprecision:\t%f'%(i, right, error, right*1.0/(right+error)))


        vutils.save_image(torch.cat([test_inputs0, test_inputs1], dim=2),
                          '{0}/{1}-0.png'.format(opt.experiment, i), pad_value=0, nrow=10)
    print(right*1.0/(right+error))
    return right*1.0/(right+error)

accu = []
for i in range(10):
    accu.append(test())
print(np.mean(np.array(accu)), np.std(np.array(accu)))

