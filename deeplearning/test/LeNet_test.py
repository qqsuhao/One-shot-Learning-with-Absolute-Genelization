# -*- coding:utf8 -*-
# @TIME     : 2021/1/11 21:56
# @Author   : SuHao
# @File     : LeNet_test.py


from __future__ import print_function
import os
import tqdm
import torch
from torch.utils.data import DataLoader
from deeplearning.models.LeNet import LeNet
from deeplearning.dataload.dataload import load_dataset
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
import torchvision.transforms as T
from deeplearning.test.evaluate import Evaluate
from deeplearning.dataload.self_transforms import Fllip, AddGaussianNoise, AddSaltPepperNoise, Style


parser = argparse.ArgumentParser()
parser.add_argument("--experiment", default=r"../experiments/LeNet_test", help="path to save experiments results")
parser.add_argument("--dataset", default="mnist", help="mnist")
parser.add_argument('--dataroot', default=r"../data", help='path to dataset')
parser.add_argument("--batchSize", type=int, default=256, help="size of the batches")
parser.add_argument("--imageSize", type=int, default=28, help="size of each image dimension")
parser.add_argument("--nc", type=int, default=1, help="number of image channels")
parser.add_argument("--cnn_pth", default=r"../experiments/lenet_train/lenet.pth", help="pretrained model")
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
    trans = T.Compose([T.Resize((opt.imageSize, opt.imageSize)), Style(2), T.ToTensor(), T.Normalize(0.5, 0.5)])
    test_dataset = load_dataset(opt.dataroot, opt.dataset, opt.imageSize, trans=trans, train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=False)
    opt.dataSize = test_dataset.__len__()

    ## model
    lenet = LeNet().to(device)
    assert lenet.load_state_dict(torch.load(opt.cnn_pth))
    print("Pretrained models have been loaded.")
    print([torch.max(i).item() for i in lenet.state_dict().values()])
    print([torch.min(i).item() for i in lenet.state_dict().values()])


    # loss
    lenet_criteria = nn.CrossEntropyLoss()
    lenet.eval()
    lenet_loss = []
    labels = []
    predictions = []
    evaluation = Evaluate(opt.experiment)
    tqdm_loader = tqdm.tqdm(test_dataloader)
    for i, (test_inputs, targets) in enumerate(tqdm_loader):
        tqdm_loader.set_description(f"Test Sample {i+1} / {opt.dataSize}")
        ## inference
        test_inputs, targets = test_inputs.to(device), targets.to(device)
        batchsize = test_inputs.size(0)
        with torch.no_grad():
            outputs = lenet(test_inputs)
        outputs = list(torch.softmax(outputs, 1)[:, 1].cpu().numpy())
        predictions.extend(outputs)
        labels.extend(list(torch.abs(targets).view(-1).detach().cpu().numpy()))


        vutils.save_image(test_inputs, '{0}/{1}-0.png'.format(opt.experiment, i), pad_value=2)

    print(labels)
    print(predictions)
    evaluation.labels = labels
    predictions = np.array(predictions)
    predictions = (predictions - np.min(predictions)) / (np.max(predictions) - np.min(predictions))
    evaluation.scores = predictions
    return evaluation.run()


test()