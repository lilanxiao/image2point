import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.optim import SGD, lr_scheduler
import MinkowskiEngine as ME
import numpy as np
from torchvision import transforms
from itertools import chain
from typing import List

from data.modelnet40 import ModelNet40Cls
from data import data_utils as d_utils
from proj_para import project_resnet, CONVBLOCKS
from resnet import resnet18
from srnet import sparse_resnet18

# TODO: other training setups.
# TODO: cannot reproduce results in paper. need debugging.
# TODO: add argparser

PRETRAIN = True             # use ImageNet pretrain weight
NUM_POINTS = 10000          # number of input points
BATCH_SIZE = 32             # batch size
VOXEL_SIZE = 0.05           # voxel size
LR = 0.1                    # initial learning rate
EPOCHS = 300                # epochs of training
DROPOUT = 0.                # dropout rate
TRAIN_WHOLE_NET = False     # if True, train whole net. If False, only train input output layers
UPDATE_BN_STAT = False      # update BN mean and variance of not.

class SparseResNetCls(nn.Module):
    def __init__(self, dim_in, dim_out, num_class, dropout_rate, backbone, source=None):
        super().__init__()
        self.backbone = backbone(dim_in)
        self.head = cls_head(dim_out, num_class, dropout_rate)
        if source is not None:
            print("Project ImageNet weights of ResNet to Sparse ResNet \n")
            project_resnet(source(True), self.backbone)
        else:
            print("Sparse ResNet random initialized \n")
    
    def forward(self, x):
        y = self.backbone(x)
        y = self.head(y)
        return y


def freeze_bn(net, modules:List[str]=CONVBLOCKS):
    for m in modules:
        getattr(net.backbone, m).eval()


def cls_head(dim_in:int, num_class:int, dropout_rate:float=0.5):
    return  nn.Sequential(
        ME.MinkowskiGlobalAvgPooling(),
        ME.MinkowskiLinear(dim_in, 1024),
        ME.MinkowskiBatchNorm(1024),
        ME.MinkowskiReLU(True),
        ME.MinkowskiDropout(dropout_rate),
        ME.MinkowskiLinear(1024, num_class)
    )


def cls_loss(pred:Tensor, labels:Tensor):
    num = pred.size(0)
    crit = nn.CrossEntropyLoss()
    loss = crit(pred, labels)
    choice = torch.argmax(pred, 1)
    correct = torch.sum((choice == labels).float()).item()
    return loss, correct, num
    

def create_collate_fn(voxel_size:float):
    def collate_fn(data_list):
        coords, feats, labels = [], [], []
        for pts, label in data_list:
            if pts.shape[1] == 3:
                f = np.ones((len(pts), 1))
                c = pts
            else:
                f = pts[:, 3:]
                c = pts[:, :3]
            c, f = ME.utils.sparse_quantize(pts, f, quantization_size=voxel_size)
            coords.append(c)
            feats.append(f)
            labels.append(label)
        coords_t, feats_t = ME.utils.sparse_collate(coords=coords, feats=feats)
        labels = torch.from_numpy(np.stack(labels)).long()
        return coords_t, feats_t.float(), labels
    return collate_fn


def main():
    t_train = transforms.Compose([
            d_utils.PointcloudToTensor(),
            d_utils.PointcloudScale(),
            d_utils.PointcloudRotate(),
            d_utils.PointcloudRotatePerturbation(),
            d_utils.PointcloudTranslate(),
            d_utils.PointcloudJitter(),
        ])
    t_val = transforms.Compose([d_utils.PointcloudToTensor()])
    
    ds_train = ModelNet40Cls(NUM_POINTS, transforms=t_train, train=True)
    ds_val = ModelNet40Cls(NUM_POINTS, transforms=t_val, train=False)
    loader_train = DataLoader(
        ds_train, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4, 
        collate_fn=create_collate_fn(VOXEL_SIZE)
    )
    loader_val = DataLoader(
        ds_val,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=create_collate_fn(VOXEL_SIZE)
    )
    
    source = resnet18 if PRETRAIN else None
    net = SparseResNetCls(1, 512, 40, DROPOUT, sparse_resnet18, source).cuda()
    
    if TRAIN_WHOLE_NET:
        trained_weights = net.parameters()
    else:
        # only train input and output layer
        trained_weights = chain(
            net.backbone.input_layer.parameters(),
            net.head.parameters()
        )
    
    optimizer = SGD(trained_weights, lr=LR, momentum=0.9, weight_decay=1e-4)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
    
    best_var_acc = 0
    for epoch in range(EPOCHS):
        # train
        net.train()
        if not UPDATE_BN_STAT and not TRAIN_WHOLE_NET:
            print("... freeze Batch Norm")
            freeze_bn(net)
        correct = 0
        total = 0
        for i, data in enumerate(loader_train):
            coords, feats, labels = data
            labels = labels.cuda()
            sin = ME.SparseTensor(feats.cuda(), coords.cuda())
            optimizer.zero_grad()
            sout = net(sin)
            loss, c, t = cls_loss(sout.F, labels)
            loss.backward()
            optimizer.step()
            correct += c
            total += t
            if i % 20 == 0:
                torch.cuda.empty_cache()  # avoid OOM
                print("Ep. {:03d} [{:d}/{:d}] loss: {:.4f}".format(
                    epoch+1,
                    i,
                    len(loader_train),
                    loss.item()
                ))
        print("Ep. {:03d} train Acc. {:.4f} ".format(epoch+1, correct/total))
        scheduler.step()
        
        # eval
        net.eval()
        correct = 0
        total = 0
        print("... validating")
        for i, data in enumerate(loader_val):
            coords, feats, labels = data
            labels = labels.cuda()
            with torch.no_grad():
                sin = ME.SparseTensor(feats.cuda(), coords.cuda())
                sout = net(sin)
                loss, c, t = cls_loss(sout.F, labels)
            correct += c
            total += t
        torch.cuda.empty_cache()  # avoid OOM
        acc = correct/total
        best_var_acc = max(acc, best_var_acc)
        print("Ep. {:03d} val Acc. {:.4f}. (Best: {:.4f})".format(epoch+1, acc, best_var_acc))
        print("-"*20)


if __name__ == "__main__":
    main()
    