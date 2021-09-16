# Image2Point (Third Party)
Unoffical implementation of [Image2Point](https://arxiv.org/abs/2106.04180) paper.


## Requirements
Following dependencies are required:
- Pytorch (tested with v.1.8)
- MinkowskiEngine (tested with v.0.5)
  - Please follow the official installation instruction [here](https://github.com/NVIDIA/MinkowskiEngine).
  - The original paper uses [torchsparse](https://github.com/mit-han-lab/torchsparse), not MinkowsiEngine.
  - No worry, the behaviours of MinkowsiEngine and torchsparse should be the same.
- numpy
- lmdb
- msgpack-numpy

## Checklists
- [X] Map pretrained weights of ResNets to Sparse 3D-ResNets
- [X] ModelNet40 classification experiment


## Usage
Enter `python train_cls.py` to run ModelNet40 experiment. In the first run, the dataset would be automatically downloaded and prepared. 
If downloading goes wrong, you can manually download the zip file from 
[here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) and put the extracted files unter `./data`.

The code trains a Sparse 3D-ResNet-18 on ModelNet40, which is intialized with ImageNet pretrained weights of a ResNet-18. With the default setup,
only the input layer and output layer are trained. Also, BN layers are frozen. 

For other setups, please modify the training script (argparser would be added later).


## Known Issues
This code cannot reach the performance reported in the original paper on ModelNet40. This code gets ~54% top-1 accuray with Sparse 3D-ResNet-18, 
when only input and output layers are trained. But the paper gets 78.69%.

(Interestingly, with random initialized weight, the model reaches ~66%, if only input and output layers are trained)

More finetuning and debugging is needed. Any suggestions are welcome.


## Ackownledgement
- The code for ModelNet40 is borrowed from: https://github.com/erikwijmans/Pointnet2_PyTorch 
- All praise to authers of the original paper. Their offical repo: https://github.com/chenfengxu714/image2point
