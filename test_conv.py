# test the behaviour of MinkowsiConvolution

import MinkowskiEngine as ME
import torch
import numpy as np

coord = torch.IntTensor([[0, 0, 0], [0, 0, 1], [0, 0, 2], 
                         [0, 1, 0], [0, 1, 1], [0, 1, 2],
                         [0, 2, 0], [0, 2, 1], [0, 2, 2],
                         [1, 0, 0], [1, 0, 1], [1, 0, 2], 
                         [1, 1, 0], [1, 1, 1], [1, 1, 2],
                         [1, 2, 0], [1, 2, 1], [1, 2, 2],
                         [2, 0, 0], [2, 0, 1], [2, 0, 2], 
                         [2, 1, 0], [2, 1, 1], [2, 1, 2],
                         [2, 2, 0], [2, 2, 1], [2, 2, 2]])                         
feat = torch.arange(27).unsqueeze(1).float()
print(coord.shape)
c, f = ME.utils.sparse_collate([coord], [feat])
sin = ME.SparseTensor(f, c)

net = ME.MinkowskiConvolution(1, 1, kernel_size=3, dimension=3)
weight = torch.arange(27)
weight = weight.view(3,3,3).permute(2, 1, 0).contiguous()
net.kernel.data.copy_(weight.view(-1, 1, 1))


sout = net(sin)
print(sout.features_at_coordinates(torch.Tensor([[0, 1, 1, 1]])))

arr = np.arange(27)
print(np.sum(arr**2))

