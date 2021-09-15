"""
Sparse 3D ResNet, following 2D ResNet, based on MinkowskiEngine 0.5
For easier weight projection, the attribute names are exactly the same. 
 
Different:
 - ME doesn't suppored grouped Conv
 - Conv in ME uses attribute name "kernel" instead of "weight"
 - Sparse ResNet has no global pooling and classification head
 - special initializations in ResNet are removed
 - sparse ResNet use three conv layers as input layer, instead of a large conv
"""

import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
import MinkowskiEngine as ME
from MinkowskiEngine import SparseTensor


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'wide_resnet50_2', 'wide_resnet101_2']


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1):
    if groups > 1:
        raise NotImplementedError
    return ME.MinkowskiConvolution(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=3,
        stride=stride,
        dilation=dilation,
        dimension=3,
        bias=False
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1):
    return ME.MinkowskiConvolution(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=1,
        stride=stride,
        dimension=3,
        bias=False
    )


class BasicBlock(nn.Module):
    expansion: int = 1
    
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None        
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = ME.MinkowskiBatchNorm
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: SparseTensor) -> SparseTensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if groups != 1:
            raise ValueError('MinkowsiEngine only supports groups=1')
        if norm_layer is None:
            norm_layer = ME.MinkowskiBatchNorm
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SparseResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        dim_in: int, 
        skip_last_downsample: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(SparseResNet, self).__init__()
        if norm_layer is None:
            norm_layer = ME.MinkowskiBatchNorm
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.input_layer = nn.Sequential(
            ME.MinkowskiConvolution(dim_in, self.inplanes, kernel_size=3, dimension=3),
            norm_layer(self.inplanes),
            ME.MinkowskiReLU(True),
            ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=3, dimension=3),
            norm_layer(self.inplanes),
            ME.MinkowskiReLU(True),
            ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=3, dimension=3, stride=2),
            norm_layer(self.inplanes),
            ME.MinkowskiReLU(True)            
        )

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], 
                                       stride=1 if skip_last_downsample else 2,
                                       dilate=replace_stride_with_dilation[2])

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.input_layer(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # NOTE: no global pooling and fc
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)



def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    dim_in: int,
    **kwargs: Any
) -> SparseResNet:
    model = SparseResNet(block, layers, dim_in, **kwargs)
    return model


def sparse_resnet18(dim_in, **kwargs: Any) -> SparseResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    """
    return _resnet(BasicBlock, [2, 2, 2, 2], dim_in, **kwargs)


def sparse_resnet34(dim_in, **kwargs: Any) -> SparseResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    """
    return _resnet(BasicBlock, [3, 4, 6, 3], dim_in, **kwargs)


def sparse_resnet50(dim_in, **kwargs: Any) -> SparseResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    """
    return _resnet(Bottleneck, [3, 4, 6, 3], dim_in, **kwargs)


def sparse_resnet101(dim_in, **kwargs: Any) -> SparseResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    """
    return _resnet(Bottleneck, [3, 4, 23, 3], dim_in, **kwargs)


if __name__ == "__main__":
    import numpy as np
    
    bs = 4
    in_channels = 1
    qunatize_size = 0.05

    coord = np.random.rand(bs, 5000, 3).astype(np.float32)
    feat = np.random.rand(bs, 5000, in_channels).astype(np.float32)
    cl = []
    fl = []
    for i in range(bs):
        c, f = ME.utils.sparse_quantize(coord[i, ...], feat[i, ...], quantization_size=qunatize_size)
        cl.append(c)
        fl.append(f)
    cl_batch = torch.cat(cl, axis=0)
    # input coordinate, input features
    coord, feat = ME.utils.sparse_collate(cl, fl)
    print(coord.shape, feat.shape)   
    sin = ME.SparseTensor(feat, coord)
    
    net = sparse_resnet50(1, skip_last_downsample=True)
    
    sout = net(sin)
    print(sout.F.shape)
    
