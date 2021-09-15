import torch
from torch.nn import Module, Conv2d
from typing import List

# only the convolutional backbone
STEM = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]
# backbone with classification head
FULL = STEM + ["fc"]
# only conv blocks without input layer
CONVBLOCKS = ["layer1", "layer2", "layer3", "layer4"]
# basic block
BASIC = ["conv1", "bn1", "conv2", "bn2", "downsample"]
# bottle neck block
BOTTLENECK = BASIC + ["conv3", "bn3"]
# down sample with conv
DOWNSAMPLE = ["conv1", "bn1"]


@torch.no_grad()
def copy_resnet(source:Module, target:Module, modules:List[str]) -> None:
    """copy a resnet's parameters to annother.
    validate concept for weight projection"""
    for n in modules:
        copy_module(getattr(source, n), getattr(target, n), n)


@torch.no_grad()
def copy_module(source:Module, target:Module, name:str)->None:
    if "conv" in name or "fc" in name:
        target.weight.data.copy_(source.weight.data)
        if target.bias is not None:
            target.bias.data.copy_(source.bias.data)
    elif "bn" in name:
        target.weight.data.copy_(source.weight.data)
        target.bias.data.copy_(source.bias.data)
        target.running_mean.data.copy_(source.running_mean.data)
        target.running_var.data.copy_(source.running_var.data)
    elif "block" in name:
        if source.__class__.__name__ == "BasicBlock":
            for n in BASIC:
                copy_module(getattr(source, n), getattr(target, n), n)
        elif source.__class__.__name__ == "Bottleneck":
            for n in BOTTLENECK:
                copy_module(getattr(source, n), getattr(target, n), n)
        else:
            raise NotImplementedError    
    elif "downsample" in name:
        if source is not None and target is not None:
            copy_module(source[0], target[0], "conv")
            copy_module(source[1], target[1], "bn")
    elif "layer" in name:
        length = len(source)
        for i in range(length):
            copy_module(source[i], target[i], "block")
    else:
        raise NotImplementedError


@torch.no_grad()
def project_resnet(source:Module, target:Module, modules:List[str]=CONVBLOCKS):
    """project weights of a ResNet to a Sparse ResNet"""
    for n in modules:
        project_module(getattr(source, n), getattr(target, n), n)


@torch.no_grad()
def project_module(source:Module, target:Module, name:str) -> None:
    """project weight of a ResNet's module to Sparse ResNet.

    Args:
        source (Module): a module in ResNet
        target (Module): a module in Sparse ResNet
        name (str): name of the module
    """
    if "conv" in name:
        project_conv_weights(source, target)
        # copy bias
        if source.bias is not None and target.bias is not None:
            target.bias.data.copy_(source.bias.data)
    elif "bn" in name:
        target.bn.weight.data.copy_(source.weight.data)
        target.bn.bias.data.copy_(source.bias.data)
        target.bn.running_mean.data.copy_(source.running_mean.data)
        target.bn.running_var.data.copy_(source.running_var.data)
    elif "block" in name:
        if source.__class__.__name__ == "BasicBlock":
            for n in BASIC:
                project_module(getattr(source, n), getattr(target, n), n)
        elif source.__class__.__name__ == "Bottleneck":
            for n in BOTTLENECK:
                project_module(getattr(source, n), getattr(target, n), n)
        else:
            raise NotImplementedError    
    elif "downsample" in name:
        if source is not None and target is not None:
            project_module(source[0], target[0], "conv")
            project_module(source[1], target[1], "bn")
    elif "layer" in name:
        length = len(source)
        for i in range(length):
            project_module(source[i], target[i], "block")
    else:
        raise NotImplementedError


@torch.no_grad()
def project_conv_weights(source:Conv2d, target):
    """
    weight of torch.nn.Con2d:   dim_out, dim_in, k, k
    kernel of 3D Mink.Conv:     k**3, dim_in, dim_out
    """
    weight = source.weight.data                     # (dim_out, dim_in, k, k)
    dim_out, dim_in, k, _ = weight.size()
    temp = torch.clone(weight)
    # NOTE: kernel spatial dimension in ME: z,y,x
    temp = temp.permute(3, 2, 1, 0).contiguous()    # (k, k, dim_in, dim_out)
    temp = temp.unsqueeze(0)                        # (1, k, k, dim_in, dim_out)
    temp = temp.repeat(k, 1, 1, 1, 1)               # (k, k, k, dim_in, dim_out)
    # copy to Sparse Conv kernel
    if k > 1:
        target.kernel.data.copy_(temp.view(-1, dim_in, dim_out).contiguous())
    else:
        target.kernel.data.copy_(temp.view(dim_in, dim_out).contiguous())


def test_copy_resnet(resnet):
    net1 = resnet(True)
    net2 = resnet(False)
    
    copy_resnet(net1, net2, FULL)
    x = torch.rand(2, 3, 224, 224)
    y1 = net1(x)
    y2 = net2(x)
    error = torch.max(y1 - y2).item()
    if error < 1e-6:
        print("pass")
    else:
        print("fail")


def test_project_resnet(resnet:Module, sparse_resnet:Module):
    project_resnet(resnet, sparse_resnet, CONVBLOCKS)
    print("ok")


if __name__ == "__main__":
    from resnet import resnet18, resnet50
    from srnet import sparse_resnet18, sparse_resnet50
    test_copy_resnet(resnet18)
    test_copy_resnet(resnet50)
    test_project_resnet(resnet18(True), sparse_resnet18(1))
    test_project_resnet(resnet50(True), sparse_resnet50(1))
    