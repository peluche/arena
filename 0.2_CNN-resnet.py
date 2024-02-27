# %%
import os
import sys
import numpy as np
import einops
from typing import Union, Optional, Tuple, List, Dict
import torch as t
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
import functools
from pathlib import Path
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from tqdm.notebook import tqdm
from dataclasses import dataclass
from PIL import Image
import json

# Make sure exercises are in the path
chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part2_cnns"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow, line, bar
import part2_cnns.tests as tests
from part2_cnns.utils import print_param_count

# device = t.device("cuda" if t.cuda.is_available() else "cpu")
device = t.device("mps" if t.backends.mps.is_available() else "cpu")

# %%

class ReLU(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        # return t.maximum(x, t.tensor(0.0))
        return x.max(t.zeros_like(x))


tests.test_relu(ReLU)


# %%
class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        '''
        A simple linear (technically, affine) transformation.

        The fields should be named `weight` and `bias` for compatibility with PyTorch.
        If `bias` is False, set `self.bias` to None.
        '''
        super().__init__()
        n = in_features**-0.5
        self.weight = nn.Parameter(t.zeros([out_features, in_features]).uniform_(-n, n))
        self.bias = nn.Parameter(t.zeros([out_features]).uniform_(-n, n)) if bias else None
        self.is_bias = bias

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (*, in_features)
        Return: shape (*, out_features)
        '''
        res = x @ self.weight.T
        return res if self.bias is None else res + self.bias

    def extra_repr(self) -> str:
        return f'Weight: {self.weight.shape=}'


tests.test_linear_forward(Linear)
tests.test_linear_parameters(Linear)
tests.test_linear_no_bias(Linear)


# %%
class Flatten(nn.Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: t.Tensor) -> t.Tensor:
        '''
        Flatten out dimensions from start_dim to end_dim, inclusive of both.
        '''
        s = input.shape #(x,y,z) -> (x, y*z)
        sd = self.start_dim
        ed = self.end_dim + 1
        if self.start_dim < 0:
            sd = len(s) + sd
        if self.end_dim < 0:
            ed = len(s) + ed
        p = 1
        for x in s[sd:ed]:
            p *= x
        new_shape = list(s[:sd]) + [p] + list(s[ed:])
        return t.reshape(input, new_shape)

    def extra_repr(self) -> str:
        return f'Flatten: {self.start_dim=} {self.end_dim=}'


tests.test_flatten(Flatten)


# %%

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = Flatten(1, -1)
        self.layer0 = Linear(784, 100, True)
        self.relu = ReLU()
        self.layer1 = Linear(100, 10, True)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.layer1(self.relu(self.layer0(self.flatten(x))))


tests.test_mlp(SimpleMLP)


# %%
MNIST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def get_mnist(subset: int = 1):
    '''Returns MNIST training data, sampled by the frequency given in `subset`.'''
    mnist_trainset = datasets.MNIST(root="./data", train=True, download=True, transform=MNIST_TRANSFORM)
    mnist_testset = datasets.MNIST(root="./data", train=False, download=True, transform=MNIST_TRANSFORM)

    if subset > 1:
        mnist_trainset = Subset(mnist_trainset, indices=range(0, len(mnist_trainset), subset))
        mnist_testset = Subset(mnist_testset, indices=range(0, len(mnist_testset), subset))

    return mnist_trainset, mnist_testset


mnist_trainset, mnist_testset = get_mnist()
mnist_trainloader = DataLoader(mnist_trainset, batch_size=64, shuffle=True)
mnist_testloader = DataLoader(mnist_testset, batch_size=64, shuffle=False)

# %%

@dataclass
class SimpleMLPTrainingArgs():
    '''
    Defining this class implicitly creates an __init__ method, which sets arguments as 
    given below, e.g. self.batch_size = 64. Any of these arguments can also be overridden
    when you create an instance, e.g. args = SimpleMLPTrainingArgs(batch_size=128).
    '''
    batch_size: int = 64
    epochs: int = 3
    learning_rate: float = 1e-3
    subset: int = 10

@t.inference_mode()
def validate(model: SimpleMLP, loader: DataLoader):
    batch_loss = 0
    batch_size = 0
    for imgs, labels in loader:
      imgs = imgs.to(device)
      labels = labels.to(device)
      logits = model(imgs)
      loss = (logits.argmax(dim=-1) == labels).sum()
      batch_loss += loss
      batch_size += imgs.shape[0]
    return (batch_loss/batch_size).item()


def train(args: SimpleMLPTrainingArgs):
    '''
    Trains the model, using training parameters from the `args` object.
    '''
    model = SimpleMLP().to(device)

    mnist_trainset, mnist_testset = get_mnist(subset=args.subset)
    mnist_trainloader = DataLoader(mnist_trainset, batch_size=args.batch_size, shuffle=True)
    mnist_testloader = DataLoader(mnist_testset, batch_size=args.batch_size, shuffle=True)

    optimizer = t.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_list = []
    validation_loss = []
    validation_loss.append(validate(model, mnist_testloader))
    for epoch in tqdm(range(args.epochs)):
        for imgs, labels in mnist_trainloader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_list.append(loss.item())  
        validation_loss.append(validate(model, mnist_testloader))

    line(
        loss_list, 
        yaxis_range=[0, max(loss_list) + 0.1],
        labels={"x": "Num batches seen", "y": "Cross entropy loss"}, 
        title="SimpleMLP training on MNIST",
        width=700
    )

    line(
        validation_loss, 
        yaxis_range=[0, max(validation_loss) + 0.1],
        labels={"x": "Num batches seen", "y": "Validation loss"}, 
        title="SimpleMLP Validation loss training on MNIST",
        width=700
    )


args = SimpleMLPTrainingArgs()
train(args)


# %%

class Conv2d(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0
    ):
        '''
        Same as torch.nn.Conv2d with bias=False.

        Name your weight field `self.weight` for compatibility with the PyTorch version.
        '''
        super().__init__()
        print(f'Stride: {stride=}')
        if stride <= 0:  
          print("AAAAAAAAAAAAA")
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        # TODO kernel
        n_in = in_channels*kernel_size*kernel_size
        n_out = out_channels*kernel_size*kernel_size
        n = (6**0.5)/ (n_in+n_out+1)**0.5
        self.kernel = t.zeros([out_channels, in_channels, kernel_size, kernel_size]).uniform_(-n, n)
        self.weight = nn.Parameter(self.kernel)

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Apply the functional conv2d, which you can import.'''
        if self.stride <= 0:
            print("AAAAAAAAAAAAA:)))))))")
            print(self.extra_repr())
        return nn.functional.conv2d(input=x, weight=self.weight, stride=self.stride, padding=self.padding, bias=None)

    def extra_repr(self) -> str:
        return f'Conv2d: {self.in_channels=} {self.out_channels=} {self.stride=}'


tests.test_conv2d_module(Conv2d)
m = Conv2d(in_channels=24, out_channels=12, kernel_size=3, stride=2, padding=1)
print(f"Manually verify that this is an informative repr: {m}")


# %%

class MaxPool2d(nn.Module):
    def __init__(self, kernel_size: int, stride: Optional[int] = None, padding: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Call the functional version of max_pool2d.'''
        # result is not as smooth as for mean-pooling
        # return einops.reduce(x, 'b (h h2) (w w2) c -> h (b w) c', 'max', h2=self.kernel_size, w2=self.kernel_size)
        return nn.functional.max_pool2d(x, self.kernel_size, stride=self.stride, padding=self.padding)

    def extra_repr(self) -> str:
        '''Add additional information to the string representation of this class.'''
        return ", ".join([f"{key}={getattr(self, key)}" for key in ["kernel_size", "stride", "padding"]])


tests.test_maxpool2d_module(MaxPool2d)
m = MaxPool2d(kernel_size=3, stride=2, padding=1)
print(f"Manually verify that this is an informative repr: {m}")


# %%

class Sequential(nn.Module):
    _modules: Dict[str, nn.Module]

    def __init__(self, *modules: nn.Module):
        super().__init__()
        for index, mod in enumerate(modules):
            self._modules[str(index)] = mod

    def __getitem__(self, index: int) -> nn.Module:
        index %= len(self._modules) # deal with negative indices
        return self._modules[str(index)]

    def __setitem__(self, index: int, module: nn.Module) -> None:
        index %= len(self._modules) # deal with negative indices
        self._modules[str(index)] = module

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Chain each module together, with the output from one feeding into the next one.'''
        for mod in self._modules.values():
            x = mod(x)
        return x

# %%
class BatchNorm2d(nn.Module):
    # The type hints below aren't functional, they're just for documentation
    running_mean: Float[Tensor, "num_features"]
    running_var: Float[Tensor, "num_features"]
    num_batches_tracked: Int[Tensor, ""] # This is how we denote a scalar tensor

    def __init__(self, num_features: int, eps=1e-05, momentum=0.1):
        '''
        Like nn.BatchNorm2d with track_running_stats=True and affine=True.

        Name the learnable affine parameters `weight` and `bias` in that order.
        '''
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.register_buffer("running_mean", t.zeros(num_features))
        self.register_buffer("running_var", t.ones(num_features))
        self.register_buffer("num_batches_tracked", t.tensor(0))
        self.weight = nn.Parameter(t.ones(num_features))
        self.bias = nn.Parameter(t.zeros(num_features))

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        Normalize each channel.

        Compute the variance using `torch.var(x, unbiased=False)`
        Hint: you may also find it helpful to use the argument `keepdim`.

        x: shape (batch, channels, height, width)
        Return: shape (batch, channels, height, width)
        '''
        # running_mean <- (1 - momentum) * running_mean + momentum * new_mean
        m = self.running_mean
        v = self.running_var
        if self.training:
            m = x.mean(dim=(0,2,3))
            v = t.var(x, unbiased=False, dim=(0,2,3))
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * m
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * v
            self.num_batches_tracked += 1
        return ((x - m.view(1, -1, 1, 1))/(v.view(1, -1, 1, 1) + self.eps)**0.5)*self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)

    def extra_repr(self) -> str:
        '''Add additional information to the string representation of this class.'''
        return ", ".join([f"{key}={getattr(self, key)}" for key in ["num_features","eps", "momentum"]])


tests.test_batchnorm2d_module(BatchNorm2d)
tests.test_batchnorm2d_forward(BatchNorm2d)
tests.test_batchnorm2d_running_mean(BatchNorm2d)

# %%

class AveragePool(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, channels, height, width)
        Return: shape (batch, channels)
        '''
        return x.mean(dim=(2,3))


# %%
    
class ResidualBlock(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, first_stride=1):
        '''
        A single residual block with optional downsampling.

        For compatibility with the pretrained model, declare the left side branch first using a `Sequential`.

        If first_stride is > 1, this means the optional (conv + bn) should be present on the right branch. Declare it second using another `Sequential`.
        '''
        super().__init__()
        self.first_stride = first_stride
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.left = Sequential(
            Conv2d(in_channels=in_feats, out_channels=out_feats, stride=first_stride, kernel_size=3, padding=1),
            BatchNorm2d(num_features=out_feats), 
            ReLU(), 
            Conv2d(in_channels=out_feats, out_channels=out_feats, stride=1,kernel_size=3, padding=1),
            BatchNorm2d(num_features=out_feats))
        if (first_stride > 1):
          self.right = Sequential(
              Conv2d(in_channels=in_feats, out_channels=out_feats, stride=first_stride,kernel_size=1, padding=0),
              BatchNorm2d(num_features=out_feats)
          )
        else:
            self.right = nn.Identity()
        self.relu = ReLU()


    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        Compute the forward pass.

        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / stride, width / stride)

        If no downsampling block is present, the addition should just add the left branch's output to the input.
        '''
        return self.relu(self.left(x) + self.right(x))

# %%
class BlockGroup(nn.Module):
    def __init__(self, n_blocks: int, in_feats: int, out_feats: int, first_stride=1):
        '''An n_blocks-long sequence of ResidualBlock where only the first block uses the provided stride.'''
        super().__init__()
        self.n_blocks = n_blocks
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.first_stride = first_stride
        seq = [ResidualBlock(in_feats=in_feats, out_feats=out_feats, first_stride=first_stride)] + [ResidualBlock(in_feats=out_feats, out_feats=out_feats, first_stride=1) for x in range(n_blocks-1)]
        self.weight = Sequential(
            *seq
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        Compute the forward pass.

        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / first_stride, width / first_stride)
        '''
        return self.weight(x)

# %%

class ResNet34(nn.Module):
    def __init__(
        self,
        n_blocks_per_group=[3, 4, 6, 3],
        out_features_per_group=[64, 128, 256, 512],
        first_strides_per_group=[1, 2, 2, 2],
        n_classes=1000,
    ):
        super().__init__()
        self.n_blocks_per_group = n_blocks_per_group
        self.out_features_per_group = out_features_per_group
        self.first_strides_per_group = first_strides_per_group
        self.n_classes = n_classes
        head_seq = [Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3), BatchNorm2d(num_features=64), ReLU(), MaxPool2d(kernel_size=3, stride=2)]
        all_in_feats = [64] + out_features_per_group[:-1]
        in_features_per_group = [x//2 for x in out_features_per_group]
        block_grp_seq = [BlockGroup(*params) for params in zip(n_blocks_per_group, all_in_feats, out_features_per_group, first_strides_per_group)]
        self.head = Sequential(*head_seq)
        self.block_grp_seq = Sequential(*block_grp_seq)
        tail_seq = [AveragePool(), Flatten(1,-1), Linear(512,1000)]
        self.tail = Sequential(*tail_seq)

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, channels, height, width)
        Return: shape (batch, n_classes)
        '''
        x = self.head(x)
        x = self.block_grp_seq(x)
        return self.tail(x)


my_resnet = ResNet34()

# %%

from pprint import pprint as pp

def copy_weights(my_resnet: ResNet34, pretrained_resnet: models.resnet.ResNet) -> ResNet34:
    '''Copy over the weights of `pretrained_resnet` to your resnet.'''

    # Get the state dictionaries for each model, check they have the same number of parameters & buffers
    mydict = my_resnet.state_dict()
    pretraineddict = pretrained_resnet.state_dict()
    assert len(mydict) == len(pretraineddict), "Mismatching state dictionaries."

    # Define a dictionary mapping the names of your parameters / buffers to their values in the pretrained model
    state_dict_to_load = {
        mykey: pretrainedvalue
        for (mykey, myvalue), (pretrainedkey, pretrainedvalue) in zip(mydict.items(), pretraineddict.items())
    }

    # Load in this dictionary to your model
    my_resnet.load_state_dict(state_dict_to_load)

    return my_resnet


pretrained_resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
my_resnet = copy_weights(my_resnet, pretrained_resnet)
print("Success!")
# %%
IMAGE_FILENAMES = [
    "chimpanzee.jpg",
    "golden_retriever.jpg",
    "platypus.jpg",
    "frogs.jpg",
    "fireworks.jpg",
    "astronaut.jpg",
    "iguana.jpg",
    "volcano.jpg",
    "goofy.jpg",
    "dragonfly.jpg",
]

IMAGE_FOLDER = section_dir / "resnet_inputs"

images = [Image.open(IMAGE_FOLDER / filename) for filename in IMAGE_FILENAMES]

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGENET_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

prepared_images = t.stack([IMAGENET_TRANSFORM(img) for img in images], dim=0)

assert prepared_images.shape == (len(images), 3, IMAGE_SIZE, IMAGE_SIZE)

# %%

def predict(model, images: t.Tensor) -> t.Tensor:
    '''
    Returns the predicted class for each image (as a 1D array of ints).
    '''
    return t.argmax(model(images), dim=-1)


with open(section_dir / "imagenet_labels.json") as f:
    imagenet_labels = list(json.load(f).values())

# Check your predictions match those of the pretrained model
my_predictions = predict(my_resnet, prepared_images)
pretrained_predictions = predict(pretrained_resnet, prepared_images)
assert all(my_predictions == pretrained_predictions)
print("All predictions match!")

# Print out your predictions, next to the corresponding images
for img, label in zip(images, my_predictions):
    print(f"Class {label}: {imagenet_labels[label]}")
    display(img)
    print()

# %%