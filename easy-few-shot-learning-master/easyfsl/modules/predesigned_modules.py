from torch import nn
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.vgg import make_layers
from easyfsl.modules import VGG
from easyfsl.modules import ResNet

__all__ = [
    "resnet10",
    "resnet12",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "default_matching_networks_support_encoder",
    "default_matching_networks_query_encoder",
    "default_relation_module",
    "default_CNN",
    "vgg11_bn",
    "vgg13_bn",
    "vgg16_bn",
    "vgg19_bn",
]

from easyfsl.modules.vgg import cfg


def resnet10(**kwargs):
    """Constructs a ResNet-10 model."""
    return ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)


def resnet12(**kwargs):
    """Constructs a ResNet-12 model."""
    return ResNet(BasicBlock, [1, 1, 2, 1], planes=[64, 160, 320, 640], **kwargs)


def resnet18(**kwargs):
    """Constructs a ResNet-18 model."""
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    """Constructs a ResNet-34 model."""
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    """Constructs a ResNet-50 model."""
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    """Constructs a ResNet-101 model."""
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152(**kwargs):
    """Constructs a ResNet-152 model."""
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)


def default_matching_networks_support_encoder(feature_dimension: int) -> nn.Module:
    return nn.LSTM(
        input_size=feature_dimension,
        hidden_size=feature_dimension,
        num_layers=1,
        batch_first=True,
        bidirectional=True,
    )


def default_matching_networks_query_encoder(feature_dimension: int) -> nn.Module:
    return nn.LSTMCell(feature_dimension * 2, feature_dimension)


def default_relation_module(feature_dimension: int, inner_channels: int = 3):
    """
    Build the relation module that takes as input the concatenation of two feature maps, from
    Sung et al. : "Learning to compare: Relation network for few-shot learning." (2018)
    In order to make the network robust to any change in the dimensions of the input images,
    we made some changes to the architecture defined in the original implementation
    from Sung et al.(typically the use of adaptive pooling).
    Args:
        feature_dimension: the dimension of the feature space i.e. size of a feature vector
        inner_channels: number of hidden channels between the linear layers of  the relation module
    Returns:
        the constructed relation module
    """
    return nn.Sequential(
        nn.Sequential(
            nn.Conv2d(
                3,
                feature_dimension,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(feature_dimension, momentum=1, affine=True),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((5, 5)),
        ),
        nn.Sequential(
            nn.Conv2d(
                feature_dimension,
                feature_dimension,
                kernel_size=3,
                padding=0,
            ),
            nn.BatchNorm2d(feature_dimension, momentum=1, affine=True),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, 1)),
        ),

        # nn.Flatten(),
        # nn.Linear(feature_dimension, inner_channels),
        # nn.ReLU(),
        # nn.Linear(inner_channels, 3),
        # nn.Sigmoid(),
    )


def default_CNN():
    return nn.Sequential(
        nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=11, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((3, 3)),
        ),
        nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((3, 3)),
        ),
        nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            # nn.BatchNorm2d(feature_dimension, momentum=1, affine=True),
            nn.ReLU(),
            # nn.AdaptiveMaxPool2d((2, 2)),
        ),
        # nn.Flatten(),
        nn.Linear(128, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
        nn.Sigmoid(),
    )


def vgg11_bn():
    return VGG(make_layers(cfg['A'], batch_norm=True))


def vgg13_bn():
    return VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16_bn():
    return VGG(make_layers(cfg['D'], batch_norm=True))


def vgg19_bn():
    return VGG(make_layers(cfg['E'], batch_norm=True))