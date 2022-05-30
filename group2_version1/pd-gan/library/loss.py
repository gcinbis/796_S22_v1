import torch
import torchvision
import torch.nn as nn

class VGG19(torch.nn.Module):
    def __init__(self):
        """
        VGG-19 convolutional neural network
        """
        super().__init__()
        # load pretrained model weights
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True, progress=False).features

        # activation maps from layers ReLU1 1, ReLU2 1, ReLU3 1,
        # ReLU4 1, and ReLU5 1.
        module_1, module_2, module_3, module_4, module_5 = [], [], [], [], []
        for x in range(2):
            module_1.append(vgg_pretrained_features[x])
        for x in range(2, 7):
            module_2.append(vgg_pretrained_features[x])
        for x in range(7, 12):
            module_3.append(vgg_pretrained_features[x])
        for x in range(12, 21):
            module_4.append(vgg_pretrained_features[x])
        for x in range(21, 30):
            module_5.append(vgg_pretrained_features[x])

        # not record operations on these tensors
        for module in module_1:
            for parameter in module.parameters():
                parameter.requires_grad = False
        for module in module_2:
            for parameter in module.parameters():
                parameter.requires_grad = False
        for module in module_3:
            for parameter in module.parameters():
                parameter.requires_grad = False
        for module in module_4:
            for parameter in module.parameters():
                parameter.requires_grad = False
        for module in module_5:
            for parameter in module.parameters():
                parameter.requires_grad = False

        # Make Sequential
        self.relu_1 = nn.Sequential(*module_1)
        self.relu_2 = nn.Sequential(*module_2)
        self.relu_3 = nn.Sequential(*module_3)
        self.relu_4 = nn.Sequential(*module_4)
        self.relu_5 = nn.Sequential(*module_5)

    def forward(self, x):
        """
        Calculate activation operation of tensors
        :param x: generated image
        :return: layer outputs
        """
        out_relu_1 = self.relu_1(x)
        out_relu_2 = self.relu_2(out_relu_1)
        out_relu_3 = self.relu_3(out_relu_2)
        out_relu_4 = self.relu_4(out_relu_3)
        out_relu_5 = self.relu_5(out_relu_4)
        return out_relu_1, out_relu_2, out_relu_3, out_relu_4, out_relu_5


def calculate_feature_matching_loss(real, pred):
    """
    Calculate Feature Matching Loss
    :param real: real image prediction of discriminator
    :param pred: fake image prediction of discriminator
    :return: FeatureMatching loss
    """
    criterion = torch.nn.L1Loss()
    loss = criterion(real, pred)
    return loss


def calculate_perceptual_loss(x_1, x_2, device='cpu'):
    """
    Calculate Perceptual Loss (Equation 6)
    :param x_1: First generated image
    :param x_2: Second generated image
    :param device: cpu/cuda training device
    :return: perceptual diversity loss
    """
    eps = 0.00001
    vgg = VGG19().to(device)
    criterion = torch.nn.L1Loss()

    vgg_1_1, vgg_1_2, vgg_1_3, vgg_1_4, vgg_1_5 = vgg(x_1)
    vgg_2_1, vgg_2_2, vgg_2_3, vgg_2_4, vgg_2_5 = vgg(x_2)
    p_loss = 0

    # sum layer outputs
    p_loss += criterion(vgg_1_1, vgg_2_1)
    p_loss += criterion(vgg_1_2, vgg_2_2)
    p_loss += criterion(vgg_1_3, vgg_2_3)
    p_loss += criterion(vgg_1_4, vgg_2_4)
    p_loss += criterion(vgg_1_5, vgg_2_5)

    return 1 / (p_loss.clone().detach() + eps)


def calculate_reconstruction_loss(x_1, x_2):
    """
    Reconstruction Loss
    :param x_1: First image
    :param x_2: Second image
    :return:Reconstruction Loss
    """
    criterion = torch.nn.L1Loss()
    loss = criterion(x_1, x_2)
    return loss
