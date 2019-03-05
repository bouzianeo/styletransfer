import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = StyleLoss.gram_matrix(target_feature).detach()

    def forward(self, input):
        G = StyleLoss.gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

    @staticmethod
    def gram_matrix(input):
        """
        Compute grammian matrix.
        Allows to compare style and content images in the same basis (compute mse loss in that basis)
        """
        a, b, c, d = input.size()  # Image tensor size, a = 1 car image RGB, b

        features = input.view(a * b, c * d)  # resize input into \hat input

        G = torch.mm(features, features.t())  # Multiply tensors

        return G.div(a * b * c * d)  # matrix mean


class Normalization(nn.Module):
    """
    Class used to normalize all images : normalized img = (img - mean) / std_dev
    Default values are taken from the paper
    """
    def __init__(self, mean=[0.485, 0.456, 0.406], std_dev=[0.229, 0.224, 0.225]):
        super(Normalization, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mean = torch.tensor(mean).view(-1, 1, 1).to(self.device)
        self.std_dev = torch.tensor(std_dev).view(-1, 1, 1).to(self.device)

    def forward(self, img):
        return (img - self.mean) / self.std_dev
