import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentLoss(nn.Module):
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input_image):
        """
        Computes Content Loss between an input image and a target image (content image)
        :param input_image: Image to compute content loss on
        """
        self.loss = F.mse_loss(input_image, self.target)
        return input_image


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = StyleLoss.gram_matrix(target_feature).detach()

    def forward(self, input_image):
        """
        Computes Style Loss between an input image and a target image (style image)
        :param input_image: Image to compute style loss on
        """
        G = StyleLoss.gram_matrix(input_image)
        self.loss = F.mse_loss(G, self.target)
        return input_image

    @staticmethod
    def gram_matrix(input_image):
        """
        Compute grammian matrix.
        Allows to compare style and content images in the same basis (compute mse loss in that basis)
        """
        a, b, c, d = input_image.size()  # Image tensor size, a = 1 car image RGB

        features = input_image.view(a * b, c * d)  # resize input into \hat input

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
