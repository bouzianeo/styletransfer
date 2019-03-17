import torch
from PIL import Image
import torchvision.transforms as transforms
import os


def image_loader(style_path, content_path):
    """
    Load style and content images
    :param style_path: name of the style image
    :param content_path: name of the content image
    :return: style and content image in cv2 format
    """
    style_path = os.path.join("images/style", style_path)
    content_path = os.path.join("images/content", content_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imsize = 1024 if torch.cuda.is_available() else 128
    loader = transforms.Compose([transforms.Resize(imsize), transforms.ToTensor()])

    content_image = Image.open(content_path)
    w, h = content_image.size
    ratio = w / h
    style_image = Image.open(style_path)
    w, h = style_image.size
    if w/h <= ratio:
        h = w/ratio
    else:
        w = h*ratio
    style_image = style_image.crop((0, 0, w, h))
    style_image = loader(style_image).unsqueeze(0)
    content_image = loader(content_image).unsqueeze(0)
    return style_image.to(device, torch.float), content_image.to(device, torch.float)


def imsave(tensor, name):
    """
    Transform a PyTorch tensor into a pil image and save it
    :param tensor: Image tensor
    :param name: Name of the saved image
    """
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    image.save("images/results/"+name, "JPEG", optimize=True)
