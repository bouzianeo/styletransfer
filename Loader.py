import torch
from PIL import Image
import torchvision.transforms as transforms
import os


def image_loader(style_path, content_path):
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


# def imshow(tensor, title=None, save=False):
#     plt.figure()
#     unloader = transforms.ToPILImage()
#     image = tensor.cpu().clone()
#     image = image.squeeze(0)
#     image = unloader(image)
#     if save:
#         image.save("output.jpg", "JPEG", optimize=True)
#     plt.imshow(image)
#     if title is not None:
#         plt.title(title)
#     plt.pause(0.001)


def imsave(tensor, name):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    image.save("images/results/"+name, "JPEG", optimize=True)