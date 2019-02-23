import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms



def image_loader(device, image_name):
    imsize = 512 if torch.cuda.is_available() else 128
    loader = transforms.Compose([transforms.Resize(imsize), transforms.ToTensor()])
    ratio = 4/3
    image = Image.open(image_name)
    w, h = image.size
    if w/h <= ratio:
        h = w/ratio
    else:
        w = h*ratio
    image=image.crop((0, 0, w, h))
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def imshow(tensor, title=None, save=False):
    plt.figure()
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    if save:
        image.save("output.jpg", "JPEG", optimize=True)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)