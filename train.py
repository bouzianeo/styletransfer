from Loader import image_loader, imshow
import torch
import matplotlib.pyplot as plt
from VGG import VGG16


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    style_img = image_loader("images/kandinsky.jpg")
    content_img = image_loader("images/souki.jpg")
    input_img = content_img.clone()

    assert style_img.size() == content_img.size(), \
        "You have to to import style and content images of the same size"

    imshow(style_img, title='Style Image')
    imshow(content_img, title='Content Image')

    vggdream = VGG16()

    output = vggdream.run_style_transfer(content_img, style_img, input_img, num_steps=300,
                           style_weight=1000000, content_weight=1)

    plt.figure()
    imshow(output, title='Output Image', save=True)

    plt.ioff()
    plt.show()