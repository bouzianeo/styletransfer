from Loader import image_loader, imsave
from VGG import VGG16
import os


if __name__ == "__main__":
    # contents = os.listdir("images/content")[1:]
    # styles = os.listdir("images/style")[1:]
    # for content in contents:
    #     for style in styles:
    #         print("Content image : {} and style image : {}".format(content, style))
    #         style_img, content_img = image_loader(style, content)
    #
    #         vggdream = VGG16(style_img, content_img)
    #
    #         output = vggdream(content_img, num_steps=300, style_weight=10000, content_weight=1)  # style_weight=100000, content_weight=10
    #         imsave(output, style[:-4] + "-" + content)

    content = 'centraleext.jpg'
    style = 'picasso.jpg'
    style_img, content_img = image_loader(style, content)

    assert style_img.size() == content_img.size(), \
        "You have to to import style and content images of the same size"

    vggdream = VGG16(style_img, content_img)
    style_weights = {1, 10, 100, 1000, 10000, 100000}
    content_weights = {1, 10, 100, 1000, 10000, 100000}
    for style_weight in style_weights:
        for content_weight in content_weights:
            print("Content weigh : {} and style weight : {}".format(content_weight, style_weight))
            output = vggdream(content_img, num_steps=300, style_weight=style_weight, content_weight=content_weight)
            imsave(output, str(style_weight) + "_" + str(content_weight) + "_" + style)
