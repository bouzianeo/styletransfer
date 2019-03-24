from Loader import image_loader, imsave
from VGG import VGG16
import os


if __name__ == "__main__":
    contents = [f for f in os.listdir("images/content") if not f.startswith('.')]
    styles = [f for f in os.listdir("images/style") if not f.startswith('.')]
    for content in contents:
        for style in styles:
            print("Content image : {} and style image : {}".format(content, style))
            style_img, content_img = image_loader(style, content)

            vggdream = VGG16(style_img, content_img)

            output = vggdream(content_img, num_steps=200, style_weight=1000000, content_weight=1)  # style_weight=100000, content_weight=10
            imsave(output, style[:-4] + "-" + content)



    ########################################
    #               Grid Search            #
    ########################################
    #
    # content = 'centraleext.jpg'
    # style = 'picasso.jpg'
    # style_img, content_img = image_loader(style, content)
    #
    # assert style_img.size() == content_img.size(), \
    #     "You have to to import style and content images of the same size"
    #
    # vggdream = VGG16(style_img, content_img)
    # style_weights = {50000, 500000, 1000000, 10000000, 100000000}
    # content_weights = {1}
    # for style_weight in style_weights:
    #     for content_weight in content_weights:
    #         print("Content weigh : {} and style weight : {}".format(content_weight, style_weight))
    #         output = vggdream(content_img, num_steps=300, style_weight=style_weight, content_weight=content_weight)
    #         imsave(output, str(style_weight) + "_" + str(content_weight) + "_" + style)

    # content = 'centraleext.jpg'
    # style = 'picasso.jpg'
    # style_img, content_img = image_loader(style, content)
    #
    # assert style_img.size() == content_img.size(), \
    #     "You have to to import style and content images of the same size"
    #
    # vggdream = VGG16(style_img, content_img)
    # num_steps={50, 100, 200, 300, 500, 1000}
    # for num_step in num_steps :
    #         output = vggdream(content_img, num_steps=num_step, style_weight=1000000, content_weight=1)
    #         imsave(output, str(num_step) + "_" + style)
