from Loader import image_loader, imshow, imsave
from VGG import VGG16


if __name__ == "__main__":
    style_img = image_loader("images/picasso.jpg")
    content_img = image_loader("images/marca.jpg")
    # input_img = content_img.clone()

    assert style_img.size() == content_img.size(), \
        "You have to to import style and content images of the same size"

    vggdream = VGG16(style_img, content_img)

    output = vggdream(content_img, num_steps=300, style_weight=100000, content_weight=1)  # style_weight=100000, content_weight=10

    imsave(output)

    # TODO : initialiser chaque couche toute seule dans l'init
    # TODO : Reecrire le VGG a la main