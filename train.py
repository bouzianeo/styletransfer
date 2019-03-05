from Loader import image_loader, imshow, imsave
from VGG import VGG16


if __name__ == "__main__":
    style_img = image_loader("images/kandinsky.jpg")
    content_img = image_loader("images/souki.jpg")
    input_img = content_img.clone()

    assert style_img.size() == content_img.size(), \
        "You have to to import style and content images of the same size"

    # imshow(style_img, title='Style Image')
    # imshow(content_img, title='Content Image')

    #builder = VGG16()
    #vggdream = builder.build(style_img, content_img)
    vggdream = VGG16(style_img, content_img)

    output = vggdream(content_img, num_steps=300, style_weight=1000000, content_weight=1)

    imsave(output)

    # TODO : initialiser chaque couche toute seule dans l'init
    # TODO : Reecrire le VGG a la main