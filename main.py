from Loader import image_loader, imsave
from VGG import VGG16
import os


if __name__ == "__main__":
    files = os.listdir("images/content")[1:]
    for file in files:
        print("Processing image : ", file)
        style_img, content_img = image_loader("picasso.jpg", file)
        # input_img = content_img.clone()

        assert style_img.size() == content_img.size(), \
            "You have to to import style and content images of the same size"

        vggdream = VGG16(style_img, content_img)

        output = vggdream(content_img, num_steps=100, style_weight=100000, content_weight=1)  # style_weight=100000, content_weight=10

        imsave(output, file)

        # TODO : initialiser chaque couche toute seule dans l'init
        # TODO : Reecrire le VGG a la main