# styletransfer

Projet commun de Deep Learning et de Vision par Ordinateur de l'option ISIA de l'école CentraleSupelec

## Installation 

1. Installez le fichier de requirements 
``` 
pip install -r requirements.txt 
```

2. Lancer le fichier main
Le fichier de contenu doit etre dans le dossier images/content et le fichier de style dans images/style
```
python main.py content_image style_image
```
Par exemple,
```
python main.py centraleext.jpg picasso.jpg 
```
