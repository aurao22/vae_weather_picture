<div style="display: flex; background-color: RGB(255,114,0); padding: 30px;" >

# Weather
</div>
by : Ellande, Vincent et Aurélie

<div style="display: flex; background-color: Blue; padding: 15px;" >

## Mission 
</div>

Une appli internationale de météo vous engage pour vérifier s'il est possible de détecter et classer de manière **non supervisée** la catégorie météo d'un jeu d'images (sachant que la détectation de manière supervisée avec CNN peuve fournir de très bons résultats). Voici les 4 catégories :
- rain 🌧, 
- cloudy ☁️, 
- sunshine ☀️, 
- sunrise 🌅.

Vous décidez de réaliser un pré-traitement d'images de type SIFT / ORB / SURF de génération de descripteurs et de "bag of visual words" afin de créer un histogramme par image. Vous réaliserez une réduction de dimension de type T-SNE et afficherez un graphique T-SNE des images selon leur catégories.

Vous vérifierez ainsi la faisabilité de séparer les images météo selon le type de météo. Avant de tester avec toutes les images à la disposition de l’appli, vous décidez de tester la faisabilité sur un dataset de 400 images déjà catégorisées. Vous en avez 100 par catégorie.


## Code

[vae_weather.ipynb](vae_weather.ipynb)
