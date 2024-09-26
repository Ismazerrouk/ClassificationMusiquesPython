# **Classification des Genres Musicaux - Machine Learning et Deep Learning**

## **Description du projet**
Ce projet a pour objectif de classifier des genres musicaux à partir de fichiers audios en utilisant des approches de **Machine Learning** (Random Forest et KNN) et **Deep Learning** (réseau de neurones convolutifs). Il utilise des fichiers audio au format `.wav` et extrait des caractéristiques des fichiers audios pour réaliser la classification.

### **Méthodes utilisées :**
1. **Machine Learning :**
   - **Random Forest** : Un classificateur utilisant des arbres de décision pour prédire les genres musicaux.
   - **K-Nearest Neighbors (KNN)** : Un modèle qui utilise la distance entre les données pour classer les genres.
   
2. **Deep Learning :**
   - **Réseau de neurones convolutifs (CNN)** : Un réseau pour extraire et analyser les spectrogrammes audios.

### **Genres musicaux :**
Les genres utilisés pour l'entraînement et la classification dans ce projet sont :
- Blues
- Classical
- Hip-hop
- Jazz
- Rock

## **Prérequis**

### **Dépendances :**
- Python 3.x
- **Librairies Python** :
  - `librosa` : Pour extraire les features et travailler avec des fichiers audio.
  - `pandas` : Pour la manipulation des données.
  - `sklearn` : Pour les modèles de machine learning et la normalisation des données.
  - `matplotlib` : Pour la visualisation des résultats (matrices de corrélation et autres graphiques).
  - `numpy` : Pour la manipulation des tableaux et des matrices.
  - `random` : Pour la génération aléatoire.
  
Tu peux installer toutes les dépendances en exécutant la commande suivante :

```bash
pip install librosa pandas scikit-learn matplotlib numpy
```

## **Structure du projet**

- **Dataset** : Ce répertoire contient les fichiers audio classés par genres.
- **Dataset_numpy** : Ce répertoire contient les fichiers traités en format numpy pour le modèle de deep learning.
- **Plt** : Ce répertoire contient les graphiques générés pendant l'entraînement.

## **Utilisation**

### **1. Choix du modèle :**
Lors de l'exécution du script, l'utilisateur est invité à choisir entre :
- **Machine Learning (1)** : Utilise Random Forest et KNN pour la classification.
- **Deep Learning (2)** : Utilise un réseau de neurones convolutifs (CNN) pour analyser les spectrogrammes audios.

### **2. Extraction des features audio :**
Les fichiers `.wav` sont analysés avec **Librosa** pour extraire des caractéristiques telles que :
- Zero-crossing rate (zcr)
- Spectral centroid
- Rolloff
- MFCC (coefficients cepstraux fréquentiels de Mel)

Ces features sont utilisées pour l'entraînement des modèles.

### **3. Entraînement des modèles :**
- **Random Forest** et **KNN** sont utilisés pour entraîner un modèle basé sur les features extraites.
- **Réseau de neurones convolutifs (CNN)** est utilisé pour analyser les spectrogrammes audios.

Les résultats du modèle (taux d'erreur, précision, score d'entraînement/test) sont affichés à l'écran après l'exécution.

### **4. Sauvegarde des résultats :**
Les résultats (comme les matrices de confusion, courbes de coût et de précision) sont sauvegardés sous forme d'images dans le répertoire `plt`.

## **Instructions d'exécution**

1. Clonez ce dépôt sur votre machine locale :

```bash
git clone <url-de-votre-projet-github>
```

2. Accédez au dossier du projet :

```bash
cd <nom-du-dossier>
```

3. Exécutez le script Python :

```bash
python main.py
```

4. Suivez les instructions à l'écran pour choisir le modèle et générer le dataset.

## **Résultats**

- **Machine Learning :** Les modèles Random Forest et KNN affichent des résultats variés en fonction des genres musicaux. Les scores d'exactitude et les taux d'erreur sont présentés après l'entraînement.
- **Deep Learning :** Le réseau de neurones convolutifs produit des courbes de coût et de précision, permettant d'observer les performances de l'entraînement.

## **Visualisations**
Les matrices de corrélation et les courbes de précision sont générées et sauvegardées dans le répertoire `plt`.
