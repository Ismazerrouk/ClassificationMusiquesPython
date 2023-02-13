from projet import *
import pandas as pd  # Pour le dataframe
import random
import librosa  # Pour l'extraction des features et la lecture des fichiers wav
import librosa.display  # Pour récupérer les spectrogrammes des audio
import librosa.feature
import os
from glob import glob

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import zero_one_loss, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
os.system("clear")
ed_path = edit_path()
#Menu
print(ed_path)
path = ed_path+"/Dataset"
print(path)
path_data = ed_path+"/Dataset_numpy"
path_plt = ed_path+"/plt"
genres = ['blues','classical','hiphop','jazz','rock']
check=0
while(check==0):
    choice = input(f"Veuillez choisir une méthode:\nMachine Learning (Random Forest et KNN):1\nDeep Learning (Réseau de neurones convolutifs:2\nQuiiter:0\n")
    if(choice=='1' or choice =='2'):
        if(choice=='1'):
            audioFiles = []

            for g in genres:
                data_dir = path + "/" +g
                audioFiles.append(glob(data_dir + '/*.wav'))
            print(audioFiles)
            audioLibrosa=[]

            for i in range(len(genres)):
                for j in range(30):
                    audio, sfreq = li.load(audioFiles[i][j])
                    audioLibrosa.append(audio)

            column_names = ['zcr', 'spectral_c', 'rolloff', 'mfcc1', 'mfcc2', 'mfcc3',
                'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9',
                'mfcc10', 'mfcc11', 'mfcc12', 'mfcc13', 'mfcc14', 'mfcc15',
                'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19', 'mfcc20', 'label']

            df = pd.DataFrame(columns=column_names)


            for i in range(0, 30):
                df.loc[i] = audio_pipeline(audioLibrosa[i]) + ['blues']

            for i in range(30, 60):
                df.loc[i] = audio_pipeline(audioLibrosa[i]) + ['classical']

            for i in range(60, 90):
                df.loc[i] = audio_pipeline(audioLibrosa[i]) + ['country']

            for i in range(90, 120):
                df.loc[i] = audio_pipeline(audioLibrosa[i]) + ['disco']

            for i in range(120, 150):
                df.loc[i] = audio_pipeline(audioLibrosa[i]) + ['hiphop']

            csv = df.to_csv('music.csv', index = False)  #exportation au format csv
            #tableau des variances
            selector = VarianceThreshold(threshold=(0.2))
            selected_features = selector.fit_transform(df[['zcr', 'spectral_c', 'rolloff', 'mfcc1', 'mfcc2', 'mfcc3',
                                              'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9',
                                              'mfcc10', 'mfcc11', 'mfcc12', 'mfcc13', 'mfcc14', 'mfcc15',
                                              'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19', 'mfcc20']])

            print(pd.DataFrame(selected_features))

            # matrice de corrélation

            print(df.corr())

            # f = plt.figure(figsize=(12, 12))
            # #ax = plt.gca()
            # plt.matshow(df.corr(), fignum=f.number)
            # plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=45)
            # plt.yticks(range(df.shape[1]), df.columns, fontsize=14)
            # #ax.tick_params(axis="x", bottom=True)
            # cb = plt.colorbar()
            # cb.ax.tick_params(labelsize=14)
            # plt.title('Matrice de corrélation', fontsize=16, y=-0.15)
            # ############### RANDOM FOREST #######################

            #features = pd.read_csv('music.csv')
            features = df
            # valeurs à prédire
            labels = np.array(features['label'])
            # supprime les labels des données
            features = features.drop('label', axis = 1)
            # sauvegarde le nom de features
            feature_list = list(features.columns)
            # conversion en numpy array
            features = np.array(features)

            # séparer les données en training and testing sets
            train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 0)
            print('Training Features Shape:', train_features.shape)
            print('Training Labels Shape:', train_labels.shape)
            print('Testing Features Shape:', test_features.shape)
            print('Testing Labels Shape:', test_labels.shape)

            sc = StandardScaler()
            train_features = sc.fit_transform(train_features)
            test_features = sc.transform(test_features)

            # création du modèle
            rf = RandomForestClassifier(n_estimators=4000, max_features='sqrt', max_depth=20, min_samples_split=2, min_samples_leaf=1, bootstrap=True, criterion='gini' ,random_state=0)

            # fit le modèle
            rf.fit(train_features, train_labels)

            # prédictions
            predictions = rf.predict(test_features)

            # Zero_one_loss error
            errors = zero_one_loss(test_labels, predictions, normalize=False)
            print('zero_one_loss error :', errors)

            # Accuracy Score
            accuracy_test = accuracy_score(test_labels, predictions)
            print('accuracy_score on test dataset :', accuracy_test)

            print(classification_report(predictions, test_labels))

            ############### KNN #######################
            X = df[['zcr', 'spectral_c', 'rolloff', 'mfcc1', 'mfcc2', 'mfcc3',
           'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9',
           'mfcc10', 'mfcc11', 'mfcc12', 'mfcc13', 'mfcc14', 'mfcc15',
           'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19', 'mfcc20']]

            y = df['label']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

            random.seed(100)

            model_1 = KNeighborsClassifier(n_neighbors = 6)  #### On commence avec 6 voisins

            model_1.fit(X_train, y_train)
            print("\n Score du KNN pour 6 voisins")
            print('Train score :', model_1.score(X_train,y_train))
            print('Test score :', model_1.score(X_test,y_test))

###### Graphique afin de determiner le meilleur nombre de voisins
#
# k_1 = np.arange(1, 31)
# train_score_1, val_score_1 = validation_curve(model_1, X_train, y_train, param_name='n_neighbors', param_range=k_1, cv = 5)
#
# #5 splits sets de cross validation, puis on fait la moyenne des scores obtenus sur chacun des 5 splits
#
# plt.plot(k_1, val_score_1.mean(axis = 1), label = 'validation')
# plt.plot(k_1, train_score_1.mean(axis = 1), label = 'train')
#
# plt.ylabel('score')
# plt.xlabel('n_neighbors')
# plt.legend()
        else:
            choix = input("Voulez-vous générer le dataset ?\nOui=1\nNon=0 (Si vous êtes le professeur)")
            if(choix=='1'):
                length = int(input("Veuillez entrer le nombre de musique par valeur que vous souhaitez:")) 
                gen_data(path,path_data,genres,length)

                os.system("clear")
                #Récupération de notre tableau numpy contenant les données des musiques
                audio_files = np.load(path_data+"/audio_tab.npy",mmap_mode=None,allow_pickle=True)

                #Récupération des Spectogrammes à l'aide de Librosa (Image de taille 128x660)
                X = Spectogram(audio_files,genres)
                x_train, x_test, y_train, y_test = labels_creation(X,genres)

                #Normalisation des données
                x_train /= np.min(x_train)
                x_test /= np.min(x_train)

                #Redimensionnement de notre Dataset
                x_train = x_train.reshape(x_train.shape[0], 128, 660, 1)
                x_test = x_test.reshape(x_test.shape[0], 128, 660, 1)

                #Sauvegarde de notre Dataset
                np.save(path_data+"/X_train.npy",x_train)
                np.save(path_data+"/X_test.npy",x_test)
                np.save(path_data+"/Y_train.npy",y_train)
                np.save(path_data+"/Y_test.npy",y_test)

                #Application de notre réseau de neurones convolutif
                loss_curve, acc_curve, loss_val_curve, acc_val_curve = network_neurons(x_train,x_test,y_train,y_test,len(genres),genres,path_plt)
                plt.plot(loss_curve, label="Train")
                plt.plot(loss_val_curve, label="Test")
                plt.legend(loc='upper left')
                plt.title("Fonction coût")
                plt.savefig(path+"/Loss.png")
                plt.show()

                plt.plot(acc_curve, label="Train")
                plt.plot(acc_val_curve, label="Test")
                plt.legend(loc='upper left')
                plt.title("Précision")
                plt.savefig(path+"/Acc.png")
                plt.show()
                check=1
            else:
                if(choix=='0'):
                    os.system("clear")

                    #Récupération de notre Dataset
                    x_train = np.load(path_data+"/X_train.npy")
                    x_test = np.load(path_data+"/X_test.npy")
                    y_train = np.load(path_data+"/Y_train.npy")
                    y_test = np.load(path_data+"/Y_test.npy")
                    print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
            
                    #Application de notre réseau de neurones convolutifs
                    loss_curve, acc_curve, loss_val_curve, acc_val_curve = network_neurons(x_train,x_test,y_train,y_test,len(genres),genres,path_plt)
                    plt.plot(loss_curve, label="Train")
                    plt.plot(loss_val_curve, label="Test")
                    plt.legend(loc='upper left')
                    plt.title("Fonction coût")
                    plt.savefig(path_plt+"/Loss.png")
                    plt.show()

                    plt.plot(acc_curve, label="Train")
                    plt.plot(acc_val_curve, label="Test")
                    plt.legend(loc='upper left')
                    plt.title("Précision")
                    plt.savefig(path_plt+"/Acc.png")
                    plt.show()
        check=1
    else:
        if(choice=='0'):
            check=1
        else:
            print(f'Vous avez entré un mauvais caractère.\n Veuillez recommencer:')


"""
while(check==0):
    choice = input("Voulez-vous générer le dataset ? (Oui=1) (Non=0 Si vous êtes le professeur)")
    if(choice=='1'):
        length = int(input("Veuillez entrer le nombre de musique par valeur que vous souhaitez:")) 
        gen_data(path,path_data,genres,length)

        os.system("clear")
        #Récupération de notre tableau numpy contenant les données des musiques
        audio_files = np.load(path_data+"/audio_tab.npy",mmap_mode=None,allow_pickle=True)

        #Récupération des Spectogrammes à l'aide de Librosa (Image de taille 128x660)
        X = Spectogram(audio_files,genres)
        x_train, x_test, y_train, y_test = labels_creation(X,genres)

        #Normalisation des données
        x_train /= np.min(x_train)
        x_test /= np.min(x_train)

        #Redimensionnement de notre Dataset
        x_train = x_train.reshape(x_train.shape[0], 128, 660, 1)
        x_test = x_test.reshape(x_test.shape[0], 128, 660, 1)

        #Sauvegarde de notre Dataset
        np.save(path_data+"/X_train.npy",x_train)
        np.save(path_data+"/X_test.npy",x_test)
        np.save(path_data+"/Y_train.npy",y_train)
        np.save(path_data+"/Y_test.npy",y_test)

        #Application de notre réseau de neurones convolutif
        loss_curve, acc_curve, loss_val_curve, acc_val_curve = network_neurons(x_train,x_test,y_train,y_test,len(genres),genres,path_plt)
        plt.plot(loss_curve, label="Train")
        plt.plot(loss_val_curve, label="Test")
        plt.legend(loc='upper left')
        plt.title("Fonction coût")
        plt.savefig(path+"/Loss.png")
        plt.show()

        plt.plot(acc_curve, label="Train")
        plt.plot(acc_val_curve, label="Test")
        plt.legend(loc='upper left')
        plt.title("Précision")
        plt.savefig(path+"/Acc.png")
        plt.show()
        check=1
    else:
        if(choice=='0'):
            os.system("clear")

            #Récupération de notre Dataset
            x_train = np.load(path_data+"/X_train.npy")
            x_test = np.load(path_data+"/X_test.npy")
            y_train = np.load(path_data+"/Y_train.npy")
            y_test = np.load(path_data+"/Y_test.npy")
            print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
            
            #Application de notre réseau de neurones convolutifs
            loss_curve, acc_curve, loss_val_curve, acc_val_curve = network_neurons(x_train,x_test,y_train,y_test,len(genres),genres,path_plt)
            plt.plot(loss_curve, label="Train")
            plt.plot(loss_val_curve, label="Test")
            plt.legend(loc='upper left')
            plt.title("Fonction coût")
            plt.savefig(path_plt+"/Loss.png")
            plt.show()

            plt.plot(acc_curve, label="Train")
            plt.plot(acc_val_curve, label="Test")
            plt.legend(loc='upper left')
            plt.title("Précision")
            plt.savefig(path_plt+"/Acc.png")
            plt.show()
            check=1
        else:
            print("Vous avez entré un caractère qui ne correspond ni à 0 ou à 1.\n Veuillez recommencez:")
"""