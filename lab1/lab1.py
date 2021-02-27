#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob

import matplotlib.pyplot as plot
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from PIL import Image, ImageOps


POKEMON_PATH = './pokemon'
POKEMON_PROCESSED_PATH = './pokemon_processed'


def plot_gallery(images, titles, file_name, h, w, n_col=4):
    plot.clf()
    n_row = len(images) // n_col + (len(images) % n_col != 0)
    plot.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plot.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        if i >= len(images):
            break
        plot.subplot(n_row, n_col, i + 1)
        plot.imshow(images[i].reshape((h, w)), cmap=plot.cm.gray)
        plot.title(titles[i], size=12)
        plot.xticks(())
        plot.yticks(())
    plot.savefig(file_name)


def get_pokemon(h, w):
    """
    Processing data. Resize pokemon images to size h * w and convert to gray
    scale. For the convenience of the future, save preprocessed images to the
    other folder.
    """
    os.makedirs(POKEMON_PROCESSED_PATH, exist_ok=True)
    pokemon_names, pokemon_paths = [], []

    for pokemon_name in sorted(os.listdir(POKEMON_PATH)):
        folder = os.path.join(POKEMON_PATH, pokemon_name)
        if not os.path.isdir(folder):
            continue

        # make new dir for the pokemon
        new_folder = os.path.join(POKEMON_PROCESSED_PATH, pokemon_name)
        os.makedirs(new_folder, exist_ok=True)

        # get path of images with file extension png
        paths_png = glob.glob(os.path.join(folder, '*.png'))
        paths_jpg = glob.glob(os.path.join(folder, '*.jpg'))
        paths = paths_png + paths_jpg
        # iterate over existing pokemon's pictures and process each one
        for i, path in enumerate(sorted(paths)):
            img = Image.open(path)
            # TODO: Checkpoint 1, Preprocessing
            # 2. Convert RGB image to grayscale
            # 1. Resize image into h x w
            ####
            '''
            img = img.convert(...)
            img = ImageOps.fit(...)
            '''
            new_path = os.path.join(
                POKEMON_PROCESSED_PATH, pokemon_name, '%d.jpg' % i)
            img.save(new_path)

    # read pokemon names and paths
    for pokemon_name in sorted(os.listdir(POKEMON_PROCESSED_PATH)):
        folder = os.path.join(POKEMON_PROCESSED_PATH, pokemon_name)
        if not os.path.isdir(folder):
            continue
        paths = [os.path.join(folder, f) for f in sorted(os.listdir(folder))]
        pokemon_names.extend([pokemon_name for _ in range(len(paths))])
        pokemon_paths.extend(paths)

    # convert string label to numerical value
    n_pokemon = len(pokemon_paths)
    target_names = list(np.unique(pokemon_names))
    target = np.searchsorted(target_names, pokemon_names)

    # read data
    pokemons = []
    for i, pokemon_path in enumerate(pokemon_paths):
        img = Image.open(pokemon_path)
        pokemon = np.asarray(img, dtype=np.float32)
        pokemons.append(pokemon)
    pokemons = np.array(pokemons)

    # shuffle pokemon
    indices = np.arange(n_pokemon)
    np.random.RandomState(42).shuffle(indices)
    pokemons, target = pokemons[indices], target[indices]
    pokemons = pokemons.reshape(len(pokemons), -1)

    return pokemons, target, target_names


def main():
    np.random.seed(42)
    height, width = 200, 200
    pokemons, target, target_names = get_pokemon(height, width)

    X = pokemons.reshape(len(pokemons), -1)
    y = target
    precisions = []
    recalls = []
    for train_index, test_index in KFold(n_splits=4, shuffle=True).split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # TODO: Checkpoint 4, Apply PCA on training set before SVM
        # 1. Set the number of eigenvectors
        # 2. View eigenpokemons
        ####
        '''
        n_components = ???
        pca = PCA(n_components=n_components, whiten=True).fit(X_train)
        eigenpokemons_titles = [
            "eigenpokemon %d" % i
            for i in range(pca.components_.shape[0])]
        plot_gallery(pca.components_, eigenpokemons_titles, "PCA", height, width)
        print("Projecting the input data on the eigenpokemon orthonormal basis")
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)
        '''

        # TODO: Checkpoint 3, Train an SVM classification model
        # 1. Select appropriate parameter for GridSearchCV
        ####
        '''
        print("Fitting SVM to the training set")
        param_grid = {
                'kernel': ['rbf', 'linear'],
                'C': [???],
                'gamma': [???],
        }
        clf = GridSearchCV(SVC(class_weight='balanced'), param_grid, cv=3)
        clf = clf.fit(X_train, y_train)
        print('Best params', clf.best_params_)
        print("Evaluation SVM quality on the test set")
        y_pred = clf.predict(X_test)
        print("SVM Report")
        print(classification_report(y_test, y_pred, target_names=target_names))

        print(''.join('%10s' % name for name in ([" "] + target_names)))
        for idx, arr in enumerate(confusion_matrix(y_test, y_pred)):
            print('%10s' % target_names[idx], end='')
            for v in arr:
                print('%10s' % str(v), end='')
            print('')
        '''

        # TODO: Checkpoint 2, Train an KNN classification model
        # 1. Select appropriate paramter for GridSearchCV
        print("Fitting KNN to the training set")
        param_grid = {
            'n_neighbors': [1]
        }
        clf = GridSearchCV(KNeighborsClassifier(), param_grid, cv=3)
        clf = clf.fit(X_train, y_train)
        print('Best params', clf.best_params_)
        print("Evaluation KNN quality on the test set")
        y_pred = clf.predict(X_test)
        print("KNN Report")
        print(classification_report(y_test, y_pred, target_names=target_names))

        print(''.join('%10s' % name for name in ([" "] + target_names)))
        for idx, arr in enumerate(confusion_matrix(y_test, y_pred)):
            print('%10s' % target_names[idx], end='')
            for v in arr:
                print('%10s' % str(v), end='')
            print('')

        precision = precision_score(y_test, y_pred, average='weighted')
        precisions.append(precision)
        recall = recall_score(y_test, y_pred, average='weighted')
        recalls.append(recall)
        print('-' * 80)

    print('KFold average')
    print('precision: %.4f' % np.mean(precisions))
    print('recall   : %.4f' % np.mean(recalls))


if __name__ == "__main__":
    main()
