from sklearn import datasets, preprocessing, svm, tree, linear_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score, silhouette_score, calinski_harabasz_score, rand_score, homogeneity_score, mutual_info_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, MeanShift, Birch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def knn_classifier(train, train_target, k):
    knn_v = KNeighborsClassifier(n_neighbors=k)
    knn_v = knn_v.fit(train, train_target)
    return knn_v


def svm_classifier(train, train_target):
    svm_v = svm.SVC()
    svm_v = svm_v.fit(train, train_target)
    return svm_v


def tree_classifier(train, train_target):
    tree_v = tree.DecisionTreeClassifier()
    tree_v = tree_v.fit(train, train_target)
    return tree_v


def log_classifier(train, train_target):
    log_v = linear_model.LogisticRegression()
    log_v = log_v.fit(train, train_target)
    return log_v


#""" Loading data
iris = datasets.load_iris()
wine = datasets.load_wine()
banknotes = datasets.fetch_openml(data_id=1462, parser='auto')

dictionary = {'iris': iris, 'wine': wine, 'banknotes': banknotes}
#"""


#""" Classification
for key, value in dictionary.items():
    target = preprocessing.LabelEncoder().fit_transform(value.target)
    data = preprocessing.StandardScaler().fit_transform(value.data)
    train, test, train_target, test_target = train_test_split(data, target, test_size=0.2)

    # iris
    knn_func = knn_classifier(train, train_target, 5)
    svm_func = svm_classifier(train, train_target)
    tree_func = tree_classifier(train, train_target)
    log_func = log_classifier(train, train_target)

    classification_dict = {'knn': knn_func, 'svm': svm_func, 'tree': tree_func, 'log': log_func}

    # Error table
    disp = ConfusionMatrixDisplay.from_estimator(knn_func, test, test_target, normalize='true')
    plt.title(f"{key} knn")
    disp2 = ConfusionMatrixDisplay.from_estimator(svm_func, test, test_target, normalize='true')
    plt.title(f"{key} svm")
    disp3 = ConfusionMatrixDisplay.from_estimator(tree_func, test, test_target, normalize='true')
    plt.title(f"{key} tree")
    disp4 = ConfusionMatrixDisplay.from_estimator(log_func, test, test_target, normalize='true')
    plt.title(f"{key} sgd")
    plt.show()

    # Classification score
    for name, func in classification_dict.items():
        print(f"{key} {name} accuracy: ", accuracy_score(test_target, func.predict(test)))
        print(f"{key} {name} precision: ", precision_score(test_target, func.predict(test), average='macro'))
        print(f"{key} {name} recall: ", recall_score(test_target, func.predict(test), average='macro'))
        print(f"{key} {name} f1: ", f1_score(test_target, func.predict(test), average='macro'))
        # Average score
        print(f"{key} {name} average score: ", (
            accuracy_score(test_target, func.predict(test)) + 
            precision_score(test_target, func.predict(test), average='macro', zero_division=1) + 
            recall_score(test_target, func.predict(test), average='macro') + 
            f1_score(test_target, func.predict(test), average='macro')
            )/4)
        print()
#"""


#""" Dimensionality reduction
iris.data = preprocessing.StandardScaler().fit_transform(iris.data)
iris_lda = LinearDiscriminantAnalysis(n_components=2).fit_transform(iris.data, iris.target)

wine.data = preprocessing.StandardScaler().fit_transform(wine.data)
wine_lda = LinearDiscriminantAnalysis(n_components=2).fit_transform(wine.data, wine.target)

blobs = datasets.make_blobs(n_samples=1000, n_features=6, centers=6)
moons = datasets.make_moons(n_samples=1000, noise=0.1)
dictionary2 = {'blobs': blobs, 'moons': moons, 'iris': (iris_lda, iris.target), 'wine': (wine_lda, wine.target)}
#"""


#""" Clustering
for key, value in dictionary2.items():
    x, y = value
    clusters = len(np.unique(y))

    train, test, train_target, test_target = train_test_split(x, y, test_size=0.2, random_state=0)

    k_mean = KMeans(n_init='auto').fit_predict(train)
    mean_shift = MeanShift().fit_predict(train)
    birch = Birch(n_clusters=clusters).fit_predict(train)

    groub_dict = {'k_mean': k_mean, 'mean_shift': mean_shift, 'birch': birch}
    
    fig, ax = plt.subplots(2, 2)

    unique = np.unique(k_mean)
    for i in unique:
        ax[0, 0].scatter(train[k_mean == i, 0], train[k_mean == i, 1], label=i)
    ax[0, 0].title.set_text(f"{key} k_mean")
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])

    unique = np.unique(mean_shift)
    for i in unique:
        ax[1, 0].scatter(train[mean_shift == i, 0], train[mean_shift == i, 1], label=i)
    ax[1, 0].title.set_text(f"{key} mean_shift")
    ax[1, 0].set_xticks([])
    ax[1, 0].set_yticks([])

    unique = np.unique(birch)
    for i in unique:
        ax[1, 1].scatter(train[birch == i, 0], train[birch == i, 1], label=i)
    ax[1, 1].title.set_text(f"{key} birch")
    ax[1, 1].set_xticks([])
    ax[1, 1].set_yticks([])

    ax[0, 1].scatter(x[:, 0], x[:, 1], c=y)
    ax[0, 1].title.set_text(f"{key} original")
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])
    plt.show()

    for name, data in groub_dict.items():
        if len(np.unique(data)) > 1:
            print(f"{key} {name} silhouette_score: ", silhouette_score(train, data))
            print(f"{key} {name} calinski_harabasz_score: ", calinski_harabasz_score(train, data))
        print(f"{key} {name} rand_score: ", rand_score(train_target, data))
        print(f"{key} {name} homogeneity_score: ", homogeneity_score(train_target, data))
        print(f"{key} {name} mutual_info_score: ", mutual_info_score(train_target, data))
        print()
#"""
