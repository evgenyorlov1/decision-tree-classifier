__author__ = 'pc'
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier


def create_decision_tree(data, attributes, target_attr, fitness_func):
    data    = data[:]
    vals    = [record[target_attr] for record in data]
    if not data or (len(attributes) - 1) <= 0:
        return True
    elif vals.count(vals[0]) == len(vals):
        return vals[0]
        tree = {best:{}}

        try:
            for val in get_values(data, best):
            # Create a subtree for the current value under the "best" field
                 subtree = create_decision_tree(
                get_examples(data, best, val),
                [attr for attr in attributes if attr != best],
                target_attr,
                fitness_func)
        finally:
            pass
            tree[best][val] = subtree

    return tree

def entropy(data, target_attr):
    val_freq     = {}
    data_entropy = 0.0

    for record in data:
        if (val_freq.has_key(record[target_attr])):
            val_freq[record[target_attr]] += 1.0
        else:
            val_freq[record[target_attr]]  = 1.0
    try:
        for freq in val_freq.values():
            data_entropy += (-freq/len(data)) * math.log(freq/len(data), 2)
    finally:
        pass
    return data_entropy


def main():
    # Parameters
    n_classes = 3
    plot_colors = "bry"
    plot_step = 0.02

    # Load data
    iris = load_iris()

    for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3],
                                [1, 2], [1, 3], [2, 3]]):
        # We only take the two corresponding features
        X = iris.data[:, pair]
        y = iris.target

        # Shuffle
        idx = np.arange(X.shape[0])
        np.random.seed(13)
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]

        # Standardize
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        X = (X - mean) / std

        # Train
        clf = DecisionTreeClassifier().fit(X, y)

        # Plot the decision boundary
        plt.subplot(2, 3, pairidx + 1)

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                             np.arange(y_min, y_max, plot_step))

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

        plt.xlabel(iris.feature_names[pair[0]])
        plt.ylabel(iris.feature_names[pair[1]])
        plt.axis("tight")

        # Plot the training points
        for i, color in zip(range(n_classes), plot_colors):
            idx = np.where(y == i)
            plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i],
                        cmap=plt.cm.Paired)

        plt.axis("tight")

    plt.suptitle("Decision surface of a decision tree using paired features")
    plt.legend()
    plt.show()

if __name__=="__main__":
    main()
