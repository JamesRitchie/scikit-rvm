from rvm import RVR
import numpy as np

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV

# # Generate sample data
# X = np.sort(5 * np.random.rand(200, 1), axis=0)
# y = np.sinc(X).ravel() + 5
#
# ###############################################################################
# # Add noise to targets
# y += np.random.normal(0, 0.1, y.shape)

boston = load_boston()
X = boston.data
y = boston.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

param_grid = [
    {
        'coef1': [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    }
]

clf = GridSearchCV(RVR(), param_grid, cv=5)
clf.fit(X_train, y_train)

print("Best params:")
print(clf.best_params_)
print()
print("Best score:")
print(clf.best_score_)
print("Relevance Vectors: ")
print(clf.best_estimator_.relevance_.shape[0])
print()
print("Test score: {}".format(clf.score(X_test, y_test)))


# pred = clf.predict(X)
#
# plt.scatter(X, y, c='k', label='data')
# plt.hold('on')
#
# plt.plot(X[:, 0], pred)
# plt.show()
