import timeit


# Time classification with the iris dataset.

setup = """
from sklearn.datasets import load_iris
from skrvm import RVC

iris = load_iris()

X = iris.data
y = iris.target

clf = RVC()
"""

time = timeit.timeit("clf.fit(X, y)", setup=setup, number=10)

print("10 runs of Iris classification fitting took {} seconds.".format(time))
