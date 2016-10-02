##########
scikit-rvm
##########

.. image:: https://travis-ci.org/JamesRitchie/scikit-rvm.svg?branch=master
    :target: https://travis-ci.org/JamesRitchie/scikit-rvm
.. image:: https://coveralls.io/repos/JamesRitchie/scikit-rvm/badge.svg?branch=master&service=github
  :target: https://coveralls.io/github/JamesRitchie/scikit-rvm?branch=master



scikit-rvm is a Python module implementing the `Relevance Vector Machine`_ (RVM)
machine learning technique using the `scikit-learn`_ API.

==========
Quickstart
==========

With NumPy, SciPy and scikit-learn available in your environment, install with::

    pip install https://github.com/JamesRitchie/scikit-rvm/archive/master.zip

Regression is done with the RVR class::

    >>> from skrvm import RVR
    >>> X = [[0, 0], [2, 2]]
    >>> y = [0.5, 2.5 ]
    >>> clf = RVR(kernel='linear')
    >>> clf.fit(X, y)
    RVR(alpha=1e-06, beta=1e-06, beta_fixed=False, bias_used=True, coef0=0.0,
    coef1=None, degree=3, kernel='linear', n_iter=3000,
    threshold_alpha=1000000000.0, tol=0.001, verbose=False)
    >>> clf.predict([[1, 1]])
    array([ 1.49995187])

Classification is done with the RVC class::

    >>> from skrvm import RVC
    >>> from sklearn.datasets import load_iris
    >>> clf = RVC()
    >>> clf.fit(iris.data, iris.target)
    RVC(alpha=1e-06, beta=1e-06, beta_fixed=False, bias_used=True, coef0=0.0,
    coef1=None, degree=3, kernel='rbf', n_iter=3000, n_iter_posterior=50,
    threshold_alpha=1000000000.0, tol=0.001, verbose=False)
    >>> clf.score(iris.data, iris.target)
    0.97999999999999998

======
Theory
======

The RVM is a sparse Bayesian analogue to the Support Vector Machine, with a
number of advantages:

* It provides probabilistic estimates, as opposed to the SVM's point estimates.
* Typically provides a sparser solution than the SVM, which tends to have the
  number of support vectors grow linearly with the size of the training set.
* Does not need a complexity parameter to be selected in order to avoid
  overfitting.

However it is more expensive to train than the SVM, although prediction is
faster and no cross-validation runs are required.

The RVM's original creator Mike Tipping provides a selection of papers offering
detailed insight into the formulation of the RVM (and sparse Bayesian learning
in general) on a `dedicated page`_, along with a Matlab implementation.

Most of this implementation was written working from Section 7.2 of Christopher
M. Bishops's `Pattern Recognition and Machine Learning`_.

============
Contributors
============

* `James Ritchie`_
* `Jonathan Feinberg`_

===================
Future Improvements
===================


* Implement the fast Sequential Sparse Bayesian Learning Algorithm outlined in
  Section 7.2.3 of `Pattern Recognition and Machine Learning`_
* Handle ill-conditioning errors more gracefully.
* Implement more kernel choices.
* Create more detailed examples with IPython notebooks.

.. Hyperlinks
.. _Relevance Vector Machine: https://en.wikipedia.org/wiki/Relevance_vector_machine
.. _scikit-learn: http://scikit-learn.org/
.. _Pattern Recognition and Machine Learning: http://research.microsoft.com/en-us/um/people/cmbishop/prml/
.. _dedicated page: http://www.miketipping.com/sparsebayes.htm
.. _James Ritchie: https://github.com/JamesRitchie
.. _Jonathan Feinberg: https://github.com/jonathf
