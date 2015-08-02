"""Relevance Vector Machine classes for regression and classification."""
import numpy as np

from scipy.optimize import minimize
from scipy.special import expit

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.metrics.pairwise import (
    linear_kernel,
    rbf_kernel,
    polynomial_kernel
)
from sklearn.multiclass import OneVsOneClassifier
from sklearn.utils.validation import check_X_y


class BaseRVM(BaseEstimator):

    """Base Relevance Vector Machine class.

    Implementation of Mike Tipping's Relevance Vector Machine using the
    scikit-learn API. Add a posterior over weights method and a predict
    in subclass to use for classification or regression.
    """

    def __init__(
        self,
        kernel='rbf',
        degree=3,
        coef1=None,
        coef0=0.0,
        n_iter=3000,
        tol=1e-3,
        alpha=1e-6,
        threshold_alpha=1e9,
        beta=1.e-6,
        beta_fixed=False,
        bias_used=True,
        verbose=False
    ):
        """Copy params to object properties, no validation."""
        self.kernel = kernel
        self.degree = degree
        self.coef1 = coef1
        self.coef0 = coef0
        self.n_iter = n_iter
        self.tol = tol
        self.alpha = alpha
        self.threshold_alpha = threshold_alpha
        self.beta = beta
        self.beta_fixed = beta_fixed
        self.bias_used = bias_used
        self.verbose = verbose

    def get_params(self, deep=True):
        """Return parameters as a dictionary."""
        params = {
            'kernel': self.kernel,
            'degree': self.degree,
            'coef1': self.coef1,
            'coef0': self.coef0,
            'n_iter': self.n_iter,
            'tol': self.tol,
            'alpha': self.alpha,
            'threshold_alpha': self.threshold_alpha,
            'beta': self.beta,
            'beta_fixed': self.beta_fixed,
            'bias_used': self.bias_used,
            'verbose': self.verbose
        }
        return params

    def set_params(self, **parameters):
        """Set parameters using kwargs."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def _apply_kernel(self, x, y):
        """Apply the selected kernel function to the data."""
        if self.kernel == 'linear':
            phi = linear_kernel(x, y)
        elif self.kernel == 'rbf':
            phi = rbf_kernel(x, y, self.coef1)
        elif self.kernel == 'poly':
            phi = polynomial_kernel(x, y, self.degree, self.coef1, self.coef0)
        elif callable(self.kernel):
            phi = self.kernel(x, y)
            if len(phi.shape) != 2:
                raise ValueError(
                    "Custom kernel function did not return 2D matrix"
                )
            if phi.shape[0] != x.shape[0]:
                raise ValueError(
                    "Custom kernel function did not return matrix with rows"
                    " equal to number of data points."""
                )
        else:
            raise ValueError("Kernel selection is invalid.")

        if self.bias_used:
            phi = np.append(phi, np.ones((phi.shape[0], 1)), axis=1)

        return phi

    def _prune(self, alpha, alpha_old, gamma, phi, sigma, m, X):
        """Remove basis functions based on alpha values."""
        keep_alpha = alpha < self.threshold_alpha

        if not np.any(keep_alpha):
            keep_alpha[0] = True
            if self.bias_used:
                keep_alpha[-1] = True

        if self.bias_used:
            if not keep_alpha[-1]:
                self.bias_used = False
            X = X[keep_alpha[:-1]]
        else:
            X = X[keep_alpha]

        alpha = alpha[keep_alpha]
        alpha_old = alpha_old[keep_alpha]
        gamma = gamma[keep_alpha]
        phi = phi[:, keep_alpha]
        sigma = sigma[np.ix_(keep_alpha, keep_alpha)]
        m = m[keep_alpha]

        return alpha, alpha_old, gamma, phi, sigma, m, X

    def fit(self, X, y):
        """Fit the RVR to the training data."""
        X, y = check_X_y(X, y)

        n_samples, n_features = X.shape

        phi = self._apply_kernel(X, X)

        n_basis_functions = phi.shape[1]

        alpha = self.alpha * np.ones(n_basis_functions)
        beta = self.beta

        m = np.zeros(n_basis_functions)

        alpha_old = alpha

        for i in range(self.n_iter):
            m, sigma = self._posterior(m, alpha, beta, phi, y)

            gamma = 1 - alpha*np.diag(sigma)
            alpha = gamma/(m ** 2)

            if not self.beta_fixed:
                beta = (n_samples - np.sum(gamma))/(
                    np.sum((y - np.dot(phi, m)) ** 2))

            pruned = self._prune(alpha, alpha_old, gamma, phi, sigma, m, X)
            alpha, alpha_old, gamma, phi, sigma, m, X = pruned

            if self.verbose:
                print("Iteration: {}".format(i))
                print("Alpha: {}".format(alpha))
                print("Beta: {}".format(beta))
                print("Gamma: {}".format(gamma))
                print("m: {}".format(m))
                print("Relevance Vectors: {}".format(X.shape[0]))
                print()

            delta = np.amax(np.absolute(alpha - alpha_old))

            if delta < self.tol and i > 1:
                break

            alpha_old = alpha

        self.alpha_ = alpha
        self.beta_ = beta
        self.sigma_ = sigma

        if self.bias_used:
            self.bias = m[-1]
        else:
            self.bias = None

        self.m_ = m
        self.relevance_ = X

        return self


class RVR(BaseRVM, RegressorMixin):

    """Relevance Vector Machine Regression.

    Implementation of Mike Tipping's Relevance Vector Machine for regression
    using the scikit-learn API.
    """

    def _posterior(self, m, alpha, beta, phi, y):
        """Compute the posterior distriubtion over weights."""
        inv_sigma = np.diag(alpha) + beta * np.dot(phi.T, phi)
        sigma = np.linalg.inv(inv_sigma)
        m = beta * np.dot(sigma, np.dot(phi.T, y))

        return m, sigma

    def predict(self, X, eval_MSE=False):
        """Evaluate the RVR model at x."""
        phi = self._apply_kernel(X, self.relevance_)

        y = np.dot(phi, self.m_)

        if eval_MSE:
            MSE = (1/self.beta_) + np.dot(phi, np.dot(self.sigma_, phi.T))
            return y, MSE[:, 0]
        else:
            return y


class RVC(BaseRVM, ClassifierMixin):

    """Relevance Vector Machine Classification.

    Implementation of Mike Tipping's Relevance Vector Machine for
    classification using the scikit-learn API.
    """

    def __init__(self, n_iter_posterior=50, **kwargs):
        """Copy params to object properties, no validation."""
        self.n_iter_posterior = n_iter_posterior
        super(RVC, self).__init__(**kwargs)

    def get_params(self, deep=True):
        """Return parameters as a dictionary."""
        params = super(RVC, self).get_params(deep=deep)
        params['n_iter_posterior'] = self.n_iter_posterior
        return params

    def _classify(self, m, phi):
        return expit(np.dot(phi, m))

    def _log_posterior(self, m, alpha, phi, t):

        y = self._classify(m, phi)

        log_p = -1 * (np.sum(np.log(y[t == 1]) + np.log(1-y[t == 0]), 0))
        log_p = log_p + 0.5*np.dot(m.T, np.dot(np.diag(alpha), m))

        jacobian = np.dot(np.diag(alpha), m) - np.dot(phi.T, (t-y))

        return log_p, jacobian

    def _hessian(self, m, alpha, phi, t):
        y = self._classify(m, phi)
        B = np.diag(y*(1-y))
        return np.diag(alpha) + np.dot(phi.T, np.dot(B, phi))

    def _posterior(self, m, alpha, beta, phi, t):

        result = minimize(
            fun=self._log_posterior,
            hess=self._hessian,
            x0=m,
            args=(alpha, phi, t),
            method='Newton-CG',
            jac=True,
            options={
                'maxiter': self.n_iter_posterior
            }
        )

        m = result.x
        sigma = np.linalg.inv(self._hessian(m, alpha, phi, t))
        return m, sigma

    def fit(self, X, y):
        """Check target values and fit model."""
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        if n_classes < 2:
            raise ValueError("Need 2 or more classes.")
        elif n_classes == 2:
            t = np.zeros(y.shape)
            t[y == self.classes_[1]] = 1
            super(RVC, self).fit(X, t)
        else:
            self.multi_ = None
            self.multi_ = OneVsOneClassifier(self)
            self.multi_.fit(X, y)

    def predict_proba(self, X):
        """Return an array of class probabilities."""
        phi = self._apply_kernel(X, self.relevance_)
        y = self._classify(self.m_, phi)
        return np.column_stack((1-y, y))

    def predict(self, X):
        """Return an array of classes for each input."""
        if len(self.classes_) == 2:
            y = self.predict_proba(X)
            res = np.empty(y.shape[0], dtype=self.classes_.dtype)
            res[y[:, 1] <= 0.5] = self.classes_[0]
            res[y[:, 1] >= 0.5] = self.classes_[1]
            return res
        else:
            return self.multi_.predict(X)
