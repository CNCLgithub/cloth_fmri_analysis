import numpy as np
from sklearn.linear_model import Ridge


class BaseDecoder:
    def fit(self, X, Y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def get_params(self):
        return {}



class LinearDecoder(BaseDecoder):
    def __init__(self):
        self.W = None
        self.b = None

    def fit(self, X, Y):

        # add bias column
        X_aug = np.concatenate(
            [X, np.ones((X.shape[0], 1))],
            axis=1
        )

        W_aug, _, _, _ = np.linalg.lstsq(X_aug, Y, rcond=None)

        self.W = W_aug[:-1]
        self.b = W_aug[-1]

    def predict(self, X):
        if self.W is None:
            raise RuntimeError("LinearDecoder is not fitted yet.")
        return X @ self.W + self.b

    def get_params(self):
        return {"W": self.W, "b": self.b}



class RidgeDecoder(BaseDecoder):

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.model = Ridge(alpha=alpha, fit_intercept=True)

    def fit(self, X, Y):
        self.model.fit(X, Y)

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self):

        W = self.model.coef_

        if W.ndim == 1:
            W = W[:, None]
        else:
            W = W.T

        b = self.model.intercept_

        if np.ndim(b) == 0:
            b = np.array([b])

        return {"W": W, "b": b}