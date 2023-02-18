import torch as t
import torch.nn as nn


class HSIC(nn.Module):
    def __init__(self, sigma_x=1.0, sigma_y=None):
        super(HSIC, self).__init__()

        if sigma_y is None:
            sigma_y = sigma_x

        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

    def _kernel_x(self, X):
        raise NotImplementedError

    def _kernel_y(self, Y):
        raise NotImplementedError

    def estimator(self, input1, input2):
        kernel_XX = self._kernel_x(input1)
        kernel_YY = self._kernel_y(input2)

        tK = kernel_XX - t.diag(kernel_XX)
        tL = kernel_YY - t.diag(kernel_YY)

        N = len(input1)

        hsic = (
            t.trace(tK @ tL)
            + (t.sum(tK) * t.sum(tL) / (N - 1) / (N - 2))
            - (2 * t.sum(tK, 0).dot(t.sum(tL, 0)) / (N - 2))
        )

        return hsic / (N * (N - 3))

    def forward(self, input1, input2, **kwargs):
        return self.estimator(input1, input2)



class RbfHSIC(HSIC):
    def _kernel(self, X, sigma):
        X = X.view(len(X), -1)
        XX = X @ X.t()
        X_sqnorms = t.diag(XX)
        X_L2 = -2 * XX + X_sqnorms.unsqueeze(1) + X_sqnorms.unsqueeze(0)
        gamma = 1 / (2 * sigma ** 2)

        kernel_XX = t.exp(-gamma * X_L2)
        return kernel_XX

    def _kernel_x(self, X):
        return self._kernel(X, self.sigma_x)

    def _kernel_y(self, Y):
        return self._kernel(Y, self.sigma_y)


class MinusRbfHSIC(RbfHSIC):
    def forward(self, input1, input2, **kwargs):
        return -self.estimator(input1, input2)