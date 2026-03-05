import numpy as np


class SGD:
    """Vanilla Stochastic Gradient Descent."""

    def __init__(self, lr=0.01, weight_decay=0.0):
        self.lr = lr
        self.wd = weight_decay

    def update(self, layers):
        for layer in layers:
            layer.W -= self.lr * (layer.grad_W + self.wd * layer.W)
            layer.b -= self.lr * layer.grad_b


class Momentum:
    """SGD with momentum — accumulates velocity over time."""

    def __init__(self, lr=0.01, beta=0.9, weight_decay=0.0):
        self.lr = lr
        self.beta = beta
        self.wd = weight_decay
        self.v = []

    def update(self, layers):
        if not self.v:
            self.v = [{'W': np.zeros_like(l.W), 'b': np.zeros_like(l.b)} for l in layers]

        for i, layer in enumerate(layers):
            self.v[i]['W'] = self.beta * self.v[i]['W'] + layer.grad_W
            self.v[i]['b'] = self.beta * self.v[i]['b'] + layer.grad_b
            layer.W -= self.lr * (self.v[i]['W'] + self.wd * layer.W)
            layer.b -= self.lr * self.v[i]['b']


class NAG:
    """Nesterov Accelerated Gradient — looks ahead before updating."""

    def __init__(self, lr=0.01, beta=0.9, weight_decay=0.0):
        self.lr = lr
        self.beta = beta
        self.wd = weight_decay
        self.v = []

    def update(self, layers):
        if not self.v:
            self.v = [{'W': np.zeros_like(l.W), 'b': np.zeros_like(l.b)} for l in layers]

        for i, layer in enumerate(layers):
            v_prev_W = self.v[i]['W'].copy()
            v_prev_b = self.v[i]['b'].copy()

            self.v[i]['W'] = self.beta * self.v[i]['W'] + layer.grad_W
            self.v[i]['b'] = self.beta * self.v[i]['b'] + layer.grad_b

            layer.W -= self.lr * ((1 + self.beta) * self.v[i]['W'] - self.beta * v_prev_W + self.wd * layer.W)
            layer.b -= self.lr * ((1 + self.beta) * self.v[i]['b'] - self.beta * v_prev_b)


class RMSProp:
    """RMSProp — adapts learning rate using squared gradient moving average."""

    def __init__(self, lr=0.001, beta=0.9, eps=1e-8, weight_decay=0.0):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.wd = weight_decay
        self.s = []

    def update(self, layers):
        if not self.s:
            self.s = [{'W': np.zeros_like(l.W), 'b': np.zeros_like(l.b)} for l in layers]

        for i, layer in enumerate(layers):
            self.s[i]['W'] = self.beta * self.s[i]['W'] + (1 - self.beta) * layer.grad_W ** 2
            self.s[i]['b'] = self.beta * self.s[i]['b'] + (1 - self.beta) * layer.grad_b ** 2

            layer.W -= self.lr * (layer.grad_W / (np.sqrt(self.s[i]['W']) + self.eps) + self.wd * layer.W)
            layer.b -= self.lr * layer.grad_b / (np.sqrt(self.s[i]['b']) + self.eps)


class Adam:
    """Adam — combines momentum and RMSProp with bias correction."""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.wd = weight_decay
        self.m = []
        self.v = []
        self.t = 0

    def update(self, layers):
        self.t += 1
        if not self.m:
            self.m = [{'W': np.zeros_like(l.W), 'b': np.zeros_like(l.b)} for l in layers]
            self.v = [{'W': np.zeros_like(l.W), 'b': np.zeros_like(l.b)} for l in layers]

        for i, layer in enumerate(layers):
            for p in ['W', 'b']:
                grad = layer.grad_W if p == 'W' else layer.grad_b

                # Update biased moment estimates
                self.m[i][p] = self.beta1 * self.m[i][p] + (1 - self.beta1) * grad
                self.v[i][p] = self.beta2 * self.v[i][p] + (1 - self.beta2) * grad ** 2

                # Bias-corrected estimates
                m_hat = self.m[i][p] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i][p] / (1 - self.beta2 ** self.t)

                if p == 'W':
                    layer.W -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.wd * layer.W)
                else:
                    layer.b -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class Nadam:
    """Nadam — Adam with Nesterov momentum lookahead."""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.wd = weight_decay
        self.m = []
        self.v = []
        self.t = 0

    def update(self, layers):
        self.t += 1
        if not self.m:
            self.m = [{'W': np.zeros_like(l.W), 'b': np.zeros_like(l.b)} for l in layers]
            self.v = [{'W': np.zeros_like(l.W), 'b': np.zeros_like(l.b)} for l in layers]

        for i, layer in enumerate(layers):
            for p in ['W', 'b']:
                grad = layer.grad_W if p == 'W' else layer.grad_b

                self.m[i][p] = self.beta1 * self.m[i][p] + (1 - self.beta1) * grad
                self.v[i][p] = self.beta2 * self.v[i][p] + (1 - self.beta2) * grad ** 2

                m_hat = self.m[i][p] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i][p] / (1 - self.beta2 ** self.t)

                # Nesterov correction on momentum
                m_nadam = (self.beta1 * m_hat + (1 - self.beta1) * grad / (1 - self.beta1 ** self.t))

                if p == 'W':
                    layer.W -= self.lr * (m_nadam / (np.sqrt(v_hat) + self.eps) + self.wd * layer.W)
                else:
                    layer.b -= self.lr * m_nadam / (np.sqrt(v_hat) + self.eps)


def get_optimizer(name, lr, weight_decay=0.0):
    """Factory function to get optimizer by name."""
    optimizers = {
        'sgd':      SGD(lr=lr, weight_decay=weight_decay),
        'momentum': Momentum(lr=lr, weight_decay=weight_decay),
        'nag':      NAG(lr=lr, weight_decay=weight_decay),
        'rmsprop':  RMSProp(lr=lr, weight_decay=weight_decay),
        'adam':     Adam(lr=lr, weight_decay=weight_decay),
        'nadam':    Nadam(lr=lr, weight_decay=weight_decay),
    }
    return optimizers[name]