import torch


class Sigmoid:
    """
    sigmoidレイヤー
    """

    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x: torch.tensor) -> torch.tensor:
        out = 1 / (1 + torch.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx


class Affine:
    """
    全結合層
    """

    def __init__(self, W, b):
        """

        Parameter
        ---------
        W: 重み
        b: バイアス
        """
        self.params = [W, b]
        self.grads = [torch.randn(W.shape), torch.randn(b.shape)]
        self.x = None

    def forward(self, x):
        W, b = self.params
        out = torch.matmul(x, W) + b
        self.x = x
        return out

    def backward(self, dout):
        W, b = self.params
        dx = torch.matmul(dout, W.T)
        dW = torch.matmul(self.x.T, dout)
        db = torch.sum(dout, axis=0)

        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx


def softmax(x: torch.tensor) -> torch.tensor:
    x_exp = torch.exp(x)
    # 行列をベクトルで割り算する時は、unsqueezeで次元を追加する
    return x_exp / torch.sum(x_exp, axis=1).unsqueeze(1)


def cross_entropy_error(y_true, y_pred):
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(1, -1)
        y_true = y_true.reshape(1, -1)
    batch_size = y_pred.shape[0]
    loss = (
        -torch.sum(
            torch.log(
                y_pred[torch.arange(batch_size), torch.argmax(y_true, axis=1)] + 1e-7
            )
        )
        / batch_size
    )
    return loss


class Softmax:
    """
    softmaxを計算する
    sigmoid関数を多次元に拡張したイメージ
    """

    def __init__(self):
        self.params = []
        self.out = None

    def forward(self, x):
        self.out = softmax(x)
        return self.out

    def backward(self, dout):
        dx = self.out * dout
        sumdx = torch.sum(dx, keepdim=True)
        dx -= self.out * sumdx
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y_pred = None
        self.y_true = None

    def forward(self, x, y_true):
        self.y_true = y_true
        self.y_pred = softmax(x)
        loss = cross_entropy_error(self.y_true, self.y_pred)
        return loss

    def backward(self, dout=1):
        batch_size = self.y_pred.shape[0]
        dx = (self.y_pred - self.y_true) / batch_size
        return dx


class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [torch.zeros_like(W)]
        self.x = None

    def forward(self, x):
        (W,) = self.params
        out = torch.matmul(x, W)
        self.x = x
        return out

    def backward(self, dout):
        (W,) = self.params
        dx = torch.matmul(dout, W.T)
        dW = torch.matmul(self.x.T, dout)
        # 3点リーダー、変数のメモリアドレスを固定しながら
        # 要素を変更
        self.grads[0][...] = dW
        return dx
