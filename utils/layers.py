from collections import Counter
from os import replace

import numpy as np
import torch
from numpy.lib.function_base import bartlett
from numpy.random.mtrand import sample


def softmax(x):
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))
    return x


def cross_entropy_error(y_pred, y_true):
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(1, -1)
        y_true = y_true.reshape(1, -1)
    if y_true.shape == y_pred.shape:
        y_true = y_true.argmax(axis=1)

    batch_size = y_pred.shape[0]
    loss = (
        -np.sum(np.log(y_pred[np.arange(batch_size), y_true.astype(int)] + 1e-7))
        / batch_size
    )

    return loss


class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None

    def forward(self, x):
        (W,) = self.params
        out = np.dot(x, W)
        self.x = x
        return out

    def backward(self, dout):
        (W,) = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        # 3点リーダー、変数のメモリアドレスを固定しながら
        # 要素を変更
        self.grads[0][...] = dW
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
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        self.x = x
        return out

    def backward(self, dout):
        W, _ = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)

        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx


class Softmax:
    """
    softmaxを計算する
    sigmoid関数を多次元に拡張したイメージ
    """

    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        self.out = softmax(x)
        return self.out

    def backward(self, dout):
        dx = self.out * dout
        sumdx = np.sum(dx, axis=1, keepdim=True)
        dx -= self.out * sumdx
        return dx


class SoftmaxWithLoss:
    def __init__(self) -> None:
        self.params, self.grads = [], []
        self.y_pred = None
        self.y_true = None

    def forward(self, x, y_true):
        self.y_true = y_true
        self.y_pred = softmax(x)

        if self.y_true.shape == self.y_pred.shape:
            self.y_true = self.y_true.argmax(axis=1)

        loss = cross_entropy_error(self.y_pred, self.y_true)
        return loss

    def backward(self, dout=1):
        batch_size = self.y_true.shape[0]
        dx = self.y_pred.copy()
        dx[np.arange(batch_size), self.y_true] -= 1
        dx *= dout
        dx /= batch_size
        return dx


class Sigmoid:
    """
    sigmoidレイヤー
    """

    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx


class SigmoidWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.loss = None
        self.y_pred = None
        self.y_true = None

    def forward(self, x, y_true):
        self.y_true = y_true
        self.y_pred = 1 / (1 + np.exp(-x))
        self.loss = cross_entropy_error(
            np.c_[1 - self.y_pred, self.y_pred], self.y_true
        )

        return self.loss

    def backward(self, dout=1):
        batch_size = self.y_true.shape[0]
        dx = (self.y_pred - self.y_true) * dout / batch_size
        return dx


class Embedding:
    """
    重みパラメータから「単語IDに該当する行（ベクトル）」を抜き出すレイヤ
    重みパラメータからindexを指定して取り出す

    単語の埋め込み（word embedding）に由来
    """

    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        idx = idx.astype(int)
        (W,) = self.params
        self.idx = idx
        out = W[idx]
        return out

    def backward(self, dout):
        (dW,) = self.grads
        dW[...] = 0
        np.add.at(dW, self.idx, dout)

        return None


class EmbeddingDot:
    def __init__(self, W):
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None

    def forward(self, h, idx):
        """
        Parameters
        ----------
        h: 中間層ニューロン
        idx: 注目する単語IDのインデックス
        """
        target_W = self.embed.forward(idx)
        out = np.sum(target_W * h, axis=1)

        # 順伝搬のときに計算した結果を一時的に保持
        self.cache = (h, target_W)
        return out

    def backward(self, dout):
        h, target_W = self.cache
        dout = dout.reshape(-1, 1)
        dtarget_W = dout * h
        self.embed.backward(dtarget_W)
        dh = dout * target_W
        return dh


class UnigramSampler:
    """
    word2vecを二値分類で学習させるとき、
    正例だけでなく負例も入れると精度が向上する。
    負例を出現頻度を考慮してランダムに選択するためのクラス
    """

    def __init__(self, corpus, power, sample_size):
        self.sample_size = sample_size
        self.vocab_size = None
        self.word_p = None

        counts = Counter(corpus)
        vocab_size = len(counts)
        self.vocab_size = vocab_size

        self.word_p = np.zeros(vocab_size)
        for i in range(vocab_size):
            self.word_p[i] = counts[i]
        self.word_p = np.power(self.word_p, power)
        self.word_p /= np.sum(self.word_p)

    def get_negative_sample(self, target):
        batch_size = target.shape[0]
        negative_sample = np.zeros((batch_size, self.sample_size))
        for i in range(batch_size):
            p = self.word_p.copy()
            target_idx = target[i]
            p[target_idx] = 0
            p /= p.sum()
            negative_sample[i, :] = np.random.choice(
                self.vocab_size, size=self.sample_size, replace=False, p=p
            )

        return negative_sample


class NegativeSamplingLoss:
    """
    正例、負例を二値分類で学習する
    """

    def __init__(self, W, corpus, power=0.75, sample_size=5):
        """
        Parameters
        ----------
        W: 出力側の重み
        corpus: コーパス
        power: 確率分布の累乗の値（これを入れるとサンプリングされない負例の確率が増える）
        sample_size: 負例のサンプリング数

        """
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size)

        # 正例1つ + 負例 sample_size
        # index 0が正例とする
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size + 1)]
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size + 1)]

        self.params, self.grads = [], []
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, h, target):
        """
        Parameters
        ----------
        h: 中間層のニューロン
        target: 正例のターゲット
        """
        batch_size = target.shape[0]
        negative_sample = self.sampler.get_negative_sample(target)

        # 正例
        score = self.embed_dot_layers[0].forward(h, target)
        correct_label = np.ones(batch_size)
        loss = self.loss_layers[0].forward(score, correct_label)

        # 負例
        negative_label = np.zeros(batch_size)
        for i in range(self.sample_size):
            negative_target = negative_sample[:, i]
            score = self.embed_dot_layers[1 + i].forward(h, negative_target)
            loss += self.loss_layers[1 + i].forward(score, negative_label)
        return loss

    def backward(self, dout=1):
        dh = 0
        for loss0, loss1 in zip(self.loss_layers, self.embed_dot_layers):
            dscore = loss0.backward(dout)
            dh += loss1.backward(dscore)
        return dh


class RNN:
    def __init__(self, Wx, Wh, b):
        """
        Wx: 入力xに対する重み
        Wh: 一つ前のレイヤから受け取る入力に対する重み
        """
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev):
        """
        h_prev: 一つ前のRNNレイヤから受け取る入力"""
        Wx, Wh, b = self.params
        tmp = np.dot(h_prev, Wh) + np.dot(x, Wx) + b
        h_next = np.tanh(tmp)
        self.cache = (x, h_prev, h_next)
        return h_next

    def backward(self, dh_next):
        Wx, Wh, b = self.params
        x, h_prev, h_next = self.cache
        dt = dh_next * (1 - h_next ** 2)
        db = np.sum(dt, axis=0)
        dWh = np.dot(h_prev.T, dt)
        dh_prev = np.dot(dt, Wh.T)
        dWx = np.dot(x.T, dt)
        dx = np.dot(dt, Wx.T)
        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db
        return dx, dh_prev
