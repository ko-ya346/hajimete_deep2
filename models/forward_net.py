import torch
from utils.layers import Affine, Sigmoid, SoftmaxWithLoss


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        # init W, b
        # 重みが小さいほうが学習がうまく進む
        W1 = 0.01 * torch.randn(I, H)
        b1 = torch.zeros(H)
        W2 = 0.01 * torch.randn(H, O)
        b2 = torch.zeros(O)

        # レイヤー生成
        self.layers = [Affine(W1, b1), Sigmoid(), Affine(W2, b2)]
        self.loss_layer = SoftmaxWithLoss()

        # layerの重みをまとめる
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def forward(self, x, y_true):
        score = self.predict(x)
        loss = self.loss_layer.forward(score, y_true)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout