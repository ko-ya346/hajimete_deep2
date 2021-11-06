import numpy as np

from models.forward_net import TwoLayerNet


def main():
    x = np.random.randn(10, 2)
    print(x)

    model = TwoLayerNet(2, 4, 3)
    s = model.predict(x)
    print(s)


main()
