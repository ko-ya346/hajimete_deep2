from datasets.spiral import load_data
from models.forward_net import TwoLayerNet
from utils.optimizer import SGD
from utils.trainer import Trainer


class CFG:
    epoch = 300
    batch_size = 30
    hidden_size = 10
    lr = 0.1
    input_size = 2
    output_size = 3
    eval_interval = 20


def main():
    x, y_true = load_data()

    model = TwoLayerNet(
        input_size=CFG.input_size,
        hidden_size=CFG.hidden_size,
        output_size=CFG.output_size,
    )
    optimizer = SGD(lr=CFG.lr)

    trainer = Trainer(model, optimizer)
    trainer.fit(x, y_true, CFG)


main()
