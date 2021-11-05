from typing import Optional

import torch

from datasets.spiral import load_data
from models.forward_net import TwoLayerNet
from utils.optimizer import SGD


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
    x = torch.from_numpy(x).float()
    y_true = torch.from_numpy(y_true).float()

    model = TwoLayerNet(
        input_size=CFG.input_size,
        hidden_size=CFG.hidden_size,
        output_size=CFG.output_size,
    )
    optimizer = SGD(lr=CFG.lr)

    data_size = len(x)
    max_iter = data_size // CFG.batch_size

    total_loss = 0
    loss_count = 0
    loss_list = []
    params_dict = {"params": [], "grads": []}

    for epoch in range(CFG.epoch):
        idx = torch.randperm(data_size)
        x = x[idx]
        y_true = y_true[idx]

        for iters in range(max_iter):
            batch_x = x[iters * CFG.batch_size : (iters + 1) * CFG.batch_size]
            batch_y_true = y_true[iters * CFG.batch_size : (iters + 1) * CFG.batch_size]

            loss = model.forward(batch_x, batch_y_true)
            model.backward()
            optimizer.update(model.params, model.grads)
            total_loss += loss
            loss_count += 1

            if (iters + 1) % 10 == 0:
                avg_loss = total_loss / loss_count
                print(f"epoch: {epoch}, iter: {iters}, loss: {avg_loss}")
                loss_list.append(avg_loss)
                total_loss, loss_count = 0, 0
            params_dict["params"].append(model.params)
            params_dict["grads"].append(model.grads)


main()
