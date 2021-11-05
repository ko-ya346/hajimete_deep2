import torch


class Trainer:
    """
    modelとoptimizerを渡して学習してくれるクラス
    """

    def __init__(self, model, optimizer) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_list = []
        self.eval_interval = None
        self.current_epoch = 0

    def fit(self, x, y_true, cfg):
        data_size = len(x)
        max_iters = data_size // cfg.batch_size
        self.eval_interval = cfg.eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss, loss_count = 0, 0

        for epoch in range(cfg.epoch):
            idx = torch.randperm(data_size)
            x = x[idx]
            y_true = y_true[idx]

            for iters in range(max_iters):
                batch_x = x[iters * cfg.batch_size : (iters + 1) * cfg.batch_size]
                batch_y_true = y_true[
                    iters * cfg.batch_size : (iters + 1) * cfg.batch_size
                ]

                loss = model.forward(batch_x, batch_y_true)
                model.backward()
                optimizer.update(model.params, model.grads)
                total_loss += loss
                loss_count += 1

                if iters % cfg.eval_interval == 0:
                    avg_loss = total_loss / loss_count
                    print(
                        f"epoch: {self.current_epoch}, iter: {iters}, loss: {avg_loss}"
                    )
                    self.loss_list.append(avg_loss)
                    total_loss, loss_count = 0, 0
            self.current_epoch += 1
