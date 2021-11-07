import numpy as np

from datasets import ptb
from models.simple_rnnlm import SimpleRnnlm
from utils.optimizer import SGD


class CFG:
    batch_size = 10
    wordvec_size = 100
    hidden_size = 100
    time_size = 5
    lr = 0.1
    epoch = 100


def main():
    corpus, word2id, id2word = ptb.load_data("train")
    corpus_size = 1000
    corpus = corpus[:corpus_size]
    vocab_size = int(max(corpus) + 1)
    xs = corpus[:-1]  # 入力
    ts = corpus[1:]
    data_size = len(xs)

    max_iters = data_size // (CFG.batch_size * CFG.time_size)
    time_idx = 0
    total_loss = 0
    loss_count = 0

    ppl_list = []
    model = SimpleRnnlm(vocab_size, CFG.wordvec_size, CFG.hidden_size)
    optimizer = SGD(lr=CFG.lr)

    # ミニバッチの各サンプルの読み込み開始位置
    jump = (corpus_size - 1) // CFG.batch_size
    offsets = [i * jump for i in range(CFG.batch_size)]

    for epoch in range(CFG.epoch):
        for iter in range(max_iters):
            batch_x = np.empty((CFG.batch_size, CFG.time_size), dtype="i")
            batch_t = np.empty((CFG.batch_size, CFG.time_size), dtype="i")

            # batchサンプル作成
            for t in range(CFG.time_size):
                for i, offset in enumerate(offsets):
                    # i: バッチ番号
                    # t: 時間のindex
                    batch_x[i, t] = xs[(offset + time_idx) % data_size]
                    batch_t[i, t] = ts[(offset + time_idx) % data_size]
                time_idx += 1

            loss = model.forward(batch_x, batch_t)
            model.backward()
            optimizer.update(model.params, model.grads)
            total_loss += loss
            loss_count += 1

        ppl = np.exp(total_loss / loss_count)
        print(f"epoch: {epoch}, perplexity: {ppl}")
        ppl_list.append(float(ppl))
        total_loss, loss_count = 0, 0


main()
