from datasets import ptb
from models.cbow import CBOW
from utils.optimizer import SGD
from utils.trainer import Trainer
from utils.utils import create_contexts_target


class CFG:
    window_size = 5
    hidden_size = 100
    batch_size = 100
    epoch = 10
    eval_interval = 20


def main():
    corpus, word2id, _ = ptb.load_data("train")
    vocab_size = len(word2id)
    contexts, target = create_contexts_target(corpus, CFG.window_size)
    model = CBOW(vocab_size, CFG.hidden_size, CFG.window_size, corpus)

    optimizer = SGD()
    trainer = Trainer(model, optimizer)
    trainer.fit(contexts, target, CFG)


main()
