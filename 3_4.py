from models.simple_cbow import SimpleCBOW
from utils.optimizer import SGD
from utils.trainer import Trainer
from utils.utils import convert_one_hot, create_contexts_target, preprocess


class CFG:
    window_size = 1
    hidden_size = 5
    batch_size = 3
    epoch = 1000
    eval_interval = 20


text = "You say goodbye and I say hello."
corpus, word2id, id2word = preprocess(text)

vocab_size = len(word2id)
contexts, target = create_contexts_target(corpus, CFG.window_size)
target = convert_one_hot(target, vocab_size)
contexts = convert_one_hot(contexts, vocab_size)

model = SimpleCBOW(vocab_size, CFG.hidden_size)
optimizer = SGD()
trainer = Trainer(model, optimizer)
trainer.fit(contexts, target, CFG)
