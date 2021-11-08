from os import read

import pytorch_lightning as pl
from pytorch_lightning.core import datamodule
from pytorch_lightning.utilities.seed import seed_everything

from datasets.addition_dataset import AdditionDataModule, AdditionDataset, Tokenizer
from models.seq2seq import PLModel
from utils.factory import read_yaml


def main():

    CONFIG_PATH = "./config/seq2seq.yaml"
    DATAPATH = "./datasets/addition.txt"

    # load config
    cfg = read_yaml(CONFIG_PATH)
    seed_everything(cfg.General.seed)

    # load data
    tokenizer = Tokenizer(DATAPATH)
    input_size = len(tokenizer.char2id)
    ## outputも同じ語彙を使用する
    output_size = len(tokenizer.char2id)

    sep_idx = 40000
    x_train = tokenizer.questions[:sep_idx, :]
    t_train = tokenizer.answers[:sep_idx, :]
    x_val = tokenizer.questions[sep_idx:, :]
    t_val = tokenizer.answers[sep_idx:, :]
    datamodule = AdditionDataModule(x_train, t_train, x_val, t_val, cfg)

    # set model
    ## batch_sizeはtrainとvalid別で指定しているが、どちらも同じ値
    ## 変えると何かまずいのか
    model = PLModel(input_size, output_size, cfg.dataloader.train.batch_size, cfg)
    model.train()
    trainer = pl.Trainer(max_epochs=cfg.General.epoch)
    trainer.fit(model, datamodule=datamodule)


main()
