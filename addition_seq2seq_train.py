import warnings

import pytorch_lightning as pl
import torch
from pytorch_lightning import callbacks
from pytorch_lightning.callbacks import early_stopping
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities.seed import seed_everything

from datasets.addition_dataset import AdditionDataModule, Tokenizer
from models.seq2seq import PLModel
from utils.factory import read_yaml

warnings.filterwarnings("ignore")


def main():

    CONFIG_PATH = "./config/seq2seq.yaml"
    DATAPATH = "./datasets/addition.txt"

    # load config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    loss_checkpoint = callbacks.ModelCheckpoint(
        filename="best_loss", monitor="valid_loss", mode="min", dirpath="./output"
    )
    early_stopping = EarlyStopping(monitor="valid_loss", patience=3)
    trainer = pl.Trainer(
        max_epochs=cfg.General.epoch,
        callbacks=[loss_checkpoint, early_stopping],
        **cfg.General.trainer
    )
    trainer.fit(model, datamodule=datamodule)


main()
