import warnings
from glob import glob

import pytorch_lightning as pl
import torch
from pytorch_lightning import callbacks
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

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
    print(x_train[:3])
    x_train = torch.flip(x_train, [1])
    t_train = torch.flip(t_train, [1])
    print(x_train[:3])
    x_val = tokenizer.questions[sep_idx:, :]
    t_val = tokenizer.answers[sep_idx:, :]
    datamodule = AdditionDataModule(x_train, t_train, x_val, t_val, cfg)

    print(tokenizer.char2id)
    # set model
    ## batch_sizeはtrainとvalid別で指定しているが、どちらも同じ値
    ## 変えると何かまずいのか
    model = PLModel(input_size, cfg.dataloader.train.batch_size, cfg, tokenizer)

    loss_checkpoint = callbacks.ModelCheckpoint(
        filename="best_loss", monitor="valid_loss", mode="min", dirpath="./output"
    )
    early_stopping = EarlyStopping(monitor="valid_loss", patience=30)
    logger = TensorBoardLogger("./output")
    trainer = pl.Trainer(
        max_epochs=cfg.General.epoch,
        logger=logger,
        callbacks=[loss_checkpoint, early_stopping],
        **cfg.General.trainer
    )
    trainer.fit(model, datamodule=datamodule)

    # CV score

    event_path = glob("./output/default/version_*")[-1]
    event_acc = EventAccumulator(event_path, size_guidance={"scalars": 0})
    event_acc.Reload()

    scalars = {}
    for tag in event_acc.Tags()["scalars"]:
        events = event_acc.Scalars(tag)
        scalars[tag] = [event.value for event in events]

    print("valid_loss")
    print(scalars["valid_loss"])


main()
