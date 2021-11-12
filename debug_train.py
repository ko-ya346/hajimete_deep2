import warnings
from glob import glob

import pytorch_lightning as pl
import torch
from pytorch_lightning import callbacks
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.data import DataLoader

from datasets.addition_dataset import AdditionDataModule, AdditionDataset, Tokenizer
from models.seq2seq import Attention, Decoder, DecoderWithAttention, Encoder, PLModel
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

    # TODO: attention_sizeもパラメータに入れる
    attention_size = 64

    sep_idx = 40000
    x_train = tokenizer.questions[:sep_idx, :]
    t_train = tokenizer.answers[:sep_idx, :]
    # 順序を反転させた学習データを追加
    # x_train = torch.cat((x_train, torch.flip(x_train, [1])), axis=0)
    # t_train = torch.cat((t_train, t_train))
    x_val = tokenizer.questions[sep_idx:, :]
    t_val = tokenizer.answers[sep_idx:, :]
    train_dataset = AdditionDataset(x_train, t_train)
    train_datamodule = DataLoader(
        train_dataset, batch_size=cfg.dataloader.train.batch_size
    )

    # model
    encoder = Encoder(
        input_size,
        cfg.model.params.hidden_size,
        cfg.model.params.num_layers,
        output_size,
    )

    attention_flg = True

    if attention_flg:
        decoder = DecoderWithAttention(
            input_size,
            cfg.model.params.hidden_size,
            cfg.model.params.num_layers,
            output_size,
            attention_size,
        )
    else:
        decoder = Decoder(
            input_size,
            cfg.model.params.hidden_size,
            cfg.model.params.num_layers,
            output_size,
        )
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    batch_size = cfg.dataloader.train.batch_size

    for question, answer in train_datamodule:
        encoder.train()
        decoder.train()
        question = question.to(device)
        answer = answer.to(device)
        print("question.shape", question.shape)
        encoder_init = encoder.init_hidden(batch_size)

        encoder_out, states = encoder.forward(question, encoder_init)
        print("encoder_out.shape: ", encoder_out.shape)
        print("states[0].shape: ", states[0].shape)
        print("states[1].shape: ", states[1].shape)

        if attention_flg:
            out = decoder.forward(encoder_out, answer, states)
        else:
            out = decoder.forward(answer, states)
        print(out.shape)
        return


main()
