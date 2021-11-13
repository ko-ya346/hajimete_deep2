import torch
import torch.nn as nn
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class AdditionDataset(Dataset):
    def __init__(self, questions, answers=None):
        self._X = questions
        self._y = answers

    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx):
        if self._y is None:
            return self._X[idx]
        return self._X[idx], self._y[idx]


class AdditionDataModule(LightningDataModule):
    def __init__(self, x_train, t_train, x_val, t_val, conf):
        # 親クラスの__init__を生かす
        super().__init__()
        self._x_train = x_train
        self._t_train = t_train
        self._x_val = x_val
        self._t_val = t_val
        self._conf = conf

    def __create_dataset(self, train=True):
        return (
            AdditionDataset(self._x_train, self._t_train)
            if train
            else AdditionDataset(self._x_val, self._t_val)
        )

    def train_dataloader(self):
        dataset = self.__create_dataset(train=True)
        return DataLoader(dataset, **self._conf.dataloader.train)

    def val_dataloader(self):
        dataset = self.__create_dataset(train=False)
        return DataLoader(dataset, **self._conf.dataloader.valid)


class Tokenizer:
    """
    addition.txtデータセットを学習用に前処理
    """

    def __init__(self, filepath):
        self.questions, self.answers = self._preprocess(filepath)

    def _preprocess(self, filepath):
        """
        addition.txtを質問と答えに分けて、空白と改行文字を削除
        """
        questions = []
        answers = []
        with open(filepath, "r") as f:
            for line in f:
                idx = line.find("_")
                questions.append(line[:idx])
                answers.append(line[idx:])

        questions = [txt.rstrip() for txt in questions]
        answers = [txt.replace("\n", "").rstrip() for txt in answers]

        # tokenizer的なものを生成
        self.all_char = sorted(list(set("".join(questions) + "".join(answers))))

        self.char2id = dict(zip(self.all_char, range(len(self.all_char))))
        self.id2char = {v: k for k, v in self.char2id.items()}
        self.input_size = len(self.char2id)

        # questionsとanswersをidのリストに変換
        questions_id = [torch.tensor([self.char2id[qq] for qq in q]) for q in questions]
        answers_id = [torch.tensor([self.char2id[qq] for qq in q]) for q in answers]

        # 長さをそろえる
        questions_id_padded = nn.utils.rnn.pad_sequence(
            questions_id, padding_value=self.char2id["_"]
        ).transpose(1, 0)
        answers_id_padded = nn.utils.rnn.pad_sequence(
            answers_id, padding_value=self.char2id["_"]
        ).transpose(1, 0)
        return questions_id_padded, answers_id_padded
