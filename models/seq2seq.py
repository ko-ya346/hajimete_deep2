import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers):
        """
        vocab_size: 語彙数
        hidden_dim: 単語IDのベクトルの次元数(好きに決めれる)
        num_layers: LSTMレイヤの総数(好きに決めれる)
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.init_h = nn.Linear(vocab_size, vocab_size)
        self.init_c = nn.Linear(vocab_size, vocab_size)
        self.device = "cuda"

        # batch_first: (seq, batch, feature) -> (batch, seq, feature) に置き換える
        self.lstm = nn.LSTM(hidden_dim, vocab_size, num_layers, batch_first=True)

    def forward(self, x, states):
        x = self.embedding(x)
        states = self.set_hidden_state(states)
        x, (h, c) = self.lstm(x, states)
        return x, (h, c)

    def init_hidden(self, batch_size):
        state_dim = (self.num_layers, batch_size, self.vocab_size)
        return (
            torch.randn(state_dim, device=self.device),
            torch.randn(state_dim, device=self.device),
        )

    def set_hidden_state(self, states):
        h = self.init_h(states[0])
        c = self.init_c(states[1])
        return (h, c)


class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, vocab_size, num_layers, batch_first=True)
        self.out = nn.Linear(vocab_size, vocab_size)
        self.init_h = nn.Linear(vocab_size, vocab_size)
        self.init_c = nn.Linear(vocab_size, vocab_size)
        self.device = "cuda"

    def forward(self, x, states):
        """
        states: encoderが出力した隠れ状態
        """
        states = self.set_hidden_state(states)
        x = self.embedding(x)
        x = F.relu(x)
        x, (h, c) = self.lstm(x, states)
        x = self.linear(x)
        return x, (h, c)

    def set_hidden_state(self, states):
        h = self.init_h(states[0])
        c = self.init_c(states[1])
        return (h, c)


class Attention(nn.Module):
    """
    decoderのLSTM層のあとに付け足すattention層
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(
            encoder_out
        )  # (batch_size, encoder_seq_length, attention_dim)
        att2 = self.decoder_att(
            decoder_hidden
        )  # (batch_size, decoder_seq_length, attention_dim)
        att2 = att2.transpose(2, 1)  # (batch_size, attention_dim, decoder_seq_length)
        att = torch.bmm(
            att1, att2
        )  # (batch_size, encoder_seq_length, decoder_seq_length)
        alpha = self.softmax(
            att
        )  # (batch_size, encoder_seq_length, decoder_seq_length)
        return alpha


class DecoderWithAttention(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, attention_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, vocab_size, num_layers, batch_first=True)

        self.init_h = nn.Linear(vocab_size, vocab_size)
        self.init_c = nn.Linear(vocab_size, vocab_size)

        self.attention = Attention(vocab_size, vocab_size, attention_dim)
        self.device = "cuda"

    def forward(self, encoder_out, targets, states):
        encoder_emb = self.embedding(targets)
        encoder_emb = F.relu(encoder_emb)  # (batch_size, encoder_seq_length, embed_dim)

        states = self.set_hidden_state(states)

        lstm_out, _ = self.lstm(
            encoder_emb, states
        )  # (batch_size, decoder_seq_length, vocab_size)
        attention_weight = self.attention(
            encoder_out, lstm_out
        )  # (batch_size, encoder_seq_length, decoder_seq_length)
        attention_weight = attention_weight.transpose(
            2, 1
        )  # (batch_size, decoder_seq_length, encoder_seq_length)

        out = torch.bmm(
            attention_weight, encoder_out
        )  # (batch_size, decoder_seq_length, embed_dim)

        return out

    def set_hidden_state(self, states):
        h = self.init_h(states[0])
        c = self.init_c(states[1])
        return (h, c)


class Seq2seq(nn.Module):
    """
    Encoder, Decoderをまとめただけ
    """

    def __init__(self, vocab_size, batch_size, cfg):
        super().__init__()
        self.attention_flg = cfg.model.params.attention_flg
        self.encoder = Encoder(
            vocab_size,
            cfg.model.params.hidden_dim,
            cfg.model.params.num_layers,
        )
        if self.attention_flg:
            self.decoder = DecoderWithAttention(
                vocab_size,
                cfg.model.params.hidden_dim,
                cfg.model.params.num_layers,
                cfg.model.params.attention_dim,
            )
        else:
            self.decoder = Decoder(
                vocab_size,
                cfg.model.params.hidden_dim,
                cfg.model.params.num_layers,
            )
        self.softmax = nn.Softmax(dim=2)
        self.encoder_hidden = self.encoder.init_hidden(batch_size)
        # decoderの初期重みはencoderから受け取るので不要
        # self.decoder_hidden = decoder.init_hidden(batch_size)

    def forward(self, x, t):
        encoder_out, states = self.encoder(x, self.encoder_hidden)
        if self.attention_flg:
            out = self.decoder.forward(encoder_out, t, states)
        else:
            out = self.decoder.forward(t, states)

        out = self.softmax(out)

        return out


class PLModel(pl.LightningModule):
    def __init__(self, vocab_size, batch_size, cfg):
        """
        cfg: dict
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.cfg = cfg
        self.__build_model()
        self._criterion = eval(self.cfg.loss)()

    def __build_model(self):
        self.model = Seq2seq(self.vocab_size, self.batch_size, self.cfg)

    def forward(self, x, t):
        out = self.model(x, t)
        return out

    def __share_step(self, batch):
        feature, target = batch

        score = self.forward(feature, target)
        target = F.one_hot(target, num_classes=self.vocab_size).float()

        loss = self._criterion(score, target)
        pred = score.detach().cpu()
        target = target.detach().cpu()
        return loss, pred, target

    def training_step(self, batch, batch_idx):
        loss, pred, target = self.__share_step(batch)
        return {"loss": loss, "pred": pred, "target": target}

    def validation_step(self, batch, batch_idx):
        loss, pred, target = self.__share_step(batch)
        return {"loss": loss, "pred": pred, "target": target}

    def __share_epoch_end(self, outputs, mode):
        preds = []
        targets = []
        for out in outputs:
            pred, target = out["pred"], out["target"]
            preds.append(pred)
            targets.append(target)
        preds = torch.cat(preds)
        targets = torch.cat(targets)

        preds_id = torch.argmax(preds, dim=2)
        targets_id = torch.argmax(targets, dim=2)
        print(preds_id.shape)
        print(targets_id.shape)
        print(preds_id[:3])
        print(targets_id[:3])
        # 正解率
        metrics = torch.sum(preds_id == targets_id) / len(preds_id) * 100
        self.log(f"{mode}_loss", metrics)

    def training_epoch_end(self, outputs):
        self.__share_epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs):
        self.__share_epoch_end(outputs, "valid")

    def configure_optimizers(self):
        # TODO: optimizerの引数のself.parametersってなんだ
        optimizer = eval(self.cfg.optimizer.name)(
            self.parameters(), **self.cfg.optimizer.params
        )
        return [optimizer]
