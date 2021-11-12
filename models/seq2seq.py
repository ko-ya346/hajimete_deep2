import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# TODO: パラメータの変数名をわかりやすくしたい。size -> dimにしたい（打ちやすいので）
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        """
        input_size: vocab_size
        hidden_size: 単語IDのベクトルの次元数(好きに決めれる)
        num_layers: LSTMレイヤの総数(好きに決めれる)
        output_size: 出力長の最大長さ

        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # TODO: Embeddingの第二引数（埋め込み次元）は自由に決められるので
        # emb_dimというインスタンス変数を追加したい
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.init_h = nn.Linear(input_size, output_size)
        self.init_c = nn.Linear(input_size, output_size)
        self.device = "cuda"

        # batch_first: (seq, batch, feature) -> (batch, seq, feature) に置き換える
        self.lstm = nn.LSTM(hidden_size, output_size, num_layers, batch_first=True)

    def forward(self, xs, states):
        # embeddingのあと、次元どうなるっけ？
        xs = self.embedding(xs)
        states = self.set_hidden_state(states)
        xs, (h, c) = self.lstm(xs, states)
        return xs, (h, c)

    def init_hidden(self, batch_size):
        state_dim = (self.num_layers, batch_size, self.input_size)
        return (
            torch.randn(state_dim, device=self.device),
            torch.randn(state_dim, device=self.device),
        )

    def set_hidden_state(self, states):
        h = self.init_h(states[0])
        c = self.init_c(states[1])
        return (h, c)


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # TODO: Embeddingの第二引数（埋め込み次元）は自由に決められるので
        # emb_dimというインスタンス変数を追加したい
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, output_size, num_layers, batch_first=True)
        self.linear = nn.Linear(output_size, output_size)
        self.init_h = nn.Linear(input_size, output_size)
        self.init_c = nn.Linear(input_size, output_size)
        self.device = "cuda"

    def forward(self, xs, states):
        """
        states: encoderの最終
        """
        states = self.set_hidden_state(states)
        xs = self.embedding(xs)
        xs = F.relu(xs)
        xs, (h, c) = self.lstm(xs, states)
        xs = self.linear(xs)
        return xs, (h, c)

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
        """
        attention_dimってhidden_dimと一緒？
        """
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
    def __init__(
        self, input_size, hidden_size, num_layers, output_size, attention_size
    ):
        super().__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # TODO: Embeddingの第二引数（埋め込み次元）は自由に決められるので
        # emb_dimというインスタンス変数を追加したい
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, output_size, num_layers, batch_first=True)

        self.init_h = nn.Linear(input_size, output_size)
        self.init_c = nn.Linear(input_size, output_size)

        # TODO: Attentionの引数、隠れ状態の次元数がencoderとdecoderで一致していない場合に対応してない
        # インスタンス変数に加えるべき
        self.attention = Attention(input_size, output_size, attention_size)
        self.device = "cuda"

    def forward(self, encoder_out, targets, states):
        encoder_emb = self.embedding(targets)
        encoder_emb = F.relu(
            encoder_emb
        )  # (batch_size, encoder_seq_length, embed_size)

        states = self.set_hidden_state(states)

        lstm_out, _ = self.lstm(
            encoder_emb, states
        )  # (batch_size, decoder_seq_length, output_size)
        print("lstm_out.shape)", lstm_out.shape)
        attention_weight = self.attention(
            encoder_out, lstm_out
        )  # (batch_size, encoder_seq_length, decoder_seq_length)
        attention_weight = attention_weight.transpose(
            2, 1
        )  # (batch_size, decoder_seq_length, encoder_seq_length)

        out = torch.bmm(
            attention_weight, encoder_out
        )  # (batch_size, decoder_seq_length, embed_size)
        print("out.shape", out.shape)

        return out

    def set_hidden_state(self, states):
        h = self.init_h(states[0])
        c = self.init_c(states[1])
        return (h, c)


class Seq2seq(nn.Module):
    """
    Encoder, Decoderをまとめただけ
    """

    def __init__(self, input_size, output_size, batch_size, cfg):
        super().__init__()
        self.encoder = Encoder(
            input_size,
            cfg.model.params.hidden_size,
            cfg.model.params.num_layers,
            input_size,
        )
        self.decoder = Decoder(
            input_size,
            cfg.model.params.hidden_size,
            cfg.model.params.num_layers,
            output_size,
        )
        self.softmax = nn.Softmax(dim=2)
        self.encoder_hidden = self.encoder.init_hidden(batch_size)
        # decoderの初期重みはencoderから受け取るので不要
        # self.decoder_hidden = decoder.init_hidden(batch_size)

    def forward(self, xs, ts):
        _, states = self.encoder(xs, self.encoder_hidden)
        output_decoder, _ = self.decoder(ts, states)
        output = self.softmax(output_decoder)

        return output


class PLModel(pl.LightningModule):
    def __init__(self, input_size, output_size, batch_size, cfg):
        """
        cfg: dict
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.cfg = cfg
        self.__build_model()
        self._criterion = eval(self.cfg.loss)()

    def __build_model(self):
        self.model = Seq2seq(
            self.input_size, self.output_size, self.batch_size, self.cfg
        )

    def forward(self, x, t):
        out = self.model(x, t)
        return out

    def __share_step(self, batch):
        feature, target = batch

        score = self.forward(feature, target)
        target = F.one_hot(target, num_classes=self.input_size).float()

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

        # 正解率
        metrics = torch.sum(preds == targets) / len(preds) * 100
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
