# 目的
- LSTMを実装したい
当初はtorch.tensorを使って挑戦したが難しいので、一旦写経することにした。

# 11/8
- seq2seqの実装
- Encoder, Decoderを使ったモデル
- pytorch_lightningでデータセットとモデルを書いた
- loss計算の部分頑張った
    - 正解データをone hot化してBCEwithlogitslossに投げた
- 学習が進まない、lossが下がらない困った 

# 11/9
- 効果なし
    - LSTMのレイヤー増やす
    - hidden size増やす


# 11/13
- 学習が中々進まないので色々見直す
    - encoder
        - 初期重みを[0-1]に変更
        - 出力にlinearをかます
    - decoder with attention
        - context vectorとlstm_outをcatしてlinearかます
            - もともとはcontext vectorを出力してた。あほ。。。
            - この修正が予測に一番影響大きかった気がする
    - その他
        - loss計算をcross entropyにして、ブランク文字のスコアを無視するようにした
            - 何もしないと全てブランク文字を予測するようになってた
    - 一応lossが順調に下がり、予測も正解ラベルに近いものを吐くようになった。よかった
        - 改良前は全て同じ文字を吐いてた
