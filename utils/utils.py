import numpy as np
import torch


def preprocess(text: str):
    """
    文章を単語毎に分けてトークン化

    Output
    ------
    corpus: list
        単語IDに変換した文章
    word_to_id: dict
        key -> 単語
        value -> ID
    id_to_word: dict
        key -> ID
        value -> 単語

    """
    words = text.lower().replace(".", " .").split(" ")

    word_to_id = {}
    id_to_word = {}
    id = 0

    for word in words:
        if word not in word_to_id:
            word_to_id[word] = id
            id_to_word[id] = word
            id += 1
    corpus = [word_to_id[w] for w in words]

    return corpus, word_to_id, id_to_word


def create_co_matrix(corpus, vocab_size, window_size=1):
    """
    共起行列を作成する。
    共起行列。。。周囲に存在する単語をカウントした行列


    """
    corpus_size = len(corpus)
    co_matrix = torch.zeros((vocab_size, vocab_size))
    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - i
            right_idx = idx + i
            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1
    return co_matrix


def cos_similarity(vec1, vec2, eps=1e-8):
    """
    コサイン類似度を計算
    """
    nx = vec1 / (torch.sqrt(torch.sum(vec1 ** 2)) + eps)
    ny = vec2 / (torch.sqrt(torch.sum(vec2 ** 2)) + eps)
    print(nx, ny)
    return torch.matmul(nx, ny)


def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    """
    queryの単語に対して類似した単語を上位から順に表示する

    Parameters
    ----------
    query: str
        類似度を調べたい単語
    word_to_id: dict
        key -> word
        value -> ID
    id_to_word: dict
        key -> ID
        value -> word
    word_matrix:
        単語ベクトルをまとめた行列
    top:
        上位からいくつ表示するか
    """
    # queryの単語が辞書になければ終了
    if query not in word_to_id:
        print(f"{query} is not found.")
        return

    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]
    vocab_size = len(id_to_word)
    similarity = torch.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    _, sorted_sim = torch.sort(similarity, descending=True)
    return sorted_sim[:top]


def create_contexts_target(corpus, window_size=1):
    """
    context（周囲の単語）からtargetを予測するモデル
    （word2vec）用に、contextとtargetを用意する
    """
    target = corpus[window_size:-window_size]
    contexts = []

    for idx in range(window_size, len(corpus) - window_size):
        cs = []
        for t in range(-window_size, window_size + 1):
            if t:
                cs.append(corpus[idx + t])
        contexts.append(cs)
    return np.array(contexts), np.array(target)


def convert_one_hot(corpus, vocab_size):
    """ """
    N = corpus.shape[0]

    if corpus.ndim == 1:
        one_hot = torch.zeros((N, vocab_size))
        for idx, word_id in enumerate(corpus):
            one_hot[idx, word_id] = 1
    elif corpus.ndim == 2:
        C = corpus.shape[1]
        one_hot = torch.zeros((N, C, vocab_size))
        for idx_0, word_ids in enumerate(corpus):
            for idx_1, word_id in enumerate(word_ids):
                one_hot[idx_0, idx_1, word_id] = 1
    return one_hot


if __name__ == "__main__":
    text = "You say goodbye and I say hello."
    corpus, word2id, _ = preprocess(text)
    context, target = create_contexts_target(corpus)
    context_onehot = convert_one_hot(context, len(word2id))
    print(context_onehot)
    print(type(corpus))

    from collections import Counter

    print(Counter(corpus))
