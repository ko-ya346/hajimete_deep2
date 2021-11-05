"""
You say goodbye and I say hello.
という文章の
youとiの類似度を計算する
"""

from utils.utils import cos_similarity, create_co_matrix, most_similar, preprocess

text = "You say goodbye and I say hello."
corpus, word_to_id, id_to_word = preprocess(text)
print(word_to_id)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
print(C)

c0 = C[word_to_id["you"]]
c1 = C[word_to_id["i"]]
print(c0, c1)
print(cos_similarity(c0, c1))


for id in most_similar("you", word_to_id, id_to_word, C, top=5).tolist():
    print(id_to_word[id])
