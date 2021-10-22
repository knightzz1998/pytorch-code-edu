# -*- coding: utf-8 -*-
# 作者 : 王天赐

from gensim.models import Word2Vec

word2vec_model_path = "out/weibo_word2vec.model"


class Word2vecModel():
    def __init__(self) -> None:
        self.model_path = word2vec_model_path
        self.model = Word2Vec.load(self.model_path)
        self.vectors = self.model.wv
        self.vectors_array = self.model.wv.vectors
        self.word_list = self.vectors.index_to_key
        self.word_dict = self.vectors.key_to_index

    def get_model(self):
        return self.model

    def get_word_dict(self):
        return self.word_dict

    def get_word_list(self):
        return self.word_list

    def get_vectors(self):
        return self.vectors

    def get_vectors_array(self):
        return self.vectors_array
