from gensim.models.word2vec import Word2Vec, LineSentence
from gensim import corpora, models, similarities
from data_helper import data_helper
from distance_helper import distance_helper
import jieba
import numpy as np


class word2vec_sim(object):
    def __init__(self):
        self.newwords = []
        self.stopwords = []

    def word2vec_train(self, train_data_path, model_path):
        train_data = data_helper.get_train_data(train_data_path)
        # train_data is list of sentence
        model = Word2Vec(train_data, hs=1, min_count=1, window=3, size=100)
        model.save(model_path)

    def tfidf_train(self, train_data_path, model_path):
        # https://blog.csdn.net/m0_37306360/article/details/76826121
        train_data = data_helper.get_train_data(train_data_path)
        # train_data is list of split sentence
        # get dictionary
        dictionary = corpora.Dictionary(train_data)
        corpus = [dictionary.doc2bow(text) for text in train_data]
        model = models.TfidfModel(corpus)
        model.save(model_path)

    def LDA_train(self, train_data_path, model_path, tfidf_model_path, num_topics=100):
        train_data = data_helper.get_train_data(train_data_path)
        # train_data is list of split sentence
        # get dictionary
        dictionary = corpora.Dictionary(train_data)
        corpus = [dictionary.doc2bow(text) for text in train_data]
        tfidf_model = models.TfidfModel.load(tfidf_model_path)
        corpus_tfidf = tfidf_model[corpus]
        model = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=num_topics)
        model.save(model_path)

    def LSI_train(self, train_data_path, model_path, tfidf_model_path, num_topics=100):
        train_data = data_helper.get_train_data(train_data_path)
        # train_data is list of split sentence
        # get dictionary
        dictionary = corpora.Dictionary(train_data)
        corpus = [dictionary.doc2bow(text) for text in train_data]
        tfidf_model = models.TfidfModel.load(tfidf_model_path)
        corpus_tfidf = tfidf_model[corpus]
        model = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=num_topics)
        model.save(model_path)

    def sentence_split(self, sentence):
        try:
            for newword in self.newwords:
                jieba.add_word(newword)
            sentence_seged = jieba.cut(sentence.strip())
            outstr = ''
            for word in sentence_seged:
                word = word.lower()
                if word not in self.stopwords:
                    if word != '\t':
                        outstr += word + " "
            return outstr
        except:
            raise TypeError(sentence)

    def get_word_vector(self, text, word2vec_model, tfidf_model, tfidf_dictionary):
        split_text = [w for w in text if w in word2vec_model.wv.vocab]
        vec = tfidf_dictionary.doc2bow(split_text)
        vec_dict = dict(tfidf_model[vec])
        tfidf_vec = []
        for word in split_text:
            tfidf_vec.append(vec_dict[tfidf_dictionary.token2id[word]])
        tfidf_vec = np.array(tfidf_vec).T
        word2vec_vec = word2vec_model[split_text].T
        word2vec_tfidf_vec = np.dot(word2vec_vec, tfidf_vec)
        return word2vec_tfidf_vec

    def similarity_word2vec_tfidf(self, text1, text2, word2vec_model_path, tfidf_model_path, tfidf_dictionary_path):
        """
        calculate similarity between text1 and text2 by word2vec with TF-IDF weights
        :return:
        """
        word2vec_model = Word2Vec.load(word2vec_model_path)
        tfidf_dictionary = corpora.Dictionary.load(tfidf_dictionary_path)
        tfidf_model = models.TfidfModel.load(tfidf_model_path)
        text1 = self.sentence_split(text1).split()
        text2 = self.sentence_split(text2).split()
        vector1 = self.get_word_vector(text1, word2vec_model, tfidf_model, tfidf_dictionary)
        vector2 = self.get_word_vector(text2, word2vec_model, tfidf_model, tfidf_dictionary)
        return distance_helper.cos_distance(vector1, vector2)
