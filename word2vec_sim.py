from gensim.models.word2vec import Word2Vec, LineSentence
from gensim import corpora, models, similarities
from data_helper import data_helper


class word2vec_sim(object):
    def __init__(self):
        pass

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

    def similarity(self):
        pass
