
import pandas as pd
import gensim
from nltk.corpus import stopwords
import spacy
from collections import defaultdict
import operator
import requests
import json
import time
import networkx as nx



#对文本进行预处理
'''分词'''
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
'''去掉停用词'''
def remove_stopwords(texts, stop_word_extention):
    return [[word for word in doc if word not in stop_word_extention] for doc in texts]



def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
    lemmatizer = nlp.get_pipe("lemmatizer")
    print(lemmatizer.mode)
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc])
    return texts_out
'''根据词性 词干化'''



def main(input_path):

    #加载数据
    data_frame_qc = pd.read_csv(input_path)
    ti_data = data_frame_qc.USE.to_list()

    # 对文本进行预处理
    '''分词'''
    data_words = list(sent_to_words(ti_data))

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=1, threshold=1)  # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=1)

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    # See trigram example
    # print(trigram_mod[bigram_mod[data_words[0]]])


    '''定义停用词'''

    stop_words = stopwords.words('english')
    tn_addtion = pd.read_csv('./TN_additional_stopwords.txt', header=None, names=['stop']).stop.to_list()
    uspto_addtion = pd.read_csv('./USPTO_stopwords_en.txt', header=None, names=['stop']).stop.to_list()
    technical_addtion = pd.read_csv('./technical_stopwords.txt', header=None, names=['stop']).stop.to_list()

    # 停用词扩展
    stop_word_extention = list(set(stop_words + tn_addtion + uspto_addtion + technical_addtion))
    stop_word_diy = ['involve','method','quickly','base','claimed']
    stop_word_extention = stop_word_extention+stop_word_diy


    '''去掉停用词'''
    data_words_nostops = remove_stopwords(data_words,stop_word_extention)

    '''词干化'''
    data_lemmatized_all = lemmatization(data_words_nostops)




    sentence_list = [[' '.join(doc)] for doc in data_lemmatized_all]

    for x in sentence_list:
        if len(x[0])>5:
            print (x[0].strip())



    # '''词干化'''
    # data_lemmatized_all = lemmatization(data_words)

if __name__== '__main__':
    main(r'D:\论文项目\量子计算\dii\ab_extract.csv')
