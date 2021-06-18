
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

'''根据词性 词干化'''
def lemmatization(texts): # exclude
    """https://spacy.io/api/annotation"""
    texts_out = []
    nlp = spacy.load("en_core_web_sm",disable=['parser', 'ner'])
    lemmatizer = nlp.get_pipe("lemmatizer")
    print(lemmatizer.mode)
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc])
    return texts_out

'''去掉低频词'''
def word_frequency(corpus=[[]]):
    """
    :param corpus: a list of lists representing tokenized documents
    :return: a dict containing the frequency of each word in the corpus
    """
    frequency = defaultdict(int)
    for doc in corpus:
        for w in doc:
            frequency[w] += 1
    return dict(sorted(frequency.items(), key=operator.itemgetter(1), reverse=True))


#将list中的文本传送到TechNet进行网络化处理
def TechNet_f(setencce_):

  user_id =1142396
  word =setencce_
  content_len= len(word)
  url = "http://52.221.86.148/api/ideation/concepts/getGraph"

  headers = {
      "Accept": "application/json, text/javascript, */*; q=0.01",
      "Accept-Encoding": "gzip, deflate",
      "Content-Length": str(content_len),
      "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
      "Connection": "keep-alive",
      "Content-Type": "application/json",
      "Host": "52.221.86.148",
      "Origin": "http://www.innogps.com",
      "Referer": "http://www.innogps.com/",
      "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36"
  }
  data ={"userid" :user_id ,"word" :word}
  s = requests.Session()
  s.headers.update(headers)
  resp = s.post(url=url ,json=data)
  print(resp.text)
  graph_data=json.loads(resp.text)['graph_data']
  return graph_data

#获得文本网络，创建网络
def network_create_f(graph_data):
  link_list=[]
  for x in graph_data['links']:
    '''创建边列表'''
    link_list.append((x['source'],x['target']))

  node_id_name_cent=[]
  for x in graph_data['nodes']:
    '''提取id、name、cent'''
    node_id_name_cent.append([x['id'],x['name'],x['cent']])

  id_name_dict={}
  for x in node_id_name_cent:
    '''创建ID name 映射'''
    id_name_dict.update({x[0]:x[1]})

  node_pair =[]
  for x in link_list:
    '''创建节点匹配'''
    node_pair.append((id_name_dict[x[0]],id_name_dict[x[1]]))
  return node_pair




def main(input_path):

    #加载数据
    data_frame_qc = pd.read_csv(input_path)
    ti_data = data_frame_qc.NOVELTY.to_list()

    # 对文本进行预处理
    '''分词'''
    data_words = list(sent_to_words(ti_data))


    '''定义停用词'''

    stop_words = stopwords.words('english')
    tn_addtion = pd.read_csv('./TN_additional_stopwords.txt', header=None, names=['stop']).stop.to_list()
    uspto_addtion = pd.read_csv('./USPTO_stopwords_en.txt', header=None, names=['stop']).stop.to_list()
    technical_addtion = pd.read_csv('./technical_stopwords.txt', header=None, names=['stop']).stop.to_list()

    # 停用词扩展
    stop_word_extention = list(set(stop_words + tn_addtion + uspto_addtion + technical_addtion))
    stop_word_diy = ['involve','method','quickly','base','compute','computer']
    stop_word_extention = stop_word_extention+stop_word_diy

    '''词干化'''
    data_lemmatized_all = lemmatization(data_words)


    '''去掉停用词'''
    data_words_nostops = remove_stopwords(data_lemmatized_all,stop_word_extention)



    #在这里需要考虑是否需要去掉低频词
    # freq = word_frequency(data_words_nostops)
    # data_lemmatized = [[token for token in doc if freq[token] > 1] for doc in data_words_nostops]  # 词频大于1




    #获取预处理完了的词列表 重组为句子列表
    sentence_list = [[' '.join(x)] for x in data_words_nostops]

    #对预处理好的句子列表传送至TechNet进行处理

    for s in sentence_list:
        if len(s[0].split(' '))>5:
            graph_data = TechNet_f(s[0])
            time.sleep(5)
            node_pair = network_create_f(graph_data)
            # '''无向网洛'''
            # G = nx.Graph()
            # G.add_edges_from(node_pair)
            # nx.draw(G, node_color='r', edge_color='b')
            # nx.write_gexf(G,'./network/'+str(sentence_list.index(s))+'qc_test.gexf')
            '''有向网络'''
            G = nx.DiGraph()
            G.add_edges_from(node_pair)
            nx.write_gexf(G, './network/' + str(sentence_list.index(s)) + 'qc_test.gexf')

        else:
            print(s[0])
    # for s in ti_data:
    #     if len(s.split(' '))>5:
    #         graph_data = TechNet_f(s)
    #         time.sleep(5)
    #         node_pair = network_create_f(graph_data)
    #         G = nx.Graph()
    #         G.add_edges_from(node_pair)
    #         nx.draw(G, node_color='r', edge_color='b')
    #         nx.write_gexf(G,'./network/'+str(ti_data.index(s))+'qc_test.gexf')
    #     else:
    #         print(s)
if __name__== '__main__':
    main(r'D:\论文项目\量子计算\dii\ab_extract.csv')



