import spacy
import pandas as pd
from nltk.corpus import stopwords
'''去掉停用词'''
def remove_stopwords(texts, stop_word_extention):
    return [[word for word in doc if word not in stop_word_extention] for doc in texts]
'''定义停用词'''

stop_words = stopwords.words('english')

tn_addtion = pd.read_csv('./TN_additional_stopwords.txt', header=None, names=['stop']).stop.to_list()
uspto_addtion = pd.read_csv('./USPTO_stopwords_en.txt', header=None, names=['stop']).stop.to_list()
technical_addtion = pd.read_csv('./technical_stopwords.txt', header=None, names=['stop']).stop.to_list()

# 停用词扩展
stop_word_extention = stop_words + tn_addtion + uspto_addtion + technical_addtion

nlp = spacy.load("en_core_web_lg", disable=['parser', 'ner'])
lemmatizer = nlp.get_pipe("lemmatizer")
print(lemmatizer.mode)
sentence = "Quantum state preparing method for quantum mechanical system, involves performing Hadamard transformation on qubits appended to vector stored in quantum computer register."
doc = nlp(sentence)
# print(" ".join([token.lemma_ for token in doc]))
text = [[token.lemma_ for token in doc]]
tex_no = remove_stopwords(text,stop_word_extention)
print(tex_no)

nlp = spacy.load("en_core_web_lg")
doc = nlp("Quantum state preparing method for quantum mechanical system, involves performing Hadamard transformation on qubits appended to vector stored in quantum computer register.")

for ent in doc.ents:
    print(ent.text, ent.label_)