import pickle
from MeCab import Model
from gensim.models import KeyedVectors, word2vec, Word2Vec

model_path = 'jawiki.all_vectors.300d.txt'

model = word2vec.KeyedVectors.load_word2vec_format(model_path)

print('FUCK YOU')

model.wv.save('model.kv')



#results = model.most_similar(positive='王', negative='男', topn=10)

#print(results)