import pickle
from MeCab import Model
from gensim.models import KeyedVectors, word2vec, Word2Vec

new_model_path = 'model.pkl'
model_path = 'jawiki.all_vectors.300d.txt'

with open(new_model_path, 'rb') as f:
    model = pickle.load(f)
    #model = word2vec.KeyedVectors.load_word2vec_format(new_model_path, binary=True)
    #model = word2vec.Word2Vec.load(new_model_path)
    #model = KeyedVectors.load_word2vec_format(new_model_path, binary=True)
    #model = word2vec.KeyedVectors.load_word2vec_format(model_path)

results = model.most_similar(positive=['王', '女'], negative=['男'], topn=10)

print(results)