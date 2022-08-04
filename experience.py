import pickle
from gensim.models import KeyedVectors

model_path = 'jawiki.all_vectors.300d.txt'
model = KeyedVectors.load_word2vec_format(model_path, binary=False)

new_model_path = 'model.pkl'

print('meow')

with open(new_model_path, 'wb') as f:
    pickle.dump(model, f)