from gensim.models import word2vec
import gensim.models as models

#model=word2vec.Word2Vec.load("worddict.dic")
#r=model.similarity("左","右")
#print(r)

import blstm
import utils

utils.cross_validation(blstm.lstm_cross,5)