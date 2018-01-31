from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from gensim.models import word2vec
import gensim.models as models
import numpy as np
import prepare
import time

def print_score(y,y_pred):
    tag_names=["S","B","M","E"]
    print(classification_report(y, y_pred, target_names=tag_names))

# compute precision,recall,f score
def get_score(y,y_pred):
    score=precision_recall_fscore_support(y, y_pred, average='micro')
    return score[0],score[1],score[2]

def cross_validation(func,folders=5):
    word_dict=word2vec.Word2Vec.load("worddict.dic")
    sents,tags=prepare.data_split()
    __sents=[]
    for t in sents:
        __sents.append([word_dict[t_i].tolist() for t_i in t])
    sents=__sents
    sets=list(zip(sents,tags))
    split_size=int(len(sets)/folders)
    splits=[]
    for i in range(folders):
        splits.append(sets[i*split_size:(i+1)*split_size])
    recall,precision,f=0,0,0
    for i in range(folders):
        test_set=splits[i]
        train_set=[]
        for j in range(folders):
            if j!=i:
                train_set+=splits[j]
        train_sents=[s[0] for s in train_set]
        train_tags=[s[1] for s in train_set]
        test_sents=[s[0] for s in test_set]
        test_tags=[s[1] for s in test_set]
        __recall,__precision,__f=func(train_sents,train_tags,test_sents,test_tags,epoch=1)
        recall+=__recall
        precision+=__precision
        f+=__f
    print("recall:%f%%"%(recall*100/folders))
    print("precision:%f%%"%(precision*100/folders))
    print("f measure:",f/folders)
        
