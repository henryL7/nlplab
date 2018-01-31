import Reader
import pickle
from gensim.models import word2vec
import gensim.models as models

def tagging(w):
    w_tags=[]
    if len(w)==1:
        w_tags.append((w[0],"S"))
    else:
        w_tags.append((w[0],"B"))
        i=1
        while i<len(w)-1:
            w_tags.append((w[i],"M"))
            i+=1
        w_tags.append((w[i],"E"))
    return w_tags

def get_sents():
    return pickle.load(open("sents.pkl","rb"))

def get_tags():
    return pickle.load(open("tags.pkl","rb"))

def get_tag_vecs():
    return pickle.load(open("tag_vecs.pkl","rb"))

def data_split(sents=get_tag_vecs()):
    chars,tags=[],[]
    for sent in sents:
        chars.append([t[0] for t in sent])
        tags.append([t[1] for t in sent])
    return chars,tags

def get_train_test():
    tuples=get_tag_vecs()
    size=int(len(tuples)*0.2)
    train_tuples=tuples[size:]
    test_tuples=tuples[:size]
    train_chars,train_tags=data_split(train_tuples)
    test_chars,test_tags=data_split(test_tuples)
    return train_chars,train_tags,test_chars,test_tags


def train_word2vec():
    sents=get_sents()
    model=word2vec.Word2Vec(sents,size=200,window=5,min_count=0,workers=2,iter=1000)
    model.save("worddict.dic")

tag2vec={"S":0,"B":1,"M":2,"E":3}

if __name__ == "__main__":

    seg_reader=Reader.Reader("./LDC2010T07/ctb7.0/data/utf-8/")
    text,_=seg_reader.readPos()
    punc=["，","。","？","！","《","》","【","】","：","；","、","“","”","’","‘",""]
    punc=set(punc)
    start=u"4e00"
    end=u"9fa5"
    tags=[]
    sents=[]
    for s in text:
        s_tags=[]
        sent=[]
        for w in s:
            s_tags+=tagging(w)
            sent+=w
        tags.append(s_tags)
        sents.append(sent)
    vecs=[]
    for t in tags:
        vecs.append([(t_i[0],tag2vec[t_i[1]]) for t_i in t])
    print(vecs)
    
    pickle.dump(tags,open("tags.pkl","wb"))
    pickle.dump(sents,open("sents.pkl","wb"))
    pickle.dump(vecs,open("tag_vecs.pkl","wb"))
    
    
            

