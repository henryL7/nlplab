import pickle
import prepare
import torch
import utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from gensim.models import word2vec
import gensim.models as models
import torch
import time

word_size=200
embedding_size=50
linear_size=20


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(word_size,embedding_size,1,True,True,0.5,True)
        self.linear=nn.Linear(embedding_size*2,linear_size,True)
        self.linear1=nn.Linear(linear_size,4,True)
        self.softmax=nn.LogSoftmax()
        
    def forward(self, inputs):
        inter,_ = self.lstm(inputs)
        #shape=inter.data.size()
        #length=shape[1]
        #out=Variable(torch.FloatTensor(length,embedding_size*2))
        #for i in range(length)
        #print(inter.data.size())
        vec=inter[0]
        out=self.linear(vec)
        out=self.linear1(F.relu(out))
        out=self.softmax(out)
        return out

def a2ft(x):
    return Variable(torch.FloatTensor(x))

def a2lt(x):
    return Variable(torch.LongTensor(x))

def vec_flat(x):
    y=[]
    for x_i in x:
        y+=x_i
    return y

def lstm_test(sents,tags):
    model=LSTM()
    model.load_state_dict(torch.load("blstm.pkl"))
    tags=vec_flat(tags)
    tags_p=[]
    for s in sents:
        out=model(torch.unsqueeze(a2ft(s),0))
        out=out.data.numpy()
        for out_i in out:
            max_idx=np.argmax(out_i)
            tags_p.append(max_idx)
    utils.print_score(tags,tags_p)
    return utils.get_score(tags,tags_p)

def lstm_cross(train_chars,train_tags,test_chars,test_tags,epoch=10,batch_size=25):
    length=len(train_chars)
    model=LSTM()
    count=0
    criterion = nn.NLLLoss(size_average=True)
    optimizer = optim.Adadelta(model.parameters())
    time1=time.clock()
    while count<epoch:
        i=0
        while i+batch_size<length:
            loss=a2ft([0])
            for j in range(batch_size):
                out=model(torch.unsqueeze(a2ft(train_chars[i+j]),0))
                loss+=criterion(out,a2lt(train_tags[i+j]))
            i+=batch_size
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(model.lstm.parameters(),5)
            optimizer.step()
            if i%100:
                print(loss.data[0])
        count+=1
        
    time2=time.clock()
    print(time2-time1)
    torch.save(model.state_dict(),"blstm.pkl")
    return lstm_test(test_chars,test_tags)

def lstm_cross_cuda(train_chars,train_tags,test_chars,test_tags,epoch=10,batch_size=25):
    length=len(train_chars)
    model=LSTM()
    model=model.cuda()
    count=0
    criterion = nn.NLLLoss(size_average=True).cuda()
    optimizer = optim.Adadelta(model.parameters())
    time1=time.clock()
    while count<epoch:
        i=0
        while i+batch_size<length:
            loss=a2ft([0]).cuda()
            for j in range(batch_size):
                out=model(torch.unsqueeze(a2ft(train_chars[i+j]).cuda(),0))
                loss+=criterion(out,a2lt(train_tags[i+j]).cuda())
            i+=batch_size
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(model.lstm.parameters(),5)
            optimizer.step()
            if i%100:
                print(loss.data[0])
        count+=1
        
    time2=time.clock()
    print(time2-time1)
    torch.save(model.state_dict(),"blstm.pkl")
    return lstm_test(test_chars,test_tags)


if __name__ == "__main__":
    word_dict=word2vec.Word2Vec.load("worddict.dic")
    train_chars,train_tags,test_chars,test_tags=prepare.get_train_test()
    length=len(train_chars)
    __train_chars=[]
    for t in train_chars:
        __train_chars.append([word_dict[t_i].tolist() for t_i in t])
    train_chars=__train_chars
    __test_chars=[]
    for t in test_chars:
        __test_chars.append([word_dict[t_i].tolist() for t_i in t])
    test_chars=__test_chars
    model=LSTM()
    epoch=10
    count=0
    batch_size=25
    criterion = nn.NLLLoss(size_average=True)
    optimizer = optim.Adadelta(model.parameters())
    while count<epoch:
        i=0
        while i+batch_size<length:
            loss=a2ft([0])
            for j in range(batch_size):
                out=model(torch.unsqueeze(a2ft(train_chars[i+j]),0))
                loss+=criterion(out,a2lt(train_tags[i+j]))
            i+=batch_size
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(model.lstm.parameters(),5)
            optimizer.step()
            if i%100:
                print(loss.data[0])
        count+=1
    
    torch.save(model.state_dict(),"blstm.pkl")
    lstm_test(test_chars,test_tags)


