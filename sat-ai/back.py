import torch
from torch import nn
import string
from collections import defaultdict
import nltk
import copy
import re
device=torch.device('cuda')
nltk.download('punkt')
letters=string.printable
glove=torch.load('./glove50').to(device)
glove.weight.requires_grad=False
def make_embeddings(fname,save_name="glove50",vocab_fname='vocab'):
    with open(fname,'r') as e:
        vocab={}
        embeds=[]
        count=1
        for line in e:
            word=line.split(' ')[0]
            embed=line.split(' ')[1:]
            vocab[count]=word
            embeds.append(np.fromstring(embed,sep=' '))
            count+=1
        embedding=torch.tensor(np.stack(embeds)).float()
        embedder=nn.Embedding.from_pretrained(embedding)
        torch.save(save_name,embedder)
        with open(vocab_fname,'w') as w:
            w.write(str(vocab))
def get_embeddings(module_name,vocab_name):
    embedder=torch.load(module_name)
    with open(vocab_name,'r') as v:
        vocab=eval(next(v))
    return embedder,vocab
def get_vocab(name):
    with open(name,'r') as v:
        vocab=eval(next(v))
    return vocab
def rdict(src):
    return {src[i]:i for i in src}
def make_word_vocab(corpus):
    '''corpus should be list of split strings'''
    words={}
    for i in corpus:
        words=words.union(set(i))
    return {j:i+1 for i,j in enumerate(words)}
try:
    rvocab=get_vocab('vocab.txt')
except FileNotFoundError:
    rvocab=get_vocab('vocab copy.txt')
vocab=rdict(rvocab)
vocab=defaultdict(lambda:0,vocab)
torch.manual_seed(1)
#vocab={j:i+1 for i,j in enumerate(letters)}
#rvocab={vocab[i]:i for i in vocab}
#vocab=defaultdict(lambda:0,vocab)
#rvocab=defaultdict(lambda:'<idk>',vocab)
import re
def nothing(*args,**kwargs):
    return
def word2tensor(x):
    nums=[vocab[i] for i in x]
    return torch.tensor(nums) if len(x)>0 else torch.zeros(1)
def tensor2word(tens):
    return ''.join([rvocab[int(i)] for i in tens.reshape(-1)])
'''
def sdpa(q,k,v):
    # q:(b,seq1,d_k)
    # k:(b,seq2,d_k)
    # v:(b,seq2,d_v)
    return torch.bmm(torch.softmax(torch.bmm(q,k.transpose(1,2))/(q.shape[-1]**0.5),dim=-1),v)
def sdpa_scale(q,k,v):
    matmuled=torch.bmm(q,k.transpose(1,2))/(q.shape[0]**0.5)
    idxes=torch.triu_indices(q.shape[1],k.shape[1],1)
    matmuled[idxes]=float('-inf')
    return torch.bmm(torch.softmax(matmuled,dim=-1),v)
'''
class Sdpa(nn.Module):
    def __init__(self,nothing=None):
        super().__init__()
    def forward(self,q,k,v)->torch.Tensor:
        return torch.bmm(torch.softmax(torch.bmm(q,k.transpose(1,2))/(q.shape[-1]**0.5),dim=-1),v)
class Sdpa_scale(nn.Module):
    def __init__(self,nothing=None):
        super().__init__()
    def forward(self,q,k,v)->torch.Tensor:
        matmuled=torch.bmm(q,k.transpose(1,2))/(q.shape[0]**0.5)
        idxes=torch.triu_indices(q.shape[1],k.shape[1],1)
        matmuled[idxes]=float('-inf')
        return torch.bmm(torch.softmax(matmuled,dim=-1),v)
a=Sdpa()
b=torch.randn(2,3,4)
print(a(b,b,b).shape)
class mhead_atten(nn.Module):
    def __init__(self,d_k,d_v=None,hk=None,hv=None,heads=8,mask=False,batch_first=False):
        '''
        d_k:input size of key and query 
        d_v: input size of value
        hk: head size of key and query
        hv: head size of value
        heads: how many heads
        '''
        super().__init__()
        if d_v is None:
            d_v=d_k
        if hk is None:
            hk=d_k
        if hv is None:
            hv=d_v
        self.kw=nn.Linear(d_k,hk*heads)
        self.vw=nn.Linear(d_v,hv*heads)
        self.qw=nn.Linear(d_k,hk*heads)
        self.union=nn.Linear(hv*heads,d_v)
        self.head_size=hk
        self.hv=hv
        self.heads=heads
        if mask:
            self.attention=Sdpa_scale()
        else:
            self.attention=Sdpa()
        self.bfirst=batch_first
    def forward(self,q,k,v):
        if self.bfirst:
            batch,seq,dk=q.shape
            vbatch,vseq,vk=v.shape
        else:
            seq,batch,dk=q.shape
            vseq,vbatch,vk=v.shape
        dv=v.shape[-1]
        q=self.qw(q).reshape(batch,seq,self.heads,self.head_size)
        k=self.kw(k).reshape(vbatch,vseq,self.heads,self.head_size)
        v=self.vw(v).reshape(vbatch,vseq,self.heads,self.hv)
        qt=q.transpose(1,2).reshape(self.heads*batch,seq,self.head_size)
        kt=k.transpose(1,2).reshape(self.heads*batch,seq,self.head_size)
        vt=v.transpose(1,2).reshape(self.heads*vbatch,seq,self.hv)
        attended=self.attention(qt,kt,vt).reshape(batch,self.heads,vseq,self.hv).transpose(1,2)
        return self.union(attended.reshape(batch,vseq,self.hv*self.heads))
class EncoderLayer(nn.Module):
    def __init__(self,d_v,attention,expanded=1024,*args,**kwargs):
        super().__init__()
        self.attention=attention
        self.inputs=(d_v,attention,expanded)
        self.ff=nn.Sequential(
            nn.Linear(d_v,expanded),
            nn.LeakyReLU(),
            nn.Linear(expanded,d_v)
        )
        self.norm1=nn.LayerNorm(d_v)
        self.norm2=nn.LayerNorm(d_v)
    def forward(self,x):
        attended=self.attention(x,x,x)
        normed1=self.norm1(attended+x)
        return self.norm2(self.ff(normed1)+x)

class Encoder(nn.Module):
    def __init__(self,encoder_block,depth):
        super().__init__()
        self.instance=encoder_block
        self.transformers=nn.Sequential(*[copy.deepcopy(self.instance) for i in range(depth)])
    def forward(self,x):
        return self.transformers(x)
class DecoderLayer(nn.Module):
    def __init__(self,d_v,masked_attention,attention,expanded=1024,*args,**kwargs):
        super().__init__()
        self.masked_attention=masked_attention 
        self.attention=attention
        self.ff=nn.Sequential(
            nn.Linear(d_v,expanded),
            nn.LeakyReLU(),
            nn.Linear(expanded,d_v)
        )
        self.norm1=nn.LayerNorm(d_v)
        self.norm2=nn.LayerNorm(d_v)
        self.norm3=nn.LayerNorm(d_v)
    def forward(self,encoded,x):
        mattended=self.masked_attention(x,x,x)
        normed1=self.norm1(mattended+x)
        combine_attended=self.attention(encoded,x,x)
        normed2=self.norm2(combine_attended+normed1)
        return self.norm2(self.ff(normed2)+normed2)
class Decoder(nn.Module):
    def __init__(self,layer,depth):
        super().__init__()
        self.transformers=nn.ModuleList([copy.deepcopy(layer) for i in range(depth)])
    def forward(self,encoded,x):
        for i in self.transformers:
            x=i(encoded,x)
        return x
class Transformer(nn.Module):
    def __init__(self,d_v,out,encoder,decoder):
        super().__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.fc=nn.Linear(d_v,out)
    def forward(self,inputs,outputs):
        encoded=self.encoder(inputs)
        attended=self.decoder(encoded,outputs)
        return self.fc(attended)
class sat_transformerv1(nn.Module):
    def __init__(self,word_embed,text_encoder,qencoder,question_decoder,chooser,out):
        super().__init__()
        self.tencoder=text_encoder
        self.qdecoder=question_decoder
        #print(question_decoder,'qd')
        #print(chooser,'chooser')
        self.chooser=chooser
        self.word_embed=glove#nn.Embedding(len(vocab),word_embed)
        self.fc=nn.Linear(word_embed,out)
        self.pos_embed=nn.Embedding(6000,word_embed)
        self.idxs=torch.arange(600).to(device)
        self.qencoder=qencoder
    def first_forward(self,text,question,choice):
        text_embedded=self.word_embed(text).transpose(0,1)
        tb,ts,tk=text_embedded.shape
        question_embedded=self.word_embed(question).transpose(0,1)
        qb,qs,qk=question_embedded.shape
        echoice=self.word_embed(choice).transpose(0,1)
        cb,ct,ck=echoice.shape
        echoice+=self.pos_embed(self.idxs[:ct]).expand(cb,ct,ck).to(device)
        text_embedded+=self.pos_embed(self.idxs[:ts]).expand(tb,ts,tk).to(device)
        question_embedded+=self.pos_embed(self.idxs[:qs]).expand(qb,qs,qk).to(device)
        text_encoded=self.tencoder(text_embedded)
        #print('qembed',question_embedded.shape,'tembed',text_embedded.shape,'tenc',text_encoded.shape)
        qencoded=self.qencoder(question_embedded)
        enc1=self.qdecoder(text_encoded,qencoded)
        #print('qdecode',enc1.shape,'choice',choice.shape)
        final=self.chooser(enc1,echoice)
        return torch.sigmoid(self.fc(final).mean(0)), enc1
    def second_forward(self,enc1,choice):
        echoice=self.word_embed(choice).transpose(0,1)
        cb,ct,ck=echoice.shape
        echoice+=self.pos_embed(self.idxs[:ct]).expand(cb,ct,ck).to(device)
        final=self.chooser(enc1,echoice)
        return torch.sigmoid(self.fc(final).mean(0))
    def forward(self,*args):
        if len(args)==2:
            return self.second_forward(*args)
        else:
            return self.first_forward(*args)

def mapfn(src,fn,out=list):
    return out(map(fn,src))

def shift_convolver(corpus,minimum=1,to_tensor=True):
    res=[]
    if to_tensor:
        fn=lambda x: word2tensor(x).to(device)
    else:
        fn=lambda x: x
    for i in range(minimum,len(corpus)-1):
        pair=(word2tensor(corpus[0:i]),word2tensor(corpus[0:i+1]))
        res.append(pair)
    return res
def one_ahead_convolver(corpus,minimum=1):
    res=[]
    for i in range(minimum,len(corpus)-1):
        pair=(word2tensor(corpus[0:i]).reshape(1,-1),word2tensor(corpus[i+1]).reshape(1,-1))
        res.append(pair)
    return res
def shfit_convolver(corpus,minimum):
    res=[]
    for i in range(minimum,len(corpus)-1):
        pair=(word2tensor(corpus[0:i]).reshape(1,-1),word2tensor(corpus[:i+1]).reshape(1,-1))
        res.append(pair)
    return res

def rtransformer_validate(data,model,input_window=None,do_print=False):
    if input_window is None:
        input_window=data[0][0][0].shape[-1]-1
    avgcorrect=0
    count=0
    with torch.no_grad():
        for inputs,outputs in data:
            count+=1
            to_encode=inputs[:,:input_window]
            pred=model(to_encode,inputs[:,input_window:])
            m,idx=torch.max(pred,dim=-1)
            string_pred=tensor2word(idx)
            string_output=tensor2word(output)
            percent=0
            for i in range(len(string_pred)):
                if string_pred[i]==string_output[i]:
                    percent+=1
            avgcorrect+=percent/len(string_pred)
            if do_print:
                print(string_pred)
    return avgcorrect/count

def rtransformer_train(data,validation,model,input_window=None):
    '''use input_window=-1 for rnn-like transformers with one decoder input'''
    optim=torch.optim.SGD(model.parameters(),lr=0.01)
    lossf=nn.BCELoss()
    if input_window is None:
        input_window=data[0][0][0].shape[-1]-1
    for e in range(30):
        for entry in data:
            for inputs,outputs in entry:
                to_encode=inputs[:,:input_window]
                pred=model(to_encode,inputs[:,input_window:]).transpose(1,2)
                loss=lossf(pred,outputs)
                optim.zero_grad()
                loss.backward()
                optim.step()
        print(rtransformer_validate(data[0],model,input_window))

df=open('./SAT Suite Question Bank-Results-6q.pdf','rb')
r=pdt.PDF(df)
r=[i.lower() for i in r]
def check_if_good(q):
    return ('which choice provides the best evidence' not in q and
            True)
def remove_beginning_chars(line):
    try:
        for i in range(len(line)):
            if len(line[i])!=0 and len(line[i+1])!=0:
                return line[i:]
    except:
        print(line)
        return [line[-1]]

def get_passage(text):
    split=text.split('\n')
    #print(text)
    #print(repr(text))
    #print([repr(k) for k in split])
    for i in range(2,len(split)):
        #print(split[i])
        if 'line' in split[i] or 'by' in split[i-1] or 'by' in split[i-2]:
            good=split[i:]
            #print(good)
            return flatten_list([nltk.word_tokenize(i) for i in good])[1:]
    return 
def flatten_list(x):
    res=[]
    for i in x:
        res+=i
    return res
def get_question_and_answer(text):
    split=text.split('\n')
    question=nltk.word_tokenize(split[0])
    answers=[nltk.word_tokenize(i[6:]) for i in split[1:5]]
    found=re.search('choice [abcd]',text)
    #print(found)
    #print(text[found])
    if found is None:
        print(text)
        assert False
    correct_answer=text[found.end()-1]
    return (question,answers,correct_answer)

def rfind(source,target):
    idxs=[]
    i=source.find(target)
    idxs.append(i)
    while i!=-1:
        i=source.find(target,i+1)
        idxs.append(i)
    return idxs
letter2choice={"a":0,"b":1,'c':2,'d':3}
def sat_validate(model,data):
    total=0
    corrects=0
    model.eval()
    with torch.no_grad():
        for passage,question,answers,correct in data:
            preds=[]
            ans=answers[0]
            pred,enc=model(passage,question,ans)
            preds.append(float(pred))
            for ans in answers[1:]:
                pred=model(enc,ans)
                preds.append(float(pred))
            answered=preds.index(max(preds))
            if answered==letter2choice[correct]:
                corrects+=1
            total+=1
    model.train()
    print(corrects,total,corrects/total,preds,correct)
    return total,corrects
def extract_vocab(file):
    seen=set()
    try:
        for word in file:
            seen=seen.union(set(word))
    except:
        print(file,word)
        raise
    return {j:i+1 for i,j in enumerate(seen)}

#vocab=extract_vocab([get_passage(r[i]) for i in range(0,len(r),2)])
scale_by=[1/4,3/4]
scale_by=[16/9,16]
def sat_train(model,train,val,optim,epochs=30):
    lossf=nn.BCELoss()
    for e in range(epochs):
        avgloss=0
        for passage,question,answers,correct in train:
            all_loss=torch.tensor(0.,requires_grad=False).to(device)
            predictions=[]
            count=0
            ans=answers[0]
            pred,enc=model(passage,question,answers[0])
            pred=model(enc,ans)
            target=torch.tensor([float(letter2choice[correct]==count)])
            loss=lossf(pred,target.reshape(-1,1).to(device))
            all_loss+=loss
            #optim.zero_grad()
            #loss.backward()
            #optim.step()
            count+=1
            avgloss+=float(loss)
            predictions.append(float(pred))
            #print(target,'target')
            for ans in answers[1:]:
                pred=model(enc,ans)
                target=torch.tensor([float(letter2choice[correct]==count)])
                #print(target,'target','count',count,'correct',correct)
                loss=lossf(pred,target.reshape(-1,1).to(device))
                all_loss+=loss*scale_by[int(target)]
                #optim.zero_grad()
                #loss.backward()
                #optim.step()
                count+=1
                avgloss+=float(loss)
                predictions.append(float(pred))
            #print(all_loss)
            all_loss.backward()
            optim.step()
            optim.zero_grad()
        sat_validate(model,val)
        print(f'epoch: {e}','avg',avgloss/len(train),predictions,correct)

def make_full_dataset(file):
    res=[]
    for i in range(0,len(file),2):
        q,ans,c=get_question_and_answer(file[i+1])
        if check_if_good(file[i+1].split('\n')[0]):
            passage=get_passage(file[i])
            #print(passage)
            #print(file[i])
            numbered_passage=torch.tensor(mapfn(passage,lambda x: vocab[x])).reshape(1,-1).to(device)
            numbered_q=torch.tensor(mapfn(q,lambda x: vocab[x])).reshape(1,-1).to(device)
            #print(ans)
            numbered_ans=[torch.tensor(mapfn(i,lambda x: vocab[x])).reshape(1,-1).to(device) for i in ans]
            res.append((numbered_passage,numbered_q,numbered_ans,c))
    return res
def make_data(fname):
    with open(fname,'rb') as df:
        r=pdt.PDF(df)
        r=[i.lower() for i in r]
        return make_full_dataset(r)
res1=make_full_dataset(r)
res2=make_data('./SAT Suite Question Bank-20q.pdf')
res3=make_data('./SAT Suite Question Bank-20q2.pdf')
both=res1+res2+res3
