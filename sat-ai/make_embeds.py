import torch
import numpy as np
import json
def make_embeddings(fname,save_name="glove100",vocab_fname='vocab'):
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
