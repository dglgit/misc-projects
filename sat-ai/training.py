import torch 
from torch import nn
from back import *
torch.manual_seed(1)
torch.cuda.manual_seed(1)
#d_k,d_v=None,hk=None,hv=None,heads=8,mask=False
atten_kwargs={'d_k':100,'hk':25,'heads':8}
'''
passage_encoder=Encoder(EncoderLayer(100,expanded=1024,attention=mhead_atten(**atten_kwargs)),8)
question_decoder=Decoder(DecoderLayer(100,attention=mhead_atten(**atten_kwargs),\
    masked_attention=mhead_atten(mask=True,**atten_kwargs)),8)
qencoder=Encoder(EncoderLayer(100,expanded=1024,attention=mhead_atten(**atten_kwargs)),8)
choice_decoder=Decoder(DecoderLayer(100,attention=mhead_atten(**atten_kwargs),\
                                    masked_attention=mhead_atten(mask=True,**atten_kwargs)),8)
spassage_encoder=torch.jit.script(passage_encoder)
squestion_decoder=torch.jit.script(question_decoder)
schoice_decoder=torch.jit.script(choice_decoder)
###
'''
passage_encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(100,10),12)
question_decoder=nn.TransformerDecoder(nn.TransformerDecoderLayer(100,10),12)
choice_decoder=nn.TransformerDecoder(nn.TransformerDecoderLayer(100,10),12)
qencoder=torch.jit.script(nn.TransformerEncoder(nn.TransformerEncoderLayer(100,10),12))
spassage_encoder=torch.jit.script(passage_encoder)
squestion_decoder=torch.jit.script(question_decoder)
schoice_decoder=torch.jit.script(choice_decoder)
#'''
model=sat_transformerv1(100,spassage_encoder,qencoder,squestion_decoder,schoice_decoder,1).to(device)
optim=torch.optim.SGD(model.parameters(),0.001,momentum=0.7)
sat_train(model,both[:-10],both[-10:],optim=optim,epochs=50)
