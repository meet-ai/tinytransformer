#code:utf-8
import torch 
from torch import nn
import math
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
from utils import *
import torch.nn.functional as F
from memory_profiler import profile
import lightning as L
from torch import optim
#torch.cuda.memory._record_memory_history()

#writer = SummaryWriter('./result_tensorboard')



class AttentionHead(nn.Module):
    #head 是子空间的概念,意思是让输入的形式更多样化. 灵感来自于人类可以多个维度的去输入数据来进行学习.
    #不切分 head 计算量会比较大,类似于 group wise 
    def __init__(self,hidden_dim, mask=None, dropout=0.1):
        super(AttentionHead, self).__init__()
        # 这里使用了相同的 Linear hidden 和 output
        # 避免句子经过多次 att 之后会膨胀.
        self.Q = nn.Linear(hidden_dim, hidden_dim)
        self.K = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, hidden_dim) 
        if dropout is not None:
            self.dropout = dropout
            self.dp = nn.Dropout(dropout)
        self.mask = mask

    #兼容 self-attention 和 cross-attention 
    @profile
    def forward(self, q, k, v, mask=None):
        #output shape  [seq_len, hidden]
        query = self.Q(q)
        key = self.K(k)
        value = self.V(v)

        #注意力的结算过程
        # V* (Q*K/sqrt(length_of_input_dim)
        # matmul vs Linear 一个是可学习的算子 一个是单纯的计算
        # Q.shape = [seq_len, dim_of_embedding]
        scores = torch.matmul(query, key.transpose(-2, -1)) / (key.size(-1) ** 0.5)

        # mask 有两种,一种是 encoder 的 mask , 用来把 pad=0 转换成接近极小的数，这样softmax算法不会因为 e**0 失效. 
        # 第二种是 decoder 的 mask 用来把未来 token 隐藏掉. 原理是每一个生成的 token 是用各单词的权重*V ,把权重里面未来token去掉就不会利用到未来数据
        if mask!=None:
            # mask 的shape 和 scores 的 shape 不一样. 但是因为维度广播机制, 会对shape进行填充.
            # masked_fill 把 mask 中 True 的位置进行替换
            scores.masked_fill(mask,-1e9)


        #softmax 是多分类的激活函数, 把多个分类分布的概率
        #softmax 使用最后一个维度作为分类的维度
        # 这里不是分类问题为何也用 softmax 
        # 点积得到的结果 shape 是 SentenceASize, SetenceBSize  一般Size是相同的
        # F.softmax 得到的结果是A*B softmax 在 B 在每个单词上相对于A的相关性
        # value 和 query 是否 
        
        # dropout 一般用在复杂算子的后面,来让神经元泛化性更强
        attention_weights = F.softmax(scores, dim=-1)
        if self.dropout != None:
            attention_weights = self.dp(attention_weights)

        #q,k,v sequence_len 是同一个值,保证 atten_weights 能和 value 计算
        
        # [seq_len,sql_len] x [seq_len,hidden]
        # output [seq_len,hidden]
        # 这里为了能计算, value 是被点乘数
        # softmax(score) * v 的本质是, 重新使用上下文的相关性权重，来在每一个 hidden_dim index 上重组每一个单词，这样就相当于新组织了一局携带上下文信息的话
        context = torch.matmul(attention_weights, value)
        
        return context 



class MultiHeadAttention(nn.Module):
    def __init__(self, headcnt, hidden_dim):
        #super不需要代入参数
        super(MultiHeadAttention,self).__init__()
        print("MultiHeadAttention")
        #multi-head-attention 有一个优化方案,不用 for loop 把 concat 的特征放在一个矩阵里面只执行一次 attention 操作就可以
        #这里先以逻辑优先把思路串通
        #print(headcnt,hidden_dim)
        self.headcnt = headcnt
        self.mha = nn.ModuleList([AttentionHead(hidden_dim//headcnt) for i in range(headcnt)])
        self.join_linear = nn.Linear(hidden_dim, hidden_dim)
    
    #@profile
    def forward(self,q,k,v,mask):
        #output shape [nhead*seq_len , nhead*seq_len ]
        #k reshape 
        # 对于每个 head,传入的 q,k,v,mask 是否都需要进行 split
        # 每个 q,k,v 都要进行切分
        qs = torch.chunk(q,self.headcnt,dim=-1)
        ks = torch.chunk(k,self.headcnt,dim=-1)
        vs = torch.chunk(v,self.headcnt,dim=-1)
        
        head_outputs = [self.mha[i](qs[i],ks[i],vs[i],mask)  for i in range(len(qs))]
        #return head_outputs[0]
        multihead_output = torch.cat(head_outputs, dim=-1)
        return self.join_linear(multihead_output)
    
class Encoder(nn.Module):
    def __init__(self,nhead,embed_dim):
        super(Encoder,self).__init__()
        #self-att    
        self.attention = MultiHeadAttention(nhead,embed_dim)
        self.dropout = nn.Dropout(0.1)
        self.att_norm = nn.LayerNorm(embed_dim)
    
        #ffn
        inner_dim = 1024
        self.ffn_linear_1 = nn.Linear(embed_dim,inner_dim)
        self.ffn_linear_2 = nn.Linear(inner_dim,embed_dim)
        self.ffn_dropout = nn.Dropout(0.1)
        self.ffn_relu = nn.ReLU()
        self.ffn_norm = nn.LayerNorm(embed_dim)
    #@profile
    def forward(self, tokens, mask=None):
        #self-att
        residual = tokens
        att = self.attention(tokens,tokens,tokens,mask)
        att = residual + self.dropout(att)
        att = self.att_norm(att)
        #return att
        # ffn 
        residual = att
        ffn = self.ffn_dropout(self.ffn_linear_2(self.ffn_relu(self.ffn_linear_1(att))))
        ffn = att + residual
        ffn = self.ffn_norm(ffn)
        return ffn

class MultiEncoder(nn.Module):
    def __init__(self,voc_size=8000,nenc=4,nhead=4,hidden_dim=256):
        super(MultiEncoder,self).__init__()
        self.encoders = nn.ModuleList([Encoder(nhead, hidden_dim) for i in range(nenc)])
    
    def forward(self, src, src_pad_mask):
        output = src
        for layer in self.encoders:
            output = layer(output,src_pad_mask)
        return output

class Decoder(nn.Module):
    def __init__(self,nhead, embed_dim):
        super(Decoder,self).__init__()
        # MHA 相当于做了一个具有冗余的查找表 , 
        # 让句子和查找表之间建立组合成一个更复杂的数据, 而不仅仅是句子本身. 
        # 还包含句子内部词汇之间的关系. 是一个更高纬度的数据.
        # 那能不能去掉 v * lookup 这个过程呢? 
        # 因为 lookup 里面是相关性百分比, 已经失去原始语义 
        self.self_att = MultiHeadAttention(nhead,embed_dim)
        self.self_att_dp = nn.Dropout(0.1)        #dropout 算子有的代码是同一个 有的代码是不同的
        self.self_att_norm = nn.LayerNorm(embed_dim)

        self.cross_att_dp = nn.Dropout(0.1)
        self.cross_att_norm = nn.LayerNorm(embed_dim)
        self.cross_att = MultiHeadAttention(nhead,embed_dim)
        #ffn 的两次线性变化,把维度提升又降了下来 
        inner_dim = 1024

        self.ffn_linear_1 = nn.Linear(embed_dim,inner_dim)
        self.ffn_linear_2 = nn.Linear(inner_dim,embed_dim)
        self.ffn_dropout = nn.Dropout(0.1)
        self.ffn_relu = nn.ReLU()
        self.ffn_norm = nn.LayerNorm(embed_dim)

    def forward(self, src, target, srcmask, tgtmask):
        #self-attention
        #当进行训练的时候,self-att需要传入 seq mask
        #当进行推理的时候,self-att不需要传入 seq mask
        #seq mask 的作用是将训练并行化了.即传入完整的一句话,训练每个位置的token预测
        #那在预测的时候，虽然有每一个token的预测,但只会取最后一个token的结果作为预测的结果. (Greedy Decoding)
        residual = target
        att = self.self_att_dp(self.self_att(target, target, target, mask=tgtmask))
        att = residual + self.self_att_dp(att)
        att = self.self_att_norm(att)
   
        # dropout 一般跟随在复杂的算子后面, 降低训练出来算子过拟合的概率 让算子中的每一个神经元都起到作用
        # encoder-decoder-att 
        residual = att
        att = self.cross_att(att,src,src,mask=srcmask)
        att = self.cross_att_dp(att)
        att = self.cross_att_norm(residual+att)

        #ffn
        ffn = self.ffn_linear_2(self.ffn_relu(self.ffn_linear_1(att)))
        ffn = att + ffn
        ffn = self.ffn_norm(ffn)

        return ffn

class MultiDecoder(nn.Module):
    def __init__(self,voc_size=8000,ndec=4,nhead=4,embed_dim=256):
        super(MultiDecoder,self).__init__()
        self.decoders = nn.ModuleList([Decoder(nhead, embed_dim) for i in range(ndec)])
    def forward(self, src, target,srcmask,tgtmask): 
        for layer in self.decoders:
            target = layer(src,target,srcmask,tgtmask)
        return target


class PositionEmbedding(nn.Module):
    def __init__(self, max_seq_len, hidden_dim):
        super(PositionEmbedding,self).__init__()
        self.pe = get_pe(max_seq_len,hidden_dim)
        self.requires_grad_(False)

    def forward(self,x):
        #PE长度是固定的，输入 x 的长度是不固定的,因此要注意限制长度. seq_len 维度是1
        seq_len = x.size(1)
        #指定第1个维度可能存在裁切
        return self.pe[:, :seq_len] + x[:,:seq_len]




class TTransformer(L.LightningModule):
    def __init__(self,voc_src_size=8000,voc_tgt_size=8000,nenc=4,ndec=4,nhead=4,hidden_dim=256):
        super(TTransformer,self).__init__()
        self.src_embedding = nn.Embedding(voc_src_size,hidden_dim)
        self.tgt_embedding = nn.Embedding(voc_tgt_size,hidden_dim)

        self.src_pe_embedding = PositionEmbedding(1024, hidden_dim)#.requires_grad_(False)
        self.tgt_pe_embedding = PositionEmbedding(1024, hidden_dim)#.requires_grad_(False)

        self.encoders =  MultiEncoder() 

        # decoder 第一层的输入是 shift tokens 
        # decoder 第二层的输入是 encoder 出来的 lookup_table*v，携带者自相关性信息 的 sentence.
        self.decoders =  MultiDecoder() # nn.ModuleList([Decoder(nhead,hidden_dim) for i in range(ndec)])
        self.voc_linear = nn.Linear(hidden_dim, voc_tgt_size)
      

    def forward(self, src, tgt):
        print("src:",src)
        src_mask = gen_src_mask(src)
        tgt_mask = gen_tgt_mask(tgt)

        src = self.src_pe_embedding(self.src_embedding(src))
        src = self.encoders(src, src_mask)

        tgt = self.tgt_pe_embedding(self.tgt_embedding(tgt))
        tgt =  self.decoders(src, tgt, src_mask, tgt_mask)

        #F.softmax 返回
        #这里返回的是所有位置的预测，所以最后的形状是 seq_len*voc_tgt_size
        #也就是训练的时候，会一次把所有的位置的预测训练完毕
        #所以需要 seq_mask 把相关性表里面用于生成新token的 att 操作进行mask处理
        #相当于把每个token的预测 mask 到这个 token 相关性表了.
        return F.softmax(self.voc_linear(tgt),dim=-1)

    def training_step(self, batch, batch_idx):

        #src 是中文 seq 
        #tgt 是英文 seq
        #forward 一次只得到一个 token 
        src, tgt = batch
        predict = self.forward(src, tgt)
        loss = nn.functional.mse_loss(predict, predict)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer



if __name__== "__main__":
    ttransformer = TTransformer()
    from torch.profiler import profile, record_function, ProfilerActivity 
    src = torch.rand(10*1024).reshape(10,1024).long()
    tgt = torch.rand(10*1024).reshape(10,1024).long()
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True,record_shapes=True) as prof:
        with record_function("model_inference"):    
            ttransformer(src,tgt)

    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
    #snapshot = torch.cuda.memory._snapshot()
    #from pickle import dump
    #dump(snapshot, open('snapshot.pickle', 'wb'))
    #pprint(snapshot['segments'])
    #out1 = torch.rand(3*2*2*4).reshape(3,2,2,4)
    #grid1 = make_grid(out1.view(-1,1,out1.shape[2],out1.shape[3]), nrow=8)
    #writer.add_image("grid1",grid1,global_step=1)

    #print(ttransformer)
   

    