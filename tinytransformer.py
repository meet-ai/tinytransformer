#code:utf-8
import torch 
from torch import nn
import math
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
from utils import *
writer = SummaryWriter('./result_tensorboard')

# 这里有个 dropout
class AttentionHead(nn.Module):
    #head 是子空间的概念,意思是让输入的形式更多样化. 灵感来自于人类可以多个维度的去输入数据来进行学习.
    # encoder 的 mask 用于删除 padding ， 那 encoder mask 就跟token长度有关系了.
    # decoder 的 mask 用于删除 padding 
    # decoder 的 mask 在使用之前是否需要对句子
    def __init__(self,hidden_dim, mask=None, dropout=0.1):
        super(AttentionHead, self).__init__()
        # 这里使用了相同的 Linear hidden 和 output
        self.Q = nn.Linear(hidden_dim, hidden_dim)
        self.K = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, hidden_dim) 
        if dropout is not None:
            self.dropout = dropout
            self.dp = nn.Dropout(dropout)
        self.mask = mask

    #兼容 self-attention 和 cross-attention 
    def forward(self, q, k, v, mask=None):

        #output shape  [seq_len, hidden]
        query = self.query_projection(q)
        key = self.key_projection(k)
        value = self.value_projection(v)

        #注意力的结算过程
        # V* (Q*K/sqrt(length_of_input_dim)
        # matmul vs Linear 一个是可学习的算子 一个是单纯的计算
        # Q.shape = [SetenceSize, dim_of_embedding]
        scores = torch.matmul(query, key.transpose(-2, -1)) / (key.size(-1) ** 0.5)

        # mask 有两种,一种是 encoder 的 mask , 用来把 pad=0转换
        # 第二种是 decoder 的 mask 用来把未来 token 隐藏掉.
        if mask!=None:
            seq_len = value.shape[1]
            #torch.tril 是指定某些位置, 填充上三角部位为 True 
            mask = torch.tril(torch.ones(sql_len, seq_len)) == 0
            # 对 mask 中 True 的位置填充 -1e9
            scores[:,mask]=-1e9

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
    def __init__(self,headcnt,hidden_dim):
        #super不需要代入参数
        super(MultiHeadAttention,self).__init__()
        print("MultiHeadAttention")
        #multi-head-attention 有一个优化方案,不用 for loop 把 concat 的特征放在一个矩阵里面只执行一次 attention 操作就可以
        #这里先以逻辑优先把思路串通
        self.mha = nn.ModuleList([AttentionHead(hidden_dim) for i in range(headcnt)])
        self.join_linear = nn.Linear(hidden_dim*headcnt, hidden_dim)
    
    def forward(self, q,k,v):
        #output shape [nhead*seq_len , nhead*seq_len ]
        #k reshape 
        head_outputs = [head(q,k,v)  for head in self.mha]
        
        multihead_output = torch.cat(head_outputs, dim=-1)
        return self.join_linear(head_outputs)
        

class DecoderLayer(nn.Module):
    def __init__(self,nhead, embed_dim):
        super(DecoderLayer,self).__init__()
        # MHA 相当于做了一个具有冗余的查找表 , 
        # 让句子和查找表之间建立组合成一个更复杂的数据, 而不仅仅是句子本身. 
        # 还包含句子内部词汇之间的关系. 是一个更高纬度的数据.
        # 那能不能去掉 v * lookup 这个过程呢? 
        # 不能 因为 lookup 里面是相关性百分比, 已经失去原始语义 
        self.attention = MultiHeadAttention(nhead,embed_dim)
        self.dropout = nn.Dropout(0.1)
        self.att_norm = nn.LayerNorm(embed_dim)
        #self.pe = PositionEmbedding(embed_dim)
    
        self.attention2 = MultiHeadAttention(nhead,embed_dim)
        #ffn 的两次线性变化,把维度提升又降了下来 
        inner_dim = 2048
        self.ffn_linear_1 = nn.Linear(embed_dim,inner_dim)
        self.ffn_linear_2 = nn.Linear(inner_dim,embed_dim)
        self.ffn_dropout = nn.Dropout(0.1)
        self.ffn_relu = nn.ReLU()
        self.ffn_norm = nn.LayerNorm(embed_dim)

    def forward(self, src, target, srcmask, tgtmask):
        #self-attention
        att = self.attention(src,src,src,mask=srcmask)
        att = tokens + self.dropout_att(att)
        att = self.att_norm(att)

        # 为什么在 + 之前做 dropout 操作    
        # dropout 一般跟随在复杂的算子后面, 降低训练出来算子过拟合的概率 让算子中的每一个神经元都起到作用
        ffn = self.dropout(self.fn_linear_2(self.ffn_relu(self.ffn_linear_1(att))))
        ffn = att + ffn
        ffn = self.ffn_norm(ffn)

        #encoder-decoder attention 
        return ffn

class Decoder(nn.Module):
    def __init__(self,voc_size=8000,ndec=4,nhead=6,embed_dim=256):
        super(Decoder,self).__init__()
        self.decoders = nn.ModuleList([DecoderLayer(nhead, embed_dim) for i in range(ndec)])
    def forward(self, src, target,srcmask,tgtmask):
        
        target = token
        for layer in self.decoders:
            target = layer(src,target,srcmask,tgtmask)
        return target

class EncoderLayer(nn.Module):
    def __init__(self,nhead,embed_dim):
        super(EncoderLayer,self).__init__()
        # attention 的 Linear Shape
        # seq_len 
        self.attention = MultiHeadAttention(nhead,embed_dim)
        self.dropout = nn.Dropout(0.1)
        self.att_norm = nn.LayerNorm(embed_dim)
    
        #ffn
        #1. ffn 的 Linear shape 
        # inner_dim
        inner_dim = 2048
        self.ffn_linear_1 = nn.Linear(embed_dim,inner_dim)
        self.ffn_linear_2 = nn.Linear(embed_dim,inner_dim)
        self.ffn_dropout = nn.Dropout(0.1)
        self.ffn_relu = nn.ReLU()
        self.ffn_norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens):
        #+ pe
        # tokens  = tokens + get_pe()
        att = self.attention(tokens,tokens,tokens)
        att = tokens + self.dropout_att(att_src)
        att = self.att_norm(att)

# ffn 有些 dropout 实现在两个 linear 之间
        ffn = self.dropout(self.fn_linear_2(self.ffn_relu(self.ffn_linear_1(att))))
        ffn = att +ffn
        ffn = self.ffn_norm(ffn)
        return ffn

class Encoder(nn.Module):
    def __init__(self,voc_size=8000,nenc=4,nhead=6,hidden_dim=256):
        super(Encoder,self).__init__()
        self.encoders = nn.ModuleList([EncoderLayer(nhead, hidden_dim) for i in range(nenc)])
    
    def forward(self, src, src_pad_mask):
        output = None
        for layer in self.encoders:
            output = layer(src)
        return output

class PositionEmbedding(nn.Module):
    def __init__(self, max_seq_len, hidden_dim):
        self.pe = get_pe(max_seq_len,hidden_dim)
        self.pe.requires_grad_(False)

    def forward(self,x):
        #PE长度是固定的，输入 x 的长度是不固定的,因此要注意限制长度. seq_len 维度是1
        seq_len = x.size(1)
        #指定第1个维度可能存在裁切
        return self.pe[:, :seq_len] + x[:,:sql_len]




class TTransformer(nn.Module):
    def __init__(self,voc_size=8000,nenc=4,ndec=4,nhead=6,hidden_dim=256):
        super(TTransformer,self).__init__()
        self.input_embedding = nn.Embedding(voc_size,hidden_dim)
        self.pe_embedding = PositionEmbedding(1024, hidden_dim)
        self.encoder =  Encoder() 

        # decoder 第一层的输入是 shift tokens 
        # decoder 第二层的输入是 encoder 出来的 lookup_table*v，携带者自相关性信息的 sentence.
        self.decoder =  Decoder() # nn.ModuleList([Decoder(nhead,hidden_dim) for i in range(ndec)])
        self.voc_linear = nn.Linear(hidden_dim, voc_size)

    def forward(self, src, tgt):
        for enc in self.encoder:
            enc_result = encoder(src)
        encode_result = self.decoder(enc_result,)
        decode_result =  self.decoder(self.encoder(self.pe_embedding(self.input_embedding(x))))
        return F.softmax(self.res_linear(decode_result),dim=-1)

class PEGenerator():
    def __init__(self, embed_size, constant=10000):
        self.constant = constant
        self.embed_size = embed_size
    
    def get(self, seq_index, embed_index):
        #sin/cos 取决于 embed_index
        if embed_index%2==0:
            return math.sin(seq_index/math.pow(self.constant,embed_index/self.embed_size))
        else:
            return math.cos(seq_index/math.pow(self.constant,embed_index/self.embed_size))

def get_pe(seq_len,embed_size):
    peg = PEGenerator(embed_size)
    pe = []
    for i in range(seq_len):
        embed_slice = []
        for j in range(embed_size):
            embed_slice.append(peg.get(i,j))
        pe.append(embed_slice)
    import numpy
    pe = torch.from_numpy(numpy.array(pe)).type(torch.float32)
    return pe


if __name__== "__main__":
    ttransformer = TTransformer()
    out1 = torch.rand(3*2*2*4).reshape(3,2,2,4)
    #grid1 = make_grid(out1.view(-1,1,out1.shape[2],out1.shape[3]), nrow=8)
    #writer.add_image("grid1",grid1,global_step=1)

    #print(ttransformer)
   

    