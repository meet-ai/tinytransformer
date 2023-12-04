

import torch 
from torch import nn
# 
class AttentionHead(nn.Module):
    def __init__(self,hidden_dim, output_dim):
        super(AttentionHead, self).__init__()
        self.Q = nn.Linear(hidden_dim, output_dim)
        self.K = nn.Linear(hidden_dim, output_dim)
        self.V = nn.Linear(hidden_dim, output_dim) 

    def forward(self, q,k,v):
        query = self.query_projection(q)
        key = self.key_projection(k)
        value = self.value_projection(v)

        #注意力的结算过程
        # V* (Q*K/sqrt(length_of_input_dim)
        # matmul vs Linear 一个是可学习的算子 一个是单纯的计算
        # Q.shape = [SetenceSize, dim_of_embedding]
        scores = torch.matmul(query, key.transpose(-2, -1)) / (key.size(-1) ** 0.5)
        #softmax 是多分类的激活函数, 把多个分类分布的概率
        #softmax 使用最后一个维度作为分类的维度
        # 这里不是分类问题为何也用 softmax 
        # 因为点积得到的结果 shape 是 SentenceASize, SetenceBSize 
        # F.softmax 得到的结果是A*B softmax 在 B 在每个单词上相对于A的相关性
        # value 和 query 是否 
        attention_weights = F.softmax(scores, dim=-1)
        
        #q,k,v sequence_len 是同一个值,保证 atten_weights 能和 value 计算
        context = torch.matmul(attention_weights, value)
        
        return context 



class MultiHeadAttention(nn.Module):
    def __init__(self,headcnt,hidden_dim,output_dim):
        #super不需要代入参数
        super(MultiHeadAttention,self).__init__()
        print("MultiHeadAttention")
        self.mha = nn.ModuleList([AttentionHead(hidden_dim,output_dim) for i in range(headcnt)])
        self.join_linear = nn.Linear(headcnt * hidden_dim, output_dim)
    
    def forward(self, q,k,v):
        head_outputs = [head(q,k,v)  for head in self.mha]
        #
        multihead_output = torch.cat(head_outputs, dim=-1)
        self.join_linear(head_outputs)
        

class Decoder(nn.Module):
    def __init__(self, embed_dim):
        super(Decoder,self).__init__()
        self.attention = MultiHeadAttention(3,512,1024)
        self.ffd = nn.Sequential( nn.Linear(embed_dim, 4 * embed_dim),nn.ReLU(),nn.Linear(4 * embed_dim, embed_dim))

class Encoder(nn.Module):
    def __init__(self,embed_dim):
        super(Encoder,self).__init__()
        self.attention = MultiHeadAttention(3,512,1024)
        self.ffd = nn.Sequential( nn.Linear(embed_dim, 4 * embed_dim),nn.ReLU(),nn.Linear(4 * embed_dim, embed_dim))

        


class TTransformer(nn.Module):
    def __init__(self,nenc,ndec):
        super(TTransformer,self).__init__()
        self.encoder = nn.ModuleList([Encoder(1024) for i in range(nenc)])
        self.decoder = nn.ModuleList([Decoder(1024) for i in range(ndec)])
        


if __name__== "__main__":
    ttransformer = TTransformer(6,6)
    print(ttransformer)
    print("main")

    