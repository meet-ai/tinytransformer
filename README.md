<!--
 * @Author: meetai meetai@gmx.com
 * @Date: 2023-12-04 14:14:19
 * @LastEditors: meetai meetai@gmx.com
 * @LastEditTime: 2024-03-05 16:04:29
 * @FilePath: /tinytransformer/README.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
# tinytransformer


## transformer 实现的几个关键技术
1. transformer 网络结构
2. encoder/decoder
3. attention 与 Q K V 
4. Q K V 与矩阵运算
5. self-attention 和 cross-attention
6. position encoding
7. padding mask 
8. 训练过程中的数据流


### transformer 网络结构
1. 从语句到 token:tokenizer
2. 从 token 到输入的表示:word-embedding
3. 添加位置信息:pos-embedding
4. 提取语句特征:encoder
5. 特征翻译:decoder
6. 输出:


### 对文本进行处理
1. 对数据进行预处理
2. 分词,把文本分割成小单元. 
3. 从文本创建词汇表,确保能包含所有需要处理的文本.
4. 把输入的句子转换成词汇表索引向量, 输入到神经网络

这些处理的过程叫 tokenizer, tokenizer 把输入的句子原始文本转换到更适合算法理解的 token index 向量. 这个过程中,生成词表是文本处理的重要目标.


### 使用 SentencePiece 训练和合并词表
SentencePiece是一个由谷歌开源的文本分词器工具. 有下面几个特点
1. SentencePiece 将句子视为 Unicode 编码序列，使得子词算法不依赖于特定语言，从而支持多语言处理. 因此他是支持中文的.
2. 可以直接使用预训练模型 也可以自己训练
3. 集成了 BPE、WordPiece、Unigram 等子词分词算法，以及字符和词级别的分词.

SentencePiece 训练词汇表过程
```
import sentencepiece as spm
spm.SentencePieceTrainer.train(
    input='data/corpus_cn.txt',
    model_prefix='tokenizer',
    vocab_size=50000,
    #user_defined_symbols=['foo', 'bar'],
    character_coverage=1.0,
    model_type="bpe",
)
```
其中 corpus_cn.txt 文本是按行存储的中文文档.具体可见 data/corpus_cn.txt 文本内容. 

model_type 使用的 BPE, BytePairEncoding  
#### BytePairEncoding 分词算法
考虑一个简单的ASCII文本文件，其中包含以下字符串：
```
"hello world"
```
在这个字符串中，我们可以找到以下连续的字节对：
```
"h" (字节值为68) 后面紧跟着 "e" (字节值为69)  
"e" 后面紧跟着 "l" (字节值为76)  
"l" 后面紧跟着 "l" (字节值相同)  
"l" 后面紧跟着 "o" (字节值为79)  
"o" 后面紧跟着空格字符 " " (字节值为32)  
" " 后面紧跟着 "w" (字节值为87)  
"w" 后面紧跟着 "o" (字节值相同)  
"o" 后面紧跟着 "r" (字节值为114)  
"r" 后面紧跟着 "l" (字节值相同)  
"l" 后面紧跟着 "d" (字节值为100)  
```
在 BPE 算法中，我们会统计这些连续字节对的出现频率。例如，如果我们在大量文本数据中发现 "el" 和 "ll" 这对字节对频繁出现，我们可能会将它们合并为一个新的单元（例如，一个新的字符或子词），从而减少整个文本数据集的大小. 简言之两两合并. 通过 BPE 实现了统计压缩的能力.

#### SentencePiece 高级用法: 合并词汇表
SentencePiece 的模型存储以 ProtoBuf 序列化字符串为基础. 而模型数据本身包含了每一个 Piece (即Token). 所以要合并词表,实际上是对模型文件的读取和合并字段,然后重写模型.这里给出伪代码
```
sp_model_cn = Load('a.model')
sp_mode_en = Load('b.model')

merged = merge_model_piece(sp_model_cn,sp_model_en)
write_model(merged)
```


### word-embedding 
因为词汇表一般会比较大，需要进行一下维度压缩.压缩到一个512，1024之类的维度长度上. 一般使用 nn.Embedding 来生成一个 Embedding Layer 对输入的句子进行处理.
```
    self.src_embed = nn.Embedding(voc_size, hidden_dim)
```
对比  Embedding 和 one-hot 
- Embedding 向量经过训练能够捕捉单词的语义信息。例如，具有相似含义的单词在向量空间中会彼此接近。这意味着，即使单词在字面上不完全相同，它们在语义上相似的程度也能通过 Embedding 向量的距离反映出来。而 one-hot 编码不考虑语义信息，它仅仅表示单词的存在与否，无法捕捉单词之间的语义关系。
- one-hot 编码的向量通常是稀疏的，大部分元素为零，这可能导致计算过程中的不稳定性。而 Embedding 向量是密集的，每个维度都携带信息，这有助于提高距离计算的稳定性和准确性。

使用 Embedding 
```
#一句话经过  Tokenizer 之后转换成 LongTensor 
sentence = torch.LongTensor([1,2,3,4,5])   #1,2,3表示词汇表中第一个词，第二个词，第三个词 
                                           #这里需要注意的是， Embedding 输入的参数是索引,所以是 LongTensor ,而不是 FloatTensor
embeded_sentence = self.src_embed(sentence)
```

### 神经网络处理输入
考虑当前数据已经处理完毕，神经网络该如何处理输入的语句? 
1. 神经网络应当充分对输入的句子进行理解和特征处理
2. 神经网络对于输入的句子 




### 给输入语句添加 position 信息

### 
