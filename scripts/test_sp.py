import torch

import sentencepiece as spm

# 加载预训练好的 SentencePiece 模型
tokenizer  = "tokenizer/tokenizer_cn_en.model"

# load
chinese_sp_model = spm.SentencePieceProcessor()
chinese_sp_model.Load(tokenizer)




print(chinese_sp_model.EncodeAsPieces('hello world 你好世界')) # 
print(chinese_sp_model.EncodeAsIds('hello world 你好世界'))

#['▁he', 'll', 'o', '▁world', '▁你', '好', '世界']
#[109, 414, 19949, 387, 29218, 67170, 20026]