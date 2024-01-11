import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
#from transformers import LlamaTokenizer
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm

tokenizer_cn  = "tokenizer/tokenizer_cn.model"
tokenizer_en ="tokenizer/tokenizer_en.model"

# load
chinese_sp_model = spm.SentencePieceProcessor()
chinese_sp_model.Load(tokenizer_cn)

en_sp_model = spm.SentencePieceProcessor()
en_sp_model.Load(tokenizer_en)

en_spm = sp_pb2_model.ModelProto()
en_spm.ParseFromString(en_sp_model.serialized_model_proto())

chinese_spm = sp_pb2_model.ModelProto()
chinese_spm.ParseFromString(chinese_sp_model.serialized_model_proto())

# print number of tokens
#print(len(llama_tokenizer), len(chinese_sp_model))
#print(llama_tokenizer.all_special_tokens)
#print(llama_tokenizer.all_special_ids)
#print(llama_tokenizer.special_tokens_map)

## Add Chinese tokens to LLaMA tokenizer
en_spm_tokens_set = set(p.piece for p in en_spm.pieces)
print(len(chinese_spm.pieces))
print(f"Before:{len(en_spm_tokens_set)}")
for p in chinese_spm.pieces:
    piece = p.piece
    if piece not in en_spm_tokens_set:
        new_p = sp_pb2_model.ModelProto().SentencePiece()
        new_p.piece = piece
        new_p.score = 0
        en_spm.pieces.append(new_p)
print(f"New model pieces: {len(en_spm.pieces)}")

## Save
output_sp_dir = 'tokenizer'
output_hf_dir = 'tokenizer'  # the path to save Chinese-LLaMA tokenizer
#os.makedirs(output_sp_dir, exist_ok=True)

with open(output_sp_dir + '/tokenizer_cn_en.model', 'wb') as f:
    f.write(en_spm.SerializeToString())
    
##tokenizer = LlamaTokenizer(vocab_file=output_sp_dir + '/tokenizer_cn_en.model')

#tokenizer.save_pretrained(output_hf_dir)
#kprint(f"Chinese-LLaMA tokenizer has been saved to {output_hf_dir}")

# Test
#llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)
#chinese_llama_tokenizer = LlamaTokenizer.from_pretrained(output_hf_dir)
#print(tokenizer.all_special_tokens)
#print(tokenizer.all_special_ids)
#print(tokenizer.special_tokens_map)
#text = '''白日依山尽，黄河入海流。欲穷千里目，更上一层楼。
#The primary use of LLaMA is research on large language models, including'''
#print("Test text:\n", text)
#print(f"Tokenized by LLaMA tokenizer:{llama_tokenizer.tokenize(text)}")
#print(f"Tokenized by Chinese-LLaMA tokenizer:{chinese_llama_tokenizer.tokenize(text)}")