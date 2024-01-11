
import unittest
from utils import *

class TestMask(unittest.TestCase):

    def test_upper(self):
        attention_map = torch.rand(3,5,5)  #shape:BatchSize, SeqLen, SeqLen
        src = torch.rand(3,5) #shape:BatchSize, SeqLen
        mask = gen_src_mask(src)
        print("mask",mask)
        attention_map.masked_fill(mask,-1e9)
        print("att",attention_map)

    def test_gen_tgt_mask(self):
        attention_map = torch.rand(2,4,4)  #shape:BatchSize, SeqLen, SeqLen
        src = torch.tensor([[1,2,3,10000],[3,4,5,10000]]) #shape:BatchSize, SeqLen
        mask = gen_tgt_mask(src)
        print("mask",mask)
        r = attention_map.masked_fill(mask,-1e9)
        print("att",r)
        
    def test_voc(self):
        from torchtext.vocab import build_vocab
        from torchtext.data.utils import get_tokenizer
        tokenizer = get_tokenizer('basic_english')
        voc = build_vocab(text_data, tokenizer=tokenizer)
        print(voc.get_itos())
        
if __name__ == '__main__':
    unittest.main()