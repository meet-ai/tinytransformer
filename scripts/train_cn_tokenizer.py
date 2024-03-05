'''
Author: meetai meetai@gmx.com
Date: 2024-01-11 17:07:35
LastEditors: meetai meetai@gmx.com
LastEditTime: 2024-03-04 17:58:42
FilePath: /tinytransformer/scripts/train_cn_tokenizer.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import sentencepiece as spm
spm.SentencePieceTrainer.train(
    input='data/corpus_cn.txt',
    model_prefix='tokenizer',
    vocab_size=30000,
    #user_defined_symbols=['foo', 'bar'],
    character_coverage=1.0,
    model_type="bpe",
)