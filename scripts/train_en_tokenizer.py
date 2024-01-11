import sentencepiece as spm
spm.SentencePieceTrainer.train(
    input='data/corpus_en.txt',
    model_prefix='tokenizer',
    vocab_size=20000,
    #user_defined_symbols=['foo', 'bar'],
    character_coverage=1.0,
    model_type="bpe",
)