import torch

def gen_src_mask(src, pad_idx=0):
    # pad_idx  index of vocabularay, usually 0
    print(f"src:{src},pad_idx:{pad_idx} ")
    mask = src==pad_idx

    #unsequeeze 提升两个维度
    # masked_fill 将 True 进行替换
    return mask.unsqueeze(1)

def gen_tgt_mask(target_seq, pad_idx=10000):
    # target mask 由 pad_mask 和 sequence mask 一起按位与
    pad_mask = (target_seq==pad_idx).unsqueeze(1)

    #sequence mask 是一个下三角
    seq_len = target_seq.shape[1]
    target = torch.ones(seq_len, seq_len)
    seq_mask = (torch.tril(target).type(torch.ByteTensor).to('cpu')==0)
    print("seq mask", seq_mask)
    return pad_mask | seq_mask
     