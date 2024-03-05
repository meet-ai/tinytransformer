'''
Author: meetai meetai@gmx.com
Date: 2024-01-11 17:31:17
LastEditors: meetai meetai@gmx.com
LastEditTime: 2024-03-04 16:03:48
FilePath: /tinytransformer/scripts/spm_gen_cn_txt.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

#train chinese tokenizer and english tokenizer
#merge tokenizer


# 读取所有中文，合并成一个文本，再进行分词
with open("data/cn.txt", "r", encoding="utf-8") as fp:
    data = fp.read().replace(" ", "").strip().split("\n")

sentences = []
for d in data:
    d = d.strip()
    if "===" in d or len(d) == 0 :
        continue
    sentences.append(d)

#写入到 corpus_cn 文本中
with open("data/corpus_cn.txt", "w", encoding="utf-8") as fp:
    fp.write("\n".join(sentences))
    
#with open("data/cn.txt", "r", encoding="utf-8") as fp:
#    data = fp.read().strip().split("\n")
#
#sentences = []
#for d in data:
#    d = d.strip()
#    if "===" in d or len(d) == 0 :
#        continue
#    sentences.append(d)
#
#with open("data/corpus_cn.txt", "w", encoding="utf-8") as fp:
#    fp.write("\n".join(sentences))
