
#train chinese tokenizer and english tokenizer
#merge tokenizer


    
with open("data/en.txt", "r", encoding="utf-8") as fp:
    data = fp.read().strip().split("\n")

sentences = []
for d in data:
    d = d.strip()
    if "===" in d or len(d) == 0 :
        continue
    sentences.append(d)

with open("data/corpus_en.txt", "w", encoding="utf-8") as fp:
    fp.write("\n".join(sentences))