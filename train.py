
from tinytransformer import *
from dataset import *
from torch import utils

dataset = LLMDataset('data/corpus_cn.txt','data/corpus_en.txt')
train_loader = utils.data.DataLoader(dataset)
ttransformer = TTransformer()
for param in ttransformer.parameters():
    if param.requires_grad==False:
        print(param) 
trainer = L.Trainer(limit_train_batches=100, max_epochs=1)
trainer.fit(model=ttransformer, train_dataloaders=train_loader)