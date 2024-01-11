
from tinytransformer import *
from dataset import *

dataset = LLMDataset()
train_loader = utils.data.DataLoader(dataset)
ttransformer = TTransformer()
trainer = L.Trainer(limit_train_batches=100, max_epochs=1)
trainer.fit(model=ttransformer, train_dataloaders=train_loader)