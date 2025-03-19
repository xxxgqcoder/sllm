import datetime

import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model.model import SLLModel
from dataset.data_loader import PretrainDataset

train_data_path = 'data/pretrain/pretrain_hq.jsonl'
tokenizer_path = 'tokenizer/minimind_tokenizer'

learning_rate = 5e-6
layer_num = 8
embed_dim = 512
atten_head_num = 8
max_seq_len = 512
epoch_num = 1
batch_size = 256

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
vocab_size = tokenizer.vocab_size
print(f"loaded tokenizer from {tokenizer_path}")

# dataset
train_dataset = PretrainDataset(
    data_path=train_data_path,
    tokenizer=tokenizer,
    max_length=max_seq_len,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    pin_memory=True,
    drop_last=False,
    shuffle=False,
)

device = torch.accelerator.current_accelerator(
) if torch.accelerator.is_available() else 'cpu'
print(f'using device: {device}')

model = SLLModel(
    vocab_size=vocab_size,
    layer_num=layer_num,
    embed_dim=embed_dim,
    atten_head_num=atten_head_num,
    max_seq_len=max_seq_len,
).to(device)

param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'total model param num: {param_num}')

optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss(reduction='none')

model.train()
for epoch in range(epoch_num):
    print(f'training epoch: {epoch}')
    start = datetime.datetime.now()
    for batch, (X, Y, loss_mask) in enumerate(train_loader):
        # compute prediction and loss
        X = X.to(device)
        Y = Y.to(device)
        loss_mask = loss_mask.to(device)
        pred = model(X)
        loss = loss_fn(pred.view(-1, pred.size(-1)), Y.view(-1)).view(Y.size())

        loss = (loss * loss_mask).sum() / loss_mask.sum()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss = loss.item()
        elapse_sec = (datetime.datetime.now() - start).seconds
        print(
            f"{datetime.datetime.now()}, epoch: {epoch}, step: {batch}, loss: {loss:.06f}, {elapse_sec / (batch+1):.2f} seconds / batch, total elapse {elapse_sec / 3600:.2f} hours"
        )
    
    print(f'{datetime.datetime.now()}, finish epoch {epoch}, total elapse {elapse_sec / 3600:.2f} hours, total {batch+1} batch')
    
