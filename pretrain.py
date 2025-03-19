import os
import datetime

import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model.model import SLLModel
from dataset.data_loader import PretrainDataset
from model.utils import latest_checkpoint

train_data_dir = 'data/pretrain/pretrain_hq.jsonl'
tokenizer_dir = 'tokenizer/minimind_tokenizer'
checkpoint_dir = 'output/checkpoints'

learning_rate = 5e-6
layer_num = 8
embed_dim = 512
atten_head_num = 8
max_seq_len = 512
epoch_num = 2
batch_size = 256


def train_model(
    train_data_dir,
    tokenizer_dir,
    checkpoint_dir,
    learning_rate,
    layer_num,
    embed_dim,
    atten_head_num,
    max_seq_len,
    epoch_num,
    batch_size,
):
    # set env
    os.makedirs(checkpoint_dir, exist_ok=True)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    vocab_size = tokenizer.vocab_size
    print(f"loaded tokenizer from {tokenizer_dir}")

    # dataset
    train_dataset = PretrainDataset(
        data_path=train_data_dir,
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

    # model
    device = torch.accelerator.current_accelerator(
    ) if torch.accelerator.is_available() else 'cpu'
    print(f'using device: {device}')

    model = SLLModel(
        vocab_size=vocab_size,
        layer_num=layer_num,
        embed_dim=embed_dim,
        atten_head_num=atten_head_num,
        max_seq_len=max_seq_len,
    )
    model = model.to(device)

    param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'total model param num: {param_num}')

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss(reduction='none')

    # try to load latest checkpoint
    start_epoch = 0
    checkpoint_path = latest_checkpoint(checkpoint_dir)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path,
                                weights_only=False,
                                map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        start_epoch = int(checkpoint['epoch']) + 1
        loss = checkpoint['loss']
        print(
            f"loading checkpoint model from {checkpoint_path}, epoch: {checkpoint['epoch']}, loss: {loss}"
        )

    train_loop(
        train_loader=train_loader,
        start_epoch=start_epoch,
        epoch_num=epoch_num,
        checkpoint_dir=checkpoint_dir,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        model=model,
    )

    print(f'finish model training')


def train_loop(
    train_loader,
    start_epoch,
    epoch_num,
    checkpoint_dir,
    optimizer,
    loss_fn,
    device,
    model,
):
    # training
    model.train()
    for epoch in range(start_epoch, epoch_num):
        print(f'training epoch: {epoch}')
        start = datetime.datetime.now()
        for batch, (X, Y, loss_mask) in enumerate(train_loader):
            if batch > 2:
                break
            # compute prediction and loss
            X = X.to(device)
            Y = Y.to(device)
            loss_mask = loss_mask.to(device)
            pred = model(X)
            loss = loss_fn(pred.view(-1, pred.size(-1)),
                           Y.view(-1)).view(Y.size())

            loss = (loss * loss_mask).sum() / loss_mask.sum()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss = loss.item()
            elapse_sec = (datetime.datetime.now() - start).seconds
            print(
                f"{datetime.datetime.now()}, epoch: {epoch}, step: {batch}, loss: {loss:.06f}, {elapse_sec / (batch+1):.2f} seconds / batch, total elapse {elapse_sec / 3600:.2f} hours"
            )

        print(
            f'{datetime.datetime.now()}, finish epoch {epoch}, total elapse {elapse_sec / 3600:.2f} hours, total {batch+1} batch'
        )

        # save checkpoints
        checkpoint_path = os.path.join(checkpoint_dir, f"model_{epoch:04d}.pt")
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            },
            checkpoint_path,
        )

        print(f'model checkpoint saved to {checkpoint_path}')


if __name__ == "__main__":
    train_model(
        train_data_dir=train_data_dir,
        tokenizer_dir=tokenizer_dir,
        checkpoint_dir=checkpoint_dir,
        learning_rate=learning_rate,
        layer_num=layer_num,
        embed_dim=embed_dim,
        atten_head_num=atten_head_num,
        max_seq_len=max_seq_len,
        epoch_num=epoch_num,
        batch_size=batch_size,
    )
