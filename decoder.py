import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random
import pandas as pd
import datetime
import time
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_upper_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def make_std_mask(tgt, pad):
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & Variable(
        create_upper_mask(tgt.size(-1)).type_as(tgt_mask.data))
    return tgt_mask


def positional_encoding(seq_len, d_model):
    max_len = seq_len + 1
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    return pe


def generate_batch(bs, seq_len, d_model):
    r = torch.randint(1, d_model - 2, (bs, seq_len))
    st = torch.zeros(bs, 1)
    et = torch.ones(bs, 1) * (d_model - 1)
    mt = torch.ones(bs, 1) * (d_model - 2)

    dec_src = torch.cat((st, r, mt, r.flip(1)), 1)
    target = torch.cat((r, mt, r.flip(1), et), 1).long()

    return dec_src, target


# feed forward

class FeedForward(nn.Module):
    def __init__(self, d_model, d_middle):
        super().__init__()
        self.l1 = nn.Linear(d_model, d_middle)
        self.l2 = nn.Linear(d_middle, d_model)

    def forward(self, x):
        x = self.l2(F.relu(self.l1(x)))

        return x


class AttentionModule(nn.Module):
    def __init__(self, d_model, d_Q, d_K, d_V):
        super().__init__()
        self.Q = nn.Linear(d_model, d_Q, bias=False)
        self.K = nn.Linear(d_model, d_K, bias=False)
        self.V = nn.Linear(d_model, d_V, bias=False)

    def forward(self, q, k, v, mask=None):
        y = self.attention(self.Q(q), self.K(k), self.V(v), mask)
        return y

    def attention(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)

        attn = attn @ V
        return attn


class MultiHeadAttentionModule(nn.Module):
    def __init__(self, d_model, h, d_Q, d_K, d_V):
        super().__init__()
        self.linear = nn.Linear(h * d_V, d_model, bias=False)
        self.a_modules = nn.ModuleList(AttentionModule(d_model, d_Q, d_K, d_V) for _ in range(h))

    def forward(self, q, k, v, mask=None):
        combines = []

        for layer in self.a_modules:
            y = layer(q, k, v, mask)

            combines.append(y)

        y = torch.cat(combines, -1)

        y = self.linear(y)

        return y


class DecoderBlock(nn.Module):
    def __init__(self, d_model, d_middle, dropout, h, d_Q, d_K, d_V):
        super().__init__()
        # multihead_masked
        self.multi_head_masked = MultiHeadAttentionModule(d_model, h, d_Q, d_K, d_V)
        # norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # feed forward
        self.feed_forward = FeedForward(d_model, d_middle)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        x: decoder input
        """
        x = x + self.dropout(self.multi_head_masked(x, x, x, mask))
        x = self.norm1(x)
        x = x + self.dropout(self.feed_forward(x))
        x = self.norm2(x)
        return x


def to_cuda(model):
    """Sends model from CPU to CUDA."""
    model.cuda()
    if isinstance(model, nn.Module):
        for child in model.children():
            to_cuda(child)


class Decoder(nn.Module):
    def __init__(self, num_blocks, d_model, d_middle, d_token, dropout, h, d_Q, d_K, d_V):
        super().__init__()

        self.layers = nn.ModuleList(
            DecoderBlock(d_model, d_middle, dropout, h, d_Q, d_K, d_V) for _ in range(num_blocks))

        self.l1 = nn.Linear(d_model, d_token)

    def forward(self, dec_inp, pe, dec_mask=None):
        x = dec_inp + pe
        for layer in self.layers:
            x = layer(x, dec_mask)
        x = self.l1(x)

        return x


def save_model(model):
    path = '.\models\model' + time.strftime("%Y%m%d-%H%M%S")
    torch.save(model.state_dict(), path)


def demo_model(model, opt, loss, bs, num_batches, d_model, pos_enc, mask, max_seq_len):
    model.eval()

    sq_len = 10
    dec_src, target = generate_batch(bs, sq_len, d_model)
    dec_src = F.one_hot(dec_src.to(torch.int64), num_classes=d_model).float()

    dec_src = dec_src.to(device)
    target = target.to(device)

    pe = pos_enc[0, :2 * sq_len + 2, :d_model]
    msk = mask[0, :2 * sq_len + 2, : 2 * sq_len + 2]

    pred = model(dec_src, pe, msk)
    print(pred)
    pred = pred.permute(0, 2, 1)
    pred = torch.argmax(F.softmax(pred, dim=1), dim=1)
    print(pred.shape)
    print(target.shape)
    print(pred)
    print(target)


def train(decoder, opt, loss, bs, num_batches, d_model, pos_enc, mask, max_seq_len):
    # Training loop
    total_loss = 0
    output_every = 500
    for batch_num in range(num_batches):

        sq_len = random.randint(1, max_seq_len)
        dec_src, target = generate_batch(bs, sq_len, d_model)
        dec_src = F.one_hot(dec_src.to(torch.int64), num_classes=d_model).float()

        dec_src = dec_src.to(device)
        target = target.to(device)
        dec_s = dec_src.clone()

        pe = pos_enc[0, :2 * sq_len + 2, : d_model].clone()

        msk = mask[0, :2 * sq_len + 2, : 2 * sq_len + 2].clone()

        pred = decoder(dec_s, pe, msk)

        pred = pred.permute(0, 2, 1)
        # Compute the loss
        l = loss(pred, target)
        total_loss += l.item()
        # Backward pass
        l.backward()
        # Update the parameters
        opt.step()
        opt.zero_grad()
        if batch_num % output_every == 0 and batch_num != 0:
            print(total_loss / output_every)
            print(l)
            total_loss = 0


def main():
    # Model paramaters
    num_blocks = 6
    d_model = 12
    d_middle = 4 * d_model
    d_token = d_model
    dropout = 0.0
    h = 6
    d_Q = d_model
    d_K = d_model
    d_V = d_model
    MAX_SEQ_LEN = 50
    pos_enc = positional_encoding(MAX_SEQ_LEN, d_model)

    pos_enc = pos_enc.to(device)
    mask = create_upper_mask(MAX_SEQ_LEN + 2)
    mask = mask.to(device)

    decoder = Decoder(num_blocks, d_model, d_middle, d_token, dropout, h, d_Q, d_K, d_V)

    LOAD = True
    if LOAD:
        PATH = '.\models\model20230115-104010'
        decoder.load_state_dict(torch.load(PATH))

    to_cuda(decoder)

    loss = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(decoder.parameters(), lr=2.5e-4)
    bs = 32
    num_batches = 40000
    max_seq_len = 20
    train(decoder, opt, loss, bs, num_batches, d_model, pos_enc, mask, max_seq_len)
    save_model(decoder)
    bs = 1
    demo_model(decoder, opt, loss, bs, num_batches, d_model, pos_enc, mask, max_seq_len)


main()
