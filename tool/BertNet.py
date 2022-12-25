import torch
from torch import nn
from d2l import torch as d2l
import pandas as pd
import nltk
import random
import time
from torch.utils.data import DataLoader
import jieba

def _read_csv(path):
    df = pd.read_csv(path)
    l = df['reviewContent'][:1000]
    paragraphs = [nltk.tokenize.sent_tokenize(i.strip().lower()) for i in l]
    random.shuffle(paragraphs)
    return paragraphs

def _get_batch_loss_bert(net, loss, vocab_size, tokens_X,
                         segments_X, valid_lens_x,
                         pred_positions_X, mlm_weights_X,
                         mlm_Y, nsp_y):
    # 前向传播
    _, mlm_Y_hat, nsp_Y_hat = net(tokens_X, segments_X,
                                  valid_lens_x.reshape(-1),
                                  pred_positions_X)
    # 计算遮蔽语言模型损失
    mlm_l = loss(mlm_Y_hat.reshape(-1, vocab_size), mlm_Y.reshape(-1)) *\
    mlm_weights_X.reshape(-1, 1)
    mlm_l = mlm_l.sum() / (mlm_weights_X.sum() + 1e-8)
    # 计算下一句子预测任务的损失
    nsp_l = loss(nsp_Y_hat, nsp_y)
    l = mlm_l + nsp_l
    return mlm_l, nsp_l, l

def train_bert(train_iter, net, loss, vocab_size, devices, num_steps):
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    trainer = torch.optim.Adam(net.parameters(), lr=0.01)
    step, timer = 0, d2l.Timer()
    animator = d2l.Animator(xlabel='step', ylabel='loss',
                            xlim=[1, num_steps], legend=['mlm', 'nsp'])
    # 遮蔽语言模型损失的和，下一句预测任务损失的和，句子对的数量，计数
    metric = d2l.Accumulator(4)
    print("start")
    while step < num_steps:
        for tokens_X, segments_X, valid_lens_x, pred_positions_X,mlm_weights_X, mlm_Y, nsp_y in train_iter:
            print(f'循环开始{step}')
            tokens_X = tokens_X.to(devices[0])
            segments_X = segments_X.to(devices[0])
            valid_lens_x = valid_lens_x.to(devices[0])
            pred_positions_X = pred_positions_X.to(devices[0])
            mlm_weights_X = mlm_weights_X.to(devices[0])
            mlm_Y, nsp_y = mlm_Y.to(devices[0]), nsp_y.to(devices[0])
            trainer.zero_grad()
            timer.start()
            mlm_l, nsp_l, l = _get_batch_loss_bert(net, loss, vocab_size, tokens_X, segments_X, valid_lens_x,pred_positions_X, mlm_weights_X, mlm_Y, nsp_y)
            l.backward()
            trainer.step()
            metric.add(mlm_l, nsp_l, tokens_X.shape[0], 1)
            timer.stop()
            animator.add(step + 1,(metric[0] / metric[3], metric[1] / metric[3]))
            print(f'循环结束{step}',f'MLM loss {metric[0] / metric[3]:.3f}, 'f'NSP loss {metric[1] / metric[3]:.3f}')
        step = step + 1


    print(f'MLM loss {metric[0] / metric[3]:.3f}, 'f'NSP loss {metric[1] / metric[3]:.3f}')
    print(f'{metric[2] / timer.sum():.1f} sentence pairs/sec on 'f'{str(devices)}')

def load_data(batch_size, max_len,filepath):
    """Load the WikiText-2 dataset.

    Defined in :numref:`subsec_prepare_mlm_data`"""
    num_workers = d2l.get_dataloader_workers()
    paragraphs = _read_csv(filepath)
    train_set = d2l._WikiTextDataset(paragraphs, max_len)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                        shuffle=True, num_workers=num_workers,drop_last=True)
    return train_iter, train_set.vocab

def train():
    path = r'data/review_final.csv'
    batch_size, max_len = 64, 64
    train_iter, vocab = load_data(batch_size, max_len, path)

    net = d2l.BERTModel(len(vocab), num_hiddens=128, norm_shape=[128],
                        ffn_num_input=128, ffn_num_hiddens=256, num_heads=2,
                        num_layers=2, dropout=0.2, key_size=128, query_size=128,
                        value_size=128, hid_in_features=128, mlm_in_features=128,
                        nsp_in_features=128)
    devices = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss()
    train_bert(train_iter, net, loss, len(vocab), devices, 1)
    torch.save(net, 'model/bert.pth')

def get_bert_encoding(net, tokens_a, tokens_b=None):
    devices = d2l.try_all_gpus()
    path = r'data/review_final.csv'
    batch_size, max_len = 64, 64
    train_iter, vocab = load_data(batch_size, max_len, path)
    tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
    token_ids = torch.tensor(vocab[tokens], device=devices[0]).unsqueeze(0)
    segments = torch.tensor(segments, device=devices[0]).unsqueeze(0)
    valid_len = torch.tensor(len(tokens), device=devices[0]).unsqueeze(0)
    encoded_X, _, _ = net(token_ids, segments, valid_len)
    return encoded_X

def get_word(path):
    '''

    :param path: 文件路径
    :return:列表 列表中是每个词
    '''
    df = pd.read_csv(path)
    l = df['reviewContent']
    paragraphs = [i.strip().lower() for i in l]
    word = []
    for i in paragraphs:
        word += jieba.lcut(i)
    return list(set(word))

def word2embedding(word,net):
    embedding = []
    for i in word:
        encoded_text = get_bert_encoding(net, [i])
        encoded_text_word = encoded_text[:, 1, :]
        encoded_text_word = encoded_text_word.tolist()
        embedding.append(" ".join(str(i) for i in encoded_text_word[0]))
    return embedding

def to_csv(embedding,word):
    df = pd.DataFrame({'word':word,'embedding':embedding})
    df.to_csv(r'data/bert-word2embedding.csv',index=False)
