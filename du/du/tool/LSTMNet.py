import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l

def process_data(path,batch_size,rate,num_steps):
    df = pd.read_csv(path)
    l = df['reviewContent']
    rating = df['rating']
    usefulCount = df['usefulCount']
    coolCount = df['coolCount']
    funnyCount = df['funnyCount']
    flagged = df['flagged']
    paragraphs = [i.strip().lower() for i in l]

    train_rating = list(rating[:int(len(df)*rate)])
    train_usefulCount =list(usefulCount[:int(len(df)*rate)])
    train_coolCount =list(coolCount[:int(len(df)*rate)])
    train_funnyCount = list(funnyCount[:int(len(df)*rate)])
    train_total = [[train_rating[i],train_usefulCount[i],train_coolCount[i],train_funnyCount[i]]for i in range(len(train_rating))]

    test_rating = list(rating[int(len(df)*rate):])
    test_usefulCount = list(usefulCount[int(len(df)*rate):])
    test_coolCount = list(coolCount[int(len(df)*rate):])
    test_funnyCount =list(funnyCount[int(len(df)*rate):])
    test_total = [[test_rating[i], test_usefulCount[i], test_coolCount[i], test_funnyCount[i]] for i in range(len(test_rating))]

    train_sen = paragraphs[:int(len(df)*rate)]
    test_sen = paragraphs[int(len(df)*rate):]

    train_tokens = d2l.tokenize(train_sen, token='word')
    vocab = d2l.Vocab(train_tokens, min_freq=5, reserved_tokens=['<pad>'])
    test_tokens = d2l.tokenize(test_sen,token='word')

    # num_steps = 128  # 序列长度
    d2l.set_figsize()
    d2l.plt.xlabel('# tokens per review')
    d2l.plt.ylabel('count')
    d2l.plt.hist([len(line) for line in train_tokens], bins=range(0, 1000, 50))

    train_features = torch.tensor([d2l.truncate_pad(vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
    test_features = torch.tensor([d2l.truncate_pad(vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
    train_features = train_features.tolist()
    test_features = test_features.tolist()
    train_data = [train_features[i]+train_total[i] for i in range(len(train_features))]
    train_data = torch.tensor(train_data)

    test_data = [test_features[i]+test_total[i] for i in range(len(test_features))]
    test_data = torch.tensor(test_data)

    labels = [0 if x == 'Y' else 1 for x in flagged]
    train_labels = labels[:int(len(df)*rate)]
    test_labels = labels[int(len(df)*rate):]

    train_iter = d2l.load_array((train_data,torch.tensor(train_labels)), batch_size)
    test_iter = d2l.load_array((test_data,torch.tensor(test_labels)),batch_size)

    return train_iter,test_iter,vocab

def revoacb():
    df = pd.read_csv(r'data/review_final.csv')
    l = df['reviewContent']
    paragraphs = [i.strip().lower() for i in l]
    train_sen = paragraphs[:int(len(df)*0.8)]
    train_tokens = d2l.tokenize(train_sen, token='word')
    vocab = d2l.Vocab(train_tokens, min_freq=5, reserved_tokens=['<pad>'])
    return vocab

class BiRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens,
                 num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 将bidirectional以获取双设置为True向循环神经网络
        self.encoder = nn.LSTM(embed_size, num_hiddens, num_layers=num_layers,
                                bidirectional=True)
        self.decoder = nn.Linear(4 * num_hiddens, 2)

    def forward(self, inputs):
        # inputs的形状是（批量大小，时间步数）
        # 因为长短期记忆网络要求其输入的第一个维度是时间维，
        # 所以在获得词元表示之前，输入会被转置。
        # 输出形状为（时间步数，批量大小，词向量维度）
        embeddings = self.embedding(inputs.T)
        self.encoder.flatten_parameters()
        # 返回上一个隐藏层在不同时间步的隐状态，
        # outputs的形状是（时间步数，批量大小，2*隐藏单元数）
        outputs, _ = self.encoder(embeddings)
        # 连结初始和最终时间步的隐状态，作为全连接层的输入，
        # 其形状为（批量大小，4*隐藏单元数）
        encoding = torch.cat((outputs[0], outputs[-1]), dim=1)
        outs = self.decoder(encoding)
        return outs

def predict_sentiment(net, sequence, rating = 0, usefulCount = 0, coolCount = 0, funnyCount = 0):
    """预测文本序列的情感"""
    vocab = revoacb()
    sequence = torch.tensor(vocab[sequence.split()], device=d2l.try_gpu())
    sequence = sequence.tolist() + [rating, usefulCount, coolCount, funnyCount]
    sequence = torch.tensor(sequence, device=d2l.try_all_gpus()[0])
    label = torch.argmax(net(sequence.reshape(1, -1)), dim=1)
    return 'Y' if label == 1 else 'N'

def train_LSTM(lr, num_epochs):
    batch_size = 256
    train_iter, test_iter, vocab = process_data(r'data/review_final.csv', batch_size=batch_size, rate=0.9,
                                                num_steps=300)
    embed_size, num_hiddens, num_layers = 256, 256, 2
    devices = d2l.try_all_gpus()
    net = BiRNN(len(vocab), embed_size, num_hiddens, num_layers)

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.LSTM:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(init_weights)
    glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
    embeds = glove_embedding[vocab.idx_to_token]
    net.embedding.weight.data.copy_(embeds)
    net.embedding.weight.requires_grad = False
    # lr, num_epochs = 0.001, 50
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss(reduction="none")
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
                   devices)

    torch.save(net, 'LSTM_sentiment_analysis.pth')
