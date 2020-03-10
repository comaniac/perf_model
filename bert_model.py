import sys
import d2l
from mxnet import autograd, gluon, np, npx, init
from mxnet.gluon import nn

npx.set_np()

# Hyper-parameters
num_heads = 8
num_hiddens = num_heads # Force 1-to-1 attention
num_pred_hiddens = 512
ffn_num_hiddens = 1024
num_layers = 6
dropout = 0.3

### Class Declarations

class BERTEncoder(nn.Block):
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, num_layers, dropout, **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.blks = nn.Sequential()
        for _ in range(num_layers):
            self.blks.add(d2l.EncoderBlock(num_hiddens, ffn_num_hiddens, num_heads, dropout))

    def forward(self, features):
        X = features
        for blk in self.blks:
            X = blk(X, None)
        return X


class ThrptPred(nn.Block):
    def __init__(self, num_hiddens, **kwargs):
        super(ThrptPred, self).__init__(**kwargs)
        self.mlp = nn.Sequential()
        self.mlp.add(nn.Dense(num_pred_hiddens, activation='relu'))
        self.mlp.add(nn.Dense(1))

    def forward(self, X):
        # 0 is the index of the CLS token
        X = X[:, 0, :]
        # X shape: (batch size, num_hiddens)
        return self.mlp(X)


class BERTModel(nn.Block):
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, num_layers, dropout):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(num_hiddens, ffn_num_hiddens, num_heads, num_layers, dropout)
        self.nsp = ThrptPred(num_hiddens)

    def forward(self, features):
        encoded_X = self.encoder(features)
        nsp_Y_hat = self.nsp(encoded_X)
        return encoded_X, nsp_Y_hat


## Functions


def load_data(file_path):
    with open(file_path, 'r') as filep:
        next(filep) # Get rid of headers

        # Parse features to sequence. num_seq = num_feature + 1
        features = []
        thrpts = []
        cnt = 0
        for line in filep:
            if cnt > 10:
                break
            tokens = line.replace('\n', '').split(',')
            feature = [0] # initial <CLS> to 0
            for v in tokens[:-1]:
                try:
                    feature.append(float(v))
                except:
                    feature.append(0)

            # Expand featues to (batch, sequence, hidden)
            cnt += 1
            expand_feat = np.expand_dims(np.array(feature), axis=0)
            features.append(np.broadcast_to(expand_feat.T, shape=(len(feature), num_hiddens)))
            thrpts.append(float(tokens[-1]))

        splitter = int(len(features) * 0.8)
        train_feats = np.array(features[:splitter])
        train_thrpts = np.array(thrpts[:splitter])
        test_feats = np.array(features[splitter:])
        test_thrpts = np.array(thrpts[splitter:])
        return train_feats, train_thrpts, test_feats, test_thrpts


def get_batch_loss(net, loss, segments_X_shards, nsp_y_shards):
    ls = []
    for (segments_X_shard, nsp_y_shard) in zip(segments_X_shards, nsp_y_shards):
        # Forward pass
        nsp_Y_hat = net(segments_X_shard)

        # Compute prediction loss
        nsp_l = loss(nsp_Y_hat, nsp_y_shard)
        ls.append(nsp_l.mean())
        npx.waitall()
    return ls


def train(train_feats, train_thrpts, batch_size, num_steps):
    net = BERTModel(num_hiddens, ffn_num_hiddens, num_heads, num_layers, dropout)
    ctx = d2l.try_all_gpus()
    net.initialize(init.Xavier(), ctx=ctx)
    loss = gluon.loss.SoftmaxCELoss()

    trainer = gluon.Trainer(net.collect_params(), 'adam')
    step, timer = 0, d2l.Timer()
    metric = d2l.Accumulator(2)
    num_steps_reached = False
    while step < num_steps and not num_steps_reached:
        for start in range(len(train_feats), batch_size):
            segments_X_shards = train_feats[start:min(start + batch_size, len(train_feats))]
            nsp_y_shards = train_thrpts[start:min(start + batch_size, len(train_feats))]
            timer.start()
            with autograd.record():
                ls = get_batch_loss(net, loss, segments_X_shards, nsp_y_shards)
            for l in ls:
                l.backward()
            trainer.step(1)
            l_mean = sum([float(l) for l in ls]) / len(ls)
            metric.add(l_mean, 1)
            timer.stop()
            print('Step %d: %.2f' % (step, l_mean))
            step += 1
            if step == num_steps:
                num_steps_reached = True
                break

    print('NSP loss %.3f' % (metric[0] / metric[1]))
    return net



print('Loading data...')
train_feats, train_thrpts, test_feats, test_thrpts = load_data(sys.argv[1])

print('Training...')
bert = train(train_feats, train_thrpts, 8, 3)

print('Testing...')
predicts = bert(test_feats)
for idx in range(len(test_feats)):
    pred = predicts[1][idx].tolist()[0]
    real = test_thrpts[idx].tolist()
    print('%.2f <> %.2f : %.2f%%' % (pred, real, abs(pred - real) / real))
