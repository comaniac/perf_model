"""Use BERT as the performance model."""
# pylint: disable=invalid-name
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import d2l
import tqdm
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
from mxnet.gluon.utils import split_and_load

npx.set_np()

logging.basicConfig(format='[%(asctime)s] %(levelname)7s %(name)s: %(message)s')
logger = logging.getLogger()
logger.setLevel('INFO')

# Hyper-parameters
NUM_HEADS = 8
NUM_HIDDENS = NUM_HEADS # Force 1-to-1 attention
NUM_PRED_HIDDENS = 512
FFN_NUM_HIDDENS = 1024
NUM_LAYERS = 6
DROPOUT = 0.3
BATCH_SIZE = 32
NUM_EPOCHS = 20

### Class Declarations

class BERTEncoder(nn.Block):
    """BERT Encoder class."""
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, num_layers, dropout, **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.blks = nn.Sequential()
        for _ in range(num_layers):
            self.blks.add(d2l.EncoderBlock(num_hiddens, ffn_num_hiddens, num_heads, dropout))

    def forward(self, features): # pylint: disable=arguments-differ
        X = features
        for blk in self.blks:
            X = blk(X, None)
        return X


class ThrptPred(nn.Block):
    """The network to predict throughput."""
    def __init__(self, **kwargs):
        super(ThrptPred, self).__init__(**kwargs)
        self.mlp = nn.Sequential()
        self.mlp.add(nn.Dense(NUM_PRED_HIDDENS, activation='relu'))
        self.mlp.add(nn.Dense(1))

    def forward(self, X):  # pylint: disable=arguments-differ
        # 0 is the index of the CLS token
        X = X[:, 0, :]
        # X shape: (batch size, num_hiddens)
        return self.mlp(X)


class BERTModel(nn.Block):
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, num_layers, dropout):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(num_hiddens, ffn_num_hiddens, num_heads, num_layers, dropout)
        self.nsp = ThrptPred()

    def forward(self, features):  # pylint: disable=arguments-differ
        encoded_X = self.encoder(features)
        nsp_Y_hat = self.nsp(encoded_X)
        return encoded_X, nsp_Y_hat


## Functions

def expand_hidden(feature):
    #pylint: disable=redefined-outer-name
    import numpy as np
    expand_feat = np.expand_dims(np.array(feature), axis=0)
    return np.broadcast_to(expand_feat.T, shape=(len(feature), NUM_HIDDENS))

def load_data(file_path, batch_size):
    """Load tabular feature data."""
    logger.info('Parsing file...')
    with open(file_path, 'r') as filep:
        next(filep) # Get rid of headers

        # Parse features to sequence. num_seq = num_feature + 1
        features = []
        thrpts = []
        for line in tqdm.tqdm(filep):
            tokens = line.replace('\n', '').split(',')

            # initial <CLS> to 0
            features.append([0] + [float(v) for v in tokens[:-1]])
            thrpts.append(float(tokens[-1]))

    # Expand featues to (batch, sequence, hidden)
    logger.info('Expanding features...')
    with ProcessPoolExecutor(max_workers=8) as pool:
        expand_features = []
        for start in tqdm.tqdm(range(0, len(features), 8)):
            futures = [
                pool.submit(expand_hidden, feature=feature)
                for feature in features[start:min(start + 8, len(features))]
            ]
            for future in as_completed(futures):
                expand_features.append(future.result())
        features = expand_features

    splitter = int(len(features) * 0.8)
    train_feats = np.array(features[:splitter])
    train_thrpts = np.array(thrpts[:splitter])
    train_iter = gluon.data.DataLoader(
        gluon.data.ArrayDataset(train_feats, train_thrpts),
        batch_size,
        shuffle=True,
        num_workers=d2l.get_dataloader_workers())
    test_feats = np.array(features[splitter:])
    test_thrpts = np.array(thrpts[splitter:])
    return train_iter, test_feats, test_thrpts


def get_batch_loss(net, loss, segments_X_shards, nsp_y_shards):
    return loss(net(segments_X_shards)[1], nsp_y_shards)

    # ls =[]
    # for (segments_X_shard, nsp_y_shard) in zip(segments_X_shards, nsp_y_shards):
    #     pred = net(segments_X_shard)
    #     ls.append(loss(pred, nsp_y_shard).mean())

    # ls = []
    # nsp_Y_hats = net(segments_X_shards)[1]
    # logger.info(nsp_y_shards)
    # for (hat, shard) in zip(nsp_Y_hats, nsp_y_shards):
    #     ls.append(loss(np.expand_dims(hat, axis=0), np.expand_dims(shard, axis=0)))
    #return ls


def train(train_iter, ctx, num_epochs):
    """Train a BERT model."""
    net = BERTModel(NUM_HIDDENS, FFN_NUM_HIDDENS, NUM_HEADS, NUM_LAYERS, DROPOUT)
    net.initialize(init.Xavier(), ctx=ctx)
    logger.info('Model initialized on %s', str(ctx))
    loss = gluon.loss.L2Loss()

    trainer = gluon.Trainer(net.collect_params(), 'adam')
    epoch = 0
    metric = d2l.Accumulator(2)
    num_epochs_reached = False
    while epoch < num_epochs and not num_epochs_reached:
        logger.info('Epoch %d', epoch)
        progress = tqdm.tqdm(train_iter)
        for iter_idx, (batch_feat, batch_thrpt) in enumerate(progress):
            np_feat = split_and_load(batch_feat, ctx, even_split=False)[0]
            np_thrpt = split_and_load(batch_thrpt, ctx, even_split=False)[0]
            with autograd.record():
                ls = get_batch_loss(net, loss, np_feat, np_thrpt)
            ls.backward()
            trainer.step(1)
            l_mean = sum([float(l) for l in ls]) / len(ls)
            metric.add(l_mean, 1)
            if iter_idx % 30 == 0:
                progress.set_description_str(desc='Loss {:3f}'.format(l_mean), refresh=True)
        logger.info('Loss @ epoch %d: %.3f', epoch, metric[0] / metric[1])
        epoch += 1
        if epoch == num_epochs:
            num_epochs_reached = True
            break

    logger.info('Final loss %.3f', metric[0] / metric[1])
    return net


def run():
    """Main process"""
    ctx = d2l.try_all_gpus()

    logger.info('Loading data...')
    train_iter, test_feats, test_thrpts = load_data(sys.argv[1], BATCH_SIZE)
    test_feats = np.array(test_feats, ctx=ctx[0])
    test_thrpts = np.array(test_thrpts, ctx=ctx[0])

    logger.info('Training...')
    bert = train(train_iter, ctx, NUM_EPOCHS)

    logger.info('Testing...')
    predicts = bert(test_feats)
    errors = []
    for idx in range(len(test_feats)):
        pred = predicts[1][idx].tolist()[0]
        real = test_thrpts[idx].tolist()
        error = abs(pred - real) / real
        errors.append(error)
        logger.debug('Pred %.2f, Expected %.2f, Error %.2f%%', pred, real, 100 * error)
    logger.info('Average error rate: %.2f%%', 100 * np.array(errors).mean())

if __name__ == "__main__":
    run()
