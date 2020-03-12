"""Use BERT as the performance model."""
# pylint: disable=invalid-name
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import d2l
import tqdm
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
from mxnet.gluon.utils import split_and_load

import logger

npx.set_np()

log = logger.get_logger('BERT')


### Class Declarations


class BERTEncoder(nn.Block):
    """BERT Encoder class."""
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, num_layers,
                 dropout, **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.blks = nn.Sequential()
        for _ in range(num_layers):
            self.blks.add(
                d2l.EncoderBlock(num_hiddens, ffn_num_hiddens, num_heads,
                                 dropout))

    def forward(self, features):  # pylint: disable=arguments-differ
        X = features
        for blk in self.blks:
            X = blk(X, None)
        return X


class ThrptPred(nn.Block):
    """The network to predict throughput."""
    def __init__(self, num_pred_hiddens, **kwargs):
        super(ThrptPred, self).__init__(**kwargs)
        self.mlp = nn.Sequential()
        self.mlp.add(nn.Dense(num_pred_hiddens, activation='relu'))
        self.mlp.add(nn.Dense(1))

    def forward(self, X):  # pylint: disable=arguments-differ
        # 0 is the index of the CLS token
        X = X[:, 0, :]
        # X shape: (batch size, num_hiddens)
        return self.mlp(X)


class BERTModel(nn.Block):
    """Define a BERT model."""
    def __init__(self, num_hiddens, ffn_num_hiddens, num_pred_hiddens,
                 num_heads, num_layers, dropout):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(num_hiddens, ffn_num_hiddens, num_heads,
                                   num_layers, dropout)
        self.nsp = ThrptPred(num_pred_hiddens)

    def forward(self, features):  # pylint: disable=arguments-differ
        encoded_X = self.encoder(features)
        nsp_Y_hat = self.nsp(encoded_X)
        return nsp_Y_hat


## Functions


def expand_hidden(feature, num_hiddens):
    """Expand features with number of hidden layers."""
    #pylint: disable=redefined-outer-name
    import numpy as np
    expand_feat = np.expand_dims(np.array(feature), axis=0)
    return np.broadcast_to(expand_feat.T, shape=(len(feature), num_hiddens))


def load_data(file_path, batch_size, num_hiddens):
    """Load tabular feature data."""
    log.info('Parsing file...')
    with open(file_path, 'r') as filep:
        next(filep)  # Get rid of headers

        # Parse features to sequence. num_seq = num_feature + 1
        features = []
        thrpts = []
        for line in tqdm.tqdm(filep):
            tokens = line.replace('\n', '').split(',')

            # initial <CLS> to 0
            features.append([0] + [float(v) for v in tokens[:-1]])
            thrpts.append(float(tokens[-1]))

    # Expand featues to (batch, sequence, hidden)
    log.info('Expanding features...')
    with ProcessPoolExecutor(max_workers=8) as pool:
        expand_features = []
        for start in tqdm.tqdm(range(0, len(features), 8)):
            futures = [
                pool.submit(expand_hidden, feature=feature, num_hiddens=num_hiddens)
                for feature in features[start:min(start + 8, len(features))]
            ]
            for future in as_completed(futures):
                expand_features.append(future.result())
        features = expand_features

    log.info('Total data size %d', len(features))

    # 70% for training.
    # 10% for validation.
    # 20% for testing.
    splitter1 = int(len(features) * 0.7)
    splitter2 = int(len(features) * 0.8)
    train_feats = np.array(features[:splitter1])
    train_thrpts = np.array(thrpts[:splitter1])
    train_iter = gluon.data.DataLoader(
        gluon.data.ArrayDataset(train_feats, train_thrpts),
        batch_size,
        shuffle=True,
        num_workers=d2l.get_dataloader_workers())
    validate_feats = np.array(features[splitter1:splitter2])
    validate_thrpts = np.array(thrpts[splitter1:splitter2])
    test_feats = np.array(features[splitter2:])
    test_thrpts = np.array(thrpts[splitter2:])
    return train_iter, validate_feats, validate_thrpts, test_feats, test_thrpts


def get_batch_loss(net, loss, segments_X_shards, nsp_y_shards):
    """Get loss of a batch."""
    return loss(net(segments_X_shards), nsp_y_shards)


def test_acc(net, test_feats, test_thrpts, print_log=True):
    """Test the model accuracy."""
    errors = []
    for start in range(0, len(test_feats), 32):
        predicts = net(test_feats[start:min(start + 32, len(test_feats))])

        for idx, predict in enumerate(predicts):
            pred = np.exp(predict.tolist()[0])
            real = np.exp(test_thrpts[start + idx].tolist())
            error = abs(pred - real) / real
            errors.append(error)
            if print_log:
                log.info('Pred %.6f, Expected %.6f, Error %.2f%%', pred, real, 100 * error)
    if print_log:
        log.info('Average error: %.2f (std %.2f)', np.array(errors).mean(), np.array(errors).std())
    return 1 - np.array(errors).mean()


def train_bert(args, reporter=None):
    """Training process."""
    logger.enable_log_file('train-{}'.format(
        os.path.basename(args.data_file).replace('.csv', '.log')))

    ctx = d2l.try_all_gpus()

    log.info('Loading data from %s...', args.data_file)
    train_iter, validate_feats, validate_thrpts, test_feats, test_thrpts = load_data(
        args.data_file, args.batch_size, args.num_hiddens)
    validate_feats = np.array(validate_feats, ctx=ctx[0])
    validate_thrpts = np.array(validate_thrpts, ctx=ctx[0])
    test_feats = np.array(test_feats, ctx=ctx[0])
    test_thrpts = np.array(test_thrpts, ctx=ctx[0])


    log.info('Training for %d epochs...', args.epochs)
    fparams = [
        'Batch Size: {}'.format(args.batch_size),
        'Num Heads: {}'.format(args.num_heads),
        'Pred Hiddens: {}'.format(args.num_pred_hiddens),
        'Num Layers: {}'.format(args.num_layers),
        'Dropout: {}'.format(args.dropout),
        'FFN Num Hiddens: {}'.format(args.ffn_num_hiddens),
        'Epochs: {}'.format(args.epochs),
        'Learning Rate: {}'.format(args.lr),
        'Weight Decay: {}'.format(args.wd)
    ]
    log.info('Hyper-Parameters:\n\t%s', '\n\t'.join(fparams))

    net = BERTModel(args.num_hiddens, args.ffn_num_hiddens,
                    args.num_pred_hiddens, args.num_heads, args.num_layers,
                    args.dropout)
    net.initialize(init.Xavier(), ctx=ctx)
    log.info('Model initialized on %s', str(ctx))
    loss = gluon.loss.L2Loss()

    trainer = gluon.Trainer(net.collect_params(), 'adam', {
        'learning_rate': args.lr,
        'wd': args.wd
    })

    metric = d2l.Accumulator(2)
    for epoch in range(args.epochs):
        if reporter is None:
            log.info('Epoch %d', epoch)
            progress = tqdm.tqdm(
                train_iter, bar_format='{desc}{percentage:3.0f}%|{bar:50}{r_bar}')
        else:
            progress = train_iter
        for iter_idx, (batch_feat, batch_thrpt) in enumerate(progress):
            np_feat = split_and_load(batch_feat, ctx, even_split=False)[0]
            np_thrpt = split_and_load(batch_thrpt, ctx, even_split=False)[0]
            with autograd.record():
                ls = get_batch_loss(net, loss, np_feat, np_thrpt)
            ls.backward()
            trainer.step(1)
            l_mean = sum([float(l) for l in ls]) / len(ls)
            metric.add(l_mean, 1)
            if reporter is None and iter_idx % 30 == 0:
                progress.set_description_str(desc='Loss {:3f}'.format(l_mean),
                                             refresh=True)
        val_acc = test_acc(net, validate_feats, validate_thrpts, print_log=False)
        if reporter is None:
            log.info('Epoch %d: Loss %.3f, Validate Error %.3f', epoch,
                     metric[0] / metric[1], val_acc)
        else:
            reporter(epoch=epoch, accuracy=val_acc)

    if reporter is None:
        log.info('Final loss %.3f', metric[0] / metric[1])

    log.info('Testing...')
    test_acc(net, test_feats, test_thrpts)
    return net


if __name__ == "__main__":
    main_args = argparse.Namespace()
    main_args.data_file = sys.argv[1]
    main_args.batch_size = 1024
    main_args.num_heads = 8
    main_args.num_hiddens = main_args.num_heads  # Force 1-to-1 attention
    main_args.num_pred_hiddens = 512
    main_args.num_layers = 6
    main_args.dropout = 0.2
    main_args.ffn_num_hiddens = 1024
    main_args.epochs = 40
    main_args.lr = 1e-4
    main_args.wd = 1
    train_bert(main_args)
