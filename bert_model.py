"""Use BERT as the performance model."""
# pylint: disable=invalid-name
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import d2l
import tqdm
from mxnet import autograd, gluon, init, nd, np, npx
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
    def __init__(self, hiddens, **kwargs):
        super(ThrptPred, self).__init__(**kwargs)
        self.net = nn.Sequential()
        for hidden in hiddens:
            self.net.add(nn.Dense(hidden, activation='relu'))
        self.net.add(nn.Dense(1))

    def forward(self, X):  # pylint: disable=arguments-differ
        # 0 is the index of the CLS token
        X = X[:, 0, :]
        # X shape: (batch size, num_hiddens)
        return self.net(X)


class ValidPred(nn.Block):
    """The network to predict if the config is valid or not."""
    def __init__(self, hiddens, **kwargs):
        super(ValidPred, self).__init__(**kwargs)
        self.net = nn.Sequential()
        for hidden in hiddens:
            self.net.add(nn.Dense(hidden, activation='relu'))
        self.net.add(nn.Dense(1, activation='sigmoid'))

    def forward(self, X):  # pylint: disable=arguments-differ
        # X shape: (batch size, sequence length, num_hiddens)
        return self.net(X)


class BERTModel(nn.Block):
    """Define a BERT model."""
    def __init__(self, center_hiddens, center_ffn_hiddens, center_heads,
                 center_layers, reg_hiddens, cls_hiddens, dropout):
        super(BERTModel, self).__init__()
        self.center = BERTEncoder(center_hiddens, center_ffn_hiddens, center_heads,
                                  center_layers, dropout)
        # TODO(comaniac): Simplify this layer first.
        # self.reg_encoder = BERTEncoder(num_hiddens, ffn_num_hiddens, num_heads,
        #                                num_layers, dropout)
        self.reg_predictor = ThrptPred(reg_hiddens)
        self.valid_predictor = ValidPred(cls_hiddens)

    def forward(self, features):  # pylint: disable=arguments-differ
        encoded_X = self.center(features)
        valid_preds = self.valid_predictor(encoded_X)
        ret = []
        for idx, valid_pred in enumerate(valid_preds):
            if valid_pred >= 0.5:
                #encoded_X = self.reg_encoder(encoded_X)
                ret.append((valid_pred, self.reg_predictor(np.expand_dims(encoded_X[idx], axis=0))))
            else:
                ret.append((valid_pred, 0))
        return np.array(ret, ctx=features.ctx)

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


def get_batch_loss(net, valid_loss, thrpt_loss, segments_X_shards, nsp_y_shards):
    """Get loss of a batch."""
    #return loss(net(segments_X_shards), nsp_y_shards)
    alpha = 0.7 # FIXME: A more reasonable number.

    ls = []
    preds = net(segments_X_shards)
    for idx, (valid_pred, thrpt_pred) in enumerate(preds):
        label = np.array([1 if nsp_y_shards[idx] > 0 else 0], ctx=valid_pred.ctx)
        vl = valid_loss(valid_pred, label).mean()
        if nsp_y_shards[idx] == 0:
            ls.append(vl)
        else:
            tl = thrpt_loss(np.log(thrpt_pred + 1e-6),
                            np.log(nsp_y_shards[idx] + 1e-6)).mean()
            ls.append(vl + alpha * tl)
    return np.array(ls).mean()

    # ls = []
    # for idx, segments_X_shard in enumerate(segments_X_shards):
    #     valid_pred, thrpt_pred = net(np.expand_dims(segments_X_shard, axis=0))[0]
    #     label = np.array([1 if nsp_y_shards[idx] > 0 else 0], ctx=valid_pred.ctx)
    #     vl = valid_loss(valid_pred, label).mean()
    #     if nsp_y_shards[idx] == 0:
    #         ls.append(vl)
    #     else:
    #         tl = thrpt_loss(np.log(thrpt_pred + 1e-6),
    #                         np.log(nsp_y_shards[idx] + 1e-6)).mean()
    #         ls.append(vl + alpha * tl)
    # return ls



def test_acc(net, test_feats, test_thrpts, print_log=True):
    """Test the model accuracy."""
    valid_error = 0
    thrpt_errors = []
    for start in range(0, len(test_feats), 32):
        predicts = net(test_feats[start:min(start + 32, len(test_feats))])

        for idx, predict in enumerate(predicts):
            valid_pred, thrpt_pred = predict.tolist()
            real = test_thrpts[start + idx].tolist()
            valid_error += (valid_pred > 0.5 and real == 0) or (valid_pred <= 0.5 and real > 0)
            if valid_pred > 0.5:
                error = abs(thrpt_pred - real) / real
                thrpt_errors.append(error)
                if print_log:
                    log.info('Pred %.6f, Expected %.6f, Error %.2f%%',
                             thrpt_pred, real, 100 * error)
            else:
                log.info('Pred invalid, expected %.6f', real)
    if print_log:
        log.info('Valid error %.2f%%', valid_error / len(test_feats))
        log.info('Average error: %.2f (std %.2f)',
                 np.array(thrpt_errors).mean(),
                 np.array(thrpt_errors).std())
    return 1 - np.array(thrpt_errors).mean()


def train_bert(args, reporter=None):
    """Training process."""
    logger.enable_log_file('train-{}'.format(
        os.path.basename(args.data_file).replace('.csv', '.log')))

    ctx = d2l.try_all_gpus()

    log.info('Loading data from %s...', args.data_file)
    train_iter, validate_feats, validate_thrpts, test_feats, test_thrpts = load_data(
        args.data_file, args.batch_size, args.center_hiddens)
    validate_feats = np.array(validate_feats, ctx=ctx[0])
    validate_thrpts = np.array(validate_thrpts, ctx=ctx[0])
    test_feats = np.array(test_feats, ctx=ctx[0])
    test_thrpts = np.array(test_thrpts, ctx=ctx[0])


    fnet = [
        'Center Heads: {}'.format(args.center_heads),
        'Center FFN Hiddens: {}'.format(args.center_ffn_hiddens),
        'Center Layers: {}'.format(args.center_layers),
        'Valid Hiddens: {}'.format(args.cls_hiddens),
        'Regression Hiddens: {}'.format(args.reg_hiddens),
    ]
    log.info('Network\n\t%s', '\n\t'.join(fnet))
    fparams = [
        'Batch Size: {}'.format(args.batch_size),
        'Dropout: {}'.format(args.dropout),
        'Epochs: {}'.format(args.epochs),
        'Learning Rate: {}'.format(args.lr),
        'Weight Decay: {}'.format(args.wd)
    ]
    log.info('Hyper-Parameters:\n\t%s', '\n\t'.join(fparams))

    net = BERTModel(args.center_hiddens, args.center_ffn_hiddens,
                    args.center_heads, args.center_layers, args.reg_hiddens,
                    args.cls_hiddens, args.dropout)
    net.initialize(init.Xavier(), ctx=ctx)
    log.info('Model initialized on %s', str(ctx))

    # Initialize trainer.
    trainer = gluon.Trainer(net.collect_params(), 'adam', {
        'learning_rate': args.lr,
        'wd': args.wd
    })

    # Initialize loss functions.
    # FIXME: Dynamic weight.
    valid_loss = gluon.loss.SigmoidBCELoss(from_sigmoid=True, weight=0.5)
    thrpt_loss = gluon.loss.L2Loss()

    log.info('Training...')
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
                ls = get_batch_loss(net, valid_loss, thrpt_loss, np_feat, np_thrpt)
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

    main_args.center_heads = 16
    main_args.center_hiddens = main_args.center_heads  # Force 1-to-1 attention
    main_args.center_layers = 12
    main_args.center_ffn_hiddens = 1024
    main_args.reg_hiddens = [1024]
    main_args.cls_hiddens = [512, 32]

    main_args.data_file = sys.argv[1]
    main_args.batch_size = 1024
    main_args.dropout = 0.2
    main_args.epochs = 80
    main_args.lr = 1e-5
    main_args.wd = 1
    train_bert(main_args)
