"""Use BERT as the performance model."""
# pylint: disable=invalid-name
import argparse
import os
import sys

import d2l
import tqdm
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
from mxnet.gluon.utils import split_and_load

import logger

npx.set_np()

log = logger.get_logger('VALID')

INVALID_THD = 1 # Invalid throughput threshold (GFLOP/s).
INVALID_LOG_THD = np.log(INVALID_THD)

### Class Declarations


class ValidNet(gluon.HybridBlock):
    """The network to predict if the config is valid or not."""
    def __init__(self, hiddens, dropout):
        super().__init__(prefix=None, params=None)
        with self.name_scope():
            self.net = nn.HybridSequential()
            for hidden in hiddens:
                self.net.add(nn.Dense(hidden, activation='relu'))
                self.net.add(nn.Dropout(dropout))
            self.net.add(nn.Dense(2))

    def hybrid_forward(self, F, x, *args, **kwargs):
        return self.net(x)

## Functions

def load_data(file_path, batch_size):
    """Load tabular feature data."""
    log.info('Parsing file...')
    with open(file_path, 'r') as filep:
        next(filep)  # Get rid of headers

        # Parse features to sequence.
        features = []
        valids = []
        for line in tqdm.tqdm(filep):
            tokens = line.replace('\n', '').split(',')
            features.append([float(v) for v in tokens[:-1]])
            valids.append(1 if float(tokens[-1]) > INVALID_THD else 0)

    log.info('Total data size %d', len(features))

    # 70% for training.
    # 10% for validation.
    # 20% for testing.
    splitter1 = int(len(features) * 0.7)
    splitter2 = int(len(features) * 0.8)
    train_feats = np.array(features[:splitter1])
    train_valid = np.array(valids[:splitter1])

    # Calculate imbalance weight.
    num_valid = len(train_valid.nonzero()[0])
    num_invalid = len(train_valid) - num_valid
    pos_weight = num_invalid / num_valid

    # One-hot encoding
    #train_valid = nd.one_hot(train_valid.as_nd_ndarray(), 2).as_np_ndarray()

    train_iter = gluon.data.DataLoader(
        gluon.data.ArrayDataset(train_feats, train_valid),
        batch_size,
        shuffle=True,
        num_workers=d2l.get_dataloader_workers())
    validate_feats = np.array(features[splitter1:splitter2])
    validate_valids = np.array(valids[splitter1:splitter2])
    test_feats = np.array(features[splitter2:])
    test_valids = np.array(valids[splitter2:])
    return train_iter, pos_weight, validate_feats, validate_valids, test_feats, test_valids


def get_batch_loss(net, loss, feats, valids):
    """Get loss of a batch."""
    #    ls = []
    # for feat, valid in zip(feats, valids):
    #     valid_pred = net(feat)
    #     ls.append(loss(valid_pred, np.expand_dims(valid, axis=1)).mean())
    # return ls

    valid_preds = net(feats)
    return loss(valid_preds, valids).mean()

def test_acc(net, test_feats, test_valids, print_log=True):
    """Test the model accuracy."""
    valid_error = 0
    pred_min = 1
    pred_max = 0
    for start in range(0, len(test_feats), 512):
        valid_preds = net(test_feats[start:min(start + 512, len(test_feats))])
        valid_hats = test_valids[start:min(start + 512, len(test_valids))]
        for pred_prob, hat in zip(valid_preds, valid_hats):
            pred_min = min(min(pred_prob), pred_min)
            pred_max = max(max(pred_prob), pred_max)
            if hat == 0:
                valid_error += 1 if pred_prob[0] <= pred_prob[1] else 0
            else:
                valid_error += 1 if pred_prob[0] > pred_prob[1] else 0
            if print_log:
                log.debug('Expected %s, predict (%.2f, %.2f)', hat, pred_prob[0], pred_prob[1])

    valid_err_rate = 100.0 * valid_error / len(test_valids)
    if print_log:
        log.info('Valid error %.2f%%', valid_err_rate)
    log.info('Predict range %.2f to %.2f', pred_min, pred_max)
    return valid_err_rate


def train_model(args, reporter=None):
    """Training process."""
    logger.enable_log_file('train-{}'.format(
        os.path.basename(args.data_file).replace('.csv', '.log')))

    ctx = d2l.try_all_gpus()

    log.info('Loading data from %s...', args.data_file)
    (train_iter, pos_weight, validate_feats, validate_valids, test_feats, test_valids) = load_data(
        args.data_file, args.batch_size)
    validate_feats = np.array(validate_feats, ctx=ctx[0])
    validate_valids = np.array(validate_valids, ctx=ctx[0])
    test_feats = np.array(test_feats, ctx=ctx[0])
    test_valids = np.array(test_valids, ctx=ctx[0])

    # Initialize loss function.
    log.info('Positive weight for CELoss: %.2f', pos_weight)
    valid_loss = gluon.loss.SoftmaxCrossEntropyLoss() #(weight=pos_weight, sparse_label=True)


    fnet = [
        'Valid Hiddens: {}'.format(args.cls_hiddens),
        'Dropout: {}'.format(args.dropout),
    ]
    log.info('Network\n\t%s', '\n\t'.join(fnet))
    fparams = [
        'Batch Size: {}'.format(args.batch_size),
        'Epochs: {}'.format(args.epochs),
        'Learning Rate: {}'.format(args.lr),
        'Weight Decay: {}'.format(args.wd)
    ]
    log.info('Hyper-Parameters:\n\t%s', '\n\t'.join(fparams))

    net = ValidNet(args.cls_hiddens, args.dropout)
    net.hybridize()
    net.initialize(init.Xavier(), ctx=ctx)
    log.info('Model initialized on %s', str(ctx))

    # Initialize trainer.
    trainer = gluon.Trainer(net.collect_params(), 'adam', {
        'learning_rate': args.lr,
        'wd': args.wd
    })

    log.info('Training...')
    metric = d2l.Accumulator(2)
    for epoch in range(args.epochs):
        # if epoch % 10 == 0:
        #     args.lr *= 0.1
        #     trainer.set_learning_rate(args.lr)
        #     log.info('Reset learning rate to %e', args.lr)
        if reporter is None:
            log.info('Epoch %d', epoch)
            progress = tqdm.tqdm(
                train_iter, bar_format='{desc} {percentage:3.0f}%|{bar:50}{r_bar}')
        else:
            progress = train_iter
        for iter_idx, (batch_feat, batch_valids) in enumerate(progress):
            np_feat = split_and_load(batch_feat, ctx, even_split=False)[0]
            np_valid = split_and_load(batch_valids, ctx, even_split=False)[0]
            with autograd.record():
                ls = get_batch_loss(net, valid_loss, np_feat, np_valid)
            ls.backward()
            l_mean = float(ls)
            trainer.step(1)
            metric.add(l_mean, 1)
            if reporter is None and iter_idx % 30 == 0:
                progress.set_description_str(desc='Loss {:.3f}'.format(l_mean), refresh=True)
        #val_acc = test_acc(net, validate_feats, validate_valids, print_log=False)
        #if reporter is None:
        #    log.info('Epoch %d: Loss %.3f, Valid Error %.2f%%', epoch,
        #             metric[0] / metric[1], val_acc)
        #else:
        #    reporter(epoch=epoch, accuracy=val_acc)

    #if reporter is None:
    #    log.info('Final loss %.3f', metric[0] / metric[1])

    log.info('Testing...')
    test_acc(net, test_feats, test_valids)
    return net


if __name__ == "__main__":
    main_args = argparse.Namespace()

    main_args.cls_hiddens = [128, 128]

    main_args.data_file = sys.argv[1]
    main_args.batch_size = 128
    main_args.dropout = 0.1
    main_args.epochs = 20
    main_args.lr = 1e-2
    main_args.wd = 0.0001
    file_name = sys.argv[2]
    train_model(main_args).export(file_name)

