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

log = logger.get_logger('Thrpt')

INVALID_THD = 10  # Invalid throughput threshold ratio.
INVALID_LOG_THD = np.log(INVALID_THD)

### Class Declarations


class ThrptPred(nn.Block):
    """The network to predict throughput."""
    def __init__(self, hiddens, num_cls, dropout, **kwargs):
        super(ThrptPred, self).__init__(**kwargs)
        self.net = nn.Sequential()
        for hidden in hiddens:
            self.net.add(nn.Dense(hidden, activation='relu'))
            self.net.add(nn.Dropout(dropout))
        self.net.add(nn.Dense(num_cls))

    def forward(self, X):  # pylint: disable=arguments-differ
        return self.net(X)

## Functions


def find_class(thrpt, boundaries):
    """Determine which class of the given throughput should go."""
    for idx, boundary in enumerate(boundaries):
        if thrpt <= boundary:
            return idx
    return -1

def load_data(file_path, batch_size, num_cls):
    """Load tabular feature data."""
    log.info('Parsing file...')
    with open(file_path, 'r') as filep:
        next(filep)  # Get rid of headers

        # Parse features to sequence. num_seq = num_feature + 1
        features = []
        thrpts = []
        for line in tqdm.tqdm(filep):
            tokens = line.replace('\n', '').split(',')

            # Filter out invalid records
            thrpt = float(tokens[-1])
            if thrpt <= INVALID_THD:
                continue

            # initial <CLS> to 0
            features.append([0] + [float(v) for v in tokens[:-1]])
            thrpts.append(thrpt)

    log.info('Total data size %d', len(features))

    # Data balancing

    # 70% for training.
    # 10% for validation.
    # 20% for testing.
    splitter1 = int(len(features) * 0.7)
    splitter2 = int(len(features) * 0.8)
    train_feats = np.array(features[:splitter1])
    train_thrpts = np.array(thrpts[:splitter1])

    log.info('Train thrpt min max: %.2f %.2f',
             min(train_thrpts).tolist(),
             max(train_thrpts).tolist())

    # Identify throughput class boundaries.
    sorted_thrpts = [e.tolist() for e in sorted(train_thrpts)]  #np.unique(
    cls_size = len(sorted_thrpts) // num_cls
    boundaries = [sorted_thrpts[-1]]
    for ridx in range(num_cls - 1, 0, -1):
        boundaries.append(sorted_thrpts[ridx * cls_size])
    boundaries.reverse()

    # Transform throughputs to classes.
    log.info('Transforming throughput to class...')
    cls_thrpts = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        for start in tqdm.tqdm(
                range(0, len(thrpts), 8),
                bar_format='{desc}{percentage:3.0f}%|{bar:50}{r_bar}'):
            futures = [
                pool.submit(find_class, thrpt=thrpt, boundaries=boundaries)
                for thrpt in thrpts[start:min(start + 8, len(thrpts))]
            ]
            for future in as_completed(futures):
                cls_thrpts.append(future.result())
    train_thrpts = np.array(cls_thrpts[:splitter1], dtype='int32')

    # Statistics
    buckets = [0 for _ in range(num_cls)]
    for thrpt_cls in train_thrpts:
        buckets[thrpt_cls.tolist()] += 1
    log.debug('Training throughput distributions')
    for idx, (boundary, bucket) in enumerate(zip(boundaries, buckets)):
        log.debug('\t%02d (<=%.2f): %d', idx, boundary, bucket)

    train_iter = gluon.data.DataLoader(
        gluon.data.ArrayDataset(train_feats, train_thrpts),
        batch_size,
        shuffle=True,
        num_workers=d2l.get_dataloader_workers())
    validate_feats = np.array(features[splitter1:splitter2])
    validate_thrpts = np.array(cls_thrpts[splitter1:splitter2], dtype='int32')
    test_feats = np.array(features[splitter2:])
    test_thrpts = np.array(cls_thrpts[splitter2:], dtype='int32')
    return (train_iter, validate_feats, validate_thrpts, test_feats, test_thrpts, boundaries)


def get_batch_loss(net, thrpt_loss, segments_X_shards, nsp_y_shards, sample_weight):
    """Get loss of a batch."""

    thrpt_preds = net(segments_X_shards)
    return thrpt_loss(thrpt_preds, nsp_y_shards, sample_weight).mean()


def test_acc(net, test_feats, test_thrpts, print_log=True):
    """Test the model accuracy."""

    if print_log:
        log.info('\nExpected\tThrpt-Pred\tThrpt-Error')

    thrpt_pred_range = (float('inf'), 0)

    # Confusion matrix for each class.
    error = 0
    TP = []
    FN = []
    FP = []

    for start in range(0, len(test_feats), 512):
        thrpt_preds = net(test_feats[start:min(start + 512, len(test_feats))])

        for idx, thrpt_pred_prob in enumerate(thrpt_preds):
            real = test_thrpts[start + idx].tolist()
            thrpt_pred = int(np.argmax(thrpt_pred_prob).tolist())

            while len(TP) <= max(real, thrpt_pred):
                TP.append(0)
                FN.append(0)
                FP.append(0)

            if thrpt_pred == real:
                TP[real] += 1
            else:
                error += 1
                FN[real] += 1
                FP[thrpt_pred] += 1

            thrpt_pred_range = (min(thrpt_pred_range[0], thrpt_pred),
                                max(thrpt_pred_range[1], thrpt_pred))
            if print_log:
                log.info('\t%d\t%d', real, thrpt_pred)

    accuracy = 100.0 * (1.0 - (error / len(test_feats)))
    recalls = [
        '{:.2f}%'.format(100.0 * tp / (tp + fn)) if tp + fn > 0 else 'N/A'
        for tp, fn in zip(TP, FN)
    ]
    precisions = [
        '{:.2f}%'.format(100.0 * tp / (tp + fp)) if tp + fp > 0 else 'N/A'
        for tp, fp in zip(TP, FP)
    ]

    log.info('Thrpt predict range: %d, %d', thrpt_pred_range[0], thrpt_pred_range[1])
    log.info('Recalls: %s', ', '.join(recalls[-3:-1]))
    log.info('Precisions: %s', ', '.join(precisions[-3:-1]))

    if print_log:
        log.info('Accuracy: %.2f%%', accuracy)
    return accuracy


def train_bert(args, reporter=None):
    """Training process."""
    logger.enable_log_file('train-{}'.format(
        os.path.basename(args.data_file).replace('.csv', '.log')))

    ctx = d2l.try_all_gpus()

    log.info('Loading data from %s...', args.data_file)
    (train_iter, validate_feats, validate_thrpts, test_feats, test_thrpts,
     _) = load_data(args.data_file, args.batch_size, args.num_cls)
    validate_feats = np.array(validate_feats, ctx=ctx[0])
    validate_thrpts = np.array(validate_thrpts, ctx=ctx[0])
    test_feats = np.array(test_feats, ctx=ctx[0])
    test_thrpts = np.array(test_thrpts, ctx=ctx[0])

    fnet = [
        'Regression Hiddens: {}'.format(args.reg_hiddens),
        'Output Classes: {}'.format(args.num_cls)
    ]
    log.info('Network\n\t%s', '\n\t'.join(fnet))
    fparams = [
        'Batch Size: {}'.format(args.batch_size),
        'Dropout: {}'.format(args.dropout), 'Epochs: {}'.format(args.epochs),
        'Learning Rate: {}'.format(args.lr), 'Weight Decay: {}'.format(args.wd)
    ]
    log.info('Hyper-Parameters:\n\t%s', '\n\t'.join(fparams))

    net = ThrptPred(args.reg_hiddens, args.num_cls, args.dropout)
    net.initialize(init.Xavier(), ctx=ctx)
    log.info('Model initialized on %s', str(ctx))

    # Initialize trainer.
    trainer = gluon.Trainer(net.collect_params(), 'adam', {
        'learning_rate': args.lr,
        'wd': args.wd
    })

    # Initialize loss functions.
    thrpt_loss = gluon.loss.SoftmaxCrossEntropyLoss()
    sample_weight = np.array(list(range(1, 2 * args.num_cls, 2)), ctx=ctx[0])

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
                train_iter,
                bar_format='{desc} {percentage:3.0f}%|{bar:50}{r_bar}')
        else:
            progress = train_iter
        for iter_idx, (batch_feat, batch_thrpt) in enumerate(progress):
            np_feat = split_and_load(batch_feat, ctx, even_split=False)[0]
            np_thrpt = split_and_load(batch_thrpt, ctx, even_split=False)[0]
            with autograd.record():
                ls = get_batch_loss(net, thrpt_loss, np_feat, np_thrpt, sample_weight)
            ls.backward()
            trainer.step(1)
            l_mean = float(ls)
            metric.add(l_mean, 1)
            if reporter is None and iter_idx % 30 == 0:
                progress.set_description_str(desc='Loss {:.3f}'.format(l_mean), refresh=True)
        val_acc = test_acc(net, validate_feats, validate_thrpts, print_log=False)
        if reporter is None:
            log.info('Epoch %d: Loss %.3f, Accuracy %.2f%%', epoch,
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

    main_args.reg_hiddens = [512, 512]
    main_args.num_cls = 100

    main_args.data_file = sys.argv[1]
    main_args.batch_size = 128
    main_args.dropout = 0.1
    main_args.epochs = 40
    main_args.lr = 1e-3
    main_args.wd = 1e-4
    file_name = sys.argv[2]
    train_bert(main_args).save_parameters(file_name)
