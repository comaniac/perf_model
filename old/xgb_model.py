"""Train a XGBoost model."""
import argparse
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import tqdm
import xgboost as xgb

import logger

INVALID_THD = 10

log = logger.get_logger('XGB')


def create_config():
    """Create the config parser of this app."""
    parser = argparse.ArgumentParser(description='Cost Model')
    subparser = parser.add_subparsers(dest='mode',
                                      description='Execution mode')
    subparser.required = True

    train = subparser.add_parser('train', help='Train cost model')
    train.add_argument('path', help='Feature file path or folder')
    train.add_argument('--n_class',
                       type=int,
                       default=100,
                       help='The class number')

    return parser.parse_args()


def find_class(thrpt, boundaries):
    """Determine which class of the given throughput should go."""
    for idx, boundary in enumerate(boundaries):
        if thrpt <= boundary:
            return idx
    return -1


def load_data(num_cls, file_path):
    """Load tabular feature data."""

    np.random.seed(23)

    log.info('Parsing file...')
    data = []
    with open(file_path, 'r') as filep:
        headers = next(filep).replace('\n', '').split(',')[:-1]

        # Parse features to sequence. num_seq = num_feature + 1
        for line in tqdm.tqdm(filep):
            tokens = line.replace('\n', '').split(',')

            # Filter out invalid records
            if float(tokens[-1]) <= INVALID_THD:
                continue

            data.append([float(v) for v in tokens])

    data = np.array(data)
    np.random.shuffle(data)
    features = np.array(data.T[:-1].T)
    thrpts = np.array(data.T[-1].T)

    log.info('Total data size %d', len(features))

    # 80% for training.
    # 20% for testing.
    splitter = int(len(features) * 0.8)
    train_feats = np.array(features[:splitter])
    train_thrpts = np.array(thrpts[:splitter])

    log.info('Train thrpt min max: %.2f %.2f', min(train_thrpts), max(train_thrpts))

    # Identify throughput class boundaries.
    sorted_thrpts = sorted(train_thrpts)
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
    train_thrpts = np.array(cls_thrpts[:splitter], dtype='int32')

    # Statistics
    buckets = [0 for _ in range(num_cls)]
    for thrpt_cls in train_thrpts:
        buckets[thrpt_cls] += 1
    log.debug('Training throughput distributions')
    for idx, (boundary, bucket) in enumerate(zip(boundaries, buckets)):
        log.debug('\t%02d (<=%.2f): %d', idx, boundary, bucket)


    test_feats = np.array(features[splitter:])
    test_thrpts = np.array(cls_thrpts[splitter:], dtype='int32')
    return headers, train_feats, train_thrpts, test_feats, test_thrpts


def test_acc(pred_thrpts, test_thrpts):
    """Test the model accuracy."""

    log.debug('\nExpected\tThrpt-Pred\tThrpt-Error')

    thrpt_pred_range = (float('inf'), 0)

    # Confusion matrix for each class.
    error = 0
    TP = []
    FN = []
    FP = []

    for pred, real in zip(pred_thrpts, test_thrpts):
        pred = int(pred)
        while len(TP) <= max(real, pred):
            TP.append(0)
            FN.append(0)
            FP.append(0)

        if pred == real:
            TP[real] += 1
        else:
            error += 1
            FN[real] += 1
            FP[pred] += 1

        thrpt_pred_range = (min(thrpt_pred_range[0], pred), max(thrpt_pred_range[1], pred))
        log.debug('\t%d\t%d', real, pred)

    accuracy = 100.0 * (1.0 - (error / len(test_thrpts)))
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
    log.info('Accuracy: %.2f%%', accuracy)
    return accuracy


def train_model(n_class: int, data_file: str):
    """Train a cost model with provided data set.

    Parameters
    ----------
    model: str
        The model name.

    data_file: str
        Data file in CSV format.
    """

    headers, train_feats, train_thrpts, test_feats, test_thrpts = load_data(n_class, data_file)

    trained_model = train_xgb_classify(headers, train_feats, train_thrpts, n_class)
    pred = xgb_predict(trained_model, test_feats)
    test_acc(pred, test_thrpts)
    pickle.dump(trained_model, open("model.dat", "wb"))


def train_xgb_classify(headers: np.array, train_feats: np.ndarray,
                       train_res: np.ndarray, n_class: int):
    """Train an XGBoost model with multi-class objective."""
    params = {
        'max_depth': 30,
        'gamma': 0.001,
        'min_child_weight': 0,
        'eta': 0.3,
        'seed': 37,
        'objective': 'multi:softmax',
        'num_class': n_class,
        'n_gpus': 0,
        'tree_method': 'hist'
    }

    dtrain = xgb.DMatrix(data=train_feats,
                         label=train_res,
                         feature_names=headers)
    model = xgb.train(params, dtrain)
    return model


def xgb_predict(model, test_feats: np.ndarray):
    dtest = xgb.DMatrix(data=test_feats, feature_names=model.feature_names)
    return model.predict(dtest)


if __name__ == "__main__":
    CONFIGS = create_config()
    if CONFIGS.mode == 'train':
        print('Training %d classes using %s' % (CONFIGS.n_class, CONFIGS.path))
        train_model(CONFIGS.n_class, CONFIGS.path)
    else:
        raise RuntimeError('Unknown mode %s' % CONFIGS.mode)
