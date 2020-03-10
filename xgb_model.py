"""Train a XGBoost model."""
import pickle
from math import sqrt

import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error as mse


def create_config():
    """Create the config parser of this app."""
    parser = argparse.ArgumentParser(description='Cost Model')
    subparser = parser.add_subparsers(dest='mode',
                                      description='Execution mode')
    subparser.required = True

    train = subparser.add_parser('train', help='Train cost model')
    train.add_argument('path', help='Feature file path or folder')
    train.add_argument('--model',
                       default='classify',
                       choices=['regression', 'classify'],
                       help='The target model to be trained')

    return parser.parse_args()


def compute_classify_accuracy(pred: np.ndarray,
                              real: np.ndarray,
                              allow_error: int = 0) -> float:
    """Compute the model accuracy by comparing prediction and real results.

    Parameters
    ----------
    pred: np.ndarray
        The model pedicted results.

    real: np.ndarray
        Real results.

    allow_error: int
        Allowed error range (-n ~ n).

    Returns
    -------
    accuracy: float
        The accuracy in percentage.
    """
    if isinstance(pred[0], np.ndarray):
        pred = np.array([p.argmax() for p in pred])
    return 100.0 * sum([abs(p - r) <= allow_error
                        for p, r in zip(pred, real)]) / len(real)


def compute_regression_error(pred: np.ndarray, real: np.ndarray) -> float:
    """Compute the model error rate by comparing prediction and real results.

    Parameters
    ----------
    pred: np.ndarray
        The model pedicted results.

    real: np.ndarray
        Real results.

    Returns
    -------
    error: float
        The error rate in percentage.
    """
    return sqrt(mse(pred, real))


def train_model(model: str, data_file: str):
    """Train a cost model with provided data set.

    Parameters
    ----------
    model: str
        The model name.

    data_file: str
        Data file in CSV format.
    """

    np.random.seed(4)

    # Exclude the header and make numpy array.
    data = np.array([1])
    with open(data_file, 'r') as filep:
        headers = np.array(next(filep).replace('\n', '').split(',')[:-1])
        data = np.array([
            np.genfromtxt(np.array(l.replace('\n', '').split(',')))
            for l in filep
        ])
    np.random.shuffle(data)

    features = np.array(data.T[:-1].T)
    results = np.array(data.T[-1].T)

    classify_scale = int(max(results) + 1)

    # Split training and testing set.
    feat_split = int(len(features) * 0.8)
    res_split = int(len(results) * 0.8)
    train_feats = features[:feat_split]
    train_res = results[:res_split]
    test_feats = features[feat_split:]
    test_res = results[res_split:]

    print('%d for training and %d for testing' %
          (feat_split, len(features) - feat_split))

    trained_model = None
    if model == 'classify':
        trained_model = train_xgb_classify(headers, train_feats, train_res,
                                           classify_scale)
        pred = xgb_predict(trained_model, test_feats)
        print('Allowed 0 Error Accuracy %.2f%%' %
              compute_classify_accuracy(pred, test_res, allow_error=0))
        print('Allowed 1 Error Accuracy %.2f%%' %
              compute_classify_accuracy(pred, test_res, allow_error=1))
    elif model == 'regression':
        trained_model = train_xgb_regression(headers, train_feats, train_res)
        pred = xgb_predict(trained_model, test_feats)
        print('RMSE %.2f' % compute_regression_error(pred, test_res))
    else:
        raise RuntimeError('Unrecognized model: %s' % model)

    if trained_model is not None:
        pickle.dump(trained_model, open("model.dat", "wb"))


def train_xgb_regression(headers, train_feats, train_res):
    """Train an XGBoost model."""
    dtrain = xgb.DMatrix(data=train_feats,
                         label=train_res,
                         feature_names=headers)
    params = {
        'max_depth': 10,
        'gamma': 0.001,
        'min_child_weight': 0,
        'eta': 0.3,
        'silent': 0,
        'seed': 37,
        'n_gpus': 0,
        'objective': 'reg:squarederror'
    }
    model = xgb.train(params, dtrain)
    return model


def train_xgb_classify(headers: np.array, train_feats: np.ndarray,
                       train_res: np.ndarray, classify_scale: int):
    """Train an XGBoost model with multi-class objective."""
    params = {
        'max_depth': 10,
        'gamma': 0.001,
        'min_child_weight': 0,
        'eta': 0.3,
        'silent': 0,
        'seed': 37,
        'objective': 'multi:softmax',
        'num_class': classify_scale,
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
        print('Training %s using %s' % (CONFIGS.model, CONFIGS.path))
        train_model(CONFIGS.model, CONFIGS.path)
    else:
        raise RuntimeError('Unknown mode %s' % CONFIGS.mode)
