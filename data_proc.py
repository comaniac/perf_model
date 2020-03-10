"""Process config data."""
import argparse
import glob
import json
import os
from math import ceil
from typing import IO, Optional, Set

import numpy as np
import tqdm

import topi  # pylint: disable=unused-import
from tvm.autotvm.record import load_from_file
from tvm.autotvm.task import create
from tvm.autotvm.task.space import AnnotateEntity, OtherOptionEntity, SplitEntity


def create_config():
    """Create the config parser of this app."""
    # pylint: disable=redefined-outer-name
    parser = argparse.ArgumentParser(description='Cost Model')
    subparser = parser.add_subparsers(dest='mode',
                                      description='Execution mode')
    subparser.required = True
    featurize = subparser.add_parser('featurize',
                                     help='Process tuning log to features')
    featurize.add_argument('path', help='Log file path or folder')
    featurize.add_argument('--no-log-scale',
                           action='store_true',
                           help='Do not use log scale for features')
    featurize.add_argument('-o',
                           '--out',
                           default='feature',
                           help='Output directory to store features')

    classify = subparser.add_parser('classify',
                                    help='Classify throughputs to N classes')
    classify.add_argument('path', help='Feature file or path')
    classify.add_argument('--n-class',
                          default=31,
                          type=int,
                          choices=[0, 9, 17, 31],
                          help='Classify throughput to N classes')
    classify.add_argument('-o',
                          '--out',
                          default='class',
                          help='Output directory to store features')

    norm = subparser.add_parser('norm', help='Normalize throughputs')
    norm.add_argument('path', help='Feature file or path')
    norm.add_argument('-o',
                      '--out',
                      default='class',
                      help='Output directory to store features')

    log_out = subparser.add_parser('log', help='Log throughputs')
    log_out.add_argument('path', help='Feature file or path')
    log_out.add_argument('-o',
                         '--out',
                         default='class',
                         help='Output directory to store features')

    tocsv = subparser.add_parser('json2csv',
                                 help='Transform JSON feature to CSV format')
    tocsv.add_argument('path', help='JSON feature file or path')
    tocsv.add_argument('-o',
                       '--out',
                       default='csv',
                       help='Output directory to store features')

    return parser.parse_args()


def gen_key_str(inp):
    """Generate a string of target and task name"""
    return '{0}-{1}'.format(inp.task.name,
                            str(inp.target).replace(' ', '').replace('=', '-'))


def extract_feature(inp, log_scale=False):
    """Extract features.

    Parameters
    ----------
    inp: MeasureInput
        AutoTVM measure input.

    log_scale: bool
        Make log to every feature value.

    Returns
    -------
    features: Dict[str, Any]
        A map from feature name to value.
    """
    features = {}

    # Feature arguments
    in_idx = 0
    attr_idx = 0
    for arg in inp.task.args:
        if isinstance(arg, (list, tuple)) and arg[0] == 'TENSOR':
            for shape in arg[1]:
                features['in_{0}'.format(in_idx)] = shape
                in_idx += 1
        elif isinstance(arg, (list, tuple)):
            for elt in arg:
                features['attr_{0}'.format(attr_idx)] = elt
                attr_idx += 1
        else:
            features['attr_{0}'.format(attr_idx)] = str(arg)
            attr_idx += 1

    # Feature configs
    for k, v in inp.config._entity_map.items():
        if isinstance(v, SplitEntity):
            for idx, elt in enumerate(v.size):
                features['sp_{0}_{1}'.format(k, idx)] = elt
        elif isinstance(v, AnnotateEntity):
            features['an_{0}'.format(k)] = v.anns
        elif isinstance(v, OtherOptionEntity):
            features['ot_{0}'.format(k)] = v.val
        else:
            raise RuntimeError("Unsupported config entity: " + v)

    if log_scale:
        for key in features:
            if isinstance(features[key], str):
                continue
            val = features[key] if features[key] > 0 else 1e-5
            features[key] = np.around(np.log(val), 2).tolist()
    return features


def featurize(log_path: str, out_path: str, log_scale=True):
    """Parse tuning logs, extract features, and output features by target and ops to JSON files.

    Parameters
    ----------
    log_path: str
        Tuning log path (file or folder).

    out_path: str
        The folder to store parsed features.

    log_scale: bool
        Weather to use log scale for feature values.
    """

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    # Extract features for each log file.
    path = '{0}/*'.format(log_path) if os.path.isdir(log_path) else log_path
    for log_file in tqdm.tqdm(glob.glob(path)):
        model_key: Optional[str] = None
        out_file: Optional[IO] = None
        for inp, res in load_from_file(log_file):
            if model_key is None:
                model_key = gen_key_str(inp)
                out_file = open('{0}/{1}.json'.format(out_path, model_key),
                                'a')
            elif model_key != gen_key_str(inp):
                print('Key mismatch %s <> %s, skip %s' %
                      (model_key, gen_key_str(inp), str(inp)))
                continue

            features = extract_feature(inp, log_scale)

            # Compute GFLOP/s
            task = create(inp.task.name, inp.task.args, inp.target)
            if res.error_no == 0:
                features['thrpt'] = np.around(
                    task.flop / 1e9 / np.mean(res.costs), 2).tolist()
            else:
                features['thrpt'] = 0

            if out_file is None:
                break
            out_file.write(json.dumps(features))
            out_file.write('\n')

        if out_file is not None:
            out_file.close()


def classify(classify_scale: int, log_path: str, out_path: str):
    """Parse extracted features and classify throughputs to N classes.

    Parameters
    ----------
    classify_scale: int
        Classify throughputs to a number of classes.

    log_path: str
        Feature file path (file or folder).

    out_path: str
        The folder to store classified outputs.
    """
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    path = '{0}/*'.format(log_path) if os.path.isdir(log_path) else log_path
    for feat_file in tqdm.tqdm(glob.glob(path)):
        file_name = os.path.basename(feat_file)

        with open(feat_file, 'r') as filep:
            data = [json.loads(r) for r in filep]

        max_thrpt = max(data, key=lambda r: r['thrpt'])['thrpt']

        with open(os.path.join(out_path, file_name), 'w') as filep:
            for record in data:
                rank = 100 * (record['thrpt'] / max_thrpt)
                record['thrpt'] = ceil(rank / (100 / classify_scale))
                filep.write(json.dumps(record))
                filep.write('\n')


def norm(log_path: str, out_path: str):
    """Parse extracted features and normalize throughputs.

    Parameters
    ----------
    log_path: str
        Feature file path (file or folder).

    out_path: str
        The folder to store classified outputs.
    """
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    path = '{0}/*'.format(log_path) if os.path.isdir(log_path) else log_path
    for feat_file in tqdm.tqdm(glob.glob(path)):
        file_name = os.path.basename(feat_file)

        with open(feat_file, 'r') as filep:
            data = [json.loads(r) for r in filep]

        max_thrpt = max(data, key=lambda r: r['thrpt'])['thrpt']

        with open(os.path.join(out_path, file_name), 'w') as filep:
            for record in data:
                record['thrpt'] = record['thrpt'] / max_thrpt
                filep.write(json.dumps(record))
                filep.write('\n')


def log_thrpt(log_path: str, out_path: str):
    """Parse extracted features and log throughputs.

    Parameters
    ----------
    log_path: str
        Feature file path (file or folder).

    out_path: str
        The folder to store logged outputs.
    """
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    path = '{0}/*'.format(log_path) if os.path.isdir(log_path) else log_path
    for feat_file in tqdm.tqdm(glob.glob(path)):
        file_name = os.path.basename(feat_file)

        with open(feat_file, 'r') as filep:
            data = [json.loads(r) for r in filep]

        with open(os.path.join(out_path, file_name), 'w') as filep:
            for record in data:
                record['thrpt'] = np.log(
                    record['thrpt']) if record['thrpt'] > 0 else np.log(1e-5)
                filep.write(json.dumps(record))
                filep.write('\n')


def json2csv(json_path: str, out_path: str):
    """Parse feature files in JSON format to CSV format.

    Parameters
    ----------
    json_path: str
        JSON feature file path (file or folder).

    out_path: str
        The folder to CSV format feature outputs.
    """
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    path = '{0}/*'.format(json_path) if os.path.isdir(json_path) else json_path
    for json_file in tqdm.tqdm(glob.glob(path)):
        features: Set[str] = set()

        with open(json_file, 'r') as filep:
            data = [json.loads(r) for r in filep]

        # The first pass to get a super set of features.
        for record in data:
            features.update(record.keys())

        # Turn to a list and enforce thrpt (result) to be the last.
        features.remove('thrpt')
        feature_list = list(features) + ['thrpt']

        # The second pass fills out the data.
        file_name = os.path.basename(json_file).replace('.json', '.csv')
        with open(os.path.join(out_path, file_name), 'w') as filep:
            # Write titles
            filep.write(','.join(feature_list))
            filep.write('\n')

            # Write features and results
            for record in data:
                csv_record = []
                for feat in feature_list:
                    if feat not in record:
                        csv_record.append(str(None))
                    else:
                        csv_record.append(str(record[feat]))
                filep.write(','.join(csv_record))
                filep.write('\n')


if __name__ == "__main__":
    CONFIGS = create_config()
    if CONFIGS.mode == 'featurize':
        print('Featurizing %s' % CONFIGS.path)
        featurize(CONFIGS.path, CONFIGS.out, not CONFIGS.no_log_scale)
    elif CONFIGS.mode == 'classify':
        print('Classifying %s to %d classes' % (CONFIGS.path, CONFIGS.n_class))
        classify(CONFIGS.n_class, CONFIGS.path, CONFIGS.out)
    elif CONFIGS.mode == 'norm':
        print('Normalizing %s' % CONFIGS.path)
        norm(CONFIGS.path, CONFIGS.out)
    elif CONFIGS.mode == 'log':
        print('Making log to %s' % CONFIGS.path)
        log_thrpt(CONFIGS.path, CONFIGS.out)
    elif CONFIGS.mode == 'json2csv':
        print('Transofrm JSON features to CSV format')
        json2csv(CONFIGS.path, CONFIGS.out)
    else:
        raise RuntimeError('Unknown mode %s' % CONFIGS.mode)
