"""Process config data."""
import argparse
import glob
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Set

import numpy as np
import tqdm
from filelock import FileLock

import topi  # pylint: disable=unused-import
from tvm.autotvm.record import load_from_file
from tvm.autotvm.task import create
from tvm.autotvm.task.space import (AnnotateEntity, OtherOptionEntity,
                                    ReorderEntity, SplitEntity)


def create_config():
    """Create the config parser of this app."""
    # pylint: disable=redefined-outer-name
    parser = argparse.ArgumentParser(description='Cost Model')
    subparser = parser.add_subparsers(dest='mode', description='Execution mode')
    subparser.required = True
    featurize = subparser.add_parser('featurize', help='Process tuning log to features')
    featurize.add_argument('path', help='Log file path or folder')
    featurize.add_argument('-o',
                           '--out',
                           default='feature',
                           help='Output directory to store features')

    tocsv = subparser.add_parser('json2csv', help='Transform JSON feature to CSV format')
    tocsv.add_argument('path', help='JSON feature file or path')
    tocsv.add_argument('--std', action='store_true', help='Standardize feature values')
    tocsv.add_argument('-o', '--out', default='csv', help='Output directory to store features')

    return parser.parse_args()


def gen_key_str(inp):
    """Generate a string of target and task"""
    return '{0}-{1}'.format(str(inp.task), str(inp.target).replace(' ', '').replace('=', '-'))

def gen_file_str(inp):
    """Generate a string as the output file name"""
    return inp.task.name


def extract_feature(inp):
    """Extract features.

    Parameters
    ----------
    inp: MeasureInput
        AutoTVM measure input.

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
    for key, val in inp.config._entity_map.items():  # pylint: disable=protected-access
        if isinstance(val, SplitEntity):
            for idx, elt in enumerate(val.size):
                features['sp_{0}_{1}'.format(key, idx)] = elt
        elif isinstance(val, AnnotateEntity):
            if isinstance(val.anns, (int, float)):
                pval = str(val.anns)
            elif isinstance(val.anns, str):
                pval = val.anns
            elif isinstance(val.anns, list):
                pval = ';'.join([str(e) for e in val.anns])
            else:
                raise RuntimeError('Unrecognized annotate type: %s' % type(val.anns))
            features['an_{0}'.format(key)] = pval
        elif isinstance(val, OtherOptionEntity):
            features['ot_{0}'.format(key)] = val.val
        elif isinstance(val, ReorderEntity):
            features['re_{0}'.format(key)] = ';'.join([str(a) for a in val.perm])
        else:
            raise RuntimeError("Unsupported config entity: " + val)
    return features


def extract_feature_from_file(log_file: str, out_path: str):
    """Parse a log file and extract featues to the output file"""
    data: Dict[str, List[str]] = {}

    cnt = 0
    for inp, res in load_from_file(log_file):
        cnt += 1
        key = (gen_key_str(inp), gen_file_str(inp))
        if key not in data:
            data[key] = []

        try:
            features = extract_feature(inp)
        except Exception as err:
            return str(err)

        # Compute GFLOP/s
        task = create(inp.task.name, inp.task.args, inp.target)
        if res.error_no == 0:
            features['thrpt'] = np.around(task.flop / 1e9 / np.mean(res.costs), 2).tolist()
        else:
            features['thrpt'] = 0

        data[key].append(json.dumps(features))

    for (_, file_key), feats in data.items():
        out_file = '{0}/{1}.json'.format(out_path, file_key)
        lock_file = '{0}.lock'.format(out_file)
        with FileLock(lock_file):
            with open(out_file, 'a') as filep:
                for record in feats:
                    filep.write(record)
                    filep.write('\n')
    return None


def featurize(log_path: str, out_path: str):
    """Parse tuning logs, extract features, and output features by target and ops to JSON files.

    Parameters
    ----------
    log_path: str
        Tuning log path (file or folder).

    out_path: str
        The folder to store parsed features.
    """

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    # Extract features for each log file.
    with ProcessPoolExecutor(max_workers=8) as pool:
        path = '{0}/*'.format(log_path) if os.path.isdir(log_path) else log_path
        file_list = glob.glob(path)
        for start in tqdm.tqdm(range(0, len(file_list), 8)):
            futures = [
                pool.submit(extract_feature_from_file, log_file, out_path)
                for log_file in file_list[start:min(start + 8, len(file_list))]
            ]
            for ret in as_completed(futures):
                msg = ret.result()
                if msg:
                    print(msg)

    # Remove lock files
    shutil.rmtree('{0}/*.lock'.format(out_path), ignore_errors=True)


def json2csv(json_path: str, std: bool, out_path: str):
    """Parse feature files in JSON format to CSV format.

    Parameters
    ----------
    json_path: str
        JSON feature file path (file or folder).

    std: bool
        Whether to standardize features.

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

        # The second pass fills out the data to a data frame.
        dataframe = []
        for record in data:
            csv_record = []
            for feat in feature_list:
                if feat not in record:
                    csv_record.append(str(None))
                else:
                    csv_record.append(str(record[feat]))
            dataframe.append(csv_record)

        if std:
            # Metadata for mean, std, and category mapping.
            meta_file = os.path.join(out_path,
                                     os.path.basename(json_file).replace('.json', '.meta'))
            with open(meta_file, 'w') as filep:
                std_data = []

                # Let each feature be a row.
                tran_data = np.array(dataframe).T
                # pylint: disable=unsubscriptable-object
                for feat, row in zip(feature_list[:-1], tran_data[:-1]):
                    val_type = 'numeric'
                    try:  # Standardize floating values.
                        float_row = row.astype('float')
                        meta = [float_row.mean(), float_row.std()]
                        meta[1] = 1 if meta[1] == 0 else meta[1]  # Workaround for std=0
                        std_data.append((float_row - meta[0]) / meta[1])
                    except ValueError:  # String to index transformation.
                        meta = np.unique(row)
                        cate_map = {c: i for i, c in enumerate(meta)}
                        std_data.append([cate_map[c] for c in row])
                        val_type = 'category'

                    filep.write('{},'.format(feat)) # Feature name
                    filep.write('{},'.format(val_type)) # Feature value type (numeric or category)
                    filep.write(','.join([str(e) for e in meta]))
                    filep.write('\n')

                # pylint: disable=unsubscriptable-object
                std_data.append(tran_data[-1].astype('float32'))
            dataframe = np.array(std_data).T

        # Write to file
        file_name = os.path.basename(json_file).replace('.json', '.csv')
        with open(os.path.join(out_path, file_name), 'w') as filep:
            # Write titles
            filep.write(','.join(feature_list))
            filep.write('\n')

            # Write features and results
            for record in dataframe:
                filep.write(','.join([str(r) for r in record]))
                filep.write('\n')


if __name__ == "__main__":
    CONFIGS = create_config()
    if CONFIGS.mode == 'featurize':
        print('Featurizing %s' % CONFIGS.path)
        featurize(CONFIGS.path, CONFIGS.out)
    elif CONFIGS.mode == 'json2csv':
        print('Transofrm JSON features to CSV format')
        json2csv(CONFIGS.path, CONFIGS.std, CONFIGS.out)
    else:
        raise RuntimeError('Unknown mode %s' % CONFIGS.mode)
