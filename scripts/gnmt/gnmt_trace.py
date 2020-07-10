# This script is modified from MLPerf:
#   https://github.com/mlperf/inference/tree/master/others/cloud/translation/gnmt/pytorch
#
# To run it, you have to first download pretrained PyTorch GNMT model:
#   wget https://zenodo.org/record/2581623/files/model_best.pth 
#
# Usage: python gnmt_trace.py --model model_best.pth

import argparse
from ast import literal_eval

import torch

from seq2seq import models


def parse_args():
    parser = argparse.ArgumentParser(description='GNMT Translate',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # data
    dataset = parser.add_argument_group('data setup')
    dataset.add_argument('-m', '--model', required=True,
                         help='model checkpoint file')

    # general setup
    general = parser.add_argument_group('general setup')

    batch_first_parser = general.add_mutually_exclusive_group(required=False)
    batch_first_parser.add_argument('--batch-first', dest='batch_first',
                                    action='store_true',
                                    help='uses (batch, seq, feature) data \
                                    format for RNNs')
    batch_first_parser.add_argument('--seq-first', dest='batch_first',
                                    action='store_false',
                                    help='uses (seq, batch, feature) data \
                                    format for RNNs')
    batch_first_parser.set_defaults(batch_first=True)

    return parser.parse_args()


def checkpoint_from_distributed(state_dict):
    ret = False
    for key, _ in state_dict.items():
        if key.find('module.') != -1:
            ret = True
            break
    return ret


def unwrap_distributed(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value

    return new_state_dict


def main():
    args = parse_args()

    torch.backends.cudnn.enabled = False
    checkpoint = torch.load(args.model, map_location={'cuda:0': 'cpu'})

    vocab_size = checkpoint['tokenizer'].vocab_size
    model_config = dict(vocab_size=vocab_size, math=checkpoint['config'].math,
                        **literal_eval(checkpoint['config'].model_config))
    model_config['batch_first'] = args.batch_first
    model = models.GNMT(**model_config)

    state_dict = checkpoint['state_dict']
    if checkpoint_from_distributed(state_dict):
        state_dict = unwrap_distributed(state_dict)

    model.load_state_dict(state_dict)

    model.type(torch.FloatTensor)
    model.eval()

    encode_shape = (128, 67)
    len_shape = (128,)
    decode_shape = (1280,)

    # FIXME: How to generate input data for tracing
    #input_encode = torch.randint(30000, encode_shape)
    #input_enc_len = 
    #input_decode = torch.randint(, decode_shape)

    #scripted_model = torch.jit.trace(model, [input_encode, input_enc_len, decode])


if __name__ == '__main__':
    main()
