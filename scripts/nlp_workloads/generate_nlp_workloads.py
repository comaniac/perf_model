import json
import mxnet as mx
from gluonnlp.models import list_backbone_names, get_backbone
from gluonnlp.utils.misc import count_parameters

mx.npx.set_np()
batch_size = 1
sequence_length = 32
all_possible_ops = []
for name in list_backbone_names():
    model_cls, cfg, tokenizer, local_params_path, others = get_backbone(model_name=name)
    net = model_cls.from_cfg(cfg)
    net.initialize()
    net.hybridize()
    print('Save the architecture of {} to {}.json'.format(name, name))
    inputs = mx.np.random.randint(0, 10, (batch_size, sequence_length))
    token_types = mx.np.random.randint(0, 2, (batch_size, sequence_length))
    valid_length = mx.np.random.randint(1, 10, (batch_size,))
    if 'roberta' in name:
        out = net(inputs, valid_length)
    else:
        out = net(inputs, token_types, valid_length)
    sym = net._cached_graph[1]
    sym.save('{}.json'.format(name), remove_amp_cast=True)
    all_ops = set()
    with open('{}.json'.format(name), 'r') as f:
        sym_info = json.load(f)
        for ele in sym_info['nodes']:
            all_ops.add(ele['op'])
    with open('{}_all_ops.json'.format(name), 'w') as f:
        json.dump(list(all_ops), f)
    all_possible_ops.extend(list(all_ops))

with open('all_possible_ops.json', 'w') as f:
    json.dump(list(set(all_possible_ops)), f)

