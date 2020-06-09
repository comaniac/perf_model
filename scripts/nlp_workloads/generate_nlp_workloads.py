import json
import mxnet as mx
# Once numpy_nlp is merged as a branch. Rmove the line to gluonnlp
from numpy_nlp.models import list_backbone_names, get_backbone
from numpy_nlp.utils.misc import count_parameters

mx.npx.set_np()
batch_size = 1
sequence_length = 32

for name in ['google_en_uncased_bert_base',
             'google_en_uncased_bert_large',
             'google_albert_base_v2',
             'google_albert_large_v2',
             'google_albert_xlarge_v2',
             'google_albert_xxlarge_v2',
             'google_electra_small',
             'google_electra_base',
             'google_electra_large']:
    model_cls, cfg, tokenizer, local_params_path, others = get_backbone(model_name=name)
    net = model_cls.from_cfg(cfg)
    net.initialize()
    net.hybridize()
    print(net)
    inputs = mx.np.random.randint(0, 10, (batch_size, sequence_length))
    token_types = mx.np.random.randint(0, 2, (batch_size, sequence_length))
    valid_length = mx.np.random.randint(1, 10, (batch_size,))
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
