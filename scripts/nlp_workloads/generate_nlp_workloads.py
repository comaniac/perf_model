import mxnet as mx
# Once numpy_nlp is merged as a branch. Rmove the line to gluonnlp
import numpy_nlp as nlp
from numpy_nlp.models import list_backbone_names, get_backbone
from numpy_nlp.utils.misc import count_parameters

mx.npx.set_np()
batch_size = 1
sequence_length = 32

for name in list_backbone_names():
    model_cls, cfg, tokenizer, local_params_path, others = get_backbone(model_name=name)
    net = model_cls.from_cfg(cfg)
    net.hybridize()
    inputs = mx.np.random.randint(0, 10, (batch_size, sequence_length))
    token_types = mx.np.random.randint(0, 2, (batch_size, sequence_length))
    valid_length = mx.np.random.randint(1, 10, (batch_size,))
    out = net(inputs, token_types, valid_length)
    sym = net._cached_graph[1]
    sym.save('{}.json'.format(name), remove_amp_cast=True)

