import numpy as np
import mxnet as mx
import tvm
from tvm import relay
from tvm import autotvm
import tvm.contrib.graph_runtime as runtime

from hybrid_bert import get_hybrid_model
from hybrid_bert import HybridBERTClassifier, HybridBERTRegression, HybridBERTForQA

seq_length = 128
bert, _ = get_hybrid_model(
        name="bert_12_768_12",
        dataset_name="book_corpus_wiki_en_uncased",
        pretrained=False,
        use_pooler=True,
        use_decoder=False,
        use_classifier=False,
        seq_length=seq_length)
net = HybridBERTClassifier(bert, num_classes=2, dropout=0.1)
net.initialize()

inputs = np.random.uniform(size=(1, seq_length)).astype('float32')
token_types = np.random.uniform(size=(1, seq_length)).astype('float32')
valid_length = np.asarray([seq_length]).astype('float32')

mx_ctx = mx.cpu()
inputs_nd = mx.nd.array(inputs, ctx=mx_ctx)
token_types_nd = mx.nd.array(token_types, ctx=mx_ctx)
valid_length_nd = mx.nd.array(valid_length, ctx=mx_ctx)
net(inputs_nd, token_types_nd, valid_length_nd)

shape_dict = {
    'data0': (1, seq_length),
    'data1': (1, seq_length),
    'data2': (1,)
}
mod, params = relay.frontend.from_mxnet(net, shape_dict)

target = "llvm -mcpu=skylake-avx512 -libs=cblas"  # c5
# target = "cuda -libs=cublas,cudnn" # p3/g4
# target = "llvm -device=arm_cpu -target=aarch64-linux-gnu -mattr=+neon" # a1

tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)
for idx, task in enumerate(tasks):
    print(idx)
    print(task)
print(len(tasks))
assert False

with tvm.transform.PassContext(opt_level=3):
    graph, lib, params = relay.build(mod["main"], target, params=params)

ctx = tvm.context(target)
ex = runtime.create(graph, lib, ctx)
ex.set_input(data0=inputs, data1=token_types, data2=valid_length, **params)
ex.run()

print("Benchmarking...")
ftimer = ex.module.time_evaluator("run", ctx, min_repeat_ms=2000)
prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
      (np.mean(prof_res), np.std(prof_res)))
