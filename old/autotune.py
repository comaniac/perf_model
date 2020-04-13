"""Train a BERT model using AutoGluon."""
import sys

import autogluon as ag

from bert_model import train_bert


@ag.args(
    data_file=sys.argv[1],
    epochs=ag.Choice(40, 80),
    lr=ag.Real(1e-5, 1e-4, log=True),
    batch_size=ag.Choice(256, 512, 1024),
    wd=ag.Real(1e-3, 10, log=True),
    num_heads=ag.Choice(8, 16, 32, 64),
    num_pred_hiddens=ag.Choice(256, 512, 1024, 2048),
    ffn_num_hiddens=ag.Choice(512, 1024, 2048),
    num_layers=ag.Choice(6, 8, 12),
    dropout=ag.Real(1e-1, 8e-1, log=True),
)
def run_training(args, reporter):
    """Launch training process."""
    args.num_hiddens = args.num_heads
    return train_bert(args, reporter)


def run():
    """Run tuning."""
    scheduler = ag.scheduler.FIFOScheduler(run_training,
                                           resource={'num_cpus': 1, 'num_gpus': 0},
                                           num_trials=1,
                                           time_attr='epoch',
                                           reward_attr="accuracy")
    scheduler.run()
    scheduler.join_jobs()

if __name__ == "__main__":
    run()
