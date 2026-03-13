"""
Evaluate paper's pretrained 6pt model on their 6pt test set.
Adapted from spinorhelicity/main.py with npt_list=[6] and 6pt model/data paths.
"""

import os
import sys
import types
import numpy as np
import torch
import json

# Mock apex module (NVIDIA apex not needed for eval-only mode)
apex_mock = types.ModuleType('apex')
apex_mock.parallel = types.ModuleType('apex.parallel')
apex_mock.amp = types.ModuleType('apex.amp')
sys.modules['apex'] = apex_mock
sys.modules['apex.parallel'] = apex_mock.parallel
sys.modules['apex.amp'] = apex_mock.amp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, 'spinorhelicity'))

from environment.utils import initialize_exp, AttrDict
from environment import build_env
from add_ons.slurm import init_signal_handler, init_distributed_mode
from model import build_modules, check_model_params
import environment
from training.trainer import Trainer
from training.evaluator import Evaluator

np.seterr(all='raise')


def main(params):
    init_distributed_mode(params)
    logger = initialize_exp(params)
    init_signal_handler()
    environment.utils.CUDA = not params.cpu

    env = build_env(params)
    modules = build_modules(env, params)
    trainer = Trainer(modules, env, params)
    evaluator = Evaluator(trainer)

    torch.random.manual_seed(42)
    scores = evaluator.run_all_evals()
    for k, v in scores.items():
        logger.info("%s -> %.6f" % (k, v))
    logger.info("__log__:%s" % json.dumps(scores))


if __name__ == '__main__':

    parameters = AttrDict({

        # Name
        'exp_name': 'eval_paper_6pt',
        'dump_path': os.path.join(REPO_ROOT, 'results', 'paper_model_eval'),
        'exp_id': 'beam5_6pt',
        'save_periodic': 0,
        'tasks': 'spin_hel',

        # environment parameters
        'env_name': 'char_env',
        'npt_list': [6],
        'max_scale': 2,
        'max_terms': 3,
        'max_scrambles': 3,
        'min_scrambles': 1,
        'save_info_scr': False,
        'save_info_scaling': False,
        'int_base': 10,
        'numeral_decomp': True,
        'max_len': 1000,
        'l_scale': 0.75,
        'numerator_only': True,
        'reduced_voc': True,
        'all_momenta': False,

        # model parameters
        'emb_dim': 512,
        'n_enc_layers': 3,
        'n_dec_layers': 3,
        'n_heads': 8,
        'dropout': 0,
        'n_max_positions': 2560,
        'attention_dropout': 0,
        'sinusoidal_embeddings': False,
        'share_inout_emb': True,
        'positional_encoding': True,
        'reload_model': os.path.join(REPO_ROOT, 'spinorhelicity', 'n6_simplifier.pth'),

        # Trainer param
        'export_data': False,
        'reload_data': f"spin_hel,{os.path.join(REPO_ROOT, 'spinorhelicity', 'data', 'n6_i1_3', 'data.prefix.counts.train_1000')},{os.path.join(REPO_ROOT, 'spinorhelicity', 'data', 'n6_i1_3', 'data.prefix.counts.valid_1000')},{os.path.join(REPO_ROOT, 'spinorhelicity', 'data', 'n6_i1_3', 'data.prefix.counts.test_1000')}",
        'reload_size': '',
        'epoch_size': 1000,
        'max_epoch': 1,
        'amp': -1,
        'fp16': False,
        'accumulate_gradients': 1,
        'optimizer': "adam,lr=0.0001",
        'clip_grad_norm': 5,
        'stopping_criterion': '',
        'validation_metrics': '',
        'reload_checkpoint': '',
        'env_base_seed': -1,
        'batch_size': 32,

        # Evaluation
        'eval_only': True,
        'test_file': True,
        'valid_file': False,
        'numerical_check': 2,
        'eval_verbose': 1,
        'eval_verbose_print': False,
        'beam_eval': True,
        'beam_size': 5,
        'beam_length_penalty': 1,
        'beam_early_stopping': True,
        'nucleus_sampling': False,
        'nucleus_p': 0.95,
        'temperature': 1.5,
        'scaling_eval': False,

        # SLURM/GPU param
        'cpu': False,
        'local_rank': -1,
        'master_port': -1,
        'num_workers': 1,
        'debug_slurm': False,
        'lib_path': '',
        'mma_path': None,
    })

    check_model_params(parameters)
    main(parameters)
