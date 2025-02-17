import importlib
import os
import platform
import subprocess
import functools


import torch
import transformers

def get_cls_by_name(name: str) -> type:
    """Get class by its name and module path.

    Args:
        name (str): e.g., transfomers:T5ForConditionalGeneration, modeling_t5:my_class

    Returns:
        type: found class for `name`
    """
    module_name, cls_name = name.split(':')
    return getattr(importlib.import_module(module_name), cls_name)


def get_git_hash_commit() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

def collect_run_configuration(args, env_vars=['CUDA_VISIBLE_DEVICES']):
    args_dict = dict(vars(args))
    args_dict['ENV'] = {}
    for env_var in env_vars:
        args_dict['ENV'][env_var] = os.environ.get(env_var, '')
    # args_dict['HVD_INIT'] = hvd.is_initialized()
    # if hvd.is_initialized():
    #     args_dict['HVD_SIZE'] = hvd.size()
    args_dict['MACHINE'] = platform.node()
    args_dict['COMMIT'] = get_git_hash_commit()
    return args_dict
