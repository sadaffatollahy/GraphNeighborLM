import random, os
import numpy as np
import torch
from config import parse_args_llama

def seed_everything(seed: int):
    """
    Set a fixed random seed for reproducibility across multiple libraries and environments.

    This function ensures that experiments in machine learning and data science are reproducible
    by setting the same random seed for various sources of randomness, including Python's `random` module,
    NumPy, and PyTorch (for both CPU and GPU). It also configures PyTorch's CuDNN backend to enforce
    deterministic behavior.

    Args:
        seed (int): The seed value to set for all random number generators.

    Libraries Affected:
        - `random`: Sets the random seed for Python's built-in random module.
        - `os`: Sets the `PYTHONHASHSEED` environment variable to ensure deterministic hashing in Python.
        - `numpy`: Sets the random seed for NumPy.
        - `torch`: Sets the seed for PyTorch's random number generation on both CPU and GPU.

    PyTorch-Specific Settings:
        - `torch.backends.cudnn.deterministic = True`:
            Forces PyTorch to use deterministic algorithms for CuDNN operations.
        - `torch.backends.cudnn.benchmark = True`:
            Enables dynamic algorithm optimization in CuDNN for faster execution in some cases.

    Example:
        >>> seed_everything(42)
        >>> random.randint(0, 10)  # Always generates the same number
        >>> np.random.rand(3)  # Produces the same array for a fixed seed
        >>> torch.randn(3, 3)  # Produces the same tensor for a fixed seed

    Notes:
        - While `torch.backends.cudnn.deterministic = True` ensures reproducibility, it may slow down computations.
        - `torch.backends.cudnn.benchmark = True` may introduce slight variability in speed for different input sizes.

    """
    random.seed(seed) #for random library
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed) #for numpy
    torch.manual_seed(seed) #for cpu
    torch.cuda.manual_seed(seed)#for gpu
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
     


from torch_geometric.data import Batch


def collate_fn(original_batch):
    """

    Custom collate function for batching data in a PyTorch DataLoader.

    This function processes a batch of individual samples (dictionaries) from the dataset and combines them into a
    single batch dictionary. If the data contains graphs (PyTorch Geometric `Data` objects), they are batched
    into a single graph using `Batch.from_data_list`.

    Args:
        original_batch (list of dict): A batch of samples from the dataset. Each sample is a dictionary where keys
                                        represent feature names (e.g., "question", "labels", "desc", "graph"), and values
                                        are the corresponding data for each sample.

    Returns:
        dict: A dictionary containing batched data for all keys:
              - For non-graph data, values are lists of the corresponding data from each sample.
              - For graph data (key: "graph"), a single `Batch` object from PyTorch Geometric is returned.

              Example output:
              {
                  "input_ids": [[101, 200, 300], [102, 201, 301]],
                  "labels": [1, 0],
                  "graph": Batch(...)  # Batched PyTorch Geometric graph object
              }

    Notes:
        - If the dataset contains a "graph" key, it is expected to be a PyTorch Geometric `Data` object.
        - Non-graph features are combined into lists for easier downstream processing.

    """
    batch = {}
    for k in original_batch[0].keys():
        batch[k] = [d[k] for d in original_batch]
    if 'graph' in batch:
        batch['graph'] = Batch.from_data_list(batch['graph'])
    return batch


import math


def adjust_learning_rate(param_group, LR, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    min_lr = 5e-6
    if epoch < args.warmup_epochs: #In epoch 1, the learning rate starts at 0 and gradually increases to LR over the warmup_epochs.
        lr = LR * epoch / args.warmup_epochs
    else:
      #After warmup_epochs, the learning rate decreases to min_lr following a half-cycle cosine schedule.
        lr = min_lr + (LR - min_lr) * 0.5 * (1.0 + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.num_epochs - args.warmup_epochs)))
    param_group["lr"] = lr
    return lr



import os
import torch


def print_trainable_params(model):
    trainable_params = 0
    all_param = 0

    for _, param in model.named_parameters():
        num_params = param.numel()

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param


def _save_checkpoint(model, optimizer, cur_epoch, args, is_best=False):
    """
    Save the checkpoint at the current epoch.
    """
    os.makedirs(f'{args.output_dir}/{args.dataset}', exist_ok=True)

    param_grad_dic = {
        k: v.requires_grad for (k, v) in model.named_parameters()
    }
    state_dict = model.state_dict() #learnable parameter
    for k in list(state_dict.keys()):
        if k in param_grad_dic.keys() and not param_grad_dic[k]:
            # delete parameters that do not require gradient
            del state_dict[k]
    save_obj = {
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "config": args,
        "epoch": cur_epoch,
    }
    path = f'{args.output_dir}/{args.dataset}/model_name_{args.model_name}_llm_model_name_{args.llm_model_name}_llm_frozen_{args.llm_frozen}_max_txt_len_{args.max_txt_len}_max_new_tokens_{args.max_new_tokens}_gnn_model_name_{args.gnn_model_name}_patience_{args.patience}_num_epochs_{args.num_epochs}_seed{args.seed}_checkpoint_{"best" if is_best else cur_epoch}.pth'
    print("Saving checkpoint at epoch {} to {}.".format(cur_epoch, path))
    torch.save(save_obj, path)


def _reload_best_model(model, args):
    """
    Load the best checkpoint for evaluation.
    """
    checkpoint_path = f'{args.output_dir}/{args.dataset}/model_name_{args.model_name}_llm_model_name_{args.llm_model_name}_llm_frozen_{args.llm_frozen}_max_txt_len_{args.max_txt_len}_max_new_tokens_{args.max_new_tokens}_gnn_model_name_{args.gnn_model_name}_patience_{args.patience}_num_epochs_{args.num_epochs}_seed{args.seed}_checkpoint_best.pth'

    print("Loading checkpoint from {}.".format(checkpoint_path))

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)

    return model


def _reload_model(model, checkpoint_path):
    """
    Load the best checkpoint for evaluation.
    """

    print("Loading checkpoint from {}.".format(checkpoint_path))

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)

    return model




import json
import pandas as pd
import re
import string
def get_accuracy_expla_graphs(path):
    df = pd.read_json(path, lines=True)
    # compute accuracy
    correct = 0
    for pred, label in zip(df["pred"], df["label"]):
        matches = re.findall(r"support|Support|Counter|counter", pred.strip())
        if len(matches) > 0 and matches[0].lower() == label:
            correct += 1

    return correct / len(df)



eval_funcs = {
    "expla_graphs": get_accuracy_expla_graphs,
}




from transformers.utils import logging
import os
import wandb
import gc
from tqdm import tqdm
import torch
import json
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from models.__init__ import load_model, llama_model_path
from dataset.__init__ import load_dataset
# from src.utils.evaluate import eval_funcs
# from src.config import parse_args_llama
# from src.utils.ckpt import _save_checkpoint, _reload_best_model
# from src.utils.collate import collate_fn
# from src.utils.seed import seed_everything
# from src.utils.lr_schedule import adjust_learning_rate
logging.set_verbosity_info()  


def main(args):

    # Step 1: Set up wandb
    seed = args.seed
    print(f"Initializing training for project: {args.project}")
    
    

    seed_everything(seed=args.seed)
    print(args)

    dataset = load_dataset[args.dataset]() #load Explanation_graph
    idx_split = dataset.get_idx_split() #train-val-test index

    # Step 2: Build Node Classification Dataset
    train_dataset = [dataset[i] for i in idx_split['train']] #contains label, desc, question, graph...
    val_dataset = [dataset[i] for i in idx_split['val']]
    test_dataset = [dataset[i] for i in idx_split['test']]

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, pin_memory=True, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, drop_last=False, pin_memory=True, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, drop_last=False, pin_memory=True, shuffle=False, collate_fn=collate_fn)

    # Step 3: Build Model
    args.llm_model_path = llama_model_path[args.llm_model_name] #edumunozsala/llama-2-7b-int4-python-code-20k model

    model = load_model[args.model_name](graph_type=dataset.graph_type, args=args, init_prompt=dataset.prompt)#load graph_llm

    # Step 4 Set Optimizer
    params = [p for _, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{'params': params, 'lr': args.lr, 'weight_decay': args.wd}, ],
        betas=(0.9, 0.95)
    )


    trainable_params, all_param = model.print_trainable_params()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

    # Step 5. Training
    num_training_steps = args.num_epochs * len(train_loader)
    progress_bar = tqdm(range(num_training_steps))
    best_val_loss = float('inf')

    for epoch in range(args.num_epochs):

        model.train()
        epoch_loss, accum_loss = 0., 0.

        for step, batch in enumerate(train_loader):

            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()

            clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1) # Limits the gradient norm to a maximum of 0.1 to prevent exploding gradients.

            if (step + 1) % args.grad_steps == 0: #This condition ensures that the learning rate adjustment only occurs after a specific number of gradient accumulation steps
                adjust_learning_rate(optimizer.param_groups[0], args.lr, step / len(train_loader) + epoch, args) # step / len(train_loader) + epoch : The learning rate is adjusted smoothly not only at the start of each epoch but continuously during the epoch as well.

            optimizer.step() #update weight
            epoch_loss, accum_loss = epoch_loss + loss.item(), accum_loss + loss.item()


            #Log to wandb
            if (step + 1) % args.grad_steps == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(f"Learning Rate: {lr}")
                print(f"Accumulated Loss: {accum_loss / args.grad_steps}")
                accum_loss = 0.


            progress_bar.update(1)

        print(f"Epoch: {epoch}|{args.num_epochs}: Train Loss (Epoch Mean): {epoch_loss / len(train_loader)}")
        #wandb.log({'Train Loss (Epoch Mean)': epoch_loss / len(train_loader)})


#ُStep6 : evaluation on validation
        val_loss = 0.
        eval_output = []
        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                loss = model(batch)
                val_loss += loss.item()
            val_loss = val_loss/len(val_loader)
            print(f"Epoch: {epoch}|{args.num_epochs}: Val Loss: {val_loss}")
            #wandb.log({'Val Loss': val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(model, optimizer, epoch, args, is_best=True)
            best_epoch = epoch

        print(f'Epoch {epoch} Val Loss {val_loss} Best Val Loss {best_val_loss} Best Epoch {best_epoch}')

        if epoch - best_epoch >= args.patience:
            print(f'Early stop at epoch {epoch}')
            break

    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()

    # Step 7. Evaluating on test with infernece part
    os.makedirs(f'{args.output_dir}/{args.dataset}', exist_ok=True)
    path = f'{args.output_dir}/{args.dataset}/model_name_{args.model_name}_llm_model_name_{args.llm_model_name}_llm_frozen_{args.llm_frozen}_max_txt_len_{args.max_txt_len}_max_new_tokens_{args.max_new_tokens}_gnn_model_name_{args.gnn_model_name}_patience_{args.patience}_num_epochs_{args.num_epochs}_seed{seed}.csv'
    print(f'path: {path}')

    model = _reload_best_model(model, args)
    model.eval()
    progress_bar_test = tqdm(range(len(test_loader)))
    with open(path, "w") as f:
        for step, batch in enumerate(test_loader):
            with torch.no_grad():
                output = model.inference(batch)
                df = pd.DataFrame(output)
                for _, row in df.iterrows():
                    f.write(json.dumps(dict(row)) + "\n")
            progress_bar_test.update(1)

    # Step 6. Post-processing & compute metrics
    acc = eval_funcs[args.dataset](path)
    print(f'Test Acc {acc}')
    #wandb.log({'Test Acc': acc})



if __name__ == "__main__":

    args = parse_args_llama()


    main(args)
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    gc.collect()