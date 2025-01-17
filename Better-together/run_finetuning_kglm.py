import json
import logging
import os
import shutil
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
import torch
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler
from huggingface_hub import hf_hub_download
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import Dataset


from lm_experiments_tools.trainer import Trainer, TrainerArgs


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    logger.info(f"Using device: {torch.cuda.get_device_name(0)}")
    logger.info(f"Total available GPUs: {torch.cuda.device_count()}")
    torch.cuda.set_device(0)  #
else:
    logger.info("CUDA device not available, falling back to CPU")

torch.set_num_threads(4)

logger.info(f"Final device: {device}")

import transformers  # noqa: E402
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser  # noqa: E402
from lm_experiments_tools.utils import collect_run_configuration, get_cls_by_name # noqa: E402


train_df = pd.read_csv("/home/ahmadi/sadaf/GraphNeighborLM/Better-together/data-preparation/datasets/wikidata5m/verbalized_train.csv")
valid_df = pd.read_csv("/home/ahmadi/sadaf/GraphNeighborLM/Better-together/data-preparation/datasets/wikidata5m/verbalized_valid.csv")

parser = HfArgumentParser(TrainerArgs)
parser.add_argument('--task_name', type=str, help='Scrolls task name: "gov_report", "summ_screen_fd", "qmsum", '
                                                  '"narrative_qa", "qasper", "quality", "contract_nli"')
parser.add_argument('--validate_only', action='store_true', default=False,
                    help='Skip training and run only validation. (default: False)')
parser.add_argument('--working_dir', type=str, default='.',
                    help='working dir, should be a dir with t5-experiments repo (default: .)')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--show_valid_examples', type=int, default=2,
                    help='how many valid examples to show during training (default: 0)')

parser.add_argument('--input_seq_len', type=int, default=128, help='input sequnce length (default: 128).')
parser.add_argument('--target_seq_len', type=int, default=16, help='target sequnce length, should be set to '
                                                                   'max(len(target))+1 for EOS (default: 16).')
parser.add_argument('--data_n_workers', type=int, default=2, help='number of dataloader workers (default: 2)')

parser.add_argument('--input_prefix', type=str, default='', help='add task prefix to an input string (default: "")')
parser.add_argument('--drop_neighborhood', action='store_true', default=False, 
                    help='not to include neighborhood in model input')
parser.add_argument('--index_path', default=None, type=str, 
                    help='path to index for hits metric')

parser.add_argument('--inference_entities_path', default=None, type=str, 
                    help='path to names of verbalized entities from inference graph')
# model args
parser.add_argument('--from_pretrained', type=str, help='model name in HF Model Hub (default: "t5-small")')
## 
parser.add_argument('--cpt_path', type=str, help='path of checkpoint folder')

parser.add_argument('--model_cfg', type=str, help='path to model configuration file (default: "")')
parser.add_argument('--model_cls', type=str, default='transformers:BertForPreTraining',
                    help='model class name to use (default: transformers:BertForPreTraining)')
parser.add_argument('--model_type', type=str, default='encoder-decoder',
                    help='model type, encoder, encoder-decoder, decoder, affects preprocessing '
                         '(default: encoder-decoder)')

# tokenizer
# todo: add wordpiece tokenizers support?
parser.add_argument('--tokenizer', type=str, default=None, help='path or name of pre-trained HF Tokenizer')

# optimizer args
parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer name: AdamW, Adafactor. (default: AdamW)')
parser.add_argument('--weight_decay', type=float, default=0.0, help='optimizer weight decay (default: 0.0)')
parser.add_argument('--scale_parameter', action='store_true', default=False,
                    help='Adafactor scale_parameter (default: False)')
parser.add_argument('--relative_step', action='store_true', default=False,
                    help='Adafactor relative_step (default: False)')
parser.add_argument('--warmup_init', action='store_true', default=False,
                    help='Adafactor warmup_init (default: False)')

class KGLMDataset(Dataset):
    def __init__(self, df, neighborhood=True):
        self.df = df
        self.neighborhood = neighborhood
        self.length = len(df)

    def __getitem__(self, idx):

        item = {}
        row = self.df.iloc[idx]

        if self.neighborhood:
            item["input"] = row['verbalization']
        else:
            verbalization = row['verbalization']
            inp = '[SEP]'.join(verbalization.split('[SEP]')[:2])
            item["input"] = inp

        item["outputs"] = row['verbalized_tail']
        return item

    def __len__(self):

        return self.length



if __name__ == '__main__':
    args = parser.parse_args()
    logger.info('Running in environment.')
    logger.info('Using a single GPU setup.')
    logger.info(f'FP16 training is set to: {args.fp16}') #default = false

    if args.model_path is None: # path where to save model, default: None
      logger.warning('model_path is not set: config, logs and checkpoints will not be saved.')

    # create model path and save configuration
    if args.model_path is not None:
        model_path = Path(args.model_path)

        if not model_path.exists():
            Path(model_path).mkdir(parents=True)
        args_dict = collect_run_configuration(args)
        # todo: if model path exists and there is config file, write new config file aside
        json.dump(args_dict, open(model_path/'config.json', 'w'), indent=4)

 #output is like below:
#         {
#     'batch_size': 32,
#     'learning_rate': 0.001,
#     'model': 'T5',
#     'ENV': {
#         'CUDA_VISIBLE_DEVICES': '0,1'
#     },
#     'MACHINE': 'colab-instance',
#     'COMMIT': '3fa4b7c2c8f67a9d8e45be68dca1e24ff8b524d1'
# }
    logger.info(f"from_pretrained: {args.from_pretrained}")
    logger.info(f"tokenizer: {args.tokenizer}")


    if args.tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer) #sepcialized tokenizer
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.from_pretrained) #general tokenizer-->t5-small


    # add sep token
    tokenizer.add_special_tokens({'sep_token': '[SEP]'})

    if args.model_type == 'encoder-decoder':
        #global_attention_first_token = False  # should be True for LED
        encode_plus_kwargs = {'truncation': True, 'padding': 'longest', 'pad_to_multiple_of': 1}
        # generate_kwargs = {'max_length': args.target_seq_len, 'min_length': args.target_seq_len}
        generate_kwargs = {}


        def collate_fn(batch):
            # print('batch', batch[0].keys(), batch[0]['input'])
            # cut too long strings because they may slow down tokenization
            inputs = [b['input'][:args.input_seq_len * 10] for b in batch] #default input_seq_len = 128
            if 'outputs' in batch[0]:
                # if we have more than 1 label per example (only in valid) take only one of them
                # to compute loss on valid
                labels = [b['outputs'][:args.target_seq_len * 10] for b in batch] #default target_seq_len = 16
            else:
                labels = [b['output'][:args.target_seq_len * 10] for b in batch]

            if args.input_prefix: #add task prefix to an input string, default = ""
                inputs = [args.input_prefix + inp for inp in inputs]


            #tokenize inputs
            features = tokenizer.batch_encode_plus(list(inputs), max_length=args.input_seq_len, return_tensors='pt',
                                                   **encode_plus_kwargs)
            #{'input_ids': [27, 8, 3, 9, 1695, 1523, 13, 8, 3, 27168, 5], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}


            #tokenize labels
            with tokenizer.as_target_tokenizer():
                labels = tokenizer.batch_encode_plus(list(labels), max_length=args.target_seq_len, return_tensors='pt',
                                                     **encode_plus_kwargs).input_ids
            labels[labels == tokenizer.pad_token_id] = -100
            features['labels'] = labels

            #             features = {
            #     'input_ids': Tensor([...]),
            #     'attention_mask': Tensor([...]),
            #     'labels': Tensor([...])
            # }


            if 'outputs' in batch[0]:
                features['target_text'] = [b['outputs'] for b in batch]
            else:
                features['target_text'] = [b['output'] for b in batch]
            # if 'global_attention_mask' in features:
            #     raise RuntimeError('What global attention mask for Longformer and LongformerEncoder-Decoder should be?')
            return features

#             features = {
#     'input_ids': Tensor([...]),         # توکن‌های ورودی
#     'attention_mask': Tensor([...]),    # ماسک توجه ورودی‌ها
#     'labels': Tensor([...]),            # توکن‌های برچسب‌ها
#     'target_text': ["This is target text 1", "This is target text 2"]  # متن برچسب‌ها
# }

    logger.info(f'Preparing dataset for: {args.task_name}')

    train_dataset = KGLMDataset(train_df, neighborhood=not args.drop_neighborhood)#it's time to use train dataset for model

    per_worker_batch_size = args.batch_size * args.gradient_accumulation_steps
    kwargs = {'pin_memory': True, 'num_workers': args.data_n_workers}
    train_dataloader = DataLoader(
    train_dataset,
    batch_size=per_worker_batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    **kwargs
)

    logger.info(f'Preparing validation data for: {args.task_name}')
    valid_dataset = KGLMDataset(valid_df, neighborhood=not args.drop_neighborhood)
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=per_worker_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        **kwargs
    )

    #log on validation data every N batches (default: None)
    if args.valid_interval is None:

      args.valid_interval = args.log_interval  #log to report loss, metrics on training data every N batches (default: None)



    model_cls = get_cls_by_name(args.model_cls)  # "transformers:T5ForConditionalGeneration"
    logger.info(f'Using model class: {model_cls}')
    logger.info(f'Loading pretrained model: {args.from_pretrained}')
    model = model_cls.from_pretrained(args.from_pretrained)

    ## load cpt
    if args.cpt_path: #loading best model wight from checkpoint
        model_cpt = os.path.join(args.cpt_path, "model_best.pth")
        cpt = torch.load(model_cpt, map_location='cpu')
        model.load_state_dict(cpt['model_state_dict'])



    logger.info('Using AdamW optimizer')
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
        )


    def keep_for_metrics_fn(batch, output):
      # select data from batch and model output that would be used to compute metrics
      data = {}
      if 'generation_outputs' in output:
          data['labels'] = batch['target_text']  # برچسب‌های اصلی (متن هدف)
          data['generation_outputs'] = output['generation_outputs']  # متن تولیدشده توسط مدل
      return data


    def metrics_fn(data):
      # compute metrics based on stored labels, predictions, ...
      metrics = {}
      y, p = None, None

      if args.model_type == 'encoder-decoder' and 'generation_outputs' in data:
          # replace -100 with pad token in labels
          y = data['labels']
          # print('!', data['generation_outputs'].shape)
          p = tokenizer.batch_decode(data['generation_outputs'], skip_special_tokens=True)
          if args.show_valid_examples > 0:
          # if args.show_valid_examples > 0:
              for i in range(min(args.show_valid_examples, len(y))):
                  logger.info(f'y: {y[i]}')
                  logger.info(f'p: {p[i]}')
                  logger.info(f'p ids: {data["generation_outputs"][i]}')
                  logger.info('-' * 50)


      if y is not None and p is not None:
          metrics['exact_match'] = accuracy_score(y, p) * 100

      return metrics


    trainer = Trainer(args, model, optimizer, train_dataloader, valid_dataloader, 
                      keep_for_metrics_fn=keep_for_metrics_fn, metrics_fn=metrics_fn,
                      generate_kwargs=generate_kwargs if args.use_generate_on_valid else {})

    if not args.validate_only:
        # train loop
        trainer.train()
        # make sure all workers are done
  
        # run validation after training
        if args.save_best:
            best_model_path = str(Path(args.model_path) / 'model_best.pth')

            logger.info(f'Loading best saved model from {best_model_path}')
            trainer.load(best_model_path)
        if valid_dataloader is not None:

            logger.info('Runnning validation on valid data:')
            trainer.validate(valid_dataloader, write_tb=False)
    else:
        # run validation, do not write to tensorboard

        logger.info('Running validation on train set:')
        trainer.validate(train_dataloader, split='train', write_tb=False)
        if valid_dataloader is not None:

            logger.info('Running validation on valid data:')
            trainer.validate(valid_dataloader, write_tb=False)