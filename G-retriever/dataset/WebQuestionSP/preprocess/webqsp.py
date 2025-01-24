
import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import datasets
from tqdm import tqdm
from dataset.pcst.retrieval import retrieval_via_pcst
import os
import torch
from torch.utils.data import Dataset
from datasets import load_from_disk, concatenate_datasets

model_name = 'sbert'
path = '/home/ahmadi/sadaf/GraphNeighborLM/G-retriever/dataset/WebQuestionSP'
path_nodes = f'{path}/nodes'
path_edges = f'{path}/edges'
path_graphs = f'{path}/graphs'

cached_graph = f'{path}/cached_graphs'
cached_desc = f'{path}/cached_desc'




class WebQSPDataset(Dataset):  # This is for PyTorch-compatible datasets
    def __init__(self):
        super().__init__()
        self.prompt = 'Please answer the given question.'
        self.graph = None
        self.graph_type = 'Knowledge Graph'

        # Load dataset using HuggingFace's library
        sample_dataset = load_from_disk(f'{path}/processed_dataset')
        self.combined_dataset = concatenate_datasets(
            [sample_dataset['train'], sample_dataset['validation'], sample_dataset['test']]
        )
        self.q_embs = torch.load(f'{path}/q_embs.pt')

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.combined_dataset)

    def __getitem__(self, index):
        """Get an item by index."""
        data = self.combined_dataset[index]
        question = f'Question: {data["question"]}\nAnswer: '
        graph = torch.load(f'{cached_graph}/{index}.pt')
        desc = open(f'{cached_desc}/{index}.txt', 'r').read()
        label = ('|').join(data['answer']).lower()

        return {
            'id': index,
            'question': question,
            'label': label,
            'graph': graph,
            'desc': desc,
        }

    def get_idx_split(self):
        """Load the saved train/val/test split indices."""
        with open(f'{path}/split/train_indices.txt', 'r') as file:
            train_indices = [int(line.strip()) for line in file]
        with open(f'{path}/split/val_indices.txt', 'r') as file:
            val_indices = [int(line.strip()) for line in file]
        with open(f'{path}/split/test_indices.txt', 'r') as file:
            test_indices = [int(line.strip()) for line in file]

        return {'train': train_indices, 'val': val_indices, 'test': test_indices}



def preprocess():
    os.makedirs(cached_desc, exist_ok=True)
    os.makedirs(cached_graph, exist_ok=True)

    # Load the dataset from disk
    dataset_dict = datasets.load_from_disk(f'{path}/processed_dataset')

    # Combine train, validation, and test splits for preprocessing
    # combined_dataset = datasets.concatenate_datasets(
    #     [dataset_dict['train'], dataset_dict['validation'], dataset_dict['test']]
    # )

    q_embs = torch.load(f'{path}/q_embs.pt')

    for index in tqdm(range(len(dataset_dict))):
        if os.path.exists(f'{cached_graph}/{index}.pt'):
            continue

        nodes = pd.read_csv(f'{path_nodes}/{index}.csv')
        edges = pd.read_csv(f'{path_edges}/{index}.csv')
        if len(nodes) == 0:
            print(f'Empty graph at index {index}')
            continue
        graph = torch.load(f'{path_graphs}/{index}.pt')
        q_emb = q_embs[index]
        subg, desc = retrieval_via_pcst(graph, q_emb, nodes, edges, topk=3, topk_e=5, cost_e=0.5)
        torch.save(subg, f'{cached_graph}/{index}.pt')
        open(f'{cached_desc}/{index}.txt', 'w').write(desc)


if __name__ == '__main__':

    preprocess()

    dataset = WebQSPDataset()

    data = dataset[1]
    for k, v in data.items():
        print(f'{k}: {v}')

    split_ids = dataset.get_idx_split()
    for k, v in split_ids.items():
        print(f'# {k}: {len(v)}')