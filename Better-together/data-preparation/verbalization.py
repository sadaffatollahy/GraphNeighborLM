
import pandas as pd
from tqdm import tqdm
import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import argparse, sys
from loguru import logger


# Configure logging
logger.add("verbalizer.log", level="INFO", rotation="10 MB")

class Verbalizer:
    def __init__(self, df, similarity_matrix=None, relation2index=None, entity2text=None, relation2text=None):
        self.df = df
        self.similarity_matrix = similarity_matrix
        self.relation2index = relation2index
        self.entity2text = entity2text
        self.relation2text = relation2text
        self.sep = '[SEP]'

    def get_neighbourhood(self, node_id, relation_id=None, tail_id=None, limit=None):
        neighs = []

        if tail_id is None:
            # Neighbors for test dataset
            head_matches = self.df[self.df['head'] == node_id].copy()
            if limit:
                head_matches = head_matches.head(limit)
            neighs.extend(head_matches.to_dict('records'))

            tail_matches = self.df[self.df['tail'] == node_id].copy()
            if limit:
                tail_matches = tail_matches.head(limit)
            for _, row in tail_matches.iterrows():
                row = row.to_dict()
                row['relation'] = 'inverse of ' + row['relation']
                neighs.append(row)
        else:
            # Neighbors for train dataset
            head_matches = self.df[(self.df['head'] == node_id) &
                                   ((self.df['tail'] != tail_id) |
                                    (self.df['relation'] != relation_id))].copy()
            if limit:
                head_matches = head_matches.head(limit)
            neighs.extend(head_matches.to_dict('records'))

            tail_matches = self.df[(self.df['tail'] == node_id) &
                                   ((self.df['head'] != tail_id) |
                                    (self.df['relation'] != relation_id))].copy()
            if limit:
                tail_matches = tail_matches.head(limit)
            for _, row in tail_matches.iterrows():
                row = row.to_dict()
                row['relation'] = 'inverse of ' + row['relation']
                neighs.append(row)

        return neighs

    def verbalize(self, head, relation, tail=None, inverse=False):
        relation_prefix = 'inverse of ' if inverse else ''
        limit = 200 if inverse else None

        neighbourhood = self.get_neighbourhood(head, relation, tail, limit)
        relation_text = relation_prefix + self.relation2text[relation]

        # Sort based on similarity
        neighbourhood.sort(
            key=lambda x: self.similarity_matrix[self.relation2index[self.relation2text[x['relation']]]]
                         [self.relation2index[relation_text]],
            reverse=True
        )

        neighbourhood = neighbourhood[:512]
        verbalization = f"predict {self.sep} {self.entity2text[head]} {relation_text} {self.sep} "

        verbalization += " ".join(
            list(
                map(
                    lambda x: f"{self.relation2text[x['relation']]} " +
                              f"{self.entity2text[x['tail']] if x['head'] == head else self.entity2text[x['head']]} {self.sep}",
                    neighbourhood
                )
            )
        )

        return " ".join(verbalization.split()).strip()


def verbalize_dataset(input_df, verbalizer):
    docs = []

    for i, doc in tqdm(input_df.iterrows(), total=len(input_df)):
        try:
            direct_verbalization = verbalizer.verbalize(doc['head'], doc['relation'], doc['tail'])
            docs.append({
                'id': i * 2,
                'verbalization': direct_verbalization,
                'head': doc['head'],
                'tail': doc['tail'],
                'relation': doc['relation'],
                'verbalized_tail': verbalizer.entity2text[doc['tail']]
            })

            inverse_verbalization = verbalizer.verbalize(doc['tail'], doc['relation'], doc['head'], inverse=True)
            docs.append({
                'id': i * 2 + 1,
                'verbalization': inverse_verbalization,
                'head': doc['tail'],
                'tail': doc['head'],
                'relation': "inverse of " + doc['relation'],
                'verbalized_tail': verbalizer.entity2text[doc['head']]
            })

        except Exception as e:
            print(f"Exception {e} on {i}th triplet")

    return pd.DataFrame(docs)


parser = argparse.ArgumentParser()

parser.add_argument("--relation_vectors_path", help="path to the embeddings of verbalized relations", default="/home/ahmadi/sadaf/GraphNeighborLM/Better-together/data-preparation/data/embeddings/fasttext_vecs-wikidata5m.npy")
parser.add_argument("--rel2ind_path", help="path to the mapping of textual relations to the index of corresponding vectors", default="/home/ahmadi/sadaf/GraphNeighborLM/Better-together/data-preparation/data/relation2ind-wikidata5m.json")
parser.add_argument("--entity_mapping_path", help="path to the entity2text mapping", default="/home/ahmadi/sadaf/GraphNeighborLM/Better-together/data-preparation/data/mappings/wd5m_aliases_entities_v3.txt")
parser.add_argument("--relation_mapping_path", help="path to the relation2text mapping", default="/home/ahmadi/sadaf/GraphNeighborLM/Better-together/data-preparation/data/relation2text-wikidata5m.json")
parser.add_argument("--train_path", help="train KG path", default='/home/ahmadi/sadaf/GraphNeighborLM/Better-together/data-preparation/datasets/wikidata5m/subset_train.txt')
parser.add_argument("--valid_path", help="valid KG path", default='/home/ahmadi/sadaf/GraphNeighborLM/Better-together/data-preparation/datasets/wikidata5m/subset_val.txt')
parser.add_argument("--test_path", help="test KG path", default='/home/ahmadi/sadaf/GraphNeighborLM/Better-together/data-preparation/datasets/wikidata5m/subset_test.txt')
parser.add_argument("--train_verbalized_output", help="verbalized train KG path", default='/home/ahmadi/sadaf/GraphNeighborLM/Better-together/data-preparation/datasets/wikidata5m/verbalized_train.csv')
parser.add_argument("--valid_verbalized_output", help="verbalized valid KG path", default='/home/ahmadi/sadaf/GraphNeighborLM/Better-together/data-preparation/datasets/wikidata5m/verbalized_valid.csv')
parser.add_argument("--test_verbalized_output", help=" verbalized test KG path", default='/home/ahmadi/sadaf/GraphNeighborLM/Better-together/data-preparation/datasets/wikidata5m/verbalized_test.csv')

args = parser.parse_args()

# Load data and mappings
try:
    vecs = np.load(args.relation_vectors_path)
    similarity_matrix = cosine_similarity(vecs)

    with open(args.rel2ind_path, 'r') as f:
        rel2ind = json.load(f)

    entity_mapping = {}
    with open(args.entity_mapping_path, 'r') as f:
        for line in f:
            _id, name = line.strip().split('\t')
            entity_mapping[_id] = name

    with open(args.relation_mapping_path, 'r') as f:
        relation_mapping = json.load(f)
except Exception as e:
    logger.error(f"Error loading files: {e}")
    sys.exit(1)


columns = ["head", "relation", "tail"]
train_df = pd.read_csv(args.train_path, sep='\t', header=None, names=columns)
test_df = pd.read_csv(args.test_path, sep='\t', header=None, names=columns)
valid_df = pd.read_csv(args.valid_path, sep='\t', header=None, names=columns)

# Verbalize datasets
verbalizer = Verbalizer(train_df, similarity_matrix=similarity_matrix,
                        relation2index=rel2ind, entity2text=entity_mapping, relation2text=relation_mapping)

logger.info('Verbalizing train KG...')
train_verbalized = verbalize_dataset(train_df, verbalizer)

logger.info('Verbalizing valid KG...')
valid_verbalized = verbalize_dataset(valid_df, verbalizer)

logger.info('Verbalizing test KG...')
test_verbalized = verbalize_dataset(test_df, verbalizer)

# Save results
train_verbalized.to_csv(args.train_verbalized_output, index=False)
valid_verbalized.to_csv(args.valid_verbalized_output, index=False)
test_verbalized.to_csv(args.test_verbalized_output, index=False)