import torch
import numpy as np
from pcst_fast import pcst_fast
from torch_geometric.data.data import Data
import faiss

def retrieval_via_pcst(graph, q_emb, textual_nodes, textual_edges, topk=3, topk_e=3, cost_e=0.5):
    c = 0.01
    if len(textual_nodes) == 0 or len(textual_edges) == 0:
        desc = textual_nodes.to_csv(index=False) + '\n' + textual_edges.to_csv(index=False, columns=['src', 'edge_attr', 'dst'])
        graph = Data(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr, num_nodes=graph.num_nodes)
        return graph, desc

    root = -1  # unrooted. In an unrooted graph, the algorithm finds the best subgraph without being constrained by a specific starting node.
    num_clusters = 1 #he number of connected subgraphs
    pruning = 'gw'# Greedy pruning with weights
    verbosity_level = 0 #Run silently without printing additional debug information.

    #Prize for nodes
# جایگزین کل بخش محاسبه n_prizes
    if topk > 0:
        # تبدیل داده‌ها به NumPy و ساخت ایندکس Faiss
        node_embeddings = graph.x.cpu().numpy().astype('float32')  # امبدینگ‌های نودها
        d = node_embeddings.shape[1]  # ابعاد بردارها
        faiss_index = faiss.IndexFlatL2(d)  # ایندکس FlatL2 برای محاسبه فاصله اقلیدسی

        # افزودن امبدینگ‌ها به ایندکس
        faiss_index.add(node_embeddings)

        # تبدیل q_emb به NumPy برای Faiss
        query_vector = q_emb.cpu().numpy().astype('float32')

        # جستجو در Faiss
        topk = min(topk, graph.num_nodes)  # اطمینان از اینکه topk بیشتر از تعداد نودها نباشد
        distances, indices = faiss_index.search(query_vector, topk)  # فاصله‌ها و ایندکس‌ها

        # تولید n_prizes
        n_prizes = torch.zeros(graph.num_nodes)  # جایزه‌ها برای همه نودها صفر است
        for rank, idx in enumerate(indices[0]):
            n_prizes[idx] = topk - rank  # هرچه رتبه کمتر باشد، امتیاز بالاتر است
    else:
        n_prizes = torch.zeros(graph.num_nodes)



    # جایگزین بخش محاسبه e_prizes
    if topk_e > 0:
        # ساخت ایندکس Faiss برای یال‌ها
        edge_embeddings = graph.edge_attr.cpu().numpy().astype('float32')  # امبدینگ‌های یال‌ها
        d = edge_embeddings.shape[1]  # ابعاد بردار یال‌ها
        faiss_edge_index = faiss.IndexFlatL2(d)  # استفاده از فاصله اقلیدسی (L2)

        # افزودن امبدینگ‌های یال‌ها به ایندکس
        faiss_edge_index.add(edge_embeddings)

        # جستجو در Faiss برای یال‌ها
        topk_e = min(topk_e, graph.edge_attr.size(0))  # اطمینان از اینکه topk_e بیشتر از تعداد یال‌ها نیست
        e_distances, e_indices = faiss_edge_index.search(query_vector, topk_e)  # فاصله‌ها و ایندکس‌ها

        # تولید e_prizes
        e_prizes = torch.zeros(graph.num_edges)  # جایزه‌ها برای همه یال‌ها صفر است
        for rank, idx in enumerate(e_indices[0]):
            e_prizes[idx] = topk_e - rank  # هرچه رتبه کمتر باشد، امتیاز بالاتر است

        # کاهش هزینه یال‌ها برای اطمینان از انتخاب حداقل یک یال
        cost_e = min(cost_e, e_prizes.max().item() * (1 - c / 2))
    else:
        e_prizes = torch.zeros(graph.num_edges)



    #check the main condition in algorithm
    costs = []
    edges = []
    vritual_n_prizes = []

    virtual_edges = []
    virtual_costs = []
    mapping_n = {}
    mapping_e = {}
    for i, (src, dst) in enumerate(graph.edge_index.T.numpy()):
        prize_e = e_prizes[i]


        if prize_e <= cost_e:
            mapping_e[len(edges)] = i
            edges.append((src, dst))
            costs.append(cost_e - prize_e)
        else:
            virtual_node_id = graph.num_nodes + len(vritual_n_prizes)
            mapping_n[virtual_node_id] = i
            virtual_edges.append((src, virtual_node_id))
            virtual_edges.append((virtual_node_id, dst))
            virtual_costs.append(0)
            virtual_costs.append(0)
            vritual_n_prizes.append(prize_e - cost_e)

    prizes = np.concatenate([n_prizes, np.array(vritual_n_prizes)])
    num_edges = len(edges)
    if len(virtual_costs) > 0:
        costs = np.array(costs+virtual_costs)
        edges = np.array(edges+virtual_edges)

    vertices, edges = pcst_fast(edges, prizes, costs, root, num_clusters, pruning, verbosity_level)


    #Generate subgraph
    selected_nodes = vertices[vertices < graph.num_nodes]
    selected_edges = [mapping_e[e] for e in edges if e < num_edges]
    virtual_vertices = vertices[vertices >= graph.num_nodes]
    if len(virtual_vertices) > 0:
        virtual_vertices = vertices[vertices >= graph.num_nodes]
        virtual_edges = [mapping_n[i] for i in virtual_vertices]
        selected_edges = np.array(selected_edges+virtual_edges)

    edge_index = graph.edge_index[:, selected_edges]
    selected_nodes = np.unique(np.concatenate([selected_nodes, edge_index[0].numpy(), edge_index[1].numpy()]))

    n = textual_nodes.iloc[selected_nodes]
    e = textual_edges.iloc[selected_edges]



    # ایجاد دیکشنری برای دسترسی سریع به نام نودها
    node_dict = dict(zip(n['node_id'], n['node_attr']))

    # تبدیل یال‌ها به جملات و اضافه کردن SEP
    sentences = []
    for _, edge in e.iterrows():
        # پیدا کردن نام‌های نودها بر اساس شناسه‌ها
        src_node = node_dict.get(edge['src'])
        dst_node = node_dict.get(edge['dst'])
        
        # ایجاد جمله
        sentence = f"{src_node} {edge['edge_attr']} {dst_node}"
        sentences.append(sentence)

    # اضافه کردن SEP بین جملات
    final_sentence = " SEP ".join(sentences)

    print(final_sentence)


    desc=final_sentence
    #desc = n.to_csv(index=False)+'\n'+e.to_csv(index=False, columns=['src', 'edge_attr', 'dst'])

    mapping = {n: i for i, n in enumerate(selected_nodes.tolist())}

    x = graph.x[selected_nodes]
    edge_attr = graph.edge_attr[selected_edges]
    src = [mapping[i] for i in edge_index[0].tolist()]
    dst = [mapping[i] for i in edge_index[1].tolist()]
    edge_index = torch.LongTensor([src, dst])
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(selected_nodes))

    return data, desc