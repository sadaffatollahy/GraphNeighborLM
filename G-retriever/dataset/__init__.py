from dataset.Explanation_graph.preprocess.expla_graphs import ExplaGraphsDataset
from dataset.WebQuestionSP.preprocess.webqsp import WebQSPDataset
# from src.dataset.scene_graphs import SceneGraphsDataset
# from src.dataset.scene_graphs_baseline import SceneGraphsBaselineDataset

# from src.dataset.webqsp_baseline import WebQSPBaselineDataset


load_dataset = {
    'expla_graphs': ExplaGraphsDataset,
    'webqsp': WebQSPDataset,
    # 'scene_graphs': SceneGraphsDataset,
    # 'scene_graphs_baseline': SceneGraphsBaselineDataset,
    # 'webqsp_baseline': WebQSPBaselineDataset,
}