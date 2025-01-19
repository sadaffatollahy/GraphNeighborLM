import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TransformerConv, GATConv


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads=-1):
      """
      :param in_channels:Number of input features for each node
      :param hidden_channels:Number of input features for each node
      :param out_channels:Number of output features for each node
      :param num_layers:Total number of graph convolutional layers.
      :param dropout:Dropout probability for regularization.
      :param num_heads:Dropout probability for regularization.
      """
      super(GCN, self).__init__()
      self.convs = torch.nn.ModuleList() #an empty ModuleList to hold the graph convolutional layers (GCNConv).

      #First layer:
      self.convs.append(GCNConv(in_channels, hidden_channels)) #Adds the first graph convolutional layer to self.convs.
      self.bns = torch.nn.ModuleList() # an empty ModuleList to hold the batch normalization layers
      self.bns.append(torch.nn.BatchNorm1d(hidden_channels)) #Adds a batch normalization layer to self.bns

      #Intermediate layers
      for _ in range(num_layers - 2):
          self.convs.append(GCNConv(hidden_channels, hidden_channels))
          self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

      #Final layer
      self.convs.append(GCNConv(hidden_channels, out_channels))


      self.dropout = dropout #tores the dropout probability for use during the forward pass.

    def reset_parameters(self):
      """model starts training with fresh weights, preventing the influence of previously learned weights from affecting new experiments.
      """
      for conv in self.convs:
          conv.reset_parameters()
      for bn in self.bns:
          bn.reset_parameters()

    def forward(self, x, adj_t, edge_attr):
      """
      x:Node feature matrix of shape (num_nodes, in_channels)
      adj_t:adjacency matrix in transposed format, representing the graph structure.Encodes which nodes are connected and enables propagation of information between nodes.
      edge_attr:Features associated with edges in the graph.
      """
      for i, conv in enumerate(self.convs[:-1]):
          x = conv(x, adj_t) #Updates the node features by aggregating information from neighboring nodes and applying a learned transformation.
          x = self.bns[i](x)
          x = F.relu(x)
          x = F.dropout(x, p=self.dropout, training=self.training)
      x = self.convs[-1](x, adj_t) #Applies the final graph convolutional layer to produce the output node representations.
      return x, edge_attr
    

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads=4):
        super(GAT, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=num_heads, concat=False)) #The outputs of multiple attention heads are averaged (not concatenated) into a single feature vector per node.
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels, heads=num_heads, concat=False))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GATConv(hidden_channels, out_channels, heads=num_heads, concat=False))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index=edge_index, edge_attr=edge_attr)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x,edge_index=edge_index, edge_attr=edge_attr)
        return x, edge_attr
    

class GraphTransformer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads=-1):  # num_heads=-1 : when model create we should define it
        super(GraphTransformer, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(TransformerConv(in_channels=in_channels, out_channels=hidden_channels//num_heads, heads=num_heads, edge_dim=in_channels, dropout=dropout)) #edge_dim : to influence edge attribute
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(TransformerConv(in_channels=hidden_channels, out_channels=hidden_channels//num_heads, heads=num_heads, edge_dim=in_channels, dropout=dropout,))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(TransformerConv(in_channels=hidden_channels, out_channels=out_channels//num_heads, heads=num_heads, edge_dim=in_channels, dropout=dropout,))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t, edge_attr):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index=adj_t, edge_attr=edge_attr)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index=adj_t, edge_attr=edge_attr)
        return x, edge_attr
    

load_gnn_model = {
    'gcn': GCN,
    'gat': GAT,
    'gt': GraphTransformer,
}