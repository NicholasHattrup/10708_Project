from ChemGraph import *
import torch.utils.data as data
from copy import deepcopy

class SDS(data.Dataset):

    def __init__(self, root_path, ids, graph_library):
        self.root = root_path
        self.ids = ids
        self.graph_library = graph_library

    def __getitem__(self, index):
        file_id = self.ids[index]
        for i,G in enumerate(self.graph_library):
            if G.graph.graph["FilePath"] == self.root+file_id:
                print(index, i)
        G = self.graph_library[index]
        nodes = []
        edges = {}
        graph = nx.to_numpy_matrix(G.graph)
        for i in G.graph.nodes:
            nodes.append(G.graph.nodes[i]['Features'])

        for (a, b, f) in G.graph.edges.data('Features'):
            feat = deepcopy(f).tolist()
            feat.append(G.graph.edges[a, b]['Distance'])
            edges[(a, b)] = feat
            target = [G.graph.graph['Gap']]

        return (graph, nodes, edges), target
        
        
    def __len__(self):
        return len(self.ids)
