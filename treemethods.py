import networkx as nx

class RegressionTree:
    """
    
    """
    def __init__(self):
        self.graph = nx.Graph()
        self.graph.add_node(1, variable = None, cutoff = None)
        self.nodes = 1
        self.X = None
        self.y = None
        self.learned = False
        
    def self.fit(self, X, y):
        self.X = X
        self.y = y
    
    def set_node(self, node_number, variable, cuttoff):
        self.graph.node[node_number]['variable'] = variable
        self.graph.node[node_number]['cutoff'] = cutoff
        
    def new_node(self, parent):
        self.nodes += 1
        self.graph.add_edge(parent, self.nodes)