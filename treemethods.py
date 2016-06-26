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
        
    def fit(self, X, y):
        self.X = X
        self.y = y
    
    def set_node(self, node_number, variable, cuttoff):
        self.graph.node[node_number]['variable'] = variable
        self.graph.node[node_number]['cutoff'] = cutoff
        
    def new_node(self, parent):
        self.nodes += 1
        self.graph.add_edge(parent, self.nodes)
        
    def CART(self):
        min_error = np.inf
        min_feature = None
        min_split = None
        for feature in range(np.shape(self.X)[1]):
            feature_vector = X[:, feature]
            sorted_vector = np.unique(np.sort(feature_vector))
            feature_splits = (sorted_vector[1:] + sorted_vector[:-1]) / 2
            for split in feature_splits:
                lower_class_average = np.mean(y[feature_vector < split])
                upper_class_average = np.mean(y[feature_vector > split])
                lower_class_errors = y[feature_vector < split] - lower_class_average
                upper_class_errors = y[feature_vector > split] - upper_class_average
                total_error = np.inner(lower_class_errors, lower_class_errors) + np.inner(upper_class_errors, upper_class_errors)
                if total_error < min_error:
                    min_error = total_error
                    min_feature = feature
                    min_split = split
        print(min_error, min_feature, min_split)
            
        
a = RegressionTree()        
a.fit(X, y)    
a.CART()