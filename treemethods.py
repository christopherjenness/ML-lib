import networkx as nx

class RegressionTree:
    """
    
    """
    def __init__(self):
        self.graph = nx.DiGraph()
        self.graph.add_node(1, variable = None, cutoff = None)
        self.nodes = 1
        self.X = None
        self.y = None
        self.learned = False
        
    def fit(self, X, y, height):
        self.X = X
        self.y = y
        for layer in range(height):
            self.add_layer()
    
    def set_node(self, node_number, variable, cutoff):
        self.graph.node[node_number]['variable'] = variable
        self.graph.node[node_number]['cutoff'] = cutoff
        
    def new_nodes(self, parent, number):
        for i in range(number):
            self.nodes += 1
            self.graph.add_edge(parent, self.nodes)
        
    def CART(self, inputs, values):
        min_error = np.inf
        min_feature = None
        min_split = None
        for feature in range(np.shape(inputs)[1]):
            feature_vector = inputs[:, feature]
            sorted_vector = np.unique(np.sort(feature_vector))
            feature_splits = (sorted_vector[1:] + sorted_vector[:-1]) / 2
            for split in feature_splits:
                lower_class_average = np.mean(values[feature_vector < split])
                upper_class_average = np.mean(values[feature_vector > split])
                lower_class_errors = values[feature_vector < split] - lower_class_average
                upper_class_errors = values[feature_vector > split] - upper_class_average
                total_error = np.inner(lower_class_errors, lower_class_errors) + np.inner(upper_class_errors, upper_class_errors)
                if total_error < min_error:
                    min_error = total_error
                    min_feature = feature
                    min_split = split
        return min_feature, min_split
        
    def add_split(self, node_number, X, y):
        min_feature, min_split = self.CART(X, y)
        self.set_node(node_number, min_feature, min_split)
        self.new_nodes(node_number, 2)
        
    def get_predecessors(self, node_number):
        predecessors = []
        current_node = node_number
        while len(self.graph.predecessors(current_node)) > 0:
            current_node = self.graph.predecessors(current_node)[0]
            predecessors.append(current_node)
        return predecessors
        
    def partition_data(self, node_number):
        predecessors = self.get_predecessors(node_number)
        predecessors.reverse()
        predecessors.append(node_number)
        data_indices = np.array(range(len(self.y)))
        node_count = 0
        while node_count < len(predecessors) - 1:
            current_node = predecessors[node_count]
            next_node = predecessors[node_count + 1]
            current_variable = self.graph.node[current_node]['variable']
            current_cutoff = self.graph.node[current_node]['cutoff']
            if current_cutoff == None:
                return []
            if next_node == min(self.graph.successors(current_node)):
                data_indices = data_indices[self.X[data_indices, current_variable] < current_cutoff]
            else:
                data_indices = data_indices[self.X[data_indices, current_variable] > current_cutoff]
            node_count +=1
        return data_indices
        
    def get_leaves(self):
        leaves = []
        for node in self.graph.nodes():
            if len(self.graph.successors(node)) == 0:
                leaves.append(node)
        return leaves
        
    def add_layer(self):
        leaves = self.get_leaves()
        for leaf in leaves:
            data_indices = self.partition_data(leaf)
            leaf_X = self.X[data_indices, :]
            leaf_y = self.y[data_indices]
            self.add_split(leaf, leaf_X, leaf_y)
            
    def compute_class_averages(self):
        for i in range(2, self.nodes + 1):
            print(i)
            parent = self.graph.predecessors(i)[0]
            if self.graph.node[parent]['cutoff'] == None:
                self.graph.node[i]['classval'] = np.nan
            else:
                node_indices = self.partition_data(i)
                classval = self.y[node_indices].mean()
                self.graph.node[i]['classval'] = classval
            
        
        
a = RegressionTree()        
a.fit(X, y, 4)    
a.compute_class_averages()
print(a.graph.node)









