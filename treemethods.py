import numpy as np
import networkx as nx
import abc
from scipy.stats import mode

class BaseTree(object):
    """
    
    """
    __metaclass__ = abc.ABCMeta
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.graph.add_node(1, variable = None, cutoff = None)
        self.nodes = 1
        self.X = None
        self.y = None
        self.learned = False
    
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
        
    def add_split(self, node_number, data, values):
        min_feature, min_split = self.CART(data, values)
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
            print('_____')
            print(leaf, data_indices)
            leaf_X = self.X[data_indices, :]
            leaf_y = self.y[data_indices]
            self.add_split(leaf, leaf_X, leaf_y)
            
    def compute_class_averages(self):
        for i in range(2, self.nodes + 1):
            parent = self.graph.predecessors(i)[0]
            if self.graph.node[parent]['cutoff'] == None:
                self.graph.node[i]['classval'] = self.graph.node[parent]['classval']
            else:
                node_indices = self.partition_data(i)
                classval = self.y[node_indices].mean()
                self.graph.node[i]['classval'] = classval

    def fit(self, X, y, height):
        self.X = X
        self.y = y
        for layer in range(height):
            self.add_layer()
        self.compute_class_averages()
        self.learned = True
        
    def predict(self, x):
        if not self.learned:
            raise NameError('Fit model first')
        current_node = 1
        leaves = self.get_leaves()
        while current_node not in leaves:
            children = self.graph.successors(current_node)
            current_variable = self.graph.node[current_node]['variable']
            current_cutoff = self.graph.node[current_node]['cutoff']
            if current_variable == None:
                return self.graph.node[current_node]['classval']
            if x[current_variable] > current_cutoff:
                current_node = children[1]
            else:
                current_node = children[0]
        return self.graph.node[current_node]['classval']
        
    @abc.abstractmethod    
    def compute_class_averages(self):
        return
    
    @abc.abstractmethod
    def CART(self, inputs, values):
        return 
            
class RegressionTree(BaseTree):
    
    def __init__(self):
        BaseTree.__init__(self)
        
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
        
    def compute_class_averages(self):
        for i in range(2, self.nodes + 1):
            parent = self.graph.predecessors(i)[0]
            if self.graph.node[parent]['cutoff'] == None:
                self.graph.node[i]['classval'] = self.graph.node[parent]['classval']
            else:
                node_indices = self.partition_data(i)
                classval = self.y[node_indices].mean()
                self.graph.node[i]['classval'] = classval
    
        
class ClassificationTree(BaseTree):
    
    def __init__(self):
        BaseTree.__init__(self)
        
    def CART(self, inputs, values):
        """
        Uses misclassification error function.
        """
        min_error = np.inf
        min_feature = None
        min_split = None
        for feature in range(np.shape(inputs)[1]):
            feature_vector = inputs[:, feature]
            sorted_vector = np.unique(np.sort(feature_vector))
            feature_splits = (sorted_vector[1:] + sorted_vector[:-1]) / 2
            for split in feature_splits:
                lower_class_mode = mode(values[feature_vector < split]).mode[0]
                upper_class_mode = mode(values[feature_vector > split]).mode[0]
                lower_class_errors = np.sum(values[feature_vector < split] != lower_class_mode)
                upper_class_errors = np.sum(values[feature_vector > split] != upper_class_mode)
                total_error = upper_class_errors + lower_class_errors
                if total_error < min_error:
                    min_error = total_error
                    min_feature = feature
                    min_split = split
        return min_feature, min_split
        
    def compute_class_averages(self):
        for i in range(2, self.nodes + 1):
            parent = self.graph.predecessors(i)[0]
            if self.graph.node[parent]['cutoff'] == None:
                self.graph.node[i]['classval'] = self.graph.node[parent]['classval']
            else:
                node_indices = self.partition_data(i)
                classval = mode(self.y[node_indices]).mode[0]
                self.graph.node[i]['classval'] = classval   
                
class Prim(BaseTree):
    """
    Patient Rule Induction Method
    """
    def partition_data(self, node_number):
        predecessors = self.get_predecessors(node_number)
        predecessors.reverse()
        predecessors.append(node_number)
        data_indices = np.array(range(len(self.y)))
        node_count = 0
        while node_count < len(predecessors) - 1:
            current_node = predecessors[node_count]
            next_node = predecessors[node_count + 1]
            cutoff_dict = self.graph.node[current_node]['cutoffs']
            for key in cutoff_dict:
                current_variable = key
                current_cutoff_min = cutoff_dict[key][0]
                current_cutoff_max = cutoff_dict[key][0]
                boxed_data = data_indices[(self.X[data_indices, current_variable] < current_cutoff_max) & (self.X[data_indices, current_variable] > current_cutoff_min)]
                if next_node == min(self.graph.successors(current_node)):
                    print('asdf')
                    data_indices = boxed_data
                else:
                    print('asdfasdf')
                    data_indices = np.delete(data_indices, boxed_data)
                node_count +=1
        return data_indices
    
    @staticmethod    
    def partition_data_nodeless(inputs, cutoff_dict):
        data_indices = np.array(range(np.shape(inputs)[0]))
        for key in cutoff_dict:
            current_variable = key
            current_cutoff_min = cutoff_dict[key][0]
            current_cutoff_max = cutoff_dict[key][1]
            boxed_data = data_indices[(inputs[data_indices, current_variable] < current_cutoff_max) & (inputs[data_indices, current_variable] > current_cutoff_min)]
            data_indices = boxed_data
        return data_indices
        
        
    def CART(self, inputs, values):
        inputs = inputs
        values = values
        # Aim for box with 10% of initial box size
        target_partition_size = int(len(values) * 0.1)
        # Initalizes Boxes
        # cutoffs is a dictionary where each key is a feature and each value is 
        # a list [min_cutoff, max_cutoff]
        cutoffs = {}
        for feature in range(np.shape(inputs)[1]):
            cutoffs[feature] = [-np.inf, np.inf]
        response_mean = np.mean(values)
        # Contracting phase
        for i in range(4):
            best_feature = None
            best_cutoff = [-np.inf, np.inf]
            response_mean_improvement = 0
            for feature in range(np.shape(inputs)[1]):
                feature_vector = inputs[:, feature]
                sorted_vector = np.unique(np.sort(feature_vector))
                feature_splits = (sorted_vector[1:] + sorted_vector[:-1]) / 2
                lower_split = feature_splits[int(len(feature_splits) * 0.1)]
                upper_split = feature_splits[int(len(feature_splits) * 0.9)]
                upper_class_average = np.mean(values[feature_vector > lower_split])
                lower_class_average = np.mean(values[feature_vector < upper_split])
                max_average = max(upper_class_average, lower_class_average)
                if max_average - response_mean > response_mean_improvement:
                    response_mean_improvement = max_average - response_mean
                    best_feature = feature
                    if upper_class_average > lower_class_average:
                        best_cutoff = [lower_split, np.inf]
                    else:
                        best_cutoff = [-np.inf, upper_split]
            cutoffs[best_feature] = best_cutoff
            boxed_indices = self.partition_data_nodeless(inputs, cutoffs)
            inputs = inputs[boxed_indices, :]
            values = values[boxed_indices]
        return cutoffs
        
    def add_split(self, node_number, data, values):
        cutoffs = self.CART(data, values)
        self.set_node(node_number, cutoffs)
        self.new_nodes(node_number, 2)
                        
    def set_node(self, node_number, cutoffs):
        self.graph.node[node_number]['cutoffs'] = cutoffs
        
    def new_nodes(self, parent, number):
        for i in range(number):
            self.nodes += 1
            self.graph.add_edge(parent, self.nodes)
                        
                        
                        
                        

    
    
    
    
    