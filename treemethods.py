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
        self.graph.add_node(1)
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
            leaf_X = self.X[data_indices, :]
            leaf_y = self.y[data_indices]
            self.add_split(leaf, leaf_X, leaf_y)


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
        data = self.X
        data_indices = np.array(range(len(self.y)))
        node_count = 0
        while node_count < len(predecessors) - 1:
            temp_data = data[data_indices]
            current_node = predecessors[node_count]
            next_node = predecessors[node_count + 1]
            cutoff_dict = self.graph.node[current_node]['cutoffs']
            if cutoff_dict == None:
                return None
            in_box = self.partition_data_nodeless(temp_data, cutoff_dict)
            if in_box == None:
                return None
            if next_node == min(self.graph.successors(current_node)):
                data_indices = data_indices[in_box]
            else:
                data_indices = np.delete(data_indices, in_box)
            node_count +=1
            if len(data_indices)==0:
                return []
        return data_indices
    
    @staticmethod    
    def partition_data_nodeless(inputs, cutoff_dict):
        data_indices = np.array(range(np.shape(inputs)[0]))
        if cutoff_dict == None:
            return None
        for key in cutoff_dict:
            current_variable = key
            current_cutoff_min = cutoff_dict[key][0]
            current_cutoff_max = cutoff_dict[key][1]
            boxed_data = data_indices[(inputs[data_indices, current_variable] < current_cutoff_max) & (inputs[data_indices, current_variable] > current_cutoff_min)]
            data_indices = boxed_data
        return data_indices
        
        
    def CART(self, inputs, values):
        target_bin_size = 100
        cutoffs = {}
        if len(values) <= target_bin_size:
            return cutoffs
        best_variable = None
        best_cutoff = [-np.inf, np.inf]
        mean_response = np.mean(values)
        for feature in range(np.shape(inputs)[1]):
            feature_vector = inputs[:, feature]
            sorted_vector = np.unique(np.sort(feature_vector))
            feature_splits = (sorted_vector[1:] + sorted_vector[:-1]) / 2
            split = int(len(feature_splits) * 0.1)
            boxed_data = values[inputs[:, feature] > feature_splits[split]]
            if np.mean(boxed_data) > mean_response:
                mean_response = np.mean(boxed_data)
                best_cutoff = [feature_splits[split], np.inf]
                best_variable = feature
        if best_variable == None:
            print("realy exit", len(values))
            print (cutoffs)
            return cutoffs
        for i in range(np.shape(inputs)[1]):
            cutoffs[i] = [-np.inf, np.inf]
        cutoffs[best_variable] = best_cutoff
        return cutoffs

                    
            
    def add_split(self, node_number, data, values):
        cutoffs = self.CART(data, values)
        self.graph.node[node_number]['cutoffs'] = cutoffs
        self.new_nodes(node_number, 2)
        
    def new_nodes(self, parent, number):
        for i in range(number):
            self.nodes += 1
            self.graph.add_edge(parent, self.nodes)
            
    def compute_class_averages(self):
        for i in range(2, self.nodes + 1):
            parent = self.graph.predecessors(i)[0]
            if self.graph.node[parent]['cutoffs'] == {}:
                self.graph.node[i]['classval'] = self.graph.node[parent]['classval']
            else:
                node_indices = self.partition_data(i)
                if len(node_indices) == 0:
                    self.graph.node[i]['classval'] = self.graph.node[parent]['classval'] 
                else:
                    classval = self.y[node_indices].mean()
                    self.graph.node[i]['classval'] = classval
                
    def predict(self, x):
        if not self.learned:
            raise NameError('Fit model first')
        current_node = 1
        leaves = self.get_leaves()
        while current_node not in leaves:
            children = self.graph.successors(current_node)
            if self.graph.node[current_node]['cutoffs'] == None:
                return self.graph.node[current_node]['classval']
            within_box = True
            for key in self.graph.node[current_node]['cutoffs']:
                current_variable = key
                current_cutoff = self.graph.node[current_node]['cutoffs'][key]
                if x[current_variable] < self.graph.node[current_node]['cutoffs'][key][0] or x[current_variable] > self.graph.node[current_node]['cutoffs'][key][1]:
                    within_box = False
            if within_box:
                current_node = children[0]
            else:
                current_node = children[1]
        return self.graph.node[current_node]['classval']
        
    def add_layer(self):
        leaves = self.get_leaves()
        for leaf in leaves:
            print(leaf)
            data_indices = self.partition_data(leaf)
            leaf_X = self.X[data_indices, :]
            leaf_y = self.y[data_indices]
            self.add_split(leaf, leaf_X, leaf_y)
                        
                        
                        
                        

    
    
    
    
    