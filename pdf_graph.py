import pymupdf
import networkx as nx
import matplotlib.pyplot as plt
import math
import pandas as pd
from sklearn.decomposition import PCA
from itertools import combinations



class PDFGraph(nx.MultiDiGraph):
    # This takes an 'origin' (int, int), and returns (int, -1*int):
    _invert_y = lambda origin:\
        (origin[0], -1*origin[1])
    
    # This returns a bool, True if the given node is used in the list of edegs:
    _in_edges = lambda edges, node: \
        (node[0] in [edge[0] for edge in edges]) or (node[0] in [edge[1] for edge in edges])

    def __init__(self, text_elements:list=None, edges=None, nodes=None, max_distance:float=50.0, min_distance:float=0.0, **attr):
        if text_elements != None:
            # print(f"creating {type(self)} with text_elements")
            self._base_constructor(text_elements, max_distance, min_distance, **attr)

        elif edges != None and nodes != None:
            # print(f"creating {type(self)} with edges and nodes")
            self._sub_graph_constructor(edges, nodes, **attr)
        
        elif text_elements == None and edges == None and nodes == None:
            print(f"({type(self)}) Error: Must provide one of (text_elements) or (edges and nodes)")
            exit()
        
        else:
            print(f"({type(self)}) Error: Must provide both (edges and nodes)")

    def _sub_graph_constructor(self, edges, nodes, **attr):
        super().__init__(None, **attr)

        self.add_nodes_from(nodes)
        self.add_edges_from(edges)

        # Rebuild the pos and labels attributes for use in draw():
        self.pos = {}
        self.labels = {}
        for node in self.nodes(data=True):
            self.pos[node[0]] = PDFGraph._invert_y(node[1]["origin"])
            self.labels[node[0]] = node[1]["text"]

    def _base_constructor(self, text_elements:list, max_distance:float=50.0, min_distance:float=0.0, **attr):
        super().__init__(None, **attr)

        self.pos = {}
        self.labels = {}
        
        # Iterate Over Text Elements in the list (page):
        element_index = 0
        for text_element in text_elements:
            # Add text element as node:
            self.add_node(node_for_adding=element_index, **text_element)
            
            # Add position and text to pos and labels dicts:
            self.pos[element_index] =PDFGraph._invert_y(text_element["origin"])
            self.labels[element_index] = text_element["text"]

            # Increment element index by 1:
            element_index += 1


        # Iterate over nodes and create edges:
        for center_node in self.nodes(data=True):
            for neighbor_node in self.nodes(data=True):
                if center_node != neighbor_node:
                    # calculate distance between center_node and neighbor_node:
                    distance = math.dist(center_node[1]["origin"], neighbor_node[1]["origin"])

                    # Check if distance between center_node and neighbor_node is within bounds:
                    if distance < min_distance or distance > max_distance: continue


                    angle = math.degrees(math.atan2(
                            center_node[1]["origin"][0]-neighbor_node[1]["origin"][0],
                            center_node[1]["origin"][1]-neighbor_node[1]["origin"][1]
                    ))      
                    
                    # Add edge to the graph:
                    self.add_edge(
                        center_node[0],
                        neighbor_node[0],
                        
                        # '**attr's:
                        distance=distance,
                        angle=angle
                    )


    def draw(self, **kwargs):
        nx.draw(
            G=self,
            pos=self.pos,
            labels=self.labels,
            **kwargs
        )

    def get_all_subgraphs(self, target_node, max_nodes:int=-1, min_nodes:int=1) -> list['PDFGraph']:
        if min_nodes > max_nodes and max_nodes != -1:
            raise Warning("(PDFGraph) Wraning: min_nodes cannot be greater than max_nodes")
        
        # Get all neighbors of target_node:
        all_neighbors = list(self.neighbors(target_node))

        # Set min and max node counts:
        if max_nodes < 1: max_nodes = len(all_neighbors)+1
        else: max_nodes = min(max_nodes, len(all_neighbors)+1)
        min_nodes = max(1, min_nodes)

        # Get all combinations of target_node and neighbors of target_node:
        range_min = max(min_nodes-1, 1)
        range_max = min(max_nodes, len(all_neighbors)+1)
        combos = [[target_node] + list(combo) for n in range(range_min, range_max) for combo in combinations(all_neighbors, n)]

        # If 1 node is allows, add target_node on its own as a combo: 
        if min_nodes == 1:
            combos = [[target_node]] + combos
        
        # Iterate over combos to create subgraphs:
        sub_graphs = []
        for combo in combos:
            edges = [edge for edge in self.edges(data=True, nbunch=combo) if edge[0] == target_node and edge[1] in combo]
            nodes = [node for node in self.nodes(data=True) if node[0] in combo]

            sub_graphs.append(PDFGraph(edges=edges, nodes=nodes))

        # Return list of subgraphs:
        return sub_graphs

    def subgraph(self, target_node, n:int=-1) -> 'PDFGraph':
        # Get all edges between neighbors and target_node:
        edges = [
            edge for edge in self.edges(data=True) if \
            # (edge[0] in self.neighbors(target_node) and edge[1] == target_node) or\
            (edge[1] in self.neighbors(target_node) and edge[0] == target_node)
        ]

        # Sort the list of edges by the 'distance' **attr of the edge and slice from 0-n:
        edges.sort(key=lambda x: x[2]['distance'])

        # Slice the now ordered list of edges from 0-n (if n > 0):
        if n > 0: edges = edges[:n]

        # Get all neighbors of the target_node that are in the list of edges:
        neighbors = [
            node for node in self.nodes(data=True) if 
            (node[0] in self.neighbors(target_node) or node[0] == target_node) and\
            PDFGraph._in_edges(edges=edges, node=node)                              
        ]
        
        # If no neighbors or edges exist, create a PDFGraph with a single text element (target_node):
        if edges == [] or neighbors == []:
            return PDFGraph(text_elements=[self.nodes(data=True)[target_node]])

        # If neighbors exist, create a PDFGraph with lists of edges and nodes:
        return PDFGraph(edges=edges, nodes=neighbors)

    def flatten(self, n_min:int=1, n_max:int=-1) -> pd.DataFrame:
        # Create list to hold tabular data as dicts:
        tabular_dicts_list = [] 
        
        # Iterate over all nodes in page graph:
        for target_node in self.nodes:
            
            # Get SubGraph (One 'element' within the doc to classify)
            for sub_graph in self.get_all_subgraphs(target_node=target_node, min_nodes=n_min, max_nodes=n_max):
                
                line_dict = {
                    "num_nodes": len(sub_graph.nodes),
                    "target_node": target_node,
                }

                # Add all attributes from the target_node to the dict:
                for attribute in self.nodes(data=True)[target_node].keys():
                    if attribute not in line_dict:
                        line_dict[attribute] = [self.nodes(data=True)[target_node][attribute]]
                    else:
                        line_dict[attribute].append(self.nodes(data=True)[target_node][attribute])

                # Iterate over nodes in SubGraph:
                for node in sub_graph.nodes(data=True):
                    if node[1] == self.nodes(data=True)[target_node]: continue
                    
                    # Add all attributes from this node to the dict:
                    for attribute in node[1].keys():
                        if attribute not in line_dict:
                            line_dict[attribute] = [node[1][attribute]]
                        else:
                            line_dict[attribute].append(node[1][attribute])
                    
                    # Add all attributes from the edge between this node and target_node to the dict:
                    edge_data = self.get_edge_data(target_node, node[0])
                    if edge_data != None:
                        for attribute in edge_data[0].keys():
                            if attribute not in line_dict:
                                line_dict[attribute] = [edge_data[0][attribute]]
                            else:
                                line_dict[attribute].append(edge_data[0][attribute])

                # Append line_dict to list of tabular dicts:
                tabular_dicts_list.append(line_dict)

        return pd.DataFrame(tabular_dicts_list)

    @staticmethod
    def gen_page_graphs(file_path:str, **attr) -> list['PDFGraph']:
        # Open document:
        with pymupdf.open(file_path) as doc:
            
            # Iterate over Pages in Document:
            graphs = []
            for page in doc:
                blocks = page.get_text("dict")["blocks"]

                # Iterate Over Blocks in Page:
                text_elements = [] # Temp list of dicts for the block
                for block in blocks:
                    if "lines" in block.keys():

                        # Iterate Over Lines in Block:
                        for line in block["lines"]:
                            if "spans" in line.keys():

                                # Add each span to the temp list of text_elements
                                text_elements += [span for span in line["spans"]]

                # Create a PDFGraph for this page and append it to the list of graphs
                graphs.append(PDFGraph(text_elements, **attr))
        
        return graphs



# Create DataFrame from a dense column (column of lists)
expand_dense_column = lambda dense_column, default=0.0: pd.DataFrame(
    # If an element of the column isn't a list, replace w/ an empty list:
    dense_column.apply(lambda x: [] if type(x) != list else x).values.tolist()
    # Fill emprty cells in the DataFrame with 0.0
    ).fillna(default)

def reduce_tabular_data(tabular_data:pd.DataFrame) -> pd.DataFrame:
    # Create reduced_data DataFrame and add 'num_nodes' column to it:
    reduced_data = pd.DataFrame(columns=['num_nodes', 'target_node', 'size', 'distance', 'angle', 'text'])
    reduced_data['num_nodes'] = tabular_data['num_nodes']
    reduced_data['target_node'] = tabular_data['target_node']

    # Create and Fit PCAs to the columns that need reduction:
    pca_size = PCA(n_components=1)
    pca_size.fit(expand_dense_column(tabular_data["size"]))

    pca_distance= PCA(n_components=1)
    pca_distance.fit(expand_dense_column(tabular_data["distance"]))
    
    pca_angle = PCA(n_components=1)
    pca_angle.fit(expand_dense_column(tabular_data["angle"]))
    
    # Reduce the columns using the PCAs:
    reduced_data['size'] = pca_size.transform(expand_dense_column(tabular_data['size']))
    reduced_data['distance'] = pca_distance.transform(expand_dense_column(tabular_data['distance']))
    reduced_data['angle'] = pca_angle.transform(expand_dense_column(tabular_data['angle']))

    # Use string concatination to joing list elements from 'text' column:
    reduced_data['text'] = tabular_data['text'].fillna("").apply(
        lambda elements: "|".join([str(element) for element in elements])
    )

    # Return the reduced data:
    return reduced_data


if __name__ == "__main__":
    pdf_file_path = r""
    graphs = PDFGraph.gen_page_graphs(pdf_file_path, max_distance=40)

    # Draw and save a PNG of the entire page graph:
    graphs[0].draw(
        font_size=6,
        node_size=50,
    )
    plt.savefig("PDFGraph.png")
    plt.close()

    # Pick some nodes and save PNGs of their subgraphs:
    nodes = [3, 4, 8]
    for node in nodes:
        target_node = list(graphs[0].nodes(data=False))[node]
        sub_g = graphs[0].subgraph(target_node, n=6)
        sub_g.draw(
            font_size=6,
            node_size=50,
        )
        plt.savefig(f"PDFGraph - subgraph {nodes.index(node)}.png")
        plt.close()

    # Create and save flattened version of page graph:
    flattened_data = graphs[0].flatten(n_min=2)
    flattened_data.to_excel("tabular_graphs.xlsx")

    # Crate and save reduced version of the flattened data:
    reduced_data = reduce_tabular_data(flattened_data)
    reduced_data.to_excel("reduced_tabular_graphs.xlsx")

                            
                            