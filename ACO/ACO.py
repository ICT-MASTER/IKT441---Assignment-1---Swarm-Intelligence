import numpy as np

class ACO:

    def __init__(self, nodes, edges):

        self.nodes = nodes
        self.edges = edges

        self.MAX_COST = ACO.get_sum()

        pass


    @staticmethod
    def has_visited_all(self, ant_edges):
        """
        Check if the ant has visited all nodes
        :param ant_edges:
        :return: boolean
        """

        # Retrieve all visited_nodes
        visited_nodes = [edge.second for edge in ant_edges]

        # Determine weither all nodes has been visited
        return set(self.nodes).issubset(visited_nodes)


    def get_sum(self):
        """
        Get cost sum for all edges
        :return:
        """
        return np.sum(self.edges)