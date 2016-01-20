

class ACO:



    def __init__(self, nodes, edges):

        self.nodes = nodes
        self.edges = edges

        self.MAX_COST = ACO.get_sum(edges)

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


    def get_sum(edges):
        """
        Get cost sum for all edges
        :return:
        """
        return sum(e.cost for e in edges)