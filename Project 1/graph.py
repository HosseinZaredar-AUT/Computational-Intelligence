from dataclasses import dataclass, field

@dataclass
class Edge :
   src : int
   dst : int
   weight : int

@dataclass
class Graph :

    num_nodes : int
    edgelist : list
    parent : list = field(default_factory = list)
    rank : list = field(default_factory = list)

    # mst stores edges of the minimum spanning tree
    mst : list = field(default_factory = list)

    def FindParent(self, node) :

        if self.parent[node] == node :
           return node
        return self.FindParent(self.parent[node])

    def KruskalMST(self) :

        # Sort objects of an Edge class based on attribute (weight)
        self.edgelist.sort(key=lambda Edge : Edge.weight)

        self.parent = [None] * self.num_nodes
        self.rank   = [None] * self.num_nodes

        for n in range(self.num_nodes) :
            self.parent[n] = n # Every node is the parent of itself at the beginning
            self.rank[n] = 0   # Rank of every node is 0 at the beginning

        for edge in self.edgelist :
            root1 = self.FindParent(edge.src)
            root2 = self.FindParent(edge.dst)

            # Parents of the source and destination nodes are not in the same subset
            # Add the edge to the spanning tree
            if root1 != root2 :
               self.mst.append(edge)
               if self.rank[root1] < self.rank[root2] :
                  self.parent[root1] = root2
                  self.rank[root2] += 1
               else :
                  self.parent[root2] = root1
                  self.rank[root1] += 1

        
        node_set = set()
        cost = 0
        all_edges = []
        for edge in self.mst:
            node_set.add(edge.src)
            node_set.add(edge.dst)
            cost += edge.weight
            all_edges.append((edge.src, edge.dst, edge.weight))
        
        # checking if it spans all the tree
        if len(node_set) == self.num_nodes:
            return True, cost, all_edges
        else:
            return False, None, None
            