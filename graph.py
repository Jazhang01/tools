class IDObject(object):
    def __init__(self, id):
        self.id = id

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return self.id

    def __str__(self):
        return str(self.id)

    def __repr__(self):
        return str(self.id)


class Vertex(IDObject):
    def __init__(self, id):
        super().__init__(id)


class Edge(IDObject):
    def __init__(self, id, v1=None, v2=None):
        assert isinstance(v1, Vertex), isinstance(v2, Vertex)
        super().__init__(id)
        self.v1 = v1
        self.v2 = v2


class Graph(object):
    def __init__(self):
        """ Initializes the graph """

    def vertices(self):
        """ Returns a set of vertices """

    def add_vertex(self, v):
        """ Adds a vertex 'v' to the graph """

    def remove_vertex(self, v):
        """ Removes a vertex 'v' from the graph, and corresponding edges """

    def add_edge(self, v, w):
        """ Adds an edge between vertices 'v' and 'w', if they exist """

    def remove_edge(self, v, w):
        """ Removes the edge between vertices 'v' and 'w' """


class DiGraph(Graph):
    def __init__(self):
        super().__init__()
        self.edges = {}

    def vertices(self):
        return self.edges.keys()

    def add_vertex(self, v):
        assert isinstance(v, Vertex)
        if v not in self.edges.keys():
            self.edges[v] = []

    def remove_vertex(self, v):
        assert isinstance(v, Vertex)
        for _v in self.edges.keys():
            if v in self.edges[_v]:
                self.edges[_v].remove(v)
        self.edges.pop(v)

    def add_edge(self, v, w):
        """ Adds a directed edge from vertex 'v' to 'w', if they exist """
        assert isinstance(v, Vertex), isinstance(w, Vertex)
        if v in self.edges and w in self.edges:
            self.edges[v].append(w)

    def remove_edge(self, v, w):
        """ Removes the directed edge from vertex 'v' to 'w' """
        assert isinstance(v, Vertex), isinstance(w, Vertex)
        self.edges[v].remove(w)

    def __str__(self):
        return str(self.edges)

    def __repr__(self):
        return str(self.edges)

