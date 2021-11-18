from graph import Graph, DiGraph, Vertex

g = DiGraph()
v1 = Vertex(5)
v2 = Vertex(6)
v3 = Vertex(7)
g.add_vertex(v1)
g.add_vertex(v2)
g.add_vertex(v3)
g.add_edge(v1, v2)
g.add_edge(v2, v3)
g.add_edge(v3, v1)
g.remove_vertex(v1)
print(g)
