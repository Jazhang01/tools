import graph

class Resistor(graph.Edge):
    def __init__(self, id, resistance, node1, node2):
        super().__init__(id, node1, node2)
        self.resistance = resistance
        self.node1 = node1
        self.node2 = node2


class Capacitor(graph.Edge):
    def __init__(self, id, capacitance, node1, node2):
        super().__init__(id, node1, node2)
        self.capacitance = capacitance
        self.node1 = node1
        self.node2 = node2


class DCSource(graph.Edge):
    def __init__(self, id, voltage, node_in, node_out):
        super().__init__(id, node_in, node_out)
        self.voltage = voltage
        self.node_in = node_in
        self.node_out = node_out


class Node(graph.Vertex):
    def __init__(self, id, elements=None):
        super().__init__(id)
        self.elements = elements

