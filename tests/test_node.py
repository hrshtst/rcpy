from rebasicspy.node import Model, Node


class TestNode:
    def test_init(self):
        node = Node()
        assert node.name == "node_0"

    def test_repr(self):
        node = Node()
        assert repr(node) == "Node(node_0, 0)"

    def test_str(self):
        node = Node()
        assert str(node) == "node_0"

    def test_auto_assigned_id(self):
        n1 = Node()
        assert n1.id == 0

        n2 = Node()
        assert n2.id == 1

    def test_auto_assigned_name(self):
        n1 = Node()
        assert n1.name == "node_0"

        n2 = Node()
        assert n2.name == "node_1"

    def test_set_name(self):
        n = Node()
        n.name = "super_node"
        assert n.name == "super_node"

    def test_iter(self):
        n1, n2, n3 = Node(), Node(), Node()
        expected = [n1.name, n2.name, n3.name]
        for n in Node:
            assert n.name in expected
            expected.remove(n.name)
        assert len(expected) == 0

    def test_find(self):
        nodes = [Node() for _ in range(5)]
        found = Node.find("node_2")
        assert found is nodes[2]

    def test_find_not_found(self):
        _ = [Node() for _ in range(5)]
        found = Node.find("non_exist")
        assert found is None


class DerivedNode(Node):
    default_name = "derived"

    def __init__(self):
        super().__init__()


def test_derived_node():
    d1 = DerivedNode()
    d2 = DerivedNode()

    assert d1.name == "derived_0"
    assert d2.name == "derived_1"


class TestModel:
    def test_init(self):
        model = Model()
        assert model.graph == {}

    def test_add(self):
        n = Node()
        m = Model()
        m.add(n)
        assert n in m.graph
        assert m.graph[n] == []

    def test_add_exists_already(self):
        n = Node()
        m = Model()
        m.add(n)
        m.add(n)
        assert n in m.graph
        assert m.graph[n] == []

    def test_connect_1(self):
        n1 = Node()
        n2 = Node()
        m = Model()
        m.connect(n1, n2)
        assert m.graph[n1] == [n2]
        assert m.graph[n2] == []

    def test_connect_2(self):
        n1, n2, n3 = Node(), Node(), Node()
        m = Model()
        m.connect(n1, n2)
        m.connect(n2, n3)
        assert m.graph[n1] == [n2]
        assert m.graph[n2] == [n3]
        assert m.graph[n3] == []

    def test_connect_3(self):
        n1, n2, n3 = Node(), Node(), Node()
        m = Model()
        m.connect(n1, n2)
        m.connect(n1, n3)
        assert m.graph[n1] == [n2, n3]
        assert m.graph[n2] == []
        assert m.graph[n3] == []

    def test_connect_exists_already(self):
        n1, n2, n3 = Node(), Node(), Node()
        m = Model()
        m.connect(n1, n2)
        m.connect(n1, n3)
        assert m.graph[n1] == [n2, n3]
        assert m.graph[n2] == []
        assert m.graph[n3] == []
        m.connect(n1, n2)
        assert m.graph[n1] == [n2, n3]
        assert m.graph[n2] == []
        assert m.graph[n3] == []

    def test_no_change_in_graph_when_connect_and_add(self):
        n1 = Node()
        n2 = Node()
        m = Model()
        m.connect(n1, n2)
        assert m.graph[n1] == [n2]
        assert m.graph[n2] == []
        m.add(n1)
        assert m.graph[n1] == [n2]
        assert m.graph[n2] == []
