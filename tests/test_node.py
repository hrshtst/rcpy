from rebasicspy.node import Node


class TestNode:
    def test_init(self):
        node = Node()
        assert node.name == "node_0"

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


class DerivedNode(Node):
    default_name = "derived"

    def __init__(self):
        super().__init__()


def test_derived_node():
    n1 = Node()
    n2 = Node()
    d1 = DerivedNode()
    d2 = DerivedNode()

    assert n1.name == "node_0"
    assert n2.name == "node_1"
    assert d1.name == "derived_0"
    assert d2.name == "derived_1"
