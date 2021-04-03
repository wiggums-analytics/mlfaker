"""
This is a demo of how I would approach coding up a dynamic (directed, possibly cyclic)
graph.

Because the graph has cycles, so there is no concept of a "source".
Therefore the entire graph must be initialized with state and that state evolved over
discrete time steps.

Here I'm just going to implement 2 simple examples of linear non-categorical graphs
where the nodes are just linear functions of their inputs. This framework can be
extended to include more complicated node functions that can invlove stateful internal
parameters
"""
import uuid
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd


class Node:
    """Nodes that represent a random variable in our system.

    This is a base class that can be inherited and customized.

    They can accept inputs and output a single output of fixed type.
    They also keep track of a local "history" and use that to influence the output.

    uid:
        A unique identifer for this node. Must be sortable and hashable.
    """

    uid: uuid.UUID

    def __init__(self):
        self.uid = uuid.uuid4()

    def interact(self, *args, **kwargs) -> Any:
        # TODO: implement this
        raise NotImplementedError()


class DynamicGraph:
    """Dynamic graphs are directed, possibly cyclic, weighted, graphs that represent
    a dynamical system. They have an `evolve` method can be called with an accompanying
    state vector to propagate the state vector 1 unit in time. Depending on the subclass
    used, state vectors can have data of varying types.

    An assumption is that a node's state can only be affected by the state of it's
    parent nodes at the previous timestep. ie: "non-local" time interactions are not
    supported at the graph level. However, a node can keep track of it's own "history"
    and use that to influence it's output function
    """

    adj: Dict[uuid.UUID, Dict[uuid.UUID, float]]
    nodes: Dict[uuid.UUID, Node]

    def __init__(self):
        self.adj = {}
        self.nodes = {}

    def add_node(self, node: Node) -> None:
        self.nodes[node.uid] = node

    def add_edge(self, u: Node, v: Node, w: float = 1.0) -> None:
        self.add_node(u)
        self.add_node(v)
        if u in self.adj:
            self.adj[u.uid][v.uid] = w
        else:
            self.adj[u.uid] = {v.uid: w}

    def evolve(self, state: pd.Series) -> pd.Series:
        # TODO: implement this along with Node.interact(...)
        # This will require an API to inspect the interaction
        # signatures to make sure the arguments are satified
        # as well as the types match
        raise NotImplementedError()


class LinearGraph(DynamicGraph):
    """Assumes each node's fucntion output is a linear sum of it's inputs.
    Only works with numerical data. Ignores any interaction functions on
    the nodes.

    These assumptions allows us to recast the problem
    as matrix multiplication to speed things up.

    M:
        2D matrix of weights that annotate the edges
        of the graph. Initializes to `None` to allow
        for a dynamic api to build up the graph and
        gets constructed on a call to `evolve`.
    N:
        Index lookup in M for a given node
    """

    M: Optional[np.array]
    N: Optional[Dict[uuid.UUID, int]]

    def __init__(self):
        super().__init__()
        self._reset()

    def _reset(self):
        self.M = None
        self.N = None

    def add_node(self, node: Node) -> None:
        self._reset()
        super().add_node(node)

    def add_edge(self, u: Node, v: Node, w: float = 1.0) -> None:
        self._reset()
        super().add_edge(u, v, w)

    def _build(self) -> None:
        n = len(self.nodes)
        self.N = {uid: i for i, uid in enumerate(sorted(list(self.nodes)))}
        self.M = np.zeros(shape=(n, n))
        for i, u in enumerate(self.N):
            for v, w in self.adj[u].items():
                self.M[i][self.N[v]] = w

    def evolve(self, state: pd.Series) -> pd.Series:
        if self.M is None:
            self._build()
        return pd.Series(np.matmul(self.M, state.values), index=state.index)


class HeatFlowGraph(LinearGraph):
    """Graph that models the linear heat flow equation: df/dt = - grad(f)

    M:
        The directed graph laplacian
    """

    def _build(self) -> None:
        n = len(self.nodes)
        self.N = {uid: i for i, uid in enumerate(sorted(list(self.nodes)))}
        self.M = np.zeros(shape=(n, n))
        for i, u in enumerate(self.N):
            for v, w in self.adj[u].items():
                self.M[i][self.N[v]] = w
        degrees = np.sum(self.M, axis=1)
        degrees = np.where(degrees > 0, degrees, np.ones(n))
        self.M -= np.diag(degrees)

    def evolve(self, state: pd.Series) -> pd.Series:
        delta = super().evolve(state)
        return state + delta / len(self.N)


class Simulation:
    """A class that accepts a DynamicGraph and an associated initial state vector
    and evolves it in time.

    """

    graph: DynamicGraph
    state: pd.Series

    def __init__(self, graph: DynamicGraph, state: pd.Series):
        self.graph = graph
        self._validate_state(state)
        self.state = state

    def _validate_state(self, state: pd.Series) -> None:
        nodes = set(self.graph.nodes.keys())
        idx = set(state.index)
        if not nodes == idx:
            raise ValueError(
                f"State vector index is not aligned with graph nodes: {nodes ^ idx}"
            )
        idx_list = state.index.to_list()
        if idx_list != sorted(idx_list):
            raise ValueError("State vector index must be sorted ascending by node uid")

    def run(self, steps: int) -> None:
        if not steps > 0:
            raise ValueError("Steps must be > 0")
        for _ in range(steps):
            self.state = self.graph.evolve(self.state)


# DEMO
if __name__ == "__main__":
    # First, do a simple 2 node linear graph
    print("\n\n 2 state linear system: oscillator \n\n")

    G = LinearGraph()
    a, b = Node(), Node()
    G.add_edge(a, b)
    G.add_edge(b, a)

    initial_state = pd.Series([1.0, 0.0], index=[a.uid, b.uid]).sort_index()
    print("Initial state is:")
    print(initial_state)

    sim = Simulation(G, initial_state)
    # evolve the graph and let the state oscillate...
    for _ in range(4):
        sim.run(1)
        print("New state is:")
        print(sim.state)

    print("\n\n~~~~~\n\n")

    # Changing the initial state alters evolution
    # Eigenstates are stable vs time:
    print("\n\n 2 state linear system: Eigenstate \n\n")
    initial_state = pd.Series([1.0, 1.0], index=[a.uid, b.uid]).sort_index()
    print("Initial state is:")
    print(initial_state)

    sim = Simulation(G, initial_state)
    for _ in range(4):
        sim.run(1)
        print("New state is:")
        print(sim.state)

    print("\n\n~~~~~\n\n")

    # Now let's look at the heat flow graph
    print("\n\n 2 state heat flow system \n\n")
    G = HeatFlowGraph()
    a, b, c = Node(), Node(), Node()
    G.add_edge(a, b)
    G.add_edge(b, a)
    G.add_edge(a, c)
    G.add_edge(c, a)

    # we'll start with one "hot" node
    initial_state = pd.Series(
        [10.0, 0.0, 0.0], index=[a.uid, b.uid, c.uid]
    ).sort_index()
    print("Initial state is:")
    print(initial_state)
    sim = Simulation(G, initial_state)
    # the graph should "thermalize" with time
    for _ in range(4):
        sim.run(1)
        print("New state is:")
        print(sim.state)
