# Copyright (c) 2021, Paul Almasan [^1]
#
# [^1]: Universitat Politècnica de Catalunya, Computer Architecture
#     department, Barcelona, Spain. Email: felician.paul.almasan@upc.edu

import gym
import numpy as np
import networkx as nx
import random
from gym import error, spaces, utils
from random import choice
import pylab
import json 
import gc
import matplotlib.pyplot as plt

def create_geant2_graph():
    Gbase = nx.Graph()
    Gbase.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
    Gbase.add_edges_from(
        [(0, 1), (0, 2), (1, 3), (1, 6), (1, 9), (2, 3), (2, 4), (3, 6), (4, 7), (5, 3),
         (5, 8), (6, 9), (6, 8), (7, 11), (7, 8), (8, 11), (8, 20), (8, 17), (8, 18), (8, 12),
         (9, 10), (9, 13), (9, 12), (10, 13), (11, 20), (11, 14), (12, 13), (12,19), (12,21),
         (14, 15), (15, 16), (16, 17), (17,18), (18,21), (19, 23), (21,22), (22, 23)])

    return Gbase

def create_nsfnet_graph():
    Gbase = nx.Graph()
    Gbase.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    Gbase.add_edges_from(
        [(0, 1), (0, 2), (0, 3), (1, 2), (1, 7), (2, 5), (3, 8), (3, 4), (4, 5), (4, 6), (5, 12), (5, 13),
         (6, 7), (7, 10), (8, 9), (8, 11), (9, 10), (9, 12), (10, 11), (10, 13), (11, 12)])

    return Gbase

def create_small_top():
    Gbase = nx.Graph()
    Gbase.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8])
    Gbase.add_edges_from(
        [(0, 1), (0, 2), (0, 3), (1, 2), (1, 7), (2, 5), (3, 8), (3, 4), (4, 5), (4, 6), (5, 0),
         (6, 7), (6, 8), (7, 8), (8, 0), (8, 6), (3, 2), (5, 3)])

    return Gbase

def create_gbn_graph():
    Gbase = nx.Graph()
    Gbase.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    Gbase.add_edges_from(
        [(0, 2), (0, 8), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4), (3, 9), (4, 8), (4, 10), (4, 9),
         (5, 6), (5, 8), (6, 7), (7, 8), (7, 10), (9, 10), (9, 12), (10, 11), (10, 12), (11, 13),
         (12, 14), (12, 16), (13, 14), (14, 15), (15, 16)])

    return Gbase

def generate_nx_graph(topology):
    """
    Generate graphs for training with the same topology.
    """
    if topology == 0:
        G = create_nsfnet_graph()
    elif topology == 1:
        G = create_geant2_graph()
    elif topology == 2:
        G = create_small_top()
    else:
        G = create_gbn_graph()

    # nx.draw(G, with_labels=True)
    # plt.show()
    # plt.clf()

    # Node id counter
    incId = 1
    # Put all distance weights into edge attributes.
    for i, j in G.edges():
        G.get_edge_data(i, j)['edgeId'] = incId
        G.get_edge_data(i, j)['betweenness'] = 0
        G.get_edge_data(i, j)['numsp'] = 0  # Indicates the number of shortest paths going through the link
        # We set the edges capacities to 200
        G.get_edge_data(i, j)["capacity"] = float(200)
        G.get_edge_data(i, j)['bw_allocated'] = 0
        incId = incId + 1

    return G


def compute_link_betweenness(g, k):
    n = len(g.nodes())
    betw = []
    for i, j in g.edges():
        # we add a very small number to avoid division by zero
        b_link = g.get_edge_data(i, j)['numsp'] / ((2.0 * n * (n - 1) * k) + 0.00000001)
        g.get_edge_data(i, j)['betweenness'] = b_link
        betw.append(b_link)

    mu_bet = np.mean(betw)
    std_bet = np.std(betw)
    return mu_bet, std_bet

class Env1(gym.Env):
    """
    Description:
    The self.graph_state stores the relevant features for the GNN model

    self.graph_state[:][0] = CAPACITY
    self.graph_state[:][1] = BW_ALLOCATED
  """
    def __init__(self):
        self.graph = None
        self.initial_state = None
        self.source = None
        self.destination = None
        self.demand = None
        self.graph_state = None
        self.diameter = None

        # Nx Graph where the nodes have features. Betweenness is allways normalized.
        # The other features are "raw" and are being normalized before prediction
        self.first = None
        self.firstTrueSize = None
        self.second = None
        self.between_feature = None

        # Mean and standard deviation of link betweenness
        self.mu_bet = None
        self.std_bet = None

        self.max_demand = 0
        self.K = 4
        self.listofDemands = None
        self.nodes = None
        self.ordered_edges = None
        self.edgesDict = None
        self.numNodes = None
        self.numEdges = None

        self.state = None
        self.episode_over = True
        self.reward = 0
        self.allPaths = dict()

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def num_shortest_path(self, topology):
        self.diameter = nx.diameter(self.graph)

        # Iterate over all node1,node2 pairs from the graph
        for n1 in self.graph:
            for n2 in self.graph:
                if (n1 != n2):
                    # Check if we added the element of the matrix
                    if str(n1)+':'+str(n2) not in self.allPaths:
                        self.allPaths[str(n1)+':'+str(n2)] = []
                    
                    # First we compute the shortest paths taking into account the diameter
                    # This is because large topologies might take too long to compute all shortest paths 
                    [self.allPaths[str(n1)+':'+str(n2)].append(p) for p in nx.all_simple_paths(self.graph, source=n1, target=n2, cutoff=self.diameter*2)]

                    # We take all the paths from n1 to n2 and we order them according to the path length
                    self.allPaths[str(n1)+':'+str(n2)] = sorted(self.allPaths[str(n1)+':'+str(n2)], key=lambda item: (len(item), item))

                    path = 0
                    while path < self.K and path < len(self.allPaths[str(n1)+':'+str(n2)]):
                        currentPath = self.allPaths[str(n1)+':'+str(n2)][path]
                        i = 0
                        j = 1

                        # Iterate over pairs of nodes increase the number of sp
                        while (j < len(currentPath)):
                            self.graph.get_edge_data(currentPath[i], currentPath[j])['numsp'] = \
                                self.graph.get_edge_data(currentPath[i], currentPath[j])['numsp'] + 1
                            i = i + 1
                            j = j + 1

                        path = path + 1

                    # Remove paths not needed
                    del self.allPaths[str(n1)+':'+str(n2)][path:len(self.allPaths[str(n1)+':'+str(n2)])]
                    gc.collect()


    def _first_second_between(self):
        self.first = list()
        self.second = list()

        # For each edge we iterate over all neighbour edges
        for i, j in self.ordered_edges:
            neighbour_edges = self.graph.edges(i)

            for m, n in neighbour_edges:
                if ((i != m or j != n) and (i != n or j != m)):
                    self.first.append(self.edgesDict[str(i) +':'+ str(j)])
                    self.second.append(self.edgesDict[str(m) +':'+ str(n)])

            neighbour_edges = self.graph.edges(j)
            for m, n in neighbour_edges:
                if ((i != m or j != n) and (i != n or j != m)):
                    self.first.append(self.edgesDict[str(i) +':'+ str(j)])
                    self.second.append(self.edgesDict[str(m) +':'+ str(n)])


    def generate_environment(self, topology, listofdemands):
        # The nx graph will only be used to convert graph from edges to nodes
        self.graph = generate_nx_graph(topology)

        self.listofDemands = listofdemands

        self.max_demand = np.amax(self.listofDemands)

        # Compute number of shortest paths per link. This will be used for the betweenness
        self.num_shortest_path(topology)

        # Compute the betweenness value for each link
        self.mu_bet, self.std_bet = compute_link_betweenness(self.graph, self.K)

        self.edgesDict = dict()

        some_edges_1 = [tuple(sorted(edge)) for edge in self.graph.edges()]
        self.ordered_edges = sorted(some_edges_1)

        self.numNodes = len(self.graph.nodes())
        self.numEdges = len(self.graph.edges())

        self.graph_state = np.zeros((self.numEdges, 2))
        self.between_feature = np.zeros(self.numEdges)

        position = 0
        for edge in self.ordered_edges:
            i = edge[0]
            j = edge[1]
            self.edgesDict[str(i)+':'+str(j)] = position
            self.edgesDict[str(j)+':'+str(i)] = position
            betweenness = (self.graph.get_edge_data(i, j)['betweenness'] - self.mu_bet) / self.std_bet
            self.graph.get_edge_data(i, j)['betweenness'] = betweenness
            self.graph_state[position][0] = self.graph.get_edge_data(i, j)["capacity"]
            self.between_feature[position] = self.graph.get_edge_data(i, j)['betweenness']
            position = position + 1

        self.initial_state = np.copy(self.graph_state)

        self._first_second_between()

        self.firstTrueSize = len(self.first)

        # We create the list of nodes ids to pick randomly from them
        self.nodes = list(range(0,self.numNodes))

    def make_step(self, state, action, demand, source, destination):
        self.graph_state = np.copy(state)
        self.episode_over = True
        self.reward = 0

        i = 0
        j = 1
        currentPath = self.allPaths[str(source) +':'+ str(destination)][action]

        # Once we pick the action, we decrease the total edge capacity from the edges
        # from the allocated path (action path)
        while (j < len(currentPath)):
            self.graph_state[self.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][0] -= demand
            if self.graph_state[self.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][0] < 0:
                # FINISH IF LINKS CAPACITY <0
                return self.graph_state, self.reward, self.episode_over, self.demand, self.source, self.destination 
            i = i + 1
            j = j + 1

        # Leave the bw_allocated back to 0
        self.graph_state[:,1] = 0

        # Reward is the allocated demand or 0 otherwise (end of episode)
        # We normalize the demand to don't have extremely large values
        self.reward = demand/self.max_demand
        self.episode_over = False

        self.demand = random.choice(self.listofDemands)
        self.source = random.choice(self.nodes)

        # We pick a pair of SOURCE,DESTINATION different nodes
        while True:
            self.destination = random.choice(self.nodes)
            if self.destination != self.source:
                break

        return self.graph_state, self.reward, self.episode_over, self.demand, self.source, self.destination

    def reset(self):
        """
        Reset environment and setup for new episode. Generate new demand and pair source, destination.

        Returns:
            initial state of reset environment, a new demand and a source and destination node
        """
        self.graph_state = np.copy(self.initial_state)
        self.demand = random.choice(self.listofDemands)
        self.source = random.choice(self.nodes)

        # We pick a pair of SOURCE,DESTINATION different nodes
        while True:
            self.destination = random.choice(self.nodes)
            if self.destination != self.source:
                break

        return self.graph_state, self.demand, self.source, self.destination
    
    def eval_sap_reset(self, demand, source, destination):
        """
        Reset environment and setup for new episode. This function is used in the "evaluate_DQN.py" script.
        """
        self.graph_state = np.copy(self.initial_state)
        self.demand = demand
        self.source = source
        self.destination = destination

        return self.graph_state