import networkx as nx
matrix = [[0, 1, 0], [1, 0 , 1], [0, 1, 0]]
G = nx.Graph(matrix)
print(G.edges)