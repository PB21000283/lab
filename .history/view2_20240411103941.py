import networkx as nx

# 创建一个图
G = nx.Graph()
# 添加边
G.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4), (4, 5)])
# 可视化原图
nx.draw(G, with_labels=True, node_color='lightblue')

# 使用最小生成树简化图
T = nx.minimum_spanning_tree(G)
# 可视化简化后的图
nx.draw(T, with_labels=True, node_color='lightgreen')