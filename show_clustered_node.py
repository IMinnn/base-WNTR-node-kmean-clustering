import networkx as nx
import wntr
import csv
import plotly
import numpy as np
import pandas as pd

"""
author:He min
time: 2020.12.05
mail:hemincug@foxmail.com
"""


# 整理聚类结果成一个字典， 类别：[ID1,ID2,...]
def statistics_cluster(cluster_file):
    """
    将整理聚类结果成一个字典
    :param cluster_file:聚类结果的CSV文件。文件中第一列为节点ID，第三列为聚类的类别编号
    :return: 返回字典（类别编号：节点ID1，节点ID2...）
    """

    # 获取聚类结果数据
    clustering_result = np.loadtxt(cluster_file, dtype=str, delimiter=',')
    clustering_result = clustering_result[1:, :]

    nodelist = clustering_result[:, 0]
    node_cluster_type = clustering_result[:, 3]
    node_cluster_type = list(map(int, node_cluster_type))
    node_cluster_type = np.array(node_cluster_type)

    # 字典  ID：类别
    node_attribute = pd.Series(node_cluster_type, nodelist)
    node_attribute = dict(node_attribute)

    result_dict = {}
    temp_value = None
    for node, value in node_attribute.items():
        if value == temp_value:
            result_dict[value] += [node]
        else:
            temp_value = value
            result_dict[value] = [node]

    return result_dict


# 将聚类结果添加至图节点的属性中
def add_node_cluster(G, clustered_node_file):
    """
    将聚类结果添加至图节点的属性‘cluster’中
    :param G: 由wntr水网络模型转成是networkx图
    :param clustered_node_file: 聚类结果的CSV文件。文件中第一列为节点ID，第三列为聚类的类别编号
    :return: 返回修改后的包含的cluster属性的networkX图
    """

    # 获取聚类结果数据
    clustering_result = np.loadtxt(clustered_node_file, dtype=str, delimiter=',')
    clustering_result = clustering_result[1:, :]

    nodelist = clustering_result[:, 0]

    node_cluster_type = clustering_result[:, 3]
    node_cluster_type = list(map(int, node_cluster_type))
    node_cluster_type = np.array(node_cluster_type)

    # 字典  ID：类别
    node_attribute = pd.Series(node_cluster_type, nodelist)
    node_attribute = dict(node_attribute)

    # 添加节点类别属性
    for node in G.nodes():
        if node not in node_attribute:
            G.nodes[node]['cluster'] = "-1"
        else:
            G.nodes[node]['cluster'] = node_attribute[node]

    return G


# 修改泵节点type属性
def modify_pump_type(G, clustered_node_file):
    """
    修改泵节点type属性，由'Junction'改为‘Pump’
    :param G: networkx图
    :param clustered_node_file: 聚类结果文件
    :return: 修正后的networkx图
    """
    clustering_result = np.loadtxt(clustered_node_file, dtype=str, delimiter=',')
    clustering_result = clustering_result[1:, :]

    nodelist = clustering_result[:, 0]

    node_cluster_type = clustering_result[:, 3]
    node_cluster_type = list(map(int, node_cluster_type))
    node_cluster_type = np.array(node_cluster_type)

    # 字典  ID：类别
    node_attribute = pd.Series(node_cluster_type, nodelist)
    node_attribute = dict(node_attribute)

    for node in G.nodes():
        if node not in node_attribute and G.nodes[node]['type'] == "Junction":
            G.nodes[node]['type'] = "Pump"
    return G


# 检索分割聚类情况
def check_breakcluster(G, cluster_dict):
    """
    检查聚类异常的节点，并修正，返回修正后的图
    :param G: networkx图，包含cluster属性
    :param cluster_dict: 由聚类结果整理后的字典 类别ID：节点ID1，节点ID2...
    :return: 修正后的networkx图
    """
    for node in G.nodes():
        if G.degree[node] == 1 and G.nodes[node]['type'] == 'Junction':
            neigs = G[node]
            for key in neigs:
                neig = key
            node_cluster = G.nodes[node]['cluster']
            neig_cluster = G.nodes[neig]['cluster']
            if node_cluster != neig_cluster and len(cluster_dict[node_cluster]) > 1:
                G.nodes[node]['cluster'] = neig_cluster
                sup = 0
                sop = sup
    return G


# 输出修正后的聚类结果
def out_modified_cluster(G, filename):

    dict = {}
    for node in G.nodes():
        cluster = G.nodes[node]['cluster']
        if cluster in dict:
            dict[cluster] += [node]
        else:
            dict[cluster] = [node]

    with open(filename, mode='w', newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['ID', 'Modified_cluster_id'])
        for key in dict:
            for node in dict[key]:
                writer.writerow([node, key])


# 显示聚类结果，可交互
def show_inter(G, filename, cluster_number = None):
    """
    可视化修正后的聚类结果
    :param G: 纠正后的networkx图
    :param filename: 生成的可视化结果.html文件
    :return:
    """
    link_width = 1
    node_labels = True
    node_cmap = 'Jet'

    # Create edge trace
    edge_trace = plotly.graph_objs.Scatter(x=[], y=[], text=[], hoverinfo='text', mode='lines',
                                           line=dict(color='#888', width=link_width))

    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])

    # Create node trace
    node_trace = plotly.graph_objs.Scatter(x=[], y=[], text=[], hoverinfo='text', mode='markers',
                                           marker=dict(showscale=False, colorscale=node_cmap, cmin=1, cmax=500,
                                                       reversescale=False,
                                                       color=[], size=16,
                                                       colorbar=dict(thickness=15, xanchor='left', titleside='right'),
                                                       line=dict(width=1)))
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])

        try:
            # Add node attributes
            cluster = G.nodes[node]['cluster']
            color1 = int((float(cluster) + 1) * int(500/cluster_number))

            node_trace['marker']['color'] += tuple([color1])
            # Add node labels

            if node_labels:
                node_info = G.nodes[node]['type'] + ': ' + str(node) + '<br>' + \
                            "Cluster" + ': ' + str(G.nodes[node]['cluster'])

                node_trace['text'] += tuple([node_info])
        except:
            node_trace['marker']['color'] += tuple(['#888'])
            if node_labels:
                node_info = G.nodes[node]['type'] + ': ' + str(node)

                node_trace['text'] += tuple([node_info])

    # Create figure
    data = [edge_trace, node_trace]
    layout = plotly.graph_objs.Layout(
        title="ky4",
        titlefont=dict(size=16),
        showlegend=False,
        width=1400,
        height=900,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))

    fig = plotly.graph_objs.Figure(data=data, layout=layout)
    if filename:
        plotly.offline.plot(fig, filename=filename, auto_open=True)
    else:
        plotly.offline.plot(fig, auto_open=True)



def main(cata):
    print("show result...")
    # 输入文件
    # inp_file = "data/ky4/ky4.inp"   # inp管网文件
    inp_file = "data/" + cata + "/" + cata + ".inp"  # inp管网文件
    # clustered_node_file = "outdata/cluster_result.csv"   # 聚类结果
    clustered_node_file = "outdata/" + cata + "/result.csv"  # 聚类结果
    # modify_file = "data/modify_id_list.csv"           # 泵节点列表，用于修改泵节点type
    modify_file = "data/" + cata + "/modify_id_list.csv"  # 泵节点列表，用于修改泵节点type

    # 输出文件
    show_file = "outdata/" + cata + "/" + cata + ".html"
    # show_mod_file = "outdata/ky4(mod).html"     # 聚类可视化
    show_mod_file = "outdata/" + cata + "/" + cata + "(mod).html"
    file_mod="outdata/" + cata + "/" + cata + "_mod_cluster.csv"

    wn = wntr.network.WaterNetworkModel(inp_file)


    G = wn.get_graph()
    G = G.to_undirected()

    # 将聚类的结果整理成字典返回，类别ID：节点ID
    cluster_dict = statistics_cluster(clustered_node_file)

    cluster_number = len(cluster_dict)
    # 将泵节点的type属性由‘Junction’改为‘Pump’,返回修正后的图
    G = modify_pump_type(G, clustered_node_file)

    # 将聚类结果添加至属性‘cluster’中，返回修改后的图
    G = add_node_cluster(G, clustered_node_file)

    #show_inter(G, show_file, cluster_number)

    # 检查聚类异常的节点，并修正，返回修正后的图
    G = check_breakcluster(G, cluster_dict)

    out_modified_cluster(G, file_mod)

    # 可视化聚类结果，返回html文件
    #show_inter(G, show_mod_file, cluster_number)
    print("succ!")

if __name__ == "__main__":
    cata = "ky2"
    main(cata)
