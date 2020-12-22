from time import time
import wntr
import numpy as np
import math
import pandas as pd
import csv
import matplotlib.pyplot as plt
from matplotlib.axes._axes import _log as matplotlib_axes_logger

from clustering import KMeans
from clustering import Node

matplotlib_axes_logger.setLevel('ERROR')  # 只显示error
np.seterr(divide='ignore', invalid='ignore')


# 通过水力模拟获取节点压力数据，并返回包含各节点坐标与压力信息的numpy数组数据
def sim_data(inp_file='data/Net3/Net3.inp'):
    print('start')
    start = time()
    wn = wntr.network.WaterNetworkModel(inp_file)
    sim = wntr.sim.WNTRSimulator(wn, mode='PDD')

    result = sim.run_sim()
    end = time()
    print('end')
    run_time = end - start
    print('Run time for no burst:' + "%.2f" % run_time + 's')

    pressure_all_time = result.node['pressure']
    pressure_mean = pressure_all_time.mean()  # 各时段压力值平均

    node_property_table = []
    for name, node in wn.junctions():
        x = node.coordinates[0]
        y = node.coordinates[1]
        if y == 0:
            y = 1  # 数据中坐标存在0，不方便后续计算，微调
        jun_xy = [name, x, y, round(pressure_mean[name], 2)]
        node_property_table.append(jun_xy)

    # 完整的节点属性表，ID，X，Y，各时段压力平均值
    node_property_table = np.array(node_property_table)
    return node_property_table

def data1(xyfile='data/ky4/nodexy.csv',pressurefile='data/ky4/matrix_1h.csv'):
    table_pressure = np.loadtxt(pressurefile, delimiter=",", dtype=float)  #
    pressure_mean = np.mean(table_pressure, axis=1)
    table_idxy = np.loadtxt(xyfile, delimiter=",", dtype=str)  #

    node_property_table = np.c_[table_idxy, pressure_mean]
    return node_property_table

def data(inp_file, pressurefile):
    table_file = np.loadtxt(pressurefile, delimiter=",", dtype=str)  #
    table_file = table_file[1:, :]  # 去掉时间戳
    node_id = table_file[:, 0]  # 获取节点ID列表
    table_pressure = table_file[:, 1:]  # 获取个时间段压力变化值
    table_pressure = table_pressure.astype(np.float)
    pressure_mean = np.mean(table_pressure, axis=1)

    wn = wntr.network.WaterNetworkModel(inp_file)
    G = wn.get_graph()
    node_xy = []
    for node in node_id:
        x, y = G.nodes[node]['pos']
        node_xy.append([x, y])
    node_property_table = np.c_[node_id, node_xy, pressure_mean]

    return node_property_table

def normalization(input_table):
    input_table = np.array(input_table)
    max_data_row = input_table.max(0)  # 计算每一列的最大值
    min_data_row = input_table.min(0)  #
    range_data_row = max_data_row-min_data_row  # 每一列的极差
    row = input_table.shape[1]
    for i in range(row):
        input_table[:, i] = (input_table[:, i]-min_data_row[i])/range_data_row[i] + 1

    return input_table



# 通过信息熵计算空间属性与非空间属性（工况：压力）权值
def cal_weight(node_property_table):
    z = 0.5  # 置信系数
    node_property_table_no_name = node_property_table[:, 1:]  # 丢掉name列
    node_property_table_no_name = node_property_table_no_name.astype(float)  # 字符串转float

    row = node_property_table_no_name.shape[0]  # 获取行数
    col = node_property_table_no_name.shape[1]  # 获取列数

    # 归一化
    # node_property_to_norm = np.array(node_property_table_no_name)
    node_property_to_norm = normalization(node_property_table_no_name)
    # for i in range(row):
    #    sum_property = np.sum(node_property_table_no_name[i])
    #    for j in range(col):
    #        node_property_to_norm[i][j] = node_property_table_no_name[i][j] / sum_property

    # 计算各属性信息熵
    node_property_temp = np.log(node_property_to_norm)  # 对数列取自然对数
    node_property_temp = node_property_temp * node_property_to_norm  # 元素相乘
    sum_property = np.sum(node_property_temp, axis=0)  # 对各列求和

    comentropy = np.ones(col)
    for i in range(col):
        comentropy[i] = -1 * (1 / math.log(row)) * sum_property[i]

    # 计算各属性权值，通过区分度
    w = np.ones(col)
    w -= comentropy
    w_sum = np.sum(w)
    for i in range(col):
        w[i] /= w_sum

    w[2:] = w[2:] * z  # 将非空间属性的权值乘上置信系数
    return w



def main(dataset_fn, output_fn, clusters_no, w):
    geo_locs = []
    # read location data from csv file and store each location as a Point(latit,longit) object
    df = pd.read_csv(dataset_fn)
    for index, row in df.iterrows():
        loc_ = Node([float(row['X']), float(row['Y']), float(row['PreChange'])], row['ID'])
        geo_locs.append(loc_)
    # run k_means clustering
    w = np.array(w)
    model = KMeans(geo_locs, clusters_no, w)
    flag = model.fit(True)
    if flag == -1:
        print("No of points are less than cluster number!")
    else:
        # save clustering results is a list of lists where each list represents one cluster
        model.save(output_fn)
        model.showresult(True)

def run(file_cata, clusters_number):
    # 输入
    # data/ky2/ky2.inp
    # outdata/ky2/
    inpfile = 'data/' + file_cata + '/' + file_cata + '.inp'
    pressure_change = 'data/' + file_cata + "/pressure_change_matrix.csv"

    # 中间输出
    node_pro_file = "outdata/" + file_cata + '/' + "node_property.csv"

    # 输出
    output_fn = "outdata/" + file_cata + '/' + "result.csv"

    #clusters_number = 20
    node_property_table = data(inpfile, pressure_change)

    with open(node_pro_file, mode='w', newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['ID', 'X', 'Y', 'PreChange'])
        for row in range(node_property_table.shape[0]):
                writer.writerow(node_property_table[row])
    # np.savetxt(node_pro_file, node_property_table, fmt='%s', delimiter=",")  # 写入Net3聚类结果
    # 计算属性权值
    w = cal_weight(node_property_table)

    main(node_pro_file, output_fn, clusters_number, w)




if __name__ == "__main__":
    z = 0.5  # 置信系数
    n_iter = 150  # 迭代次数
    K = 5  # 聚类数量

    file_type = "ky8"
    k_cluster = 20
    run(file_type, k_cluster)



