import random as rand
import math
import numpy as np
import matplotlib.pyplot as plt
import csv


class Node:
    def __init__(self, attribute_):
        self.id = 'rom'
        self.attribute = np.array(attribute_)

    def __init__(self, attribute_, id_):
        self.id = id_
        self.attribute = np.array(attribute_)

    # 欧式距离算法
    def Euc_dis(self, another_node):
        return math.sqrt(
            math.pow(self.coordx - another_node.coordx, 2.0) + math.pow(self.coordy - another_node.coordy, 2.0))

    # 空间属性+非空间属性 算法
    def Self_dis(self, another_node, w):
        dis = (self.attribute - another_node.attribute) ** 2
        dis = np.sum(dis * w, axis=0)
        return dis


class KMeans:
    def __init__(self, geo_locs_, k_, w_):
        self.geo_locations = geo_locs_
        self.k = k_
        self.w = w_
        self.clusters = None  # clusters of nodes
        self.means = []  # means of clusters
        self.debug = False  # debug flag

    def next_random(self, index, nodes, clusters):
        # this method returns the next random node
        # pick next node that has the maximum distance from other nodes
        dist = {}
        for node_1 in nodes:
            if self.debug:
                print("node_1: {} {}".format(node_1.attribute[0], node_1.attribute[0]))
            # compute this node distance from all other nodes in cluster
            for cluster in clusters.values():
                node_2 = cluster[0]
                if self.debug:
                    print("node_2: {} {}".format(node_2.attribute[0], node_2.attribute[0]))
                if node_1 not in dist:
                    # dist[node_1] = math.sqrt(math.pow(node_1.latit - node_2.latit,2.0) + math.pow(node_1.longit - node_2.longit,2.0))
                    dist[node_1] = node_1.Self_dis(node_2, self.w)
                else:
                    # dist[node_1] += math.sqrt(math.pow(node_1.latit - node_2.latit,2.0) + math.pow(node_1.longit - node_2.longit,2.0))
                    dist[node_1] += node_1.Self_dis(node_2, self.w)
        if self.debug:
            for key, value in dist.items():
                print("({}, {}) ==> {}".format(key.attribute[0], key.attribute[1], value))
        # now let's return the node that has the maximum distance from previous nodes
        count_ = 0
        max_ = 0
        for key, value in dist.items():
            if count_ == 0:
                max_ = value
                max_node = key
                count_ += 1
            else:
                if value > max_:
                    max_ = value
                    max_node = key
        return max_node

    def initial_means(self, nodes):
        # compute the initial means
        # pick the first node at random
        node_ = rand.choice(nodes)
        if self.debug:
            print("node#0: {} {}".format(node_.attribute[0], node_.attribute[1]))
        clusters = dict()
        clusters.setdefault(0, []).append(node_)
        nodes.remove(node_)
        # now let's pick k-1 more random nodes
        for i in range(1, self.k):
            node_ = self.next_random(i, nodes, clusters)
            if self.debug:
                print("node#{}: {} {}".format(i, node_.attribute[0], node_.attribute[0]))
            # clusters.append([node_])
            clusters.setdefault(i, []).append(node_)
            nodes.remove(node_)
        # compute mean of clusters
        self.means = self.compute_means(clusters)
        if self.debug:
            print("initial means:")
            self.print_means(self.means)

    def compute_means(self, clusters):
        means = []
        for cluster in clusters.values():
            mean_node = Node([0.0, 0.0, 0.0], 'rom')
            cnt = 0.0
            for node in cluster:
                # print "compute: node(%f,%f)" % (node.latit, node.longit)
                mean_node.attribute[0] += node.attribute[0]
                mean_node.attribute[1] += node.attribute[1]
                cnt += 1.0
            mean_node.attribute[0] = mean_node.attribute[0] / cnt
            mean_node.attribute[1] = mean_node.attribute[1] / cnt
            means.append(mean_node)
        return means

    def assign_nodes(self, nodes):
        # assign nodes to the cluster with the smallest mean
        if self.debug:
            print("assign nodes")
        clusters = dict()
        for node in nodes:
            dist = []
            if self.debug:
                print("node({},{})".format(node.attribute[0], node.attribute[1]))
            # find the best cluster for this node
            for mean in self.means:
                # dist.append(math.sqrt(math.pow(node.latit - mean.latit,2.0) + math.pow(node.longit - mean.longit,2.0)))
                dist.append(node.Self_dis(mean, self.w))
            # let's find the smallest mean
            if self.debug:
                print(dist)
            cnt_ = 0
            index = 0
            min_ = dist[0]
            for d in dist:
                if d < min_:
                    min_ = d
                    index = cnt_
                cnt_ += 1
            if self.debug:
                print("index: {}".format(index))
            clusters.setdefault(index, []).append(node)
        return clusters

    def update_means(self, means, threshold):
        # compare current means with the previous ones to see if we have to stop
        for i in range(len(self.means)):
            mean_1 = self.means[i]
            mean_2 = means[i]
            if self.debug:
                print("mean_1({},{})".format(mean_1.attribute[0], mean_1.attribute[1]))
                print("mean_2({},{})".format(mean_2.attribute[0], mean_2.attribute[1]))
                # if math.sqrt(math.pow(mean_1.latit - mean_2.latit,2.0) + math.pow(mean_1.longit - mean_2.longit,2.0)) > threshold:
            if mean_1.Self_dis(mean_2, self.w) > threshold:
                return False
        return True

    def save(self, filename="output.csv"):
        # save clusters into a csv file
        with open(filename, mode='w', newline="") as csv_file:
            writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['ID', 'X', 'Y', 'Cluster_id'])
            cluster_id = 0
            for cluster in self.clusters.values():
                for node in cluster:
                    writer.writerow([node.id, node.attribute[0], node.attribute[1], cluster_id])
                cluster_id += 1

    def print_clusters(self, clusters=None):
        if not clusters:
            clusters = self.clusters
        # debug function: print cluster nodes
        cluster_id = 0
        for cluster in clusters.values():
            print("nodes in cluster #{}".format(cluster_id))
            cluster_id += 1
            for node in cluster:
                print("node({},{})".format(node.attribute[0], node.attribute[1]))

    def print_means(self, means):
        # print means
        for node in means:
            print("{} {}".format(node.attribute[0], node.attribute[1]))

    def fit(self, plot_flag):
        # Run k_means algorithm
        if len(self.geo_locations) < self.k:
            return -1  # error
        nodes_ = [node for node in self.geo_locations]
        # compute the initial means
        self.initial_means(nodes_)

        stop = False
        iterations = 1
        print("Starting K-Means...")
        while not stop:
            # assignment step: assign each node to the cluster with the closest mean
            nodes_ = [node for node in self.geo_locations]
            clusters = self.assign_nodes(nodes_)
            if self.debug:
                self.print_clusters(clusters)
            means = self.compute_means(clusters)
            if self.debug:
                print("means:")
                self.print_means(means)
                print("update mean:")
            stop = self.update_means(means, 0.01)
            if not stop:
                self.means = []
                self.means = means
            iterations += 1
        print("K-Means is completed in {} iterations. Check outputs.csv for clustering results!".format(iterations))
        self.clusters = clusters

        return 0

    def showresult(self, plot_flag):
        if plot_flag:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            cnt = 0
            cmap = plt.cm.get_cmap(name='hsv', lut=self.k)

            for cluster in self.clusters.values():
                latits = []
                longits = []
                for node in cluster:
                    latits.append(node.attribute[0])
                    longits.append(node.attribute[1])
                ax.scatter(longits, latits, s=60, c=cmap(cnt), marker='o')
                cnt += 1
            plt.show()
            return 0
