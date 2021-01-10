import main
import show_clustered_node
import leak_location_base_mSVM

if __name__ == '__main__':
    file = "ky2"
    k = 20
    main.run(file, k)  # 输出result.csv文件，为聚类结果
    show_clustered_node.main(file) # modify_id_list.cev文件，为输出初步修正后的聚类结果文件,须有上一步的result.csv文件

    leak_location_base_mSVM.main(file)