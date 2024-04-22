import pandas as pd
import numpy as np
import networkx as nx


# # 读取 .txt 文件
# txt_file_path = 'data_sea/collaboration/collaboration.txt'
# with open(txt_file_path, 'r') as txt_file:
#     txt_content = txt_file.readlines()
#
# # 假设 .txt 文件的内容是以制表符分隔的数据，每行有多列
# data = []
# for line in txt_content:
#     line = line.strip()  # 去除换行符
#     columns = line.split('\t')  # 假设使用制表符分隔
#     data.append(columns)
#
# # 构建 DataFrame
# df = pd.DataFrame(data, columns=[0, 1, 2])  # 根据实际列数修改列名
#
# # 保存为 .csv 文件
# csv_file_path = 'data_sea/collaboration/collaboration.csv'
# df.to_csv(csv_file_path, index=False)


for i in range(10):
    data = pd.read_csv('data_sea/condmat/condmat.csv')
    # test
    samdata = data.sample(frac=0.1)
    # others for train
    # data2 = data.append(samdata)
    data2 = pd.concat([data, samdata])
    data2 = data2.drop_duplicates(keep=False)
    samdata.to_csv('data_sea/condmat/condmatsam'+str(i)+'.csv', index=False)
    data2.to_csv('data_sea/condmat/condmat2'+str(i), sep='\t', index=False, header=False)
    # data2.to_csv('data_sea/ucsocial/ucsocial_numpy'+str(i)+'.csv', index=False)



# 这是划分不同比例的训练集和测试集的数据
# ratio = 0.7
# for j in range(7):
#     for i in range(10):
#         data = pd.read_csv('data_sea/celegans/celegans.csv')
#         # test
#         samdata = data.sample(frac=ratio)
#         # others for train
#         # data2 = data.append(samdata)
#         data2 = pd.concat([data, samdata])
#         data2 = data2.drop_duplicates(keep=False)
#         samdata.to_csv('data_sea/celegans/celeganssam'+str(j)+str(i)+'.csv', index=False)
#         data2.to_csv('data_sea/celegans/celegans2'+str(j)+str(i), sep='\t', index=False, header=False)
#         data2.to_csv('data_sea/celegans/celegans_numpy'+str(j)+str(i)+'.csv', index=False)
#     ratio = ratio - 0.1