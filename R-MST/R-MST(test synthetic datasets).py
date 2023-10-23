import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from collections import OrderedDict
import time
from sklearn import metrics


# 根据反向近邻求噪声
def zaoSheng(n,k1,k_mindist,distMatrix):
    fknn = OrderedDict()
    zaoshenglist=[]
    lenzaosheng=0
    zaosheng = np.zeros(n, dtype=int)
    if k1!=0:
        for i in range(n):
            fknn[i] = []
        k = 0
        while k != -1:
            for j in range(n):
                fknn[k_mindist[j][k]].append(j)
            k = k + 1
            if k == k1:
                break
        lenfknn = np.zeros(n, dtype=int)
        for i in range(n):
            lenfknn[i] = len(fknn[i])
        juli = np.zeros(n, dtype=float)
        for i in range(n):
            for j in range(k1):
                juli[i] = juli[i] + distMatrix[i, k_mindist[i][j]]
        mean_juli = np.mean(juli)
        median_fknn=np.median(lenfknn)
        for i in range(n):
            if juli[i] > mean_juli and lenfknn[i]<median_fknn:
            # if lenfknn[i]<median_fknn:
                zaosheng[i]=1
                zaoshenglist.append(i)
                lenzaosheng = lenzaosheng + 1
    return zaosheng,lenzaosheng,zaoshenglist

def getRho(n,k1,k_mindist):
    fknn = OrderedDict()
    for i in range(n):
        fknn[i] = []
    k = 0
    while k != -1:
        for j in range(n):
            fknn[k_mindist[j][k]].append(j)
        k = k + 1
        if k == k1:
            break
    lenfknn = np.zeros(n, dtype=int)
    for i in range(n):
        lenfknn[i] = len(fknn[i])
    rho = np.zeros(n, dtype=float)
    for i in range(n):
        if len(fknn[i])==0:
            continue
        else:
            for j in range(len(fknn[i])):
                rho[i] = rho[i] + lenfknn[fknn[i][j]]
        rho[i]=lenfknn[i]
    return rho,fknn

#求每个点的最近距离k个点
def k_Mindist(n,distMatrix,k):
    k_mindist=np.zeros((n,k), dtype=int)
    for i in range(n):
        sortdist = np.argsort(distMatrix[i])
        if sortdist[0]!=i:
            for j in range(1,len(sortdist)):
                if sortdist[j]==i:
                    temp=sortdist[0]
                    sortdist[0]=i
                    sortdist[j]=temp
                    break
        k_mindist[i]=sortdist[1:k+1]
    return k_mindist

def hexindianshibie(n,k,k_mindist1,rho,distMatrix):
    sortrho = np.sort(rho)
    maxrhoindex=np.zeros(n, dtype=int)
    median_rho = np.median(rho)
    kdist = np.zeros(n, dtype=float)
    for i in range(0,n):
        for j in range(0,k):
            kdist[i]=kdist[i]+distMatrix[i,k_mindist1[i][j]]
    meankdist=np.mean(kdist)
    maxkdist=np.max(kdist)
    for i in range(0,n):
        maxrho=rho[i]
        maxrhoindex[i]=i
        if rho[i]<=median_rho and kdist[i]>(meankdist+(maxkdist-meankdist)/2):
            continue
        else:
            for j in range(0,k):
                if rho[k_mindist1[i][j]]>maxrho:
                    maxrho=rho[k_mindist1[i][j]]
                    maxrhoindex[i]=k_mindist1[i][j]
    return maxrhoindex

#删除重复核心点
def delete_repeat_hexindian(maxrhoindex):
    finalmaxrhoindex=np.unique(maxrhoindex)
    return finalmaxrhoindex

def prime(gaorho,distMatrix):
    gaorho=gaorho.tolist()
    n=len(gaorho)
    selected_nodes = []  # 存储选中节点的集合
    candidate_nodes = gaorho  # 备选的集合
    result_edges = []
    dist_edges=[]
    for i in range(n):
        if i == 0:  # 这里随机选择第一个顶点作为起始点，这个起始点
            selected_nodes.append(gaorho[i])
            candidate_nodes.remove(gaorho[i])
        else:
            candidate_cost = distMatrix[selected_nodes, :][:, candidate_nodes]
            idx = np.nonzero(candidate_cost == np.min(candidate_cost))  # 选出从选中节点的出度中最短的边
            #         print(candidate_cost)
            start_p = selected_nodes[idx[0][0]]
            end_p = candidate_nodes[idx[1][0]]
            #         print(start_p, end_p)
            selected_nodes.append(end_p)
            candidate_nodes.remove(end_p)
            result_edges.append([start_p, end_p])
            dist_edges.append(distMatrix[start_p, end_p])
    return selected_nodes,result_edges,dist_edges


def qiebian(dist_edges,result_edges,maxrhoindex,k_mindist1,k):
    dist_edges_sort = np.argsort(dist_edges)
    cut_edges = []
    for i in range(len(dist_edges_sort) - 1, 0, -1):
        dingdian_and_fushudian1 = []
        dingdian_and_fushudian2 = []
        dingdian_and_fushudian1.append(result_edges[dist_edges_sort[i]][0])
        dingdian_and_fushudian2.append(result_edges[dist_edges_sort[i]][1])
        for j in range(0,len(maxrhoindex)):
            if maxrhoindex[j] == result_edges[dist_edges_sort[i]][0]:
                dingdian_and_fushudian1.append(j)
            if maxrhoindex[j] == result_edges[dist_edges_sort[i]][1]:
                dingdian_and_fushudian2.append(j)
        biaoji=0
        for j in range(0,len(dingdian_and_fushudian1)):
            for h in range(0,len(dingdian_and_fushudian2)):
                if dingdian_and_fushudian1[j] in k_mindist1[dingdian_and_fushudian2[h]]:
                    if dingdian_and_fushudian2[h] in k_mindist1[dingdian_and_fushudian1[j]]:
                        biaoji=1
                        break
            if biaoji==1:
                break
        if biaoji==1:
            break
        if biaoji==0:
            cut_edges.append(dist_edges_sort[i])
    return cut_edges

#每个噪声点分配给离他最近的非噪声点所属的簇
def fenpeizaoshengdian(zaoshenglist,distMatrix,n,zaosheng):
    mindistzaosheng=np.zeros(n,dtype=int)
    for i in range(len(zaoshenglist)):
        mindist=np.max(distMatrix[zaoshenglist[i]])
        for j in range(2,n):
            if i==j:
                continue
            if zaosheng[j]==1:
                continue
            if distMatrix[zaoshenglist[i]][j]<mindist:
                mindist=distMatrix[zaoshenglist[i]][j]
                mindistzaosheng[zaoshenglist[i]]=j
    return mindistzaosheng

def selected_nodes_Cluster(selected_nodes,cut_edges,result_edges):
    selected_nodes_cluster = OrderedDict()  # 存储选中节点的集合字典
    selected_nodes_label = np.ones(len(selected_nodes), dtype=int)
    for i in range(len(cut_edges)):
        selected_nodes_label[selected_nodes.index(result_edges[cut_edges[i]][1])] = -1
    count = 1
    selected_nodes_cluster[count] = []
    for i in range(len(selected_nodes)):
        if selected_nodes_label[i] == -1:
            count = count + 1
            selected_nodes_cluster[count] = []
            selected_nodes_cluster[count].append(selected_nodes[i])
        else:
            selected_nodes_cluster[count].append(selected_nodes[i])
    return selected_nodes_cluster,count


if __name__ == '__main__':
    # data = np.array(pd.read_table(r'D:\project\数据集\heartshaped.txt', sep=',', usecols=[0,1],engine='python', header=None))
    # data = np.array(pd.read_table(r'D:\project\数据集\ED_Hexagon.txt', sep=',', usecols=[0,1],engine='python', header=None))
    # data = np.array(pd.read_table(r'D:\project\数据集\r15.orig.txt', sep='\t', usecols=[0,1],engine='python', header=None))
    # data = np.array(pd.read_table(r'D:\project\数据集\cth3.txt', sep='\t', engine='python', header=None))
    # data = np.array(pd.read_table(r'D:\project\数据集\ls3.txt', sep='\t', engine='python', header=None))
    # data = np.array(pd.read_table(r'D:\project\数据集\Aggregation.txt', sep='\t', usecols=[0,1],engine='python', header=None))
    # data = np.array(pd.read_table(r'D:\project\数据集\t4.txt', sep='\t', engine='python', header=None))
    # data = np.array(pd.read_table(r'D:\project\数据集\t7.txt', sep='\t', engine='python', header=None))
    # data = np.array(pd.read_table(r'D:\project\数据集\D31.txt', sep=',', usecols=[0,1],engine='python', header=None))
    data = np.array(pd.read_table(r'D:\project\数据集\jain.txt', sep='\t', usecols=[0,1],engine='python', header=None))
    # data = np.array(pd.read_table(r'D:\project\数据集\threecircles.txt', sep='   ', usecols=[0,1],engine='python', header=None))
    # data = np.array(pd.read_table(r'D:\project\数据集\worms_2d.txt', sep=' ', usecols=[0,1],engine='python', header=None))
    # data = np.array(pd.read_table(r'D:\project\数据集\2d-20c-no0.txt', sep=',', usecols=[0,1],engine='python', header=None))
    # data, y2 = datasets.make_moons(n_samples=4000, noise=0.03,shuffle=True,random_state=44)
    # data, y2 = datasets.make_blobs(n_samples=20000,n_features=2,cluster_std=0.3, centers=10,shuffle=True,random_state=44)

    x = data[:, 0]
    y = data[:, 1]
    plt.scatter(x,y)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    print("开始记录时间")
    start_time = time.time()  # 记录程序开始执行的当前时间
    k1=2
    k2=9
    x = data[:, 0]
    n = x.shape[0]
    distList = pdist(data, metric='euclidean')
    distMatrix = squareform(distList)
    k_mindist=k_Mindist(n, distMatrix, k1)
    zaosheng,lenzaosheng,zaoshenglist=zaoSheng(n,k1,k_mindist,distMatrix)
    mindistzaosheng = fenpeizaoshengdian(zaoshenglist, distMatrix, n,zaosheng)
    data1 = []
    data1todata = np.zeros(n - lenzaosheng, dtype=int)
    count = -1
    for i in range(n):
        if zaosheng[i] == 1:
            continue
        else:
            data1.append(data[i])
            count = count + 1
            data1todata[count] = i
    distList = pdist(data1, metric='euclidean')
    distMatrix = squareform(distList)
    data1 = np.array(data1)
    xx = data1[:, 0]
    yy = data1[:, 1]
    n = xx.shape[0]
    k_mindist1 = k_Mindist(n, distMatrix, k2)
    rho,fknn = getRho(n,k2,k_mindist1)
    maxrhoindex = hexindianshibie(n,k2,k_mindist1,rho,distMatrix)
    finalmaxrhoindex = delete_repeat_hexindian(maxrhoindex)
    print("代表点的个数：",len(finalmaxrhoindex))
    selected_nodes, result_edges, dist_edges = prime(finalmaxrhoindex, distMatrix)
    cut_edges = qiebian(dist_edges, result_edges, maxrhoindex, k_mindist1, k2)
    selected_nodes_cluster,count=selected_nodes_Cluster(selected_nodes,cut_edges,result_edges)
    b=[]
    for i in range(n):
        b.append(i)
    dirho=list(set(b).difference(set(finalmaxrhoindex)))  # b中有⽽a中没有的
    for i in range(len(dirho)):
        for j in range(len(selected_nodes_cluster)):
            if maxrhoindex[dirho[i]] in selected_nodes_cluster[j+1]:
                selected_nodes_cluster[j+1].append(dirho[i])
                break
    aset = []
    for i in range(count):
        aset.append(i + 1)
    cluster_data = (-1) * np.ones(x.shape[0], dtype=int)
    for i in range(len(aset)):
        for j in range(len(selected_nodes_cluster[aset[i]])):
            cluster_data[data1todata[selected_nodes_cluster[aset[i]][j]]] = data1todata[aset[i]]
    for i in range(x.shape[0]):
        if cluster_data[i] == -1:
            cluster_data[i] = cluster_data[mindistzaosheng[i]]
    stop_time = time.time()  # 记录执行结束的当前时间
    func_time = stop_time - start_time  # 得到中间功能的运行时间
    print('共花费：',func_time)
    clusterSetFinally = OrderedDict()
    for i in range(len(aset)):
        clusterSetFinally[aset[i]] = []
    for i in range(x.shape[0]):
        for j in range(len(aset)):
            if cluster_data[i] == data1todata[aset[j]]:
                clusterSetFinally[aset[j]].append(i)
                break
    print(len(clusterSetFinally))
    for k, v in clusterSetFinally.items():
        E = data[v]
        plt.scatter(E[:, 0], E[:, 1])
    plt.xticks([])
    plt.yticks([])
    # plt.savefig('D:\project\论文需要的图\测试时间月亮图\月亮图4000.png')
    plt.show()
    label = np.array(pd.read_table(r'D:\project\数据集\jain.txt', sep='\t', usecols=[2],engine='python', header=None))
    label=label.reshape(len(label),)
    ari = np.round((metrics.adjusted_rand_score(label, cluster_data)), 5)  # 调兰德指数
    nmi = np.round((metrics.normalized_mutual_info_score(label, cluster_data)), 5)  # 标准化互信息
    homo = np.round((metrics.homogeneity_score(label, cluster_data)), 5)
    print("k1=",k1,"k2=",k2)
    print('ari', round(ari, 4))
    print('nmi', round(nmi, 4))
    print('homo', round(homo, 4))

