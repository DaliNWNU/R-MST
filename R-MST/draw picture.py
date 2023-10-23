import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from collections import OrderedDict


#根据反向近邻求噪声
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
    print(sortrho)
    maxrhoindex=np.zeros(n, dtype=int)
    median_rho = np.median(rho)
    print(median_rho)
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
    # cut_edges.append(dist_edges_sort[len(dist_edges_sort) - 1])
    # cut_edges.append(dist_edges_sort[len(dist_edges_sort) - 2])
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
    # data = np.array(pd.read_table(r'D:\project\数据集\Aggregation.txt', sep='\t', usecols=[0,1],engine='python', header=None))
    # data = np.array(pd.read_table(r'D:\project\数据集\flame有噪声.txt', sep='\t', engine='python', header=None))
    # data = np.array(pd.read_table(r'D:\project\数据集\Spiral.txt', sep='\t', engine='python', header=None))
    # data = np.array(pd.read_table(r'D:\project\数据集\cth3.txt', sep='\t', engine='python', header=None))
    # data = np.array(pd.read_table(r'D:\project\数据集\d6.txt', sep='\t', engine='python', header=None))
    # data = np.array(pd.read_table(r'D:\project\数据集\ls3.txt', sep='\t', engine='python', header=None))
    # data = np.array(pd.read_table(r'D:\project\数据集\Compound.txt', sep='\t', usecols=[0,1],engine='python', header=None))
    data = np.array(pd.read_table(r'D:\project\数据集\jain.txt', sep='\t', usecols=[0,1],engine='python', header=None))
    # data = np.array(pd.read_table(r'D:\project\数据集\lineblobs.txt', sep='   ', usecols=[0,1],engine='python', header=None))
    # data = np.array(pd.read_table(r'D:\project\数据集\threecircles.txt', sep='   ', usecols=[0,1],engine='python', header=None))
    # data = np.array(pd.read_table(r'D:\project\数据集\Twomoons.txt', sep=' ', usecols=[1,2],engine='python', header=None))
    # data = np.array(pd.read_table(r'D:\project\数据集\Zigzag.txt', sep=' ', usecols=[0,1],engine='python', header=None))
    # data = np.array(pd.read_table(r'D:\project\数据集\pathbased.txt', sep='\t', usecols=[0,1],engine='python', header=None))
    # data = np.array(pd.read_table(r'D:\project\数据集\t4.txt', sep='\t', engine='python', header=None))
    # data = np.array(pd.read_table(r'D:\project\数据集\t7.txt', sep='\t', engine='python', header=None))
    # data, y1 = datasets.make_circles(n_samples=2000, factor=0.6, noise=0.04,shuffle=True,random_state=44)
    # data, y2 = datasets.make_moons(n_samples=500, noise=0.1,shuffle=True,random_state=44)
    # data, y2 = datasets.make_blobs(n_samples=1000,n_features=2,cluster_std=0.8, centers=10,shuffle=True,random_state=44)
    # data = np.array(pd.read_table(r'D:\project\数据集\D31.txt', sep=',', usecols=[0,1],engine='python', header=None))
    # data = np.array(pd.read_table(r'D:\project\数据集\grid.txt', sep=' ', usecols=[0,1],engine='python', header=None))
    # data = np.array(pd.read_table(r'D:\project\数据集\worms_2d.txt', sep=' ', usecols=[0,1],engine='python', header=None))
    # data = np.array(pd.read_table(r'D:\project\数据集\2d-20c-no0.txt', sep=',', usecols=[0,1],engine='python', header=None))
    # data = np.array(pd.read_table(r'D:\project\数据集\heartshaped.txt', sep=',', usecols=[0,1],engine='python', header=None))


    x = data[:, 0]
    y = data[:, 1]
    plt.scatter(x,y,s=0.002,c='black')
    plt.scatter(x,y,c='deepskyblue',s=50,marker='o',edgecolors='black')
    plt.xticks([])
    plt.yticks([])
    # plt.savefig('D:\project\论文需要的图\英文论文算法步骤图\英文论文算法步骤图1.png')
    plt.show()
    n = x.shape[0]
    distList = pdist(data, metric='euclidean')
    distMatrix = squareform(distList)
    k1=0
    k2=7
    k_mindist = k_Mindist(n, distMatrix, k1)
    zaosheng, lenzaosheng, zaoshenglist = zaoSheng(n, k1, k_mindist, distMatrix)
    mindistzaosheng = fenpeizaoshengdian(zaoshenglist, distMatrix, n,zaosheng)
    #
    data1 = []
    data1todata = np.zeros(n - lenzaosheng, dtype=int)
    count = -1
    for i in range(n):
        if zaosheng[i]==1:
            plt.scatter(data[i,0],data[i,1],c='black',s=50,marker='s')
            continue
        else:
            data1.append(data[i])
            count = count + 1
            data1todata[count] = i
            plt.scatter(data[i,0],data[i,1],c='deepskyblue',s=50,marker='o',edgecolors='black')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('D:\project\论文需要的图\英文论文算法步骤图\英文论文算法步骤图2.png')
    plt.show()
    distList = pdist(data1, metric='euclidean')
    distMatrix = squareform(distList)
    data1 = np.array(data1)


    xx = data1[:, 0]
    yy = data1[:, 1]
    plt.scatter(xx, yy,c='deepskyblue',s=50,marker='o',edgecolors='black')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('D:\project\论文需要的图\英文论文算法步骤图\英文论文算法步骤图3.png')
    plt.show()
    n = xx.shape[0]
    k_mindist1 = k_Mindist(n, distMatrix, k2)
    rho, fknn = getRho(n, k2, k_mindist1)
    maxrhoindex =hexindianshibie(n, k2, k_mindist1, rho,distMatrix)
    finalmaxrhoindex = delete_repeat_hexindian(maxrhoindex)
    print('数量', len(maxrhoindex), len(finalmaxrhoindex))
    for i in range(n):
        if i in finalmaxrhoindex:
            plt.scatter(data1[i,0],data1[i,1],marker='*',c='red',s=30)
        else:
            plt.scatter(data1[i,0],data1[i,1],c='skyblue',s=30,marker='o')
    for i in range(n):
        start = data1[i]
        end = data1[maxrhoindex[i]]
        plt.arrow(start[0], start[1], end[0]-start[0], end[1]-start[1],head_width=0.3, lw=0.2, facecolor="black", length_includes_head=True,fill=False)  # ⻓度计算包含箭头箭尾
    plt.xticks([])
    plt.yticks([])
    plt.savefig('D:\project\论文需要的图\英文论文算法步骤图2\代表点指向图1.png')
    plt.show()
    for i in range(n):
        if i in finalmaxrhoindex:
            plt.scatter(data1[i,0],data1[i,1],marker='*',c='red',s=30)
        else:
            plt.scatter(data1[i,0],data1[i,1],c='skyblue',s=30,marker='o')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('D:\project\论文需要的图\英文论文算法步骤图2\代表点无指向图1.png')
    plt.show()
    print("构建最小生成树点的数量", len(finalmaxrhoindex))
    print("开始记录时间")

    selected_nodes, result_edges, dist_edges = prime(finalmaxrhoindex, distMatrix)

    for i in range(len(selected_nodes)):
        plt.scatter(data1[selected_nodes[i], 0], data1[selected_nodes[i], 1], marker='*',c='red',s=50)
    for i in range(len(result_edges)):
        plt.plot([data1[result_edges[i][0], 0], data1[result_edges[i][1], 0]],
                 [data1[result_edges[i][0], 1], data1[result_edges[i][1], 1]], color='b')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('D:\project\论文需要的图\英文论文算法步骤图2\英文论文算法步骤图5.png')
    plt.show()

    cut_edges = qiebian(dist_edges, result_edges, maxrhoindex, k_mindist1, k2)
    print('聚类簇数', len(cut_edges) + 1)

    for i in range(len(selected_nodes)):
        plt.scatter(data1[selected_nodes[i], 0], data1[selected_nodes[i], 1], marker='*',c='red',s=30)
    for i in range(len(result_edges)):
        if i not in cut_edges:
            plt.plot([data1[result_edges[i][0], 0], data1[result_edges[i][1], 0]],
                     [data1[result_edges[i][0], 1], data1[result_edges[i][1], 1]], color='b')
        if i in cut_edges:
            plt.plot([data1[result_edges[i][0], 0], data1[result_edges[i][1], 0]],
                     [data1[result_edges[i][0], 1], data1[result_edges[i][1], 1]], color='darkorange')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('D:\project\论文需要的图\英文论文算法步骤图2\英文论文算法步骤图6.png')
    plt.show()

    selected_nodes_cluster,count=selected_nodes_Cluster(selected_nodes,cut_edges,result_edges)
    for k, v in selected_nodes_cluster.items():
        E = data1[v]
        plt.scatter(E[:, 0], E[:, 1],s=50,marker='o',edgecolors='black')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('D:\project\论文需要的图\英文论文算法步骤图\英文论文算法步骤图7.png')
    plt.show()

    b=[]
    for i in range(n):
        b.append(i)
    dirho=list(set(b).difference(set(finalmaxrhoindex)))  # b中有⽽a中没有的

    for i in range(len(dirho)):
        for j in range(len(selected_nodes_cluster)):
            if maxrhoindex[dirho[i]] in selected_nodes_cluster[j+1]:
                selected_nodes_cluster[j+1].append(dirho[i])
                break


    for k, v in selected_nodes_cluster.items():
        E = data1[v]
        plt.scatter(E[:, 0], E[:, 1],s=50,marker='o',edgecolors='black')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('D:\project\论文需要的图\英文论文算法步骤图\英文论文算法步骤图8.png')
    plt.show()

    aset=[]
    for i in range(count):
        aset.append(i+1)
    cluster_data = (-1) * np.ones(x.shape[0], dtype=int)
    for i in range(len(aset)):
        for j in range(len(selected_nodes_cluster[aset[i]])):
            cluster_data[data1todata[selected_nodes_cluster[aset[i]][j]]] = data1todata[aset[i]]
    for i in range(x.shape[0]):
        if cluster_data[i] == -1:
            cluster_data[i] = cluster_data[mindistzaosheng[i]]
    clusterSetFinally = OrderedDict()
    for i in range(len(aset)):
        clusterSetFinally[aset[i]] = []
    for i in range(x.shape[0]):
        for j in range(len(aset)):
            if cluster_data[i] == data1todata[aset[j]]:
                clusterSetFinally[aset[j]].append(i)
                break
    print(clusterSetFinally)

    for k, v in clusterSetFinally.items():
        E = data[v]
        plt.scatter(E[:, 0], E[:, 1],s=50,marker='o',edgecolors='black')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('D:\project\论文需要的图\英文论文算法步骤图\英文论文算法步骤图9.png')
    plt.show()



