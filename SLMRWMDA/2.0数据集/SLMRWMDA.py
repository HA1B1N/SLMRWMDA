#!/usr/bin/env python
# -*- coding: utf-8 -*-
#from numba import jit
import numpy as np
import numpy.linalg as LA
import scipy.linalg as LG

alp = 0.2
Il = 1
Ir = 2
kx = 0.6
# import time
#@jit
def solve_l1l2(W, lambda1):
    nv = W.shape[1]  # 求矩阵W列数
    F = W.copy()
    for p in range(nv):
        nw = LA.norm(W[:, p], "fro")
        #        z=np.mat(np.zeros((495,383)))
        if nw > lambda1:
            F[:, p] = (nw - lambda1) * W[:, p] / nw
        else:
            F[:, p] = np.zeros((W[:, p].shape[0], 1))
    #        F[:,p]=solve_l2(W[:,p],p,lambda1)
    return F


a = np.mat(np.zeros((495, 383)))  # 定义一个495*383的0矩阵
AA = a.astype(int)  # 把矩阵mta化为整形

# 读取miRNA-disease关联数字编
b = np.loadtxt('knowndiseasemirnainteraction.txt')
B = b.astype(int)

# 根据已知miRNA-disease把矩阵AA相应位置元素变为1从而得到邻接矩阵AA
for x in B:
    AA[x[0] - 1, x[1] - 1] = 1
# 读取疾病语义类似性矩阵1
c1 = np.loadtxt('疾病语义类似性矩阵1.txt')

# 读取疾病语义类似性矩阵2
c2 = np.loadtxt('疾病语义类似性矩阵2.txt')

# 读取疾病语义类似性加权矩阵
c = np.loadtxt('疾病语义类似性加权矩阵1.txt')
C = 0.5 * (c1 + c2)  # 语义类似性矩阵

# 读取miRNA功能类似性矩阵
D = np.loadtxt('miRNA功能类似性矩阵.txt')

# 读取miRNA功能类似性加权矩阵
d = np.loadtxt('miRNA功能类似性加权矩阵.txt')

# 根据已知SM-miRNA数字关联把矩阵AA相应位置元素变为1从而得到邻接矩阵AA
for x in B:
    AA[x[0] - 1, x[1] - 1] = 1

# LOOCV开始
for count in range(1):
    print('第' + str(count + 1) + '次随机抽样')
    array = np.random.permutation(5430)
    for y in range(5):
        A = AA.copy()
        array1 = array[0 + y * 1086:1086 * (y + 1)]
        for z1 in array1:
            A[B[z1, 0] - 1, B[z1, 1] - 1] = 0  # 将数据分成五份，其中一份作为验证集

        alpha = 0.1
        J = np.mat(np.zeros((383, 383)))
        X = np.mat(np.zeros((383, 383)))
        E = np.mat(np.zeros((495, 383)))
        Y1 = np.mat(np.zeros((495, 383)))
        Y2 = np.mat(np.zeros((383, 383)))
        mu = 10 ** -4
        max_mu = 10 ** 10
        rho = 1.1
        epsilon = 10 ** -6

        print('第' + str(y + 1) + '次循环')
        while True:
            [U, sigma1, V] = LG.svd(X + Y2 / mu, lapack_driver='gesvd')
            G = [sigma1[k] for k in range(len(sigma1)) if sigma1[k] > 1 / mu]
            svp = len(G)
            if svp >= 1:
                sigma1 = sigma1[0:svp] - 1 / mu
            else:
                sigma1 = [0]
                svp = 1
            J = np.mat(U[:, 0:svp]) * np.mat(np.diag(sigma1)) * np.mat(V[0:svp, :])
            ATA = A.T * A
            X = (ATA + np.eye(383)).I * (ATA - A.T * E + J + (A.T * Y1 - Y2) / mu)

            temp1 = A - A * X
            E = solve_l1l2(temp1 + Y1 / mu, alpha / mu)
            Y1 = Y1 + mu * (temp1 - E)
            Y2 = Y2 + mu * (X - J)
            mu = min(rho * mu, max_mu)
            if LA.norm(temp1 - E, np.inf) < epsilon and LA.norm(X - J, np.inf) < epsilon: break
        P = A * X

        #    print('come=',come)

        # 计算疾病之间高斯类似性矩阵KD,疾病之间集成相似性矩阵SD
        gamad1 = 1
        sum1 = 0
        for nm in range(383):
            sum1 = sum1 + LA.norm(P[:, nm], "fro") ** 2  # F范数
        gamaD1 = gamad1 * 383 / sum1  # 计算参数gamaD1
        KD = np.mat(np.zeros((383, 383)))  # 定义一个383*383的0矩阵，初始化疾病之间高斯相似性矩阵
        for ab in range(383):
            for ba in range(383):
                KD[ab, ba] = np.exp(-gamaD1 * LA.norm(P[:, ab] - P[:, ba], "fro") ** 2)
                #    SD=np.mat(np.zeros((383,383)))#定义一个383*383的0矩阵，初始化疾病之间集成相似性
        SD = np.multiply((C + KD) * 0.5, c) + np.multiply(KD, 1 - c)
        # for e in range(383):
        #   for f in range(383):
        #      if c[e,f]==1:#疾病之间语义相似性加权矩阵相应位置元素==1
        #         SD[e,f]=0.5*(C[e,f]+KD[e,f])#疾病之间集成相似性相应位置元素用疾病之间语义相似性
        #     elif c[e,f]==0:
        #        SD[e,f]=KD[e,f]#疾病之间集成相似性相应位置元素用疾病之间高斯相似性
        # 把SD规范化



        # 计算miRNA之间高斯类似性矩阵mtKM,miRNA之间集成相似性矩阵mtSM
        gamad2 = 1
        sum2 = 0
        for mn in range(495):
            sum2 = sum2 + LA.norm(P[mn, :], "fro") ** 2  # F范数
        gamaD2 = gamad2 * 495 / sum2  # 计算参数gamaD1
        KM = np.mat(np.zeros((495, 495)))  # 定义一个495*495的0矩阵，初始化miRNA之间高斯相似性矩阵
        for cd in range(495):
            for dc in range(495):
                KM[cd, dc] = np.exp(-gamaD2 * LA.norm(P[cd, :] - P[dc, :], "fro") ** 2)
        SM = np.multiply((D + KM) * 0.5, d) + np.multiply(KM, 1 - d)
        #    SM=np.mat(np.zeros((495,495)))#定义一个495*495的0矩阵，初始化miRNA之间集成相似性
        #    for g in range(495):
        #        for h in range(495):
        #           if d[g,h]==1:#miRNA之间功能相似性加权矩阵相应位置元素==1
        #               SM[g,h]=0.5*(D[g,h]+KM[g,h])#miRNA之间集成相似性相应位置元素用miRNA之间功能相似性
        #           elif d[g,h]==0:
        #               SM[g,h]=KM[g,h]#miRNA之间集成相似性相应位置元素用miRNA之间高斯相似性
        # 把SM规范化



        # 在疾病相似性网络、miRNA-miRNA相似性网络上引入RWR算法


        RM = np.mat(np.zeros((495, 495)))
        MM = np.mat(np.zeros((495, 495)))
        M0 = np.mat(np.zeros((495, 1)))
        for MM1 in range(495):
            M0 = SM[:, MM1] / np.sum(SM[:, MM1])
            RM[:, MM1] = M0
            #while LA.norm(MM[:, MM1] - RM[:, MM1], 1) > 10 ** -6:
            #MM[:, MM1] = RM[:, MM1]
            RM[:, MM1] = (1 - kx) * SM * RM[:, MM1] + kx * M0
                #print(MM1, "RM迭代中")

        RD = np.mat(np.zeros((383, 383)))
        DD = np.mat(np.zeros((383, 383)))
        D0 = np.mat(np.zeros((383, 1)))
        for DD1 in range(383):
            D0 = SD[:, DD1] / np.sum(SD[:, DD1])
            RD[:, DD1] = D0
            #while LA.norm(DD[:, DD1] - RD[:, DD1], 1) > 10 ** -6:
            #DD[:, DD1] = RD[:, DD1]
            RD[:, DD1] = (1 - kx) * SD * RD[:, DD1] + kx * D0
                #print(DD1, "RD迭代中")

        # for mm1 in range(495):
        #     sumcoltmp = sum(SM[:, mm1])
        #     if sumcoltmp > 0:
        #         SM[:, mm1] = SM[:, mm1] / sumcoltmp
        # for dd1 in range(383):
        #     sumcoltmp = sum(SD[:, dd1])
        #     if sumcoltmp > 0:
        #         SD[:, dd1] = SD[:, dd1] / sumcoltmp

        SM1 = RM.copy()
        for mm1 in range(495):
            for mm2 in range(495):
                #            sum5=0
                #            sum6=0
                #            for mm in range(495):
                #                sum5=sum5+SM1[mm1,mm]
                #                sum6=sum6+SM1[mm2,mm]NP.
                RM[mm1, mm2] = RM[mm1, mm2] / (np.sqrt(np.sum(SM1[mm1, :])) * np.sqrt(np.sum(SM1[mm2, :])))

        SD1 = RD.copy()
        for nn1 in range(383):
            for nn2 in range(383):
                #            sum3=0
                #            sum4=0
                #            for dd in range(383):
                #                sum3=sum3+SD1[nn1,dd]
                #                sum4=sum4+SD1[nn2,dd]
                RD[nn1, nn2] = RD[nn1, nn2] / (np.sqrt(np.sum(SD1[nn1, :])) * np.sqrt(np.sum(SD1[nn2, :])))


        # S = np.mat(np.random.rand(495, 383))
        # Si = 0.4 * SM2 * S * SD2 + 0.6 * P
        # while LA.norm(Si - S, 1) > 10 ** -6:
        #     S = Si
        #     Si = 0.4 * SM2 * S * SD2 + 0.6 * P
        # 双随机游走
        P = P / np.sum(P[:])
        R = P
        Max_Iter = max(Il, Ir)
        # Ri = np.zeros((495, 383))
        # while LA.norm(Ri - R, 1) > 10 **-6:
        # Ri = R
        LR = (1 - alp) * R * RD + alp * P
        RR = (1 - alp) * RM * R + alp * P
        R = (LR + RR) / 2

        # 计算由1变0元素的得分
        for z2 in array1:
            i = B[z2, 0] - 1
            j = B[z2, 1] - 1
            S1 = R[i, j]
            #            print('S1=%f'%S1)

            # global LOOOCV
            list1 = [S1]  # 定义一个只有1变0元素得分的列表list1
            for mi in range(495):
                for di in range(383):
                    if AA[mi, di] == 0:
                        SG = R[mi, di]
                        list1.append(SG)  # 把AA中其余0元素的得分添加到list1中
                    else:
                        continue
            list2 = sorted(list1, reverse=True)  # 把得分数组list1按从大到小排序
            arrayp1 = np.array(list2)
            locationp1 = np.average(np.where(arrayp1 == S1))  # 找出由1变0元素得分的排序位置并求平均值
            list3 = []
            list3.append(str(locationp1 + 1))
            list3.append('\t')
            Global = open('5_fold_ris2.txt', 'a+')
            Global.writelines(list3)
            Global.close()
        if y == 4:
            Global = open('5_fold_ris2.txt', 'a+')
            Global.writelines('\n')
            Global.close()



