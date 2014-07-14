# -*- coding=utf-8 -*-
'''
Created on 2014-7-14
相似度度量方式
@author: mafing
'''
import math
import numpy as np
def kl(p, q):
    """
    Kullback-Leibler divergence D(P || Q) for discrete distributions计算KL散度
    Parameters
    ----------
    p, q : array-like, dtype=float, shape=n
        Discrete probability distributions.
    thanks to gist : https://gist.github.com/larsmans/3104581
    """
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)
    sum_pq = np.sum(np.where(p != 0, p * np.log(p / q), 0))
    sum_qp = np.sum(np.where(q != 0, q * np.log(q / p), 0))
    return (sum_pq+sum_qp)/2 # symmetric

def cos_dist(a, b):##计算余弦相似度
    if a.shape[1] != b.shape[1]:
        return None
    part_up = 0.0
    a_sq = 0.0
    b_sq = 0.0
    for i in range(0,a.shape[1]):
        a1 = a[0,i]
        b1 = b[0,i]
        part_up += a1*b1
        a_sq += a1**2
        b_sq += b1**2
    part_down = math.sqrt(a_sq*b_sq)
    if part_down == 0.0:
        return None
    else:
        return part_up / (part_down+0.0) 

def jaccardSim(list1,list2):
    '''求jaccard相似度'''
    if list1 == ["none"] or list2 == ["none"]:#若存在空集
        return 0
    try:
        listInter  = [val for val in list1 if val in list2]#交集
        listMerge  = list(set(list1+list2))
        return (len(listInter)+0.0)/(len(listMerge)+0.0)
    except:
        return 0

def datePublishSim(date1,date2):
    '''求年份之间的相似度'''
    maxmargin = 100#假设年份之间最大长度为100
    minus = 0 #年份之间差值初始值为0
    if date1 in "none" or date2 in "none":
        minus = 100
    else:
        minus = math.fabs(int(date1)-int(date2))
    return ((1+1.0/maxmargin)/(minus+1) - 1.0/maxmargin)
    

if __name__ == "__main__":
    """主函数开始"""