# -*- coding=utf-8 -*-
'''
Created on 2014-7-14
通用函数集合
@author: mafing
'''
import os
import random
import math
import numpy as np
def stopWords():#获取停用词
    stop_words = set()
    '''
    add more stopwords
    '''
    f = open("../Data/stopWords.txt")
    for l in f:
        l = l.strip()
        stop_words.add(l)
    return stop_words 

def readInfo():
    '''
            读取所有entity信息
    '''
    infoDic = {}
    fDic = open("../Data/task1_out_ttl.txt")
    for l in fDic:
        l = l.strip()
        tmpList = l.split("\t")
        try:
            if tmpList[0] not in infoDic:
                infoDic[tmpList[0]] = {tmpList[1]:tmpList[2]}
            else:
                infoDic[tmpList[0]][tmpList[1]]=tmpList[2] 
        except:
            print l
    return infoDic  

def getAttr(entityDic):
    '''获取某一类实体的所有属性，随机挑选50个实例，将他们的attr合并去重   '''
    InfoDic = readInfo()
    setAttr = set()
    tmpList = entityDic.items()
    for i in range(0,50):
        sampleInt = random.randint(0,len(entityDic)-1)#随机一个数
        tmpKey = tmpList[sampleInt][0]
        tmpDic = InfoDic[tmpKey]
        attrList = [val for val in tmpDic]
        for val in attrList:
            setAttr.add(val)
    setAttr.remove("type")
    setAttr.remove("sid")
    #print  "该组的属性值为：\t"+"\t".join(list(setAttr))
    return setAttr


def mixSim(list1,list2):
    '''求两个列表交集大小'''
    if list1 == ["none"] or list2 == ["none"]:#若存在空集
        return 0
    listInter  = [val for val in list1 if val in list2]#交集
    return len(listInter)

def getListIndexValBack(listValue,listIndex):
    '''根据给定的listIndex  返回listValue的值'''
    tmpList = []
    for indexI in listIndex:
        tmpList.append(listValue[indexI])
    return tmpList


if __name__ == "__main__":
    '''主函数开始'''
    