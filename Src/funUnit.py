# -*- coding=utf-8 -*-
'''
Created on 2014年6月23日
基础函数集合
@author: Mafing
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

def getTrainFileList(path = "C:/PythonWork/BaiduEntitySimilarity/Data/Type/"):
    #获取所有待测量数据集
    fileList =[path+val for val in os.listdir(path) if "train-" in val]
    return fileList
def getFeatureFileList(featureType= "tfIdf",path = "C:/PythonWork/BaiduEntitySimilarity/Data/Type/"):
    #获取所有特征数据集
    fileList =[path+val for val in os.listdir(path) if (featureType) in val]
    return fileList

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
    '''获取某一类实体的所有属性
    随机挑选50个实例，将他们的attr合并去重
    '''
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


def getRatio(listB):#给定list，根据其内已有的比例，返回其4、3、2、1各占的比例
    tmpList = sorted(listB,reverse = True)#由大到小排列
    first  = tmpList.index(3)
    second = tmpList.index(2)
    third  = tmpList.index(1)
    return [first,second,third]
def getQuarter(listB):#根据四分之一来分配
    number    = len(listB)
    firstQtr  = int(number/4)             #取前四分之一的节点quarter
    secondQtr = int(number/2)
    thirdQtr  = int(number/4)*3  
    return [firstQtr,secondQtr,thirdQtr]  

def transToQuarter(listA,listB,listRat=None):
    #将第一个参数所得的相似度按照四分之一的规则正规化,第二个参数是判断4,3,2,1的index
    if listRat == None:
        listRat = getRatio(listB)#默认采用比例方式
    if len(listRat)!=3:
        print "输入的分割List不正确"
        return None
    tmpList = sorted(listA,reverse = True)#tmpList由大到小排列,不改变listA的顺序
    listAPlus = []
    firstQtr  = tmpList[listRat[0]]              #根据输入的阈值进行分配
    secondQtr = tmpList[listRat[1]]
    thirdQtr  = tmpList[listRat[2]]
    for val in listA:
        if val > firstQtr:
            listAPlus.append(4)
        elif val > secondQtr:
            listAPlus.append(3)
        elif val > thirdQtr:
            listAPlus.append(2)
        else:
            listAPlus.append(1)
    return listAPlus 

def logRegtransToQuarter(listA,listB,listRat=None):
    #直接按照给出的结果进行预测(1.5->1,1.5~2.5->2,……)
    if listRat == None:
        listRat = getRatio(listB)#默认采用比例方式
    if len(listRat)!=3:
        print "输入的分割List不正确"
        return None
    listAPlus = []
    firstQtr  = 3.5              #根据输入的阈值进行分配
    secondQtr = 2.5
    thirdQtr  = 1.5
    for val in listA:
        if val > firstQtr:
            listAPlus.append(4)
        elif val > secondQtr:
            listAPlus.append(3)
        elif val > thirdQtr:
            listAPlus.append(2)
        else:
            listAPlus.append(1)
    return listAPlus 

def getListIndexValBack(listValue,listIndex):
    tmpList = []
    for indexI in listIndex:
        tmpList.append(listValue[indexI])
    return tmpList
        
def featureInit(tag = 1):
    '''分成4种，只有name,des;之后添加其他标称型;之后添加其他标称型的count;之后添加LDA'''
    featureAttr = set() 
    featureAttr.add("name")
    featureAttr.add("description")
    if tag == 1:
        return featureAttr
    featureAttr.add("inLanguage")
    featureAttr.add("datePublish")
    featureAttr.add("country")
    featureAttr.add("actor")
    featureAttr.add("director")
    featureAttr.add("editor")
    if tag == 2:
        return featureAttr
    featureAttr.add("CountDirector")
    featureAttr.add("CountLanguange")
    featureAttr.add("CountCountry")
    featureAttr.add("CountEditor")
    featureAttr.add("CountActor")
    if tag == 3:
        return featureAttr
    featureAttr.add("NameLDACosSim")
    featureAttr.add("NameLDAHellingerSim")
    featureAttr.add("DesLDACosSim")
    featureAttr.add("DesLDAHellingerSim")
    return featureAttr  
   
def cutDocuments():
    import jieba
    stop_words = set()
    '''
    add more stopwords
    '''
    f = open("../Data/stopWords.txt")
    for l in f:
        l = l.strip()
        stop_words.add(l)
    corpus = []
    infoDic = readInfo()
    for val in infoDic:
        if infoDic[val]["type"] in "Movie":
            name = infoDic[val]["name"]
            description = infoDic[val].get("description","")
            corpus.append(name+"\t"+description)
    for i in range(0,len(corpus)):
        l  = corpus[i]
        w = jieba.cut(l)
        w = [val for val in w if val not in stop_words]
        tmp = " ".join(w)
        corpus[i] = tmp
    fOut = open("../Data/cutMovie.txt","w+")
    fOut.write("\n".join(corpus))#写入分词后的结果

if __name__ == "__main__":
    entityDic = readInfo()
    typeDic = {}
    for val in entityDic:
        type = entityDic[val]["type"]
        if type not in typeDic:
            typeDic[type] = 0
        else:
            typeDic[type] = typeDic[type] + 1
    SumPair = 0
    for val in typeDic:
        print val+"\t"+str(typeDic[val])
        SumPair += typeDic[val]*typeDic[val]
    print str(SumPair)
    print str(math.sqrt(SumPair))
    
    
                
        
        