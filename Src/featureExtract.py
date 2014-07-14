# -*- coding=utf-8 -*-
'''
Created on 2014年6月23日
将每个实体对转为feature 向量,并存到文件中
@author: Mafing
'''
import funUnit
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def tfIdfExtract(InfoDic,entityDic,attrSet):
    '''
             输入整体的train集上entity集合，返回其内各个属性对应的tf-idf文档向量
     sortEntity = {}#存放该类别下出现在train集中的所有实体sid-在train集中排序id
    '''
    tfidfAttr = {}#存放每类attr对于的tfidf矩阵
    CorPusDic = {}#存放 attr-corpusList
    for val in attrSet:#初始化
        CorPusDic[val] = []
    sortEntity=sorted(entityDic.items(),key=lambda e:e[1])   #按读入顺序排序,由小到大
    sidTmp = sortEntity[0][0]#取出第一条entity，找到其所处的type
    type = InfoDic[sidTmp].get("type")
    typeEntityInfoDicKey = ([val 
                          for val in InfoDic 
                          if InfoDic[val].get("type") in type])#存放属于该类别的所有实体key
    typeEntityInfoDicValue = ([InfoDic[val] 
                          for val in InfoDic 
                          if InfoDic[val].get("type") in type])#存放属于该类别的所有实体属性
    typeEntityInfoDic = dict(zip(typeEntityInfoDicKey, typeEntityInfoDicValue))
    typeEntityDic = {}#存放属于该类别实体对应到tf-idf矩阵中的行数  eg.sid-23
    stop_words = funUnit.stopWords()#添加停用词
    i = 0
    for entityWithId in typeEntityInfoDic:#对于属于该类的实体，每一条找出其属性、切词、加入corpus
        typeEntityDic[entityWithId] = i
        i += 1
        key = entityWithId
        tmpDic = InfoDic[key]
        for attr in attrSet:
            l = tmpDic.get(attr)
            if l  == None:#若该属性缺失
                l = ""
            w = jieba.cut(l)#切词
            w = [val for val in w if val not in stop_words]
            tmp = " ".join(w)
            CorPusDic[attr].append(tmp) #加入corpus
    for attr in attrSet:
        tfidfAttr[attr] = []
        vectorizer = TfidfVectorizer(min_df=1)#长度不低于1
        tmpTypeEntiy  = vectorizer.fit_transform(CorPusDic[attr])#存放了该类型下所有实体的向量列表
        for val in sortEntity:
            sidTmp = val[0]
            lineId = typeEntityDic[sidTmp]
            tmpList = tmpTypeEntiy[lineId]
            tfidfAttr[attr].append(tmpList)
    '''生成完毕''' 
    return tfidfAttr   



def tfIdfCompute(filePath):
    '''
    path = "C:/PythonWork/BaiduData/Src/Type/train-Movie.txt"
            首先 将entity分类，对于每个类别，其中的每个属性，求其tf-idf的向量
            然后 从分类后的train集中，对于每一条测试数据，计算它们在各个属性维度上tf-idf向量的余弦相似度
            按照 id1,id2,name1,name2,att1-Sim,att2-Sim,……,Tag 这样的形式输出

    '''
    f = open(filePath)
    InfoDic = funUnit.readInfo()
    simDic = {}#存放<E1"\t"E2>Sim字典
    entityDic = {}#存放所有entity的序号
    entityId = 0
    for l in f:
        l = l.replace("\r\n","")
        tmpList = l.split("\t")
        if tmpList[0] not in entityDic:#将实体排序
            entityDic[tmpList[0]] = entityId
            entityId += 1
        if tmpList[1] not in entityDic:
            entityDic[tmpList[1]] = entityId
            entityId += 1   
        simDic[tmpList[0]+"\t"+tmpList[1]] = tmpList[2].replace("\n","")
    attrSet = funUnit.getAttr(entityDic)#存放了所有属性的set
    
    tfidfAttr = tfIdfExtract(InfoDic,entityDic,attrSet)#存放每类attr对应的tfidf矩阵
    
    fOut = open("../Data/type/feature-tfIdf-"+filePath.split("-")[1],"w+")
    fOut.write("key1\tkey2\tname1\tname2\t"+"\t".join(list(attrSet))+"\tTag"+"\n")
    for val in simDic:
        key1 = val.split("\t")[0]
        key2 = val.split("\t")[1] 
        tag = simDic[val]
        simList = []
        i1 = entityDic[key1]
        i2 = entityDic[key2]
        for attr in attrSet:
            tf1 = tfidfAttr[attr][i1]
            tf2 = tfidfAttr[attr][i2]
            tmpSim = cosine_similarity(tf1, tf2)[0,0] 
            simList.append(str(tmpSim))
        fOut.write(key1+"\t"+key2+"\t"+InfoDic[key1].get("name")+"\t"+InfoDic[key2].get("name")+"\t"+"\t".join(simList)+"\t"+tag+"\n")

if __name__ == "__main__":
    fileList = funUnit.getTrainFileList("../Data/type/") 
    for filePath in fileList:
        tfIdfCompute(filePath)
