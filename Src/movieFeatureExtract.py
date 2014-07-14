# -*- coding=utf-8 -*-
'''
Created on 2014年6月24日

@author: Mafing
'''
import simMeasure
import funUnit
import jieba
import datetime
import time 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def attrTfIdfExtract(InfoDic,entityDic,attr):
    '''
             输入整体的train集上entity集合，返回其内attr对应属性的tf-idf文档向量
     sortEntity = {}#存放该类别下出现在train集中的所有实体sid-在train集中排序id
    '''
    tfidfAttr = []#存放该类attr对于的tfidf矩阵
    CorPusDic = []#存放 attr-corpusList
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
        l = tmpDic.get(attr)
        if l  == None:#若该属性缺失
            l = ""
        w = jieba.cut(l)#切词
        w = [val for val in w if val not in stop_words]
        tmp = " ".join(w)
        CorPusDic.append(tmp) #加入corpus
    vectorizer = TfidfVectorizer(min_df=1)#长度不低于1
    tmpTypeEntiy  = vectorizer.fit_transform(CorPusDic)#存放了该类型下所有实体的向量列表
    for val in sortEntity:
        sidTmp = val[0]
        lineId = typeEntityDic[sidTmp]
        tmpList = tmpTypeEntiy[lineId]
        tfidfAttr.append(tmpList)
    '''生成完毕''' 
    return tfidfAttr   
def attrListExtract(InfoDic,sortEntity,attr):
    dicAttr = {}#存放返回的属性列表
    for entityVal in sortEntity:
        sid = entityVal[0]
        
        dicAttr[sid] = InfoDic[sid].get(attr,"none").split(",")
    return dicAttr

def featureGenerate(filePath):
    '''
            生成特征文件，并且返回 attrList,tagDic,featureDic
    tagDic     = {sid,tag}
    featureDic = {sid,feature}
    path = "C:/PythonWork/BaiduData/Src/Type/train-Movie.txt"
            首先 将entity分类，对于每个类别，其中的每个属性，求其相似度
            然后 从分类后的train集中，对于每一条测试数据，计算它们在各个属性维度上tf-idf向量的余弦相似度
            按照 id1,id2,name1,name2,att1-Sim,att2-Sim,……,Tag 这样的形式输出
    '''
    f = open(filePath)
    InfoDic = funUnit.readInfo()
    simDic = {}#存放<E1"\t"E2>Sim字典
    entityDic = {}#存放所有entity的序号
    entityId = 0
    
    
    
    for l in f:
        l = l.replace("\r\n","").replace("\n","")
        tmpList = l.split("\t")
        if tmpList[0] not in entityDic:#将实体排序
            entityDic[tmpList[0]] = entityId
            entityId += 1
        if tmpList[1] not in entityDic:
            entityDic[tmpList[1]] = entityId
            entityId += 1  
        try: 
            simDic[tmpList[0]+"\t"+tmpList[1]] = tmpList[2].replace("\n","")
        except:
            print l
        if len(entityDic) == 4778:
            break
    attrSet = funUnit.getAttr(entityDic)#存放了所有属性的set
    attrSet.remove("url")#移除url属性
    '''start  添加额外的特征'''
    attrSet.add("CountCountry")
    attrSet.add("CountActor")
    attrSet.add("CountDirector")
    attrSet.add("CountEditor")
    attrSet.add("CountLanguange")
    attrSet.add("NameLDACosSim")
    attrSet.add("NameLDAHellingerSim")
    attrSet.add("NameLDAKLSim")
    attrSet.add("DesLDACosSim")
    attrSet.add("DesLDAHellingerSim")
    attrSet.add("DesLDAKLSim")
    '''end  添加额外的特征'''
    featureAttr = {}#存放各个属性对应的特征向量
    #featureAttr = tfIdfExtract(InfoDic,entityDic,attrSet)#存放每类attr对应的tfidf矩阵
    sortEntity=sorted(entityDic.items(),key=lambda e:e[1])   #按读入顺序排序,由小到大
    '''下面采集其他的attr，不存在的则填充none'''
    featureAttr["datePublish"] = attrListExtract(InfoDic,sortEntity,"datePublish")
    featureAttr["country"    ] = featureAttr["CountCountry"  ] = attrListExtract(InfoDic,sortEntity,"country")
    featureAttr["actor"      ] = featureAttr["CountActor"    ] = attrListExtract(InfoDic,sortEntity,"actor")
    featureAttr["director"   ] = featureAttr["CountDirector" ] = attrListExtract(InfoDic,sortEntity,"director")
    featureAttr["editor"     ] = featureAttr["CountEditor"   ] = attrListExtract(InfoDic,sortEntity,"editor")
    featureAttr["inLanguage" ] = featureAttr["CountLanguange"] = attrListExtract(InfoDic,sortEntity,"inLanguage")
    featureAttr["NameLDACosSim"  ] = featureAttr["NameLDAHellingerSim"] = featureAttr["NameLDAKLSim"] = attrListExtract(InfoDic,sortEntity,"name")
    featureAttr["DesLDACosSim"   ] = featureAttr["DesLDAHellingerSim" ] = featureAttr["DesLDAKLSim"] = attrListExtract(InfoDic,sortEntity,"description")
    
    
    featureAttr["name"]= attrTfIdfExtract(InfoDic,entityDic,"name")#存放该类attr对应的tfidf矩阵
    featureAttr["description"] = attrTfIdfExtract(InfoDic,entityDic,"description")#存放该类attr对应的tfidf矩阵
    
    '''start 初始化lda模型'''
    from gensim import corpora, models, similarities,matutils
    dictionary = corpora.Dictionary.load('../Data/model/deerwester.dict')
    corpus = corpora.MmCorpus('../Data/model/deerwester.mm')
    lda = models.ldamodel.LdaModel.load('../Data/model/model.lda')
    stop_words = funUnit.stopWords()#添加停用词
    '''end 初始化lda模型'''
    
    

    fOut = open("../Data/type/feature-test-1"+filePath.split("-")[1],"w+")
    fOut.write("key1\tkey2\tname1\tname2\t"+"\t".join(list(attrSet))+"\tTag"+"\n")
    f = open(filePath)#再次读取
    for l in f:
        l = l.replace("\r\n","").replace("\n","")
        tmpList = l.split("\t") 
        key1   = tmpList[0]
        key2   = tmpList[1]   
        tag    = tmpList[2].replace("\n","") 
        simList = []
        for attr in attrSet:
            if attr in "name" or attr in "description":#存储方式为list 而非dic
                f1 = featureAttr[attr][entityDic[key1]]
                f2 = featureAttr[attr][entityDic[key2]]
                tmpSim = cosine_similarity(f1, f2)[0,0] 
            elif attr in "datePublish":
                f1 = featureAttr[attr].get(key1)[0]#将其内数值去除，方便下面计算相似度
                f2 = featureAttr[attr].get(key2)[0]
                tmpSim = simMeasure.datePublishSim(f1,f2)
            elif "Count" in attr:
                f1 = featureAttr[attr].get(key1)
                f2 = featureAttr[attr].get(key2)
                tmpSim = simMeasure.mixSim(f1, f2)
            elif "LDA" in attr:
                f1 = featureAttr[attr].get(key1)
                f2 = featureAttr[attr].get(key2)
                f1 = jieba.cut(",".join(f1))#切词
                f1 = [val for val in f1 if val not in stop_words]
                f2 = jieba.cut(",".join(f2))#切词
                f2 = [val for val in f2 if val not in stop_words]
                f1 = dictionary.doc2bow(f1)
                f2 = dictionary.doc2bow(f2)
                vec_lda1 = lda[f1]
                vec_lda2 = lda[f2]
                if "Cos" in attr:
                    tmpSim = matutils.cossim(vec_lda1, vec_lda2)
                elif "HellingerSim" in attr:
                    dense1 = matutils.sparse2full(vec_lda1, lda.num_topics)
                    dense2 = matutils.sparse2full(vec_lda2, lda.num_topics)
                    tmpSim = np.sqrt(0.5 * ((np.sqrt(dense1) - np.sqrt(dense2))**2).sum())
                else:    
                    dense1 = matutils.sparse2full(vec_lda1, lda.num_topics)
                    dense2 = matutils.sparse2full(vec_lda2, lda.num_topics)
                    tmpSim = simMeasure.kl(dense1, dense2)
            else:#其余使用jaccardSim
                f1 = featureAttr[attr].get(key1)
                f2 = featureAttr[attr].get(key2)
                tmpSim = simMeasure.jaccardSim(f1, f2)
            simList.append(str(tmpSim))
        try:
            fOut.write(key1+"\t"+key2+"\t"+InfoDic[key1].get("name")+"\t"+InfoDic[key2].get("name")+"\t"+"\t".join(simList)+"\t"+tag+"\n")
        except:
            print l
if __name__ == "__main__":
    '''
    fileList = funUnit.getTrainFileList("C:/PythonWork/BaiduEntitySimilarity/Data/type/")
    for filePath in fileList:
        if "MovieFilter2" in filePath:
            featureGenerate(filePath)
    '''
    filePath = "../Data/type/test-Movie.txt"
    #filePath = "../Data/test/test-random-Movie.txt"
    featureGenerate(filePath)