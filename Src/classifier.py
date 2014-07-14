# -*- coding=utf-8 -*-
'''
Created on 2014年6月23日
利用分类器进行学习
@author: Mafing
'''
import numpy as np
import evaluatPro
import funUnit
import math
from sklearn import svm
from sklearn import linear_model
from sklearn import ensemble
from sklearn import lda
from sklearn import tree
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def testTfidf(filePath):
    corpus = []
    sidList = {}
    i = 0
    f = open("../Data/cutResult.txt")
    for l in f:
        tmpSid = (l.split("\t")[0]).strip()
        sidList[tmpSid] = i
        i += 1 
        l = l.strip()
        corpus.append(l)
    vectorizer = TfidfVectorizer(min_df=1)#长度不低于1
    tfidf  = vectorizer.fit_transform(corpus)
    #tfidfArray = tfidf.toarray() 
    infoDic = funUnit.readInfo()
    fKey = open(filePath)#对每个类型打开文件
    '''第一步，读取attrList，并让name放在第一位'''
    tmpList = fKey.readline().replace("\n","").split("\t")
    attrList= tmpList[4:-1]
    tmp = attrList.index("name")
    attrList[tmp] = attrList[0]
    attrList[0] = "name"
    
    fileType= (filePath.split("-"))[-1]
    sidPair = []
    tagData = []
    testTagData = []
    for l in fKey:
        l = l.strip()
        tmpList = l.split("\t")
        key1 = tmpList[0].strip()
        key2 = tmpList[1].strip()
        sim = tmpList[-1]
        
        sim2 = 0
        try:
            i1 = sidList[infoDic[key1].get("sid")]
            i2 = sidList[infoDic[key2].get("sid")] 
            tf1 = tfidf[i1]
            tf2 = tfidf[i2]
            sim2 = cosine_similarity(tf1, tf2)[0,0]
        except:
            print l
        
        sidPair.append([key1,key2])
        tagData.append(int(sim))
        testTagData.append(sim2)
    
    testTagData = funUnit.transToQuarter(testTagData,tagData)
    '''第四步，对预测值进行评价'''
    loss,pre = evaluatPro.countDistant(testTagData,tagData,0)
    print "tf-Idf-"+fileType+"\t"+str(loss)+"\t"+str(pre)
    '''第五步，将预测值和对应的详细信息输出'''
    infoDic = funUnit.readInfo()
    evaluatPro.printAll(sidPair,testTagData,tagData,attrList,infoDic,"../Data/result/"+"tf-Idf"+"-"+fileType+".txt")
    
def testTfidfPlus(filePath):
    featureData = []
    corpus = []
    sidList = {}
    i = 0
    f = open("../Data/cutResult.txt")
    for l in f:
        tmpSid = (l.split("\t")[0]).strip()
        sidList[tmpSid] = i
        i += 1 
        l = l.strip()
        corpus.append(l)
    vectorizer = TfidfVectorizer(min_df=1)#长度不低于1
    tfidf  = vectorizer.fit_transform(corpus)
    #tfidfArray = tfidf.toarray() 
    infoDic = funUnit.readInfo()
    fKey = open(filePath)#对每个类型打开文件
    '''第一步，读取attrList，并让name放在第一位'''
    tmpList = fKey.readline().replace("\n","").split("\t")
    attrList= tmpList[4:-1]
    tmp = attrList.index("name")
    attrList[tmp] = attrList[0]
    attrList[0] = "name"
    
    fileType= (filePath.split("-"))[-1]
    sidPair = []
    tagData = []
    for l in fKey:
        l = l.strip()
        tmpList = l.split("\t")
        key1 = tmpList[0].strip()
        key2 = tmpList[1].strip()
        sim = tmpList[-1]
        tagData.append(sim)
        try:
            i1 = sidList[infoDic[key1].get("sid")]
            i2 = sidList[infoDic[key2].get("sid")] 
            tf1 = tfidf[i1].todense()
            tf2 = tfidf[i2].todense()
            s = np.concatenate((tfidf[i1], tfidf[i2]), axis=1)
            featureData.append(np.append(np.array(tf1),np.array(tf2)))
        except:
            print l
        sidPair.append([key1,key2])
        tagData.append(int(sim))
    testTagData1 = splitData([featureData,tagData],10,testSVM)
    testTagData2 = splitData([featureData,tagData],10,testLogReg)
    testTagData2 = funUnit.logRegtransToQuarter(testTagData2,tagData)
    '''第四步，对预测值进行评价'''
    loss,pre = evaluatPro.countDistant(testTagData1,tagData,0)
    print "tf-Idf-SVM-"+fileType+"\t"+str(loss)+"\t"+str(pre)
    '''第五步，将预测值和对应的详细信息输出'''
    evaluatPro.printAll(sidPair,testTagData1,tagData,attrList,infoDic,"../Data/result/"+"SVM-tfIdf"+"-"+fileType+".txt")
    '''第四步，对预测值进行评价'''
    loss,pre = evaluatPro.countDistant(testTagData2,tagData,0)
    print "tf-Idf-LogReg-"+fileType+"\t"+str(loss)+"\t"+str(pre)
    '''第五步，将预测值和对应的详细信息输出'''
    evaluatPro.printAll(sidPair,testTagData2,tagData,attrList,infoDic,"../Data/result/"+"LogReg-tfIdf"+"-"+fileType+".txt")

def testSVM(trainData,testData):
    featureData = trainData[0][:]
    tagData = trainData[1][:]
    clf = svm.SVC(kernel='linear', C=4)
    clf.fit(np.array(featureData), np.array(tagData))
    testTagArray = clf.predict(testData[0])
    return testTagArray    
    
def testLogReg(trainData,testData):
    '''
    logistic regression方法  tag = 1
    '''
    featureData = trainData[0][:]
    tagData = trainData[1][:]
    #clf = linear_model.LinearRegression()
    clf = linear_model.LogisticRegression()
    clf.fit(featureData, tagData)
    testTagArray = clf.predict(testData[0])
    return testTagArray

def testRandomForest(trainData,testData):
    '''
    RandomForest        tag = 3
    '''
    featureData = trainData[0][:]
    tagData = trainData[1][:]
    clf = ensemble.RandomForestClassifier(n_estimators=10,max_features="sqrt")
    #clf = ensemble.ExtraTreesClassifier(n_estimators=10)
    clf.fit(featureData, tagData)
    testTagArray = clf.predict(testData[0])
    return testTagArray

def testAdaBoost(trainData,testData):
    '''
    AdaBoost            tag = 4
    '''
    featureData = trainData[0][:]
    tagData = trainData[1][:]
    clf = ensemble.AdaBoostClassifier(n_estimators=100)
    #   tree.DecisionTreeClassifier()   svm.SVC(kernel='linear', C=4),algorithm='SAMME'
    clf.fit(featureData, tagData)
    testTagArray = clf.predict(testData[0])
    return testTagArray

def testGradientBoosting(trainData,testData):
    '''
    GradientBoosting    tag = 5
    '''
    featureData = trainData[0][:]
    tagData = trainData[1][:]
    clf = ensemble.GradientBoostingClassifier(n_estimators=100,learning_rate=1)
    #clf = ensemble.ExtraTreesClassifier(n_estimators=10)
    clf.fit(featureData, tagData)
    testTagArray = clf.predict(testData[0])
    return testTagArray

def testBagging(trainData,testData):
    '''
    Bagging             tag = 6
    '''
    featureData = trainData[0][:]
    tagData = trainData[1][:]
    clf = ensemble.BaggingClassifier(n_estimators=10,base_estimator=linear_model.LogisticRegression())
    #svm.SVC(kernel='linear', C=4)   lda.LDA()   linear_model.LogisticRegression()
    #clf = ensemble.ExtraTreesClassifier(n_estimators=10)
    clf.fit(featureData, tagData)
    testTagArray = clf.predict(testData[0])
    return testTagArray

def testLDA(trainData,testData):
    '''
    logistic regression方法  tag = 7
    '''
    featureData = trainData[0][:]
    tagData = trainData[1][:]
    clf = lda.LDA()
    #clf = ensemble.ExtraTreesClassifier(n_estimators=10)
    clf.fit(featureData, tagData)
    testTagArray = clf.predict(testData[0])
    return testTagArray

def splitData(AllData,cv,classifier = testSVM):
    '''
    all data
    cv 为 交叉验证的折数
    return 对每条数据的预测结果
    '''
    featureData = AllData[0][:]
    tagData = AllData[1][:]
    testTagData = [0 for val in tagData]#初始测试数据为0
    testLenth = len(tagData)/cv 
    for i in range(0,cv):
        startId = i*testLenth
        endId= (i+1)*testLenth
        if i == cv-1:#考虑最后面的情况
            endId = len(featureData)
        testData = {}
        testData[0] = featureData[startId:endId]
        testData[1]  = tagData[startId:endId]
        trainData = {}
        trainData[0] = featureData[0:startId] + featureData[endId:]
        trainData[1] = tagData[0:startId] + tagData[endId:]
        testTagArray = classifier(trainData,testData)
        testTagData[startId:endId] = testTagArray
    return testTagData

def getAllData(filePathOfTrain,filePathOfTest,outPutFilePath,tag=1):
    '''
    给一个文件名，进行相似度计算，并输出它的loss、正确率,以及预测的详细信息
    tag=1 SVM
    tag=2 logistic regression
    '''
    classfier = None
    className = None
    if tag == 1:
        classfier = testSVM
        className = "SVM"
    elif tag == 2:
        classfier = testLogReg
        className = "LogReg"
    elif tag == 3:
        classfier = testRandomForest
        className = "RandomForest"
    elif tag == 4:
        classfier = testAdaBoost
        className = "AdaBoost"
    elif tag == 5:
        classfier = testGradientBoosting
        className = "GradientBoosting"
    elif tag == 6:
        classfier = testBagging
        className = "Bagging"
    elif tag == 7:
        classfier = testLDA
        className = "LDA"
    
    '''读取训练数据'''
        
    sidPair = []
    featureData = []
    tagData = []
    f = open(filePathOfTrain)
    '''第一步，读取attrList，并让name放在第一位'''
    tmpList = f.readline().replace("\n","").split("\t")
    attrList= tmpList[4:-1]
    tmp = attrList.index("name")
    attrList[tmp] = attrList[0]
    attrList[0] = "name"
        
    '''第二步，读取所有的data'''
    for l in f:
        tmpList = l.replace("\n","").split("\t")
        tmpData = tmpList[4:-1]
        tmpData = [float(val) for val in tmpData]
        featureData.append(tmpData)
        tagData.append(int(tmpList[-1]))
        sidPair.append(tmpList[0:2])
        
    '''读取测试数据'''
        
    sidPairOfTest = []
    featureDataOfTest = []
    tagDataOfTest = []
    fOfTest = open(filePathOfTest)
    '''第一步，读取attrList，并让name放在第一位'''
    tmpListOfTest = fOfTest.readline().replace("\n","").split("\t")
    attrListOfTest= tmpListOfTest[4:-1]
    tmpOfTest = attrListOfTest.index("name")
    attrListOfTest[tmpOfTest] = attrListOfTest[0]
    attrListOfTest[0] = "name"
        
    '''第二步，读取所有的data'''
    for l in fOfTest:
        tmpListOfTest = l.replace("\n","").split("\t")
        tmpDataOfTest = tmpListOfTest[4:-1]
        tmpDataOfTest = [float(val) for val in tmpDataOfTest]
        featureDataOfTest.append(tmpDataOfTest)
        tagDataOfTest.append(int(tmpListOfTest[-1]))
        sidPairOfTest.append(tmpListOfTest[0:2])
    
    testTagArray=classfier([featureData,tagData],[featureDataOfTest,tagDataOfTest])
    
    outPut = open(outPutFilePath,"w+")
    for i in range(0,len(sidPairOfTest)):
        outPut.write(str(testTagArray[i])+"\n")
    return
        
    '''第三步，分割data，并进行交叉预测   
    testTagData = splitData([featureData,tagData],10,classfier)
    #if tag == 2:
        #testTagData = funUnit.transToQuarter(testTagData,tagData)
        #testTagData = funUnit.logRegtransToQuarter(testTagData,tagData)
    ###第四步，对预测值进行评价
    loss,pre,pre1,pre2 = evaluatPro.countDistant(testTagData,tagData,0)
    fileType= (filePathOfTrain.split("-"))[-1]
    print fileType+"\t"+className+"\t"+str(pre)+"\t"+str(pre1)+"\t"+str(pre2)
    #print fileType+"\t"+className+"\t"+str(loss)+"\t"+str(pre)
    ###第五步，将预测值和对应的详细信息输出
    infoDic = funUnit.readInfo()
    evaluatPro.printAll(sidPair,testTagData,tagData,attrList,infoDic,"../Data/result/"+className+"-"+fileType+"2.txt")
    '''
    
if __name__ == "__main__":
    fileList = funUnit.getFeatureFileList(path = "../Data/type/") 
    tag=4
    #for tag in range(3,4):
    getAllData("../Data/type/feature-train-Movie.txt","../Data/type/feature-test-1Movie.txt","../Data/type/result-1Movie.txt",tag)
    '''
    for filePath in fileList:
        #getAllData(filePath,tag=1)
        getAllData(filePath,tag=2)
        #testTfidf(filePath)
        #testTfidfPlus(filePath)
    '''






    
    
