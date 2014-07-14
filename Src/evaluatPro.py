# -*- coding=utf-8 -*-
'''
Created on 2014年6月23日
进行验证操作
@author: Mafing
'''
import funUnit
import math
def countDistant(testTagData,trueTagData,tag = 0):
    '''
    function 输入两个list,返回loss(平方距离),pre(准确度)
    first判断是否需要将预测标签正规化(转为1,2,3,4)
    second计算损失数
    third 计算准确率
    '''
    if tag == 1:
        testTagData = funUnit.transToQuarter(testTagData,trueTagData)#对所得的预测值正规化(1)
    elif tag == 2:
        testTagData = funUnit.transToQuarter(testTagData,trueTagData,listRat = funUnit.getQuarter(trueTagData))#对所得的预测值正规化(2)
    loss = 0
    for i in range(0,len(trueTagData)):
        loss += math.pow((testTagData[i]-trueTagData[i]),2)
    loss = math.sqrt((loss+0.0))/len(testTagData)    
    pre = 0
    pre1 = 0
    pre2 = 0
    n1 = 0
    n2 = 0
    for i in range(0,len(trueTagData)):
        if testTagData[i] == trueTagData[i]:
            pre += 1
            if testTagData[i] == 1:
                pre1 += 1
            else:
                pre2 += 1
        if trueTagData[i] == 1:
            n1 += 1
        else:
            n2 += 1
    pre =  (pre + 0.0)/len(testTagData)
    pre1 = (pre1 + 0.0)/n1
    pre2 = (pre2 + 0.0)/n2
    return loss,pre,pre1,pre2 


def printAll(sidPairList,preTagList,truTagList,
             attrList = ["name","description"],
             InfoDic = funUnit.readInfo(),
             filePath = "../Data/result/result.txt"):
    #给一串Entity,给定预测Tag，和fileName，输出其所有的信息到文件中
    f = open(filePath,"w+")
    attrListPlus = []
    for val in attrList:
        attrListPlus.append(val)
        attrListPlus.append(val)
    f.write("预测相似度\t实际相似度\t"+"\t".join(attrListPlus)+"\n")
    for i in range(0,len(sidPairList)):
        sid1 = sidPairList[i][0]
        sid2 = sidPairList[i][1]
        tmpDic1 = InfoDic[sid1]
        tmpDic2 = InfoDic[sid2]
        attrValue = []
        for attr in attrList:
            attrValue.append(tmpDic1.get(attr,"None"))
            attrValue.append(tmpDic2.get(attr,"None"))
        f.write(str(preTagList[i])+"\t"+str(truTagList[i])+"\t"+"\t".join(attrValue)+"\n")


