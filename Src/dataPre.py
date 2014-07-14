# -*- coding=utf-8 -*-
'''
Created on 2014年6月23日
进行数据预处理工作
@author: Mafing
'''
import funUnit
def processJson(inputJsonFile, outputJsonFile):
    '''将最初始的输入转为<sdi,attr,value>三元组'''
    import json     
    f1=open("task1_bad_line1.txt","w+")
    fin = open(inputJsonFile, 'r')
    fout = open(outputJsonFile, 'w+')
    for eachLine in fin:

        line = eachLine.strip().decode("GBK","ignore").encode("utf-8","ignore")                
        #去除每行首位可能的空格，并且转为UTF-8进行处理
        tmpList = line.split("\t")
        tmpList[0] = tmpList[0].strip()
        
        line = tmpList[1].replace("}",',"sid":"'+tmpList[0]+'"}')
        js = None
        try:
            js = json.loads(line)                              #加载Json文件
        except Exception,e:
            print 'bad line'
            print line
            f1.write(line+'\n')
            continue
        for val in js:
            if type(js[val]) == type([1,2]):#若为list型
                x=js["sid"]+'\t'+val+'\t'+(",".join(js[val])).replace(u"\t",".").replace("\r","")
            else:
                x=js["sid"]+'\t'+val+'\t'+(js[val]).replace(u"\t",".").replace("\r","")
            fout.write(x+'\n')

    fin.close()                                                #关闭文件
    fout.close()



def outPutDifferentTypeResult():
    #将初始的train集合按不同类别存放
    infoDic = funUnit.readInfo()
    fKey = open("../Data/1.txt")
    typeDic = {}
    c=0
    for l in fKey:##将每一行存放成六列数据的形式
        l = l.strip()
        
        tmpList = l.split("\t")
        key1 = tmpList[0].strip()
        key2 = tmpList[1].strip() 
        sim = tmpList[2]
        if key1 not in key2:#去除异常（重复）的行
            tmpType = infoDic[key1].get("type")
            tmpList = [key1,key2,sim]
            if tmpType not in typeDic:
                typeDic[tmpType] = [tmpList]
            else:
                typeDic[tmpType].append(tmpList)
    for tmpType in typeDic:
        fTmp = open("../Data/type/test-"+tmpType+".txt","w+")
        for tmpList in typeDic[tmpType]:
            fTmp.write("\t".join(tmpList)+"\n")



if __name__ == "__main__":
    #processJson('../Data/entity.json', '../Data/task1_out_ttl.txt')
    outPutDifferentTypeResult()
    
    