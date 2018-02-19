import pandas as pd
import numpy as np
import os
import sys
sys.path.append('C:\\Users\\an\\PycharmProjects\\LAMOST\\DLmethod')
from sltools import save_pickle


def  stratifiedSampling(df,key ,frac = 0.001):
    return df.groupby(key).apply(lambda x: x.sample(frac=frac)).reset_index(drop = True)

def  stratifiedSplit(df,key,frac = np.array([0.003,0.002])):
    numOfSplit = frac.shape[0]
    dfList = []
    dfTest = np.array([])
    for i in range(numOfSplit):
        dfList.append(np.array([]))
    for _,data in df.groupby(key):
        index = data.index.values
        # shuffle it !
        from numpy.random import  shuffle
        shuffle(index)
        numOfIndex = index.__len__()
        temp = np.split(index,
           (numOfIndex*np.cumsum(frac)).astype(int)
        )
        for i in range(numOfSplit):
            dfList[i] = np.concatenate([dfList[i],temp[i]])
        dfTest = np.concatenate([dfTest,temp[-1]])
    for i in range(numOfSplit):
        dfList[i] = df[df.index.isin(dfList[i])]
    dfTest  = df[df.index.isin(dfTest)]
    return dfList,dfTest

def read_data(index,path , useDF = True):
    import datetime,os
    fileIndex = [os.path.join(path,str(i) + '.txt') for i in index]
    start = datetime.datetime.now()
    dfList = []
    counter = 0
    for file in fileIndex:
        counter += 1
        with open(file) as f:
            dfList.append(f.readline())
        if counter %1000 == 0:
            print(counter)
            end = datetime.datetime.now()
            print(end-start)

    if useDF == False:
        data = np.stack([np.array(data.split(',')).astype('float32') for data in dfList])
    else :
        data = pd.DataFrame([np.array(data.split(',')).astype('float32') for data in dfList],index = index)
    end = datetime.datetime.now()
    print(end - start)
    return  data



if __name__ == "__main__":
    os.chdir("C:\\Users\\an\\Documents\\competition\\LAMOST")
    data = pd.read_csv(".\\data\\first_train_index_20180131.csv")

    IndexList,testList = stratifiedSplit(data,key='type',frac = np.array([0.1]*8))

    for i,index in enumerate(IndexList):
        train_data = read_data(index.id.values,".\\data\\first_train_data_20180131")
        train_label = index.type.values

        save_pickle(train_data,"train_data"+str(i))
        save_pickle(train_label,"train_label"+str(i))

    testList = testList.sample(frac=0.5)
    test_data = read_data(testList.id.values, ".\\data\\first_train_data_20180131")
    test_label = testList.type.values

    save_pickle(test_data, "test_data" )
    save_pickle(test_label, "test_label" )
