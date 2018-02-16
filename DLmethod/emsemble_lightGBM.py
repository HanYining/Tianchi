import  pandas as pd
import  numpy as np
import os
from sltools import save_pickle,load_pickle
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
from sklearn import metrics
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import  ClusterCentroids, NearMiss,RandomUnderSampler
import gc
from mxnet import gluon
from mxnet import autograd
from mxnet import nd
from mxnet.gluon import nn
import mxnet as mx
import lightgbm as lgb
from dlTools import  *

if __name__ == "__main__":

    os.chdir("C:\\Users\\an\\Documents\\competition\\LAMOST")
    netList = []
    for itr in range(4):
        net = load_pickle("./net/net"+str(itr))
        # generate a new Net
        net2 = genNet()
        net2.initialize(ctx=mx.cpu())

        net2ParaList = net2.collect_params()
        netParaList = net.collect_params()
        ParaList = list(zip(net2ParaList.keys(),netParaList.keys()))
        for para in ParaList:
            net2Para, netPara = para
            net2ParaList[net2Para].set_data(
                netParaList[netPara].data().as_in_context(mx.cpu())
            )
        netList.append(net2)




    # loading training Data and testing Data
    train_data, train_label, sdr, lab = loadTrainData(
                                                "train_data" + str(4),
                                                "train_label" + str(4)
    )


    test_data = load_pickle("test_data")
    test_label = load_pickle("test_label")

    test_data = sdr.fit_transform(test_data.values.T).T
    test_label = lab.fit_transform(test_label)
    test_data = test_data
    test_label = test_label

    save_pickle(sdr,"sdr")

    ctx = mx.cpu()
    train_data = nd.array(train_data, ctx=ctx).reshape((train_data.shape[0], 1, 1, -1))
    train_label = nd.array(train_label, ctx=ctx)
    test_data = nd.array(test_data, ctx=ctx).reshape((test_data.shape[0], 1, 1, -1))
    test_label = nd.array(test_label, ctx=ctx)

    # make predict
    resultList = np.zeros(shape=(train_data.shape[0],0))
    for net in netList:
        output = net(train_data)
        output = output.argmax(axis=1).asnumpy()
        print(metrics.confusion_matrix(train_label.asnumpy(),output))
        print(metrics.f1_score(train_label.asnumpy(), output, average='macro'))
        resultList =  np.hstack((resultList,output.reshape((-1,1))))
        print("net for train is ready ~~")
    save_pickle(train_label,"balanced_train_label")
    save_pickle(resultList,"resultList")

    del resultList,train_label
    gc.collect()

    resultList = np.zeros(shape=(test_data.shape[0], 0))
    for net in netList:
        output = net(test_data)
        output = output.argmax(axis=1).asnumpy()
        print(metrics.confusion_matrix(test_label.asnumpy(), output))
        print(metrics.f1_score(test_label.  asnumpy(), output, average='macro'))
        resultList = np.hstack((resultList, output.reshape((-1, 1))))
        print("net for train is ready ~~")
    save_pickle(test_label, "test_label")
    save_pickle(resultList, "resultList_test")


    #try to ouput the middle result

    for itr , net in enumerate(netList):
        temp = train_data
        for i in range(9):
            temp = net[i](temp)
        save_pickle(temp,"train_middle_result"+str(itr)+".dta")
        print("train_middle_result"+str(itr)+".dta is done")
    save_pickle(train_label,"train_label_middle")


    for itr , net in enumerate(netList):
        temp = test_data
        for i in range(9):
            temp = net[i](temp)
        save_pickle(temp, "test_middle_result"+str(itr)+".dta")
        print("test_middle_result"+str(itr)+".dta is done")
    save_pickle(test_label, "test_label_middle")


