from mxnet import gluon
from mxnet import autograd
from mxnet import nd
from mxnet.gluon import nn
import mxnet as mx
import os
import numpy as np
from collections import Counter
from sltools import load_pickle,save_pickle
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
from sklearn import metrics
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import  ClusterCentroids, NearMiss,RandomUnderSampler
import gc
import sys
from dlTools import  *

if __name__ == "__main__":
    os.chdir("C:\\Users\\an\\Documents\\competition\\LAMOST")
    # load train data
    for itr in range(4):
    # nowIter = sys.argv[1]
    #for itr in [0]:
        train_data  = load_pickle("train_data" + str(itr))
        train_label = load_pickle( "train_label" + str(itr))
        sdr = StandardScaler()
        enc = OneHotEncoder()
        lab = LabelEncoder()
        train_data = sdr.fit_transform(train_data.values.T).T
        train_label = lab.fit_transform(train_label)

        print("trainCounter： do  not  sampling ",Counter(train_label))

        # down sampling the data in class 2
        train_data, train_label = RandomUnderSampler(
            random_state=0, ratio={0: 523, 1: 136, 2: 4429, 3: 342}
        ).fit_sample(
            train_data, train_label
        )
        gc.collect()
        print("trainCounter：after down sampling ",Counter(train_label))

        # upsampling the data
        train_data, train_label = SMOTE(
            kind='borderline1'  # ,ratio={0:520,1:1300,2:4429,3:3420}
        ).fit_sample(
            train_data, train_label
        )
        print("trainCounter：after  up  sampling ",Counter(train_label))

        # load test data
        test_data = load_pickle("test_data")
        test_label = load_pickle("test_label")
        test_data = sdr.fit_transform(test_data.values.T).T[:4000, :]
        test_label = lab.fit_transform(test_label)[:4000]
        print("testCounter：after sample sampling ",Counter(test_label))

        # train_label = enc.fit_transform(temp)


        # initialize the net
        ctx = mx.cpu()
        net  = genNet()
        net.initialize(ctx=ctx)
        test_data = nd.array(test_data, ctx=ctx).reshape((test_data.shape[0], 1, 1, -1))
        test_label = nd.array(test_label, ctx=ctx)

        # BATCH-SGD:
        # define loss and trainer
        loss = gluon.loss.SoftmaxCrossEntropyLoss()
        trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.05,'wd': 0.01})
        #trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 0.01, 'wd': 0.01})
        for epoch in range(20):
            train_loss = 0.
            train_acc = 0.
            dataList = list(range(train_data.shape[0]))
            np.random.shuffle(dataList)
            #batchSize = 50
            batchSize = 50
            for idx in range(0,len(dataList),batchSize):
                dataUsed = dataList[idx:(idx+batchSize)]
                X = nd.array(train_data[dataUsed, :], ctx=ctx)
                X = X.reshape((X.shape[0], 1, 1, -1))
                label = nd.array(train_label[dataUsed],ctx=ctx)
                with autograd.record():
                    #print(net(X))
                    output = net(X)
                    losses = loss(output, label)
                    losses.backward()
                trainer.step(batchSize)

                train_loss += nd.sum(losses).asscalar()
                train_acc += accuracy(output, label)

            test_acc = accuracy(net(test_data),test_label)
            print("Epoch %d. Loss: %f, Train acc %f , Test acc %f" % (
                        epoch, train_loss/len(train_data), train_acc/len(train_data),test_acc/len(test_data)
                        )
                  )

        # test f1 score
        output = net(test_data)
        output = output.argmax(axis=1)

        print(metrics.f1_score(test_label.asnumpy(),output.asnumpy(), average='macro'))
        print(metrics.confusion_matrix(test_label.asnumpy(),output.asnumpy()))
        net.save_params("netParams"+str(itr))
        save_pickle(net,"net"+str(itr))

        del train_data,train_label,test_data,test_label,trainer,loss,net
        gc.collect()