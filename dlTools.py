import  pandas as pd
import  numpy as np
import os
from sltools import save_pickle,load_pickle
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
from sklearn import metrics
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids, NearMiss,RandomUnderSampler
import gc
from mxnet import gluon
from mxnet import autograd
from mxnet import nd
from mxnet.gluon import nn
import mxnet as mx


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

def accuracy(output, label):
    return nd.sum(output.argmax(axis=1)==label).asscalar()
def loadTrainData(trainLoc,labelLoc):
    train_data = load_pickle(trainLoc)
    train_label = load_pickle(labelLoc)
    sdr = StandardScaler()
    lab = LabelEncoder()
    train_data = sdr.fit_transform(train_data.values.T).T
    train_label = lab.fit_transform(train_label)

    print("trainCounter： do  not  sampling ", Counter(train_label))

    # down sampling the data in class 2
    train_data, train_label = RandomUnderSampler(
        random_state=0, ratio={0: 523, 1: 136, 2: 4429, 3: 342}
    ).fit_sample(
        train_data, train_label
    )
    gc.collect()
    print("trainCounter：after down sampling ", Counter(train_label))

    # upsampling the data
    train_data, train_label = SMOTE(
        kind='borderline1'  # ,ratio={0:520,1:1300,2:4429,3:3420}
    ).fit_sample(
        train_data, train_label
    )
    print("trainCounter：after  up  sampling ", Counter(train_label))
    return train_data,train_label,sdr,lab

def genNet(paras = [1,20,50]):
    drop_prob1 = 0.5
    net = nn.Sequential()
    with net.name_scope():
        net.add(
            OneDimensionConv(out_layer=paras[1],input_layer=paras[0],kernel_size=100),
            OneDimensionConv(out_layer=paras[1], input_layer=paras[1], kernel_size=100),
            OneDimensionMaxPool(kernel_size=20),
            OneDimensionConv(out_layer=paras[2],input_layer=paras[1],kernel_size=50),
            OneDimensionConv(out_layer=paras[2], input_layer=paras[2], kernel_size=50),
            OneDimensionMaxPool(kernel_size=20),
            # OneDimensionConv(out_layer=20, input_layer=50, kernel_size=100),
            # OneDimensionMaxPool(kernel_size=2, ifStride=True),
            nn.Flatten(),
            nn.Dropout(drop_prob1),
            nn.Dense(128,activation='relu'),
            nn.Dropout(drop_prob1),
            nn.Dense(4)
        )

    return net


class OneDimensionConv(nn.Block):
    def __init__(self,out_layer = 3,input_layer = 1,kernel_size = 100,**kwargs):

        super(OneDimensionConv,self).__init__(**kwargs)
        self.out_layer = out_layer
        self.input_layer = input_layer
        with self.name_scope():
            self.kernel = self.params.get(
                'kernel',shape = (self.out_layer,self.input_layer,1,kernel_size)
            )
            self.bias   = self.params.get('bias',shape=(self.out_layer,))

    def forward(self, x):
        conv = nd.Convolution(x, self.kernel.data(), self.bias.data(),
                       kernel=self.kernel.data().shape[2:], num_filter=self.kernel.data().shape[0])
        return nd.relu(conv)


class OneDimensionMaxPool(nn.Block):
    def __init__(self, kernel_size = 2,**kwargs):
        self.kernel_size = kernel_size
        super(OneDimensionMaxPool,self).__init__(**kwargs)
    def forward(self, x):
        conv = nd.Pooling(data=x, pool_type="max", kernel=(1,self.kernel_size),stride=(1,self.kernel_size))
        #conv = nd.Pooling(data=x, pool_type="max", kernel=(1, self.kernel_size))
        return nd.relu(conv)

class Residual(nn.Block):
    def __init__(self, channels, same_shape=True, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.same_shape = same_shape
        strides = 1 if same_shape else 2
        self.conv1 = nn.Conv1D(channels, kernel_size=3, padding=1,
                               strides=strides)
        self.bn1 = nn.BatchNorm()
        self.conv2 = nn.Conv1D(channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm()
        if not same_shape:
            self.conv3 = nn.Conv1D(channels, kernel_size=1,
                                   strides=strides)

    def forward(self,x):
        out = nd.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if not self.same_shape:
            x = self.conv3(x)
        return nd.relu(out + x)

class ResNet(nn.Block):
    def __init__(self, num_classes, verbose=False, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        self.verbose = verbose
        # add name_scope on the outermost Sequential
        with self.name_scope():
            # block 1
            # batch x 10 x 1297
            b1 = nn.Conv1D(10, kernel_size=3, strides=2)
            # block 2
            b2 = nn.Sequential()
            b2.add(
                nn.MaxPool1D(pool_size=3, strides=2),
                Residual(20, same_shape=False),
                Residual(20),
                Residual(20),

            )
            b3 = nn.Sequential()
            b3.add(
                nn.MaxPool1D(pool_size=3, strides=2),
                Residual(40, same_shape=False),
                Residual(40),
            )
            b3.add(
                nn.MaxPool1D(pool_size=3, strides=2),
                Residual(60, same_shape=False),
                Residual(60),
            )
            b4 = nn.Sequential()
            b4.add(
                nn.AvgPool1D(pool_size=3),
                nn.Dropout(.5),
                nn.Dense(num_classes)
            )
            # chain all blocks together
            self.net = nn.Sequential()
            self.net.add(b1, b2, b3, b4)

    def forward(self, x):
        out = x
        for i, b in enumerate(self.net):
            out = b(out)
            if self.verbose:
                print('Block %d output: %s' % (i + 1, out.shape))
        return out
