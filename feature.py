# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 19:02:31 2018

@author: pcdalao
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np

train_data=pd.read_csv('train_data_0209_2.csv').set_index('Unnamed: 0')
test_data=pd.read_csv('test_data_0209.csv').set_index('Unnamed: 0')
oot_data1=pd.read_csv('oot_data.csv').set_index('Unnamed: 0')
oot_data2=pd.read_csv('oot_data2.csv').set_index('Unnamed: 0')
oot_data3=pd.read_csv('oot_data3.csv').set_index('Unnamed: 0')
star_data=pd.read_csv('star_0209.csv').set_index('Unnamed: 0')

oot_data=pd.concat([oot_data1,oot_data2,oot_data3])
train_data=pd.concat([train_data,star_data],axis=0)

name=[]
for i in range(2600):
   name.append('f'+str(i))    
   
train_data.columns=name
test_data.columns=name
oot_data.columns=name
#选特征


#找出共同的前200特征



from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#标准化
scaler = StandardScaler()
train_datat=train_data.T
oot_datat=oot_data.T
test_datat=test_data.T

train_scale=pd.DataFrame(scaler.fit_transform(train_datat).T)
oot_scale=pd.DataFrame(scaler.fit_transform(oot_datat).T)
test_scale=pd.DataFrame(scaler.fit_transform(test_datat).T)
                             
selected_feat_names=[]
for i in range(2):                           #这里我们进行十次循环取交集
    tmp = set()
    rfc = RandomForestClassifier(n_jobs=-1,n_estimators=100)
    rfc.fit(train_scale, trainy)
    print("training finished")

    importances = rfc.feature_importances_
    indices = np.argsort(importances)[::-1]   # 降序排列
    for f in range(train_scale.shape[1]):
        if f < 500:                            #选出前150个重要的特征
            tmp.add(train_scale.columns[indices[f]])
            print("%2d) %-*s %f" % (f + 1, 30, train_scale.columns[indices[f]], importances[indices[f]]))
    
    selected_feat_names.append(tmp)  
    
    print(len(selected_feat_names), "features are selected")  

list1=list(set(selected_feat_names[0]).intersection(*selected_feat_names[1:]))
listchoose=list1    
train_data=train_data[list1]
oot_data=oot_data[list1]
test_data=test_data[list1]


ipca=PCA(n_components=50)
ipca.fit(train_scale)
train_PCA=ipca.transform(train_scale)
oot_PCA=ipca.transform(oot_scale)
test_PCA=ipca.transform(test_scale)



temp=pd.DataFrame(pd.Series(train_data.index).rename('ind'))
trainy=pd.merge(temp,train_index_series,left_on='ind',right_index=True,how='inner')['type']

temp=pd.DataFrame(pd.Series(test_data.index).rename('ind'))
testy=pd.merge(temp,train_index_series,left_on='ind',right_index=True,how='inner')['type']
