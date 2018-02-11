# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 23:44:11 2018

@author: pcdalao
"""
import os 
import pandas as pd
from sklearn.cross_validation import train_test_split
#读入train_index数据集
train_index=pd.read_csv('first_train_index_20180131.csv')
train_index['type'].value_counts()
train_index.index=train_index['id']
train_index_series=train_index.drop('id',axis=1)
test_index=list(train_index_series.index)[470000:]
qso_index=list(train_index_series[train_index_series['type']=='qso'].index)
galaxy_index=list(train_index_series[train_index_series['type']=='galaxy'].index)
star_index=list(train_index_series[train_index_series['type']=='star'].index)
unknown_index=list(train_index_series[train_index_series['type']=='unknown'].index)[:15000]
#分test train
qso_train, qso_test, qso_train, qso_test = train_test_split(qso_index, qso_index, test_size=0.1,random_state=222)
galaxy_train, galaxy_test, galaxy_train, galaxy_test = train_test_split(galaxy_index, galaxy_index, test_size=0.1,random_state=222)
star_train, star_test, star_train, star_test = train_test_split(star_index, star_index, test_size=0.3,random_state=222)
unknown_train, unknown_test, unknown_train, unknown_test = train_test_split(unknown_index, unknown_index, test_size=0.3,random_state=222)
'''
star       442969
unknown     34288
galaxy       5231
qso          1363
'''
oot_index=pd.read_csv('first_test_index_20180131.csv')[:35000]
oot_index1=pd.read_csv('first_test_index_20180131.csv')[35000:70000]
oot_index2=pd.read_csv('first_test_index_20180131.csv')[70000:]

#采用欠采样来提取数据

train_index=qso_train+galaxy_train+star_train+unknown_train
test_index=qso_test+galaxy_test+star_test+unknown_test

#读入文件qso
def read_data(index0):
   out=[]
   count=0
   for name in index0:
       count=count+1
       path='F:/项目/天池天文/天池/'+str(name)+'.txt'
       f = open(path, "r")
       text = f.read()
       out.append(text.split(','))
       f.close()
       if count%1000==0:
         print(count)
   out=pd.DataFrame(out,index=index0)
   return(out)
          
test=read_data(test_index)
test.to_csv('F:/项目/天池天文/test_data_0209.csv',header=True,index=True)

train=read_data(train_index)
train.to_csv('F:/项目/天池天文/train_data_0209_3.csv',header=True,index=True)

star=read_data(star_index)
star.to_csv('F:/项目/天池天文/star_0209.csv',header=True,index=True)


oot1=read_data(list(oot_index['id']))
oot_data=pd.DataFrame(oot1,index=oot_index)
oot_data.to_csv('F:/项目/天池天文/oot_data.csv',header=True,index=True)

oot2=read_data(list(oot_index1['id']))
oot_data=pd.DataFrame(oot2,index=oot_index1)
oot_data.to_csv('F:/项目/天池天文/oot_data2.csv',header=True,index=True)

 


