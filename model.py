# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn import metrics
from xgboost.sklearn import XGBClassifier

xgb1= xgb.XGBClassifier(max_depth=5, 
                        min_child_weight=10, #这个参数是指建立每个模型所需要的最小样本数,即调大这个参数能够控制过拟合。
                        n_estimators=150, 
                        learning_rate=0.03,
                        gamma=0,
                        silent=True, 
                        objective='multi:softmax', 
                        nthread=-1, 
                        seed=27,   
                        #max_delta_step=1, #如果取值为0，那么意味着无限制。如果取为正数，则其使得xgboost更新过程更加保守。通常不需要设置这个值，但在使用logistics 回归时，若类别极度不平衡，则调整该参数可能有效果
                        subsample=0.8 ,#用于训练模型的子样本占整个样本集合的比例。如果设置为0.5则意味着XGBoost将随机的从整个样本集合中抽取出50%的子样本建立树模型，这能够防止过拟合。
                        colsample_bytree=0.8, #在建立树时对特征随机采样的比例。缺省值为1[default=1]
                        #colsample_bylevel=1, #决定每次节点划分时子样例的比例[default=1]
                        #scale_pos_weight=1, 
                        #missing=None,
                         #reg_alpha=6,
                         #reg_lambda=6,
                        )
parameters = {
'n_estimators':[300],
'max_depth':[5] ,
'min_child_weight':[4,10],        
'learning_rate':[0.1],
#'reg_alpha': (0.001,0.01,0.1,1,5),     
#'reg_lambda' :(0.001,0.01,0.1,1,5)    
}

grid_search = GridSearchCV(xgb1, parameters, n_jobs=-1,scoring='f1_micro',verbose=2, cv=3)
grid_search.fit(train_PCA, trainy)

print('最佳效果：%0.3f' % grid_search.best_score_)
predict = grid_search.predict(train_PCA)
print('训练集准确性')
print(metrics.f1_score(trainy,predict,average='macro'))

predict_test =grid_search.predict(test_PCA)
print('检验集准确性')
print(metrics.f1_score(predict_test, testy,average='macro'))
print('最优参数：')
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
     print('\t%s: %r' % (param_name, best_parameters[param_name]))   

#模型导出
from sklearn.externals import joblib      
joblib.dump(grid_search, "grid_search_0209_3.m")  
xgb1=joblib.load("grid_search.m")  
#预测
oot_predict=pd.Series(grid_search.predict(oot_PCA))
oot_index=pd.read_csv('first_test_index_20180131.csv')
result=pd.concat([oot_index['id'],oot_predict],axis=1)
result.to_csv('result_0209_3.csv',header=False,index=False)
