# Tianchi

## this project contains our code about LAMOST

## sltools 中的函数load_pickle,save_pickle 用于序列化存储数据

## dltools 中存放了如下内容：

- read_data: 根据给定的list读取相关的train 的txt 数据

- getNet,OneDimensionConv,OneDimensionMaxPool 方法用于实现cnn网络（已弃用）

- Residual，ResNet ：用于实现残差神经网络

## 模型大致分为如下步骤：

- DLmethod/train_sampling.py:
    - 对index表做分层抽样，抽出8个子训练集合（每个使用10%的训练数据）训练8个模型
    - 对剩余数据抽取50%生成test
    - 读取train中的txt，生成相应的pandas.DataFrame

- DLmethod/cnnEnsemble.py:
    - 用于完成resNet模型的训练
    - 对于前7个测试集合完成抽样：
      - 先对于star down sampling： 抽取10%的star
      - 再对于其他的做upsampling： {0: 4429, 1: 4429, 2: 7000, 3: 6500+itr*200}
      - 这里，itr 指模型编号，对于不同的模型，抽取不同数量的unknown
      - upsampling 的方法是bordline SMOTE

    - 对于每一个模型，训练一个resNet 网络
      - 训练特点：使用early stopping
      - 具体选择迭代次数（epoch）的选择方法：
      - 对于每一个eopch，计算在保留的test上的f1score
      - 选择能够使得在test上f1score最大的那个epoch来训练模型


- DLmethod/readTest.py:
  - 用于完成预测
  - 每一个模型做一次预测
  - 理论上会有7个预测结果，可以用第八个模型做集成学习（待做）
  - 现在是7个投票法投票的结果
