
import lightgbm as lgb
from dlTools import  *

if __name__ == "__main__":

    os.chdir("C:\\Users\\an\\Documents\\competition\\LAMOST")

    # loading the net
    netList = []
    for itr in range(2):
        net = load_pickle("./net/net" + str(itr))
        net.collect_params().reset_ctx(mx.cpu())
        # generate a new Net


    # loading the prepare data
    test_label  = load_pickle("balanced_test_label")
    resultList  = load_pickle("resultList")
    test_result   = load_pickle("resultList_test")
    test_label    = load_pickle( "test_label" )
    for i in range(4):
        temp = load_pickle("train_middle_result" + str(itr) + ".dta")
        load_pickle("test_middle_result" + str(itr) + ".dta")


    # method 1 : voting
    def findMostFreq(x):
        return np.bincount(x).argmax()
    voteMethod = np.apply_along_axis(findMostFreq,1,test_result.astype(int))
    print(metrics.confusion_matrix(test_label.asnumpy(),voteMethod))
    print(metrics.f1_score(test_label.asnumpy(), voteMethod, average='macro'))



    resultList = resultList.astype(str)
    # fit a gbdt to extract feature
    lgbClassify = lgb.LGBMClassifier(boosting_type='gbdt',n_estimators=5,objective="multiclass"
                                     ,class_weight="balanced"
                                     )
    lgbClassify.fit(X = resultList ,y=train_label.asnumpy())

    #extract feature by lgbClassify
    lgbVote  = lgbClassify.predict(test_result)
    print(metrics.confusion_matrix(test_label.asnumpy(), lgbVote))
    print(metrics.f1_score(test_label.asnumpy(), lgbVote, average='macro'))

    #fit a logistic model or ffm model





# code for real test :
    realTestResult = []
    for i in range(5):
        realTestResult.append(load_pickle("realTest_resultList"+str(i)+".dta"))
    result  = np.vstack(realTestResult)
    def findMostFreq(x):
        return np.bincount(x).argmax()
    voteMethod = np.apply_along_axis(findMostFreq,1,result.astype(int))
    test = pd.read_csv(".\\data\\first_test_index_20180131.csv")
    test['predict'] = voteMethod
    di  = {2:"star",1:"qso",3:"unknown",0:"galaxy"}
    test['predicted class'] = test['predict'].map(di)
    test = test.rename(columns={"id":"key"})
    test[['key','predicted class']].to_csv("result2_15.csv",index = False)