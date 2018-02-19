
from dlTools import  *
if __name__ == "__main__":
    os.chdir("C:\\Users\\an\\Documents\\competition\\LAMOST\\")
    # file = pd.read_csv(".\\data\\first_test_index_20180131.csv")
    # test_data = read_data(file.id.values, ".\\data\\first_test_data_20180131")
    #
    # sdr = load_pickle("sdr")
    # test_data = sdr.fit_transform(test_data.values.T).T
    test_data = load_pickle("real_test_data_no_label")
    ctx = mx.cpu()
    test_data = nd.array(test_data, ctx=ctx).reshape((test_data.shape[0], 1, -1))
    # save_pickle(test_data, "real_test_data_no_label")

    test_dataList = [test_data[:20000,:,:],test_data[20000:40000,:,:],test_data[40000:60000,:,:]
            ,test_data[60000:80000,:,:],test_data[80000:,:,:]]

    numOfNet = 7
    # loading the net
    netList = []
    for itr in range(numOfNet):
        net = load_pickle("./net/net" + str(itr))
        net.collect_params().reset_ctx(mx.cpu())
        netList.append(net)

    for itr , test_data in enumerate(test_dataList):
        resultList = np.zeros(shape=(test_data.shape[0], 0))
        print(test_data.shape[0])
        for net in netList:
            output = net(test_data)
            output = output.argmax(axis=1).asnumpy()
            resultList = np.hstack((resultList, output.reshape((-1, 1))))
            print("net for real test  is ready ~~")
        save_pickle(resultList, "realTest_resultList"+str(itr)+".dta")
        del resultList
        gc.collect()


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
    test[['key','predicted class']].to_csv("result2_19.csv",index = False)

