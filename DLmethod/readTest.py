
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
    test_data = nd.array(test_data, ctx=ctx).reshape((test_data.shape[0], 1, 1, -1))
    # save_pickle(test_data, "real_test_data_no_label")

    test_dataList = [test_data[:20000,:,:,:],test_data[20000:40000,:,:,:],test_data[40000:60000,:,:,:]
            ,test_data[60000:80000,:,:,:],test_data[80000:,:,:,:]]


    # loading the net
    netList = []
    for itr in range(4):
        net = load_pickle("./net/net" + str(itr))
        # generate a new Net
        net2 = genNet()
        net2.initialize(ctx=mx.cpu())

        net2ParaList = net2.collect_params()
        netParaList = net.collect_params()
        ParaList = list(zip(net2ParaList.keys(), netParaList.keys()))
        for para in ParaList:
            net2Para, netPara = para
            net2ParaList[net2Para].set_data(
                netParaList[netPara].data().as_in_context(mx.cpu())
            )
        netList.append(net2)

    for itr , test_data in enumerate(test_dataList):
        resultList = np.zeros(shape=(test_data.shape[0], 0))
        for net in netList:
            output = net(test_data)
            output = output.argmax(axis=1).asnumpy()
            resultList = np.hstack((resultList, output.reshape((-1, 1))))
            print("net for real test  is ready ~~")
        save_pickle(resultList, "realTest_resultList"+str(itr)+".dta")
        del resultList
        gc.collect()


    for itrr, test_data in enumerate(test_dataList):
        for itr, net in enumerate(netList):
            temp = test_data
            for i in range(9):
                temp = net[i](temp)
            save_pickle(temp, str(itrr)+"real_test_middle_result" + str(itr) + ".dta")
            print(str(itrr)+"real_test_middle_result" + str(itr) + ".dta is done")






