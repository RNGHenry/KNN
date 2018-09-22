import tushare as ts
import numpy as np 
import pandas as pd 
import math
import operator
import random
from scipy import stats

# hyper-parameter
k = 7

# the stock code 
code = "000001"

# calculate the statistical distance
def dist(sample_1,sample_2,covar):
    diff = (sample_1[:-1] - sample_2[:-1])
    covar_I = np.linalg.inv(covar)
    dst = diff.dot(covar_I).dot(diff.T)
    return dst

# get the neighbors
def getN(training,testing,k,n):
    distances = []
    tf_data = training.T
    tf_data = tf_data[:-1]
    covar = np.cov(tf_data)
    for x in range(n):
        dst = dist(testing,training[x],covar)
        distances.append((training[x],dst))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

# vote for the candidates and find out the final forecast
def getR(neighbors):
    classvotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classvotes:
            classvotes[response] += 1
        else:
            classvotes[response] = 1
    sortedvotes = sorted(classvotes.items(),key=operator.itemgetter(1),reverse=True)
    return sortedvotes[0][0]

# calculate the testing acuuracy and the significance level
def getA(testing,predictions):
    knn_correct = 0
    guess_correct = 0
    for x in range(len(testing)-1):
        if testing[x][-1] == predictions[x]:
            knn_correct += 1
        if random.randint(0,1) == testing[x][-1]:
            guess_correct += 1
    tst = stats.chisquare([knn_correct,len(testing)-1-knn_correct],f_exp=[guess_correct,len(testing)-1-guess_correct])
    p = tst[1]
    return (knn_correct/float(len(testing)-1)) * 100.0, (guess_correct/float(len(testing)-1)) * 100.0, p

# define the main function
def main(k,code,index=False):
    sp = ts.get_k_data(str(code),start='2017-01-01',index=index)
    data = sp.assign(foresee = 0)  

    for i in range(len(sp)-1):
        j = i + 1
        dclose = sp.get_value(j,'close')
        dopen = sp.get_value(j,'open')
        if dclose - dopen > 0:
            data.set_value(i,'foresee',1)
        else:
            data.set_value(i,'foresee',0)
    data["k1"] = (data["high"] - data["low"])/data["open"]
    data["k2"] = (data["close"] - data["open"])/data["open"]
    data["k3"] = data["volume"]/100000
    data["5d"] = data["close"].rolling(window=5).mean()
    data["13d"] = data["close"].rolling(window=13).mean()
    data["20d"] = data["close"].rolling(window=20).mean()
    data["k4"] = (data["5d"] - data["20d"])
    data["k5"] = (data["5d"] - data["13d"])
    def mad(x): return np.fabs(x - x.mean()).mean()
    data["k6"] = data["close"].rolling(window=5).apply(mad)

    data = data.loc[20:,["k1","k2","k3","k4","k5","k6","foresee"]]
    redata = data[:-1]
    training_data = redata.sample(frac=0.7)
    tst = list(data.index)
    for i in list(training_data.index):
        tst.remove(i)
    testing_data = data[data.index.isin(tst)]
    training_data = training_data.values
    testing_data = testing_data.values


    predictions=[]
    for x in range(len(testing_data)):
        t = x + k
        neighbors = getN(training_data,testing_data[x],k,t)
        result = getR(neighbors)
        predictions.append(result)
        print('> predicted=' + repr(result) + ', actual=' + repr(testing_data[x][-1]))
    knn_accuracy, guess_accuracy, p_value = getA(testing_data, predictions)
    print('k = ' + str(k))
    print('code = ' + str(code))

    # print the accuracy, matched accuracy and the significance level 
    print('Accuracy: ' + repr(knn_accuracy) + '%') 
    print('Matched_Accuracy: ' + repr(guess_accuracy) + '%')
    print("P_value: " + repr(p_value))
    if predictions[-1] == 1:
        print('buy in')
        return "buy in"
    else:
        print('sell out')
        return "buy out" 


if __name__ == "__main__":
    main(k,code,index=True)
