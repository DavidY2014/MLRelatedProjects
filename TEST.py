# -*- coding: utf-8
import csv

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import fetch_20newsgroups
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression,SGDRegressor
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.metrics import classification_report,mean_squared_error
from sklearn.externals import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
#import jieba

def loadDataSetFromSklearn():
    iris = load_iris()
    #print("description\n",iris["DESCR"])
    #print("featurename\n",iris.feature_names)
    #print("feature\n",iris.data,iris.data.shape)

    x_train,x_test,y_train,y_test =train_test_split(iris.data,iris.target,test_size=0.2,random_state=22)
    print(x_train,x_train.shape)

def dict_demo():
    data = [{'city':'beijing','temperature':100},
            {'city':'shanghai','temperature':60},
            {'city':'shenzhen','temperature':70},
            {'city':'hangzhou','temperature':90}]
    transfer = DictVectorizer(sparse=False)
    data_new = transfer.fit_transform(data)
    print(data_new)
    print(transfer.get_feature_names())


def count_demo():
    data = ["this is short , i like python ","life is too long,i dislike python",
            "this is test ,call me baby"]
    transfer = CountVectorizer()
    data_new = transfer.fit_transform(data)
    print(transfer.get_feature_names())
    print(data_new.toarray())

def count_chinese_demo():
    data = ["我爱北京天安门","天安门上太阳升"]
    transfer = CountVectorizer()
    data_new = transfer.fit_transform(data)
    print(data_new.toarray())
    print(transfer.get_feature_names())

def cut_word(text):
    a=list(jieba.cut(text))
    return a;



def count_chinese_demo2():
    data = ["我爱北京天安门","天安门上太阳升"]
    data_new = []
    for sent in data:
        data_new.append(cut_word((sent)))
    #分词后进行特征提取
    transfer = CountVectorizer(stop_words=["我","上"])
    #调用fit_transform
    data_final = transfer.fit_transform(data_new)
    print(data_final.toarray())
    print(transfer.get_feature_names())

def tfidf_demo():
    data = ["this is short , i like python "]
    transfer = TfidfVectorizer()
    data_new = transfer.fit_transform(data)
    print(data_new.toarray())
    print(transfer.get_feature_names())


def minmax_demo():
    data = pd.read_csv("datingTestSet.txt")
    data = data.iloc[:,:3]
    transfer = MinMaxScaler()
    data_new = transfer.fit_transform(data)

    print(data_new)
    print('minmax')

def standard_demo():
    data = pd.read_csv("datingTestSet.txt")
    transfer = StandardScaler()
    data_new =transfer.fit_transform(data)
    print(data_new)



def variance_demo():
    data = pd.read_csv("")
    data = data.iloc[:,1:-2]
    # 方差为2的特征直接去除，可调参
    transfer = VarianceThreshold(threshold=2)
    data_new = transfer.fit_transform(data)
    print(data_new)


def pca_demo():
    data = [[2,4,6,7],
            [3,2,3,4],
            [2,3,4,5],
            [1,2,3,4],
            [4,6,7,8]]
    transfer = PCA(n_components=2)
    data_new =transfer.fit_transform(data)
    print(data_new)

def mergeMulti_demo():
    product_table = pd.read_csv("")
    user_table = pd.read_csv("")
    merge_table = pd.merge(product_table,user_table,on=["",""])
    print(merge_table)

def knn_iris():
    iris = load_iris()
    x_train,x_test,y_train,y_test  =train_test_split(iris.data,iris.target,random_state=6)
    transfer = StandardScaler()
    x_train  = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    estimator = KNeighborsClassifier(n_neighbors=3)
    estimator.fit(x_train,y_train)
    y_predict = estimator.predict(x_test)
    score = estimator.score(x_test,y_test)
    print(score)


def naviebayes():
    news = fetch_20newsgroups(subset="all")
    print(news)


def descisiontree_demo():
    iris = load_iris()
    x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,random_state=22)
    estimetor =DecisionTreeClassifier(criterion="entropy")
    estimetor.fit(x_train,y_train)
    y_predict = estimetor.predict(x_test)
    print(y_predict)
    print('is it the same:\n',y_predict == y_test)
    score = estimetor.score(x_test,y_test)
    print(score)


def linerRegression_demo():
    column_name = ['Sample code number', 'Clump Thickness',
                   'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                   'Marginal Adhesion', 'Single Epithelial Cell Size',
                   'Bare Nuclei', 'Bland Chromatin',
                   'Normal Nucleoli', 'Mitoses', 'Class']
    data = pd.read_csv("breast-cancer-wisconsin.data", names=column_name)
    # 缺失值替换
    data = data.replace(to_replace="?", value=np.nan)
    # 缺失值删除
    data.dropna(inplace=True)
    # 筛选特征值和目标值
    x = data.iloc[:, 1:-1]
    y = data["Class"]
    # 划分测试和训练集
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22)
    # 特征工程
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    #预估器
    estimator = LinearRegression()
    estimator.fit(x_train,y_train)
    print(estimator.coef_)
    print(estimator.intercept_)

def boston_linerRegression_demo1():
    boston_data = load_boston()
    x_train,x_test,y_train,y_test =train_test_split(boston_data.data,boston_data.target,random_state=10)
    #标准化
    transfer = StandardScaler()
    x_standard_train = transfer.fit_transform(x_train)
    x_standard_test = transfer.transform(x_test)
    #预估器(预估模型参数)
    estimator = LinearRegression()
    estimator.fit(x_standard_train,y_train)
    print('正规方程-参数\n',estimator.coef_)
    print('正规方程-偏置\n',estimator.intercept_)
    #评估好坏,MSE
    y_predict = estimator.predict(x_test)
    #print('预测值\n',y_predict)
    error = mean_squared_error(y_test,y_predict)
    print('mse：',error)

def boston_linerRegression_demo2():
    boston_data = load_boston()
    x_train,x_test,y_train,y_test =train_test_split(boston_data.data,boston_data.target,random_state=10)
    #标准化
    transfer = StandardScaler()
    x_standard_train = transfer.fit_transform(x_train)
    x_standard_test = transfer.transform(x_test)
    #预估器(预估模型参数)
    estimator = SGDRegressor()
    estimator.fit(x_standard_train,y_train)
    print('梯度下降-参数\n',estimator.coef_)
    print('梯度下降-偏置\n',estimator.intercept_)
    #评估好坏,MSE
    y_predict = estimator.predict(x_test)
    #print('预测值\n',y_predict)
    error = mean_squared_error(y_test,y_predict)
    print('mse：',error)

def boston_linerRegression_demo3():
    boston_data = load_boston()
    x_train,x_test,y_train,y_test =train_test_split(boston_data.data,boston_data.target,random_state=10)
    #标准化
    transfer = StandardScaler()
    x_standard_train = transfer.fit_transform(x_train)
    x_standard_test = transfer.transform(x_test)
    #预估器(预估模型参数)
    estimator = Ridge()
    estimator.fit(x_standard_train,y_train)
    print('岭回归-参数\n',estimator.coef_)
    print('岭回归-偏置\n',estimator.intercept_)
    #评估好坏,MSE
    y_predict = estimator.predict(x_test)
    #print('预测值\n',y_predict)
    error = mean_squared_error(y_test,y_predict)
    print('mse：',error)


#低方差过滤无效特征
def lowVarianceFeatureExtract_demo():
    boston_data = load_boston()
    x_train,x_test,y_train,y_test =train_test_split(boston_data.data,boston_data.target,random_state=10)
    #标准化
    #transfer = StandardScaler()
    #x_standard_train = transfer.fit_transform(x_train)
    #x_standard_test = transfer.transform(x_test)
    #data = x_train.iloc[:,1:-2]
    # 方差为2的特征直接去除，可调参
    transfer = VarianceThreshold(threshold=2)
    data_new = transfer.fit_transform(x_train)
    print(data_new)
    return None

#相关系数特征提取
def relatedParameterFeatureExtract_demo():
    boston_data = load_boston()
    x_train,x_test,y_train,y_test =train_test_split(boston_data.data,boston_data.target,random_state=10)
    #标准化
    transfer = StandardScaler()
    x_standard_train = transfer.fit_transform(x_train)
    x_standard_test = transfer.transform(x_test)

    return None


#PCA特征提取
def PCA_demo():
    boston_data = load_boston()
    x_train, x_test, y_train, y_test = train_test_split(boston_data.data, boston_data.target, random_state=10)
    # 标准化
    transfer = StandardScaler()
    x_standard_train = transfer.fit_transform(x_train)
    x_standard_test = transfer.transform(x_test)

    transfer = PCA(n_components=0.95)
    data_new = transfer.fit_transform(boston_data.data)
    print(data_new)
    print(boston_data.data.shape,data_new.shape)

def logistic_Demo():
    column_name = ['Sample code number','Clump Thickness',
                   'Uniformity of Cell Size','Uniformity of Cell Shape',
                   'Marginal Adhesion','Single Epithelial Cell Size',
                   'Bare Nuclei','Bland Chromatin',
                   'Normal Nucleoli','Mitoses','Class']
    data = pd.read_csv("breast-cancer-wisconsin.data",names=column_name)
    #缺失值替换
    data = data.replace(to_replace="?",value=np.nan)
    #缺失值删除
    data.dropna(inplace=True)
    #筛选特征值和目标值
    x = data.iloc[:,1:-1]
    y= data["Class"]
    #划分测试和训练集
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22)
    #特征工程
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    #模型训练
    estimator = LogisticRegression()
    estimator.fit(x_train,y_train)
    #显示回归参数和偏置
    #print(estimator.coef_)
    #print(estimator.intercept_)
    y_predict = estimator.predict(x_test)
    report = classification_report(y_test,y_predict,labels=[2,4],target_names=["良性","恶性"])
    print(report)
    #模型保存
    joblib.dump(estimator,'logisticmodel.pkl')
    #estimator = joblib.load('logisticmodel.pkl')


def prepare_data_forUnsupveried():
    column_name = ['Sample code number', 'Clump Thickness',
                   'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                   'Marginal Adhesion', 'Single Epithelial Cell Size',
                   'Bare Nuclei', 'Bland Chromatin',
                   'Normal Nucleoli', 'Mitoses', 'Class']
    data = pd.read_csv("breast-cancer-wisconsin.data", names=column_name)
    # 缺失值替换
    data = data.replace(to_replace="?", value=np.nan)
    # 缺失值删除
    data.dropna(inplace=True)
    # 筛选特征值和目标值
    x = data.iloc[:, 1:-1]
    y = data["Class"]
    # 划分测试和训练集
    #x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22)
    # 特征工程
    #transfer = StandardScaler()
    #x_train = transfer.fit_transform(x_train)
    #x_test = transfer.transform(x_test)
    return x

def kmeans_demo():
    #data= prepare_data_forUnsupveried()
    #高炉能源数据
    data = pd.read_csv("gaolu_csv.csv")
    # 缺失值替换
    data = data.replace(to_replace="?", value=np.nan)
    # 缺失值删除
    data.dropna(inplace=True)
    #特征工程
    transfer = StandardScaler()
    x_standard_data = transfer.fit_transform(data)
    print(x_standard_data)
    #模型训练
    estimetor = KMeans(n_clusters=3)
    estimetor.fit(x_standard_data)
    y_predict = estimetor.predict(x_standard_data)
    #获取轮廓系数
    score = silhouette_score(x_standard_data,y_predict)
    print(score)

    return None


if __name__ == '__main__':
    #count_demo()
    #count_chinese_demo2()
    #tfidf_demo()
    #minmax_demo()
    #standard_demo()
    #variance_demo()
    #pca_demo()
    #knn_iris()
    #wastedata_demo()
    #naviebayes()
    #descisiontree_demo()
    #logistic_Demo()
    #kmeans_demo()
    #linerRegression_demo()
    #boston_linerRegression_demo()
    #linerRegression_demo()
    #boston_linerRegression_demo1()
    #boston_linerRegression_demo2()
    #boston_linerRegression_demo3()
    #lowVarianceFeatureExtract_demo()
    PCA_demo()