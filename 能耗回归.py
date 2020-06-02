import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression,SGDRegressor,Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

def load_gaolu():
    # 高炉能源数据
    column_name = ['高炉工序能耗', '1高炉工序能耗',
                   '2高炉工序能耗', '高炉折煤气用量',
                   '高炉铁水产量', '高炉TRT发电量',
                   '高炉煤气发生量', '高炉喷吹煤用量',
                   '高炉焦炭使用量', '高炉鼓风用量',
                   '1高炉BT风温','2高炉BT风温',
                   '1高炉煤比','2高炉煤比',
                   '1高炉入炉焦比','2高炉入炉焦比',
                   '1高炉煤气CO2含量','2高炉煤气CO2含量',
                   '1#TRT发电率','2#TRT发电率',
                   '铁水比']
    data = pd.read_csv("gaolu_csv.csv",names=column_name )
    # 缺失值替换
    data = data.replace(to_replace="?", value=np.nan)
    # 缺失值删除
    data.dropna(inplace=True)
    print(data)
    # 特征工程
    transfer = StandardScaler()
    x_standard_data = transfer.fit_transform(data)
    print(x_standard_data)
    return x_standard_data

#kmeans无监督聚类
def kmeans_demo():
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
    #获取轮廓系数(轮廓值为[-1,1]，值越大分类效果越好)
    score = silhouette_score(x_standard_data,y_predict)
    print(score)
    return  None


#决策树特征提取（通过熵值来判断有效特征）
def descisiontreeFeatureFilter_demo():
    gaolu_data = load_gaolu()
    x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,random_state=22)
    estimetor =DecisionTreeClassifier(criterion="entropy")
    estimetor.fit(x_train,y_train)
    y_predict = estimetor.predict(x_test)
    print(y_predict)
    print('is it the same:\n',y_predict == y_test)
    score = estimetor.score(x_test,y_test)
    print(score)

#低方差过滤无效特征
def lowVarianceFeatureExtract_demo():
    column_name = ['高炉工序能耗', '1高炉工序能耗',
                   '2高炉工序能耗', '高炉折煤气用量',
                   '高炉铁水产量', '高炉TRT发电量',
                   '高炉煤气发生量', '高炉喷吹煤用量',
                   '高炉焦炭使用量', '高炉鼓风用量',
                   '1高炉BT风温', '2高炉BT风温',
                   '1高炉煤比', '2高炉煤比',
                   '1高炉入炉焦比', '2高炉入炉焦比',
                   '1高炉煤气CO2含量', '2高炉煤气CO2含量',
                   '1#TRT发电率', '2#TRT发电率',
                   '铁水比']
    # 高炉能源数据
    data = pd.read_csv("gaolu_csv.csv", names=column_name)
    # 缺失值替换
    data = data.replace(to_replace="?", value=np.nan)
    # 缺失值删除
    data.dropna(inplace=True)
    # print(data)
    # 筛选特征值和目标值
    x = data.iloc[:, 4:]
    y1 = data.iloc[:, :1]
    y2 = data.iloc[:, 1:2]
    y3 = data.iloc[:, 2:3]
    # 划分训练测试集
    x_train, x_test, y1_train, y1_test = train_test_split(x, y1, random_state=22)
    #低方差过滤
    transfer = VarianceThreshold(threshold=2)  # 方差为2的特征直接去除
    data_new = transfer.fit_transform(x)
    print(data_new)
    print('维数\n',data_new.shape)
    return None

#相关系数特征提取(用pearson系数,[-1,1] 0为无关，1为正相关，-1为负相关,p值为显著性，越小越好)
def pearsonFeatureExtract_demo():
    factor = ['Gas_consumption','Molteniron_production', 'TRTpower_generation',
                   'Gas_production', 'Coalinjection_consumption',
                   'Coke_consumption', 'Amount_of_blast',
                   '1BTwind_temperature', '2BTwind_temperature',
                   '1Coal_ratio', '2Coal_ratio',
                   '1Coke_ratio', '2Coke_ratio',
                   '1CO2_contentofgas', '2CO2_contentofgas',
                   '1TRTGenerationrate', '2TRTGenerationrate',
                   'Hotmetal_ratio']
    # 高炉能源数据
    data = pd.read_csv("gaolu_csv.csv")
    # 缺失值替换
    data = data.replace(to_replace="?", value=np.nan)
    # 缺失值删除
    data.dropna(inplace=True)
    # 源特征值
    x = data.iloc[1:, 3:]
    #print(x)
    print('相关系数为\n')
    #循环计算相关系数
    for i in range(len(factor)):
        for j in range(i,len(factor)-1):
            print(factor[i],factor[j],
                  pearsonr(x[factor[i]],x[factor[j]]))

    return None


#PCA特征提取
def PCA_demo():
    # 高炉能源数据
    data = pd.read_csv("gaolu_csv.csv")
    # 缺失值替换
    data = data.replace(to_replace="?", value=np.nan)
    # 缺失值删除
    data.dropna(inplace=True)
    # 源特征值
    x = data.iloc[1:, 3:]
    transfer = PCA(n_components=0.99)
    data_new = transfer.fit_transform(x)
    print(data_new)
    print(data.shape,data_new.shape)

#正规方程
def liner_demo1():
    column_name = ['高炉工序能耗', '1高炉工序能耗',
                   '2高炉工序能耗', '高炉折煤气用量',
                   '高炉铁水产量', '高炉TRT发电量',
                   '高炉煤气发生量', '高炉喷吹煤用量',
                   '高炉焦炭使用量', '高炉鼓风用量',
                   '1高炉BT风温', '2高炉BT风温',
                   '1高炉煤比', '2高炉煤比',
                   '1高炉入炉焦比', '2高炉入炉焦比',
                   '1高炉煤气CO2含量', '2高炉煤气CO2含量',
                   '1#TRT发电率', '2#TRT发电率',
                   '铁水比']
    # 高炉能源数据
    data = pd.read_csv("gaolu_csv.csv",names=column_name)
    # 缺失值替换
    data = data.replace(to_replace="?", value=np.nan)
    # 缺失值删除
    data.dropna(inplace=True)
    #print(data)
    # 筛选特征值和目标值
    x = data.iloc[:, 4:]
    y1 = data.iloc[:,:1]
    y2 = data.iloc[:,1:2]
    y3 = data.iloc[:,2:3]
    #print(y1)
    #print(y2)
    #print(y3)
    #划分训练测试集
    x_train, x_test, y1_train, y1_test = train_test_split(x, y1, random_state=22)
    # 特征工程
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    #预估器
    estimator1 = LinearRegression()
    estimator1.fit(x_train,y1_train)
    #回归参数
    print('正规方程-回归参数\n',estimator1.coef_)
    #偏置
    print('正规方程-偏置值\n',estimator1.intercept_)
    #评估好坏,MSE
    y_predict = estimator1.predict(x_test)
    #print('预测值\n',y_predict)
    error = mean_squared_error(y1_test,y_predict)
    print('mse\n',error)


#梯度下降
def liner_demo2():
    # 高炉能源数据
    data = pd.read_csv("gaolu_csv.csv")
    # 缺失值替换
    data = data.replace(to_replace="?", value=np.nan)
    # 缺失值删除
    data.dropna(inplace=True)
    # 筛选特征值和目标值
    x = data.iloc[:, 4:]
    y1 = data.iloc[:,:1]
    y2 = data.iloc[:,1:2]
    y3 = data.iloc[:,2:3]
    #print(y1)
    #print(y2)
    #print(y3)
    #划分训练测试集
    x_train, x_test, y1_train, y1_test = train_test_split(x, y1, random_state=22)
    # 特征工程
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    #预估器
    estimator1 = SGDRegressor()
    estimator1.fit(x_train,y1_train)
    #回归参数
    print('梯度下降-回归参数\n',estimator1.coef_)
    #偏置
    print('梯度下降-偏置值\n',estimator1.intercept_)
    #评估好坏,MSE
    y_predict = estimator1.predict(x_test)
    #print('预测值\n',y_predict)
    error = mean_squared_error(y1_test,y_predict)
    print('mse\n',error)

def liner_demo3():
    # 高炉能源数据
    data = pd.read_csv("gaolu_csv.csv")
    # 缺失值替换
    data = data.replace(to_replace="?", value=np.nan)
    # 缺失值删除
    data.dropna(inplace=True)
    # 筛选特征值和目标值
    x = data.iloc[:, 4:]
    y1 = data.iloc[:,:1]
    y2 = data.iloc[:,1:2]
    y3 = data.iloc[:,2:3]
    #print(y1)
    #print(y2)
    #print(y3)
    #划分训练测试集
    x_train, x_test, y1_train, y1_test = train_test_split(x, y1, random_state=22)
    # 特征工程
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    #预估器
    estimator1 = Ridge()
    estimator1.fit(x_train,y1_train)
    #回归参数
    print('岭回归-回归参数\n',estimator1.coef_)
    #偏置
    print('岭回归-偏置值\n',estimator1.intercept_)
    #评估好坏,MSE
    y_predict = estimator1.predict(x_test)
    #print('预测值\n',y_predict)
    error = mean_squared_error(y1_test,y_predict)
    print('mse\n',error)




if __name__ == '__main__':
    #kmeans_demo()
    #liner_demo1()
    #liner_demo2()
    #liner_demo3()
    #lowVarianceFeatureExtract_demo()
    #pearsonFeatureExtract_demo()
    PCA_demo()