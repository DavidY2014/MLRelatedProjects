# -*- coding: utf-8
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import jieba

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






if __name__ == '__main__':
    #count_demo()
    count_chinese_demo2()
