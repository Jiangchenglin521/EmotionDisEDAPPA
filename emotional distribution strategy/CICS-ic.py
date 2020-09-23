'''
This code is for emotional score generation and emotional distriution genenration
New version: developed by Chenglin Jiang (cljiangjiang@gmail.com)
'''
import jieba
import numpy as np
import gzip
import jieba
import re
import json
import tarfile
import configparser
import scipy.stats
import pickle
import os
import csv
import time
import datetime
import random
import json
import math
import warnings
from collections import Counter
from math import sqrt
from tensorflow.python.platform import gfile
import gensim
import pandas as pd
import numpy as np
import tensorflow as tf
#(1)icount 还没改，要限制在同一句表达中
#(2)多个以句号分句的句子可能不会成功。而且还未查找更具体的断句字符。
#***********************加载EL****************************
int2emotion = ['others', 'like', 'sad', 'disgust', 'angry', 'happy']
nn = ['sad','angry', 'disgust']
pp = ['like', 'happy']
la = []
lines = open('/Users/apple/Desktop/DLUT-Emotionontology-master/分离文件/分类后/other.txt', encoding='utf-8').read().splitlines()
a = [x for x in lines]
la.append(a)
lines1 = open('/Users/apple/Desktop/DLUT-Emotionontology-master/分离文件/分类后/c1.txt', encoding='utf-8').read().splitlines()
b = [x for x in lines1]
la.append(b)
lines2 = open('/Users/apple/Desktop/DLUT-Emotionontology-master/分离文件/分类后/c2.txt', encoding='utf-8').read().splitlines()
c1 = [x for x in lines2]
la.append(c1)
lines3 = open('/Users/apple/Desktop/DLUT-Emotionontology-master/分离文件/分类后/c3.txt', encoding='utf-8').read().splitlines()
d = [x for x in lines3]
la.append(d)
lines4 = open('/Users/apple/Desktop/DLUT-Emotionontology-master/分离文件/分类后/c4.txt', encoding='utf-8').read().splitlines()
e = [x for x in lines4]
la.append(e)
lines5 = open('/Users/apple/Desktop/DLUT-Emotionontology-master/分离文件/分类后/c5.txt', encoding='utf-8').read().splitlines()
f = [x for x in lines5]
la.append(f)
#打开词典文件，返回列表
def open_dict(Dict = 'hahah', path=r'/Users/apple/Desktop/Textming/Sent_Dict/Hownet/'):
    path = path + '%s.txt' % Dict
    dictionary = open(path, 'r', encoding='utf-8')
    dict = []
    for word in dictionary:
        word = word.strip('\n')
        dict.append(word)
    return dict
def judgeodd(num):
    if (num % 2) == 0:
        return 'even'
    else:
        return 'odd'


#注意，这里你要修改path路径。
deny_word = open_dict(Dict = '否定词', path= r'/Users/apple/Desktop/Textming/')
posdict = open_dict(Dict = 'positive', path= r'/Users/apple/Desktop/Textming/')
negdict = open_dict(Dict = 'negative', path= r'/Users/apple/Desktop/Textming/')

degree_word = open_dict(Dict = '程度级别词语', path= r'/Users/apple/Desktop/Textming/')
mostdict = degree_word[degree_word.index('extreme')+1 : degree_word.index('very')]#权重4，即在情感词前乘以4
verydict = degree_word[degree_word.index('very')+1 : degree_word.index('more')]#权重3
moredict = degree_word[degree_word.index('more')+1 : degree_word.index('ish')]#权重2
ishdict = degree_word[degree_word.index('ish')+1 : degree_word.index('last')]#权重0.5
def distribution(sentence, label):
    e_word = []
    init = [0, 0, 0, 0, 0, 0]
    init[label] = 1
    count = [0, 0, 0, 0, 0, 0]
    distr = [0, 0, 0, 0, 0, 0]
    segs = jieba.lcut(sentence, cut_all=False)
    for word in segs:
        if word in a:
            init[0] = 1
            count[0] += 1
            e_word.append(word)
            e_word.append(word)
        elif word in b:
            init[1] = 1
            count[1] += 1
            e_word.append(word)
        elif word in c1:
            init[2] = 1
            count[2] += 1
            e_word.append(word)
        elif word in d:
            init[3] = 1
            count[3] += 1
            e_word.append(word)
        elif word in e:
            init[4] = 1
            count[4] += 1
            e_word.append(word)
        elif word in f:
            init[5] = 1
            count[5] += 1
            e_word.append(word)

    if np.sum(init, 0) ==0 or np.sum(init, 0) ==1:
        init[label] = 1
        return init
    else:
        ban = init
        for i,j in enumerate(init):
            if j == 1:
                ban[i] = scipy.stats.norm(label,1).pdf(i)
        all = 0
        for k in range(6):
            all += ban[k]
        for j in range(6):
            aa = ban[j]/all
            ban[j] = aa
        return ban
lis = json.load(open('/Users/apple/Desktop/Textming/cate1s', encoding='utf-8'))
emo = json.load(open('/Users/apple/Desktop/Textming/cate1e', encoding='utf-8'))
l1, l2, l3, dataar = [],[],[], []
rr = 0.6
for i,j in enumerate(lis):
    distributionss = distribution(j, emo[i])
    l1.append(j)
    l2.append(emo[i])
    nvm = str(distributionss)
    l3.append(nvm)
dataar.append(l1)
dataar.append(l2)
dataar.append(l3)
np_data = np.array(dataar)
np_data = np_data.T
np.array(np_data)
save = pd.DataFrame(np_data, columns=['sentence', 'emotion label', 'distribution'])
save.to_csv('/Users/apple/Desktop/Textming/CICS-ic.csv')
print('over')