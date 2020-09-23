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

#注意，这里你要修改path路径。
deny_word = open_dict(Dict = '否定词', path= r'/Users/apple/Desktop/Textming/')
posdict = open_dict(Dict = 'positive', path= r'/Users/apple/Desktop/Textming/')
negdict = open_dict(Dict = 'negative', path= r'/Users/apple/Desktop/Textming/')

degree_word = open_dict(Dict = '程度级别词语', path= r'/Users/apple/Desktop/Textming/')
mostdict = degree_word[degree_word.index('extreme')+1 : degree_word.index('very')]#权重4，即在情感词前乘以4
verydict = degree_word[degree_word.index('very')+1 : degree_word.index('more')]#权重3
moredict = degree_word[degree_word.index('more')+1 : degree_word.index('ish')]#权重2
ishdict = degree_word[degree_word.index('ish')+1 : degree_word.index('last')]#权重0.5
def distribution(sentence, label, r):

    #print('score = ', score1)
    e_word = []
    init = [0,0,0,0,0,0]
    init[label] = 1
    count = [0,0,0,0,0,0]
    distr = [0, 0, 0, 0, 0, 0]
    segs = jieba.lcut(sentence, cut_all=False)
    for word in segs:
        if word in a :
            init[0] = 1
            count[0] += 1
            e_word.append(word)
            e_word.append(word)
        elif word in b :
            init[1] = 1
            count[1] += 1
            e_word.append(word)
        elif word in c1 :
            init[2] = 1
            count[2] += 1
            e_word.append(word)
        elif word in d :
            init[3] = 1
            count[3] += 1
            e_word.append(word)
        elif word in e :
            init[4] = 1
            count[4] += 1
            e_word.append(word)
        elif word in f :
            init[5] = 1
            count[5] += 1
            e_word.append(word)

    if np.sum(init, 0) == 0:
        print('sum == 0/1')
        init[label] =1
        return init
    if np.sum(init, 0) == 1:
        return init




    if np.sum(init, 0) > 1 and init[label] == 1:

        print('np.sum(init, 0) > 1 and init[label] ==1')
        u = 0
        v = 0
        for w in e_word:
            if w in posdict:
                u+=1
            if w in negdict:
                v+=1
#A
        if u ==0 or v ==0:
            print('u ==0 or v ==0')
#a
            if np.sum(init, 0) ==2:
                ban = init
                ban[label]=0
                init[label] = r +(1-r)*(count[label]/(count[label]+ count[ban.index(1)]))
                init[ban.index(1)] = 1 - init[label]
                return init
#b
            if np.sum(init, 0) > 2:
                ban = init
                ban[label]=0
                all = 0
                for c in range(6):
                    if init[c] ==1:
                        all+=count[c]
                init[label] = r + (1-r)*(count[label]/all)
                for k in range(6):
                    if init[k] ==1:
                        ban[k] = (1-init[label]) * (count[k]/all)

                ban[label] = init[label]
                return ban
#B
        if u!=0 and v!=0:
            print('u!=0 and v!=0')
#a
            if np.sum(init, 0) ==2:

                for i in init:
                    if i != label and  i==1:
                        init[init.index(i)] = 1-r
                init[label] = r
                return init
#b
            if np.sum(init, 0) > 2:
                print('np.sum(init, 0) > 2')

                postip = (1-r)*r
                negtip = 1-r - postip
                # print('postip: ',postip)
                # print('negtip', negtip)
                pall =0
                nall =0
                for t in pp:
                    if t != int2emotion[label]:
                        pall += count[int2emotion.index(t)]

                for l in nn:
                    if l != int2emotion[label]:
                        nall += count[int2emotion.index(l)]
                # print(pall)
                # print(nall)
                for i,j in enumerate(init):
                    # print(i)
                    # print(j)
                    if j==1 and i != label and int2emotion[i] in pp:
                        # print('count', count[i])
                        init[i] = postip * (count[i]/pall)
                    if j==1 and i != label and int2emotion[i] in nn:
                        # print('count', count[i])
                        init[i] = negtip * (count[i]/nall)

                init[label] = r
                return init

#cond.4
        if np.sum(init, 0) > 1 and init[label] == 0:
            print('np.sum(init, 0) > 1 and init[label] == 0')
            u = 0
            v = 0
            for w in e_word:
                if w in posdict:
                    u += 1
                if w in negdict:
                    v += 1
            # A
            if u == 0 or v == 0:
                # a
                if np.sum(init, 0) == 2:
                    ban = init
                    ban[label] = 0
                    init[label] = r
                    init[ban.index(1)] = 1 - init[label]
                    return init
                # b
                if np.sum(init, 0) > 2:
                    ban = init
                    ban[label] = r
                    all = 0
                    for c in range(6):
                        if init[c] == 1 and c!=label:
                            all += count[c]

                    for k in range(6):
                        if ban[k] == 1:
                            ban[k] = (1 - init[label]) * (count[k] / all)


                    return ban
            # B
            if u != 0 and v != 0:
                # a
                if np.sum(init, 0) == 2:
                    init[label] = r
                    for i in init:
                        if i != label and init[i] == 1:
                            init[i] = 1 - r
                    return init
                # b
                if np.sum(init, 0) > 2:
                    # print('np.sum(init, 0) > 2')

                    postip = (1 - r) * (1-r)
                    negtip = 1 - r - postip
                    # print('postip: ', postip)
                    # print('negtip', negtip)
                    pall = 0
                    nall = 0
                    for t in pp:
                        if t != int2emotion[label]:
                            pall += count[int2emotion.index(t)]

                    for l in nn:
                        if l != int2emotion[label]:
                            nall += count[int2emotion.index(l)]
                    # print(pall)
                    # print(nall)
                    for i, j in enumerate(init):
                        print(i)
                        print(j)
                        if j == 1 and i != label and int2emotion[i] in pp:
                            print('count', count[i])
                            init[i] = postip * (count[i] / pall)
                        if j == 1 and i != label and int2emotion[i] in nn:
                            print('count', count[i])
                            init[i] = negtip * (count[i] / nall)

                    init[label] = r
                    return init

lis = json.load(open('/Users/apple/Desktop/Textming/cate1s', encoding='utf-8'))
emo = json.load(open('/Users/apple/Desktop/Textming/cate1e', encoding='utf-8'))
l1, l2, l3, dataar = [],[],[], []
rr = 0.6
for i,j in enumerate(lis):
    distributionss = distribution(j, emo[i], rr)
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
save.to_csv('/Users/apple/Desktop/Textming/EDA-p.csv')
print('over')