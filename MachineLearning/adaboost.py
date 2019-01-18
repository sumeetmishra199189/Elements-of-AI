#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 23:04:46 2018

@author: sumeetmishra
"""
import pandas as pd
import numpy as np
import math
import pickle
import sys

argtype = sys.argv[1] 
filetype = sys.argv[2] 
modelfile = sys.argv[3] 
model = sys.argv[4] 

def gen_classifier():
    cf=1000
    ar=[]
    for i in range(cf):
        c1=np.random.randint(192,size=2)
        ar.append(list(c1))
    return ar    

def adaboost(lim1,lim2,d,tf):
    labels = [lim1, lim2]
    tf['weights']=(1/36976)
    tf1=tf[tf.labels.isin(labels)]
    tf1['dummy']=tf1['labels']
    tf1['dummy'].replace([lim1,lim2],[-1,1],inplace=True)
    #print(tf1)
    tf1=np.array(tf1)
    clf={}
    for j in d:
        classifier1=[]
        error1=0
        wt=[]
        for i in range(len(tf1[:,-2])):
            
            if(tf1[i,j[0]]>=tf1[i,j[1]]):
                classifier1.append(-1)
            else:    
                classifier1.append(1) 
     
        for i in range(len(classifier1)):        
                if classifier1[i]!= tf1[i,-1]:
                    error1=error1+tf1[i,-2]
                else:
                    wt.append(i)            
        if error1<0.8:
            er=error1/(1-error1)
            a1=math.log(1/er)            
                 
            for i in wt:
                   tf1[i,-2]*=er
            sum_wt=sum(tf1[:,-2])
            for i in range(len(tf1[:,-2])):
                tf1[i,-2]=tf1[i,-2]/sum_wt
     

            clf[(j[0],'>',j[1])]=a1
       
    return clf     


def pred(dict,lim1,lim2,tf):
    #print(sample)
    tf1=np.array(tf)
    output=[]
    
    for j in tf1:
        h1=0
        #print(j)
        for i in dict.keys():
            if j[i[0]]>j[i[2]]:
                h1+=dict[i]*(-1)
            else:
                h1+=dict[i]*(1)
        if h1<0:
            output.append(lim1)
        else:
            output.append(lim2)
    return output

if model.lower() == 'adaboost':
    if argtype.lower() == 'train':
        colnames=[i for i in range(-2,192)]
        train_file=pd.read_csv('/Users/sumeetmishra/Desktop/Elements_of_AI/Assignments/Assignment4/train-data.txt',header=None,delimiter=' ',names=colnames)
        tf=train_file.drop([-2], axis=1)
        tf.loc[:,'labels']=tf[-1]
        tf=tf.drop([-1], axis=1)
        d=gen_classifier()
        dict_1=adaboost(0,90,d,tf)
        dict_2=adaboost(0,180,d,tf)
        dict_3=adaboost(0,270,d,tf)
        dict_4=adaboost(90,180,d,tf)
        dict_5=adaboost(90,270,d,tf)
        dict_6=adaboost(180,270,d,tf)
        model_file = open(modelfile,"wb")
        pickle.dump(dict_1,model_file)
        pickle.dump(dict_2,model_file)
        pickle.dump(dict_3,model_file)
        pickle.dump(dict_4,model_file)
        pickle.dump(dict_5,model_file)
        pickle.dump(dict_6,model_file)
        model_file.close()

    elif argtype.lower() == 'test':
        colnames=[i for i in range(-2,192)]
        test_file=pd.read_csv('/Users/sumeetmishra/Desktop/Elements_of_AI/Assignments/Assignment4/test-data.txt',header=None,delimiter=' ',names=colnames)
        test=test_file.drop([-2], axis=1)
        test['labels']=test[-1]
        
        test=test.drop([-1], axis=1)
        model_file = open(modelfile,"rb")
        model_file.seek(0)
        dict_1 = pickle.load(model_file)
        dict_2 = pickle.load(model_file)
        dict_3 = pickle.load(model_file)
        dict_4 = pickle.load(model_file)
        dict_5 = pickle.load(model_file)
        dict_6 = pickle.load(model_file)
        model_file.close()
        pred1=pred(dict_1,0,90,test)
        pred2=pred(dict_2,0,180,test)
        pred3=pred(dict_3,0,270,test)
        pred4=pred(dict_4,90,180,test)
        pred5=pred(dict_5,90,270,test)
        pred6=pred(dict_6,180,270,test)
        
        result=[]
        for i in range(len(pred1)):
            zero=0
            ninety=0
            oneeighty=0
            twoseventy=0
            
            if pred1[i]==0:
                zero+=1
            else:
                ninety+=1
            
            if pred2[i]==0:
                zero+=1
            else:
                oneeighty+=1
            
            if pred3[i]==0:
                zero+=1
            else:
                twoseventy+=1
            
            if pred4[i]==90:
                ninety+=1
            else:
                oneeighty+=1 
            
            if pred5[i]==90:
                ninety+=1
            else:
                twoseventy+=1
            
            if pred6[i]==180:
                oneeighty+=1
            else:
                twoseventy+=1 
        
            if max(zero,ninety,oneeighty,twoseventy)==zero:
                result.append(0)
            elif max(zero,ninety,oneeighty,twoseventy)==ninety:
                result.append(90)
            elif max(zero,ninety,oneeighty,twoseventy)==oneeighty:
                result.append(180)
            else:
                result.append(270)    
        
        count=0
        output_file = open('adaboost_output.txt',"w+")
        for i,row in test.iterrows():
            output_file.write(test_file.loc[i,-2]+" "+str(result[i])+"\n")
            if result[i]==test.loc[i,'labels']:
                count+=1
        output_file.close()        
        
        print('The accuracy = '+str((count/len(result))*100))        
        