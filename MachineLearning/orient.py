#!/usr/bin/env python3
# orient.py : Image orientation detection
# Supriya Ayalur Balasubramanian (sayalurb)
# Sumeet Mishra (sumish)
# Varun Miranda


import sys
import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt
import pandas as pd
import math
import pickle

################################################# KNN #################################################
def euclideanDistance(x, y):
#    return np.sqrt(np.sum(np.square(x - y)))
    return np.sum(np.absolute(x - y))

def train_knn(filetype, modelfile):
    traindata=[]
    print("Loading the model file")
    mfile=open(modelfile,'w')
    data=open(filetype,'r').readlines()
    for each in data:
        traindata.append(each.split())
        mfile.write(each)
    print("Model file loaded\n")
    return traindata

def test_knn(train,test,modelfile):
    
    testdata=[]
    for each in test:
        testdata.append(each[2:])
    testdata_knn=np.array(testdata).astype(int)
    
    trainlabels=[]
    modeldata=[]
    for line in train:
        trainlabels.append(line[0:2])
        modeldata.append(line[2:])
    
    labels=np.vstack(trainlabels)
    traindata_knn=np.array(modeldata).astype(int)
    
    m=np.shape(traindata_knn)[0]
    n=np.shape(testdata_knn)[0]

    print("Calculating Euclidean Distance")
    dists = np.zeros(shape = (m,n))
    for i in range(m):
        for j in range(n):
            dists[i,j]=euclideanDistance(traindata_knn[i,:], testdata_knn[j,:])
    print("Euclidean Distance calculated\n")
    
    return dists, labels
   
def sortlabels(knnmodel, labels):
    print("Sorting the labels")
    train_labs = labels[:,1].astype(int)
    list_sorted_labels = []
    for i in range(len(testdata)):
        matrix = np.vstack((knnmodel[:,i], train_labs))
        matrix = matrix.T
        matrix_sort = matrix[np.lexsort(np.fliplr(matrix).T)]
        list_sorted_labels.append(matrix_sort[:,1])
    print("Labels sorted\n")
    return list_sorted_labels


def predictlabels(sorted_labels,testdata):
    print("Predicting the labels")
    k_range = [k for k in range(1,500)]
    percent=[]
    for k in k_range:
        labelpred = []
        for i in range(len(testdata)):
            res = mode(sorted_labels[i][:k])
            labelpred.append(res[0][0])
        pred_labels = [int(i) for i in labelpred]
        test_labels = [int(testdata[i][1]) for i in range(len(testdata))]
        photo_id = [testdata[i][0] for i in range(len(testdata))]
        count = 0
        for pred,actual in zip(test_labels,pred_labels):
            if pred == actual:
                count+=1
        score=(count/len(test_labels))*100
        percent.append(score)

        mfile=open('nearest_output.txt','w+')
        for id,orient in zip(photo_id, pred_labels):
            output=id+' '+str(orient)+'\n'
            mfile.write(output)
                       
    plt.plot(k_range, percent)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Testing Accuracy')  
    
    maxpercent=max(percent)
    index=percent.index(maxpercent)
    print(round(maxpercent,2),"% Accuracy achieved at K =",k_range[index],"for KNN")

################################################# KNN #################################################

################################################# ADABOOST #################################################
    
def gen_classifier():
    cf=500
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

################################################# ADABOOST #################################################

################################################# DECISION FOREST #################################################

"Parameters"

trees = 50
samp_size = 5000
max_depth = 5

"Functions involved in the construction of a decision tree"
"Reference: Github Code Link for defining three classes"
        
class Question:

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        val = example[self.column]
        return val >= self.value
    
    def __repr__(self):
        condition = ">="
        return "Is %s %s %s?" % (header[self.column], condition, str(self.value))

class Leaf:
    def __init__(self, rows):
        self.predictions = frequency(rows)

class Decision_Node:

    def __init__(self,question,true_branch,false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

"The three classes are thus defined which are used in the functions below"

def frequency(data):
    counts = {}
    for row in data:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

def partition(data, question):
    greater_than = []
    less_than = []
    for row in data:
        if question.match(row):
            greater_than.append(row)
        else:
            less_than.append(row)
    return greater_than, less_than

def gini(data):
    counts = frequency(data)
    gini_index = 1
    for label in counts:
        prob_of_label = counts[label] / float(len(data))
        gini_index = gini_index - math.pow(prob_of_label,2)
    return gini_index

def shannon(data):
    counts = frequency(data)
    entropy = 0
    for label in counts:
        prob_of_label = counts[label] / float(len(data))
        entropy = entropy - 1 * (prob_of_label) * (math.log((prob_of_label),2) if prob_of_label != 0 else 0)
    return entropy

def info_gain(left, right, current):
    p = float(len(left)) / (len(left) + len(right))
    return current - p * gini(left) - (1 - p) * gini(right)

#def info_gain(left, right, current):
#    p = float(len(left)) / (len(left) + len(right))
#    return current - p * shannon(left) - (1 - p) * shannon(right)

def find_best_split(data):
    best_gain = 0  
    best_question = None  
    current_uncertainty = gini(data)
    n_features = len(data[0]) - 1  
    for col in range(n_features): 
        values = set([row[col] for row in data])  
        for val in values:  
            question = Question(col, val)
            true_rows, false_rows = partition(data, question)
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue
            gain = info_gain(true_rows, false_rows, current_uncertainty)
            if gain >= best_gain:
                best_gain = gain 
                best_question = question
    return best_gain, best_question


def build_tree(data,depth):
    depth = depth + 1
    if depth < max_depth:
        gain, question = find_best_split(data)
        if gain == 0:
            return Leaf(data)
        true_rows, false_rows = partition(data, question)
        true_branch = build_tree(true_rows,depth)
        false_branch = build_tree(false_rows,depth)
        return Decision_Node(question, true_branch, false_branch)
    else:
        return Leaf(data)

def classify(row, node):
    if isinstance(node, Leaf):
        return node.predictions
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)

def print_tree(node, spacing=""):
    if isinstance(node, Leaf):
        print (spacing + "Predict", node.predictions)
        return
    
    print (spacing + str(node.question))

    print (spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    print (spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")


################################################# DECISION FOREST #################################################

# Input parameters
argtype=sys.argv[1] # train or test
filetype=sys.argv[2] # train-data.txt or test-data.txt
modelfile=sys.argv[3] #<model>_model.txt
model=sys.argv[4] #nearest or adaboost or forest or best


if argtype=="train":    
    
    if model == "nearest":
        traindata=train_knn(filetype, modelfile)
    
    elif model == "adaboost":
        colnames=[i for i in range(-2,192)]
        train_file=pd.read_csv('train-data.txt',header=None,delimiter=' ',names=colnames)
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
    
    elif model == "forest" or model == "best":
        train = pd.read_csv(filetype, sep = ' ', header = None)
        num_cols_train = train.shape[1] - 2
        train.columns = [i for i in range(-1,num_cols_train+1)]
        train_trim = train.drop(columns = [-1,0])
        train_trim = train_trim.join(train[0])
        
        "Variables"
        node_array = {}
        matrix_size = int(math.sqrt((num_cols_train)/3))
        header = []
        for i in range(1,matrix_size+1):
            for j in range(1,matrix_size+1):
                header.append("r"+str(i)+str(j))
                header.append("g"+str(i)+str(j))
                header.append("b"+str(i)+str(j))
        
        "Running Random Forest"
        decision_tree = [0]*trees
        model_file = open(modelfile,"wb")
        
        for i in range(0,trees):    
            print("tree:",i+1)
            rand_sample = train_trim.sample(n = samp_size,replace = False)
            rand_sample = np.asarray(rand_sample)
            training_data = rand_sample
            decision_tree[i] = build_tree(training_data,0)
        
            "Writing into forest_model.txt"
            pickle.dump(decision_tree[i],model_file)
        model_file.close()

elif argtype=="test":
    testdata=[]
    print("Loading the testing data")
    testfile=open(filetype,'r').readlines()
    for each in testfile:
        testdata.append(each.split())
    print("Testing data loaded\n")
    
    if model == "nearest":
        
        (knn_model, labels)=test_knn(traindata,testdata,modelfile)
        sorted_labels = sortlabels(knn_model, labels)
        predictlabels(sorted_labels,testdata)
        
    elif model == "adaboost":
        colnames=[i for i in range(-2,192)]
        test_file=pd.read_csv('test-data.txt',header=None,delimiter=' ',names=colnames)
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
        print(round(((count/len(result))*100),2),"% Accuracy achieved for Adaboost")
#        print('The accuracy = '+str((count/len(result))*100)) 

    elif model == "forest" or model == "best":
        
        "Test_data"
        test = pd.read_csv(filetype, sep = ' ', header = None)
        num_cols_test = test.shape[1] - 2
        test.columns = [i for i in range(-1,num_cols_test+1)]
        test_trim = test.drop(columns = [-1,0])
        testing_data = np.asarray(test_trim)
        model_file = open(modelfile,"rb")
        model_file.seek(0)
        
        node_array = {}
        for i in range(0,trees):
            node_array[i] = []
            decision_model = pickle.load(model_file)
            for row in testing_data:
                node = classify(row,decision_model)
                node_array[i].append(max(node, key=node.get))
        model_file.close()
        
        "Max Voting"
        max_vote = [0]*len(node_array[0])
        for i in range(0,len(node_array[0])):
            zero = 0
            ninety = 0
            one_eighty = 0
            two_seventy = 0
            for j in range(0,len(node_array)):
                if(node_array[j][i] == 0):
                    zero+=1
                elif(node_array[j][i] == 90):
                    ninety+=1
                elif(node_array[j][i] == 180):
                    one_eighty+=1
                else:
                    two_seventy+=1  
            if max(zero,ninety,one_eighty,two_seventy) == zero:
               max_vote[i] = 0
            elif max(zero,ninety,one_eighty,two_seventy) == ninety:
               max_vote[i] = 90
            elif max(zero,ninety,one_eighty,two_seventy) == one_eighty:
               max_vote[i] = 180
            else:
               max_vote[i] = 270
                     
        "Accuracy and storing the values in an output file"
        output_file = open('output_forest.txt',"w+")
        correct = 0
        total = len(test)
        for i in range(0, len(max_vote)):
            output_file.write(test[-1][i]+" "+str(max_vote[i])+"\n")
            if max_vote[i] ==  test[0][i]:
                correct+=1
        output_file.close()
        print ("The Accuracy of the Random Forest Model is",round(correct/total,4)*100,"%")
    else:
        print("Enter one of the model names: nearest, adaboost, forest, best")
else:
    print("Incorrect value for type parameter")
   
        
    
    
    
