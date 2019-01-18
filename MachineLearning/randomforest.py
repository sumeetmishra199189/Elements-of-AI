#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Nov 30 15:20:44 2018
@author: Varun Miranda, Supriya AB, Sumeet Mishra
"""

"""
DESCRIPTION:
    
First Approach:

1. In order to implement Random Forest, first a Decision Tree needs to be implemented
2. The approach I have tried earlier was to take a random sample of training data (took top 500 rows)
3. The decision tree function will iterate over various treshold values and the gini index
   will be calculated for every column. 
4. The column with the least disorder is chosen as the root node and values of that column > treshold 
   and < treshold will be subsetted and the same process repeats on both the children till the
   disorder is 0
5. Once the disorder is 0, the process terminates in a leaf node
6. A log is maintained which will have the following entries

TreeNum	ColChosen Parent_Col	 Rule	 Treshold	Disorder	   Size	 Disorder1	   Disorder2
   1	   3	        0	﻿root	    0	   0.673390302	500	  0.6388	         0.6977
   1	  24	        3	greater	   125	   0.180581746	202	  0.503985969	 0.376049383
   1	  99	       24	greater	   115	   0.089034114	111	  0.724952741	 0.17183432
   1	 174	       99	greater	   130	   0.05872	     46	  0.6944	         0.571428571
   1	 192	      174	greater	   150	   0.028	         25	  0.736842105	 0
   1	 121	      192	greater	   135	   0.022380952	 19	  0.694444444	 0.408163265
   1	  62	      121	greater	   150	   0.011085714	 12	  0.48	         0.448979592
   1	   8	       62	greater	   135	   0	          5	  0	             0
   1	 105	       62	 less	   135	   0.002	          7	  0	             0.5
   1	  59	      105	 less	   110	   0	          2	  0	             0

7. Problem with this approach: Many decision trees will have separate tables like this and 
   imposing a prediction with test data and implementing max voting will take time

Second Approach:
    
Citation: Implementation of Question, Leaf and Decision Node classes and general idea from Google Developers
Github Code Link: https://github.com/random-forests/tutorials/blob/master/decision_tree.ipynb
    
1. The algorithm is devised using the three classes: Question: that uses all the unique treshold values (0,255)
   and returns a boolean value if the value in the column is greater than the treshold, Leaf: once the tree 
   terminates in the leaf node, it's frequency of the class variables is computed, Decision Node: references the 
   question, and it's child nodes.
2. The main function of this algorithm is the build tree function, initially the depth of the tree is given
   as 0, and then it's incremented upto a max depth, beyond which the tree will terminate into a leaf.
3. To splitting the data based on the best treshold value and the best column, the find best split function is used
   which will call the gini and the information gain functions which determines the best way to split the data.
   The split is performed by the partition function.
4. The function classify takes the test data and performs the rules that it learnt from the training data and 
   predicts the outcome.
5. This entire program is done for one decision tree. Random forest takes multiple decision trees on different
   samples of training data.
6. Max Voting is then done to take the majority of the decisions into account
7. Accuracy will then be printed

Citation: Implementation of Print Tree from Google Developers

Sample print tree function can be obtained in this format for one decision tree:
    
Is b58 >= 194?
--> True:
  Is g38 >= 139?
  --> True:
    Predict {90: 27}
  --> False:
    Is b88 >= 253?
    --> True:
      Predict {90: 1}
    --> False:
      Predict {0: 6}
--> False:
  Is b78 >= 163?
  --> True:
    Is b48 >= 161?
    --> True:
      Is b88 >= 212?
      --> True:
        Predict {0: 2}
      --> False:
        Predict {90: 2}
    --> False:
      Predict {180: 29}
  --> False:
    Is b27 >= 167?
    --> True:
      Is g25 >= 90?
      --> True:
        Predict {0: 20, 90: 1, 270: 1, 180: 1}
      --> False:
        Predict {90: 4}
    --> False:
      Is b61 >= 154?
      --> True:
        Predict {270: 25, 90: 3}
      --> False:
        Predict {180: 22, 0: 15, 270: 29, 90: 12}

"""

"Library files"

import pickle
import sys
import math
import pandas as pd
import numpy as np

"Command line arguments"

argtype = sys.argv[1] 
filetype = sys.argv[2] 
modelfile = sys.argv[3] 
model = sys.argv[4] 

"Parameters"

trees = 2
samp_size = 100
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

"Reading the training file"

if model.lower() == 'forest':
    
    if argtype.lower() == 'train':
    
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
        
    elif argtype.lower() == 'test':    
        
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
        
        print("Incorrect Arguments")

else:
    
    print("Incorrect Arguments")