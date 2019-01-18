###################################
# CS B551 Fall 2018, Assignment #3
#
# Your names and user ids:
# Supriya - sayalurb
# Sumeet - sumish
#
# (Based on skeleton code by D. Crandall)
#
#
####
# Put your report here!!
####

import random
import math
import operator
import copy

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#

initial_count={}
initial_probability={}
prior_count={}
prior_probability={}
trans_count={}
trans_probability={}
emission_count={}
emission_probability={}


# Calculation of Initial, Prior, Transition and Emission probabilities
def calculate_init_trans_emiss_probability(data):
    #Initial and Prior count
    for line in data:
        firstpos=line[1][0]
        initial_count[firstpos]=(initial_count[firstpos]+1 if firstpos in initial_count.keys() else 1)
        for pos in line[1]:
            prior_count[pos]=(prior_count[pos]+1 if pos in prior_count.keys() else 1)
	
    #Initial and Prior probabilities
    for pos in prior_count.keys():
        prior_probability[pos]=float(prior_count[pos])/sum(prior_count.values())
        initial_probability[pos]=float(initial_count[pos])/sum(initial_count.values())       
    
    #Transition and emission count
    for pos in prior_probability.keys():
        trans_count[pos]={}
        emission_count[pos]={}
        for line in data:
            for i in range(len(line[1])-1): #Transition count
                if pos==line[1][i]:                    
                    trans_count[pos][line[1][i+1]]=(1 if line[1][i+1] not in trans_count[pos].keys() else trans_count[pos][line[1][i+1]]+1)                        
            
            for (word, wordpos) in zip(line[0],line[1]): #Emission count
                if pos==wordpos:
                    emission_count[pos][word]=(1 if word not in emission_count[pos].keys() else emission_count[pos][word]+1)          

    #Transition probabilities
    for pos in trans_count.keys():  
        trans_probability[pos]={}
        for nextpos in trans_count[pos].keys():
            trans_probability[pos][nextpos]=float(trans_count[pos][nextpos])/sum(trans_count[pos].values())

    #Emission probabilities
    for pos in emission_count.keys(): 
        emission_probability[pos]={}
        for word in emission_count[pos].keys():
            emission_probability[pos][word]=float(emission_count[pos][word])/sum(emission_count[pos].values()) 

class Solver:
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    def posterior(self, model, sentence, label):
#        sentence=[word.replace("'s","") for word in list(sentence)]
        if model == "Simple":
            psum=0
            for word,pos in zip(sentence,label):
                psum+=math.log(emission_probability[pos][word]) + math.log(prior_probability[pos])
            return psum
        elif model == "Complex":
            return -999
        elif model == "HMM":
            return -999
        else:
            print("Unknown algo!")
    
    # Do the training!
    #
    def train(self, data):
        calculate_init_trans_emiss_probability(data)		

    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    # Simplified Bayes Net
    def simplified(self, sentence):
        predictedsimplepos=[]
#        sentence=[word.replace("'s","") for word in list(sentence)]
        for word in sentence:
            probposgword={}
            for pos in prior_probability.keys():
                probposgword[pos]={}
                if word not in emission_probability[pos].keys():
                    emission_probability[pos][word]=1e-9
                probposgword[pos]=math.log(emission_probability[pos][word]) + math.log(prior_probability[pos])
            predictedsimplepos.append(max(probposgword.items(),key=operator.itemgetter(1))[0])
        return predictedsimplepos

    # HMM Viterbi
    # Reference: https://en.wikipedia.org/wiki/Viterbi_algorithm
    def hmm_viterbi(self, sentence):
        V = {}
       # Calculating for t=0
        V[0]={}
        for state in prior_probability.keys():
            V[0][state]={"prob": (initial_probability[state]) * (emission_probability[state][sentence[0]]), "prev": None}
       
        # Run Viterbi when t > 0
        for t in range(1, len(sentence)):
            V[t]={}
            for currst in prior_probability.keys():
                max_prob=max([(V[t-1][prevst]["prob"]*1e-9,prevst) if (currst not in trans_probability[prevst].keys()) else (V[t-1][prevst]["prob"]*trans_probability[prevst][currst],prevst) for prevst in prior_probability.keys()])
#                p=[]
#                for prevst in prior_probability.keys():
#                    if (currst not in trans_probability[prevst].keys()):
#                        p.append((V[t-1][prevst]["prob"]*1e-9,prevst))
#                    else:
#                        p.append((V[t-1][prevst]["prob"]*trans_probability[prevst][currst],prevst))
#                max_prob=max(p)
                V[t][currst] = {"prob": max_prob[0] * (emission_probability[currst][sentence[t]]), "prev": max_prob[1]}
        
        # Back-tracking
        prob = max((value["prob"],value["prev"]) for value in V[len(sentence)-1].values())
        for pos in prior_probability.keys():
            if V[len(sentence)-1][pos]["prob"]==prob[0]:
                lastpos=pos
        
        # State Sequence
        hmmseq=[]
        hmmseq.append(lastpos)
        for t in range(len(sentence)-1,0,-1):
            hmmseq.append(V[t][lastpos]["prev"])
            lastpos=V[t][lastpos]["prev"]
        hmmseq.reverse()
        
        return hmmseq
    

    def complex_mcmc(self, sentence, sample_count):
        sentence_len = len(sentence)
        sample=['noun']*sentence_len
        max_marg=[]
        for n in range(0,sample_count+1000):  # ignore first 1000 samples as warm up iterations
            for i in range(sentence_len):
                word = sentence[i]
                prob=[]
                
                if i==0:
                    for pos in prior_probability.keys():
                        if len(sentence)>1:
                            tp=(trans_probability[pos][sample[i-1]] if pos in trans_probability.keys() and sample[i-1] in trans_probability[pos].keys() else 1e-9)
                            prob.append((prior_probability[pos]*emission_probability[pos][word]*tp, pos))
                        else:
                            prob.append((prior_probability[pos]*emission_probability[pos][word], pos))
                elif i==sentence_len-1:
                    for pos in prior_probability.keys():
                        tp=(trans_probability[sample[i-1]][pos] if sample[i-1] in trans_probability.keys() and pos in trans_probability[sample[i-1]].keys() else 1e-9)
                        prob.append((prior_probability[sample[i-1]]*emission_probability[pos][word]*tp, pos))
                else:
                    for pos in prior_probability.keys():
                        tp1=(trans_probability[sample[i-1]][pos] if sample[i-1] in trans_probability.keys() and pos in trans_probability[sample[i-1]].keys() else 1e-9)
                        tp2=(trans_probability[pos][sample[i+1]] if pos in trans_probability.keys() and sample[i+1] in trans_probability[pos].keys() else 1e-9)
                        prob.append((prior_probability[sample[i-1]]*emission_probability[pos][word]*tp1*tp2, pos))                       

                maxmarg=max(prob)
                maxprob=maxmarg[0]
                sample[i]=maxmarg[1]
                probsum=0.0
                for j in range(len(prob)):
                    probsum+=prob[j][0]                
                max_marg.append(round(maxprob/probsum,2))
            
        return sample


#                    
#                    for j in range(0,len(prob)):
#                        p_sum+=prob[j]/sum(prob)
#                        if rand < p_sum:
#                            sample[i] = prob[1][j]
#                            break
#            s.append(sample)
#
##        mcmcpos=[]
#        print(s)
#        print(len(s))
#        print(s[len(s)-sample_count:])
#        for i in s[len(s)-sample_count:]:
#            print("i is:",i)
#            mcmcpos.append(i)
#        
#        print(mcmcpos)
#        return mcmcpos
       
        
    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
#        sentence=[word.replace("'s","") for word in list(sentence)]
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence,5)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        else:
            print("Unknown algo!")

