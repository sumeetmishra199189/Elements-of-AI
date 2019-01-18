###################################
# CS B551 Fall 2018, Assignment #3
#
# Your names and user ids:
# Supriya - sayalurb
# Sumeet - sumish
# Varun - varmiran
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

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#

initial_prob={}
initial_count={}
prior_count={}
prior_probability={}
trans_count={}
trans_probability={}
emission_count={}
emission_probability={}

#word_count={}
#word_prob={}


#Calculation of initial and prior probabilities
def calculate_initial_prior_probability(data):
	for line in data:
		firstpos=line[1][0]
		initial_count[firstpos]=(initial_count[firstpos]+1 if firstpos in initial_count.keys() else 1)
		for pos in line[1]:
			prior_count[pos]=(prior_count[pos]+1 if pos in prior_count.keys() else 1)
	
	for pos in prior_count.keys():
		prior_probability[pos]=float(prior_count[pos])/sum(prior_count.values())
		initial_prob[pos]=float(initial_count[pos])/sum(initial_count.values())   
  

# Calculation of Transition and emission probabilities
def calculate_trans_emiss_probability(data):
    for pos in prior_probability.keys():
        trans_count[pos]={}
        emission_count[pos]={}
        for line in data:
            for i in range(len(line[1])-1): #Transition
                if pos==line[1][i]:
                    trans_count[pos][line[1][i+1]]=(1 if line[1][i+1] not in trans_count[pos].keys() else trans_count[pos][line[1][i+1]]+1 )
                        
            for (word, wordpos) in zip(line[0],line[1]): #Emission
                if pos==wordpos:
                    emission_count[pos][word]=(1 if word not in emission_count[pos].keys() else emission_count[pos][word]+1)          


    for pos in trans_count.keys():  #Transition probabilities
        trans_probability[pos]={}
        for nextpos in trans_count[pos].keys():
            trans_probability[pos][nextpos]=float(trans_count[pos][nextpos])/sum(trans_count[pos].values())

    for pos in emission_count.keys(): #Emission probabilities
        emission_probability[pos]={}
        for word in emission_count[pos].keys():
            emission_probability[pos][word]=float(emission_count[pos][word])/sum(emission_count[pos].values()) 



#def calculate_emission_probability(data):
#    for pos in prior_probability.keys():
#        emission_count[pos]={}
#        for line in data:
#            for (word, wordpos) in zip(line[0],line[1]):
##                print word, wordpos
#                if pos==wordpos:
#                    emission_count[pos][word]=(1 if word not in emission_count[pos].keys() else emission_count[pos][word]+1)
##                    word_count[word]=(1 if word not in word_count.keys() else word_count[word]+1)
#
#    for pos in emission_count.keys():
#        emission_probability[pos]={}
#        for word in emission_count[pos].keys():
#            emission_probability[pos][word]=float(emission_count[pos][word])/sum(emission_count[pos].values()) 


class Solver:
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    def posterior(self, model, sentence, label):
        if model == "Simple":
            return -999
        elif model == "Complex":
            return -999
        elif model == "HMM":
            return -999
        else:
            print("Unknown algo!")
    
    # Do the training!
    #
    def train(self, data):
        calculate_initial_prior_probability(data)
        calculate_trans_emiss_probability(data)
#        calculate_emission_probability(data)		

    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    #
    def simplified(self, sentence):
        predictedpos=[]
        for word in sentence:
            probposgword={}
            for pos in prior_probability.keys():
                probposgword[pos]={}
#                probposgword[pos]=(math.log(emission_probability[pos][word])+math.log(prior_probability[pos]) if word in emission_probability[pos].keys() else math.log((float(1)/10000000000))+math.log(prior_probability[pos]))
                if word in emission_probability[pos].keys():
                    probposgword[pos]=math.log(emission_probability[pos][word])+math.log(prior_probability[pos])
                else:
                    probposgword[pos]=math.log((float(1)/100000000))+math.log(prior_probability[pos])
            predictedpos.append(max(probposgword.items(),key=operator.itemgetter(1))[0])
        return predictedpos


    def complex_mcmc(self, sentence):
        return [ "noun" ] * len(sentence)

    def hmm_viterbi(self, sentence):
        return [ "noun" ] * len(sentence)


    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        else:
            print("Unknown algo!")

