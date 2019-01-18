###################################
# CS B551 Fall 2018, Assignment #3
#
# Your names and user ids:
# Supriya Ayalur Balasubramanian - sayalurb
# Sumeet Mishra - sumish
#
# (Based on skeleton code by D. Crandall)
#
#

'''
**************************Report************************
1. Training the dataset bc.train:
    Used dictionaries to store the counts and calculate the below probabilites:
    1. Initial Probability - Calculates the probability of a POS tag occurring in the beginning of the sentence, P(S_1)
    2. Prior Probability - Calculates the probability of a POS tag occurring in the entire training dataset, P(S_i)
    3. Emission Probability - Calculates the probability of the word given the POS tag, P(W_i|S_i)
    4. Transition probability - Calculates the probability of the next POS tag given its previous POS tag, P(S_i+1|S_i)
    5. Transition3 probability - Calculates the probability of P(S_i+2|S_i+1, S_i) i.e, for eg: P(S_3|S_2, S_1)

2. Assumed a small proability of 1e-10, if in case a state/POS does not exist in the training dataset.

3. Simplified model:
    Implemented Naive bayes algorithm to predict the pos sequence for the simplified model.
    P(S_i|W_i) = P(W_i|S_i) * P(S_i)
    where, P(W_i|S_i) = emisison probability, and
           P(S_i) = prior probability

4. HMM Viterbi model:
	a. I am using a nested dictionary within a dictionary to store the probabilites and its previous state.
	b. Initially, the probabilities of the state at t=0 is calculated for every POS tag. 
	c. Then, the probabilities of the state for t>1 is calculated as below:
		i. Product of the transition probability from the previous state to the current state with the value of the previous state, i.e t-1 for each POS tag.
		ii. Multiply the emission probability with the max value of the product obtained from the previous step (Step c.i)
	d. Storing the probabilites and the corresponding POS tag in the dictionary.
	e. For finding the predicted POS sequence, backtrack from the last word in the sentence by pick the most probable state sequence.

5. Complex MCMC model (Gibbs Sampling):
    
    
    
    
    
6. Posterior probability:
    
'''

import random
import math
import operator

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
trans3_count={}
trans3_probability={}
pos_vi={} 
pos_mc={}
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
            
            
    # Three Transition count
    for pos in prior_probability.keys():
        trans3_count[pos]={}
        for line in data:
            for i in range(len(line[1])-2):
                nextpos=line[1][i+1]
                if nextpos not in trans3_count[pos].keys():
                    trans3_count[pos][nextpos]={}
                else:
                    if pos==line[1][i]:                    
                        trans3_count[pos][nextpos][line[1][i+2]]=(1 if line[1][i+2] not in trans3_count[pos][nextpos].keys() else trans3_count[pos][nextpos][line[1][i+2]]+1)
                    
    # Three Transition probabilities
    for pos in trans3_count.keys():  
        trans3_probability[pos]={}
        for pos1 in trans3_count[pos].keys():
            trans3_probability[pos][pos1]={}
            for pos2 in trans3_count[pos][pos1].keys():
                trans3_probability[pos][pos1][pos2]=float(trans3_count[pos][pos1][pos2])/sum(trans3_count[pos][pos1].values())
            

class Solver:
    
    def __init__(self):
        self.smooth_prob=1/1000000000
    
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
            psum=0
            p=1
            for i in range(len(sentence)):
                   
                    currpos=label[i]
                    word=sentence[i]
                    ep = emission_probability[currpos][word] if currpos in emission_probability.keys() and word in emission_probability[currpos].keys() else self.smooth_prob
                    p*=ep
                    
                    if i==0:
                        ip = initial_probability[label[i]]
                        p *= ip
                        #pos_mc[label[i]]=p
                    elif i==1:
                        prevpos=label[i-1]
                        #pp = prior_probability[prevpos]
                        tp = trans_probability[prevpos][currpos] if prevpos in trans_probability.keys() and currpos in trans_probability[prevpos].keys() else self.smooth_prob
                        p*=tp
                        #pos_mc[label[i]]=p
                    else:
                    
                        prevprevpos=label[i-2]
                        prevpos=label[i-1]
                        #pp = prior_probability[prevprevpos]
                        tp3 = trans3_probability[prevprevpos][prevpos][currpos] if prevprevpos in trans3_probability.keys() and prevpos in trans3_probability[prevprevpos].keys() and currpos in trans3_probability[prevprevpos][prevpos].keys() else self.smooth_prob
                        p*=tp3
                        #pos_mc[label[i]]=p
                    #psum+=math.log(pos_mc[label[i]])
            return math.log(p)
        elif model == "HMM":
            psum=0
            p=1
            
            for i in range(len(sentence)):
                
                  
                    currpos=label[i]
                    word=sentence[i]
                    ep = emission_probability[currpos][word] if currpos in emission_probability.keys() and word in emission_probability[currpos].keys() else self.smooth_prob
                    if i==0:
                            ip = initial_probability[label[i]]
                            p = ip*ep
                            pos_vi[label[i]]=p
                    else:
                            prevpos=label[i-1]
                            #pp = prior_probability[prevpos]
                            tp = trans_probability[prevpos][currpos] if prevpos in trans_probability.keys() and currpos in trans_probability[prevpos].keys() else self.smooth_prob
                            p = ep*tp
                            pos_vi[label[i]]=p   
                    psum+=math.log(pos_vi[label[i]])
            return psum
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
        for word in sentence:
            probposgword={}
            for pos in prior_probability.keys():
                probposgword[pos]={}
                if word not in emission_probability[pos].keys():
                    emission_probability[pos][word]=self.smooth_prob
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
                p=[]
                for prevst in prior_probability.keys():
                    if (currst not in trans_probability[prevst].keys()):
                        p.append((V[t-1][prevst]["prob"]*self.smooth_prob,prevst))
                    else:
                        p.append((V[t-1][prevst]["prob"]*trans_probability[prevst][currst],prevst))
                max_prob=max(p)
                V[t][currst] = {"prob": max_prob[0] * (emission_probability[currst][sentence[t]]), "prev": max_prob[1]}
                
        # Back-tracking
        prob = max((value["prob"],value["prev"]) for value in V[len(sentence)-1].values())
        for pos in prior_probability.keys():
            if V[len(sentence)-1][pos]["prob"]==prob[0]:
                lastpos=pos
                #pos_vi[lastpos]=prob[0]
        
        # Final State Sequence
        hmmseq=[]
        hmmseq.append(lastpos)
        for t in range(len(sentence)-1,0,-1):
            hmmseq.append(V[t][lastpos]["prev"])
            lastpos=V[t][lastpos]["prev"]
            #pos_vi[lastpos]=V[t][lastpos]["prob"]
        hmmseq.reverse()
        
        return hmmseq
   
    # Generate samples for the complex (MCMC) model using gibbs sampling
    def generate_sample(self, sentence, sample):
        sentence_len = len(sentence)
        samplecopy=list(sample)
        
        for i in range(sentence_len):
            possample=list(samplecopy)
            probabilities = []
            poslist=[]
            
            for pos in prior_probability.keys():
                p=1
                possample[i]=pos
                currpos=possample[i]
                word=sentence[i]
                ep = emission_probability[currpos][word] if currpos in emission_probability.keys() and word in emission_probability[currpos].keys() else self.smooth_prob
                p*=ep
                if i==0:
                    ip = initial_probability[currpos]
                    p *= ip
                    #pos_mc[pos]=p
                elif i==1:
                    prevpos=possample[i-1]
                    #pp = prior_probability[prevpos]
                    tp = trans_probability[prevpos][currpos] if prevpos in trans_probability.keys() and currpos in trans_probability[prevpos].keys() else self.smooth_prob
                    p *= tp
                    #pos_mc[currpos]=p
                else:
                
                    prevprevpos=possample[i-2]
                    prevpos=possample[i-1]
                    #pp = prior_probability[prevprevpos]
                    tp3 = trans3_probability[prevprevpos][prevpos][currpos] if prevprevpos in trans3_probability.keys() and prevpos in trans3_probability[prevprevpos].keys() and currpos in trans3_probability[prevprevpos][prevpos].keys() else self.smooth_prob
#                    tp = trans_probability[prevprevpos][pos] if prevprevpos in trans_probability.keys() and pos in trans_probability[prevprevpos].keys() else self.smooth_prob
#                    tp1 = trans_probability[prevpos][pos] if prevpos in trans_probability.keys() and pos in trans_probability[prevpos].keys() else self.smooth_prob
#                    tp2 = trans_probability[prevprevpos][prevpos] if prevprevpos in trans_probability.keys() and prevpos in trans_probability[prevprevpos].keys() else self.smooth_prob
                    p *=tp3
                    #pos_mc[currpos]=p
                probabilities.append(p)
                poslist.append(pos)
            
            probsum=sum(probabilities)
            probabilities=[float(prob/probsum) for prob in probabilities]
            
            rand = random.random()
            p_sum = 0
            for j in range(len(probabilities)):
                #p_sum += probabilities[j]
                if rand < probabilities[j]:
                    samplecopy[i] = poslist[j]
                    break

        return samplecopy

    # COmplex model - Gibbs Sampling
    def complex_mcmc(self, sentence):
        samples=[]
        sample=['noun']*len(sentence) # Initializing a sample with all pos tags as nouns
        
        for n in range(500):  
            sample = self.generate_sample(sentence, sample)
            if(n>=100):                   # Ignore first 100 samples as burn-in iterations
                samples.append(list(sample))

 #       for n in range(100,500):
          #  sample = self.generate_sample(sentence, sample)
           # samples.append(sample)

        # Picking the pos tag that occurs the most frequent
        mcmcpos=[]
        for i in range(len(sentence)):
            mcmcposcount=dict.fromkeys(prior_probability.keys(),0)
            for each in samples:
                mcmcposcount[each[i]]+=1
            mcmcpos.append(max(mcmcposcount, key=mcmcposcount.get))
        return mcmcpos


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

