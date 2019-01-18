#!/usr/bin/env python

import re
import pandas as pd
import timeit
import operator
#import nltk
#from nltk.stem import WordNetLemmatizer
#wordnet_lemmatizer = WordNetLemmatizer()

start=timeit.default_timer()

# Cleaning the data and creating a bag of words model
def preprocess_trainfile(file):
    for line in file:
        city=line.split()[0]
        message=filter(str.strip,(re.sub(r"(\&amp)|[\W_']+",'',w.lower()) for w in line.split()[1:]))
#        msg=filter(lambda twt:len(twt)>=4 or twt in city.split(',_')[1].lower(), message)
        msg=filter(lambda twt:twt not in stopwords, message)
#        nltk_tokens = nltk.word_tokenize(message)
#        for w in nltk_tokens:
##            print "Actual: %s  Lemma: %s"  % (w,wordnet_lemmatizer.lemmatize(w))
#            wrd=w
#            msg=wrd+' '+w
        tweet.append(msg)
        location.append(city)
        bagofword(city,msg)
        
# Counting the number of words that appear for the same city and storing it in a dict called bagofwords        
def bagofword(city,msg):
        if city in bagofwords.keys():
            for wrd in msg:
                bagofwords[city][wrd]=(bagofwords[city][wrd]+1 if wrd in bagofwords[city].keys() else 1)
        else:
            bagofwords[city]={}
            for wrd in msg:
                bagofwords[city][wrd]=(bagofwords[city][wrd]+1 if wrd in bagofwords[city].keys() else 1)
                
        tweetscountsincity[city]=(1 if city not in tweetscountsincity.keys() else tweetscountsincity[city]+1)

def probofcity(city):
    return float(tweetscountsincity[city])/sum(tweetscountsincity.values())


def probofwordgivencity(gcity,gmsg):
    probwordgivencity={}
    for gcity in tweetscountsincity.keys():
        probwordgcity=1
        for gword in gmsg:
            if gword in bagofwords[gcity].keys():
                probwordgcity=probwordgcity*(float(bagofwords[gcity][gword])/sum(bagofwords[gcity].values()))
            else:
                probwordgcity=probwordgcity*float(1)/10000000
        prob=probwordgcity*probofcity(gcity)
        probwordgivencity[gcity]=prob
    predcity=max(probwordgivencity.items(),key=operator.itemgetter(1))[0]
    return predcity



def predict(testfile):
    for line in testfile:
        city=line.split()[0]
        origtweet=line.split()[1:]
        message=filter(str.strip,(re.sub(r"(\&amp)|[\W_']+",'',w.lower()) for w in origtweet))
#        msg=filter(lambda twt:len(twt)>=4 or twt in city.split(',_')[1].lower(), message)
        msg=filter(lambda twt:twt not in stopwords, message)
        testfiletweet.append(origtweet)
        testfilelocation.append(city)
        predcity.append(probofwordgivencity(city,msg))
    
        
def output(predcity,testdf,outfile):
	with open(outfile,"w") as f:
		for i in range(len(predcity)):
			f.write(predcity[i]+' '+testdf['Location'][i]+'\n')#+testdf['Tweets'][i])

def accuracy():
    match=0
    notmatch=0
    for i in range(len(testfile)):
        if predcity[i]==testdf['Location'][i]:
            match+=1
        else:
            notmatch+=1
            
        print(match,notmatch)
        accuracypercent=(match/(match+notmatch))*100
    return accuracypercent

trainingfile='tweets.train.clean.txt' #sys.argv[1]
testingfile='tweets.test1.clean.txt' #sys.argv[2]
outputfile='Output.txt' #sys.argv[3]

location=[]
tweet=[]
bagofwords={}
tweetscountsincity={}
top5words={}
probwordgivencity={}
testfilelocation=[]
testfiletweet=[]
predcity=[]
probwordgivencity={}
#testdata=[]
#prepositions=['','is','im','in','at','a','and','the','an','as','i','to','for',\
#              'of','this','my','you','our','with','so','on','that','here','from','your','are','were']

# Reference : http://xpo6.com/list-of-english-stop-words/
stopwords=["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost",\
           "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", \
           "amount", "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", \
           "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", \
           "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", \
           "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", \
           "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", \
           "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", \
           "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", \
           "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", \
           "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", \
           "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", \
           "hundred", "i","ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", \
           "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", \
           "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must",\
           "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", \
           "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", \
           "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", \
           "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", \
           "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", \
           "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", \
           "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", \
           "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", \
           "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", \
           "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", \
           "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", \
           "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", \
           "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole",\
           "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", \
           "yourself", "yourselves", "the"]

trainfile=open(trainingfile, 'r').readlines()
testfile=open(testingfile,'r').readlines()
preprocess_trainfile(trainfile)
predict(testfile)
testdf=pd.DataFrame({'Location':testfilelocation,'Tweets':testfiletweet})
output(predcity,testdf,outputfile)
percent=accuracy()
print ('The accuracy is '+str(percent)+' %')

end=timeit.default_timer()
elapsed_time=end-start
print (elapsed_time)