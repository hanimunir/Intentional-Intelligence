# -*- coding: utf-8 -*-
import nltk
import json
from sklearn import svm

import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from nltk.corpus import wordnet as wn
from nltk.corpus import conll2000
from nltk.chunk import conlltags2tree, tree2conlltags
from nltk.chunk import ne_chunk
from nltk import pos_tag
import spacy
#nlp=spacy.load('en_core_web_sm')
np.random.seed(500)

class nlppipeline:

 def __init__(self,text):
    self.text = text
 def DependencyTagger(self,text):
      
      nlp = spacy.load('en_core_web_sm')
      doc = nlp(text)
      print(doc)
      for token in doc:
          print(token.text)
          print(token.dep_)
      return doc
          #if token.dep_=='nsubj':
              #print (token.text,token.dep_)
 def AgentExtract(self,doc,personae):
      
      for token in doc:
          print(token)
          if token.dep_=='nsubj':
              print (token.text,token.dep_)
              if token.text=="i" or token.text=="I":
                    return personae['AgentName']
              elif token.text=="it" or token.text=="It":
                    return personae['AgentName']
              elif token.text=="They" or token.text=="they":
                    return personae['AgentName']
              else:
                    return token.text

 def RecepientExtract(self,doc):
      
      for token in doc:
          print(token.text,token.dep_)
          if token.dep_=='dobj':
              print (token.text,token.dep_)
              return token.text

 def PartOfSpeech(self,text):
      tokens=SentenceToken(text)
      tagged_tokens = nltk.pos_tag(tokens)
      print (" ")  #  for line spacing only
      print (tagged_tokens)
      for w in tagged_tokens:
          print (w[0], w[1])        #{w0] will return word and w[1] will return the tag

 def SentenceToken(self,text):
      sentences = nltk.sent_tokenize(text)
      for s in sentences:
            print (s)        
            tokens = nltk.wordpunct_tokenize(s)
            return tokens
 def NameEntityTagger(self,text):
        
        ne_tree = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(text)))
        print(ne_tree)




     