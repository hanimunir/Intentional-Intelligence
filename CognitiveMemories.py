from py2neo import Graph
from py2neo import Node, Relationship,NodeMatcher
from py2neo.ogm import GraphObject, Property
from nltk.corpus import wordnet
import pandas as pd
import numpy as np
import nltk
from nltk.stem import PorterStemmer
import spacy
import json
from datetime import datetime
from datetime import datetime
from nlppipeline import nlppipeline
from Personality import Personality
from sklearn.metrics import jaccard_score
from csv import reader
import random
import threading
import queue
import time
person=None
inputarray=None
global Mode
Mode=0
global race
race=0
global context
context="Introduction"
global turn 
turn=1
global host
host="bolt://localhost:7687"
global passw
passw="123456"
global Q 
Q=""
class ConnectionGraph:
    def connectiongraph(self):
            
            graph = Graph(host,user="neo4j",password=passw)
            return graph
            



class SemanticMemory:

        def Connection(self):
            
            graph = Graph(host,user="neo4j",password=passw)
            return graph
          
                
                
        def FetchSemanticMemory(self,text):
            graph=self.Connection()
            tx=graph.begin()
            #print("1")
            nlp = spacy.load("en_core_web_sm")
            v = text
            doc = nlp(v)
            #print("2")
            Emotion=None
            REmotion=None
            SentimentX=None
            SentimentY=None
             # Token and Tag '
            #senttoken = nltk.sent_tokenize(v,language='english')
            #print(senttoken)
            #print("sentence tokenize")
            token = nltk.word_tokenize(v)
            tokens=nltk.pos_tag(token)
            #print("tokens inside SM")
            #print(tokens)
            porter = PorterStemmer()
            for w in tokens:
                    
                     if(w[1]=='VB' or w[1]=='VBZ' or w[1]=='VBP' or w[1]=='VBG' or w[1]=='VBN' or w[1]=='VBD'):
                        verbword=porter.stem(w[0])
                        #print(verbword)
                        
                        #print("Check condition")
                                ## Query for retrieve emotion accordingly given content
                        nodes = graph.run('Match(n{XAction:"%s"})-[r]->(b) return b,ID(b)'%verbword)

                        MaxSimilarity=0   
                        MaxNodeId=None
                        if(nodes!=None):
                            i=0
                            for node in nodes:
                               
                                   #print(node[0]['Content'])
                                   str1 = text
                                   str2 = node[0]['Content']
                                   a = set(str1.split())
                                   b = set(str2.split())
                                   c = a.intersection(b)
                                   Similarity=float((len(c)) / (len(a) + len(b) - len(c)))*100
                                   if(MaxSimilarity<Similarity):
                                       MaxSimilarity=Similarity
                                       MaxNodeId=node[0]['Content']
                                   i=i+1
                                   
            
                        else:
                            print("No Data Found")
                            pass
                        
                        #print(MaxSimilarity)
                        #print(MaxNodeId)
                        agentEmotionnode= graph.run('Match(n{Content:"%s"})-[r:AgentEmotionLink]->(b) return b' %MaxNodeId)
                        for n in agentEmotionnode:
                            
                            Emotion=n[0]['XEmotion']
                            

                        RecepientEmotionnode= graph.run('Match(n{Content:"%s"})-[r:RecepientEmotionLink]->(b) return b' %MaxNodeId)
                        for n in RecepientEmotionnode:
                            REmotion=n[0]['Yemotion']

                        agentsentnode= graph.run('Match(n{Content:"%s"})-[r:AgentSentimentLink]->(b) return b' %MaxNodeId)
                        for n in agentsentnode:
                            SentimentX=n[0]['XSentiment']

                        Recepientsentnode= graph.run('Match(n{Content:"%s"})-[r:RecepientSentimentLink]->(b) return b' %MaxNodeId)
                        for n in Recepientsentnode:
                            SentimentY=n[0]['YSentiment']


                        #print("Emotion")
                        #print(Emotion)
                        return Emotion,REmotion,SentimentX,SentimentY

        

#####################################################################################
class EpisodicMemory:

        def Connection(self):   
            
            graph = Graph(host,user="neo4j",password=passw)
            return graph
          
        def CreateEpisodicMemory(self,EventArray,personae):
            
            graph=self.Connection()
            tx=graph.begin()
            
            print("episode for")
            print(EventArray['Agent'])
            print(personae['AgentName'])

            
            if(EventArray['Agent']==personae['AgentName']):
                Ref=personae['PersonReference']
                print("same person")
            else:
                Ref="known by "+personae['AgentName']
            
            N=EventArray['Agent']
            nodes=self.FetchEpisodicMemory(N,Ref)
            #print(nodes)
            now = datetime.now()
            #print("now =", now)
            # dd/mm/YY H:M:S
            date_string = now.strftime("%d/%m/%Y")
            time_string = now.strftime("%H:%M:%S")
            
            if not nodes:
                #print("empty")
                Episode = Node("Episode",EventNumber=personae['Interaction'],Context=EventArray['Context'],EpisodeNumber=EventArray['Number'],AgentName=EventArray['Agent'],AgentRefID=Ref,AgentEmotion=EventArray['AgentEmotion'][0],SentiAgent=EventArray['SentiAgent'][0],Recepient=EventArray['Recepient'][0],RecepientEmotion=EventArray['RecepientEmotion'][0],SentiOther=EventArray['SentiOther'][0],PersonalityValue=EventArray['Personality'][0],AgentActions=EventArray['AgentActions'][0],Date=date_string,Time=time_string)
                #Episode = Node("Episode",AgentName=EventArray['Agent'][0],AgentRefID=Ref,AgentEmotion=EventArray['AgentEmotion'][0],SentiAgent=EventArray['SentiAgent'][0],Recepient=EventArray['Recepient'][0],RecepientEmotion=EventArray['RecepientEmotion'][0],SentiOther=EventArray['SentiOther'][0],PersonalityValue=EventArray['Personality'][0],AgentActions=EventArray['AgentActions'][0])
                print("Episodice Memory:")
                print(Episode)
                tx.create(Episode)
                tx.commit()
                
                return 0
              
            else:
               agent=EventArray['Agent']
               EpisodesAll=tx.evaluate("Match(n:Episode) WHERE n.AgentName='%s' RETURN n"%(agent))
               #print("Episodes all")
               #print(EpisodesAll)
               Episode = Node("Episode",EventNumber=personae['Interaction'],Context=EventArray['Context'],EpisodeNumber=EventArray['Number'],AgentName=EventArray['Agent'],AgentRefID=Ref,AgentEmotion=EventArray['AgentEmotion'][0],SentiAgent=EventArray['SentiAgent'][0],Recepient=EventArray['Recepient'][0],RecepientEmotion=EventArray['RecepientEmotion'][0],SentiOther=EventArray['SentiOther'][0],PersonalityValue=EventArray['Personality'][0],AgentActions=EventArray['AgentActions'][0],Date=date_string,Time=time_string)
                
               #Episode = Node("Episode",AgentName=EventArray['Agent'][0],AgentRefID=Ref,AgentEmotion=EventArray['AgentEmotion'][0],SentiAgent=EventArray['SentiAgent'][0],Recepient=EventArray['Recepient'][0],RecepientEmotion=EventArray['RecepientEmotion'][0],SentiOther=EventArray['SentiOther'][0],PersonalityValue=EventArray['Personality'][0],AgentActions=EventArray['AgentActions'][0])
               print("Episodice Memory:")
               print(Episode)
               tx.create(Episode)
               tx.create(Relationship(EpisodesAll, "EpisodeLink", Episode))
               #print("created more episode")
               tx.commit()
               return 1
            ##################################################################################################)
        def FetchEpisodicMemory(self,AgentName,Ref):
            
            graph=self.Connection()
            #tx=graph.begin()
            N=AgentName
            
            matcher = NodeMatcher(graph)
            mm=list(matcher.match("Episode").where("_.AgentName='%s'"%N,"_.AgentRefID='%s'"%Ref))
            
            return mm

        def FetchSpecificEpisodicMemory(self,n,p,Ref):

            graph=self.Connection()
       
            matcher = NodeMatcher(graph)
            m=list(matcher.match("Episode").where("_.AgentName ='%s'"%n,"_.AgentRefID='%s'"%Ref,"_.PersonalityValue='%s'"%p))
 
            return m
        def FetchSpecificPersonalitySentimentEpisodicMemory(self,n,p,Ref,sent):

            graph=self.Connection()

            if(sent=="Nothing"):
              return "None"
            
            if(sent=="good"):
                 matcher = NodeMatcher(graph)
                 st="4.0"
                 #print("specific node inside >3")
                 m=list(matcher.match("Episode").where("_.AgentName ='%s'"%n,"_.AgentRefID='%s'"%Ref,"_.SentiOther='%s'"%st,"_.PersonalityValue='%s'"%p))
                 st="5.0"
                 m1=list(matcher.match("Episode").where("_.AgentName ='%s'"%n,"_.AgentRefID='%s'"%Ref,"_.SentiOther='%s'"%st,"_.PersonalityValue='%s'"%p))
                 
                 m.extend(m1)
                 return m
            if(sent=="bad"):
                matcher = NodeMatcher(graph)
                st="1.0"
                #print("specific node inside < 3")
                m=list(matcher.match("Episode").where("_.AgentName ='%s'"%n,"_.AgentRefID='%s'"%Ref,"_.SentiOther='%s'"%st,"_.PersonalityValue='%s'"%p))
                st="2.0"
                m1=list(matcher.match("Episode").where("_.AgentName ='%s'"%n,"_.AgentRefID='%s'"%Ref,"_.SentiOther='%s'"%st,"_.PersonalityValue='%s'"%p))
                
                m.extend(m1)
                return m
            if(sent=="netural"):
                matcher = NodeMatcher(graph)
                st="3.0"
                m=list(matcher.match("Episode").where("_.AgentName='%s'"%n,"_.AgentRefID='%s'"%Ref,"_.SentiOther='%s'"%st,"_.PersonalityValue='%s'"%p))
                
                return m
            if(sent=="nothing"):
                matcher = NodeMatcher(graph)

                st="Nan"
                m=list(matcher.match("Episode").where("_.AgentName='%s'"%n,"_.AgentRefID='%s'"%Ref,"_.SentiOther='%s'"%st,"_.PersonalityValue='%s'"%p))            
                return m
                        

class Perception:

        def Connection(self):   
            
            graph = Graph(host,user="neo4j",password=passw)
            return graph
          
        def perception(self,text,personae,EpisodeNumber,context):

            graph=self.Connection()
            #E=Emotion(text)
            #emotion=E.EmotionExtract(text)
            P=Personality(text)
            #personality="INFJ" 
            personality=P.PersonalityExtract(text)
            #P.PersonalityExtract(text)
            #I=Intent(text)
            #intent=I.IntentExtract(text)
            #############################################################################
            token = nltk.word_tokenize(text)
            tokens=nltk.pos_tag(token)
            #print(tokens)
            porter = PorterStemmer()
            verbword=[]
            i=0
            #print(tokens)
            for w in tokens:
                    
                     if(w[1]=='VB' or w[1]=='VBZ' or w[1]=='VBP' or w[1]=='VBG' or w[1]=='VBN' or w[1]=='VBD'):
                        verbword=porter.stem(w[0])
                        
                      
            #####################################NLP####################################
            nlp=nlppipeline(text)
            doc=nlp.DependencyTagger(text)
            agent=nlp.AgentExtract(doc,personae)
            
            if not agent:
                agent=personae['AgentName']
                print(agent)
            #agent=agent.lower()
            recepient=nlp.RecepientExtract(doc)
            if not recepient:
                #print("Recpeient none")
                recepient='none'

            
            ###################################Related semantic knowledge#########################################
            SM=SemanticMemory()
            SemanticKnowledge=SM.FetchSemanticMemory(text)
            if(SemanticKnowledge==None):
                agentemotion="Unknown"
                Recepientemotion="Unknown"
                sentA="3"
                sentO="3"
            else:
            #print("hani sm")
                agentemotion=SemanticKnowledge[0]
                if(agentemotion==None):
                      agentemotion="Nan"
                Recepientemotion=SemanticKnowledge[1]
                if(Recepientemotion==None):
                      Recepientemotion="Nan"
                sentA=SemanticKnowledge[2]
                sentO=SemanticKnowledge[3]
                if(sentA==None):
                      sentA="Nan"
                if(sentO==None):
                      sentO="Nan"
       
            #print("Semantic Knowledge:")
            #print(SemanticKnowledge)
            ####################################################################################
            percept = {'Agent':agent,'AgentEmotion':[agentemotion],'SentiAgent':[sentA],'Recepient':[recepient],'RecepientEmotion':[Recepientemotion],'SentiOther':[sentO],'Personality':[personality],'AgentActions':[verbword],'Number':[EpisodeNumber],'Context':[context]}
            print("Perception:")
            print(percept)
            WM=WorkingMemory()
            WM.WorkingMemoryContent(percept,personae)

class PersonalSemantic:
     def Connection(self):   
            
            graph = Graph(host,user="neo4j",password=passw)
            return graph
     def PersonalSemanticLearning(self):
         print("xyz")
     def PersonalSemanticGeneration(self,percept,personae):
        #print("personal")
        graph=self.Connection()
        tx=graph.begin()
        PI=personae['PersonInterest']
        View=self.CalculateSentiment(percept)
        sentimentvalue=percept['SentiOther'][0]
        #print(percept['Personality'][0])
        P=percept['Personality'][0]
        print("personality percept")
        print(P)
        N=percept['Agent']
        print("Agent")
        print(N)
        print(personae['AgentName'])
        Ref=None
        if(percept['Agent']==personae['AgentName']):
                Ref=personae['PersonReference']
                print("agent and personae same")
        else:
                Ref="known by "+personae['AgentName']
                print("agent and personae not same")
        EM=EpisodicMemory()
        EpisodeNodes=EM.FetchEpisodicMemory(N,Ref)
        PersonalityNodes=EM.FetchSpecificEpisodicMemory(N,P,Ref)
        SpecificNodes=EM.FetchSpecificPersonalitySentimentEpisodicMemory(N,P,Ref,View)
        print("SpecificNodes")
        print(SpecificNodes)
        print("PersonalityNodes")
        print(PersonalityNodes)
        ########################################################Attention Subjective value call ###############################################
        attentionsubvalue=self.AttentionSubjective(PI)
        
        nodes=self.FetchPersonalSemantic(percept['Personality'][0],percept['Agent'],Ref,View)
        print("existing PS")
        print(nodes)
        if not nodes:
            #print("empty")
            graph=self.Connection()
            tx=graph.begin()
            personalsemantics=Node("PersonalSemantic",PersonName=percept['Agent'],PersonReference=Ref,Personality=P,Sentiment=View,AttentionObjective=0,AttentionSubjective=attentionsubvalue,Mass=1,ConfidenceValue=0,SupportValue=0)
            
            PSall=self.FetchallPersonalSemantic(percept)
            #check for first isnstance
            if not PSall:
                print("Personal Semantic Memory:")
                print(personalsemantics)
                tx.create(personalsemantics)
                tx.commit()
                #print("created First")
            else:
                print("Personal Semantic Memory:")
                print(personalsemantics)
                tx.create(personalsemantics)
                tx.create(Relationship(PSall, "AmendmentLink", personalsemantics))
                #print("created second")
                tx.commit()
        else:
           
            internalassociation=0
            if(percept['Agent']==personae['AgentName']):
                internalassociation=1
                #print("internal association")
            print("update mass")
            massnodes=nodes[0]['Mass']
            mass=self.Mass(massnodes,internalassociation)
            #confidencevalue=0
            #supportvalue=0
            confidencevalue=self.Confidence(SpecificNodes,PersonalityNodes)
            supportvalue=self.Support(EpisodeNodes,PersonalityNodes)
            attentionobjective=self.AttentionObjective(nodes)
           # m=list(matcher.match("PersonalSemantic").where("_.PersonName ='%s'"%N,"_.Sentiment='%s'"%Sent,"_.Personality='%s'"%P))
            #print("this is updation")
            n=int(nodes[0].identity)
            #print(nodes[0])
            
            #NewNode = Node("PersonalSemantic",id=n)
            #print("this is updation before merge")
            #print(NewNode)

            tx=graph.begin()
            nodes[0].__primarylabel__ = "PersonalSemantic"
            #nodes[0].__primarykey__ = "id"
            #tx.merge(nodes[0])
            #print("this is updation after merge")

            
           
            query='''MATCH (n:PersonalSemantic) WHERE ID(n)=$uol SET n.Mass=$Mass,n.ConfidenceValue=$ConfidenceValue,n.SupportValue=$SupportValue,n.AttentionObjective=$AttentionObjective,n.AttentionSubjective=$AttentionSubjective'''

            
            tx.run(query,uol=n,Mass=mass,ConfidenceValue=confidencevalue,SupportValue=supportvalue,AttentionObjective=attentionobjective,AttentionSubjective=attentionsubvalue)
            tx.commit()
            #print("new done")
            #i want to pay attention towards my studies

            
           
     def AttentionSubjective(self,Interest):
         
         PersonalInterest=['Boating','cooking','playing','talking','Travelling','Reading','Gamming','Shopping']
         AgentInterest=Interest
         if not AgentInterest:
           AgentInterest=['Travelling','cooking']
         intersection = len(list(set(PersonalInterest).intersection(AgentInterest)))
         union = (len(PersonalInterest) + len(AgentInterest)) - intersection
         Similarity=float(intersection) / union
         AttentionSubValue=(Similarity*10)
 
         return AttentionSubValue
     def AttentionObjective(self,nodes):
         AttentionObjValue=float(nodes[0]['AttentionObjective'])
         AttentionObjValue=(AttentionObjValue+1)
         return AttentionObjValue
     def Mass(self,SpecificNodes,internalassociation):
         SpecificNodes=int(SpecificNodes)+1
         MassValue=SpecificNodes+internalassociation

         return MassValue
     def CalculateSentiment(self,percept):

         if(percept['SentiOther'][0]==None):
             view="Nothing"
             return view
         if(float(percept['SentiOther'][0])>3):
             view="good"
             return view
         if(float(percept['SentiOther'][0])<3):
             view="bad"
             return view
         if(float(percept['SentiOther'][0])==3 or percept['SentiOther']=="Nan"):
             view="netural"
             return view
         if(percept['SentiOther'][0]=="Nan"):
             view="nothing"
             return view
        
         
     def Confidence(self,SpecificNodes,PersonalityNodes):

         if (SpecificNodes=="None"):
            ConfidenceValue=0
         else:
            PersonalityNodesLength=len(PersonalityNodes)
            SameNodes=len(SpecificNodes)
            ConfidenceValue=SameNodes/PersonalityNodesLength
            
            ConfidenceValue=round(ConfidenceValue,3)
         return ConfidenceValue
     def Support(self,EpisodeNodes,SpecificNodes):
         count=0
         SameNodes=0

         if (SpecificNodes=="None"):
            SupportValue=0
         else:
            count=len(EpisodeNodes)
            SameNodes=len(SpecificNodes)

            SupportValue=float(SameNodes/count)
            print("Support is")
            print(SupportValue)
            SupportValue=round(SupportValue,3)
 
         return SupportValue

     def FetchPersonalSemantic(self,P,N,Ref,Sent):
         graph=self.Connection()
 
         matcher = NodeMatcher(graph)
         m=list(matcher.match("PersonalSemantic").where("_.PersonName ='%s'"%N,"_.Sentiment='%s'"%Sent,"_.Personality='%s'"%P))
         #print(m)
         #print("print matcher")
         #query_string = "MATCH (n:PersonalSemantic) WHERE n.PersonName:{N} AND n.Personality='%s' RETURN n " % (P)
         #Enode=graph.run(query_string, {"N":N}).to_table()
         #match for current personal semantics.......personality and sentiments......
         #Enode= graph.run("Match(n:PersonalSemantic) where n.PersonName={N:'%s'} AND n.Personality={P:'%s'} return n"%(N)%(P))
         #'Match(n{PersonName:"%s"})-[r]->(b) return b' %N)
         #print("personal semantic of person")
         #print(Enode)
         #print("Exist")
         return m
     def FetchallPersonalSemantic(self,Percept):
         graph=self.Connection()
         tx=graph.begin()

         N=Percept['Agent']

         Enode= tx.evaluate("match(n:PersonalSemantic) where n.PersonName='%s' RETURN n order by ID(n) DESC"%(N))
     
         #print("all personal semantic of person")
        
         return Enode
        

class WorkingMemory:

        def Connection(self):   
            
            graph = Graph(host,user="neo4j",password=passw)
            return graph
          
        def WorkingMemoryContent(self,percept,personae):

            
            #print(percept)
            EMN=EpisodicMemory()
            EMN.CreateEpisodicMemory(percept,personae)
            SM=SemanticMemory()
            SemanticKnowledge=SM.FetchSemanticMemory(Cue)
            EM=EpisodicMemory()
            EpisodicKnowledge=EM.FetchEpisodicMemory(Cue)
            P=PersonalSemantic()
            P.PersonalSemanticGeneration(percept,personae)
            P=PersonalSemantic()
            PM=P.FetchPersonalSemanticMemory(Cue)
            print(" Personal Semantics done")
            Assessment=ResponseElicitation(SM,PM,EM,percept)
            SelectedResponse=ResponseModulation(Assessment)
            R=ResponseGeneration(SelectedResponse)

         def ResponseElicitation(SemanticMemory C,PersonalSemantic p,EpisodicMemory EM,perception p)
                nlppipeline nlp
                tokens=nlp.SentenceToken(p.content)
                NegativeWord=SentiwordNet(tokens)
		if NegativeWord==true and p.BondValue>0:
                      flag="Negative"
		      return flag
                else 
                      flag="Positive"
                      return flag
                
            def ResponseModulation(Assessment,PersonalSemantic p):
                     nlppipeline nlp
                     tokens=nlp.SentenceToken(p.content)
                     i=0
                     for t in tokens:
                   
			if TokenFoundNegative(t)==true:
                              
                             Tokens[i]=antonym_extractor(phrase=t)
                          i=i+1
                     return tokens
                            
            

def antonym_extractor(phrase):
     
     synonyms = []
     antonyms = []

     for syn in wordnet.synsets(phrase):
          
               if l.antonyms():
                    antonyms.append(l.antonyms()[0].name())


     return set(antonyms)

def ResponseGeneration(SelectedResponse)
     
      print(SelectedResponse.GetString())
       self.outputtextbox.Value=self.outputtextbox.Value+'\n'+"says:"+ SelectedResponse 
    
    
            