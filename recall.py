from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from py2neo import Graph
from py2neo import Node, Relationship,NodeMatcher
from py2neo.ogm import GraphObject, Property

Predictions=["'INTP'", "'INTP'", "'INTP'", "'INTP'", "'INFP'", "'INFP'", "'INTP'", "'INTP'", "'INTJ'", "'INFP'", "'INFP'", "'INFP'", "'INTP'", "'INTP'", "'INTP'", "'INTP'", "'INTP'", "'INFP'", "'INFP'", "'INFP'", "'INTP'", "'INFP'", "'INTJ'", "'INFP'", "'INTP'", "'INFP'", "'INTJ'", "'INFP'", "'INTP'", "'INFP'", "'INFP'", "'INFP'", "'INTJ'", "'INTP'", "'INFP'", "'INTP'", "'INFP'", "'INTP'", "'INFP'", "'INTP'", "'INFP'", "'INTP'", "'INTP'", "'INFP'", "'INTP'", "'INTP'", "'INTP'", "'INFP'", "'INFP'"]


test=["'INFJ'", "'ESTJ'", "'ISFJ'", "'ISTJ'", "'ISTJ'", "'INFJ'", "'ISTJ'", "'ISTJ'", "'ENTJ'", "'INTJ'", "'INTJ'", "'ISFJ'", "'ISTJ'", "'ISFJ'", "'INTP'", "'ISTJ'", "'INTP'", "'INFJ'", "'ESTJ'", "'ISTJ'", "'ENTJ'", "'ISTJ'", "'INTJ'", "'INTP'", "'ESTJ'", "'ESTJ'", "'ISTJ'", "'INTP'", "'ESTJ'", "'ISTJ'", "'INTP'", "'ISTJ'", "'ESTJ'", "'ENTP'", "'ISTJ'", "'ESTP'", "'INTJ'", "'ESFJ'", "'INFP'", "'ESFJ'", "'ISFJ'", "'INTJ'", "'ENFJ'", "'INTJ'", "'INTJ'", "'ISTJ'", "'ENTP'", "'ISFJ'", "'INTJ'"]

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns # used for plot interactive graph.
from py2neo import Graph
from py2neo import Node, Relationship,NodeMatcher
graph = Graph("bolt://localhost:11005",user="neo4j",password="123456")
import pandas as pd

Predictions=[]
#"'INTP'", "'INTP'", "'INTP'", "'INTP'", "'INFP'", "'INFP'", "'INTP'", "'INTP'", "'INTJ'", "'INFP'", "'INFP'", "'INFP'", "'INTP'", "'INTP'", "'INTP'", "'INTP'", "'INTP'", "'INFP'", "'INFP'", "'INFP'", "'INTP'", "'INFP'", "'INTJ'", "'INFP'", "'INTP'", "'INFP'", "'INTJ'", "'INFP'", "'INTP'", "'INFP'", "'INFP'", "'INFP'", "'INTJ'", "'INTP'", "'INFP'", "'INTP'", "'INFP'", "'INTP'", "'INFP'", "'INTP'", "'INFP'", "'INTP'", "'INTP'", "'INFP'", "'INTP'", "'INTP'", "'INTP'", "'INFP'", "'INFP'"]


test=[]
#"'INFJ'", "'ESTJ'", "'ISFJ'", "'ISTJ'", "'ISTJ'", "'INFJ'", "'ISTJ'", "'ISTJ'", "'ENTJ'", "'INTJ'", "'INTJ'", "'ISFJ'", "'ISTJ'", "'ISFJ'", "'INTP'", "'ISTJ'", "'INTP'", "'INFJ'", "'ESTJ'", "'ISTJ'", "'ENTJ'", "'ISTJ'", "'INTJ'", "'INTP'", "'ESTJ'", "'ESTJ'", "'ISTJ'", "'INTP'", "'ESTJ'", "'ISTJ'", "'INTP'", "'ISTJ'", "'ESTJ'", "'ENTP'", "'ISTJ'", "'ESTP'", "'INTJ'", "'ESFJ'", "'INFP'", "'ESFJ'", "'ISFJ'", "'INTJ'", "'ENFJ'", "'INTJ'", "'INTJ'", "'ISTJ'", "'ENTP'", "'ISFJ'", "'INTJ'"]
d=1
url = 'F:/Project/Responses1_old.csv'
df = pd.read_csv(url)
for d in range(1,50):
    print("Processing")
    print(d)
    print(df['Name'][d])
    textName=df['Name'][d]
    tx=graph.begin()
    matcher = NodeMatcher(graph)
    PersonalSemanticNodes=list(matcher.match("PersonalSemantic").where("_.PersonName ='%s'"%textName))
    print(PersonalSemanticNodes)
    if not PersonalSemanticNodes:
          print("not exist user")
    else:

          ScoreSheetNode=list(matcher.match("ScoreSheet").where("_.PersonName ='%s'"%textName))
          MaxMass=PersonalSemanticNodes[0]['Mass']
          print(PersonalSemanticNodes)
          MaxNode=PersonalSemanticNodes[0]
          for node in PersonalSemanticNodes:
                  print("hani 6")
                  if(MaxMass<node['Mass']):
                    print("hani 7")
                    MaxMass=node['Mass']
                    MaxNode=node
          PSpersonality=MaxNode['Personality']
          ppersonality=ScoreSheetNode[0]['PersonalityType']
          test.extend(PSpersonality)
          Predictions.extend(ppersonality)
recall = recall_score(test,Predictions,average='macro')
print('Recall: %f' % recall)

# precision tp / (tp + fp)
precision = precision_score(test,Predictions,average='macro')
print('Precision: %f' % precision)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(test,Predictions,average='macro')
print('F1 score: %f' % f1)
accuracy = accuracy_score(test,Predictions)
print('Accuracy: %f' % accuracy)
unique_list=[]
for x in test:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
for x in Predictions:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
labels=unique_list
print(labels)
disp = confusion_matrix(test,Predictions,labels)
print(disp)
df_cm = pd.DataFrame(disp, index = [i for i in labels],
                  columns = [i for i in labels])
plt.figure(figsize = (10,7))
sns.heatmap(df_cm, annot=True)


