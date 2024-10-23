import nltk
import csv
from csv import reader
import pandas as pd
import random
from nltk.stem import PorterStemmer
from py2neo import Graph
from py2neo import Node, Relationship,NodeMatcher
from sklearn.metrics import precision_score,recall_score
from sklearn.metrics import  accuracy_score, f1_score
from Neo4j import Perception,SemanticMemory
from Neo4j import EpisodicMemory,WorkingMemory,PersonalSemantic

df = pd.read_csv(r"F:/Project/Responses1_old.csv",encoding='latin-1')

Questionlist={'1':['At a party do you','interact with many, including strangers','interact with a few, known to me','I am in the party '],
              '2':['Are you more:','realistic than speculative','speculative than realistic','I am '],
              '3':['Is it worse to:','have your “head in the clouds”.','be “in a rut','it is worst to '],
              '4':['Are you more impressed by:','principles.','emotions.','i am more impressed by the '],
              '5':['Are more drawn toward the:','convincing.','touching.','i am more drawn towards the '],
              '6':['Do you prefer to work:','to deadlines.','just “whenever”.','I prefer to do work '],
              '7':['Do you tend to choose:','rather carefully.','somewhat impulsively.','i tend to choose things '],
              '8':['At parties do you:','stay late, with increasing energy.','leave early with decreased energy','In the party, i am '],
              '9':['Are you more attracted to:','sensible people.','imaginative people','I am attracted to the '],
              '10':['Are you more interested in:','what is actual.','what is possible','i am interested in '],
              '11':['In judging others are you more swayed by:','laws than circumstances.','circumstances than laws','i am more swayed by '],
              '12':['In approaching others is your inclination to be somewhat:','objective','personal.','in approaching others, i am '],
              '13':['Are you more:','punctual.','leisurely.','i am '],
              '14':['Does it bother you more having things:','incomplete.','completed.','i like things '],
              '15':['In your social groups do you:','keep abreast of others happenings.','get behind on the news.','in my social group,i am '],
              '16':['In doing ordinary things are you more likely to:','do it the usual way.','do it your own way.','in doing things,i '],
              '17':['Writers should:','“say what they mean and mean what they say”.','express things more by use of analogy','i think writers should '],
              '18':['Which appeals to you more:','consistency of thought.','harmonious human relationships','i request '],
              '19':['Are you more comfortable in making:','logical judgments.','value judgments','i am comfortable in making '],
              '20':['Do you want things:','settled and decided.','unsettled and undecided.','i want things '],
              '21':['Would you say you are more:','serious and determined.','easy-going','i am '],
              '22':['In phoning do you:','rarely question that it will all be said.','rehearse what you will say','i am in phoning '],
              '23':['Facts:','speak for themselves.','illustrate principles','facts '],
              '24':['Are visionaries:','somewhat annoying.','rather fascinating.','visionaries are '],
              '25':['Are you more often:','a cool-headed person.','a warm-hearted person','i am '],
              '26':['Is it worse to be:','unjust.','merciless.','it is worse to be '],
              '27':['Should one usually let events occur:','by careful selection and choice.','randomly and by chance','events should occur '],
              '28':['Do you feel better about:','having purchased.','having the option to buy.','i feel better for '],
              '29':['In company do you:','initiate conversation.','wait to be approached','In company, i always '],
              '30':['Common sense is:','rarely questionable.','frequently questionable','common sense is '],
              '31':['Children often do not:','make themselves useful enough.','exercise their fantasy enough.','children often do not '],
              '32':['In making decisions do you feel more comfortable with:','standards.','feelings','i am more comfortable with '],
              '33':['Are you more:','firm than gentle.','gentle than firm.','i am '],
              '34':['Which is more admirable:','the ability to organize and be methodical.','his ability to adapt and make do.','it is more admirable to '],
              '35':['Do you put more value on:','infinite.','open-minded.','i put more value on '],
              '36':['Does new and non-routine interaction with others:','stimulate and energize me.','tax your reserves.','new interactions with others '],
              '37':['Are you more frequently:','a practical sort of person.','a fanciful sort of person.','i am '],
              '38':['Are you more likely to:','see how others are useful.','see how others see.','i '],
              '39':['Which is more satisfying:','to discuss an issue thoroughly.','to arrive at agreement on an issue','it is more satisfying '],
              '40':['Which rules you more:','my head.','my heart.','what rule me is '],
              '41':['Are you more comfortable with work that is:','contracted.','done on a casual basis.','i am comfortable with work that is '],
              '42':['Do you tend to look for:','the orderly.','whatever turns up.','i look for '],
              '43':['Do you prefer:','many friends with brief contact.','a few friends with more lengthy contact.','i prefer '],
              '44':['Do you go more by:','facts','principles','i go more by '],
              '45':['Are you more interested in:','production and distribution','design and research','i am more interested in '],
              '46':['Which is more of a compliment:','There is a very logical person.','There is a very sentimental person.','it is more compliment of '],
              '47':['Do you value in yourself more that you are:','unwavering.','devoted.','i am '],
              '48':['Do you more often prefer the','final and unalterable statement.','tentative and preliminary statement','i prefer '],
              '49':['Are you more comfortable:','after a decision','before a decision','i am comfortable'],
              '50':['Do you:','speak easily and at length with strangers.','find little to say to strangers','i '],
              '51':['Are you more likely to trust your:','experience.','hunch.','i trust my '],
              '52':['Do you feel:','more practical than ingenious.','more ingenious than practical.','i feel '],
              '53':['Which person is more to be complimented– one of:','clear reason.','strong feeling.','the more complimented person is with '],
              '54':['Are you inclined more to be:','fair-minded.','sympathetic.','i am '],
              '55':['Is it preferable mostly to:','make sure things are arranged.','just let things happen.','it is preferable to '],
              '56':['In relationships should most things be:','re-negotiable.','random and circumstantial.','things in a relationship should be '],
              '57':['When the phone rings do you:','hasten to get to it first.','hope someone else will answer.','when phone bell rings, i '],
              '58':['Do you prize more in yourself:','a strong sense of reality.','a vivid imagination.','i feel in myself the '],
              '59':['Are you drawn more to:','fundamentals.','overtones.','i drawn to '],
              '60':['Which seems the greater error:','to be too passionate.','to be too objective.','it is wrong '],
              '61':['Do you see yourself as basically:','hard-headed.','soft-hearted.','i am '],
              '62':['Which situation appeals to you more:','structured and scheduled.','unstructured and unscheduled.','i like the situation '],
              '63':['Are you a person that is more:','routinized than whimsical.','whimsical than routinized.','i am '],
              '64':['Are you more inclined to be:','easy to approach.','somewhat reserved.','i am '],
              '65':['In writings do you prefer:','the more literal.','the more figurative.','i prefer '],
              '66':['Is it harder for you to:','identify with others.','utilize others.','it is harder for me to '],
              '67':['Which do you wish more for yourself:','clarity of reason.','strength of compassion.','i wish for myself the  '],
              '68':['Which is the greater fault:','being indiscriminate.','being critical.','it is faulty to '],
              '69':['Do you prefer the:','planned event.','unplanned event.','i prefer '],
              '70':['Do you tend to be more:','deliberate than spontaneous','spontaneous than deliberate','i am ']}

str1=[]

text=""
column1=[]
column2=[]
column3=[]
column4=[]
column5=[]
column6=[]
column7=[]
Scorelist={ '1':['',''],
              '2':['',''],
              '3':['',''],
              '4':['',''],
              '5':['',''],
              '6':['',''],
              '7':['',''],
              '8':['',''],
              '9':['',''],
              '10':['',''],
              '11':['',''],
              '12':['',''],
              '13':['',''],
              '14':['',''],
              '15':['',''],
              '16':['',''],
              '17':['',''],
              '18':['',''],
              '19':['',''],
              '20':['',''],
              '21':['',''],
              '22':['',''],
              '23':['',''],
              '24':['',''],
              '25':['',''],
              '26':['',''],
              '27':['',''],
              '28':['',''],
              '29':['',''],
              '30':['',''],
              '31':['',''],
              '32':['',''],
              '33':['',''],
              '34':['',''],
              '35':['',''],
              '36':['',''],
              '37':['',''],
              '38':['',''],
              '39':['',''],
              '40':['',''],
              '41':['',''],
              '42':['',''],
              '43':['',''],
              '44':['',''],
              '45':['',''],
              '46':['',''],
              '47':['',''],
              '48':['',''],
              '49':['',''],
              '50':['',''],
              '51':['',''],
              '52':['',''],
              '53':['',''],
              '54':['',''],
              '55':['',''],
              '56':['',''],
              '57':['',''],
              '58':['',''],
              '59':['',''],
              '60':['',''],
              '61':['',''],
              '62':['',''],
              '63':['',''],
              '64':['',''],
              '65':['',''],
              '66':['',''],
              '67':['',''],
              '68':['',''],
              '69':['',''],
              '70':['','']}

dict=[]
graph = Graph("bolt://localhost:11004",user="neo4j",password="123456")
print("processing start")
d=1

y_pred=list()
y_test=list()
#PSpersonality=['INFJ']

for d in range(1,50):
    print("hani")
    text=""
    print(df['1'][d])
    #print(df['Name'][d])
    textName=df['Name'][d]
    textRefID=""+str(df['ID'][d])+""
    textInterest=""+str(df['interest'][d])+""
    textGender=""+str(df['Gender'][d])+""
    print(textName)
    print(textRefID)
    print(textInterest)
    CurInteraction=random.randint(1,100)
    textName=textName.lower()
    PAM=Perception()
    personae={'AgentName':[textName],'PersonReference':[textRefID],'PersonInterest':[textInterest],'Interaction':[CurInteraction]}
    print("personae is")
    print(personae)
    EpisodeNumber=1
    if(df['1'][d]=='a'):
        tex=Questionlist['1'][3]+Questionlist['1'][1]
        text=text+tex+"|||"
        Scorelist['1'][0]="1"
        Scorelist['1'][1]="0"
        textinput=tex
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['1'][3]+Questionlist['1'][2]
        text=text+tex+"|||"
        Scorelist['1'][0]="0"
        Scorelist['1'][1]="1"
        textinput=tex
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
################################################

    if(df['2'][d]=='a'):
        tex=Questionlist['2'][3]+Questionlist['2'][1]
        text=text+tex+"|||"
        Scorelist['2'][0]="1"
        Scorelist['2'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['2'][3]+Questionlist['2'][2]
        text=text+tex+"|||"
        Scorelist['2'][0]="0"
        Scorelist['2'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    if(df['3'][d]=='a'):
        tex=Questionlist['3'][3]+Questionlist['3'][1]
        text=text+tex+"|||"
        Scorelist['3'][0]="1"
        Scorelist['3'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['3'][3]+Questionlist['3'][2]
        text=text+tex+"|||"
        Scorelist['3'][0]="0"
        Scorelist['3'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    if(df['4'][d]=='a'):
        tex=Questionlist['4'][3]+Questionlist['4'][1]
        text=text+tex+"|||"
        Scorelist['4'][0]="1"
        Scorelist['4'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['4'][3]+Questionlist['4'][2]
        text=text+tex+"|||"
        Scorelist['4'][0]="0"
        Scorelist['4'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    if(df['5'][d]=='a'):
        tex=Questionlist['5'][3]+Questionlist['5'][1]
        text=text+tex+"|||"
        Scorelist['5'][0]="1"
        Scorelist['5'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['5'][3]+Questionlist['5'][2]
        text=text+tex+"|||"
        Scorelist['5'][0]="0"
        Scorelist['5'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    if(df['6'][d]=='a'):
        tex=Questionlist['6'][3]+Questionlist['6'][1]
        text=text+tex+"|||"
        Scorelist['6'][0]="1"
        Scorelist['6'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['6'][3]+Questionlist['6'][2]
        text=text+tex+"|||"
        Scorelist['6'][0]="0"
        Scorelist['6'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    if(df['7'][d]=='a'):
        tex=Questionlist['7'][3]+Questionlist['7'][1]
        text=text+tex+"|||"
        Scorelist['7'][0]="1"
        Scorelist['7'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['7'][3]+Questionlist['7'][2]
        text=text+tex+"|||"
        Scorelist['7'][0]="0"
        Scorelist['7'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    if(df['8'][d]=='a'):
        tex=Questionlist['8'][3]+Questionlist['8'][1]
        text=text+tex+"|||"
        Scorelist['8'][0]="1"
        Scorelist['8'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['8'][3]+Questionlist['8'][2]
        text=text+tex+"|||"
        Scorelist['8'][0]="0"
        Scorelist['8'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")

    if(df['9'][d]=='a'):
        tex=Questionlist['9'][3]+Questionlist['9'][1]
        text=text+tex+"|||"
        Scorelist['9'][0]="1"
        Scorelist['9'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['9'][3]+Questionlist['9'][2]
        text=text+tex+"|||"
        Scorelist['9'][0]="0"
        Scorelist['9'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    if(df['10'][d]=='a'):
        tex=Questionlist['10'][3]+Questionlist['10'][1]
        text=text+tex+"|||"
        Scorelist['10'][0]="1"
        Scorelist['10'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['10'][3]+Questionlist['10'][2]
        text=text+tex+"|||"
        Scorelist['10'][0]="0"
        Scorelist['10'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    if(df['11'][d]=='a'):
        tex=Questionlist['11'][3]+Questionlist['11'][1]
        text=text+tex+"|||"
        Scorelist['11'][0]="1"
        Scorelist['11'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['11'][3]+Questionlist['11'][2]
        text=text+tex+"|||"
        Scorelist['11'][0]="0"
        Scorelist['11'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    if(df['12'][d]=='a'):
        tex=Questionlist['12'][3]+Questionlist['12'][1]
        text=text+tex+"|||"
        Scorelist['12'][0]="1"
        Scorelist['12'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['12'][3]+Questionlist['12'][2]
        text=text+tex+"|||"
        Scorelist['12'][0]="0"
        Scorelist['12'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    if(df['13'][d]=='a'):
        tex=Questionlist['13'][3]+Questionlist['13'][1]
        text=text+tex+"|||"
        Scorelist['13'][0]="1"
        Scorelist['13'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['13'][3]+Questionlist['13'][2]
        text=text+tex+"|||"
        Scorelist['13'][0]="0"
        Scorelist['13'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    if(df['14'][d]=='a'):
        tex=Questionlist['14'][3]+Questionlist['14'][1]
        text=text+tex+"|||"
        Scorelist['14'][0]="1"
        Scorelist['14'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['14'][3]+Questionlist['14'][2]
        text=text+tex+"|||"
        Scorelist['14'][0]="0"
        Scorelist['14'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    if(df['15'][d]=='a'):
        tex=Questionlist['15'][3]+Questionlist['15'][1]
        text=text+tex+"|||"
        Scorelist['15'][0]="1"
        Scorelist['15'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['15'][3]+Questionlist['15'][2]
        text=text+tex+"|||"
        Scorelist['15'][0]="0"
        Scorelist['15'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    if(df['16'][d]=='a'):
        tex=Questionlist['16'][3]+Questionlist['16'][1]
        text=text+tex+"|||"
        Scorelist['16'][0]="1"
        Scorelist['16'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['16'][3]+Questionlist['16'][2]
        text=text+tex+"|||"
        Scorelist['16'][0]="0"
        Scorelist['16'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    if(df['17'][d]=='a'):
        tex=Questionlist['17'][3]+Questionlist['17'][1]
        text=text+tex+"|||"
        Scorelist['17'][0]="1"
        Scorelist['17'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['17'][3]+Questionlist['17'][2]
        text=text+tex+"|||"
        Scorelist['17'][0]="0"
        Scorelist['17'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    if(df['18'][d]=='a'):
        tex=Questionlist['18'][3]+Questionlist['18'][1]
        text=text+tex+"|||"
        Scorelist['18'][0]="1"
        Scorelist['18'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['18'][3]+Questionlist['18'][2]
        text=text+tex+"|||"
        Scorelist['18'][0]="0"
        Scorelist['18'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    if(df['19'][d]=='a'):
        tex=Questionlist['19'][3]+Questionlist['19'][1]
        text=text+tex+"|||"
        Scorelist['19'][0]="1"
        Scorelist['19'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['19'][3]+Questionlist['19'][2]
        text=text+tex+"|||"
        Scorelist['19'][0]="0"
        Scorelist['19'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    
    if(df['20'][d]=='a'):
        tex=Questionlist['20'][3]+Questionlist['20'][1]
        text=text+tex+"|||"
        Scorelist['20'][0]="1"
        Scorelist['20'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['20'][3]+Questionlist['20'][2]
        text=text+tex+"|||"
        Scorelist['20'][0]="0"
        Scorelist['20'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    if(df['21'][d]=='a'):
        tex=Questionlist['21'][3]+Questionlist['21'][1]
        text=text+tex+"|||"
        Scorelist['21'][0]="1"
        Scorelist['21'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['21'][3]+Questionlist['21'][2]
        text=text+tex+"|||"
        Scorelist['21'][0]="0"
        Scorelist['21'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    if(df['22'][d]=='a'):
        tex=Questionlist['22'][3]+Questionlist['22'][1]
        text=text+tex+"|||"
        Scorelist['22'][0]="1"
        Scorelist['22'][1]="0"
        textinput=tex
        PAM.perception(textinput,personae,"1","test")
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['22'][3]+Questionlist['22'][2]
        text=text+tex+"|||"
        Scorelist['22'][0]="0"
        Scorelist['22'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")

    if(df['23'][d]=='a'):
        tex=Questionlist['23'][3]+Questionlist['23'][1]
        text=text+tex+"|||"
        Scorelist['23'][0]="1"
        Scorelist['23'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['23'][3]+Questionlist['23'][2]
        text=text+tex+"|||"
        Scorelist['23'][0]="0"
        Scorelist['23'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    if(df['24'][d]=='a'):
        tex=Questionlist['24'][3]+Questionlist['24'][1]
        text=text+tex+"|||"
        Scorelist['24'][0]="1"
        Scorelist['24'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['24'][3]+Questionlist['24'][2]
        text=text+tex+"|||"
        Scorelist['24'][0]="0"
        Scorelist['24'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    if(df['25'][d]=='a'):
        tex=Questionlist['25'][3]+Questionlist['25'][1]
        text=text+tex+"|||"
        Scorelist['25'][0]="1"
        Scorelist['25'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['25'][3]+Questionlist['25'][2]
        text=text+tex+"|||"
        Scorelist['25'][0]="0"
        Scorelist['25'][1]="1"
        textinput=tex
        
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
        

    if(df['26'][d]=='a'):
        tex=Questionlist['26'][3]+Questionlist['26'][1]
        text=text+tex+"|||"
        Scorelist['26'][0]="1"
        Scorelist['26'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['26'][3]+Questionlist['26'][2]
        text=text+tex+"|||"
        Scorelist['26'][0]="0"
        Scorelist['26'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    if(df['27'][d]=='a'):
        tex=Questionlist['27'][3]+Questionlist['27'][1]
        text=text+tex+"|||"
        Scorelist['27'][0]="1"
        Scorelist['27'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['27'][3]+Questionlist['27'][2]
        text=text+tex+"|||"
        Scorelist['27'][0]="0"
        Scorelist['27'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    if(df['28'][d]=='a'):
        tex=Questionlist['28'][3]+Questionlist['28'][1]
        text=text+tex+"|||"
        Scorelist['28'][0]="1"
        Scorelist['28'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['28'][3]+Questionlist['28'][2]
        text=text+tex+"|||"
        Scorelist['28'][0]="0"
        Scorelist['28'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    if(df['29'][d]=='a'):
        tex=Questionlist['29'][3]+Questionlist['29'][1]
        text=text+tex+"|||"
        Scorelist['29'][0]="1"
        Scorelist['29'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['29'][3]+Questionlist['29'][2]
        text=text+tex+"|||"
        Scorelist['29'][0]="0"
        Scorelist['29'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    if(df['30'][d]=='a'):
        tex=Questionlist['30'][3]+Questionlist['30'][1]
        text=text+tex+"|||"
        Scorelist['30'][0]="1"
        Scorelist['30'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['30'][3]+Questionlist['30'][2]
        text=text+tex+"|||"
        Scorelist['30'][0]="0"
        Scorelist['30'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    if(df['31'][d]=='a'):
        tex=Questionlist['31'][3]+Questionlist['31'][1]
        text=text+tex+"|||"
        Scorelist['31'][0]="1"
        Scorelist['31'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['31'][3]+Questionlist['31'][2]
        text=text+tex+"|||"
        Scorelist['31'][0]="0"
        Scorelist['31'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    if(df['32'][d]=='a'):
        tex=Questionlist['32'][3]+Questionlist['32'][1]
        text=text+tex+"|||"
        Scorelist['32'][0]="1"
        Scorelist['32'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['32'][3]+Questionlist['32'][2]
        text=text+tex+"|||"
        Scorelist['32'][0]="0"
        Scorelist['32'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    if(df['33'][d]=='a'):
        tex=Questionlist['33'][3]+Questionlist['33'][1]
        text=text+tex+"|||"
        Scorelist['33'][0]="1"
        Scorelist['33'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['33'][3]+Questionlist['33'][2]
        text=text+tex+"|||"
        Scorelist['33'][0]="0"
        Scorelist['33'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    if(df['34'][d]=='a'):
        tex=Questionlist['34'][3]+Questionlist['34'][1]
        text=text+tex+"|||"
        Scorelist['34'][0]="1"
        Scorelist['34'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['34'][3]+Questionlist['34'][2]
        text=text+tex+"|||"
        Scorelist['34'][0]="0"
        Scorelist['34'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    if(df['35'][d]=='a'):
        tex=Questionlist['35'][3]+Questionlist['35'][1]
        text=text+tex+"|||"
        Scorelist['35'][0]="1"
        Scorelist['35'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['35'][3]+Questionlist['35'][2]
        text=text+tex+"|||"
        Scorelist['35'][0]="0"
        Scorelist['35'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    if(df['36'][d]=='a'):
        tex=Questionlist['36'][3]+Questionlist['36'][1]
        text=text+tex+"|||"
        Scorelist['36'][0]="1"
        Scorelist['36'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['36'][3]+Questionlist['36'][2]
        text=text+tex+"|||"
        Scorelist['36'][0]="0"
        Scorelist['36'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    
    if(df['37'][d]=='a'):
        tex=Questionlist['37'][3]+Questionlist['37'][1]
        text=text+tex+"|||"
        Scorelist['37'][0]="1"
        Scorelist['37'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['37'][3]+Questionlist['37'][2]
        text=text+tex+"|||"
        Scorelist['37'][0]="0"
        Scorelist['37'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    if(df['38'][d]=='a'):
        tex=Questionlist['38'][3]+Questionlist['38'][1]
        text=text+tex+"|||"
        Scorelist['38'][0]="1"
        Scorelist['38'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['38'][3]+Questionlist['38'][2]
        text=text+tex+"|||"
        Scorelist['38'][0]="0"
        Scorelist['38'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    if(df['39'][d]=='a'):
        tex=Questionlist['39'][3]+Questionlist['39'][1]
        text=text+tex+"|||"
        Scorelist['39'][0]="1"
        Scorelist['39'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['39'][3]+Questionlist['39'][2]
        text=text+tex+"|||"
        Scorelist['39'][0]="0"
        Scorelist['39'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    if(df['40'][d]=='a'):
        tex=Questionlist['40'][3]+Questionlist['40'][1]
        text=text+tex+"|||"
        Scorelist['40'][0]="1"
        Scorelist['40'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['40'][3]+Questionlist['40'][2]
        text=text+tex+"|||"
        Scorelist['40'][0]="0"
        Scorelist['40'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    if(df['41'][d]=='a'):
        tex=Questionlist['41'][3]+Questionlist['41'][1]
        text=text+tex+"|||"
        Scorelist['41'][0]="1"
        Scorelist['41'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['41'][3]+Questionlist['41'][2]
        text=text+tex+"|||"
        Scorelist['41'][0]="0"
        Scorelist['41'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    if(df['42'][d]=='a'):
        tex=Questionlist['42'][3]+Questionlist['42'][1]
        text=text+tex+"|||"
        Scorelist['42'][0]="1"
        Scorelist['42'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['42'][3]+Questionlist['42'][2]
        text=text+tex+"|||"
        Scorelist['42'][0]="0"
        Scorelist['42'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    if(df['43'][d]=='a'):
        tex=Questionlist['43'][3]+Questionlist['43'][1]
        text=text+tex+"|||"
        Scorelist['43'][0]="1"
        Scorelist['43'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['43'][3]+Questionlist['43'][2]
        text=text+tex+"|||"
        Scorelist['43'][0]="0"
        Scorelist['43'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    if(df['44'][d]=='a'):
        tex=Questionlist['44'][3]+Questionlist['44'][1]
        text=text+tex+"|||"
        Scorelist['44'][0]="1"
        Scorelist['44'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['44'][3]+Questionlist['44'][2]
        text=text+tex+"|||"
        Scorelist['44'][0]="0"
        Scorelist['44'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    if(df['45'][d]=='a'):
        tex=Questionlist['45'][3]+Questionlist['45'][1]
        text=text+tex+"|||"
        Scorelist['45'][0]="1"
        Scorelist['45'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['45'][3]+Questionlist['45'][2]
        text=text+tex+"|||"
        Scorelist['45'][0]="0"
        Scorelist['45'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    if(df['46'][d]=='a'):
        tex=Questionlist['46'][3]+Questionlist['46'][1]
        text=text+tex+"|||"
        Scorelist['46'][0]="1"
        Scorelist['46'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['46'][3]+Questionlist['46'][2]
        text=text+tex+"|||"
        Scorelist['46'][0]="0"
        Scorelist['46'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    if(df['47'][d]=='a'):
        tex=Questionlist['47'][3]+Questionlist['47'][1]
        text=text+tex+"|||"
        Scorelist['47'][0]="1"
        Scorelist['47'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['47'][3]+Questionlist['47'][2]
        text=text+tex+"|||"
        Scorelist['47'][0]="0"
        Scorelist['47'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    if(df['48'][d]=='a'):
        tex=Questionlist['48'][3]+Questionlist['48'][1]
        text=text+tex+"|||"
        Scorelist['48'][0]="1"
        Scorelist['48'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['48'][3]+Questionlist['48'][2]
        text=text+tex+"|||"
        Scorelist['48'][0]="0"
        Scorelist['48'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    if(df['49'][d]=='a'):
        tex=Questionlist['49'][3]+Questionlist['49'][1]
        text=text+tex+"|||"
        Scorelist['49'][0]="1"
        Scorelist['49'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['49'][3]+Questionlist['49'][2]
        text=text+tex+"|||"
        Scorelist['49'][0]="0"
        Scorelist['49'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    if(df['50'][d]=='a'):
        tex=Questionlist['50'][3]+Questionlist['50'][1]
        text=text+tex+"|||"
        Scorelist['50'][0]="1"
        Scorelist['50'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['50'][3]+Questionlist['50'][2]
        text=text+tex+"|||"
        Scorelist['50'][0]="0"
        Scorelist['50'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    if(df['51'][d]=='a'):
        tex=Questionlist['51'][3]+Questionlist['51'][1]
        text=text+tex+"|||"
        Scorelist['51'][0]="1"
        Scorelist['51'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['51'][3]+Questionlist['51'][2]
        text=text+tex+"|||"
        Scorelist['51'][0]="0"
        Scorelist['51'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    if(df['52'][d]=='a'):
        tex=Questionlist['52'][3]+Questionlist['52'][1]
        text=text+tex+"|||"
        Scorelist['52'][0]="1"
        Scorelist['52'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['52'][3]+Questionlist['52'][2]
        text=text+tex+"|||"
        Scorelist['50'][0]="0"
        Scorelist['50'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    if(df['53'][d]=='a'):
        tex=Questionlist['53'][3]+Questionlist['53'][1]
        text=text+tex+"|||"
        Scorelist['53'][0]="1"
        Scorelist['53'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['53'][3]+Questionlist['53'][2]
        text=text+tex+"|||"
        Scorelist['53'][0]="0"
        Scorelist['53'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    
    if(df['54'][d]=='a'):
        tex=Questionlist['54'][3]+Questionlist['54'][1]
        text=text+tex+"|||"
        Scorelist['54'][0]="1"
        Scorelist['54'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['54'][3]+Questionlist['54'][2]
        text=text+tex+"|||"
        Scorelist['54'][0]="0"
        Scorelist['54'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    if(df['55'][d]=='a'):
        tex=Questionlist['55'][3]+Questionlist['55'][1]
        text=text+tex+"|||"
        Scorelist['55'][0]="1"
        Scorelist['55'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['55'][3]+Questionlist['55'][2]
        text=text+tex+"|||"
        Scorelist['55'][0]="0"
        Scorelist['55'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")

    if(df['56'][d]=='a'):
        tex=Questionlist['56'][3]+Questionlist['56'][1]
        text=text+tex+"|||"
        Scorelist['56'][0]="1"
        Scorelist['56'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['56'][3]+Questionlist['56'][2]
        text=text+tex+"|||"
        Scorelist['56'][0]="0"
        Scorelist['56'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")

    if(df['57'][d]=='a'):
        tex=Questionlist['57'][3]+Questionlist['57'][1]
        text=text+tex+"|||"
        Scorelist['57'][0]="1"
        Scorelist['57'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['57'][3]+Questionlist['57'][2]
        text=text+tex+"|||"
        Scorelist['57'][0]="0"
        Scorelist['57'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    if(df['58'][d]=='a'):
        tex=Questionlist['58'][3]+Questionlist['58'][1]
        text=text+tex+"|||"
        Scorelist['58'][0]="0"
        Scorelist['58'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['58'][3]+Questionlist['58'][2]
        text=text+tex+"|||"
        Scorelist['58'][0]="0"
        Scorelist['58'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    if(df['59'][d]=='a'):
        tex=Questionlist['59'][3]+Questionlist['59'][1]
        text=text+tex+"|||"
        Scorelist['59'][0]="1"
        Scorelist['59'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['59'][3]+Questionlist['59'][2]
        text=text+tex+"|||"
        Scorelist['59'][0]="0"
        Scorelist['59'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    if(df['60'][d]=='a'):
        tex=Questionlist['60'][3]+Questionlist['60'][1]
        text=text+tex+"|||"
        Scorelist['60'][0]="1"
        Scorelist['60'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['60'][3]+Questionlist['60'][2]
        text=text+tex+"|||"
        Scorelist['60'][0]="0"
        Scorelist['60'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")

    if(df['61'][d]=='a'):
        tex=Questionlist['61'][3]+Questionlist['61'][1]
        text=text+tex+"|||"
        Scorelist['61'][0]="1"
        Scorelist['61'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['61'][3]+Questionlist['61'][2]
        text=text+tex+"|||"
        Scorelist['61'][0]="0"
        Scorelist['61'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    if(df['62'][d]=='a'):
        tex=Questionlist['62'][3]+Questionlist['62'][1]
        text=text+tex+"|||"
        Scorelist['62'][0]="1"
        Scorelist['62'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['62'][3]+Questionlist['62'][2]
        text=text+tex+"|||"
        Scorelist['62'][0]="0"
        Scorelist['62'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    if(df['63'][d]=='a'):
        tex=Questionlist['63'][3]+Questionlist['63'][1]
        text=text+tex+"|||"
        Scorelist['63'][0]="1"
        Scorelist['63'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['63'][3]+Questionlist['63'][2]
        text=text+tex+"|||"
        Scorelist['63'][0]="0"
        Scorelist['63'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    if(df['64'][d]=='a'):
        tex=Questionlist['64'][3]+Questionlist['64'][1]
        text=text+tex+"|||"
        Scorelist['64'][0]="1"
        Scorelist['64'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")

    else:
        tex=Questionlist['64'][3]+Questionlist['64'][2]
        text=text+tex+"|||"
        Scorelist['64'][0]="0"
        Scorelist['64'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    if(df['65'][d]=='a'):
        tex=Questionlist['65'][3]+Questionlist['65'][1]
        text=text+tex+"|||"
        Scorelist['65'][0]="1"
        Scorelist['65'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['65'][3]+Questionlist['65'][2]
        text=text+tex+"|||"
        Scorelist['65'][0]="0"
        Scorelist['65'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    if(df['66'][d]=='a'):
        tex=Questionlist['66'][3]+Questionlist['66'][1]
        text=text+tex+"|||"
        Scorelist['66'][0]="1"
        Scorelist['66'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['66'][3]+Questionlist['66'][2]
        text=text+tex+"|||"
        Scorelist['66'][0]="0"
        Scorelist['66'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    if(df['67'][d]=='a'):
        tex=Questionlist['67'][3]+Questionlist['67'][1]
        text=text+tex+"|||"
        Scorelist['67'][0]="1"
        Scorelist['67'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['67'][3]+Questionlist['67'][2]
        text=text+tex+"|||"
        Scorelist['67'][0]="0"
        Scorelist['67'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    if(df['68'][d]=='a'):
        tex=Questionlist['68'][3]+Questionlist['68'][1]
        text=text+tex+"|||"
        Scorelist['68'][0]="1"
        Scorelist['68'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['68'][3]+Questionlist['68'][2]
        text=text+tex+"|||"
        Scorelist['68'][0]="0"
        Scorelist['68'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    if(df['69'][d]=='a'):
        tex=Questionlist['69'][3]+Questionlist['69'][1]
        text=text+tex+"|||"
        Scorelist['69'][0]="1"
        Scorelist['69'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['69'][3]+Questionlist['69'][2]
        text=text+tex+"|||"
        Scorelist['69'][0]="0"
        Scorelist['69'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    if(df['70'][d]=='a'):
        tex=Questionlist['70'][3]+Questionlist['70'][1]
        text=text+tex+"|||"
        Scorelist['70'][0]="1"
        Scorelist['70'][1]="0"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")
    else:
        tex=Questionlist['70'][3]+Questionlist['70'][2]
        text=text+tex+"|||"
        Scorelist['70'][0]="0"
        Scorelist['70'][1]="1"
        textinput=tex
        EpisodeNumber=EpisodeNumber+1
        EpisodeNum=str(EpisodeNumber)
        #PAM.perception(textinput,personae,EpisodeNum,"test")






###########################################################
    
    column1={'E':[0],'I':[0]}

    column2={'copya':[0],'copyb':[0]}
    column3={'S':[0],'N':[0]}
    column4={'copya':[0],'copyb':[0]}
    column5={'T':[0],'F':[0]}
    column6={'copya':[0],'copyb':[0]}
    column7={'J':[0],'P':[0]}

    column1['E'][0]=Scorelist['1'][0]+Scorelist['8'][0]+Scorelist['15'][0]+Scorelist['22'][0]+Scorelist['29'][0]+Scorelist['36'][0]+Scorelist['43'][0]+Scorelist['50'][0]+Scorelist['57'][0]+Scorelist['64'][0]
    column1['I'][0]=Scorelist['1'][1]+Scorelist['8'][1]+Scorelist['15'][1]+Scorelist['22'][1]+Scorelist['29'][1]+Scorelist['36'][1]+Scorelist['43'][1]+Scorelist['50'][1]+Scorelist['57'][1]+Scorelist['64'][1]
    
    column2['copya'][0]=Scorelist['2'][0]+Scorelist['9'][0]+Scorelist['16'][0]+Scorelist['23'][0]+Scorelist['30'][0]+Scorelist['37'][0]+Scorelist['44'][0]+Scorelist['51'][0]+Scorelist['58'][0]+Scorelist['65'][0]
    column2['copyb'][0]=Scorelist['2'][1]+Scorelist['9'][1]+Scorelist['16'][1]+Scorelist['23'][1]+Scorelist['30'][1]+Scorelist['37'][1]+Scorelist['44'][1]+Scorelist['51'][1]+Scorelist['58'][1]+Scorelist['65'][1]

    column3['S'][0]=Scorelist['3'][0]+Scorelist['10'][0]+Scorelist['17'][0]+Scorelist['24'][0]+Scorelist['31'][0]+Scorelist['38'][0]+Scorelist['45'][0]+Scorelist['52'][0]+Scorelist['59'][0]+Scorelist['66'][0]
    column3['N'][0]=Scorelist['3'][1]+Scorelist['10'][1]+Scorelist['17'][1]+Scorelist['24'][1]+Scorelist['31'][1]+Scorelist['38'][1]+Scorelist['45'][1]+Scorelist['52'][1]+Scorelist['59'][1]+Scorelist['66'][1]
    column3['S'][0]=column3['S'][0]+column2['copya'][0]
    column3['N'][0]=column3['N'][0]+column2['copyb'][0]
    column4['copya'][0]=Scorelist['4'][0]+Scorelist['11'][0]+Scorelist['18'][0]+Scorelist['25'][0]+Scorelist['32'][0]+Scorelist['39'][0]+Scorelist['46'][0]+Scorelist['53'][0]+Scorelist['60'][0]+Scorelist['67'][0]
    column4['copyb'][0]=Scorelist['4'][1]+Scorelist['11'][1]+Scorelist['18'][1]+Scorelist['25'][1]+Scorelist['32'][1]+Scorelist['39'][1]+Scorelist['46'][1]+Scorelist['53'][1]+Scorelist['60'][1]+Scorelist['67'][1]
    
    column5['T'][0]=Scorelist['5'][0]+Scorelist['12'][0]+Scorelist['19'][0]+Scorelist['26'][0]+Scorelist['33'][0]+Scorelist['40'][0]+Scorelist['47'][0]+Scorelist['54'][0]+Scorelist['61'][0]+Scorelist['68'][0]
    column5['F'][0]=Scorelist['5'][1]+Scorelist['12'][1]+Scorelist['19'][1]+Scorelist['26'][1]+Scorelist['33'][1]+Scorelist['40'][1]+Scorelist['47'][1]+Scorelist['54'][1]+Scorelist['61'][1]+Scorelist['68'][1]
    
    column5['T'][0]=column5['T'][0]+column4['copya'][0]
    column5['F'][0]=column5['F'][0]+column4['copyb'][0]

    column6['copya'][0]=Scorelist['6'][0]+Scorelist['13'][0]+Scorelist['20'][0]+Scorelist['27'][0]+Scorelist['34'][0]+Scorelist['41'][0]+Scorelist['48'][0]+Scorelist['55'][0]+Scorelist['62'][0]+Scorelist['69'][0]
    column6['copyb'][0]=Scorelist['6'][1]+Scorelist['13'][1]+Scorelist['20'][1]+Scorelist['27'][1]+Scorelist['34'][1]+Scorelist['41'][1]+Scorelist['48'][1]+Scorelist['55'][1]+Scorelist['62'][1]+Scorelist['69'][1]

    column7['J'][0]=Scorelist['7'][0]+Scorelist['14'][0]+Scorelist['21'][0]+Scorelist['28'][0]+Scorelist['35'][0]+Scorelist['42'][0]+Scorelist['49'][0]+Scorelist['56'][0]+Scorelist['63'][0]+Scorelist['70'][0]
    column7['P'][0]=Scorelist['7'][1]+Scorelist['14'][1]+Scorelist['21'][1]+Scorelist['28'][1]+Scorelist['35'][1]+Scorelist['42'][1]+Scorelist['49'][1]+Scorelist['56'][1]+Scorelist['63'][1]+Scorelist['70'][1]

    column7['J'][0]=column7['J'][0]+column6['copya'][0]
    column7['P'][0]=column7['P'][0]+column6['copyb'][0]
    personality=""
    if(column1['E'][0]>=column1['I'][0]):
        personality=personality+"E"
    else:
        personality=personality+"I"
   

    if(column3['S'][0]>=column3['N'][0]):
        personality=personality+"S"
    else:
        personality=personality+"N"

    if(column5['T'][0]>=column5['F'][0]):
        personality=personality+"T"
    else:
        personality=personality+"F"
    if(column7['J'][0]>=column7['P'][0]):
        personality=personality+"J"
    else:
        personality=personality+"P"

    
    #print("person data")
    #print('Recall: %.3f' % recall_score(y_test, y_pred))
    
    tx=graph.begin()
    matcher = NodeMatcher(graph)
    PersonalSemanticNodes=list(matcher.match("PersonalSemantic").where("_.PersonName ='%s'"%textName))
           
    MaxMass=PersonalSemanticNodes[0]['Mass']
    print(PersonalSemanticNodes)
    MaxNode=PersonalSemanticNodes[0]
    for node in PersonalSemanticNodes:
            print("hani 6")
            if(MaxMass<node['Mass']):
               print("hani 7")
               MaxMass=node['Mass']
               MaxNode=node
    PSpersonality=["'"+MaxNode['Personality']+"'"]
    ppersonality=["'"+personality+"'"]
    tx.commit()
    print(PSpersonality)
    print(ppersonality)
    y_pred.extend(PSpersonality)
    y_test.extend(ppersonality)
    print("persoalities are :")
    print(y_pred)
    print(y_test)
    
    #print(text)
    #print(personality)
    interact=random.randint(1,50)
    dict.append([personality,text])
    print(d)
    #d=d+1
    #graph = Graph("bolt://localhost:11004",user="neo4j",password="123456")

    
    tx=graph.begin()
    query='''MATCH(n:Personae) WHERE n.PersonName=$uol AND n.PersonReference=$RefID RETURN n'''
    nodePersonae=tx.run(query,uol=textName,RefID=textRefID)
    print(nodePersonae)
    if not nodePersonae:
        print("user not exist")
        personae=Node("Personae",PersonName=textName,PersonReference=textRefID,Gender=textGender,Interest=textInterest,Interaction=interact)
        tx.create(personae)
        Scoresheet=Node("ScoreSheet",PersonName=textName,PersonReference=textRefID,PersonalityType=personality,Q1=Scorelist['1'],Q2=Scorelist['2'],Q3=Scorelist['3'],Q4=Scorelist['4'],Q5=Scorelist['5'],Q6=Scorelist['6'],Q7=Scorelist['7'],Q8=Scorelist['8'],Q9=Scorelist['9'],Q10=Scorelist['10'],Q11=Scorelist['11'],Q12=Scorelist['12'],Q13=Scorelist['13'],Q14=Scorelist['14'],Q15=Scorelist['15'],Q16=Scorelist['16'],Q17=Scorelist['17'],Q18=Scorelist['18'],Q19=Scorelist['19'],Q20=Scorelist['20'],Q21=Scorelist['21'],Q22=Scorelist['22'],Q23=Scorelist['23'],Q24=Scorelist['24'],Q25=Scorelist['25'],Q26=Scorelist['26'],Q27=Scorelist['27'],Q28=Scorelist['28'],Q29=Scorelist['29'],Q30=Scorelist['30'],Q31=Scorelist['31'],Q32=Scorelist['32'],Q33=Scorelist['33'],Q34=Scorelist['34'],Q35=Scorelist['35'],Q36=Scorelist['36'],Q37=Scorelist['37'],Q38=Scorelist['38'],Q39=Scorelist['39'],Q40=Scorelist['40'],Q41=Scorelist['41'],Q42=Scorelist['42'],Q43=Scorelist['43'],Q44=Scorelist['44'],Q45=Scorelist['45'],Q46=Scorelist['46'],Q47=Scorelist['47'],Q48=Scorelist['48'],Q49=Scorelist['49'],Q50=Scorelist['50'],Q51=Scorelist['51'],Q52=Scorelist['40'],Q53=Scorelist['53'],Q54=Scorelist['54'],Q55=Scorelist['55'],Q56=Scorelist['56'],Q57=Scorelist['57'],Q58=Scorelist['58'],Q59=Scorelist['59'],Q60=Scorelist['60'],Q61=Scorelist['61'],Q62=Scorelist['40'],Q63=Scorelist['63'],Q64=Scorelist['64'],Q65=Scorelist['65'],Q66=Scorelist['66'],Q67=Scorelist['67'],Q68=Scorelist['68'],Q69=Scorelist['69'],Q70=Scorelist['70'])
        tx.create(Scoresheet)
        tx.commit()
    print("Processing")

#print('Recall: %.3f' % recall_score(y_test, y_pred))
#print('precision_score: %.3f' % precision_score(y_test, y_pred))
#print('f1_score: %.3f' % f1_score(y_test, y_pred))
#print('Accuracy:', accuracy_score(y_test, y_pred))
dff = pd.DataFrame(dict, columns=["Personality", "text"])
print(dff)
dff.to_csv('F:/Project/GFG.csv')



           
#print(list_of_rows)   
#for t in list_of_rows:
    #print(t)

#for t in list_of_rows:
    #print(t)
    #t oken = nltk.word_tokenize(t[0])
    #tokens=nltk.pos_tag(token)
    #print(tokens)
    #Context=" "
    #porter = PorterStemmer()
    #for w in tokens:
        #if(w[1]=='NN' or w[1]=='NNS' or w[1]=='JJ' ):
            #Context=Context+" "+w[0]
            #print(w)
        #t[1]=Context

#fields = ['Question', 'Context'] 

#print(list_of_rows)   
  
#with open('F:/Project/QuestionContext.csv', 'w') as f:
      
    # using csv.writer method from CSV package
    #write = csv.writer(f)
      
    #write.writerow(fields)
    #write.writerows(list_of_rows)


