# -*- coding: utf-8 -*-
import nltk
import json
from sklearn import svm
from Neo4j import Perception
from Neo4j import EpisodicMemory
from Neo4j import WorkingMemory
from Neo4j import Agent
import wx 

class Mywin(wx.Frame): 
   def __init__(self, parent, title,size): 
        super(Mywin, self).__init__(parent, title = title,size = size)
        #window = wx.Frame(None, title = "Welcome...! Says Personal Semantics Agent", size = (800,500)) 
        panel = wx.Panel(self) 
        label = wx.StaticText(panel,id=1,label="Enter your question below:",pos=(50,370),size=(400,30))
        font = label.GetFont()
        font.PointSize +=3
        font = font.Bold()
        label.SetFont(font)
        self.outputtextbox = wx.TextCtrl(panel,id=2,value="",pos=(50,50),size=(400,300),style = wx.TE_MULTILINE|wx.TE_READONLY) 
        self.inputtextbox = wx.TextCtrl(panel,id=1,value="",pos=(50,400),size=(400,30),style = wx.TE_PROCESS_ENTER)
        input=self.inputtextbox.Bind(wx.EVT_TEXT_ENTER,self.OnEnter)
        print(input)
        Statelabel = wx.StaticText(panel,id=11,label="Agent Subjective States:",pos=(465,20),size=(200,30),style = wx.TE_READONLY)
        font = Statelabel.GetFont()
        font.PointSize +=5
        font = font.Bold()
        Statelabel.SetFont(font)
        Statetextbox = wx.TextCtrl(panel,id=12,value="",pos=(465,50),size=(300,280))
        ConversationMode= wx.Button(panel,id=3,label="Conversation Mode",pos=(500,350),size=(200,50))
        ConversationMode.Bind(wx.EVT_BUTTON,self.OnConversationMode)
        QuestionMode= wx.Button(panel,id=3,label="Question Mode",pos=(500,400),size=(200,50))
        QuestionMode.Bind(wx.EVT_BUTTON,self.OnQuestionMode)

        
        self.Centre() 
        self.Show() 
        self.Fit()  
   def OnConversationMode(self, event): 
      print(event.GetString())

   def OnQuestionMode(self, event): 
      print(event.GetString())


   def OnEnter(self,event): 

       print(event.GetString())
       self.outputtextbox.Value=self.outputtextbox.Value+'\n'+"says:"+event.GetString() 
       self.inputtextbox.Value=""
       text = self.outputtextbox.Value
       PAM=Perception()
       PAM.perception(text,personae)


#app = wx.App() 
#Mywin(None,"Welcome...! Says Personal Semantics Agent",(800,500))
#app.MainLoop()