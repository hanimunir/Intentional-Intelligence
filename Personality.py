
import pandas as pd
import numpy as np
from scipy.stats import randint
import seaborn as sns # used for plot interactive graph. 
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import accuracy_score

class Personality:
    def __init__(self,text):
        self.text = text
    def PersonalityExtract(self,inputtext):
        # loading data
        df = pd.read_csv(r"F:/Project/mbti.csv",encoding='latin-1')
        df.shape
        df.head(2).T # Columns are shown in rows for easy reading
        # Create a new dataframe with two columns
        df1 = df[['text', 'intent']].copy()

        # Remove missing values (NaN)
        df1 = df1[pd.notnull(df1['intent'])]

        # Renaming second column for a simpler name
        df1.columns = ['text', 'intent'] 
        df1.shape
        # Percentage of complaints with text
        total = df1['text'].notnull().sum()
        round((total/len(df)*100),1)
        pd.DataFrame(df.intent.unique()).values

        # Because the computation is time consuming (in terms of CPU), the data was sampled
        df2 = df1.sample(8675, random_state=1,replace=True).copy()

        #df2=df1['text'].dropna(inplace=True)
        i=1;
        for t in df2['text']:
            #print(t)
            t=t.replace('INTJ','')
            t=t.replace('ISTJ','')
            t=t.replace('ISFJ','')
            t=t.replace('INFJ','')
            t=t.replace('ISTP','')
            t=t.replace('ISFP','')
            t=t.replace('INFP','')
            t=t.replace('INTP','')
            t=t.replace('ESTP','')
            t=t.replace('ESFP','')
            t=t.replace('ENFP','')
            t=t.replace('ENTP','')
            t=t.replace('ESTJ','')
            t=t.replace('ESFJ','')
            t=t.replace('ENFJ','')
            t=t.replace('ENTJ','')
             ##########################################
            t=t.replace('intj','')
            t=t.replace('istj','')
            t=t.replace('isfj','')
            t=t.replace('infj','')
            t=t.replace('istp','')
            t=t.replace('isfp','')
            t=t.replace('infp','')
            t=t.replace('intp','')
            t=t.replace('estp','')
            t=t.replace('esfp','')
            t=t.replace('enfp','')
            t=t.replace('entp','')
            t=t.replace('estj','')
            t=t.replace('esfj','')
            t=t.replace('enfj','')
            t=t.replace('entj','')
            t=t.replace('|||','')
            t=t.replace('http://www.youtube.com/watch?v=','<>')
            df2['text'][i]=t
    
            print("########First#########################")
            #print(t)
            #print(df2['text'][i])
            i=i+1
            #print("#################################")
            #print(t)
            # print(df2[t])
            #print("press any key")
            #text = input("Enter your value: ") 

        print("this is demo")
        # Renaming categories
       
        pd.DataFrame(df2.intent.unique())

        # Create a new column 'category_id' with encoded categories 
        df2['category_id'] = df2['intent'].factorize()[0]
        category_id_df = df2[['intent', 'category_id']].drop_duplicates()


        # Dictionaries for future use
        category_to_id = dict(category_id_df.values)
        id_to_category = dict(category_id_df[['category_id', 'intent']].values)

        # New dataframe
        df2.head()
        #fig = plt.figure(figsize=(8,6))
       # colors = ['grey','grey','grey','grey','grey','grey','grey','grey','grey',
            #'grey','darkblue','darkblue','darkblue']
        #df2.groupby('intent').text.count().sort_values().plot.barh(
           # ylim=0, color=colors, title= 'Intent classification\n')
        #plt.xlabel('Number of ocurrences', fontsize = 10);

        tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5,
                                ngram_range=(1, 2), 
                                stop_words='english')

        # We transform each into a vector
        features = tfidf.fit_transform(df2.text)

        labels = df2.category_id

       
        ##########################################################################
        import pickle
        Pkl_Filename = "Pickle_Personality_Model.pkl"  


        with open(Pkl_Filename, 'rb') as file:  
            Pickled_LR_Model = pickle.load(file)
        #########################################################################

        
        texts =[inputtext]
        
        text_features = tfidf.transform(texts)
        print("persoanlity")
        print(text_features)
        #predictions = model.predict(text_features)
        #for text, predicted in zip(texts, predictions):
          #print('"{}"'.format(text))
          #print("  - Predicted as: '{}'".format(id_to_category[predicted]))
          #print("")

        # Use the Reloaded Model to 
        # Calculate the accuracy score and predict target values
        # Predict the Labels using the reloaded Model

        Ypredict = Pickled_LR_Model.predict(text_features)  
        print("done")
        CatName=""
        for text, predicted in zip(texts, Ypredict):
          print('"{}"'.format(text))
          print("  - Predicted as: '{}'".format(id_to_category[predicted]))
          print("")
          CatName=format(id_to_category[predicted])
          print(CatName)
        return CatName


