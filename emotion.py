
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

class Emotion:
    def __init__(self,text):
        self.text = text
    def EmotionExtract(self,text):

        # loading data
        df = pd.read_csv(r"F:/Project/emotiondataset.csv",encoding='latin-1')
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
        df2 = df1.sample(8610, random_state=1,replace=True).copy()

       
        pd.DataFrame(df2.intent.unique())

        # Create a new column 'category_id' with encoded categories 
        df2['category_id'] = df2['intent'].factorize()[0]
        category_id_df = df2[['intent', 'category_id']].drop_duplicates()


        # Dictionaries for future use
        category_to_id = dict(category_id_df.values)
        id_to_category = dict(category_id_df[['category_id', 'intent']].values)

        # New dataframe
        df2.head()
        fig = plt.figure(figsize=(8,6))
        colors = ['grey','grey','grey','grey','grey','grey','grey','grey','grey',
            'grey','darkblue','darkblue','darkblue']
        df2.groupby('intent').text.count().sort_values().plot.barh(
            ylim=0, color=colors, title= 'Intent classification\n')
        plt.xlabel('Number of ocurrences', fontsize = 10);

        tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5,
                                ngram_range=(1, 2), 
                                stop_words='english')

        # We transform each complaint into a vector
        features = tfidf.fit_transform(df2.text).toarray()

        labels = df2.category_id

        ##########################################################################
        import pickle
        Pkl_Filename = "Pickle_Emotion_Model.pkl"  

        with open(Pkl_Filename, 'rb') as file:  
            Pickled_LR_Model = pickle.load(file)
        #########################################################################

        #val = input("Enter your value: ") 
        texts = [text]
        text_features = tfidf.transform(texts)
        #predictions = model.predict(text_features)
        #for text, predicted in zip(texts, predictions):
          #print('"{}"'.format(text))
          #print("  - Predicted as: '{}'".format(id_to_category[predicted]))
          #print("")

        # Use the Reloaded Model to 
        # Calculate the accuracy score and predict target values
        # Predict the Labels using the reloaded Model

        Ypredict = Pickled_LR_Model.predict(text_features)  

        CatName=""
        for text, predicted in zip(texts, Ypredict):
          print('"{}"'.format(text))
          print("  - Predicted as: '{}'".format(id_to_category[predicted]))
          print("")
          CatName=format(id_to_category[predicted])
          print(CatName)
        return CatName
