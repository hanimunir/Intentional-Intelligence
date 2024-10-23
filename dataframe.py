
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
import pickle

# loading data
df = pd.read_csv(r"F:/Project/datafile.csv",encoding='latin-1')
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
df2 = df1.sample(22500, random_state=1,replace=True).copy()

#df2=df1['text'].dropna(inplace=True)
print(df)
# Renaming categories
df2.replace({'intent': 
             {'Credit reporting, credit repair services, or other personal consumer reports': 
              'Credit reporting, repair, or other', 
              'Credit reporting': 'Credit reporting, repair, or other',
             'Credit card': 'Credit card or prepaid card',
             'Prepaid card': 'Credit card or prepaid card',
             'Payday loan': 'Payday loan, title loan, or personal loan',
             'Money transfer': 'Money transfer, virtual currency, or money service',
             'Virtual currency': 'Money transfer, virtual currency, or money service'}}, 
            inplace= True)
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
# Finding the three most correlated terms with each of the product categories
N = 3
for intent, category_id in sorted(category_to_id.items()):
  features_chi2 = chi2(features, labels == category_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  print("\n==> %s:" %(intent))
  print("  * Most Correlated Unigrams are: %s" %(', '.join(unigrams[-N:])))
  print("  * Most Correlated Bigrams are: %s" %(', '.join(bigrams[-N:])))

  X = df2['text'] # Collection of documents
y = df2['intent'] # Target or the labels we want to predict (i.e., the 13 different complaints of products)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.25,
                                                    random_state = 0)

models = [
    RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]
print('hani1')
# 5 Cross-validation
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
    print('in processing')
    
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

mean_accuracy = cv_df.groupby('model_name').accuracy.mean()
std_accuracy = cv_df.groupby('model_name').accuracy.std()

acc = pd.concat([mean_accuracy, std_accuracy], axis= 1, 
          ignore_index=True)
acc.columns = ['Mean Accuracy', 'Standard deviation']
print('accuracy')
print(acc['Mean Accuracy'])
sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
              size=8, jitter=True, edgecolor="gray", linewidth=2)
#plt.show()

cv_df.groupby('model_name').accuracy.mean()


model = LinearSVC()

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.33, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df.intent.values, yticklabels=category_id_df.intent.values)
#plt.ylabel('Actual')
#plt.xlabel('Predicted')
#plt.show()
##########################################################################
Pkl_Filename = "Pickle_RL_Model.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(model, file)

with open(Pkl_Filename, 'rb') as file:  
    Pickled_LR_Model = pickle.load(file)
#########################################################################
print("SVC Accuracy Score -> ",accuracy_score(y_pred,y_test)*100)
val = input("Enter your value: ") 
texts = [val]
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

for text, predicted in zip(texts, Ypredict):
  print('"{}"'.format(text))
  print("  - Predicted as: '{}'".format(id_to_category[predicted]))
  print("")
