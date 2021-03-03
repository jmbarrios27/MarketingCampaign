#Importing libraries
import pandas as pd
import seaborn as sns
import warnings
from imblearn.over_sampling import SMOTE
from datetime import date
import numpy as np
import pickle
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from dtreeviz.trees import dtreeviz
from matplotlib.colors import ListedColormap


#Reading dataframe
data = pd.read_csv('C:\\Users\\Asus\\Desktop\\marketing\\data\\marketing_campaign.csv',sep=';')
data.head()


#Data check
def data_check(df):
    print('dataframe shape', df.shape)
    print()
    print(df.describe())
    print()
    print('Dataframe NaN Values')
    print(df.isna().sum())
    print()
    print(df.info())


# Checking Categorical variables.
def categorical_check(data):
    return data.describe(include=[np.object])


#Generation function
def generations(year):
    if year <= 1915:
        return 'Lost Generation'
    elif year >= 1916 and year <= 1924:
        return 'The Greatest Generation'
    elif year >= 1925 and year <= 1945:
        return 'The Silent Generation'
    elif year >= 1946 and year <=1963:
        return 'Baby Boomer Generation'
    elif year >= 1964 and year <=1979:
        return 'Generation X'
    elif year >= 1980 and year <=1993:
        return 'Millenials'
    else:
        return 'Generation Z'


#Predicted Label Change
def checkFunc(value):
    if value == True:
        return 'Corrrectly Predicted'
    else:
        return 'Not Correctly Predicted'


# Customer Segmentation
def clusters(Kmeans):
    if Kmeans == 0:
        return 'Loyal'
    elif Kmeans == 1:
        return '5-Star'
    elif Kmeans == 2:
        return 'Need Attention'
    else:
        return 'Highest Potential'




# Calling data check function
data_check(data)


#Replacing Empty values with mean
data.Income.fillna(data.Income.mean(),inplace=True)


#Calling the categorical data function, to observe categorical variables summay
categorical_check(data)



#Converting it into datetime format
data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'])


#Creating new columns based on the time info
data['Year'] = data.Dt_Customer.dt.year
data['Month'] = data.Dt_Customer.dt.month
data['Day'] = data.Dt_Customer.dt.day

#Creating a Column for count of values
data['trend'] = 1


#Creating a column with generations
data['Generation'] = data['Year_Birth'].apply(generations)



#reorder column
data = data[['ID', 'Year_Birth', 'Education', 'Marital_Status', 'Income', 'Kidhome',
       'Teenhome', 'Dt_Customer','Year', 'Month', 'Day', 'Recency', 'MntWines', 'MntFruits',
       'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
       'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
       'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth',
       'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1',
       'AcceptedCmp2', 'Complain', 'Z_CostContact', 'Z_Revenue','Generation',
        'trend','Response']]



#FILTERING BY YEARS
twelve = data[data['Year']==2012]
thirteen = data[data['Year']==2013]
fourteen =data[data['Year']==2014]



#Final dataset view
data.head()


plt.figure(figsize=(8,8))
sns.distplot(data['Year_Birth'],color='blue')
plt.xlabel('Birth Year')
plt.ylabel('Density')
plt.title('CUSTOMER BIRTH YEAR DISTRIBUTION')
plt.show()

#  Customer Generation.
sns.countplot(data= data, x='Generation',palette='magma')
plt.xlabel('Generations')
plt.xticks(rotation=90)
plt.title('CUSTOMER GENERATION')
plt.show()


#Marital Status.
sns.countplot(data=data, x='Education',palette='viridis')
plt.title('CUSTOMER EDUCATION')
plt.xlabel('Status')
plt.xticks(rotation=90)
plt.show()


sns.countplot(data=data, x='Marital_Status',palette='rainbow_r')
plt.title('MARITAL STATUS')
plt.xlabel('Status')
plt.xticks(rotation=90)
plt.show()


data.Income.plot(color='lightgreen')
plt.title('Customer’s yearly household income $')
plt.ylabel('Incomes in $')
plt.xlabel('Customer count')
plt.show()


fig = plt.figure(figsize=(10,4))
#  subplot #1
plt.subplot(121)
plt.title('KIDS AT TOME')
sns.countplot(data = data, x = 'Kidhome', palette='viridis')
#  subplot #2
plt.subplot(122)
plt.title('TEENS AT HOME')
sns.countplot(data = data, x = 'Teenhome',palette='rainbow')

plt.show()


fig = plt.figure(figsize=(22,10))
#  subplot #1
plt.subplot(231)
plt.title('SUSCRIPTIONS PER YEAR')
sns.countplot(data = data, x = 'Year', palette='viridis')
#  subplot #2
plt.subplot(232)
plt.title('SUSCRIPTIONS PER MONTHS OF ALL YEARS')
sns.countplot(data = data, x = 'Month',palette='rainbow')

#  subplot #3
plt.subplot(234)
plt.title('SUSCRIPTIONS PER DAYS OF ALL MONTHS IN ALL YEARS')
sns.countplot(data = data, x = 'Day',palette='magma')
plt.show()


fig = plt.figure(figsize=(22,10))
#  subplot #1
plt.subplot(231)
plt.title('SUSCRIPTIONS, YEAR 2012')
sns.countplot(data = twelve, x = 'Month', palette='viridis')
#  subplot #2
plt.subplot(232)
plt.title('SUSCRIPTIONS PER DAYS, 2012')
sns.countplot(data = twelve, x = 'Day',palette='gist_rainbow')
plt.show()


twelvetrend = twelve.groupby(by=['Month']).sum()
twelvetrend.trend.plot(color='blue',marker='o',linestyle='-',linewidth=3.0)
plt.title('CUSTOMER SUSCRIPTION TREND BY MONTH, 2012')
plt.xticks(twelvetrend.index)
plt.show()


fig = plt.figure(figsize=(22,10))
#  subplot #1
plt.subplot(231)
plt.title('SUSCRIPTIONS, YEAR 2013')
sns.countplot(data = thirteen, x = 'Month', palette='viridis')
#  subplot #2
plt.subplot(232)
plt.title('SUSCRIPTIONS PER DAYS, 2013')
sns.countplot(data = thirteen, x = 'Day',palette='Set3')
plt.show()



thirteentrend = thirteen.groupby(by=['Month']).sum()
thirteentrend.trend.plot(color='darkred',marker='o',linestyle='-',linewidth=3.0)
plt.title('CUSTOMER SUSCRIPTION TREND BY MONTH, 2013')
plt.xticks(thirteentrend.index)
plt.show()


fig = plt.figure(figsize=(22,10))
#  subplot #1
plt.subplot(231)
plt.title('SUSCRIPTIONS, YEAR 2014')
sns.countplot(data = fourteen, x = 'Month', palette='magma')
#  subplot #2
plt.subplot(232)
plt.title('SUSCRIPTIONS PER DAYS, 2014')
sns.countplot(data = fourteen, x = 'Day',palette='Set2')
plt.show()


fourteentrend = fourteen.groupby(by=['Month']).sum()
fourteentrend.trend.plot(color='navy',marker='o',linestyle='-',linewidth=3.0,)
plt.title('CUSTOMER SUSCRIPTION TREND BY MONTH, 2014')
plt.xticks(fourteentrend.index)
plt.show()


#Days since the last purchase distribution.
plt.figure(figsize=(8,8))
sns.distplot(data['Recency'],color='red')
plt.xlabel('Days')
plt.ylabel('Density')
plt.title('DISTRIBUTION OF NUMBER OF DAYS SINCE THE LAST PURCHASE')
plt.show()



fig = plt.figure(figsize=(20,12))
plt.suptitle('DISTRIBUTION OF MONEY SPENT, FOR DIFFERENT PRODUCTOS FOR THE LAST TWO YEARS')
#  subplot #1
plt.subplot(231)
sns.distplot(data['MntWines'],color='green')
plt.xlabel('Money Spent in $, for Wines')
plt.ylabel('Density')


#  subplot #2
plt.subplot(232)
sns.distplot(data['MntFruits'],color='red')
plt.xlabel('Money Spent in $, for fruits')
plt.ylabel('Density')



#  subplot #3
plt.subplot(233)
sns.distplot(data['MntMeatProducts'],color='blue')
plt.xlabel('Money Spent in $,for meats')
plt.ylabel('Density')



# subplot #4
plt.subplot(234)
sns.distplot(data['MntFishProducts'],color='orange')
plt.xlabel('Money Spent in $, for fish')
plt.ylabel('Density')


# subplot #5
plt.subplot(235)
sns.distplot(data['MntSweetProducts'],color='pink')
plt.xlabel('Money Spent in $, for sweet products')
plt.ylabel('Density')



# subplot #6
plt.subplot(236)
sns.distplot(data['MntGoldProds'],color='black')
plt.xlabel('Money Spent in $, for gold')
plt.ylabel('Density')



plt.show()


fig = plt.figure(figsize=(20,12))
plt.suptitle('NUMBER OF PURCHASES PER PRODUCT AND WEBSITE VISITS')
#  subplot #1
plt.subplot(231)
sns.countplot(data=data, x='NumDealsPurchases')
plt.xlabel('Purchases with discount')
plt.ylabel('Nº of Customers')


#  subplot #2
plt.subplot(232)
sns.countplot(data=data, x='NumWebPurchases')
plt.xlabel('Purchases from web')
plt.ylabel('Nº of Customers')


#  subplot #3
plt.subplot(233)
sns.countplot(data=data, x='NumCatalogPurchases')
plt.xlabel('Purchases from catalog')
plt.ylabel('Nº of Customers')



# subplot #4
plt.subplot(234)
sns.countplot(data=data, x='NumStorePurchases')
plt.xlabel('Purchases from stores')
plt.ylabel('Nº of Customers')


# subplot #5
plt.subplot(236)
sns.countplot(data=data, x='NumWebVisitsMonth')
plt.xlabel('Visit on website')
plt.ylabel('Nº of Customers')


plt.show()



fig = plt.figure(figsize=(20,12))
plt.suptitle('1 IF CUSTOMER ACCEPTED CAMPAIGN, 0 IF NOT',size=20)
c_color = ['darkred','lightgreen']
#  subplot #1
plt.subplot(231)
sns.countplot(data=data, x='AcceptedCmp1',palette=c_color)
plt.xlabel('Purchases')
plt.ylabel('Nº of Customers')
plt.title('FIRST CAMPAIGN')

#  subplot #2
plt.subplot(232)
sns.countplot(data=data, x='AcceptedCmp2',palette=c_color)
plt.xlabel('Purchases')
plt.ylabel('Nº of Customers')
plt.title('SECOND CAMPAIGN')


#  subplot #3
plt.subplot(233)
sns.countplot(data=data, x='AcceptedCmp3',palette=c_color)
plt.xlabel('Purchases')
plt.ylabel('Nº of Customers')
plt.title('THIRD CAMPAIGN')


# subplot #4
plt.subplot(234)
sns.countplot(data=data, x='AcceptedCmp4',palette=c_color)
plt.xlabel('Purchases')
plt.ylabel('Nº of Customers')
plt.title('FOURTH CAMPAIGN')

# subplot #5
plt.subplot(236)
sns.countplot(data=data, x='AcceptedCmp5',palette=c_color)
plt.xlabel('Purchases')
plt.ylabel('Nº of Customers')
plt.title('FIFTH CAMPAGIN')


plt.show()

#Customer Complain
complain = data['Complain'].value_counts()
complain_color = ['lightgreen','darkred']
plt.figure(figsize=(8,6))
plt.pie(complain,labels=complain.index,autopct='%1.2f%%',explode=(0.1,0.1),colors=complain_color)
plt.title('CUSTOMER COMPLAIN')
plt.axis('equal')
plt.show()



#Customer Complain
response = data['Response'].value_counts()
response_color = ['darkred','lightgreen']
plt.figure(figsize=(8,6))
plt.pie(response,labels=response.index,autopct='%1.2f%%',explode=(0.1,0.1),colors=response_color)
plt.title('CUSTOMER RESPONSE TO THE CAMPAIGN')
plt.axis('equal')
plt.show()

#Target Class
sns.countplot(data = data, x = 'Response',palette='gist_rainbow')
plt.title('TARGET CLASS')
plt.show()


# Creating an alternaative df to use.
model_df = data

#dropping columns
model_df = model_df.drop(columns=['ID','trend','Z_CostContact','Z_Revenue','Dt_Customer'])

# Creating Dummy variables.
model_df = pd.get_dummies(model_df)


#Separating Variables
X = model_df.drop(columns=['Response'])
y = model_df['Response']


from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier(random_state=0)
model.fit(X,y)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
plt.figure(figsize=(10,8))
feat_importances.nlargest(15).plot(kind='barh',color='lightgreen')
plt.title('FEATURE IMPORTANCE')
plt.show()



#Let´s oversample the target class
smote = SMOTE()
X_smote, y_smote = smote.fit_resample(X,y)

#Let´s check the data shape
print('Attibutes shape: ',X_smote.shape)
print('Target shape:    ',y_smote.shape)

# Creating variables for the final test set.
X_dev, y_dev = X_smote, y_smote


#Creating a Final test set, to test the model
X_smote = X_smote.iloc[382:]
y_smote = y_smote.iloc[382:]

#Selecting first 800 rows to final test set
X_dev = X_dev.iloc[0:381]
y_dev = y_dev.iloc[0:381]


#Let´s check the data shape
print('Attibutes shape: ',X_dev.shape)
print('Target shape:    ',y_dev.shape)



#Train and Test Split
X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.30,random_state=0)


#RANDOM FOREST MODEL
random_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
random_classifier.fit(X_train, y_train)
random_prediction = random_classifier.predict(X_test)

#Model Accuracy
print('RANDOM FOREST CLASSIFIER')
print(confusion_matrix(y_test, random_prediction))
print(classification_report(y_test, random_prediction))
print('Model Accuracy: ',accuracy_score(y_test, random_prediction))


#Plotting a single tree.
plt.figure(figsize=(20,20))
_= tree.plot_tree(random_classifier.estimators_[1], feature_names=X_train.columns, filled=True)


#KNN Classifier
knn_classifier = KNeighborsClassifier(n_neighbors=56)
knn_classifier.fit(X_train, y_train)
clf_prediction = knn_classifier.predict(X_test)

#Model Accuracy
print('KNN Model')
print(confusion_matrix(y_test, clf_prediction))
print(classification_report(y_test, clf_prediction))
print('Model Accuracy: ',accuracy_score(y_test, clf_prediction))


from sklearn.linear_model import LogisticRegressionCV
#LOGISTIC REGRESSION MODEL
lr_classifier =  LogisticRegressionCV(cv=5, random_state=0)
lr_classifier.fit(X_train, y_train)
lr_prediction = random_classifier.predict(X_test)

#Model Accuracy
print('LOGISTIC REGRESSIONN')
print(confusion_matrix(y_test, lr_prediction))
print(classification_report(y_test, lr_prediction))
print('Model Accuracy: ',accuracy_score(y_test, lr_prediction))


#Predicting Target values for final test set
predictions = random_classifier.predict(X_dev)

#Converting into pandas object
predictions = pd.DataFrame(predictions)
predictions.columns = ['Predictions']

#Creating a Datafame for the actual values
actual_values = pd.DataFrame(y_dev)
actual_values.columns = ['ActualValues']

#Join dataframes
results = actual_values.join(predictions)
results.head(15)


#Lets comparte the values of the columns and see if the predictions were true of vales
predicted = results['ActualValues'] == results['Predictions']

#Transform predicted into pandas DF
predicted = pd.DataFrame(predicted)
predicted.columns = ['True or False']

#Applying check function
predicted['True or False'] = predicted['True or False'].apply(checkFunc)

#Plot results
pred = predicted['True or False'].value_counts()
color = ['lightgreen','darkred']
plt.figure(figsize=(8,6))
plt.pie(pred,labels=pred.index,autopct='%1.2f%%',explode=(0,0.1),colors=color)
plt.title('MODEL PREDICTION ON FINAL TEST SET')
plt.axis('equal')
plt.show()

# Saving Random forest classifier model.
filename = 'marketing_response.pkl'
pickle.dump(random_classifier, open(filename, 'wb'))

# Creating a new dataframe from the original
unsupervised = data

#Creating a new column based on spendings
unsupervised['TotalSpendings'] = unsupervised['MntWines'] + unsupervised['MntFruits']+ unsupervised['MntMeatProducts'] + unsupervised['MntFishProducts'] + unsupervised['MntSweetProducts'] +unsupervised['MntGoldProds']

#Creating a new column for customer enrolled from enrolled date, to last date of data
last_date = date(2014,10, 4)
unsupervised['Enrolled']=pd.to_datetime(unsupervised['Dt_Customer'], dayfirst=True,format = '%Y-%m-%d')
unsupervised['Enrolled'] = pd.to_numeric(unsupervised['Enrolled'].dt.date.apply(lambda x: (last_date - x)).dt.days, downcast='integer')/30

#Create dataframe with this columns
marketing_data = unsupervised[['Enrolled','Income','TotalSpendings']]


#Observing distribution

#  subplot #1
plt.subplot(331)
sns.distplot(marketing_data['Enrolled'])


#  subplot #2
plt.subplot(333)
sns.distplot(marketing_data['Income'])

#  subplot #3
plt.subplot(335)
sns.distplot(marketing_data['TotalSpendings'])

plt.show()


#some information about the data
marketing_data.describe()


#Data normalize
scaler=StandardScaler()

#Normalizing values
X_std =scaler.fit_transform(marketing_data)
X = X_std


#Elbow curve values
Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(X)
    Sum_of_squared_distances.append(km.inertia_)


# Plot elbow curve.
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()


#fitting the K-means Algorithm
kmeans = KMeans(n_clusters=4,max_iter=100,random_state=0)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

#Adding Kmeans values to model_df
marketing_data['Kmeans'] = kmeans.labels_

#Getting unique labels
u_labels = np.unique(y_kmeans)
#Getting centroinds
centroids = kmeans.cluster_centers_


# Dataframe with only the clusters
kmeans_values = marketing_data['Kmeans']
kmeans_values = pd.DataFrame(kmeans_values)
kmeans_values = kmeans_values.reset_index(drop=True)

# Applying Principal component to reduce dimensionality.
pca_a = PCA(n_components=2) # 2d pplot
pca_review = pca_a.fit_transform(X)

# Dataframe with Components and K_means Clusters
pca_review_df = pd.DataFrame(data= pca_review, columns= ['Component1','Component2'])
pca_data = pd.concat([pca_review_df, kmeans_values[['Kmeans']]],axis=1)

# Scatterplot
sns.scatterplot(
    x="Component1", y="Component2",
    hue='Kmeans',
    palette=sns.color_palette('tab10', 4),
    data=pca_data,
    legend="full",
    alpha=0.3
)
plt.title('Clusters')
plt.show()

#Countplot
total = len(marketing_data['Kmeans'])*1.
ax = sns.countplot(y="Kmeans", data=marketing_data, palette='tab10')
plt.title('CLUSTERS')
plt.xlabel('Percentage')

for p in ax.patches:
        ax.annotate('{:.1f}%'.format(100*p.get_height()/total), (p.get_y()+0.1, p.get_height()+5))
_ = ax.set_xticklabels(map('{:.1f}%'.format, 100*ax.xaxis.get_majorticklocs()/total))


#Checking results from cluster
group = marketing_data.groupby('Kmeans').describe()
group = group.transpose()
group


# Renaming cluster names.
marketing_data['Kmeans'] = marketing_data['Kmeans'].apply(clusters)

#Separating cluster
cluster = marketing_data['Kmeans']
cluster = pd.DataFrame(cluster)
cluster.columns = ['CustomerSegmentation']

data = data.join(cluster)


# Creating a dataframe with the ID and type of customer
customer_segmentation = data[['ID','CustomerSegmentation']]
customer_segmentation.head(10)


#Downloading as a csv fie
customer_segmentation.to_csv(r'C:\\Users\Asus\\Desktop\\marketing\\data\\customer_segmentation.csv')
