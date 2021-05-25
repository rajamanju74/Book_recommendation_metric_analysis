# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 21:46:26 2021

@author: rajam
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import correlation
from sklearn.metrics.pairwise import pairwise_distances
import ipywidgets as widgets
from IPython.display import display, clear_output
from contextlib import contextmanager
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import os, sys
import re
from scipy.sparse import csr_matrix




import os
os.chdir(r'C:\\datascience_py')
os.getcwd()


books = pd.read_csv('BX-Books.csv', sep=';', error_bad_lines=False, encoding="latin-1")
books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM', 'imageUrlL']


users = pd.read_csv('BX-Users.csv', sep=';', error_bad_lines=False, encoding="latin-1")
users.columns = ['userID', 'Location', 'Age']
ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
ratings.columns = ['userID', 'ISBN', 'bookRating']


pd.set_option('display.max_colwidth', -1)

books.loc[books.ISBN == '0789466953','yearOfPublication '] = 2000
books.loc[books.ISBN == '0789466953','bookAuthor '] = "James Buckley"
books.loc[books.ISBN == '0789466953','publisher '] = "DK Publishing Inc"
books.loc[books.ISBN == '0789466953','bookTitle '] = "DK Readers: Creating the X-Men, How Comic Books Come to Life (Level 4: Proficient Readers)\";James Buckley"


books.loc[books.ISBN == '078946697X','yearOfPublication '] = 2000
books.loc[books.ISBN == '078946697X','bookAuthor '] = "JMichael Teitelbaum"
books.loc[books.ISBN == '078946697X','publisher '] = "DK Publishing Inc"
books.loc[books.ISBN == '078946697X','bookTitle '] = "DK Readers: Creating the X-Men, How It All Began (Level 4: Proficient Readers)\";Michael Teitelbaum"



books.loc[books.yearOfPublication == 'Gallimard', :]



books.loc[books.ISBN == '2070426769','yearOfPublication '] = 2003
books.loc[books.ISBN == '2070426769','bookAuthor '] = "Jean-Marie Gustave Le ClÃ?Â©zio"
books.loc[books.ISBN == '2070426769','publisher '] = "Gallimard"
books.loc[books.ISBN == '2070426769','bookTitle '] = "Peuple du ciel, suivi de Les Bergers"



books.yearOfPublication = pd.to_numeric(books.yearOfPublication, errors = 'coerce')


books.loc[(books.yearOfPublication > 2006) | (books.yearOfPublication == 0), 'yearOfPublication'] = np.NAN
books.yearOfPublication.fillna(round(books.yearOfPublication.mean()), inplace = True)


books.yearOfPublication = books.yearOfPublication.astype(np.int32)


books.loc[(books.ISBN == '193169656X'), 'publisher'] = 'other'
books.loc[(books.ISBN == '1931696993'), 'publisher'] = 'other'

users.loc[(users.Age > 90) | (users.Age < 5), 'Age'] = np.nan
users.Age = users.Age.fillna(users.Age.mean())
users.Age = users.Age.astype(np.int32)


n_users = users.shape[0]
n_books = books.shape[0]
print(n_users * n_books)

new_ratings = ratings[ratings.ISBN.isin(books.ISBN)]
new_ratings = new_ratings[new_ratings.userID.isin(users.userID)]
new_ratings.size
len(new_ratings)


sparsity = 1.0 - len(new_ratings)/float(n_users*n_books)
print( 'The sparsity level of Book Crossing Dataset is' + str(sparsity*100)+'%')


#99.99863734155898%

ratings_explicit = new_ratings[new_ratings.bookRating != 0]
ratings_implicit = new_ratings[new_ratings.bookRating == 0]



users_exp_ratings = users[users.userID.isin(ratings_explicit.userID)]
users_imp_ratings = users[users.userID.isin(ratings_implicit.userID)]





counts1 = ratings_explicit['userID'].value_counts()
ratings_explicit = ratings_explicit[ratings_explicit['userID'].isin(counts1[counts1 >= 100].index)]


counts = ratings_explicit['bookRating'].value_counts()
ratings_explicit = ratings_explicit[ratings_explicit['bookRating'].isin(counts[counts >= 100].index)]
ratings_explicit.size
#df.groupby('name')['activity'].value_counts()
#count_users = ratings_explicit[ratings_explicit.groupby('userID')['ISBN'].value_counts()>10]
c1 = sum(ratings_explicit['userID'].value_counts())

#Generating ratings matrix from explicit ratings table
ratings_matrix = ratings_explicit.pivot(index='userID', columns='ISBN', values='bookRating')


userID = ratings_matrix.index
ISBN = ratings_matrix.columns
ISBN
print(ratings_matrix.shape)
ratings_matrix.size
ratings_matrix.head()
#Notice that most of the values are NaN (undefined) implying absence of ratings

# Drop books that have fewer than 10 ratings.
#ratings_matrix = ratings_matrix.dropna(axis='columns', thresh=10)
# Drop users that have given fewer than 10 ratings of these most-rated books
#ratings_matrix = ratings_matrix.dropna(thresh=10)
print(ratings_matrix.shape)
userID = ratings_matrix.index
ISBN = ratings_matrix.columns
ISBN.shape
print(n_users*n_books)


n_users = ratings_matrix.shape[0] #considering only those users who gave explicit ratings
n_books = ratings_matrix.shape[1]
print (n_users, n_books)
print(n_users*n_books)
#total
ratings_matrix.size
#non zero value count
nonzeroval = sum(ratings_matrix.count())
#for i in ratings_matrix.itertuples():
nonzeroval
s = 1.0 - len(ratings_explicit)/float(n_users*n_books)
sparsity = 1.0 - nonzeroval/float(n_users*n_books)
print( 'The sparsity level of Book Crossing Dataset is' + str(sparsity*100)+'%')
#47.88556837335109%


ratings_matrix.fillna(0, inplace = True)
ratings_matrix = ratings_matrix.astype(np.int32)





global metric,k
k=10
metric='cosine'



def findksimilarusers(user_id, ratings, metric = metric,k=k):
    similarities = []
    indicies = []
    model_knn = NearestNeighbors(metric = metric, algorithm = 'auto')
    model_knn.fit(ratings)
    
    loc = ratings.index.get_loc(user_id)
    distances, indices = model_knn.kneighbors(ratings.iloc[loc, :].values.reshape(1,-1), n_neighbors = k+1)
    similarities = 1-distances.flatten()
    return similarities, indices







def predict_userbased(user_id, item_id, ratings, metric = metric, k=k):
    prediction=0
    user_loc = ratings.index.get_loc(user_id)
    item_loc = ratings.columns.get_loc(item_id)
    similarities, indices=findksimilarusers(user_id, ratings,metric, k) 
    #similar users based on cosine similarity
    mean_rating = ratings.iloc[user_loc,:].mean() 
    #to adjust for zero based indexing
    sum_wt = np.sum(similarities)-1
    product=1
    wtd_sum = 0 
    
    for i in range(0, len(indices.flatten())):
        if indices.flatten()[i] == user_loc:
            continue;
        else: 
            ratings_diff = ratings.iloc[indices.flatten()[i],item_loc]-np.mean(ratings.iloc[indices.flatten()[i],:])
            product = ratings_diff * (similarities[i])
            wtd_sum = wtd_sum + product
    
    #in case of very sparse datasets, using correlation metric for collaborative 
    #based approach may give negative ratings
    #which are handled here as below
            
    if prediction <= 0:
        prediction = 1   
    elif prediction >10:
        prediction = 10
    
    prediction = int(round(mean_rating + (wtd_sum/sum_wt)))
    #print ('\nPredicted rating for user {0} -> item {1}: {2}'.format(user_id,item_id,prediction))

    return prediction



def findksimilaritems(item_id, ratings, metric=metric, k=k):
    similarities=[]
    indices=[]
    ratings=ratings.T
    loc = ratings.index.get_loc(item_id)
    model_knn = NearestNeighbors(metric = metric, algorithm = 'brute')
    model_knn.fit(ratings)
    
    distances, indices = model_knn.kneighbors(ratings.iloc[loc, :].values.reshape(1, -1), n_neighbors = k+1)
    similarities = 1-distances.flatten()

    return similarities,indices




def predict_itembased(user_id, item_id, ratings, metric = metric, k=k):
    prediction= wtd_sum =0
    user_loc = ratings.index.get_loc(user_id)
    item_loc = ratings.columns.get_loc(item_id)
    similarities, indices=findksimilaritems(item_id, ratings) #similar users based on correlation coefficients
    sum_wt = np.sum(similarities)-1
    product=1
    for i in range(0, len(indices.flatten())):
        if indices.flatten()[i] == item_loc:
            continue;
        else:
            product = ratings.iloc[user_loc,indices.flatten()[i]] * (similarities[i])
            wtd_sum = wtd_sum + product                              
    prediction = int(round(wtd_sum/sum_wt))
    
    #in case of very sparse datasets, using correlation metric for collaborative based approach may give negative ratings
    #which are handled here as below //code has been validated without the code snippet below, below snippet is to avoid negative
    #predictions which might arise in case of very sparse datasets when using correlation metric
    if prediction <= 0:
        prediction = 1   
    elif prediction >10:
        prediction = 10

    #print ('\nPredicted rating for user {0} -> item {1}: {2}'.format(user_id,item_id,prediction))    
    
    return prediction





def recommendItem(user_id, ratings, metric=metric):    
    if (user_id not in ratings.index.values) or type(user_id) is not int:
        print ("User id should be a valid integer from this list :\n\n {} ".format(re.sub('[\[\]]', '', np.array_str(ratings_matrix.index.values))))
    else:    
        clear_output(wait=True)
        prediction = []            
                                   
        metric = 'cosine'   
        print("cosine")
        #with suppress_stdout():
        print("Item based")
        for i in range(ratings.shape[1]):
            if (ratings[str(ratings.columns[i])][user_id] !=0): #not rated already
                prediction.append(predict_itembased(user_id, str(ratings.columns[i]) ,ratings, metric))
            else:                    
                prediction.append(-1) #for already rated items
        prediction = pd.Series(prediction)
        prediction = prediction.sort_values(ascending=False)
        recommended = prediction[:10]
        for i in range(len(recommended)):
            print ("{0}. {1}".format(i+1,books.bookTitle[recommended.index[i]].encode('utf-8')))                       
    #select.observe(on_change)       
        prediction = []          
        print("User based")
        for i in range(ratings.shape[1]):
            if (ratings[str(ratings.columns[i])][user_id] !=0): #not rated already
                prediction.append(predict_userbased(user_id, str(ratings.columns[i]) ,ratings, metric))
            else:                    
                prediction.append(-1) #for already rated items
        prediction = pd.Series(prediction)
        prediction = prediction.sort_values(ascending=False)
        recommended = prediction[:10]
        #print ("As per {0} approach....Following books are recommended..." .format(select.value))
        for i in range(len(recommended)):
            print ("{0}. {1}".format(i+1,books.bookTitle[recommended.index[i]].encode('utf-8')))                       
    #select.observe(on_change)
    #display(select)
        prediction = []            
                                   
        metric = 'euclidean' 
        print("euclidean")
        #with suppress_stdout():
        print("Item based")
        for i in range(ratings.shape[1]):
            if (ratings[str(ratings.columns[i])][user_id] !=0): #not rated already
                prediction.append(predict_itembased(user_id, str(ratings.columns[i]) ,ratings, metric))
            else:                    
                prediction.append(-1) #for already rated items
        prediction = pd.Series(prediction)
        prediction = prediction.sort_values(ascending=False)
        recommended = prediction[:10]
        for i in range(len(recommended)):
            print ("{0}. {1}".format(i+1,books.bookTitle[recommended.index[i]].encode('utf-8')))                       
    #select.observe(on_change)       
        prediction = []          
        print("User based")
        for i in range(ratings.shape[1]):
            if (ratings[str(ratings.columns[i])][user_id] !=0): #not rated already
                prediction.append(predict_userbased(user_id, str(ratings.columns[i]) ,ratings, metric))
            else:                    
                prediction.append(-1) #for already rated items
        prediction = pd.Series(prediction)
        prediction = prediction.sort_values(ascending=False)
        recommended = prediction[:10]
        #print ("As per {0} approach....Following books are recommended..." .format(select.value))
        for i in range(len(recommended)):
            print ("{0}. {1}".format(i+1,books.bookTitle[recommended.index[i]].encode('utf-8')))                       
   
            
       
            
        


recommendItem(11676,ratings_matrix)
#old
pred_user = predict_userbased(11676,'0001056107',ratings_matrix);
#2
pred_item = predict_itembased(11676,'0001056107',ratings_matrix);
#1

#old
#actual 7
pred_user = predict_userbased(2033,'0030020786',ratings_matrix)
#1
pred_item = predict_itembased(2033,'0030020786',ratings_matrix)
#9


#old
#actual 10
pred_user = predict_userbased(2033,'0060248025',ratings_matrix)
#1
pred_item = predict_itembased(2033,'0060248025',ratings_matrix)
#2

#old
#actual 1
pred_user = predict_userbased(2110,'0688047211',ratings_matrix)
#1
pred_item = predict_itembased(2110,'0688047211',ratings_matrix)
#1

#actual 6
pred_user = predict_userbased(11676,'002542730X',ratings_matrix)
#4
pred_item = predict_itembased(11676,'002542730X',ratings_matrix)
#7

#actual 9
pred_user = predict_userbased(11676,'006016848X',ratings_matrix)
#4
pred_item = predict_itembased(11676,'006016848X',ratings_matrix)
#9



#actual 3
pred_user = predict_userbased(11676,'0060958022',ratings_matrix)
#4
pred_item = predict_itembased(11676,'0060958022',ratings_matrix)
#7



#actual 5
pred_user = predict_userbased(11676,'0140293248',ratings_matrix)
#6
pred_item = predict_itembased(11676,'0140293248',ratings_matrix)
#7



#actual 10
pred_user = predict_userbased(11676,'0446604666',ratings_matrix)
#4
pred_item = predict_itembased(11676,'0446604666',ratings_matrix)
#8




#actual 0
pred_user = predict_userbased(11676,'1878424319',ratings_matrix)
#4
pred_item = predict_itembased(11676,'1878424319',ratings_matrix)
#5




#actual 9
pred_user = predict_userbased(11676,'1592400876',ratings_matrix)
#5
pred_item = predict_itembased(11676,'1592400876',ratings_matrix)
#8









ss = ratings_explicit[ratings_explicit.userID == 2110]
#ss = ratings_matrix[]
"""ss = ratings_matrix
df = ss.loc[11676]
ss = df
ss = pd.DataFrame(df)
ss = ss.T
#ss.rename(columns = {[1]:'rate'}, inplace = True)ISBN
#ss.columns = ['rate']
bookid = ss.columns
bookid = pd.DataFrame(bookid)
ss = ss.T
"""
...............................
ISBN = pd.DataFrame(ISBN)
booknum = pd.DataFrame(ISBN)
...............................
"""
chosen_user = 11676;
for data in ISBN.itertuples():
    pred_user = predict_userbased(chosen_user,data.ISBN,ratings_matrix)
    pred_item = predict_itembased(chosen_user,data.ISBN,ratings_matrix)
    df.append([ss[''],pred_user,pred_item])
    #i = i+1
    
    

test_data = ratings_explicit[ratings_explicit.userID == 11676 & ratings_explicit['ISBN'].isin(bookid['ISBN'])]

ss.columns = ['rate']
bookid['ISBN'] = bookid['ISBN'].astype(object)
chosen_user = 11676;
i = 0
bookid_list = []
for boo in bookid.itertuples():
    bookid_list.append(boo.ISBN)
ss['bookid'] = bookid_list
#bookid_list = pd.DataFrame(bookid_list)
#bookid_list['rate'] = ss['rate']
#bookid_list.dtypes

#sss =  ss
#ss = pd.DataFrame([sss,bookid_list])
#ss.dtypes
for data in ss.itertuples():
    pred_user = predict_userbased(chosen_user,data.bookid,ratings_matrix)
    pred_item = predict_itembased(chosen_user,data.bookid,ratings_matrix)
    df.append([data.rate,pred_user,pred_item])
    i = i+1
    
bookid.dtypes
df


for data in bookid.itertuples():

#0 --> original rating

#1 --> user based

#2 --> item based



#error = (actual - predicted)/actual
#error = ([i][0]-[i][1])/[i][0]

"""
metric = "cosine"


df = []
for data in ss.itertuples():
    pred_user = predict_userbased(2110,data.ISBN,ratings_matrix)
    pred_item = predict_itembased(2110,data.ISBN,ratings_matrix)
    df.append([data.bookRating,pred_user,pred_item])
    
df

#mine
df_user = []
df_item = []
len(df)
for i in range(103):
    error_user = (df[i][0]-df[i][1])/df[i][0]
    df_user.append(error_user)
    error_item = (df[i][0]-df[i][2])/df[i][0]
    df_item.append(error_item)
    


df_user_abs = [abs(ele) for ele in df_user]
df_item_abs = [abs(ele) for ele in df_item]
error_rate_user = sum(df_user_abs)/len(df_user)

error_rate_item = sum(df_item_abs)/len(df_item)

accuracy_item = 1 - error_rate_item

accuracy_user = 1 - error_rate_user    
    
accuracy_user
#14.29
accuracy_item
#61.43 --> abs

ratings_matrix
df_itempred = []
df_userpred = []
df_true = []
#for mae,rmse, r2_score
for i in range(len(df)):
    df_true.append(df[i][0])
    df_userpred.append(df[i][1])
    df_itempred.append(df[i][2]) 
    
MSE_cosine_user = np.square(np.subtract(df_true,df_userpred)).mean()
MSE_cosine_item = np.square(np.subtract(df_true,df_itempred)).mean()

from math import sqrt

RMSE_cosine_user = sqrt(MSE_cosine_user)
RMSE_cosine_item = sqrt(MSE_cosine_item)


MAE_cosine_user = abs(np.subtract(df_true,df_userpred)).mean()
MAE_cosine_item = abs(np.subtract(df_true,df_itempred)).mean()


from sklearn.metrics import r2_score
R2_cosine_user = r2_score(df_true,df_userpred)
R2_cosine_item = r2_score(df_true,df_itempred)

accuracy_cosine_item = accuracy_item
accuracy_cosine_user = accuracy_user
##print(ratings_matrix['userID'].where(ratings_matrix['B00009NDAN']>0))


#ss = ratings_matrix[]


metric = 'euclidean'

def findksimilarusers(user_id, ratings, metric = metric,k=k):
    similarities = []
    indicies = []
    model_knn = NearestNeighbors(metric = metric, algorithm = 'auto')
    model_knn.fit(ratings)
    loc = ratings.index.get_loc(user_id)
    distances, indices = model_knn.kneighbors(ratings.iloc[loc, :].values.reshape(1,-1), n_neighbors = k+1)
    similarities = 1-distances.flatten()
    return similarities, indices







def predict_userbased(user_id, item_id, ratings, metric = metric, k=k):
    prediction=0
    user_loc = ratings.index.get_loc(user_id)
    item_loc = ratings.columns.get_loc(item_id)
    similarities, indices=findksimilarusers(user_id, ratings,metric, k) #similar users based on cosine similarity
    mean_rating = ratings.iloc[user_loc,:].mean() #to adjust for zero based indexing
    sum_wt = np.sum(similarities)-1
    product=1
    wtd_sum = 0 
    
    for i in range(0, len(indices.flatten())):
        if indices.flatten()[i] == user_loc:
            continue;
        else: 
            ratings_diff = ratings.iloc[indices.flatten()[i],item_loc]-np.mean(ratings.iloc[indices.flatten()[i],:])
            product = ratings_diff * (similarities[i])
            wtd_sum = wtd_sum + product
    
    #in case of very sparse datasets, using correlation metric for collaborative 
    #based approach may give negative ratings
    #which are handled here as below
            
    prediction = int(round(mean_rating + (wtd_sum/sum_wt)))
    if prediction <= 0:
        prediction = 1   
    elif prediction >10:
        prediction = 10
    
    
    #print ('\nPredicted rating for user {0} -> item {1}: {2}'.format(user_id,item_id,prediction))

    return prediction



def findksimilaritems(item_id, ratings, metric=metric, k=k):
    similarities=[]
    indices=[]
    ratings=ratings.T
    loc = ratings.index.get_loc(item_id)
    model_knn = NearestNeighbors(metric = metric, algorithm = 'brute')
    model_knn.fit(ratings)
    
    distances, indices = model_knn.kneighbors(ratings.iloc[loc, :].values.reshape(1, -1), n_neighbors = k+1)
    similarities = 1-distances.flatten()

    return similarities,indices




def predict_itembased(user_id, item_id, ratings, metric = metric, k=k):
    prediction= wtd_sum =0
    user_loc = ratings.index.get_loc(user_id)
    item_loc = ratings.columns.get_loc(item_id)
    similarities, indices=findksimilaritems(item_id, ratings) #similar users based on correlation coefficients
    sum_wt = np.sum(similarities)-1
    product=1
    for i in range(0, len(indices.flatten())):
        if indices.flatten()[i] == item_loc:
            continue;
        else:
            product = ratings.iloc[user_loc,indices.flatten()[i]] * (similarities[i])
            wtd_sum = wtd_sum + product                              
    prediction = int(round(wtd_sum/sum_wt))
    
    #in case of very sparse datasets, using correlation metric for collaborative based approach may give negative ratings
    #which are handled here as below //code has been validated without the code snippet below, below snippet is to avoid negative
    #predictions which might arise in case of very sparse datasets when using correlation metric
    if prediction <= 0:
        prediction = 1   
    elif prediction >10:
        prediction = 10

    #print ('\nPredicted rating for user {0} -> item {1}: {2}'.format(user_id,item_id,prediction))    
    
    return prediction



predict_itembased(2110,'0688047211',ratings_matrix)

df = []
for data in ss.itertuples():
    pred_user = predict_userbased(2110,data.ISBN,ratings_matrix)
    pred_item = predict_itembased(2110,data.ISBN,ratings_matrix)
    df.append([data.bookRating,pred_user,pred_item])
    
df_userbased = []
df_itembased = []
df_true = []
"""
for i in range(103):
    df_userbased.append(df[i][1])
    df_itembased.append(df[i][2])
    df_true.append(df[i][0])

#0 --> original rating

#1 --> user based

#2 --> item based

"""

#error = (actual - predicted)/actual
#error = ([i][0]-[i][1])/[i][0]



df_user = []
df_item = []
len(df)
for i in range(len(df)):
    error_user = (df[i][0]-df[i][1])/df[i][0]
    df_user.append(error_user)
    error_item = (df[i][0]-df[i][2])/df[i][0]
    df_item.append(error_item)
    
df_item
df_user

error_rate_user = sum(df_user)/len(df_user)

error_rate_item = sum(df_item)/len(df_item)


df_user_abs = [abs(ele) for ele in df_user]
df_item_abs = [abs(ele) for ele in df_item]
error_rate_user = sum(df_user_abs)/len(df_user)

error_rate_item = sum(df_item_abs)/len(df_item)


accuracy_item = 1 - error_rate_item

accuracy_user = 1 - error_rate_user    
    
accuracy_user
#12.67
accuracy_item
#78.78
#75.32 --> abs













df = []
for data in ss.itertuples():
    pred_user = predict_userbased(2110,data.ISBN,ratings_matrix)
    pred_item = predict_itembased(2110,data.ISBN,ratings_matrix)
    df.append([data.bookRating,pred_user,pred_item])
    
df

#mine
df_user = []
df_item = []
len(df)
for i in range(len(df)):
    error_user = (df[i][0]-df[i][1])/df[i][0]
    df_user.append(error_user)
    error_item = (df[i][0]-df[i][2])/df[i][0]
    df_item.append(error_item)
    


df_user_abs = [abs(ele) for ele in df_user]
df_item_abs = [abs(ele) for ele in df_item]
error_rate_user = sum(df_user_abs)/len(df_user)

error_rate_item = sum(df_item_abs)/len(df_item)

accuracy_item = 1 - error_rate_item

accuracy_user = 1 - error_rate_user    
    
accuracy_user
#14.29
accuracy_item
#61.43 --> abs

ratings_matrix
df_itempred = []
df_userpred = []
df_true = []
#for mae,rmse, r2_score
for i in range(len(df)):
    df_true.append(df[i][0])
    df_userpred.append(df[i][1])
    df_itempred.append(df[i][2]) 
    
MSE_euclid_user = np.square(np.subtract(df_true,df_userpred)).mean()
MSE_euclid_item = np.square(np.subtract(df_true,df_itempred)).mean()

from math import sqrt

RMSE_euclid_user = sqrt(MSE_euclid_user)
RMSE_euclid_item = sqrt(MSE_euclid_item)

MAE_euclid_user = abs(np.subtract(df_true,df_userpred)).mean()
MAE_euclid_item = abs(np.subtract(df_true,df_itempred)).mean()


from sklearn.metrics import r2_score
R2_euclid_user = r2_score(df_true,df_userpred)
R2_euclid_item = r2_score(df_true,df_itempred)



accuracy_euclid_item = accuracy_item
accuracy_euclid_user = accuracy_user

from sklearn.cluster import KMeans

clusterer_KMeans = KMeans(n_clusters=7).fit(ratings_matrix)
preds_KMeans = clusterer_KMeans.predict(ratings_matrix)
#item
from sklearn.metrics import silhouette_score
for i in range(2,15):
    clusterer_KMeans = KMeans(n_clusters=i).fit(ratings_matrix)
    preds_KMeans = clusterer_KMeans.predict(ratings_matrix)

    kmeans_score = silhouette_score(ratings_matrix, preds_KMeans)
    print(i)
    print(kmeans_score)




#observation --> 7 good

#user
    i = 7
for i in range(2,15):
    clusterer_KMeans = KMeans(n_clusters=i).fit(ratings_explicit)
    preds_KMeans = clusterer_KMeans.predict(ratings_explicit)

    kmeans_score = silhouette_score(ratings_explicit, preds_KMeans)
    print(i)
    print(kmeans_score)


"""
Mean squared error,
Mean absolute error,
Variance score
"""