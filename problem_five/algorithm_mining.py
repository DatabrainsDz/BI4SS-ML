# like always -_- the libraries
import pandas as pd
import matplotlib.pyplot as plt
# loading the data
data = pd.read_csv('algo_results.csv')
# extract each subject on it's own datafram
data_algo_one = data[data["id_course"] == 'ALGO']
data_algo_two = data[data["id_course"] == 'Algo2']
d1 = data_algo_one.drop(['id_course' , 'current_year','study_level' , 'id_branch' , 'status'], axis = 1)
d2 = data_algo_two.drop(['id_course' , 'current_year','study_level' , 'id_branch' , 'status'], axis = 1)
d1.columns = ['id_student', 'algo_one', 'scholar_year', 'currentYear']
d2.columns = ['id_student', 'algo_two', 'scholar_year', 'currentYear']

data = data.drop(['id_course' , 'average' , 'scholar_year' ], axis = 1)
# mergin the two averages
result_one = pd.merge(d1,d2,on = ['id_student' , 'scholar_year' , 'currentYear'])
# merging the result averages with the rest of the data
result_two = pd.merge(result_one , data , on = ['id_student' , 'currentYear'])
result_clean = result_two.drop_duplicates()

result_clean = result_clean.drop(['scholar_year', 'currentYear', 'id_branch'],axis = 1)
result_clean.to_csv('result_clean.csv', index=False, header=True)

#################################### start the ML Models ##################
X = result_clean.iloc[: , 1:-1].values
y = result_clean.iloc[: , -1].values

### show the plot with the first two fields
plt.scatter(X[:,0] , X[:,1] , c = y)
plt.show()

### applying pca
from sklearn.decomposition import PCA
pca = PCA(2)
new_X = pca.fit_transform(X)
plt.scatter(new_X[:,0] ,new_X[:,1] , c = y)
plt.show()

from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
X[:,-1] = label.fit_transform(X[:,-1])
y = label.fit_transform(y)

from sklearn.preprocessing import StandardScaler
std = StandardScaler()
X = std.fit_transform(X)



# spiting the data test train split
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(X,y , test_size = 0.3 , random_state = 40)
################################################################
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix
############ KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 45, metric = 'minkowski',p =2)
knn.fit(x_train , y_train)
# Predicting the Test set results
y_pred = knn.predict(x_test)
print('The score For Knn:' , r2_score(y_test , y_pred))
print('The accuracy For Knn:', (y_pred == y_test).mean())
cm = confusion_matrix(y_test,y_pred)
print('Confusion Matrix is :', cm)
print('######################################')
################################################################
from sklearn.svm import SVC
############# SVM
classifier = SVC(kernel = 'rbf',random_state = 0 , gamma = 0.001 , C = 1000)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
print('The score For SVM: ' , r2_score(y_test , y_pred))
print('The accuracy For SVM is ', (y_pred == y_test).mean())
cm = confusion_matrix(y_test,y_pred)
print('Confusion Matrix is :', cm)
print('######################################')
#############################################################
from sklearn.linear_model import LogisticRegression
############# Logistic Regression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)
print('The score For Logistic Regression: ' , r2_score(y_test , y_pred))
print('The accuracy for LR is ', (y_pred == y_test).mean())
cm = confusion_matrix(y_test,y_pred)
print('Confusion Matrix is :', cm)
print('######################################')
###############################################################
from sklearn.naive_bayes import GaussianNB
############# Naive Bayes
classifier = GaussianNB()
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)
print('The score For Naive Bayes: ' , r2_score(y_test , y_pred))
print('The accuracy for NB is ', (y_pred == y_test).mean())
cm = confusion_matrix(y_test,y_pred)
print('Confusion Matrix is :', cm)
print('######################################')
############################################################
from sklearn.tree import DecisionTreeClassifier
############# Decision Trees
classifier = DecisionTreeClassifier(criterion ='entropy')
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)
print('The score For Decision Trees: ' , r2_score(y_test , y_pred))
print('The accuracy For Decision Trees is ', (y_pred == y_test).mean())
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix is :', cm)
print('######################################')
############################################################
from sklearn.ensemble import RandomForestClassifier
############# Random Forest
classifier = RandomForestClassifier( n_estimators = 50 , criterion ='entropy')
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)
print('The score For Random Forest: ' , r2_score(y_test , y_pred))
print('The accuracy For Random Forest is ', (y_pred == y_test).mean())
cm = confusion_matrix(y_test,y_pred)
print('Confusion Matrix is :', cm)
print('######################################')
############################################################
from sklearn.cluster import KMeans
############# K-Means Algorithm
kmeans = KMeans(n_clusters=2, init='k-means++', max_iter = 300, n_init= 10, random_state=0)
kmeans.fit(x_train)
results = kmeans.predict(x_test)
print('The accuracy For K-means is ', (results == y_test).mean())

