# getting libraries
import pandas as pd
import matplotlib.pyplot as plt

###### The data preprocesing phase
branchs = pd.read_csv('student_master_branchs.csv')
data = pd.read_csv('student_master.csv')

data = data.drop(['age', 'repeated','id_student', 'nationality', 'bac_wilaya'] , axis = 1)
# converting to nupy array
X = data.iloc[:,:].values
y = branchs.iloc[:,1].values

# Encoding The data
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
X[:,0] = label.fit_transform(X[:,0])
#X[:,3] = label.fit_transform(X[:,3])
#X[:,-1] = label.fit_transform(X[:,-1])
y = label.fit_transform(y)

from sklearn.preprocessing import StandardScaler
std = StandardScaler()
X = std.fit_transform(X)

from sklearn.decomposition import PCA
pca = PCA(n_components = 8)
new = pca.fit_transform(X)
print(pca.explained_variance_ratio_)
#Plot in 2D
plt.scatter(X[:,1],X[:,2] ,c = y)
plt.show()
# spiting the data test train split
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(X,y , test_size = 0.3 , random_state = 40)
################################################################
from sklearn.metrics import r2_score, f1_score
from sklearn.metrics import confusion_matrix
############ KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 56, metric = 'minkowski',p =2)
knn.fit(x_train , y_train)
# Predicting the Test set results
y_pred = knn.predict(x_test)
print('The score For Knn:' , f1_score(y_test , y_pred , average = 'macro'))
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
print('The score For SVM: ' ,f1_score(y_test , y_pred , average = 'macro'))
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
print('The score For Logistic Regression: ' , f1_score(y_test , y_pred , average = 'macro'))
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
print('The score For Naive Bayes: ' , f1_score(y_test , y_pred , average = 'macro'))
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
print('The score For Decision Trees: ' , f1_score(y_test , y_pred , average = 'macro'))
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
print('The score For Random Forest: ' , f1_score(y_test , y_pred , average = 'macro'))
print('The accuracy For Random Forest is ', (y_pred == y_test).mean())
cm = confusion_matrix(y_test,y_pred)
print('Confusion Matrix is :', cm)
print('######################################')
################################################################
from sklearn.cluster import KMeans
############# K-Means Algorithm
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter = 300, n_init= 10, random_state=0)
kmeans.fit(x_train)
results = kmeans.predict(x_test)
print('The accuracy For K-means is ', (results == y_test).mean())
################################################################
from sklearn.cluster import AgglomerativeClustering
############# agglomerative hierarchical clustering
agg = AgglomerativeClustering(n_clusters = 3)
results = agg.fit_predict(x_test)
print('The accuracy For Agglomerative Clustering On Test set is ', (results == y_test).mean())
results = agg.fit_predict(x_train)
print('The accuracy For Agglomerative Clustering On Train set is ', (results == y_train).mean())