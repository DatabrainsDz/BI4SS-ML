# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

############################# Load and Preprocessing #################
# loading the data
data = pd.read_csv('f_year.csv')
# taking of the target
target = data["status"]
target_new = target != "Ajourné"
#### Adding nationality encoding
nationality = data["nationality"]
nationality = nationality == "algérienne"
data["nationality"] = nationality
##### Adding the age
birthday = data["birthay"]
import datetime
formating = '%Y-%m-%d %H:%M:%S'
scholar_year =  data["scholar_year"].values
ages = []
for i in range(len(scholar_year)):
    date = datetime.datetime.strptime(birthday[i],formating)
    ages.append(int(scholar_year[i]) - date.year)
data["age"] = ages
# droping the data we don't need
data = data.drop(["id_student","group" , "section","section" ,"repeated",  "birthay" ,"year_study","year_cycle" , "scholar_year" , "status"], axis = 1)
# data_new = pd.get_dummies(X[:,1])

# converting to numpy arrays
X = data.iloc[:,:].values
y = target_new.iloc[:].values

# Encoding The data
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
gender_encoder = LabelEncoder()
nationality_encoder = LabelEncoder()

X[:,0] = gender_encoder.fit_transform(X[:,0])
X[:,1] = nationality_encoder.fit_transform(X[:,1])
y = label.fit_transform(y)

### savingt the Label Encoder
np.save('gender_classes.npy' , gender_encoder.classes_)
np.save('nationality_classes.npy' , nationality_encoder.classes_)

# Standerdizing
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

### savingt the Scaler
import pickle
scalerfile = 'scaler_one.sav'
pickle.dump(scaler, open(scalerfile, 'wb'))


# trying PCA
from sklearn.decomposition import PCA
pca = PCA(2)
new_X = pca.fit_transform(X)
ratio = pca.explained_variance_ratio_

# Plotting The data
plt.scatter(new_X[:,0] , new_X[:,1] , c = y )
plt.legend()
plt.show()

# spiting the data test train split
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(X,y , test_size = 0.3 , random_state = 40)

############################## Machine Learning Models #################################
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix
############ KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 56, metric = 'minkowski',p =2)
knn.fit(x_train , y_train)
# Predicting the Test set results
y_pred = knn.predict(x_test)
print('The score For Knn:' , r2_score(y_test , y_pred))
print('The accuracy For Knn:', (y_pred == y_test).mean())
cm = confusion_matrix(y_test,y_pred)
print('Confusion Matrix is :', cm)
print('######################################')
from sklearn.externals import joblib
joblib.dump(knn, 'knn_one_v1.pkl')
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
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import SGD

########## Deep Neural Network with Keras and TensorFlow
classifier = Sequential()
classifier.add(Dense(output_dim=64 , kernel_initializer = 'uniform',
                     activation = 'relu' ,input_dim = 8))
classifier.add(Dropout(0.2))
classifier.add(Dense(output_dim=16, kernel_initializer = 'uniform',activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(output_dim=4, kernel_initializer = 'uniform',activation = 'sigmoid'))
classifier.add(Dropout(0.2))
classifier.add(Dense(output_dim =1 , kernel_initializer = 'uniform',activation = 'sigmoid'))
sgd = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)
classifier.compile(optimizer = sgd , loss = 'binary_crossentropy' ,metrics=['accuracy'])
classifier.fit(x_train,y_train,batch_size=20, epochs=100)
print('######################################')
############################################################
import torch.nn.functional as F
from torch import nn
import torch.tensor
from torch import optim
########## Deep Neural Network with Pytorch
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(7, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.sigmoid(x)
        return x

model = Network()


optimizer = optim.SGD(model.parameters(), lr=0.01)

model = nn.Sequential(nn.Linear(7, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 1),
                      nn.LogSigmoid())

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003)


epochs = 5
for e in range(epochs):
    running_loss = 0
    for i in range(len(x_train)):
        x = torch.tensor(list(x_train[i]))
        output = model.forward(x)
        loss = criterion(output, y_train[i])
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(x_train)}")

################################################ END ########
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
##### Grid Search For SVM
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
scores = ['precision', 'recall']
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()
    clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
    clf.fit(x_train, y_train)
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()
    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(x_test)
    print(classification_report(y_true, y_pred))
    print()
################################################ 
    
##### Grid Search For KNN
neighbors = list(range( 1, 101 , 5))
tuned_parameters = {'n_neighbors': neighbors, 'metric' : ['minkowski'], 
                    'weights': ['uniform' , 'distance'], 'p': [2]}

scores = ['accuracy']
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()
    clf = GridSearchCV(KNeighborsClassifier(), tuned_parameters, cv=5,
                       scoring='%s' % score)
    clf.fit(x_train, y_train)
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()
    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(x_test)
    print(classification_report(y_true, y_pred))
    print()
    """Best parameters set found on development set:

{'p': 2, 'n_neighbors': 56, 'weights': 'uniform', 'metric': 'minkowski'}"""