# getting the libraries
import numpy as np
import pandas as pd
from apyori import apriori

# get the data
data = pd.read_csv('students_course_gotten.csv')

### separate the data for the levels MIAS and MGI.
mias_data = data[data['branch_level'] == 'MIAS']
mgi_data = data[data['branch_level'] == 'MGI']
mias_data = mias_data.drop(['branch_level'],axis  = 1)
mgi_data = mgi_data.drop(['branch_level'],axis  = 1)
#### separate the data based on the semester for MIAS
mias_data_s1 = mias_data[mias_data["semester"] == 1].drop(['semester'],axis  = 1)
mias_data_s2 = mias_data[mias_data["semester"] == 2].drop(['semester'],axis  = 1)
mias_data_s3 = mias_data[mias_data["semester"] == 3].drop(['semester'],axis  = 1)
mias_data_s4 = mias_data[mias_data["semester"] == 4].drop(['semester'],axis  = 1)
mias_data_s5 = mias_data[mias_data["semester"] == 5].drop(['semester'],axis  = 1)
mias_data_s6 = mias_data[mias_data["semester"] == 6].drop(['semester'],axis  = 1)
##### separate the data based on the semester for MGI
mgi_data_s1 = mgi_data[mgi_data["semester"] == 1].drop(['semester'],axis  = 1)
mgi_data_s2 = mgi_data[mgi_data["semester"] == 2].drop(['semester'],axis  = 1)
mgi_data_s3 = mgi_data[mgi_data["semester"] == 3].drop(['semester'],axis  = 1)


def write_result(dataframe, name):
        dataframe.to_csv(name, index=False, header=False)

def run_apriori(dataset , support , confidance):
    #converting data into nupy array
    X = dataset.iloc[:,:].values

    # constructign the list
    subjects = []
    current_subject = []
    current_student = X[0,0]
    for i in range(len(X)):
        if current_student == X[i,0]:
            current_subject.append(X[i,1])
        else:
            if len(current_subject) > 1:
                subjects.append(current_subject)
            current_subject = []        
            current_student= X[i,0]
            
    rules = apriori(subjects,min_support = support, min_confidence = confidance , min_lift = 1 , min_length = 2)
    
    results = list(rules)
    found = []
    for i in range(len(results)):
        found.append(list(results[i][0]))
    
    return pd.DataFrame(found)


dataFrame = run_apriori(mias_data_s6 , 0.01,0.09)
write_result(dataFrame , 'mias_data_s6.csv')

# try the best support and condidance
interval = np.linspace(0.0001, 0.01,100)
length = 0

for i in interval:
    for j in interval:
        dataFrame = run_apriori(mias_data_s1 , i,j)        
        if len(dataFrame) >= 3:
            print(" len {} sup {} and conf = {} ".format(len(dataFrame), i , j))