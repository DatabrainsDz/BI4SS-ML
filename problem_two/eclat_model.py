# The Libraries
import pandas as pd
from fim import eclat

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


def run_eclat(dataset):
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
            
    results = []
    results= eclat(subjects,out = results , supp=0.05, zmin=2)
    
    
    found = []
    for i in range(len(results)):
        found.append(list(results[i][0]))
    
    return pd.DataFrame(found)


def write_result(dataframe, name):
        dataframe.to_csv(name, index=False, header=False)
        

dataFrame = run_eclat(mias_data_s1)
write_result(dataFrame , 's1.csv')
