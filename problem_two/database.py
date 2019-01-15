#
import pandas as pd

data = pd.read_csv('mgi_data_s1.csv' , header = None)
data.sort_values()
values = data.iloc[:].values

def make_table(dictionary):
    table = []
    for i in dictionary.keys():
        current = []
        for j in dictionary[i]:
            current = [i , j]
            table.append(current)
    return table
    


def get_dic(values):
    dic = {}
    for i in values:
        for j in range(1,len(i)):
            
            if i[0] not in dic.keys():    
                dic[i[0]] = []
            if str(i[j]) !='nan':
                if i[j] not in dic[i[0]]:
                    dic[i[0]].append(i[j])        
    return dic    
    
dataFrame = pd.DataFrame(make_table(get_dic(values)))
dataFrame.to_csv('result.csv', header = False , index = False)