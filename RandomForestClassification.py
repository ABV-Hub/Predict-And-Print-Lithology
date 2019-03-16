import numpy as np
from sklearn import preprocessing,model_selection
import pandas as pd
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import confusion_matrix


df=pd.read_csv('training-data/Well-all.csv')
df=df.dropna()
x=np.array(df.drop(['Lithology'],1))
y=np.array(df['Lithology']) 

x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.20) #20% test data

clf=RandomForestClassifier(n_estimators=80)
clf.fit(x_train,y_train)
accuracy=clf.score(x_test,y_test)
print(f"Accuracy : {accuracy*100} %")

with open("output/RFprediction.csv","w") as f:
    f.write("DEPTH,Lithology,SGR,NPHI,RHOB,DT\n")
    
    df2=pd.read_csv('10thwell/Well-10_log_data.csv')
    a=np.array(df2.drop(['LITHOLOGY'],1))
    y_pred=[]
    for sample in a:
        example_measures=np.array([sample[0],sample[1],sample[2],sample[3]])
        example_measures=example_measures.reshape(1,-1)
        prediction=clf.predict(example_measures)
        y_pred.append(prediction[0])
        f.write(f"{sample[4]},{prediction[0]},{sample[0]},{sample[1]},{sample[2]},{sample[3]}\n")
    
    y_actual=[]
    a=df2["LITHOLOGY"]
    for i in a:
        y_actual.append(i)
    print(confusion_matrix(y_actual,y_pred).shape)