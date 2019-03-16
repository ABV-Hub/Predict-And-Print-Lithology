import numpy as np
from sklearn import preprocessing,model_selection,neighbors
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


df=pd.read_csv('training-data/Well-all.csv')
df.dropna(inplace=True)
x=np.array(df.drop(['Lithology'],1))
y=np.array(df['Lithology']) 

x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.10) #20% test data
clf=DecisionTreeClassifier(criterion = "gini",random_state = 100,max_depth=3, min_samples_leaf=5) 
clf.fit(x_train,y_train)

accuracy=clf.score(x_test,y_test)
print(f"Accuracy : {accuracy*100} %")

with open("output/DecisionTreeprediction.csv","w") as f:
    f.write("DEPTH,Lithology,SGR,NPHI,RHOB,DT\n")
    
    df2=pd.read_csv('10thwell/Well-10_log_data.csv')
    a=np.array(df2.drop(['LITHOLOGY'],1))
    
    for sample in a:
        example_measures=np.array([sample[0],sample[1],sample[2],sample[3]])
        example_measures=example_measures.reshape(1,-1)
        prediction=clf.predict(example_measures)
        
        f.write(f"{sample[4]},{prediction[0]},{sample[0]},{sample[1]},{sample[2]},{sample[3]}\n")