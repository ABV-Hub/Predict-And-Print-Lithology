# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 01:20:57 2019

@author: Savitri
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing,model_selection,svm,neighbors
df=pd.read_csv('ex.csv')
df.drop(['Formation'],1,inplace=True)

x=np.array((df.drop['Lthology'],1))
y=np.array(df['Lthology'])

x_train,x_test,y_train,y_test=model_selection.test_train_split(x,y,test_size=0.2)

clf=svm.SVC()
clf.fit(x_train,y_train)

accuracy=clf.score(x_test,y_test)
ac=accuracy*100
print(ac)