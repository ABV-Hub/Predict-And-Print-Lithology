
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton,QLineEdit,QLabel,QDialog
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot

'''--------------------------------------------------------'''
import numpy as np
from sklearn import preprocessing,model_selection,neighbors,svm
import pandas as pd
from sklearn.ensemble import BaggingClassifier,RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
 

class AccuracyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
    def initUI(self):
        self.setWindowTitle("Accuracy-Meter")
        self.setGeometry(100,100,400,470)
        self.label = QLabel("~ SUPERVISED ALGORITHMS ~",self)
        self.label.move(130,10)
        '''-----------------KNN-----------------------'''
        self.label = QLabel("K-NearestNeighbours\n (BaggingClassifier):",self)
        self.label.move(70,40)
        self.textbox = QLineEdit(self)
        self.textbox.move(200,40)
        self.textbox.resize(100,30)
        
        '''-----------------RF-----------------------'''
        self.label1 = QLabel("RandomForest :",self)
        self.label1.move(70,80)
        self.textbox1 = QLineEdit(self)
        self.textbox1.move(200,80)
        self.textbox1.resize(100,30)
        
        '''-----------------GB-----------------------'''
        self.label2 = QLabel("Naive Bayes :",self)
        self.label2.move(70,120)
        self.textbox2 = QLineEdit(self)
        self.textbox2.move(200,120)
        self.textbox2.resize(100,30)
        '''-----------------DT-----------------------'''
        self.label3 = QLabel("Decision Tree :",self)
        self.label3.move(70,160)
        self.textbox3 = QLineEdit(self)
        self.textbox3.move(200,160)
        self.textbox3.resize(100,30)
        '''----------------LR-----------------------'''
        self.label4 = QLabel("Logistic Regression :",self)
        self.label4.move(70,200)
        self.textbox4 = QLineEdit(self)
        self.textbox4.move(200,200)
        self.textbox4.resize(100,30)
        '''----------------SVM-----------------------'''
        self.label5 = QLabel("Support Vector Machine :",self)
        self.label5.move(70,240)
        self.textbox5 = QLineEdit(self)
        self.textbox5.move(200,240)
        self.textbox5.resize(100,30)
        
        
       
        button = QPushButton('COMPARE',self)
        button.setToolTip('ACCURACY OF DIFFERENT ALGO COMPARISION')
        button.move(150,300)
        button.resize(100,40)
        '''----------------RESULT-----------------------'''
        self.label6 = QLabel("BEST SUPERVISED ALGORITHM",self)
        self.label6.move(130,360)
        self.textbox6 = QLineEdit(self)
        self.textbox6.move(50,400)
        self.textbox6.resize(300,40)
        
        
        
        
        button.clicked.connect(self.on_click)
        self.show()
    
    
    @pyqtSlot()
    def on_click(self):
        df=pd.read_csv('training-data/Well-all.csv')
        df.dropna(inplace=True)
        x=np.array(df.drop(['Lithology'],1))
        y=np.array(df['Lithology']) 
        x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.20) #20% test data
        
        clf=BaggingClassifier(neighbors.KNeighborsClassifier(),max_samples=0.5,max_features=0.5)
        clf.fit(x_train,y_train)
        KNN=clf.score(x_test,y_test)
        
        self.textbox.setText(f"{KNN*100:1.4f} %")
        with open("output/KNNprediction.csv","w") as f:
            f.write("Lithology,DT,SGR,NPHI,RHOB,DEPTH\n")
            df2=pd.read_csv('10thwell/Well-10_log_data.csv')
            a=np.array(df2.drop(['LITHOLOGY'],1))
            for sample in a:
                example_measures=np.array([sample[0],sample[1],sample[2],sample[3]])
                example_measures=example_measures.reshape(1,-1)
                prediction=clf.predict(example_measures)
                f.write(f"{prediction[0]},{sample[3]},{sample[0]},{sample[1]},{sample[2]},{sample[4]}\n")
        
        
        '''--------------------------------RF------------------------------------'''
        clf=RandomForestClassifier(n_estimators=100)
        clf.fit(x_train,y_train)
        RF=clf.score(x_test,y_test)
        self.textbox1.setText(f"{RF*100:1.4f} %")
        
        with open("output/RFprediction.csv","w") as f:
            f.write("Lithology,DT,SGR,NPHI,RHOB,DEPTH\n")
            df2=pd.read_csv('10thwell/Well-10_log_data.csv')
            a=np.array(df2.drop(['LITHOLOGY'],1))
            for sample in a:
                example_measures=np.array([sample[0],sample[1],sample[2],sample[3]])
                example_measures=example_measures.reshape(1,-1)
                prediction=clf.predict(example_measures)
                f.write(f"{prediction[0]},{sample[3]},{sample[0]},{sample[1]},{sample[2]},{sample[4]}\n")
        
        '''----------------------------------GNB--------------------------------------------'''
        clf=GaussianNB()
        clf.fit(x_train,y_train)
        NB=clf.score(x_test,y_test)
        self.textbox2.setText(f"{NB*100:1.4f} %")
        with open("output/NBprediction.csv","w") as f:
            f.write("Lithology,DT,SGR,NPHI,RHOB,DEPTH\n")
            df2=pd.read_csv('10thwell/Well-10_log_data.csv')
            a=np.array(df2.drop(['LITHOLOGY'],1))
            for sample in a:
                example_measures=np.array([sample[0],sample[1],sample[2],sample[3]])
                example_measures=example_measures.reshape(1,-1)
                prediction=clf.predict(example_measures)
                f.write(f"{prediction[0]},{sample[3]},{sample[0]},{sample[1]},{sample[2]},{sample[4]}\n")
        
        '''---------------------------------------DECISION TREE ---------------------------------------'''
        clf=DecisionTreeClassifier(criterion = "gini",random_state = 100,max_depth=3, min_samples_leaf=5) 
        clf.fit(x_train,y_train)
        DT=clf.score(x_test,y_test)
        self.textbox3.setText(f"{DT*100:1.4f} %")
        with open("output/DecisionTreeprediction.csv","w") as f:
            f.write("Lithology,DT,SGR,NPHI,RHOB,DEPTH\n")
            df2=pd.read_csv('10thwell/Well-10_log_data.csv')
            a=np.array(df2.drop(['LITHOLOGY'],1))
            for sample in a:
                example_measures=np.array([sample[0],sample[1],sample[2],sample[3]])
                example_measures=example_measures.reshape(1,-1)
                prediction=clf.predict(example_measures)
                f.write(f"{prediction[0]},{sample[3]},{sample[0]},{sample[1]},{sample[2]},{sample[4]}\n")
                
        '''------------------------------------------LR-----------------------------------------------'''
        reg=LogisticRegression()
        reg.fit(x_train,y_train)
        LR=reg.score(x_test,y_test)
        self.textbox4.setText(f"{LR*100:1.4f} %")
        with open("output/LRprediction.csv","w") as f:
            f.write("Lithology,DT,SGR,NPHI,RHOB,DEPTH\n")
            df2=pd.read_csv('10thwell/Well-10_log_data.csv')
            a=np.array(df2.drop(['LITHOLOGY'],1))
            for sample in a:
                example_measures=np.array([sample[0],sample[1],sample[2],sample[3]])
                example_measures=example_measures.reshape(1,-1)
                prediction=reg.predict(example_measures)
                f.write(f"{prediction[0]},{sample[3]},{sample[0]},{sample[1]},{sample[2]},{sample[4]}\n")
        '''------------------------------------------SVM-----------------------------------------------'''
        clf=svm.SVC(gamma='auto')
        clf.fit(x_train,y_train)
        SM=clf.score(x_test,y_test)
        self.textbox5.setText(f"{SM*100:1.4f} %")
        with open("output/SVMprediction.csv","w") as f:
            f.write("Lithology,DT,SGR,NPHI,RHOB,DEPTH\n")
            df2=pd.read_csv('10thwell/Well-10_log_data.csv')
            a=np.array(df2.drop(['LITHOLOGY'],1))
            for sample in a:
                example_measures=np.array([sample[0],sample[1],sample[2],sample[3]])
                example_measures=example_measures.reshape(1,-1)
                prediction=clf.predict(example_measures)
                f.write(f"{prediction[0]},{sample[3]},{sample[0]},{sample[1]},{sample[2]},{sample[4]}\n")
        
        
        '''-----------------------------------------Result--------------------------------------------------'''
        
        
        best=""
        
        if KNN>RF and KNN>LR and KNN>SM and KNN>DT and KNN>NB:
            best=f"K- Nearest Neighbours with Accuracy : {KNN*100:1.4f} %"
        elif RF>KNN and RF>LR and RF>SM and RF>DT and RF>NB:
            best=f"Random Forest with Accuracy : {RF*100:1.4f} %"
        elif LR>RF and LR>KNN  and LR>SM and LR>DT and LR>NB:
            best=f"Logistic Regression with Accuracy : {LR*100:1.4f} %"
        elif SM>RF and SM>KNN  and SM>LR and SM>DT and SM>NB:
            best=f"Support Vector Machine with Accuracy : {SM*100:1.4f} %"
        elif DT>RF and DT>KNN  and DT>SM and DT>LR and DT>NB:
            best=f"Decision Tree with Accuracy : {DT*100:1.4f} %"
        else:
             best=f"Naive Bayes with Accuracy : {NB*100:1.4f} %"
            
        
        
        self.textbox6.setText(best)
        
        
        



app = QApplication(sys.argv)
a=AccuracyApp()
sys.exit(app.exec_())


