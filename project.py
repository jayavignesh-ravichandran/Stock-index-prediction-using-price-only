# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a project script file.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier 
from sklearn import neighbors
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


def QDA(X_train,y_train,X_test,y_test):
    dc_class= QuadraticDiscriminantAnalysis()
    dc_class.fit(X_train,y_train)
    predict=dc_class.predict(X_test)
    print (predict)
    print(" Discriminant accuracy",dc_class.score(X_test,y_test))
    return dc_class.score(X_test,y_test)

def svcClass(X_train,y_train,X_test,y_test):
    suv= SVC()
    suv.fit(X_train,y_train)
    predict=suv.predict(X_test)
    print (predict)
    print(" suv accuracy",suv.score(X_test,y_test))
    return suv.score(X_test,y_test)

def knnClass(X_train,y_train,X_test,y_test):
    knn= neighbors.KNeighborsClassifier()
    knn.fit(X_train,y_train)
    predict=knn.predict(X_test)
    print (predict)
    print(" knn accuracy",knn.score(X_test,y_test))
    return knn.score(X_test,y_test)

def treeClass(X_train,y_train,X_test,y_test):
    detclass= DecisionTreeClassifier()
    detclass.fit(X_train,y_train)
    predict=detclass.predict(X_test)
    print (predict)
    print(" Tree accuracy",detclass.score(X_test,y_test))
    return detclass.score(X_test,y_test)

def reglog(X_train,y_train,X_test,y_test):
    reglog= LogisticRegression()
    reglog.fit(X_train,y_train)
    predict=reglog.predict(X_test)
    print(predict)
    print("Log-regression accuracy",reglog.score(X_test,y_test))
    return reglog.score(X_test,y_test)

def randomForest(X_train,y_train,X_test,y_test):
    ran= RandomForestClassifier(n_estimators=500,n_jobs=-1)
    ran.fit(X_train,y_train)
    predict=ran.predict(X_test)
    print(predict)
    print("Random Forest accuracy",ran.score(X_test,y_test))
    return ran.score(X_test,y_test)

def tensplitfolds(X_train):
    print("Fold split function starts")
    total_length=len(X_train)
    fold=np.zeros(10,dtype=int)
    length_Cutter=(round(total_length*0.1))
    print(length_Cutter)
    i=0
    for i in range(10):
        if(i==0):
            fold[i]=length_Cutter
        if(i>0):
            fold[i]=length_Cutter+fold[i-1]
            if(fold[i]>train):
                fold[i]=fold[i]-(fold[i]-train)
        print(i,fold[i])
    print("Fold split function ends")
    return fold

def highestMeanAccuracy(QDA_accuracy,svcClass_accuracy,knnClass_accuracy,treeClass_accuracy,reglog_accuracy,randomForest_accuracy):
    highaccuracy=['technique',0]
    mean_store = np.array([QDA_accuracy,svcClass_accuracy,knnClass_accuracy,treeClass_accuracy,reglog_accuracy,randomForest_accuracy]) 
    mean_store.max()
    for i in range(6):
        if(mean_store[i]==mean_store.max()):
            if(i==0):
                highaccuracy=['QDA_accuracy',mean_store[i]]
            elif(i==1):
                highaccuracy=['svcClass_accuracy',mean_store[i]]
            elif(i==2):
                highaccuracy=['knnClass_accuracy',mean_store[i]]
            elif(i==3):
                highaccuracy=['treeClass_accuracy',mean_store[i]]
            elif(i==4):
                highaccuracy=['reglog_accuracy',mean_store[i]]
            elif(i==5):
                highaccuracy=['randomForest_accuracy',mean_store[i]]
    return highaccuracy  

def getRecommendation(highaccuracylist):
    max=0
    maxAfterTechnique=0
    accuracyComparisonList= []
    choosenTechniqueList=[]
    recommendList=[]
    chosenTech=''
    for i in range(len(highaccuracylist)):
        accuracyComparisonList.append(highaccuracylist[i][1])
    for i in range(len(accuracyComparisonList)):
        if (max<accuracyComparisonList[i]):
            max=accuracyComparisonList[i]
    qdacounter=0
    svccounter=0
    knncounter=0
    treecounter=0
    reglogcounter=0
    ranForcounter=0
    for i in range(len(highaccuracylist)):
        if(highaccuracylist[i][0]=='QDA_accuracy'):
            qdacounter=qdacounter+1
        elif(highaccuracylist[i][0]=='svcClass_accuracy'):
            svccounter=svccounter+1
        elif(highaccuracylist[i][0]=='knnClass_accuracy'):
            knncounter=knncounter+1
        elif(highaccuracylist[i][0]=='treeClass_accuracy'):
            treecounter=treecounter+1
        elif(highaccuracylist[i][0]=='reglog_accuracy'):
            reglogcounter=reglogcounter+1
        elif(highaccuracylist[i][0]=='randomForest_accuracy'):
            ranForcounter=ranForcounter+1
    counterArray=np.array([qdacounter,svccounter,knncounter,treecounter,reglogcounter,ranForcounter])

    for i in range(counterArray.size):
        if(counterArray[i]==counterArray.max()):
           if(i==0):
              chosenTech='QDA_accuracy'
           elif(i==1):
              chosenTech='svcClass_accuracy'
           elif(i==2):
              chosenTech='knnClass_accuracy'
           elif(i==3):
              chosenTech='treeClass_accuracy'
           elif(i==4):
              chosenTech='reglog_accuracy'
           elif(i==5):
              chosenTech='randomForest_accuracy'
# position number works eg: choosenTechnique[0] means 1 day returns and choosenTechnique[1] means 2 day returns   
    for i in range(len(highaccuracylist)):
        if(highaccuracylist[i][0]==chosenTech):
            choosenTechniqueList.append(highaccuracylist[i])
            print("choosenTechniqueList\n\n",choosenTechniqueList)
    print("choosenTechniqueList size \n\n",len(choosenTechniqueList))
    print("choosenTechniqueList\n\n",choosenTechniqueList[0][1])
    print("MAx available\n\n",max)
    
    for i in range(len(choosenTechniqueList)):
        if (maxAfterTechnique<choosenTechniqueList[i][1]):
            maxAfterTechnique=choosenTechniqueList[i][1]

    for i in range(len(choosenTechniqueList)):
        if(choosenTechniqueList[i][1]==maxAfterTechnique):
            recommendList=[choosenTechniqueList[i],i+1]
            print("recommendList",recommendList)
    return recommendList

def modelChosendecrypt(recommendedList):
    model=''
    if(recommendedList[0]=='reglog_accuracy'):
        model="Logarthmic Regression"
    elif(recommendedList[0]=='QDA_accuracy'):
        model="Quadratic Discriminant Analysis"
    elif(recommendedList[0]=='svcClass_accuracy'):
        model="Support Vector Machine"
    elif(recommendedList[0]=='knnClass_accuracy'):
        model="K-Nearest Neighbors"
    elif(recommendedList[0]=='treeClass_accuracy'):
        model="Decision Tree"
    elif(recommendedList[0]=='randomForest_accuracy'):
        model="Random Forest"
    return model

    
def finalModel(X_train,y_train,X_test,y_test,recommendedList):
    accuracy=0
    if(recommendedList[0]=='reglog_accuracy'):
        accuracy=reglog(X_train,y_train,X_test,y_test)
    elif(recommendedList[0]=='QDA_accuracy'):
        accuracy=QDA(X_train,y_train,X_test,y_test)
    elif(recommendedList[0]=='svcClass_accuracy'):
        accuracy=svcClass(X_train,y_train,X_test,y_test)
    elif(recommendedList[0]=='knnClass_accuracy'):
        accuracy=knnClass(X_train,y_train,X_test,y_test)
    elif(recommendedList[0]=='treeClass_accuracy'):
        accuracy=treeClass(X_train,y_train,X_test,y_test)
    elif(recommendedList[0]=='randomForest_accuracy'):
        accuracy=randomForest(X_train,y_train,X_test,y_test)
    return accuracy
    
            
    
data=pd.read_excel("D:/IIM/Term 5/Big Data Analytics/Project/Index_returns.xlsx")
s=data.columns
totalRows=len(data.index)
test=round(totalRows*0.2)
train=totalRows-test
xdata=data.copy()
dataTrain=xdata[xdata.index<train]
dataTest=xdata[xdata.index>=train]
fold=tensplitfolds(dataTrain)
highaccuracylist=[]

for i in range(5):
    dat=pd.DataFrame(dataTrain,columns=s[1:(9*(i+1))+1])
    tar=pd.DataFrame(dataTrain,columns=s[46:47])
    dtTotalRows=len(dat.index)
    print (dtTotalRows)
    dtest=round(dtTotalRows*0.2)
    dtrain=dtTotalRows-dtest
    print ("Train",dtrain)
    print ("Test",dtest)
    x=dat.copy()
    y=tar.copy()
# Cross validation
    QDA_accuracy=np.zeros(9,dtype=float)
    svcClass_accuracy=np.zeros(9,dtype=float)
    knnClass_accuracy=np.zeros(9,dtype=float)
    treeClass_accuracy=np.zeros(9,dtype=float)
    reglog_accuracy=np.zeros(9,dtype=float)
    randomForest_accuracy=np.zeros(9,dtype=float)
    for j in range(9):
        X_train = x[x.index < fold[j]]
        y_train = y[y.index < fold[j]]
        X_test = x.loc[fold[j]:fold[j+1],:]
        y_test = y.loc[fold[j]:fold[j+1],:]
        QDA_accuracy[j]=QDA(X_train,y_train,X_test,y_test)
        svcClass_accuracy[j]=svcClass(X_train,y_train,X_test,y_test)
        knnClass_accuracy[j]=knnClass(X_train,y_train,X_test,y_test)
        treeClass_accuracy[j]=treeClass(X_train,y_train,X_test,y_test)
        reglog_accuracy[j]=reglog(X_train,y_train,X_test,y_test)
        randomForest_accuracy[j]=randomForest(X_train,y_train,X_test,y_test)
    highaccuracy=['technique',0]
    highAccuracy=highestMeanAccuracy(QDA_accuracy.mean(),svcClass_accuracy.mean(),
                        knnClass_accuracy.mean(),
                        treeClass_accuracy.mean(),reglog_accuracy.mean(),
                        randomForest_accuracy.mean())
    highaccuracylist.append(highAccuracy)
    print("Highest Accuracy",highAccuracy)
print ('\n \n \n',highaccuracylist)
recommendation=[]
recommendation= getRecommendation(highaccuracylist)
print ('\n \n \n',recommendation)
print ('\n \n \n',recommendation[1])
recommendedList=[]
recommendedList.append(recommendation[0][0])
recommendedList.append(recommendation[0][1])
recommendedList.append(recommendation[1])
print ("Your recommendedList",recommendedList)

dat_train=pd.DataFrame(dataTrain,columns=s[1:(9*(recommendedList[2]))+1])
tar_train=pd.DataFrame(dataTrain,columns=s[46:47])

dat_test=pd.DataFrame(dataTest,columns=s[1:(9*(recommendedList[2]))+1])
tar_test=pd.DataFrame(dataTest,columns=s[46:47])

finalChosenModel=modelChosendecrypt(recommendedList)
finalModelAccuracy=finalModel(dat_train,tar_train,dat_test,tar_test,recommendedList)

print ("\n\n***************************\n\n")
print ("Final Model Chosen:",finalChosenModel)
print ("Final Accuracy of the model=",finalModelAccuracy)
print ("\n\n***************************\n\n\n\n")




    
