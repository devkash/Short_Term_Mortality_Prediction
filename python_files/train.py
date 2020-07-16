import itertools
import pandas as pd
import numpy as np
import math
from scipy import interp
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve,auc
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier  
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dense
from keras.models import Model
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.utils import to_categorical
from random import randint
from time_diff import time_diff

def plot_confusion_matrix(cm,classes,normalize=False,title='Normalized Confusion Matrix',cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def plot_2d_space(X, y, label='Classes'):   
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(
            X[y==l, 0],
            X[y==l, 1],
            c=c, label=l, marker=m
        )
    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()


def random_drop(features,target):
    i=0
    while i != len(target)-1:
        if target[i] == 0 and randint(1,10000) > 1:
            target = np.delete(target, i, axis=0)
            features = np.delete(features, i, axis=0)
        i+=1

    return features,target

#def random_forest(train_x,train_y,test_x,test_y):
#    #Classifier
#    rf = RandomForestClassifier(n_estimators=25,random_state=42,class_weight={0: 1.,1: 2000.},verbose=0)
#    rf.fit(train_x,train_y)
#
#    print("\nVALIDATION:")
#    print("\tscore: ",rf.score(val_x,val_y))
#    predictions = rf.predict(val_x)
#    print("\trecall: ",recall_score(val_y,predictions))
#    print("\tprecision: ",precision_score(val_y,predictions))
#    print("\tf1: ",f1_score(val_y,predictions))
#    print(confusion_matrix(val_y,predictions))
#
#    print("\nTESTING:")
#    print("\tscore: ",rf.score(test_x,test_y))
#    predictions = rf.predict(test_x)
#    print("\trecall: ",recall_score(test_y,predictions))
#    print("\tprecision: ",precision_score(test_y,predictions))
#    print("\tf1: ",f1_score(test_y,predictions))
#    print(confusion_matrix(test_y,predictions))
#
#
#    feature_importances = pd.DataFrame(rf.feature_importances_, index = feature_list, columns=['importance']).sort_values('importance', ascending=False)
#    print(feature_importances)
#    return

#def SVM(train_x,train_y,val_x,val_y,test_x,test_y):
#    clf = SVC(kernel='linear')
#    clf.fit(train_x,train_y)
#
#    print("\nVALIDATION:")
#    print("\tscore: ",clf.score(val_x,val_y))
#    predictions = clf.predict(val_x)
#    print("\trecall: ",recall_score(val_y,predictions))
#    print("\tprecision: ",precision_score(val_y,predictions))
#    print("\tf1: ",f1_score(val_y,predictions))
#    print(confusion_matrix(val_y,predictions))
#
#    print("\nTESTING:")
#    print("\tscore: ",clf.score(test_x,test_y))
#    predictions = clf.predict(test_x)
#    print("\trecall: ",recall_score(test_y,predictions))
#    print("\tprecision: ",precision_score(test_y,predictions))
#    print("\tf1: ",f1_score(test_y,predictions))
#    print(confusion_matrix(test_y,predictions))
#
#    feature_importances = pd.DataFrame(clf.feature_importances_, index = feature_list, columns=['importance']).sort_values('importance', ascending=False)
#    print(feature_importances)
#    return

def bayes_network(train_x,train_y,val_x,val_y,test_x,test_y):
    return

def decision_tree(train_x,train_y,val_x,val_y,test_x,test_y):
    clf = DecisionTreeClassifier() 
    clf.fit(train_x,train_y)

    print("\nVALIDATION:")
    print("\tscore: ",clf.score(val_x,val_y))
    predictions = clf.predict(val_x)
    print("\trecall: ",recall_score(val_y,predictions))
    print("\tprecision: ",precision_score(val_y,predictions))
    print("\tf1: ",f1_score(val_y,predictions))
    print(confusion_matrix(val_y,predictions))

    print("\nTESTING:")
    print("\tscore: ",clf.score(test_x,test_y))
    predictions = clf.predict(test_x)
    print("\trecall: ",recall_score(test_y,predictions))
    print("\tprecision: ",precision_score(test_y,predictions))
    print(confusion_matrix(test_y,predictions))

    feature_importances = pd.DataFrame(clf.feature_importances_, index = feature_list, columns=['importance']).sort_values('importance', ascending=False)
    print(feature_importances)
    return

def MLP(x_train,y_train,x_val,y_val,x_test,y_test):
    epochs, batch_size, verbose = 100, 16, 1
    y_train = to_categorical(y_train,2)
    y_val = to_categorical(y_val,2)
    y_test = to_categorical(y_test,2)

    #Define simple neural net with stochastic gradient descent
    model = Sequential()
    model.add(Dense(768,input_dim=11,init="uniform",activation="sigmoid"))
    model.add(Dense(384,activation="sigmoid",kernel_initializer="uniform"))
    model.add(Dense(2))
    model.add(Activation("sigmoid"))
    sgd = SGD(lr=0.01)
    model.compile(loss="binary_crossentropy",optimizer=sgd,metrics=["accuracy"])
    model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size,verbose=verbose)
    val_loss,val_accuracy = model.evaluate(x,y,batch_size=batch_size,verbose=verbose)
    test_loss,test_accuracy = model.evaluate(x_test,y_test,batch_size=batch_size,verbose=verbose)
    print("Val: ",val_loss,val_accuracy)
    print("Test: ",test_loss,test_accuracy)
    return


def find_op(y_vals,thresh):
    return [1 if y >= thresh else 0 for y in y_vals]

def random_forest(x,y):
    rf = RandomForestClassifier(n_estimators=25,random_state=2,verbose=0)
    rf.fit(x,y)
    return rf

def svm(x,y):
    clf = SVC(kernel='linear',verbose=1)
    clf.fit(x,y)
    return clf

def mlp(x,y):
    epochs,batch_size,verbose = 10,32,1

    mlp = Sequential()
    mlp.add(Dense(768,input_dim=11,init="uniform",activation="relu"))
    mlp.add(Dense(384,activation="relu",kernel_initializer="uniform"))
    mlp.add(Dense(2))
    #mlp.add(Activation("relu"))
    sgd = SGD(lr=0.01)
    mlp.compile(loss="binary_crossentropy",optimizer=sgd,metrics=["accuracy"])
    mlp.fit(x,y,epochs=epochs,batch_size=batch_size,verbose=verbose)


#features = pd.read_csv('all.csv',delimiter='\t')
features = pd.read_csv('all_first_day.csv',delimiter='\t')
col = features.columns.values.tolist()
#all_rows = features[['sofa_sofa','sapsii_sapsii','oasis_oasis','apsiii_apsiii','saps_saps','TARGET','sapsii_prob_sapsii','oasis_prob_oasis','apsiii_prob_apsiii']]

#'''
all_rows = features[['TARGET','sofa_sofa','sapsii_sapsii','oasis_oasis','apsiii_apsiii','saps_saps','sapsii_prob_sapsii','oasis_prob_oasis','apsiii_prob_apsiii']]
#'''

'''
all_rows = features[['TARGET',
#'sofa_sofa', 
'respiration_sofa', 'coagulation_sofa', 'liver_sofa', 'cardiovascular_sofa', 'cns_sofa', 'renal_sofa',

#'sapsii_sapsii',
'sapsii_prob_sapsii', 'age_score_sapsii', 'hr_score_sapsii', 'sysbp_score_sapsii', 'temp_score_sapsii', 'pao2fio2_score_sapsii', 'uo_score_sapsii', 'bun_score_sapsii', 'wbc_score_sapsii', 'potassium_score_sapsii', 'sodium_score_sapsii', 'bicarbonate_score_sapsii', 'bilirubin_score_sapsii', 'gcs_score_sapsii', 'comorbidity_score_sapsii', 'admissiontype_score_sapsii',

#'oasis_oasis',
'oasis_prob_oasis', 'age_score_oasis', 'preiculos_score_oasis', 'gcs_score_oasis', 'heartrate_score_oasis', 'meanbp_score_oasis', 'resprate_score_oasis', 'temp_score_oasis', 'urineoutput_score_oasis', 'mechvent_score_oasis', 'electivesurgery_score_oasis',

#'apsiii_apsiii',
'apsiii_prob_apsiii', 'hr_score_apsiii', 'meanbp_score_apsiii', 'temp_score_apsiii', 'resprate_score_apsiii', 'pao2_aado2_score_apsiii', 'hematocrit_score_apsiii', 'wbc_score_apsiii', 'creatinine_score_apsiii', 'uo_score_apsiii', 'bun_score_apsiii', 'sodium_score_apsiii', 'albumin_score_apsiii', 'bilirubin_score_apsiii', 'glucose_score_apsiii', 'acidbase_score_apsiii', 'gcs_score_apsiii',

#'saps_saps',
'age_score_saps', 'hr_score_saps', 'sysbp_score_saps', 'resp_score_saps', 'temp_score_saps', 'uo_score_saps', 'vent_score_saps', 'bun_score_saps', 'hematocrit_score_saps', 'wbc_score_saps', 'glucose_score_saps', 'potassium_score_saps', 'sodium_score_saps', 'bicarbonate_score_saps', 'gcs_score_saps']]
'''

#print(len(all_rows))
#print(all_rows['TARGET'].sum())
all_rows = all_rows.dropna()
#print(len(all_rows))
#print(all_rows['TARGET'].sum())
#quit()

benchmark = all_rows[['sapsii_prob_sapsii','oasis_prob_oasis','apsiii_prob_apsiii']]
sapsii_bench = np.array(benchmark['sapsii_prob_sapsii'])
oasis_bench = np.array(benchmark['oasis_prob_oasis'])
apsiii_bench = np.array(benchmark['apsiii_prob_apsiii'])

target = np.array(all_rows['TARGET'])
#features = np.array(all_rows[['sofa_sofa','sapsii_sapsii','oasis_oasis','apsiii_apsiii','saps_saps']])
features = np.array(all_rows.drop(['TARGET'],axis=1))

'''
train_x,test_x,train_y,test_y = train_test_split(features,target,test_size=0.8,random_state=42)
sm = SMOTE(random_state=42,ratio=1.0)
features_res, target_res = sm.fit_sample(train_x,train_y)
model = svm(features_res, target_res)
#probs = model.predict_proba(test_x)
#preds = probs[:,1]
preds = model.decision_function(test_x)
fpr,tpr,thresh = roc_curve(test_y,preds)
roc_auc = auc(fpr, tpr)
'''

n_splits = 5
kfold = StratifiedKFold(n_splits = n_splits, shuffle=True, random_state=42)

#fpr,tpr,thresh = [],[],[]
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
count = 0
for train, test in kfold.split(features,target):
    sm = SMOTE(random_state=42,ratio=1.0)
    features_res, target_res = sm.fit_sample(features[train],target[train])

    ###############################################
    # Model Function Call
    model = random_forest(features_res, target_res)
    ###############################################

    count += 1
    print(count)

    probs = model.predict_proba(features[test])
    preds = probs[:,1]
    fpr,tpr,thresh = roc_curve(target[test],preds)

    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
roc_auc = auc(mean_fpr, mean_tpr)
roc_auc = auc(fpr, tpr)

dist = math.sqrt(2)
opt = thresh[0]
for i in range(0,len(fpr)):
    new_dist = math.sqrt((fpr[i]**2)+((1-tpr[i])**2))
    if new_dist < dist:
        dist = new_dist
        opt = thresh[i]

##################################################
# Figure Titles
title = 'Random Forest'
##################################################

cnf = confusion_matrix(target[test],find_op(preds,opt))
#cnf = confusion_matrix(test_y,find_op(preds,opt))
plt.figure(1)
plot_confusion_matrix(cnf,classes=['0','1'],title=title,normalize=1)
plt.show()

#sapsii
fpr_sapsii,tpr_sapsii,thresh_sapsii = roc_curve(target,sapsii_bench)
roc_auc_sapsii = auc(fpr_sapsii, tpr_sapsii)

#oasis
fpr_oasis,tpr_oasis,thresh_oasis = roc_curve(target,oasis_bench)
roc_auc_oasis = auc(fpr_oasis, tpr_oasis)

#apsiii
fpr_apsiii,tpr_apsiii,thresh_apsiii = roc_curve(target,apsiii_bench)
roc_auc_apsiii = auc(fpr_apsiii, tpr_apsiii)


plt.figure(2)
plt.title(title)
plt.plot(fpr, tpr, 'b', label = 'AUC (SVM) = %0.2f' % roc_auc)
plt.plot(fpr_sapsii, tpr_sapsii, 'r', label = 'AUC (sapsii) = %0.2f' % roc_auc_sapsii)
plt.plot(fpr_oasis, tpr_oasis, 'g', label = 'AUC (oasis) = %0.2f' % roc_auc_oasis)
plt.plot(fpr_apsiii, tpr_apsiii, 'm', label = 'AUC (apsiii) = %0.2f' % roc_auc_apsiii)

plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()




#Extract target data
#target = np.array(features[''])
#features = features.drop(['Label','Domain'],axis=1)
#
#
##Extract training features to array
#feature_list = list(features.columns)
#features = np.array(features)
#
#features,target = random_drop(features,target)
##Print means of each data column (identify nans)
##print('\n',np.mean(target),)
##for i in range(11):
##    print(np.mean(features[:,i]))
##print()
#
##Split data into training, validation, and testing
#train_features,test_x,train_target,test_y = train_test_split(features,target,test_size=0.20,random_state=42)
#train_x,val_x,train_y,val_y = train_test_split(train_features,train_target,test_size = 0.25,random_state=42)
#
##Apply SMOTE
#sm = SMOTE(random_state=42,ratio=1.0)
#train_x_res, train_y_res = sm.fit_sample(train_x,train_y)
#
##Apply SMOTE + Tomek
##smt = SMOTETomek(ratio='auto')
##train_x_res, train_y_res = smt.fit_sample(train_x,train_y)
##
##plot_2d_space(train_x_res,train_y_res,'SMOTE + Tomek')
#
#print "\nRunning Random Forest Classifier..."
#random_forest(train_x_res,train_y_res,val_x,val_y,test_x,test_y)
##print "Running SVM Classifier..."
##SVM(train_x_res,train_y_res,val_x,val_y,test_x,test_y)
#print "\nRunning Decision Tree Classifier..."
#decision_tree(train_x_res,train_y_res,val_x,val_y,test_x,test_y)



#MLP
#MLP(train_x_res,train_y_res,val_x,val_y,test_features,test_target)
