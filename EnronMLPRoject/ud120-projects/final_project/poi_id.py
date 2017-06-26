#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
sys.path.append("../outliers/")


from numpy import inf
from numpy import nan
from outlier_cleaner import outlierCleaner
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data 
from tester import test_classifier
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
#import xgboost
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import FeatureUnion
import matplotlib.pyplot as plt

### Task 1: Select what features you'll use.
import numpy as np
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'deferral_payments',   'bonus', 'total_stock_value', 'total_payments','deferred_income',   'exercised_stock_options', 'other', 'from_poi_to_this_person', 'from_this_person_to_poi', 'shared_receipt_with_poi']#['poi','salary', 'deferral_payments', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'exercised_stock_options', 'long_term_incentive', 'restricted_stock', 'from_poi_to_this_person', 'from_this_person_to_poi', 'shared_receipt_with_poi']#['poi','salary', 'deferral_payments', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'exercised_stock_options', 'long_term_incentive', 'restricted_stock', 'from_poi_to_this_person', 'from_this_person_to_poi', 'shared_receipt_with_poi']#['poi','salary', 'total_payments', 'restricted_stock_deferred', 'expenses', 'exercised_stock_options', 'long_term_incentive', 'other','restricted_stock',  'from_poi_to_this_person', 'from_this_person_to_poi'] # You will need to use more features

######################### full list of features##########################
#['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees','to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
## This is where the outlier total is removed. After inspecting the data visualy, which is commented in the final code, I do not see any other values that need to be removed. 
for key in data_dict.keys():
    if key == 'TOTAL':
        print data_dict.pop(key,None)
        
    #print "and its value is"
    #print data_dict[key]
#for key in data_dict.keys():
#    print key  
### Task 2: Remove outliers

reg = LinearRegression()

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
### to create a new feature i am going to use the ratio of all the stock related income i.e. total_stock_value + exercised_stock_options + restricted+stock_deffered against the total_payments made to the employee


my_dataset = data_dict

#new_feature_keys = {'ratio_of_stocks_vs_total_payment'}

#new_feature_dict  = {a: data_dict[a] for a in new_feature_keys }
#print new_feature_dict.keys()

##my_dataset['Stock_Salary'] = my_dataset[,2]+my_dataset['total_stock_value'] 

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = False)
print data.shape
#### add new feature here: This is the ratio of total stock value over the total payment. 
new_feature = data[:,4] / data[:,5]
new_feature[new_feature == inf] = 0
new_feature[np.isnan(new_feature)] = 0
print new_feature
#print new_feature

#new_feature = featureFormat(new_feature, new_feature_keys, sort_keys = False)

np.delete(data, [4,5], axis=1)

#np.concatenate([data, new_feature],axis = 1 )
data = np.column_stack([data, new_feature])
#print "Print the array after stacking"
#print data.shape
labels, features = targetFeatureSplit(data)


plt.figure()

features = np.array(features)
labels = np.array(labels)


#print features.shape
#print labels
#for col in range(len(features[0])):
#    if (all(i >= 0 for i in features[col])) == False:
#        for indx, value in enumerate(features[col]):
#            if value < 0:
                #print features[col][indx]
#                print features[col:,indx]
#indxlst = []
#for indx, row in enumerate(features):
#    if(all(i >= 0 for i in row)) ==False:
#        indxlst.append(indx)
        #print indx
        #features = np.delete(features, indx, axis =0)
        #print row
#print indxlst    
#features = np.delete(features,indxlst,axis =0)
#labels = np.delete(labels,indxlst,axis =0)
#    for col, rowval in enumerate(features)
    
                #features[col][indx] = -1 * features[col][indx]
                
#for col in range(len(features[0])):
    #print features_list[col+1]
#    if (all(i > 0 for i in features[col])) == False:
#        for i in features[col]:
#            if i < 0:
                #print features[col][i]
                #print labels[col]
                #print i
#    A = [features[col], labels]
#    plt.boxplot(A)
#    plt.title(features_list[col+1])
#    plt.show()


    #print col
    #print features[col]


#print len(labels)
#### try outlier cleaner 
##reg.fit(features, labels)
##predictions = reg.predict(features)
##cleaned_data = outlierCleaner( predictions, features, labels)
##features, labels, errors = zip(*cleaned_data)

################### MixMaxscaler ###############
min_max_scaler = preprocessing.MinMaxScaler()
#min_max_scaler = preprocessing.StandardScaler()

pca = PCA()
skb = SelectKBest()
estimators = [('skb',SelectKBest()),('pca', PCA())]
combined = FeatureUnion(estimators)
#pca.fit(features)
#features = pca.transform(features)
#print len(labels)
#print data
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

#clf = GaussianNB()

#clf = DecisionTreeClassifier()

#clf = SVC()

#clf = RandomForestClassifier()

clf = KNeighborsClassifier()

#clf = AdaBoostClassifier()

##### This is n example of pipeline
### anova_svm = Pipeline([('anova', anova_filter), ('svc', clf)])


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.2, random_state=32)
#clf.fit(features_train, labels_train)
#clf.predict(features_train)
#print clf.score(features_train, labels_train)
#print clf.score(features_test, labels_test)
###################### Stratified shuffle split common for all #################
sss = StratifiedShuffleSplit(n_splits=100,test_size=0.3,random_state=60)

##################### SVC classifier ###############################
#pipeline_svc = Pipeline([('pca',pca),('svc',clf)])
#svc_parameters = {'svc__kernel':['rbf'],'svc__C':[1,10],'pca__n_components':range(2,6)}
#gs = GridSearchCV(pipeline_svc, svc_parameters, n_jobs =-1, cv =sss)#, scoring = 'f1')
#gs.fit(features,labels)
#clf = gs.best_estimator_
#print clf
#print gs.score(features_test,labels_test)

#################### K nearest neighbour classifier ################

pipeline_KNN = Pipeline([('feature_sel',combined),('knn',clf)])
KNN_parameters = {'knn__n_neighbors': range(2,len(features_list)),"feature_sel__skb__k" : range(2,(len(features_list)-1)),'feature_sel__pca__n_components':range(2,(len(features_list)-1))}#"skb__k" : range(2,(len(features_list)-1))}#'pca__n_components':range(2,6)}
#gs = GridSearchCV(pipeline_KNN, KNN_parameters, n_jobs=-1, cv =sss)
#gs.fit(features,labels)
#clf = gs.best_estimator_
#print clf
#print gs.score(features_test, labels_test)

################ Gaussian NB classifier #################

#pipeline_GNB = Pipeline([('skb',skb),('gnb',clf)])
#pipeline_GNB = Pipeline([('feature_sel',combined),('gnb',clf)])
#GNB_parameters = {"feature_sel__skb__k" : range(2,(len(features_list)-1)),'feature_sel__pca__n_components':range(2,(len(features_list)-1))}#(len(features_list)-1))}
#GNB_parameters = {"skb__k" : range(2,(len(features_list)-1))}
#gs = GridSearchCV(pipeline_GNB, GNB_parameters, n_jobs =-1, cv =sss)
#gs.fit(features,labels)
#clf = gs.best_estimator_
#print clf
#print gs.score(features_test, labels_test)


################# Random Forest Classifier ############
#pipeline_RF = Pipeline([('scaling',min_max_scaler),('skb',skb),('rf',clf)])
#RF_parameters = { #'pca__n_components':range(2,6),
#           "skb__k" : range(2,(len(features_list)-1)),
#          "rf__n_estimators"      : [250, 300, 400, 500],
#          "rf__criterion"         : ["gini", "entropy"],
#          #"rf__max_features"      : range(3, 5),
#         "rf__max_depth"         : range(10, 20),
#           "rf__min_samples_split" : range(2, 4) ,
#           "rf__bootstrap": [True, False]}

gs = GridSearchCV(pipeline_KNN, KNN_parameters, n_jobs =-1, cv =sss, scoring ="f1")
gs.fit(features_train,labels_train)
clf = gs.best_estimator_
#print clf
#print gs.score(features_test, labels_test)
print gs.best_estimator_.named_steps['feature_sel']

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
print "Classifier form tester file : "
test_classifier(clf, my_dataset, features_list, folds = 1000)
dump_classifier_and_data(clf, my_dataset, features_list)
