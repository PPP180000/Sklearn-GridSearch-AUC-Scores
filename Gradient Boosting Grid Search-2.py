#!/usr/bin/env python
# coding: utf-8

# In[781]:


import pandas as pd
import numpy as np
from sklearn import ensemble 
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import learning_curve, GridSearchCV, train_test_split


# In[782]:


#read data
#https://www.kaggle.com/wendykan/lending-club-loan-data/download
loan_data = pd.read_csv("Desktop/loan.csv")
#print(loan_data.shape)


# In[783]:


print(loan_data.dtypes) 


# In[784]:


#loan_data.describe()


# In[785]:


loan_data['home_ownership']


# In[786]:


loan_data.describe().transpose() #what happened to the other variables?
# do you see any outliers?


# In[787]:


# Show number of missings in each of the above features. There are many ways. I do it this way:
# Deduct number of observations from count column in the describe table.
# To work with describe, let's check its type.
#type(loan_data.describe())


# In[788]:


#loan_data.shape


# In[ ]:





# In[789]:


# for i in loan_data:
#     if loan_data[i].isnull().sum() > 150000:
#         model_data = loan_data.drop(loan_data[i],1)
#         model_data


# In[790]:


# We are going to use sklearn to develop the model. 
# sklearn does not work with character variables; so we need to convert them to numeric. 

temp_data = loan_data.drop(["id",'member_id','url','sub_grade','emp_title','sec_app_open_acc','sec_app_revol_util',
                           'sec_app_open_act_il',  
                            #"sec_app_num_rev_accts ", "sec_app_chargeoff_within_12_mths",
                            #"sec_app_collections_12_mths_ex_med ", 
                            'sec_app_mths_since_last_major_derog','hardship_flag',
                            'hardship_type','hardship_reason','hardship_status','deferral_term','hardship_amount','hardship_start_date',
                            'hardship_end_date','payment_plan_start_date','hardship_length','hardship_dpd','hardship_loan_status',
                            'orig_projected_additional_accrued_interest','hardship_payoff_balance_amount','hardship_last_payment_amount',
                            'disbursement_method','debt_settlement_flag','debt_settlement_flag_date','settlement_status','settlement_date',
                            'settlement_amount','settlement_percentage','settlement_term',
                           'revol_bal_joint','sec_app_earliest_cr_line','sec_app_inq_last_6mths','sec_app_mort_acc',
                           'sec_app_num_rev_accts',"sec_app_chargeoff_within_12_mths",'sec_app_collections_12_mths_ex_med','mths_since_recent_bc_dlq',
                           'desc','mths_since_last_delinq','mths_since_last_record','issue_d','title','zip_code',
                           'addr_state','earliest_cr_line','initial_list_status','last_pymnt_d','next_pymnt_d','last_credit_pull_d',
                           'application_type','verification_status_joint'],1)


# In[791]:


temp_data.shape


# In[ ]:







# In[792]:


temp_data['home_ownership']


# In[793]:


#temp_data.dtypes


# In[794]:


temp_data[:].isnull().sum()


# In[795]:


#model_data = temp_data[~ temp_data["emp_length"].isnull()]


# In[796]:


temp_data['emp_length'] = temp_data['emp_length'].replace({"1 year": 1,'10+ years' : 10, "2 years": 2,"3 years": 3,                                 
                                                       "4 years": 4,"5 years": 5,"6 years": 6,"7 years": 7,"8 years": 8,
                                                        "9 years": 9, "< 1 year": 1, 'NaN': 0 })



# In[797]:


temp_data['emp_length']


# In[798]:


#set(temp_data['term'])


# In[799]:


temp_data['term'] = temp_data['term'].replace({' 36 months':36, ' 60 months':60})


# In[800]:


set(temp_data['term'])


# In[801]:


temp_data['grade'] = temp_data['grade'].replace({"A": 100.00,'B' : 80.00, 'C': 70.00, 'D': 60.00, "E": 50.00, "F": 40.00, "G": 30.00})


# In[802]:


temp_data['grade']


# In[ ]:





# In[803]:


#temp_data['home_ownership'] = temp_data['home_ownership'].replace({'ANY': 1, 'MORTGAGE': 2, 'NONE': 0 , 'OTHER':3, 'OWN':4, 'RENT':5})


# In[804]:


temp_data['home_ownership']


# In[ ]:





# In[805]:


set(temp_data['home_ownership'])


# In[806]:


temp_data['verification_status'] = temp_data['verification_status'].replace({'Not Verified':0, 'Source Verified':50, 'Verified':100})


# In[807]:


set(temp_data['verification_status'])


# In[808]:


temp_data['loan_status'] = temp_data['loan_status'].replace({'Charged Off':1,'Current':1,'Default':0,'Does not meet the credit policy. Status:Charged Off':0,
 'Does not meet the credit policy. Status:Fully Paid':0,'Fully Paid':1,'In Grace Period':1,'Late (16-30 days)':1,'Late (31-120 days)':0})


# In[809]:


set(temp_data['loan_status'])


# In[810]:


temp_data['pymnt_plan'] = temp_data['pymnt_plan'].replace({'n':0, 'y':1})


# In[811]:


set(temp_data['pymnt_plan'])


# In[812]:


# temp_data['purpose'] = temp_data['purpose'].replace({'car':1,
#  'credit_card':2,
#  'debt_consolidation':3,
#  'educational':4,
#  'home_improvement':5,
#  'house':6,
#  'major_purchase':7,
#  'medical':8,
#  'moving':9,
#  'other':0,
#  'renewable_energy':10,
#  'small_business':11,
#  'vacation':12,
#  'wedding':13})


# In[813]:


set(temp_data['purpose'])


# In[814]:


list = []
for i in temp_data:
    if temp_data[i].dtypes == object:
        print(temp_data[i])
        #we don't get any objects means removed all the objects and converted some 
        #into float by adding values in the features


# In[ ]:





# In[815]:


temp_data.dtypes


# In[816]:


#model_data = temp_data[~ temp_data["mort_acc"].isnull()]


# In[817]:


#model_data.drop(["mort_acc"],1,inplace=True)


# In[818]:


model_data = temp_data["mort_acc"].isnull().dropna().copy()


# In[819]:


print(temp_data["mort_acc"].isnull().sum())
print(model_data.shape)


# In[820]:


# model_data = temp_data[~ temp_data["emp_length"].isnull()]
# model_data = temp_data[~ temp_data["dti"].isnull()]
# model_data = temp_data[~ temp_data["delinq_2yrs"].isnull()]
# model_data = temp_data[~ temp_data["inq_last_6mths"].isnull()]
# model_data = temp_data[~ temp_data["open_acc"].isnull()]
# model_data = temp_data[~ temp_data["pub_rec"].isnull()]
# model_data = temp_data[~ temp_data["revol_util"].isnull()]
# model_data = temp_data[~ temp_data["total_acc"].isnull()]
# model_data = temp_data[~ temp_data["mo_sin_old_il_acct"].isnull()]
model_data = temp_data[~ temp_data["mths_since_recent_revol_delinq"].isnull()]


# In[ ]:





# In[821]:


model_data.shape


# In[822]:


model_data['avg_cur_bal'].mean()


# In[823]:


model_data['emp_length'].fillna(1, inplace = True)
model_data['dti'].fillna(10, inplace = True)
model_data['revol_util'].fillna(60, inplace = True)
model_data['mo_sin_old_il_acct'].fillna(88, inplace = True)
model_data['mo_sin_old_rev_tl_op'].fillna(260, inplace = True)
model_data['mo_sin_rcnt_rev_tl_op'].fillna(10, inplace = True)
model_data['mo_sin_rcnt_tl'].fillna(6, inplace = True)
model_data['mths_since_recent_bc'].fillna(15, inplace = True)
model_data['mths_since_recent_inq'].fillna(7, inplace = True)
model_data['num_accts_ever_120_pd'].fillna(0, inplace = True)
model_data['num_actv_bc_tl'].fillna(6, inplace = True)
model_data['num_actv_rev_tl'].fillna(4, inplace = True)

model_data['num_bc_sats'].fillna(4, inplace = True)
model_data['num_bc_tl'].fillna(7, inplace = True)
model_data['num_il_tl'].fillna(11, inplace = True)
model_data['num_op_rev_tl'].fillna(7, inplace = True)
model_data['num_rev_accts'].fillna(11, inplace = True)
model_data['num_rev_tl_bal_gt_0'].fillna(10, inplace = True)
model_data['num_sats'].fillna(4, inplace = True)
model_data['num_tl_120dpd_2m'].fillna(0, inplace = True)
model_data['num_tl_30dpd'].fillna(0, inplace = True)

model_data['num_tl_90g_dpd_24m'].fillna(0, inplace = True)
model_data['num_tl_op_past_12m'].fillna(3, inplace = True)
model_data['pct_tl_nvr_dlq'].fillna(66, inplace = True)
model_data['percent_bc_gt_75'].fillna(0, inplace = True)
model_data['tot_hi_cred_lim'].fillna(186287.010, inplace = True)
model_data['total_il_high_credit_limit'].fillna(45241, inplace = True)
model_data['policy_code'].fillna(1, inplace = True)
model_data['dti_joint'].fillna(0, inplace = True)

model_data['acc_now_delinq'].fillna(0, inplace = True)
model_data['tot_cur_bal'].fillna(150452, inplace = True)
model_data['open_acc_6m'].fillna(1, inplace = True)
model_data['open_act_il'].fillna(2, inplace = True)
model_data['open_il_12m'].fillna(1, inplace = True)
model_data['open_il_24m'].fillna(1, inplace = True)
model_data['mths_since_rcnt_il'].fillna(21, inplace = True)

model_data['total_bal_il'].fillna(36937, inplace = True)
model_data['il_util'].fillna(69, inplace = True)
model_data['open_rv_12m'].fillna(1, inplace = True)
model_data['open_rv_24m'].fillna(2, inplace = True)
model_data['max_bal_bc'].fillna(5250, inplace = True)
model_data['all_util'].fillna(57, inplace = True)
model_data['total_rev_hi_lim'].fillna(31189, inplace = True)
model_data['inq_fi'].fillna(1, inplace = True)
model_data['total_cu_tl'].fillna(1, inplace = True)

model_data['inq_last_12m'].fillna(2, inplace = True)
model_data['acc_open_past_24mths'].fillna(4.66, inplace = True)
model_data['bc_open_to_buy'].fillna(9488, inplace = True)
model_data['bc_util'].fillna(58, inplace = True)
model_data['chargeoff_within_12_mths'].fillna(0, inplace = True)

model_data['mths_since_last_major_derog'].fillna(23, inplace = True)
model_data['annual_inc_joint'].fillna(6013, inplace = True)
model_data['tot_coll_amt'].fillna(6013, inplace = True)
model_data['avg_cur_bal'].fillna(14203, inplace = True)


# In[824]:


model_data['avg_cur_bal'].isnull().sum()


# In[825]:


model_data.columns[56]


# In[ ]:





# In[826]:


#model_data.dropna(subset=['mo_sin_rcnt_tl'])


# In[827]:


model_data['chargeoff_within_12_mths'].isnull().sum()


# In[828]:


# Sklearn does not work with missing values; 
# so we need to impute missing values.
# first calculate number of missing

(model_data.shape[0] - model_data.describe().transpose()["count"]).values
    


# In[829]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
import sys
le = LabelEncoder()
model_data['home_ownership'] = le.fit_transform(model_data['home_ownership'])
# model_data = le.fit_transform(model_data['loan_status'])
# model_data = le.fit_transform(model_data['purpose'])


#model_data.iloc[:,8].shape


# In[830]:


# X_data = model_data.drop(['home_ownership'],1)
# X_data.shape


# In[831]:


#ohe = OneHotEncoder(categorical_features=model_data['home_ownership'])
#ohe
ct = ColumnTransformer(transformers=[('home_ownership',OneHotEncoder(), [8,11,13])], remainder='passthrough')
model_data = ct.fit_transform(model_data)



# In[832]:


model_data


# In[ ]:





# In[846]:


model_data[:].shape


# In[851]:


# Now we are ready to run gradient boosting.
# Define Y and X
Y = model_data[:,8]
X = np.delete(model_data,[8],1)


# In[852]:


X.shape


# In[853]:


# # run a simple model
# params = {'n_estimators': 3,'max_leaf_nodes':6,'learning_rate': 0.1, 'random_state':1}
# model = ensemble.GradientBoostingClassifier(**params)
# model.fit(X, Y)


# In[854]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=1)


# In[855]:


sel = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1))
sel.fit(X_train, y_train)


# In[856]:


print(sel.get_support())


# In[869]:


#removing unnecessary features
Y = model_data[:,8]
X = np.delete(model_data,[0,1,2,3,4,5,6,7,8,11,13,16,18,19,20,21,22,23,24,25,28,29,31,32,34,35,36,37,40,41,42,45,47,48,49,51,52,53,54,55,56,57,
                         59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,78,79,82,83,84,85,86,88,89,90,91,92,93,94,95,96,98,99,100,101,103,104,105,108,109],1)


# In[870]:


X.shape


# In[871]:


Y.shape


# In[872]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=1)


# In[873]:


sel = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1))
sel.fit(X_train, y_train)


# In[874]:


print(sel.get_support())


# In[875]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[877]:


# run a simple model
params = {'n_estimators': 20,'max_leaf_nodes':6,'learning_rate': 0.1, 'random_state':1, 'max_features': 21}
classifier = ensemble.GradientBoostingClassifier(**params)
classifier.fit(X_train, y_train)


# In[881]:


# calculate AUC
from sklearn.metrics import roc_auc_score
from sklearn import ensemble 
from sklearn.externals import joblib
roc_auc_score(Y, classifier.predict(X))      


# In[ ]:





# StandardScaler() : Scaling is used to give same weights to each variables so 
# that in our optimization problem will give us the best value instead of giving different values each time

# In[882]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[883]:


X_test


# In[896]:


from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier


def build_classifier():  
    classifier = Sequential()
    classifier.add(Dense(units=5, kernel_initializer='glorot_uniform', activation='relu'))
    classifier.add(Dense(units=5, kernel_initializer='glorot_uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='glorot_uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier
  
classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, nb_epoch=3)
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=1)

print(accuracies.mean())
print(accuracies.std())

# def another_classifier(optimizer):
#     classifier = Sequential()
#     classifier.add(Dense(units=6, kernel_initializer="glorot_uniform", activation="relu"))
#     classifier.add(Dense(units=6, kernel_initializer="glorot_uniform", activation="relu"))
#     classifier.add(Dense(units=1, kernel_initializer="glorot_uniform", activation="sigmoid"))
#     classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
#     return classifier

# classifier = KerasClassifier(build_fn=build_classifier)
# parameters = {
#     'batch_size': [25,32], 
#     'nb_epoch':[1,2], 
#     'optimizer': ['adam', 'rmsprop']
#   }
# grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10)
# grid_search = grid_search.fit(X_train, y_train)

# best_parameters = grid_search.best_params_
# best_accuracy = grid_search.best_score_

# print(best_parameters)
# print(best_accuracy)

# classify()


# In[884]:


# run a simple model

classifier = ensemble.GradientBoostingClassifier(**params)
classifier.fit(X_train, y_train)

# Predict
Y_pred = classifier.predict(X_test)
Y_prob = classifier.predict_proba(X_test)

# Make the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, Y_pred)
# print(cm)

# Calculate AUC 
from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test, Y_pred))


# In[889]:


# now we run a grid search on GBM parameters
# save the results to choose the best parameters
import pandas as pd
from sklearn.metrics import roc_auc_score
results = pd.DataFrame(columns=["# Trees", "Max Features", "Learning Rate",
                                "Max Leaf Nodes", "Min Split","AUC_Test","AUC_Train"])
    
for n_estimators in [1, 2]:
        for max_features in ['sqrt','auto']:
                for learning_rate in [0.001, 0.01, 0.1, 0.5]:
                    for max_leaf_nodes in [6, 8, 12, 16, 20]:
                            for min_samples_split in [0.00001, 0.1, 0.2, 0.3]:
        
                                params = {'n_estimators': n_estimators,  
                                  'max_features':max_features,'max_leaf_nodes':max_leaf_nodes, 
                                  'learning_rate': learning_rate, 
                                  'min_samples_split': min_samples_split,
                                  'subsample':0.8, 'random_state':1}
                                model = ensemble.GradientBoostingClassifier(**params)
                                model.fit(X, Y)
                                results.loc[len(results)]=[n_estimators, max_features, learning_rate,
                                               max_leaf_nodes, min_samples_split,
                                               roc_auc_score(Y_pred, model.predict(X_test)),
                                                          roc_auc_score(y_train, model.predict(X_train))]


# In[ ]:


results#.sort_values(['AUC_Train', "AUC_Test"], ascending = False).head(10)


# In[ ]:


from sklearn.metrics import roc_curve#results['AUC_Test']
fpr, tpr, thresholds = roc_curve(Y_pred, model.predict(X_test))


# In[ ]:


#choosing best parameters from gridsearch
params = {'n_estimators': 2,'max_features':'auto','max_leaf_nodes':20,
          'learning_rate': 0.5,'min_samples_split': 0.3,'subsample':0.8, 'random_state':1}
best_model = ensemble.GradientBoostingClassifier(**params)
best_model.fit(X_train, y_train)


# In[ ]:





# In[ ]:


feature_importance = pd.DataFrame()
#feature_importance['Variable'] = 
feature_importance['Importance'] = best_model.feature_importances_

# feature_importance values in descending order
feature_importance.sort_values(by='Importance', ascending=False).head(10)


# In[ ]:


proba_classes = pd.DataFrame()
proba_classes['Y']= y_test
proba_classes['Class:0'] = best_model.predict_proba(X_test)[:, 0]
proba_classes['Class:1'] = best_model.predict_proba(X_test)[:, 1]
proba_classes.head()


# In[ ]:


plt.figure(figsize=(10,6))
proba_classes[Y_prob <= 1]['Class:1'].hist(alpha=0.5,color='blue',
                                              bins=100,label='Y = 1')
proba_classes[Y_prob>=0]['Class:0'].hist(alpha=0.5,color='orange',
                                              bins=100,label='Y = 0')
plt.legend()
plt.xlabel('Class Probability');


# In[ ]:


plt.plot(fpr, tpr)
plt.show()


# In[ ]:


plt.plot(Y_prob)
plt.show()


# In[ ]:


Y_pred


# In[ ]:


feature_importance = pd.DataFrame()
feature_importance['Variable'] = X_train.columns
feature_importance['Importance'] = best_model.feature_importances_

# feature_importance values in descending order
feature_importance.sort_values(by='Importance', ascending=False).head(10)


# In[ ]:


indx = range(len(Y_prob))
indx


# In[ ]:


plt.plot(Y_prob)
plt.show()


# In[ ]:


Y_prob


# In[ ]:





# In[ ]:


Y_prob[5][1]


# In[ ]:


# Applying Grid Search to find the best model with the best parameters
# from sklearn.model_selection import GridSearchCV
# parameters = [{
#   'n_estimators': [20],
#   'max_leaf_nodes': [70],
#   'learning_rate': [0.1],
#   'random_state': [1]
# }]
# grid_search = GridSearchCV(
#   estimator=classifier,
#   param_grid=parameters,
#   scoring="accuracy",
#   cv=10,
#   n_jobs=1
# )
# grid_search = grid_search.fit(Y_pred, Y_prob)


# best_accuracy = grid_search.best_score_
# best_parameters = grid_search.best_params_

# print("Best Accuracy is: ", best_accuracy)
# print("Best Parameters are: ", best_parameters)


# In[ ]:


plot.grid_search(grid_search.grid_scores_, change='n_estimators', kind='bar')

# plt.hist(Y_prob)
plt.show()


# In[ ]:



from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense


# In[ ]:


model = Sequential()
model.add(Dense(units=64, kernel_initializer = 'glorot_uniform', activation='relu', input_dim=100))
#model.add(Dense(units=10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])



# In[ ]:




