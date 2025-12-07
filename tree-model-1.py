import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier

print ("########################\n"\
        "# Ensemble learning   ##\n"\
        "########################\n")
# Classification tree advantage:
# 1. Easy to understand
# 2. Easy to interpret
# 3. Easy to use
# 4. Flexiable: Capable to descrive non-linear dependency
# 5. No need to preprocessing - like standardization, normalization etc.
#####################
# Limitation:      ##
#####################
# If a single point is removed from the dataset, CART learning may change drastically
# Sensitive to small variation of dataset.
# Unconstrained CARTs may overfit the training set.

# In Ensambled learning, multiple models are used to predict independently.
# Finally meta-model aggregates the prediction and final result is less prone to error
# and robust.

SEED = 1
print ("\n############ Read data from datafile ############\n")
cancer_df = pd.read_csv("cancer_dataset.csv")
print (cancer_df.head())
#print (cancer_df.columns)

cancer_df.drop('Unnamed: 32', axis=1, inplace=True)

y = cancer_df['diagnosis']
X = cancer_df.drop(["diagnosis"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    random_state=SEED)
lr = LogisticRegression(random_state=SEED)
dt = DecisionTreeClassifier(random_state=SEED)
knn = KNN()

classifier = [('Logistic Regression', lr),
              ('K nearest neighbor', knn),
              ('Classification Tree', dt)
              ]

for clf_name, clf in classifier:
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print ("Accuracy score of ", clf_name, "model :", accuracy_score(y_test, y_pred))

vc = VotingClassifier(estimators=classifier)
vc.fit(X_train, y_train)
y_pred = vc.predict(X_test)
print ("Accuracy: {}".format(accuracy_score(y_test, y_pred)))

print ("\n#####################################")
print ("\n## BAGGING - BOOTSTRAP AGGREGATION ##")  
print ("\n#####################################")
# Like ensambled learning, same dataset to broken up into bootstrap sample and feed 
# to multiple instance of the algorithm and finally all the output is aggregated based 
# on the nature of the problem.
# For Classification - > Majority voting method is used
# For Regression ----- > Average prediction is used

SEED = 1
print ("\n############ Read data from datafile ############\n")
cancer_dframe = pd.read_csv("cancer_dataset.csv")
print (cancer_dframe.head())
#print (cancer_dframe.columns)
cancer_dframe.drop('Unnamed: 32', axis=1, inplace=True)

y = cancer_dframe['diagnosis']
X = cancer_dframe.drop(["diagnosis"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3,
                                                    stratify=y,
                                                    random_state=SEED)

dt = DecisionTreeClassifier(max_depth=4, 
                            min_samples_leaf=0.16,
                            random_state=SEED)
bc = BaggingClassifier(estimator=dt, n_estimators=300, n_jobs=-1)
bc.fit(X_train, y_train)
y_pred = bc.predict(X_test)
acc_score = accuracy_score(y_test, y_pred)
print ("Accuracy score : {}".format(acc_score))
print ("\n#################")
print ("\n# Bagging      ##")
print ("\n#################")
# Some of the instances might be used several times where as other samples are not 
# used at all.
# 63% traiining instances are sampled and 37% anr not sampled, those are considered as 
# OOB instances.

# Using bootstrap instances train the model and using oob instances test the model.
# Funally the oov score is found by the average of those all oob scores.
SEED = 1
print ("\n############ Read data from datafile ############\n")
cancer_dframe = pd.read_csv("cancer_dataset.csv")
print (cancer_dframe.head())
#print (cancer_dframe.columns)
cancer_dframe.drop('Unnamed: 32', axis=1, inplace=True)

y = cancer_dframe['diagnosis']
X = cancer_dframe.drop(["diagnosis"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3,
                                                    stratify=y,
                                                    random_state=SEED)

dt = DecisionTreeClassifier(max_depth=4, 
                            min_samples_leaf=0.16,
                            random_state=SEED)
bc = BaggingClassifier(estimator=dt, 
                       n_estimators=300, 
                       oob_score=True,
                       n_jobs=-1)
bc.fit(X_train, y_train)
y_pred = bc.predict(X_test)
acc_score = accuracy_score(y_test, y_pred)
print ("Test Set accuracy score : {}".format(acc_score))
oob_accurace = bc.oob_score_
print ("OOB accuracy score : {}".format(oob_accurace))

###################
# Random Forest  ##
###################
# It is an ensambled method.
# Base estimator used is Discision tree.
# Each estimator are trained on different bootstrap sample having the same size
# Out of all features, d no of features are sampled without replacement.
# Random forest makes the final prediction based on the type of problem - 
# Classification - majority voting
# Regression - averaging

SEED = 1
print ("\n############ Read data from datafile ############\n")
cancer_dataframe = pd.read_csv("cancer_dataset.csv")
print (cancer_dataframe.head())
cancer_dataframe.drop('Unnamed: 32', axis=1, inplace=True)

y = cancer_dataframe['diagnosis']
X = cancer_dataframe.drop(["diagnosis"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3,
                                                    random_state=SEED)

rf = RandomForestClassifier(n_estimators=400,
                            min_samples_leaf=0.12,
                            random_state=SEED)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print (y_pred)
acc_score = accuracy_score(y_test, y_pred)
print ("Accuracy Score:", acc_score)

#####################################################################################
SEED = 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y,
                                                    random_state=SEED)
dt = DecisionTreeClassifier(max_depth=1, random_state=SEED)
adb_clf = AdaBoostClassifier(base_estimator=dt, n_estimator=100)
adb_clf.fit(X_train, y_train)
y_pred_proba = adb_clf.predict_proba(X_test)