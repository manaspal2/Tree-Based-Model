import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import cross_val_score

###########################
# Classification Tree
###########################
# Way of deriving: Multiple question of if-else
# Goal : 1. Infer the class labels
#        2. It is able to capture non-linear relationship between feature and labels.
#        3. Tree doesn't need to standardize the data
###########################
# Decision tree diagram
###########################
#                 Variable A (If less than 0.051)
#                           |
#                           |
#          +--------------------------------------+
#          |     True                  False      |
#          |                                      |
#      Variable B (If less than 0.010)     Variable B (If less than 0.015)
#          |                                      |
#          |                                      |
#   +--------------------+                +--------------------+
#   |                    |                |                    |
#  True                False            True                 False
# Property            Property         Property             Property
# of data             of data          of data              Of data

print ("\n############ Read data from datafile ############\n")
cancer_df = pd.read_csv("cancer_dataset.csv")
print (cancer_df.head())

print ("\n############ Select interesting data from datafile ############\n")
cancer_subset_df = cancer_df[["radius_mean", "concave points_mean", "diagnosis"]]
print (cancer_subset_df.head())

print ("\n############ Plot the dependent and independent variable data ############\n")
sns.scatterplot(data=cancer_subset_df,
                x="radius_mean",
                y="concave points_mean",
                hue="diagnosis",
                style="diagnosis")
plt.xlabel('X Axis: radius_mean')
plt.ylabel('Y Axis: concave points_mean')
plt.show()

############ Split the train and test set of data ############
y = cancer_subset_df['diagnosis']
X = cancer_subset_df.drop(["diagnosis"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    stratify=y, 
                                                    random_state=1)
dt = DecisionTreeClassifier(max_depth=2, random_state=1)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
print ("\nAccuracy score of Decision Tree Model:", accuracy_score(y_test, y_pred))

###################################
# Descision tree looks like below #
###################################
#                        Root Node [No parent node]
#                           |
#                           |
#          +--------------------------------------+
#          |      True                  False     |
#          |                                      |<=============== Branch
#      Internal Node                        Internal Node 
# [One parent node, two child node]     [One parent node, two child node]
#          |                                      |
#          |                                      |
#   +--------------------+                +--------------------+
#   |                    |                |                    |
#  Leaf                Leaf             Leaf                 Leaf
#

dt_gini = DecisionTreeClassifier(criterion='gini', random_state=1)
dt_gini.fit(X_train, y_train)
y_pred = dt_gini.predict(X_test)
print ("\nAccuracy score of Decision Tree Model:", accuracy_score(y_test, y_pred))


#########################################
# Descision tree for regression problem #
#########################################
print ("\n############ Read data from datafile ############\n")
car_df = pd.read_csv("auto_mpg/test-file.txt", 
                        sep='\s+',
                        header=None, 
                        names=['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'car name'])
print ("\n")
print (car_df.head())

print ("\n############ Data for plotting scatter plot ############\n")
test_df = car_df[['mpg', 'displacement']]
print (test_df)
sns.scatterplot(data=test_df,
                x="displacement",
                y="mpg",)
plt.xlabel('---- X Axis: displacement ----->')
plt.ylabel('---- Y Axis: mpg -----> ')
plt.show()

# The graphe shows non-linearity. Limear regression model will not be able to capture the same.
grouped_counts = test_df.groupby('mpg').count()
print(grouped_counts.to_string())


y = test_df['mpg']
print ("\nTarget Variable\n")
print (y)
X = test_df.drop(["mpg"], axis=1)
print ("\nIndependent Variable\n")
print (X)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    stratify=y, 
                                                    random_state=3)
dt_car = DecisionTreeClassifier(max_depth=4, min_samples_leaf=0.1, random_state=3)
# min_samples_leaf --> Means each leaf contains atleast 10% of training data
dt_car.fit(X_train, y_train)
y_pred = dt_car.predict(X_test)

mse_dt = MSE(y_test, y_pred)
rmse_dt = mse_dt**(1/2)
print ("\nRoot mean square error:", rmse_dt)

#############################################
# Model problems                           ##
#############################################
# Overfitting -> The model performs well with training dataset but performs poorly 
# with test dataset
# Underfitting -> The model performs poorly with training and test dataset, both the 
# cases error will be almost same.
# Generalization error - > This says that how much a model generalize on a unseen data
# Generalization error = Bias * Bias + Variance + Irreducible error (Generally contributed by noise)

# high bias model always cause underfitting
# high variance model alwasy causes overfitting

# Model coomplexity is a way to control these. Model complexity reduces the generalization error
# reduces the bias but increases variable. So the situation is bias-variance tradeoff.
# Need to have an optimum model complexity.

# Estimate generalization error:
# ==============================
# How to find it:-
# 1. Split the data to training and test data
# 2. Fit the function with training set
# 3. Evaluate the error of the function with test set
# 4. Generalization error of the function ~ test set error of the function

# Test data should not be used unless we are confident about the function's performance
# Cross validation (CV) is the solution.
# To find CV, split the training data into K-Folds. Find the error from each training data folds.
# CV Error = (Sum of all error from each folds/Number of folds)

# If CV Error of Function > training set error of Function, function is suffering from variance.
# Means function is overfitted with the training set.
# To solve this:- 
# ===============
# 1. Reduce model complexity
# 2. Gather more data
SEED = 123
y = test_df['mpg']
X = test_df.drop(["mpg"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    random_state=SEED)
dt_car = DecisionTreeClassifier(max_depth=4, min_samples_leaf=0.14, random_state=SEED)
MSE_CV = - cross_val_score(dt_car,
                         X_train,
                         y_train,
                         cv=10,
                         scoring='neg_mean_squared_error',
                         n_jobs=-1)
print (MSE_CV)
dt_car.fit(X_train,y_train)
y_predict_train = dt_car.predict(X_train)
y_predict_test = dt_car.predict(X_test)
print ("Mean Square error:", MSE_CV.mean())
print ("Train Set MSE:", MSE(y_train, y_predict_train))
print ("Test Set MSE:", MSE(y_test, y_predict_test))