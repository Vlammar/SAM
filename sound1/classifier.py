from feature_extract import *
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


X,y= loadDatas()
#X,y= computeAndSaveDatas ()
X = X.reshape(X.shape[0],-1)
#X_mfcc = X[:,:975*3]
#X_chromas = X[:,975*3:]
print(X.shape)
#print(X_mfcc.shape)
#print(y.shape)
#X = StandardScaler().fit_transform(X)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

classifier_and_params = {
		 "KNN":(KNeighborsClassifier(),{'n_neighbors':[1,3,5,9]}),
		 "SVM":(SVC(),{'kernel':('linear', 'rbf'), 'C':[1, 10,20],'max_iter':[1e5]}), 
		 "Decision Tree":(DecisionTreeClassifier(),{'max_depth':[1,5,10,20]}),
		 "Random Forest":(RandomForestClassifier(),{'max_depth':[1,5,10,20]}),
		 "AdaBoost":(AdaBoostClassifier(),{}),
         "Naive Bayes":(GaussianNB(),{})
      }

for clf_key in classifier_and_params:
	clf, param = classifier_and_params[clf_key]
	gridCV = GridSearchCV(cv=5,estimator=clf,param_grid=param)
	gridCV.fit(X, y)
	
	print(clf_key,' ',gridCV.best_params_,':',gridCV.best_score_)

