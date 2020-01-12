from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def getBestClassifier(X,y):
	classifier_and_params = {
			 "KNN":(KNeighborsClassifier(),{'n_neighbors':[1,3,5,9,50]}),
			 "AdaBoost":(AdaBoostClassifier(),{}),
			 "Random Forest":(RandomForestClassifier(),{'max_depth':[1,5,10,20,100]}),
		     "Naive Bayes":(GaussianNB(),{}),
			 "Decision Tree":(DecisionTreeClassifier(),{'max_depth':[1,5,10,20,100]}),
			 "SVM":(SVC(),{'kernel':('linear', 'rbf'), 'C':[1, 10,20,200],'max_iter':[1e3]})
		  }

	for clf_key in classifier_and_params:
		clf, param = classifier_and_params[clf_key]
		gridCV = GridSearchCV(cv=5,estimator=clf,param_grid=param)
		gridCV.fit(X, y)	
		print(clf_key,' ',gridCV.best_params_,':',gridCV.best_score_)
