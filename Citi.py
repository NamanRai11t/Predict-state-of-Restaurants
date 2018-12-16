import pandas as pd
import numpy as np

from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
#from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
#to train dataset
features  = pd.read_csv('./train.csv')

features.drop('UIDX', axis=1, inplace=True)
features.drop('CAMIS', axis=1, inplace=True)
features.drop('PHONE', axis=1, inplace=True)
features.drop('BUILDING', axis=1, inplace=True)
#features.drop('STREET', axis=1, inplace=True)
features.drop('ZIPCODE', axis=1, inplace=True)
features.drop('INSPECTION DATE', axis=1, inplace=True)
features.drop('VIOLATION DESCRIPTION', axis=1, inplace=True)
# features.drop('GRADE', axis=1, inplace=True)
features.drop('GRADE DATE', axis=1, inplace=True)
features.drop('RECORD DATE', axis=1, inplace=True)
features.drop('DBA', axis=1, inplace=True)
features.drop('INSPECTION TYPE', axis=1, inplace=True)


to_drop = ['Not Applicable']
features = features[~features['CRITICAL FLAG'].isin(to_drop)]

features = features[pd.notnull(features['SCORE'])]


labels = features['CRITICAL FLAG'].unique().tolist()
mapping = dict( zip(labels,range(len(labels))) )
features.replace({'CRITICAL FLAG': mapping},inplace=True)

grades = features['GRADE'].unique().tolist()
mapping_grades = dict( zip(grades,range(len(grades))) )
features.replace({'GRADE': mapping_grades},inplace=True)
features['GRADE']=features['GRADE'].astype('category')


labels_boro = features['STREET'].unique().tolist()
mapping_street = dict( zip(labels_boro,range(len(labels_boro))))
features.replace({'STREET': mapping_street},inplace=True)
features['STREET']=features['STREET'].astype('category')


labels_boro = features['BORO'].unique().tolist()
mapping_boro = dict( zip(labels_boro,range(len(labels_boro))))
features.replace({'BORO': mapping_boro},inplace=True)
features['BORO']=features['BORO'].astype('category')

labels_CUISINE_DESCRIPTION = features['CUISINE DESCRIPTION'].unique().tolist()
mapping_cuisine_description = dict( zip(labels_CUISINE_DESCRIPTION,range(len(labels_CUISINE_DESCRIPTION))) )
features.replace({'CUISINE DESCRIPTION': mapping_cuisine_description},inplace=True)
features['CUISINE DESCRIPTION']=features['CUISINE DESCRIPTION'].astype('category')

labels_ACTION = features['ACTION'].unique().tolist()
mapping_action = dict( zip(labels_ACTION,range(len(labels_ACTION))) )
features.replace({'ACTION': mapping_action},inplace=True)
features['ACTION']=features['ACTION'].astype('category')

labels_VIOLATION_CODE = features['VIOLATION CODE'].unique().tolist()
mapping_violation_code = dict( zip(labels_VIOLATION_CODE,range(len(labels_VIOLATION_CODE))) )
features.replace({'VIOLATION CODE': mapping_violation_code},inplace=True)
features['VIOLATION CODE']=features['VIOLATION CODE'].astype('category')

labels_INSPECTION_NO = features['INSPECTION NO.'].unique().tolist()
mapping_inspection_no = dict( zip(labels_INSPECTION_NO,range(len(labels_INSPECTION_NO))) )
features.replace({'INSPECTION NO.': mapping_inspection_no},inplace=True)
'''
labels_INSPECTION_TYPE = features['INSPECTION TYPE'].unique().tolist()
mapping_inspection_type = dict( zip(labels_INSPECTION_TYPE,range(len(labels_INSPECTION_TYPE))) )
features.replace({'INSPECTION TYPE': mapping_inspection_type},inplace=True)
features['INSPECTION TYPE']=features['INSPECTION TYPE'].astype('category')
'''
#print(features.describe())
#print(features.dtypes)
labels = np.array(features['CRITICAL FLAG'])

features= features.drop('CRITICAL FLAG', axis=1)


features = np.array(features)

#Splitting


validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(features, labels, test_size=validation_size, random_state=seed)


'''
# Spot-Check Algorithms
models = []
#models.append(('LR', LogisticRegression()))
#models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
#models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = KFold(n_splits=10, random_state=seed)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
'''
scaler = StandardScaler()
print("training start")

#to fit model
knn =  AdaBoostClassifier()
knn.fit(scaler.fit_transform(X_train), Y_train)
print("training done")
#to test
features  = pd.read_csv('test.csv')

list_uidx = features['UIDX'].tolist()
list_score = features['SCORE'].tolist()

features.drop('UIDX', axis=1, inplace=True)
features.drop('CAMIS', axis=1, inplace=True)
features.drop('PHONE', axis=1, inplace=True)
features.drop('DBA', axis=1, inplace=True)
features.drop('BUILDING', axis=1, inplace=True)
# features.drop('STREET', axis=1, inplace=True)
features.drop('ZIPCODE', axis=1, inplace=True)
features.drop('INSPECTION DATE', axis=1, inplace=True)
features.drop('VIOLATION DESCRIPTION', axis=1, inplace=True)
# features.drop('GRADE', axis=1, inplace=True)
features.drop('GRADE DATE', axis=1, inplace=True)
features.drop('RECORD DATE', axis=1, inplace=True)


Inspection_no = []
Inspection_type = []
for i in (features['INSPECTION TYPE']):
	if(type(i)!=float):
		a = i.split("/")
		Inspection_no.append(a[1])
		Inspection_type.append(a[0])
	else:
		a = ' '
		Inspection_no.append(a)
		Inspection_type.append(a)

features.drop('INSPECTION TYPE', axis=1, inplace=True)
features['INSPECTION NO.'] = Inspection_no
features['INSPECTION TYPE'] = Inspection_type

features.drop('INSPECTION TYPE', axis=1, inplace=True)

mapping_boro_test={}
for i in features['BORO']:
	length=len(mapping_boro)
	if i in mapping_boro:
		mapping_boro_test[i] = mapping_boro[i]
	else:
		mapping_boro_test[i] = length
		length=length+1
features.replace({'BORO': mapping_boro_test},inplace=True)
features['BORO']=features['BORO'].astype('category')

mapping_grades_test={}
for i in features['GRADE']:
	length=len(mapping_grades)
	if i in mapping_grades:
		mapping_grades_test[i] = mapping_grades[i]
	else:
		mapping_grades_test[i] = length
		length=length+1
features.replace({'GRADE': mapping_grades_test},inplace=True)
features['GRADE']=features['GRADE'].astype('category')


mapping_street_test={}
for i in features['STREET']:
	length=len(mapping_street)
	if i in mapping_street:
		mapping_street_test[i] = mapping_street[i]
	else:
		mapping_street_test[i] = length
		length=length+1
features.replace({'STREET': mapping_street_test},inplace=True)
features['STREET']=features['STREET'].astype('category')
print("done")

mapping_cuisine_description_test={}
for i in features['CUISINE DESCRIPTION']:
	length=len(mapping_cuisine_description)
	if i in mapping_cuisine_description:
		mapping_cuisine_description_test[i] = mapping_cuisine_description[i]
	else:
		mapping_cuisine_description_test[i] = length
		length=length+1
features.replace({'CUISINE DESCRIPTION': mapping_cuisine_description_test},inplace=True)
features['CUISINE DESCRIPTION']=features['CUISINE DESCRIPTION'].astype('category')


mapping_action_test={}
for i in features['ACTION']:
	length=len(mapping_action)
	if i in mapping_action:
		mapping_action_test[i] = mapping_action[i]
	else:
		mapping_action_test[i] = length
		length=length+1
features.replace({'ACTION': mapping_action_test},inplace=True)
features['ACTION']=features['ACTION'].astype('category')

mapping_violation_code_test={}
for i in features['VIOLATION CODE']:
	length=len(mapping_violation_code)
	if i in mapping_violation_code:
		mapping_violation_code_test[i] = mapping_violation_code[i]
	else:
		mapping_violation_code_test[i] = length
		length=length+1
features.replace({'VIOLATION CODE': mapping_violation_code_test},inplace=True)
features['VIOLATION CODE']=features['VIOLATION CODE'].astype('category')

mapping_inspection_no_test={}
for i in features['INSPECTION NO.']:
	length=len(mapping_inspection_no)
	if i in mapping_inspection_no:
		mapping_inspection_no_test[i] = mapping_inspection_no[i]
	else:
		mapping_inspection_no_test[i] = length
		length=length+1
features.replace({'INSPECTION NO.': mapping_inspection_no_test},inplace=True)
'''
mapping_inspection_type_test={}
for i in features['INSPECTION TYPE']:
	length=len(mapping_inspection_type)
	if i in mapping_inspection_type:
		mapping_inspection_type_test[i] = mapping_inspection_type[i]
	else:
		mapping_inspection_type_test[i] = length
		length=length+1
features.replace({'INSPECTION TYPE': mapping_inspection_type_test},inplace=True)
features['INSPECTION TYPE']=features['INSPECTION TYPE'].astype('category')
#print(features.describe())
'''

features = np.array(features)

#to prdict
CRITICAL_FLAG =[]
for i in range(len(list_uidx)):
	if(list_score[i] not in range(-100,500)):

		CRITICAL_FLAG.append("Not Applicable")
	else:
		# Make predictions on validation dataset
		list_of_list = []
		list_of_list.append(features[i][:])
		prediction = knn.predict_proba(scaler.transform(list_of_list))
		if(prediction[0][1]<0.5):
			CRITICAL_FLAG.append("Critical")
		else:
			CRITICAL_FLAG.append("Not Critical")

Final = pd.DataFrame({"CRITICAL FLAG": CRITICAL_FLAG, "UIDX":list_uidx})
cols = Final.columns.tolist()

cols = cols[-1:] + cols[:-1]
Final = Final[cols]
Final.to_csv("RESULT.csv", index=False)




