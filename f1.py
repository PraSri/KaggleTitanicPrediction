import pandas as pd
import numpy as np
import re
from sklearn.tree import DecisionTreeClassifier
testFile = pd.read_csv('all/test.csv')
trainFile = pd.read_csv('all/train.csv')
all_data = [testFile, trainFile]

# fearure 1: Pclass
f1 = trainFile[["Pclass", "Survived"]].groupby(["Pclass"], as_index = False).mean()

# feature 2: Sex
f2 = trainFile[["Sex", "Survived"]].groupby(["Sex"], as_index = False).mean()

# feature 3: Family size
for data in all_data:
  data['family_size'] = data['SibSp'] + data['Parch'] + 1
f3 = trainFile[["family_size", "Survived"]].groupby(["family_size"], as_index = False).mean()

# feature 3.1: is_alone
for data in all_data:
  data['is_alone'] = 0
data.loc[data['family_size'] == 1, 'is_alone'] = 1
f31 = trainFile[["is_alone", "Survived"]].groupby(["is_alone"], as_index = False).mean()

# feature 4: embarked
f4 = trainFile[["Embarked", "Survived"]].groupby(["Embarked"], as_index = False).mean()

# feature 5: Fare
for data in all_data:
  data['Fare'] = data['Fare'].fillna(data['Fare'].median())
trainFile['category_Fare'] = pd.qcut(trainFile['Fare'], 4)
f5 = trainFile[["category_Fare", "Survived"]].groupby(["category_Fare"], as_index = False).mean()

# feature 6: Age
for data in all_data:
  avg_age = data['Age'].mean()
  std_age = data['Age'].std()
  null_age = data['Age'].isnull().sum()
random_list = np.random.randint(avg_age - std_age, avg_age + std_age, size = null_age)
data['Age'][np.isnan(data['Age'])] = random_list
data['Age'] = data['Age'].astype(int)
trainFile['category_age'] = pd.cut(trainFile['Age'], 5)
f6 = trainFile[["category_age", "Survived"]].groupby(["category_age"], as_index = False).mean()

# feature 7: Name
def get_title(name):
	title_search = re.search('([A-Za-z]+)\.', name)
	if title_search:
		return title_search.group(1)	
	return ""

for data in all_data:
  data['title'] = data['Name'].apply(get_title)
for data in all_data:
  data['title'] = data['title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
  data['title'] = data['title'].replace('Mlle', 'Miss')
  data['title'] = data['title'].replace('Ms', 'Miss')
  data['title'] = data['title'].replace('Mme', 'Mrs')

# crosstab title and Sex table by crosstab function of pandas as title will be rows and sex will be column
cross = pd.crosstab(trainFile['title'], trainFile['Sex'])

f7 = trainFile[["title", "Survived"]].groupby(["title"], as_index = False).mean()

# print(f1)
# print(f2)
# print(f3)
# print(f31)
# print(f4)
# print(f5)
# print(f6)
# print(cross)
# print(f7)
# print(all_data)

# mapping data

#Mapping Sex

for data in all_data:
  sex_map = {
    'female': 0,
    'male': 1
  }
  data['Sex'] = data['Sex'].map(sex_map).astype(int)

  # Mapping title
  title_map = {
  'Mr': 1,
  'Miss': 2,
  'Mrs': 3,
  'Master': 4,
  'Rare': 5
  }
  data['title'] = data['title'].map(title_map).astype(int)
  data['title'] = data['title'].fillna(0).astype(int)
  # Mapping Embarked
  # embark_map = {'S': 0, 'C': 1, 'Q': 2}
  # data['Embarked'] = data['Embarked'].map(embark_map).astype(int)
  data.loc[data['Embarked']=='S','Embarked'] = 0
  data.loc[data['Embarked']=='C','Embarked'] = 1 
  data.loc[data['Embarked']=='Q','Embarked'] = 2
  # Mapping Fare
  data.loc[data['Fare'] <= 7.91, 'Fare'] = 0
  data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare'] = 1
  data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare'] = 2
  data.loc[data['Fare'] > 31, 'Fare'] = 3
  data['Fare'] = data['Fare'].astype(int)
  # Mapping Age
  data.loc[data['Age'] <= 16, 'Age'] = 0
  data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1
  data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2
  data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3
  data.loc[data['Age'] > 64, 'Age'] = 4

drop_elements = ["Name", "Ticket", "Cabin", "SibSp", "Parch", "family_size"]

trainFile = trainFile.drop(drop_elements, axis = 1)
trainFile = trainFile.drop(['PassengerId', 'category_Fare', 'category_age'], axis = 1)
testFile = testFile.drop(drop_elements, axis = 1)

# print(trainFile.head(10))

X_train = trainFile.drop("Survived",axis=1).fillna(0)
Y_train = trainFile["Survived"].fillna(0)
X_test = testFile.drop("PassengerId",axis=1).copy().fillna(0)

# print(X_train)
# print(Y_train)

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train , Y_train)
Y_pred = decision_tree.predict(X_test)
accuracy = round(decision_tree.score(X_train,Y_train)*100,2)
# print("Modal accuracy" , accuracy)

submission = pd.DataFrame({"PassengerId":testFile["PassengerId"],"Survived":Y_pred})
submission.to_csv('submission.csv',index = False)





