import numpy as np
import pandas as pd
from sklearn import tree

#used this for printing table and showing full columns
desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',10)

input_file = "/Users/maxkiehn/Desktop/maLearn/MLCourse/PastHires.csv"
df = pd.read_csv(input_file, header = 0)

#prints out 13 rows and columns from the above CSV with their assocaited values
print(df.head(13))

#Assigns a number to a value, so 1 = yes, 0 = no. map(d) will transform that data into 1s and 0s
d = {'Y': 1, 'N': 0}
df['Hired'] = df['Hired'].map(d)
df['Employed?'] = df['Employed?'].map(d)
df['Top-tier school'] = df['Top-tier school'].map(d)
df['Interned'] = df['Interned'].map(d)
d = {'BS': 0, 'MS': 1, 'PhD': 2}
df['Level of Education'] = df['Level of Education'].map(d)
print(df.head(13))

#prints all columns
features = list(df.columns[:6])
print(features)

#this will construct the data tree
y = df["Hired"]
X = df[features]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,y)

from IPython.display import Image
from six import StringIO
import pydotplus

dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=features)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())

#prints out png tree file
graph.write_png("treeGraph.png")

#Random forest of 10 decision trees to predict emplyment. There is a random element so end value may change
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(X, y)

#Predict employment of an employed 10-year veteran
print (clf.predict([[10, 1, 4, 0, 0, 0]]))
#...and an unemployed 10-year veteran
print (clf.predict([[10, 0, 4, 0, 0, 0]]))