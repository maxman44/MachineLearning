from sklearn.datasets import load_iris

#loads iris data for flowers, data set includes width and length of petals and sepals of many iris flowers
iris = load_iris()

numSamples, numFeatures = iris.data.shape
print(numSamples)
print(numFeatures)
print(list(iris.target_names))

from sklearn.model_selection import train_test_split

#Dividing the data by 20% reserved for testing and training with the remaining 80%. X is features (pedal size), y is labels (species)
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=0)

import xgboost as xgb

#convert data into what DMatrix expects. One for testing, one for training
train = xgb.DMatrix(X_train, label=y_train)
test = xgb.DMatrix(X_test, label=y_test)

#define hyperparameters. Softmax is best since this is a multiple classification problem
param = {
    'max_depth': 4,
    'eta': 0.3,
    'objective': 'multi:softmax',
    'num_class': 3}
epochs = 10

#single lines training execution to train model using these parameters as a first guess
model = xgb.train(param, train, epochs)

#
predictions = model.predict(test)
print(predictions)