import os
import matplotlib.pyplot as plt
#%matplotlib inline
import pandas as pd
os.chdir('/home/mark/Desktop/ml4meremortals-master/chapter2')

# %% Read in data

data = pd.read_csv('OnlineNewsPopularitySample.csv', header = None,
                   names =
                   ['url', 'title_length', 'article_length', 'keywords', 'shares'],
                   index_col = 0)
print(f'Length of data is {len(data)}')

# %%  Plot it

fig = plt.figure(1)
plt.xlabel('Title (words)')
plt.ylabel('Shares')
plt.scatter(data.title_length, data.shares, color='r', s=5.0)
fig = plt.figure(2)
plt.xlabel('Article (words)')
plt.ylabel('Shares')
plt.scatter(data.article_length, data.shares, color='r', s=5.0)
fig = plt.figure(3)
plt.xlabel('Keywords')
plt.ylabel('Shares')
plt.scatter(data.keywords, data.shares, color='r', s=5.0)

# %% linear regression - single feature

from sklearn.linear_model import LinearRegression

model = LinearRegression()
target = data.shares
predictors = data[['article_length']]
model.fit(predictors, target)
print(f'Weight is {model.coef_}')
print(f'Bias is {model.intercept_}')

fig = plt.figure()
plt.xlabel('Article (words)')
plt.ylabel('Shares')
plt.scatter(data.article_length, data.shares, color='r', s=5.0)
plt.plot(data.article_length, 
         data.article_length * model.coef_[0]
         + model.intercept_)
# %%  linear regression - multiple features

model = LinearRegression()
predictors = data.drop('shares', axis = 'columns')
model.fit(predictors, target)
print(f'Weights are{model.coef_}')
print(f'Bias is {model.intercept_}')

fig = plt.figure()
plt.xlabel('Article (words)')
plt.ylabel('Shares')
plt.scatter(data.article_length, data.shares, color='r', s=5.0)
plt.plot(data.article_length, 
         (data.title_length * model.coef_[0]
         + data.article_length * model.coef_[1]
         + data.keywords * model.coef_[2]
         + model.intercept_))

# %%  non-linear regression - multiple features

# different data
data = pd.read_csv('OnlineNewsPopularityNonLinear.csv', header = None,
                   names =
                   ['url', 'title_length', 'article_length', 'keywords', 'shares'],
                   index_col = 0)
print(f'Length of data is {len(data)}')

model = LinearRegression()

target = data.shares
predictors = data.drop('shares', axis = 'columns')
model.fit(predictors, target)
print(f'Weights are{model.coef_}')
print(f'Bias is {model.intercept_}')

fig = plt.figure()
plt.xlabel('Article (words)')
plt.ylabel('Shares')
plt.scatter(data.article_length, data.shares, color='r', s=5.0)
plt.plot(data.article_length, 
         (data.title_length * model.coef_[0]
         + data.article_length * model.coef_[1]
         + data.keywords * model.coef_[2]
         + model.intercept_))
# or let the model do it!
#plt.plot(data.article_length, model.predict(predictors))

# %%  K-Nearest Neighbors for Classification - described, not used

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=7, weights='distance')

# %%  Linear Classification

from sklearn.linear_model import SGDClassifier

os.chdir('/home/mark/Desktop/ml4meremortals-master/chapter3')
traindata = pd.read_csv('OnlineNewsPopularityClassification.csv',
                      header=None, index_col=0, skiprows=1)
testdata = pd.read_csv('OnlineNewsPopularityClassification_test.csv',
                      header=None, index_col=0, skiprows=1)
unseendata = pd.read_csv('OnlineNewsPopularityClassification_newsamples.csv',
                      header=None, index_col=0, skiprows=1)

# train the model
Y=traindata.iloc[:,-1]
X=traindata.iloc[:,:-1]

classifier = SGDClassifier(max_iter=2000)
classifier.fit(X,Y)

# test the model
testX=testdata.iloc[:,:-1]
testY=testdata.iloc[:,-1]
# evaluate

print(f'Accuracy against the test data: {classifier.score(testX, testY):.1%}')

# predict
unseenX=unseendata.iloc[:,:-1]

predictedY = classifier.predict(unseenX)
print(f'Predictions: {predictedY}')