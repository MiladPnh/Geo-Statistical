import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import style
style.use('fivethirtyeight')
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate
import scipy.stats as sps

# Load in the data
df = pd.read_csv('blood.csv', header=None)
#Random Sampling Incorporation
df = df.sample(n=100, random_state=1)
cols = [c for c in df.columns]
X = cols[:-1]
Y = cols[-1]

Tree_model = DecisionTreeClassifier(criterion="entropy", max_depth=2)
predictions = np.mean(cross_validate(Tree_model, X, Y, cv=100)['test_score'])
print('The accuracy is: ', predictions * 100, '%')


class Boosting:
    def __init__(self, dataset, T, test_dataset):
        self.dataset = dataset
        self.T = T
        self.test_dataset = test_dataset
        self.alphas = None
        self.models = None
        self.accuracy = []
        self.predictions = None

    def fit(self):
        # Set the descriptive features and the target feature
        X = self.dataset.drop(['target'], axis=1)
        Y = self.dataset['target'].where(self.dataset['target'] == 1, -1)
        # Initialize the weights of each sample with wi = 1/N and create a dataframe in which the evaluation is computed
        Evaluation = pd.DataFrame(Y.copy())
        Evaluation['weights'] = 1 / len(self.dataset)  # Set the initial weights w = 1/N

        # Run the boosting algorithm by creating T "weighted models"

        alphas = []
        models = []

        for t in range(self.T):
            # Train the Decision Stump(s)
            Tree_model = DecisionTreeClassifier(criterion="entropy",
                                                max_depth=1)

            model = Tree_model.fit(X, Y, sample_weight=np.array(Evaluation['weights']))

            # Append the single weak classifiers to a list which is later on used to make the
            # weighted decision
            models.append(model)
            predictions = model.predict(X)
            score = model.score(X, Y)
            # Add values to the Evaluation DataFrame
            Evaluation['predictions'] = predictions
            Evaluation['evaluation'] = np.where(Evaluation['predictions'] == Evaluation['target'], 1, 0)
            Evaluation['misclassified'] = np.where(Evaluation['predictions'] != Evaluation['target'], 1, 0)
            # Calculate the misclassification rate and accuracy
            accuracy = sum(Evaluation['evaluation']) / len(Evaluation['evaluation'])
            misclassification = sum(Evaluation['misclassified']) / len(Evaluation['misclassified'])
            # Caclulate the error
            err = np.sum(Evaluation['weights'] * Evaluation['misclassified']) / np.sum(Evaluation['weights'])

            # Calculate the alpha values
            alpha = np.log((1 - err) / err)
            alphas.append(alpha)
            # Update the weights wi --> These updated weights are used in the sample_weight parameter
            # for the training of the next decision stump.
            Evaluation['weights'] *= np.exp(alpha * Evaluation['misclassified'])
            # print('The Accuracy of the {0}. model is : '.format(t+1),accuracy*100,'%')
            # print('The missclassification rate is: ',misclassification*100,'%')

        self.alphas = alphas
        self.models = models

    def predict(self):
        X_test = self.test_dataset.drop(['target'], axis=1).reindex(range(len(self.test_dataset)))
        Y_test = self.test_dataset['target'].reindex(range(len(self.test_dataset))).where(self.dataset['target'] == 1,
                                                                                          -1)

        # With each model in the self.model list, make a prediction

        accuracy = []
        predictions = []

        for alpha, model in zip(self.alphas, self.models):
            prediction = alpha * model.predict(
                X_test)  # We use the predict method for the single decisiontreeclassifier models in the list
            predictions.append(prediction)
            self.accuracy.append(
                np.sum(np.sign(np.sum(np.array(predictions), axis=0)) == Y_test.values) / len(predictions[0]))
            # Goal: Create a list of accuracies which can be used to plot the accuracy against the number of base learners used for the model
            # 1. np.array(predictions) --> This is the array which contains the predictions of the single models.

            # 2. np.sum(np.array(predictions),axis=0) --> Summs up the first elements of the lists, that is 0,998+...+...+0.99. This is
            # done since the formula for the prediction wants us to sum up the predictions of all models for each instance in the dataset.

            # 3. np.sign(np.sum(np.array(predictions),axis=0)) --> Since our test target data are elements of {-1,1} and we want to
            # have our prediction in the same format, we use the sign function.

            # 4. np.sum(np.sign(np.sum(np.array(predictions),axis=0))==Y_test.values)/len(predictions[0]) --> With the last step we have transformed the array
            # into the shape 8124x1 where the instances are elements {-1,1}

            # 5. self.accuracy.append(np.sum(np.sign(np.sum(np.array(predictions),axis=0))==Y_test.values)/len(predictions[0])) -->
            # After we have computed the above steps, we add the result to the self.accuracy list. This list has the shape n x 1, that is,
            # for a model with 5 base learners this list has 5 entries where the 5th entry represents the accuracy of the model when all
            # 5 base learners are combined.

        self.predictions = np.sign(np.sum(np.array(predictions), axis=0))


######Plot the accuracy of the model against the number of stump-models used##########
number_of_base_learners = 50
fig = plt.figure(figsize=(10, 10))
ax0 = fig.add_subplot(111)
for i in range(number_of_base_learners):
    model = Boosting(df, i, df)
    model.fit()
    model.predict()
ax0.plot(range(len(model.accuracy)), model.accuracy, '-b')
ax0.set_xlabel('# models used for Boosting ')
ax0.set_ylabel('accuracy')
print('With a number of ', number_of_base_learners, 'base models we receive an accuracy of ', model.accuracy[-1] * 100,
      '%')

plt.show()


#SKlearn Adaboost

from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

for label in df.columns:
    df[label] = LabelEncoder().fit(df[label]).transform(df[label])

X = df.drop(['target'], axis=1)
Y = df['target']
# model = DecisionTreeClassifier(criterion='entropy',max_depth=1)
# AdaBoost = AdaBoostClassifier(base_estimator= model,n_estimators=400,learning_rate=1)
AdaBoost = AdaBoostClassifier(n_estimators=400, learning_rate=1, algorithm='SAMME')
AdaBoost.fit(X, Y)
prediction = AdaBoost.score(X, Y)
print('The accuracy is: ', prediction * 100, '%')






'''
#Results
From Scratch Code Vs SKlearn
With a number of  50 base models we receive an accuracy of  98.67 %                    Vs.      The accuracy is:  100.00 %   ==> #Blood.csv 
With a number of  50 base models we receive an accuracy of  98.72 %                    Vs.      The accuracy is:  96.68 %   ==> #Bank.csv 
With a number of  50 base models we receive an accuracy of  92.67 %                    Vs.      The accuracy is:  89.57 %   ==> #breast-cancer.csv 
With a number of  50 base models we receive an accuracy of  93.69 %                    Vs.      The accuracy is:  95.40 %   ==> #chess-krvk.csv 
With a number of  50 base models we receive an accuracy of  91.49 %                    Vs.      The accuracy is:  99.65 %   ==> #cylinder-bands.csv 
With a number of  50 base models we receive an accuracy of  90.52 %                    Vs.      The accuracy is:  86.01 %   ==> #credit-approval.csv 
With a number of  50 base models we receive an accuracy of  88.40 %                    Vs.      The accuracy is:  88.30 %   ==> #flags.csv 
With a number of  50 base models we receive an accuracy of  90.28 %                    Vs.      The accuracy is:  91.31 %   ==> #hepatitis.csv 
With a number of  50 base models we receive an accuracy of  97.23 %                    Vs.      The accuracy is:  99.00 %   ==> #ionosphere.csv 
'''