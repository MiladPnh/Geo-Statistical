import sklearn.feature_selection
import sklearn.decomposition
import sklearn.metrics
import sklearn.linear_model
import sklearn.ensemble
import sklearn.neighbors
import sklearn.tree
import sklearn.svm
from ML import types
from ML import dataTransformation
from ML import modeling
from ML import utilities

# Paths
pathToData = 'ExampleData/myData.csv'

# Read data sets
myData = types.DataSet('My Training Data',
                                   pathToData)
splitData = dataTransformation.splitDataSet(myData,
                                                        testProportion=0.3,
                                                        randomSeed=89271)
trainingData = splitData.trainDataSet
testingData = splitData.testDataSet

# Tune models for training data set
tuneScoringMethod = 'r2'
rfParameters = [{'n_estimators': [50, 75, 100]}]
rfMethod = types.ModellingMethod('Random Forest',
                                             sklearn.ensemble.RandomForestRegressor)
rfConfig = types.TuneModelConfiguration('Tune Random Forest',
                                                    rfMethod,
                                                    rfParameters,
                                                    tuneScoringMethod)
knnParameters = [{'n_neighbors': [2, 5]}]
knnMethod = types.ModellingMethod('K Nearest Neighbors',
                                              sklearn.neighbors.KNeighborsRegressor)
knnConfig = types.TuneModelConfiguration('Tune KNN',
                                                     knnMethod,
                                                     knnParameters,
                                                     tuneScoringMethod)

predictorConfigs = [rfConfig, knnConfig]
tunedModelResults = modeling.tuneModels([trainingData],
                                                    predictorConfigs)

# Apply the tuned models to some test data
applyModelConfigs = []
for tunedModelResult in tunedModelResults:
    applyModelConfig = types.ApplyModelConfiguration(tunedModelResult.description,
                                                                 tunedModelResult.modellingMethod,
                                                                 tunedModelResult.parameters,
                                                                 trainingData,
                                                                 testingData)
    applyModelConfigs.append(applyModelConfig)
applyModelResults = modeling.applyModels(applyModelConfigs)

# Score the test results
r2Method = types.ModelScoreMethod('R Squared',
                                              sklearn.metrics.r2_score)
meanOEMethod = types.ModelScoreMethod('Mean O/E',
                                                  modeling.meanObservedExpectedScore)
testScoringMethods = [r2Method, meanOEMethod]
testScoreModelResults = modeling.scoreModels(applyModelResults,
                                                         testScoringMethods)

# Create a dataframe where each row is a different dataset-model combination
scoreModelResultsDF = utilities.createScoreDataFrame(testScoreModelResults)
print(scoreModelResultsDF)

# Visualization
utilities.barChart(scoreModelResultsDF,
                               'R Squared',
                               'R Squared for Each Model',
                               'ExampleData/rSquared.png',
                               '#2d974d')

utilities.scatterPlot(scoreModelResultsDF,
                                  'Mean O/E',
                                  'R Squared',
                                  'Mean O/E by R Squared for Each Model',
                                  'ExampleData/meanOEbyRSquared.png',
                                  '#2d974d')


# Some other things you can do

# Scale data
scaledTrainingData, scaler = dataTransformation.scaleDataSet(trainingData)
scaledTestingData = dataTransformation.scaleDataSetByScaler(testingData, scaler)

# Perform feature engineering
pcaConfig = types.FeatureEngineeringConfiguration('PCA n5',
                     'extraction', sklearn.decomposition.PCA, {'n_components': 5})

pcaTrainingData, transformer = dataTransformation.\
    engineerFeaturesForDataSet(trainingData, pcaConfig)
pcaTestingData = dataTransformation.engineerFeaturesByTransformer(
                                                         testingData, transformer)

# Create stacking ensemble
predictorConfigs = []
for tunedModelResult in tunedModelResults:
    predictorConfig = types.PredictorConfiguration(tunedModelResult.modellingMethod.description,
                                                     tunedModelResult.modellingMethod.function,
                                                     tunedModelResult.parameters)
    predictorConfigs.append(predictorConfig)

stackMethod = types.ModellingMethod('Stacking Ensemble',
                                                types.StackingEnsemble)
stackParameters = {'basePredictorConfigurations': predictorConfigs,
                   'stackingPredictorConfiguration': predictorConfigs[0]}
stackApplyModelConfig = types.ApplyModelConfiguration(
    'Stacking Ensemble',
    stackMethod,
    stackParameters,
    trainingData,
    testingData)

stackResult = modeling.applyModel(stackApplyModelConfig)

