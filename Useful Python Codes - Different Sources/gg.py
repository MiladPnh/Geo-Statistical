from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.model_selection import ParameterGrid, KFold

clf = GBR()

grid = {
    'min_samples_split': [2, 3],
    'max_depth': [3, 8, 15],
    'n_estimators': [10, 20, 50],
    'max_features': [0.8,'sqrt'],
    'subsample': [1, 0.7],
    'min_samples_leaf': [1, 3],
    'learning_rate': [0.01, 0.1]
}

sampler = ParameterGrid(grid)




for params in sampler:
    print(params)


