# HousePrices

Project that predicts the sales prices of houses given a training dataset. Currently the program uses a GradientBoostingRegressor model from the scikit-learn library to predict sales prices, based on the numerous features (~80) given in the training dataset. Categorical features were factorized into integer values so as to be used in training the GradientBoostingRegressor model. The Groupby object of the pandas library, as well as the Matplotlib and seaborn libraries were used to visualize and make observations on the training dataset.

Certain portions of code are commented out. These were used in cross-validation methods to determine optimal parameters to the GradientBoostingRegressor model, using the rmsle function. Unfortunately these values did not prove optimal on the actual test data offered by kaggle.com, which is noted in the program.
