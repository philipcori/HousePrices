
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV


def main():

    train = pd.read_csv('C:/Users/phili/PycharmProjects/HousePrices/train.csv')
    test = pd.read_csv('C:/Users/phili/PycharmProjects/HousePrices/test.csv')

    print(train.groupby('YrSold').SalePrice.median())

    #print correlation values between features and SalePrice
    corr = train.corr()
    corr.sort_values(["SalePrice"], ascending=False, inplace=True)
    print(corr.SalePrice)

    #filter out outliers
    train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index)

    #notes on observed correlations:
    #SalePrice and: GrLivArea - linear, TotalBsmtSF - linear, OverallQual & YearBuilt - quad?

    #plot SalePrice vs GrLivArea
    fig, ax = plt.subplots()
    ax.scatter(x=train['GrLivArea'], y=train['SalePrice'])
    plt.ylabel('SalePrice', fontsize=13)
    plt.xlabel('GrLivArea', fontsize=13)

    # Correlation map to see how features are correlated with SalePrice
    corrmap = train.corr()
    plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmap, vmax=0.9, square=True)
    plt.show()


    for column in train:
        if(train[column].dtype == object):                              #if column is filled with strings, aka categories:
            train[column] = pd.factorize(train[column].values)[0]       #replace categories with corresponding integers values
        else:
            train[column] = train[column].fillna(train[column].mean())          #otherwise fill in missing values of columns with numerical values


    train['is2010'] = train['YrSold'] == 2010               #add 'is2010' feature because of decrease in median SalePrice during 2010


    X,Y = XYSplit(train)

    # trainX, testX, trainY, testY = train_test_split(X,Y,test_size=0.33,random_state=0)
    # rfr = RandomForestRegressor(n_estimators=1000,random_state=0).fit(trainX,np.log(trainY))
    # optimatal min_samples_split, min_samples_leaf, n_estimators for GBR: 20,1,360, rmsle = 0.124... DOESN'T WORK ON TEST DATA FROM KAGGLE

    print("training...")

    gbr = GradientBoostingRegressor(n_estimators=100).fit(X,Y)

    for column in test:
        if (test[column].dtype == object):
            test[column] = pd.factorize(test[column].values)[0]
        else:
            test[column] = test[column].fillna(test[column].mean())

    test['is2010'] = test['YrSold'] == 2010

    pred_gbr = (gbr.predict(test))

    # rmsle_rfr = rmsle(testY.values,pred_rfr)


    my_solution = pd.DataFrame(pred_gbr,test['Id'],columns=['SalePrice'])
    my_solution.to_csv("out6.csv", index_label=["Id"])


#returns RMSLE error
def rmsle(real,predicted):
    sum=0.0
    for i in range(len(predicted)):
        p = np.log(predicted[i])
        r = np.log(real[i])
        sum += (p - r)**2
    return (sum/len(predicted))**0.5

#splits a dataframe into features and feature to be predicted (SalePrice)
def XYSplit(train):
    X = train.copy()
    y = train.SalePrice
    del X['SalePrice']
    return X, y


if __name__ == '__main__':
        main()