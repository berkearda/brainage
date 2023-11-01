from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

class FeatureSelector:
    def __init__(self, df_x, df_y):
        self.df_x = df_x
        self.df_y = df_y

    def select_features(self, n_features=10):
        # Remove or replace NaNs
        self.df_x = self.df_x.dropna()  # or self.df_x.fillna(self.df_x.mean())

        model = LinearRegression()
        rfe = RFE(estimator=model, n_features_to_select=n_features)
        fit = rfe.fit(self.df_x, self.df_y.loc[self.df_x.index])
        self.df_x = self.df_x.loc[:, fit.support_]