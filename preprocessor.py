import pandas as pd

class DataPreprocessor:
    def __init__(self, df_x, df_y):
        self.df_x = df_x
        self.df_y = df_y

    def handle_missing_values(self, data_type='x', strategy='mean'):
        df = self.df_x if data_type == 'x' else self.df_y
        if strategy == 'mean':
            df = df.fillna(df.mean())
        elif strategy == 'median':
            df = df.fillna(df.median())
        elif strategy == 'drop':
            df = df.dropna()
        else:
            raise ValueError(f"Unknown strategy {strategy}")

        if data_type == 'x':
            self.df_x = df
        else:
            self.df_y = df

        return df

    def get_z_score_summary(self, data_type='x'):
        df = self.df_x if data_type == 'x' else self.df_y
        z_scores = (df - df.mean()) / df.std()
        z_summary = pd.DataFrame({
            'min': z_scores.min(),
            'max': z_scores.max(),
            'mean': z_scores.mean(),
            'std': z_scores.std()
        })
        return z_summary

    def remove_outliers(self, threshold=3):
        z_scores = (self.df_x - self.df_x.mean()) / self.df_x.std()
        self.df_x = self.df_x[(z_scores < threshold).all(axis=1)]
        self.df_y = self.df_y.loc[self.df_x.index]