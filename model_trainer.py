from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class ModelTrainer:
    def __init__(self, df_x, df_y):
        self.df_x = df_x
        self.df_y = df_y

    def train_and_evaluate(self):
        X_train, X_test, y_train, y_test = train_test_split(self.df_x, self.df_y, test_size=0.2)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Mean Squared Error
        mse = mean_squared_error(y_test, y_pred)
        print(f'Mean Squared Error: {mse}')
        
        # R2 Score
        r2 = r2_score(y_test, y_pred)
        print(f'R2 Score: {r2}')