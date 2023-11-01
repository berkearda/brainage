import pandas as pd
from preprocessor import DataPreprocessor
from feature_selector import FeatureSelector
from model_trainer import ModelTrainer
from visualizer import DataVisualizer

class MriWorkflow:
    def __init__(self, path_to_data_x, path_to_data_y):
        self.data_x = pd.read_csv(path_to_data_x)
        self.data_y = pd.read_csv(path_to_data_y)
        
        assert len(self.data_x) == len(self.data_y), "Data X and Data Y must have the same number of rows."
        
        self.processor = DataPreprocessor(self.data_x, self.data_y)
        self.selector = FeatureSelector(self.data_x, self.data_y)
        self.trainer = ModelTrainer(self.data_x, self.data_y)
        self.visualizer = DataVisualizer()

    def execute(self):
        # Handle missing values
        self.data_x = self.processor.handle_missing_values(data_type='x', strategy='mean')
        
        # Check for NaN values
        has_nan = self.data_x.isna().any().any()
        print(f"Contains NaN values after imputation: {has_nan}")

        # Print the head of the DataFrame to inspect the first few rows
        print("Head of the data safter imputing missing values:")
       # Print the first row of self.x_data
        print("First row of self.x_data:")

        # Drop rows where target values are missing
        missing_y_indices = self.data_y[self.data_y.isna().any(axis=1)].index
        self.data_x.drop(missing_y_indices, inplace=True)
        self.data_y.drop(missing_y_indices, inplace=True)
        print("After dropping rows with missing target values:", self.data_x.shape)  # Debugging line

        # Remove outliers
        self.processor.remove_outliers(threshold=3)
        print("After removing outliers:", self.data_x.shape)  # Debugging line
        
        # Feature selection
        self.selector.select_features(n_features=10)
        
        # Get and visualize z_score_summary
        z_summary_x = self.processor.get_z_score_summary(data_type='x')
        self.visualizer.plot_z_score_summary(z_summary_x)
        
        # Train and evaluate model
        self.trainer.train_and_evaluate()
