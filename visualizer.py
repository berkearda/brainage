import matplotlib.pyplot as plt
import seaborn as sns

class DataVisualizer:
    def __init__(self):
        pass

    @staticmethod
    def plot_z_score_summary(z_summary):
        """
        Plot Z-score summary as a boxplot.
        :param z_summary: DataFrame with Z-score summary.
        """
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=z_summary)
        plt.title('Boxplot of Z-score Summary Statistics Across All Features')
        plt.ylabel('Metrics')
        plt.xlabel('Z-score')
        plt.show()