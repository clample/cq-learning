import matplotlib as mpl
import matplotlib.pyplot as plt

class Plot:

    def collisions_over_time_plot(self, results):
        plt.figure()
        for label, result in results.items():
            plt.plot(result.get_average_collisions(), label=label)

        plt.xlabel("Episode")
        plt.ylabel("No of collisions")
        plt.legend()
        plt.savefig(f"plots/{self.file_prefix}-collisions-over-time.png")
    
    def __init__(self, file_prefix):
        self.file_prefix = file_prefix
