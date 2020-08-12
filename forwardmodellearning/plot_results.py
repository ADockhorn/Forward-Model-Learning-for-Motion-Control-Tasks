import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import numpy as np


def store_data_into_csv():
    """ Store training results into CSV files to calculate a smoothed loess with R before plotting.
    """
    for ENV_NAME in ['CartPole-v1', 'Pendulum-v0', 'Acrobot-v1', 'LunarLander-v2', 'Swimmer-v2']:
        print("processing results of:", ENV_NAME)

        if ENV_NAME in {"CartPole-v1", "Acrobot-v1", "LunarLander-v2"}:
            model_names = ["DFM", "DQN", "SARSA", "CEM"]
        else:
            model_names = ["DFM", "NAF"]

        folders = os.listdir(f"results\\{ENV_NAME}")

        histories = {model: [] for model in model_names}
        for folder in folders:
            for model in model_names:
                if os.path.exists(f"results\\{ENV_NAME}\\{folder}\\training_results_{model}"):
                    try:
                        with open(f"results\\{ENV_NAME}\\{folder}\\training_results_{model}", "rb") as file:
                            data = pickle.load(file)
                            if isinstance(data, dict):
                                histories[model].append(data)
                    except Exception:
                        pass

        colors = sns.color_palette("muted", 8)
        sns.set(style="darkgrid")

        for i, model in enumerate(model_names):
            if len(histories[model]) == 0:
                continue
            scatter_data_x = []
            scatter_data_y = []
            end_performance = []
            for history in histories[model]:
                scatter_data_x.extend(history["nb_steps"])
                if model == "NAF":
                    scatter_data_y.extend([x*100 for x in history["episode_reward"]])
                    end_performance.extend([x*100 for x in history["episode_reward"][-10:]])
                else:
                    scatter_data_y.extend(history["episode_reward"])
                    end_performance.extend(history["episode_reward"][-10:])
            print(model, np.mean(end_performance))
            a = np.array([scatter_data_x, scatter_data_y]).transpose()
            np.savetxt(f"results\\csv\\{ENV_NAME}_{model}.csv", a, "%4.8f")


def plot_r_csv_files():
    """ Plot the csv files return by the R script which includes the loess smoothed values.
    """
    for ENV_NAME in ['CartPole-v1', 'Pendulum-v0', 'Acrobot-v1', 'LunarLander-v2', 'Swimmer-v2']:
        print("processing results of:", ENV_NAME)

        if ENV_NAME in {"CartPole-v1", "Acrobot-v1", "LunarLander-v2"}:
            model_names = ["DFM", "DQN", "SARSA", "CEM"]
        else:
            model_names = ["DFM", "NAF"]

        histories = {model: [] for model in model_names}
        for model in model_names:
            if os.path.exists(f"results\\out\\{ENV_NAME}_{model}.csv"):
                try:
                    data = np.genfromtxt(f"results\\out\\{ENV_NAME}_{model}.csv", delimiter=",", skip_header=1)
                    histories[model] = data
                except Exception:
                    pass

        colors = sns.color_palette("muted", 8)
        sns.set(style="darkgrid", font_scale=1.3)

        cols = 0
        for i, model in enumerate(model_names):
            if len(histories[model]) == 0:
                continue
            if model == "DDQN":
                continue
            cols += 1
            x, y, y1, y2 = histories[model][:, 1], histories[model][:, 2], \
                           histories[model][:, 2] - histories[model][:, 3], \
                           histories[model][:, 2] + histories[model][:, 3]
            if model == "DFM":
                model = "DDFM"
            plt.plot(x, y, c=colors[i], label=model)
            # plt.scatter(scatter_data_x, scatter_data_y)
            # plt.fill_between(x, y1, y2, color=colors[i], alpha=0.2)
        plt.legend()
        plt.xlabel("training steps")
        plt.ylabel("average episode return")
        plt.legend(bbox_to_anchor=(0.5, -0.20), loc=9, borderaxespad=0., ncol=cols, facecolor="white",
                   edgecolor="white")
        # plt.title(ENV_NAME)
        plt.tight_layout()
        plt.savefig(f"results\\{ENV_NAME}.png")
        plt.savefig(f"results\\{ENV_NAME}.pdf")
        plt.show()


if __name__ == "__main__":
    # values were smoothed using the loess function of r. please refer to our R-script "smoother.r"
    plot_r_csv_files()
    # store_data_into_csv()
