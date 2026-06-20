import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# This function returns arrays for generations,
# metric mean and standard deviation values aggregated over runs,
# given the run folders and metric name
def aggregate_metric_values_over_runs(run_folders, metric):
    # this will be filled with arrays of metric values for each experiment run
    metric_values_per_runs = []

    # Iterating over each of the experiment run folders within the experiment folder
    for run_folder in run_folders:
        # storing the metrics csv file as a data frame
        df = pd.read_csv(os.path.join(run_folder, "metrics.csv"))
        
        # adding array of metric values for this run to the list of arrays
        metric_values_per_runs.append(df[metric].to_numpy(dtype=float))

    # reshaping the list of arrays of metric values per run into a 2d array
    metric_values_per_runs = np.vstack(metric_values_per_runs)

    # returning arrays for generations and 
    # metric mean/standard deviation per generation 
    # aggregated over experiment runs
    return (
        df["generation"].to_numpy(), 
        np.mean(metric_values_per_runs, axis=0), 
        np.std(metric_values_per_runs, axis=0)
    )

# This function produces aggregated graphs of all the experiment runs given the experiment folder
def plot_aggregated_experiment_runs(experiment_folder):
    # storing all the run folders which are part of the experiment folder
    run_folders = sorted(glob.glob(os.path.join(experiment_folder, "run*")))

    ## FITNESS GRAPH ##
    plt.figure()
    ax = plt.gca()

    # Getting generations, elite fitness values averaged over runs, and standard deviation
    generations, mean, std = aggregate_metric_values_over_runs(run_folders, "elite_fitness")

    # Plotting the aggregated elite fitness values against generations
    ax.plot(generations, mean, label="elite fitness", marker="o", markersize=2, linewidth=1)

    # Lightly filling standard deviation area
    ax.fill_between(generations, mean - std, mean + std, alpha=0.1)

    # Getting generations, mean creature fitness values averaged over runs, and standard deviation
    generations, mean, std = aggregate_metric_values_over_runs(run_folders, "mean_fitness")
    # Plotting the aggregated mean creature fitness values against generations
    ax.plot(generations, mean, label="mean fitness", marker="o", markersize=2, linewidth=1)
    # Lightly filling standard deviation area
    ax.fill_between(generations, mean - std, mean + std, alpha=0.1)

    # setting graph title, labels, legend
    ax.set_title(f"Fitness")
    ax.set_xlabel("generation")
    ax.set_ylabel("fitness")
    ax.legend()
    ax.grid(True)

    ## MOVEMENT GRAPH ##
    # creating two subplots in the horizontal movement graph
    # 1: minimum radius from mountain center
    # 2: ratio of movement taking longer paths
    fig, axs = plt.subplots(2, 1, sharex=True)

    # Title for both subplots
    fig.suptitle(f"Horizontal Movement")

    # First subplot axis for minimum radius graph
    ax = axs[0]
    # Getting generations, elite creature minimum mountain radius aggregated over runs, and standard deviation
    generations, mean, std = aggregate_metric_values_over_runs(run_folders, "elite_min_radius")
    # Plotting the aggregated elite creature minimum mountain radius values against generations
    ax.plot(generations, mean, label="elite minimum mountain radius", marker="o", markersize=2, linewidth=1)
    # Lightly filling standard deviation area
    ax.fill_between(generations, mean - std, mean + std, alpha=0.1)

    # Getting generations, mean creatures minimum mountain radius aggregated over runs, and standard deviation
    generations, mean, std = aggregate_metric_values_over_runs(run_folders, "mean_min_radius")
    # Plotting the aggregated elite creature minimum mountain radius values against generations
    ax.plot(generations, mean, label="mean minimum mountain radius", marker="o", markersize=2, linewidth=1)
    # Lightly filling standard deviation area
    ax.fill_between(generations, mean - std, mean + std, alpha=0.1)

    # Setting radius label and legend
    ax.set_ylabel("mountain radius")
    ax.legend()
    ax.grid(True)

    # Second subplot for ratio of movement on longer paths
    ax = axs[1]
    # Getting generations, elite creatures longer paths ratios aggregated over runs, and standard deviation
    generations, mean, std = aggregate_metric_values_over_runs(run_folders, "elite_long_path_ratio")
    # Plotting the elite creature aggregated longer path movement ratios against generations
    ax.plot(generations, mean, label="elite long paths ratio", marker="o", markersize=2, linewidth=1)
    # Lightly filling standard deviation area
    ax.fill_between(generations, mean - std, mean + std, alpha=0.1)

    # Getting generations, mean creatures longer paths ratios aggregated over runs, and standard deviation
    generations, mean, std = aggregate_metric_values_over_runs(run_folders, "mean_long_path_ratio")
    # Plotting the mean creature aggregated longer path movement ratios against generations
    ax.plot(generations, mean, label="mean long paths ratio", marker="o", markersize=2, linewidth=1)
    # Lightly filling standard deviation area
    ax.fill_between(generations, mean - std, mean + std, alpha=0.1)

    # labels and legend for longer paths ratio
    ax.set_xlabel("generation")
    ax.set_ylabel("ratio of movement on longer paths")
    ax.legend()
    ax.grid(True)

    ## MOUNTAIN CLIMBING GRAPH ##
    # creating two subplots in the mountain climbing graph
    # 1: highest platform reached
    # 2: simulation time ratio with creature/mountain contact
    fig, axs = plt.subplots(2, 1, sharex=True)

    # Title for both subplots
    fig.suptitle(f"Mountain Climbing")

    # First subplot axis for highest platform reached
    ax = axs[0]

    # Getting generations, elite creatures highest platform heights aggregated over runs, and standard deviation
    generations, mean, std = aggregate_metric_values_over_runs(run_folders, "elite_highest_platform_reached")
    # Plotting the elite creature highest platform reached heights against generations
    ax.plot(generations, mean, label="elite highest platform", marker="o", markersize=2, linewidth=1)
    # Lightly filling standard deviation area
    ax.fill_between(generations, mean - std, mean + std, alpha=0.1)

    # Getting generations, mean creatures highest platform heights aggregated over runs, and standard deviation
    generations, mean, std = aggregate_metric_values_over_runs(run_folders, "mean_highest_platform_reached")
    # Plotting the mean creature highest platform reached heights against generations
    ax.plot(generations, mean, label="mean highest platform", marker="o", markersize=2, linewidth=1)
    # Lightly filling standard deviation area
    ax.fill_between(generations, mean - std, mean + std, alpha=0.1)

    # setting label and legend for highest platform height
    ax.set_ylabel("platform height")
    ax.legend()
    ax.grid(True)
    
    # Second subplot for mountain contact time ratio
    ax = axs[1]
    # Getting generations, elite creatures mountain contact time ratios aggregated over runs, and standard deviation
    generations, mean, std = aggregate_metric_values_over_runs(run_folders, "elite_mountain_collision_ratio")
    # Plotting the elite creature mountain contact ratio against generations
    ax.plot(generations, mean, label="elite mountain contact ratio", marker="o", markersize=2, linewidth=1)
    # Lightly filling standard deviation area
    ax.fill_between(generations, mean - std, mean + std, alpha=0.1)

    # Getting generations, mean creatures mountain contact time ratios aggregated over runs, and standard deviation
    generations, mean, std = aggregate_metric_values_over_runs(run_folders, "mean_mountain_contact_ratio")
    # Plotting the mean creature mountain contact ratio against generations
    ax.plot(generations, mean, label="mean mountain contact ratio", marker="o", markersize=2, linewidth=1)
    # Lightly filling standard deviation area
    ax.fill_between(generations, mean - std, mean + std, alpha=0.1)

    # labels and legend for mountain contact ratio
    ax.set_xlabel("generation")
    ax.set_ylabel("mountain contact ratio")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True)
    

    ## STABILITY GRAPH ##
    # creating three subplots in the stability graph
    # 1: flying and approaching walls
    # 2: motor effort (total and useless)
    # 3: number of links
    fig, axs = plt.subplots(3, 1, sharex=True)

    # Title for all stability subplots
    fig.suptitle(f"Stability")

    # Axis for flying and wall approaching subplot
    ax = axs[0]
    # Getting generations, elite creatures flying ratios aggregated over runs, and standard deviation
    generations, mean, std = aggregate_metric_values_over_runs(run_folders, "elite_flying_ratio")
    # Plotting the elite creatures flying ratio against generations
    ax.plot(generations, mean, label="elite flying ratio", marker="o", markersize=2, linewidth=1)
    # Lightly filling standard deviation area
    ax.fill_between(generations, mean - std, mean + std, alpha=0.1)

    # Getting generations, mean creatures flying ratios aggregated over runs, and standard deviation
    generations, mean, std = aggregate_metric_values_over_runs(run_folders, "mean_flying_ratio")
    # Plotting the mean creatures flying ratio against generations
    ax.plot(generations, mean, label="mean flying ratio", marker="o", markersize=2, linewidth=1)
    # Lightly filling standard deviation area
    ax.fill_between(generations, mean - std, mean + std, alpha=0.1)

    # Getting generations, elite creatures wall proximity ratios aggregated over runs, and standard deviation
    generations, mean, std = aggregate_metric_values_over_runs(run_folders, "elite_wall_proximity_ratio")
    # Plotting the elite creatures wall proximity ratio against generations
    ax.plot(generations, mean, label="elite wall ratio", marker="o", markersize=2, linewidth=1)
    # Lightly filling standard deviation area
    ax.fill_between(generations, mean - std, mean + std, alpha=0.1)

    # Getting generations, mean creatures wall proximity ratios aggregated over runs, and standard deviation
    generations, mean, std = aggregate_metric_values_over_runs(run_folders, "mean_wall_proximity_ratio")
    # Plotting the mean creatures wall proximity ratio against generations
    ax.plot(generations, mean, label="mean wall ratio", marker="o", markersize=2, linewidth=1)
    # Lightly filling standard deviation area
    ax.fill_between(generations, mean - std, mean + std, alpha=0.1)

    # labels and legend for wall proximity/flying ratio
    ax.set_ylabel("wall/flying ratio")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True)

    # Axis for motor effort subplot
    ax = axs[1]

    # Getting generations, elite creatures total motors effort aggregated over runs, and standard deviation
    generations, mean, std = aggregate_metric_values_over_runs(run_folders, "elite_total_motor_effort")
    # Plotting the elite creatures total motor effort against generations
    ax.plot(generations, mean, label="elite motor effort", marker="o", markersize=2, linewidth=1)
    # Lightly filling standard deviation area
    ax.fill_between(generations, mean - std, mean + std, alpha=0.1)

    # Getting generations, mean creatures total motors effort aggregated over runs, and standard deviation
    generations, mean, std = aggregate_metric_values_over_runs(run_folders, "mean_total_motor_effort")
    # Plotting the mean creatures total motor effort against generations
    ax.plot(generations, mean, label="mean motor effort", marker="o", markersize=2, linewidth=1)
    # Lightly filling standard deviation area
    ax.fill_between(generations, mean - std, mean + std, alpha=0.1)

    # Getting generations, elite creatures useless motors effort aggregated over runs, and standard deviation
    generations, mean, std = aggregate_metric_values_over_runs(run_folders, "elite_useless_motor_effort")
    # Plotting the elite creatures useless motor effort against generations
    ax.plot(generations, mean, label="elite useless motor effort", marker="o", markersize=2, linewidth=1)
    # Lightly filling standard deviation area
    ax.fill_between(generations, mean - std, mean + std, alpha=0.1)

    # Getting generations, mean creatures useless motors effort aggregated over runs, and standard deviation
    generations, mean, std = aggregate_metric_values_over_runs(run_folders, "mean_useless_motor_effort")
    # Plotting the mean creatures useless motor effort against generations
    ax.plot(generations, mean, label="mean useless motor effort", marker="o", markersize=2, linewidth=1)
    # Lightly filling standard deviation area
    ax.fill_between(generations, mean - std, mean + std, alpha=0.1)

    # labels and legend for motor effort
    ax.set_ylabel("motor effort")
    ax.legend()
    ax.grid(True)

    # Axis for links subplot
    ax = axs[2]
    # Getting generations, elite creatures links aggregated over runs, and standard deviation
    generations, mean, std = aggregate_metric_values_over_runs(run_folders, "elite_links")
    # Plotting the elite creatures links number against generations
    ax.plot(generations, mean, label="elite links", marker="o", markersize=2, linewidth=1)
    # Lightly filling standard deviation area
    ax.fill_between(generations, mean - std, mean + std, alpha=0.1)

    # Getting generations, mean creatures links aggregated over runs, and standard deviation
    generations, mean, std = aggregate_metric_values_over_runs(run_folders, "mean_links")
    # Plotting the mean creatures links number against generations
    ax.plot(generations, mean, label="mean links", marker="o", markersize=2, linewidth=1)
    # Lightly filling standard deviation area
    ax.fill_between(generations, mean - std, mean + std, alpha=0.1)

    # labels and links legend
    ax.set_xlabel("generation")
    ax.set_ylabel("links")
    ax.legend()
    ax.grid(True)
    
    plt.show()

if __name__ == "__main__":
    import sys
    assert len(sys.argv) == 2, "Usage: python graph_experiment_results.py result_logs/<experiment_folder>"

    # Passing the experiment folder name from the terminal to the plotting function
    # This function produces aggregated graphs of all the experiment runs
    plot_aggregated_experiment_runs(sys.argv[1])


