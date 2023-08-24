import matplotlib.pyplot as plt
import os


def plot_accuracies(accuracies_dict, ood_subset_for_hp, save_dir='./plots', file_format='png'):
    """
    Plot average accuracy across all test distributions versus `ood_subset_for_hp` for multiple baselines.

    Parameters:
        - accuracies_dict (dict): Dictionary where keys are names of the baselines and values are the corresponding list of accuracies.
        - ood_subset_for_hp (list): List of OOD subsets.
        - save_dir (str): Directory to save the plot.
        - file_format (str): Format to save the image, default is 'png'. Can be 'jpg' or 'png'.
    """

    plt.figure(figsize=(10, 6))

    # Loop over all baselines and plot them with different colors.
    for baseline, accuracies in accuracies_dict.items():
        plt.plot(ood_subset_for_hp, accuracies, '-o', label=baseline)

    plt.xlabel('Number of OOD Examples for Loss Learning')
    plt.ylabel('Average Test Accuracy')
    plt.title('Average Test Accuracy vs Number of OOD Examples for Loss Learning')
    plt.legend(loc='best')
    plt.grid(True)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, 'acc_vs_ood_subset_for_hp.' + file_format)
    plt.savefig(save_path)

    plt.show()

