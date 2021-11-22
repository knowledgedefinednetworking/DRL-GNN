import argparse
import os
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

if __name__ == "__main__":
    # python parse.py -d ./Logs/expDQN_agentLogs.txt
    parser = argparse.ArgumentParser(description='Parse file and create plots')

    parser.add_argument('-d', help='data file', type=str, required=True, nargs='+')
    args = parser.parse_args()

    aux = args.d[0].split(".")
    aux = aux[1].split("exp")
    differentiation_str = str(aux[1].split("Logs")[0])

    list_score_test = []
    epsilon_decay = []
    list_losses = []

    if not os.path.exists("./Images"):
        os.makedirs("./Images")

    with open(args.d[0]) as fp:
        for line in fp:
            arrayLine = line.split(",")
            if arrayLine[0]==">":
                list_score_test.append(float(arrayLine[1]))
            elif arrayLine[0]=="-":
                epsilon_decay.append(float(arrayLine[1]))
            elif arrayLine[0]==".":
                list_losses.append(float(arrayLine[1]))

    plt.plot(list_score_test, label="Score")
    plt.xlabel("Episodes")
    plt.title("GNN+DQN Testing score")
    plt.ylabel("Average Score Test")
    plt.legend(loc="lower right")
    plt.savefig("./Images/AvgTestScore" + differentiation_str)
    plt.close()

    # Plot epsilon evolution
    plt.plot(epsilon_decay)
    plt.xlabel("Episodes")
    plt.ylabel("Epsilon value")
    plt.savefig("./Images/Epsilon" + differentiation_str)
    plt.close()

    # Plot Loss evolution
    ysmoothed = savgol_filter(list_losses, 51, 3)
    plt.plot(list_losses, color='lightblue')
    plt.plot(ysmoothed)
    plt.xlabel("Batch")
    plt.title("Average loss per batch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.savefig("./Images/AvgLosses" + differentiation_str)
    plt.close()