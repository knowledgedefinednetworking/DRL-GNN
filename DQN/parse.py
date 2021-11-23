import argparse
import os
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

if __name__ == "__main__":
    # python parse.py -d ./Logs/expsample_DQN_agentLogs.txt
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

    model_id = -1
    reward = 0
    with open(args.d[0]) as fp:
        for line in reversed(list(fp)):
            arrayLine = line.split(":")
            if arrayLine[0]=='MAX REWD':
                model_id = arrayLine[2].split(",")[0]
                reward = arrayLine[1].split(" ")[1]
                break
    
    print("Best model_id: "+model_id+" with Average Score Test of "+reward)

    plt.plot(list_score_test, label="Score")
    plt.xlabel("Episodes")
    plt.title("GNN+DQN Testing score")
    plt.ylabel("Average Score Test")
    plt.legend(loc="lower right")
    plt.savefig("./Images/AvgTestScore_" + differentiation_str)
    plt.close()

    # Plot epsilon evolution
    plt.plot(epsilon_decay)
    plt.xlabel("Episodes")
    plt.ylabel("Epsilon value")
    plt.savefig("./Images/Epsilon_" + differentiation_str)
    plt.close()

    # Plot Loss evolution
    ysmoothed = savgol_filter(list_losses, 51, 3)
    plt.plot(list_losses, color='lightblue')
    plt.plot(ysmoothed)
    plt.xlabel("Batch")
    plt.title("Average loss per batch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.savefig("./Images/AvgLosses_" + differentiation_str)
    plt.close()