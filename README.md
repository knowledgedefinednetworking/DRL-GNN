# Deep Reinforcement Learning meets Graph Neural Networks: exploring a routing optimization use case
#### Link to paper: [[here](https://arxiv.org/abs/1910.07421)]
#### P. Almasan, J. Suárez-Varela, A. Badia-Sampera, K. Rusek, P. Barlet-Ros, A. Cabellos-Aparicio.
 
Contact: <felician.paul.almasan@upc.edu>

[![Twitter Follow](https://img.shields.io/twitter/follow/PaulAlmasan?style=social)](https://twitter.com/PaulAlmasan)
[![GitHub watchers](https://img.shields.io/github/watchers/knowledgedefinednetworking/DRL-GNN?style=social&label=Watch)](https://github.com/knowledgedefinednetworking/DRL-GNN)
[![GitHub forks](https://img.shields.io/github/forks/knowledgedefinednetworking/DRL-GNN?style=social&label=Fork)](https://github.com/knowledgedefinednetworking/DRL-GNN)
[![GitHub stars](https://img.shields.io/github/stars/knowledgedefinednetworking/DRL-GNN?style=social&label=Star)](https://github.com/knowledgedefinednetworking/DRL-GNN)

## Abstract
Recent advances in Deep Reinforcement Learning (DRL) have shown a significant improvement in decision-making problems. The networking community has started to investigate how DRL can provide a new breed of solutions to relevant optimization problems, such as routing. However, most of the state-of-the-art DRL-based networking techniques fail to generalize, this means that they can only operate over network topologies seen during training, but not over new topologies. The reason behind this important limitation is that existing DRL networking solutions use standard neural networks (e.g., fully connected), which are unable to learn graph-structured information. In this paper we propose to use Graph Neural Networks (GNN) in combination with DRL. GNN have been recently proposed to model graphs, and our novel DRL+GNN architecture is able to learn, operate and generalize over arbitrary network topologies. To showcase its generalization capabilities, we evaluate it on an Optical Transport Network (OTN) scenario, where the agent needs to allocate traffic demands efficiently. Our results show that our DRL+GNN agent is able to achieve outstanding performance in topologies unseen during training.  

# Instructions to execute

[See the execution instructions](https://github.com/knowledgedefinednetworking/DRL-GNN/blob/master/DQN/README.md)

## Description

To know more details about the implementation used in the experiments contact: [felician.paul.almasan@upc.edu](mailto:felician.paul.almasan@upc.edu)

Please cite the corresponding article if you use the code from this repository:

```
@article{almasan2019deep,
  title={Deep reinforcement learning meets graph neural networks: Exploring a routing optimization use case},
  author={Almasan, Paul and Su{\'a}rez-Varela, Jos{\'e} and Badia-Sampera, Arnau and Rusek, Krzysztof and Barlet-Ros, Pere and Cabellos-Aparicio, Albert},
  journal={arXiv preprint arXiv:1910.07421},
  year={2019}
}
```
