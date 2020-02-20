# Deep Reinforcement Learning meets Graph Neural Networks: exploring a routing optimization use case
#### Link to paper: [[here](https://arxiv.org/abs/1910.07421)]
#### P. Almasan, J. Suárez-Varela, A. Badia-Sampera, K. Rusek, P. Barlet-Ros, A. Cabellos-Aparicio.
 
## Abstract
Recent advances in Deep Reinforcement Learning (DRL) have shown a significant improvement in decision-making problems. The networking community has started to investigate how DRL can provide a new breed of solutions to relevant optimization problems, such as routing. However, most of the state-of-the-art DRL-based networking techniques fail to generalize, this means that they can only operate over network topologies seen during training, but not over new topologies. The reason behind this important limitation is that existing DRL networking solutions use standard neural networks (e.g., fully connected), which are unable to learn graph-structured information. In this paper we propose to use Graph Neural Networks (GNN) in combination with DRL. GNN have been recently proposed to model graphs, and our novel DRL+GNN architecture is able to learn, operate and generalize over arbitrary network topologies. To showcase its generalization capabilities, we evaluate it on an Optical Transport Network (OTN) scenario, where the agent needs to allocate traffic demands efficiently. Our results show that our DRL+GNN agent is able to achieve outstanding performance in topologies unseen during training.  
 
## Description

To know more details about the implementation used in the experiments contact: [almasan@ac.upc.edu](mailto:almasan@ac.upc.edu)