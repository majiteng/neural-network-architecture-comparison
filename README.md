# neural-network-architecture-comparison
Code and dataset for the paper "Machine Learning-Aided Microwave Circuit Design: Challenges and Prospects"


Two circuit analysis cases were compared to showcase the abilities of various supervised ML models. The first case was to predict the scattering parameters of a three-port grid-like microwave coupler comprising 35 identical cells in the design space. The second case required predicting the scattering parameters of a microwave branchline coupler, with annotations for the width and length of each transmission line. For each task, 30,000 random structures were created and labeled.

There are four Jupyter Notebook files corresponding to four different cases: "Case1_MLP", "Case1_ResNet", "Case2_MLP", and "Case2_GNN". In Case 1, raw data is saved in file "three_port_coupler". In Case 2, raw data is saved in "DoubleBox_Coupler". In the "Case2_GNN" notebook, the dataset in "DoubleBox_Coupler" needs to be handled in graph format, which is done by the file "preprocessing_graph_dataloader.ipynb". The processed graph data is then saved into the file "DoubleBox_Branchline_data".

To make a fair comparison, the hyperparameters, including the learning rate and the number of neurons in each layer, are fine-tuned using the NN intelligence (NNI) hyperparameter tuning tools (https://github.com/microsoft/nni).

The corresponding paper has been submitted to an IEEE magazine and is currently under peer review. The full manuscript will be available once it has been published.

Abstract:

The progress of future wireless communication systems heavily relies on microwave technologies. However, as higher frequency bands are standardized for wireless communications and transistor density increases exponentially as predicted by Moore's Law, the complexity and cost associated with microwave circuit design rise significantly. Engineers strive to depart from the conventional laborious design procedures and explore more efficient and precise methods of microwave circuit design. Machine learning (ML) has emerged as a promising technique to assist us. Many studies have shown that ML can improve wireless access by managing spectrum with dynamic spectrum access (DSA), predicting and optimizing quality of service (QoS), and detecting anomalies for network security. From the physical layer, ML can assist device-level optimization and facilitate microwave circuit design. This article provides an overview of the enabling technologies of ML-aided microwave circuit design for communication purposes. We articulate the critical need for model training in order to effectively deploy ML in microwave circuit design and investigate innovative techniques to accelerate ML training.
