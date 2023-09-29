## Build AI Based model to Detect Stress HVR Data

in here we Explore Data Usage Features Extraction and Ai modeling using **Pytorch || Tensorflow**

* Resources Papers

    * [Heart Rate Variability in Psychology: A Review of HRV Indices and an Analysis Tutorial](https://www.mdpi.com/1424-8220/21/12/3998)
    * [A multi-module case-based biofeedback system for stress treatment](https://sci-hub.se/https://doi.org/10.1016/j.artmed.2010.09.003)
    * [Stress Detection by Machine Learning and Wearable Sensors](https://dl.acm.org/doi/abs/10.1145/3397482.3450732)


### Per-Processing Stage

we used WESAD device wchih contain most features decribe the physoiology of subject , our goal is to extract the HRV features that corrspond to status of subject_ID **Stress** OR **Non-Stress** 

* Output File Extraction :

    the total numbers of Class Stress and Non-Stress 
    Number of Samples Per Class

    | Class         | Count |
    |---------------|-------|
    | stress        | 880    |
    | non-stress    | 385   |
    | total_Window_len | 118   |
    

* [x] Task: balancing the Dataset over class using **Oversampling Techniques**

    **algo used** : SMOTE-Tomek Links:

    **Approach**: SMOTE-Tomek Links combines SMOTE with Tomek links, which are pairs of instances (one from the majority class and one from the minority class) that are each other's nearest neighbors and have different class labels. SMOTE-Tomek removes these pairs, which are considered ambiguous.

    **Use Cases**: It's useful when we want to balance the dataset and remove ambiguous samples that might confuse the Model

* [x] Task: best selected Features **using best Selection features to improce the linear model**



* Split Data into Frequnecy Domaine and Time Domaine:

    After we Extrat the feature HRV we will now Split into Two DataSet to use an different model AI 

    **Notation**: Check Folder Features Extraction [Here](src/featureTestTraining/binary/Features) and [Notabook](src/featureTestTraining/binary/Split_Data_Features.ipynb)

    - **List Features Frequnecy || Time || Non-Linear Domain** :

        | Feature   Spatial | Feature   Frequency | Feature  Non-Linear |
        | ----------------- | ------------------- | ------------------- |
        | HR_mean           | LF                  | SD1                 |
        | HR_std            | HF                  | SD2                 |
        | meanNN            | ULF                 | pA                  |
        | SDNN              | VLF                 | pQ                  |
        | medianNN          | LFHF                | ApEn                |
        | meanSD            | total_power         | shanEn              |
        | SDSD              | lfp                 | D2                  |
        | RMSSD             | hfp                 | label               |
        | pNN20             | label               |
        | pNN50             |                     |
        | TINN              |                     |
        | label             |                     |

## Modeling AI Stage

**Notation**:***all the model was test on data **SWELL** models not train on or seen before for each different domaine Features HRV***


* Split Data into Frequnecy Domaine and Time Domaine:

After we Extrat the feature HRV we will now Split into Two DataSet to use an different model AI 

### Frequnecy model **[Linear Model-notebook](src/models/Linear-Model-with-Best-Selection-Features.ipynb)** : 

here we will use type of model based on Fully Connected Layer   which will conatin Learnbel Matix that leanr from **Frequnecy Domain Features** the benifit is Fast Training .

```python 

import torch
import torch.nn as nn
import torch.fft

class LinearStress(nn.Module):
    def __init__(self):
        super(LinearStress, self).__init__()
        self.fc = nn.Sequential(
                        nn.Linear(11, 128),
                        nn.Dropout(0.5),
                        nn.ReLU(),
                        nn.Linear(128, 2),
                        nn.Dropout(0.5),
                        nn.LogSoftmax(dim=1))
        
    def forward(self, x):
        return self.fc(x)   

```

* Results StressLinear :

    |    Tester | Accuracy | Precision | Recall | F1    |
    |--------|----------|-----------|--------|-------|
    | Tester | 0.9446   | 0.9437    | 0.9437 | 0.9437|


* Model Complexity Analysis:

    1. LinearStress Model

    | Property                                       | Value                          |
    |-----------------------------------------------|--------------------------------|
    | Total Parameters                               | 156                            |
    | Total MACs (Multiply-Accumulate Operations)    | 167.0                          |


    2. The LinearStress model consists of a Sequential block, which is composed of the following layers:

    | Layer                 | Parameters                  | MACs                           | Input Features | Output Features | Bias  |
    |-----------------------|-----------------------------|--------------------------------|----------------|-----------------|-------|
    | Linear Layer (Layer 0)| 132 (84.615% of total)       | 132.0 (79.042% of total)       | 11             | 11              | True  |
    | ReLU Activation (Layer 1)| No additional parameters | 11.0 (6.587% of total)        | -              | -               | -     |
    | Linear Layer (Layer 2)| 24 (15.385% of total)        | 24.0 (14.371% of total)        | 11             | 2               | True  |
    | LogSoftmax Layer (Layer 3)| No additional parameters | No MACs used (dim=1)          | -              | -               | -     |

**Computational Complexity:**

- Total MACs: 167.0

**Number of Parameters:**

- Total Parameters: 156


### Spatial model **[Stress-Attention-Notebook](src/models/TimeDomaineLearnRepresentationLinear-Attention.ipynb)**:

here we will use type of model based on Hybrid-Linear wihtin Attention which will conatin Attention mechanism that learn from **Spatial Domain Features HRV**.

```python
class MultiLayerSelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_layers=1):
        super(MultiLayerSelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(True),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_layers)
        ])
        self.attentions = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Softmax(dim=1)
            ) for _ in range(num_layers)
        ])

    def forward(self, encoder_outputs):
        energy = encoder_outputs
        for i in range(self.num_layers):
            energy = self.projections[i](energy)
            attention_scores = self.attentions[i](energy)
            context = attention_scores * encoder_outputs
            context = torch.sum(context, dim=1)
            encoder_outputs = context  
        return context

```


* Results StressAttention:

    | Tester | Accuracy | Precision | Recall | F1    |
    |--------|----------|-----------|--------|-------|
    | Tester | 0.9385   | 0.8927    | 0.9937 | 0.9405|


* Model Complexity Analysis:

| Property                                       | Value                          |
|-----------------------------------------------|--------------------------------|
| Total Parameters                               | 1.17k                          |
| Total MACs (Multiply-Accumulate Operations)    | 1.25 KMac                      |

| Layer                 | Parameters                  | MACs                           | Input Features | Output Features | Bias  |
|-----------------------|-----------------------------|--------------------------------|----------------|-----------------|-------|
| Linear Layer (Layer 0)| 132 (11.244% of total)       | 132.0 (10.568% of total)       | 11             | 11              | True  |
| ReLU Activation (Layer 1)| No additional parameters | 11.0 (0.881% of total)        | -              | -               | -     |
| Linear Layer (Layer 2)| 132 (11.244% of total)       | 132.0 (10.568% of total)       | 11             | 11              | True  |
| Softmax Activation (Layer 3)| No additional parameters | No MACs used (dim=1)          | -              | -               | -     |
| Linear Layer (Layer 4)| 264 (22.487% of total)       | 275.0 (22.018% of total)       | 11             | 11              | True  |
| Linear Layer (Layer 5)| 12 (1.022% of total)         | 12.0 (0.961% of total)        | 11             | 1               | True  |
| Linear Layer (Layer 6)| 768 (65.417% of total)       | 768.0 (61.489% of total)       | 11             | 64              | True  |
| ReLU Activation (Layer 7)| No additional parameters | 64.0 (5.124% of total)        | -              | -               | -     |
| Linear Layer (Layer 8)| 130 (11.073% of total)       | 130.0 (10.408% of total)       | 64             | 2               | True  |

**Computational Complexity:**
- Total MACs: 1.25 KMac

**Number of Parameters:**
- Total Parameters: 1.17k



**NOTE**: ***we will use **LTSM Attention** at the Task of detecting Stress in Arasoual Stage ,*** 



### Machine Learning Models : 

in here i implemented a collections of ML models at once include all the Algorithms which will picked up one :

* Decision Tree Classifier
* Random Forest Classifier
* AdaBoost Classifier
* K-Nearest Neighbors (KNN) Classifier
* Linear Discriminant Analysis (LDA)
* Support Vector Machine (SVM) Classifier
* Gradient Boosting Classifier

**Results**: all the results save in the [file](src/FeatureTestTraining/binary/results/Metric_Value.csv)

|   Model   |  Subject  | Metric_Value |
|-----------|-----------|--------------|
|   DT_AUC  | Subject_2 |   0.952091   |
|   DT_F1   | Subject_2 |   0.975610   |
|   DT_ACC  | Subject_2 |   0.963636   |
|   RF_AUC  | Subject_2 |   0.928571   |
|   RF_F1   | Subject_2 |   0.976190   |
|   ...     |   ...     |     ...      |


### Addtionaly Infos :

definitions of the units commonly used to measure computational complexity:

* Parameters:

    Definition: Parameters in a model represent the learnable components that the model uses to make predictions. They are the weights and biases in neural networks. The total number of parameters indicates the model's capacity to capture patterns in data.
    Importance: More parameters can lead to a more complex model that can fit the training data better, but it may also increase the risk of overfitting if not managed properly.

* MACs (Multiply-Accumulate Operations):

    Definition: MACs represent the number of arithmetic operations (multiplications and additions) performed by a model during inference. It is a measure of the computational workload required for one forward pass through the model.
    Importance: MACs give an estimate of the computational cost of running a model, which is crucial for assessing its efficiency and suitability for deployment on different hardware platforms.

* FLOPs" (Floating-Point Operations per Second):

    Definition: FLOPs measure the rate at which a computer or processor can perform floating-point arithmetic operations. It is often used to gauge the computational power of hardware.
    Importance: FLOPs are important for understanding the speed and efficiency of hardware devices, especially when dealing with large-scale computations in deep learning.

