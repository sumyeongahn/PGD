## PGD: Per-sample gradient Debiasing (ICLR'23)

[OpenReview](https://openreview.net/forum?id=7mgUec-7GMv)  
Sumyeiong Ahn*, Seongyoon Kim*, Se-Young Yun  (KAIST AI)



### Abstract
The performance of deep neural networks is strongly influenced by the training dataset setup. In particular, when attributes having a strong correlation with the target attribute are present, the trained model can provide unintended prejudgments and show significant inference errors (i.e., the dataset bias problem). Various methods have been proposed to mitigate dataset bias, and their emphasis is on weakly correlated samples, called bias-conflicting samples. These methods are based on explicit bias labels provided by human. However, such methods require human costs. Recently, several studies have tried to reduce human intervention by utilizing the output space values of neural networks, such as feature space, logits, loss, or accuracy. However, these output space values may be insufficient for the model to understand the bias attributes well. In this study, we propose a debiasing algorithm leveraging gradient called PGD (Per-sample Gradient-based Debiasing). PGD comprises three steps: (1) training a model on uniform batch sampling, (2) setting the importance of each sample in proportion to the norm of the sample gradient, and (3) training the model using importance-batch sampling, whose probability is obtained in step (2). Compared with existing baselines for various datasets, the proposed method showed state-of-the-art accuracy for the classification task. Furthermore, we describe theoretical understandings of how PGD can mitigate dataset bias. 


### Run command 

Step 1) Constructing dataset
~~~
cd datagen
python generator.py --data colored_mnist --bias_ratio 0.005,0.01,0.05
~~~

Step 2) Training
~~~
python train.py --reproduce --bratio 0.005 --dataset colored_mnist --train --exp run0 
~~~
