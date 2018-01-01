# What I've learnt (in reverse chronological order):

## Course 2: Improving Deep Neural Networks
- [Followed a tensorflow tutorial, and implemented a deep learning model using Tensorflow, to recognize numbers from 0 to 5 from the SIGNS dataset](Tensorflow+Tutorial.ipynb)

- [Implemented the following optimization algorithms: Mini-batch Gradient Descent, Mini-batch Momentum and Mini-batch Adam](Optimization+methods.ipynb)

- [Implemented 1-dimensional and N-dimensional gradient checking](Gradient+Checking+v1.ipynb)

- [Modelled 3-layer NN without regularization, with L2 regularization and with dropout](Regularization.ipynb)

- [Modelled 3-layer NN without zero initializations, with random initializations and with He initializations](Initialization.ipynb)

## Course 1: Neural Networks and Deep Learning
- [Built and applied a deep neural network to supervised learning](Deep+Neural+Network+-+Application+v3.ipynb)

- [implemented all the functions required for building a deep neural network](Building+your+Deep+Neural+Network+-+Step+by+Step+v5.ipynb)

- [Built and trained a 2-class classification neural network with a single hidden layer, and implemented forward propagation and backpropagation](Planar+data+classification+with+one+hidden+layer+v4.ipynb)

- [Built a logistic regression classifier to recognize cats](Logistic+Regression+with+a+Neural+Network+mindset+v4.ipynb)

- [Practised Numpy basics (math functions, broadcasting and vectorization)](Python+Basics+With+Numpy+v3.ipynb)

***
***

# Notes

## Course 2: Improving Deep Neural Networks

### Setting up your Machine Learning Application

<u>Train/Test/Dev sets</u>
- dev set goals - “hold out cross validation set” - test different algorithms and figure out which works better
- test set goals - given the final classifier figure out how how well it’s doing
- make sure that the dev and test sets come from the same distribution
- not having a test set might be okay

<u>Bias/Variance</u>
- high bias ( —> underfitting) —> high and similar errors with train set and test set
- high variance ( —> overfitting) —> low error with train set and high error with test set
- high bias and high variance —> high error with train set and significantly higher error with test set
- low bias and low variance —> low errors with train and test sets

<i>Note: the error comparisons mentioned above are with respect to the optimal (Bayes) error mentioned</i>
           
<u>Basic recipe for machine learning</u>
Higher bias
- train longer
- bigger network
- find better neural network architecture

High variance
- More data
- Regularization
- find better neural network architecture

### Regularizing NN

Regularization -- add picture of notes

Why does regularization reduce overfitting?
- L~2~ regularization introduces term in cost equation that penalizes weight matrices for being too large
- overfitting —> If lambda is large then W is small then Z is small then tanh is roughly linear then evey layer computes something roughly linear and so it reduces overfitting  

Dropout regularization
- go through each and eliminate some nodes randomly from that layer —> you end with a smaller, diminished network; repeat this for every training example
- Inverted dropout is the most popular implementation

Why does drop out work?
- It's like L~2~ regularization, but more adaptive
- Intuiation : can’t rely on any one feature, so have to spread out weights —> shrinks weights —> e.g., useful in computer vision because there are so many input features overfitting is very common and dropout is very useful here
- downside cost function is no longer well defined —> may not go down with every iteration of gradient descent

data auigmentation hack: to get more data for computer vision, just modify existing images: flip horizontally, randomly rotate and zoom etc.

Early stopping
- stop iterating when dev set error is minimum
- downside: this is against the practice of orthogonalization
- orthogonalization: focus on one task at a time—> first optimize cost function (gradient descent etc…) —> second do not over fit (regularization etc…)
- Early stopping couples the first and second tasks

### Setting Up an Optimization Problem:

Normalize Inputs —> why? —> because then all features will be on the same scale —> cost function will be more “round” —> it’ll be easier to optimize 

Vanishing/exploding gradients —> careful choice of initializing weights partially solves this problem

Weight initializations 
- multiply randomly initialized weights by variance 
- for Relu variance is sqrt(2 / number of neuron in input layer) 
- for tanh it’s sqrt(1 / number of neuron in input layer) )

Checking derivative equation: use triangle of width of 2 epsilon

Gradient checking for a neural vector —> [insert picture] —> only use to debug, doesn't work with dropout

### Practical aspects of deep learning quiz notes:

1. If 10 millions examples —> tran/dev/test split should be 98/1/1%
2. Dev and test set should come from same distribution 
3. If NN seems to have have high bias: increase num of units in each hidden layer and make NN deeper
4. If low error with train set but high error with dev set —> increase regularization parameter lambda and get more training data
5. Weight decay is a regularization technique (such as L2 regularization) that results in gradient descent shrinking weight on every iteration
6. When regularization hyperparameter lambda is increased —> weights are pushed toward becoming smaller toward 0
7. With inverted dropout technique, at test time: dp not apply dropout (do not randomly eliminate units) and do not keep the 1/keep_prob factor in the calculations used in training
8. Increasing the parameter keep_prob from (say) .5 to .6 will likely cause: reduction of the regularization effect and case the neural network to end up with a lower training test set error
9. Techniques useful for reducing variance/overfitting —> data augmentation, L2 regularization, Dropout
10. Normalize the inputs x to make the cost function faster to optimize. 

### Optimization Algorithms

Mini-batch gradient descent
- split training sets into “mini-batches” —> process one mini-batch at a time, at one step of gradient descent
- unlike in batch gradient step where one pass over training data leads to one step of gradient descent, with mini-batch gradient descent one pass over training data leads to many number of gradient descent steps
- cost may not decrease on every iteration of mini-batch gradient descent
- Batch gradient descent —> too slow
- Stochastic gradient descent —> loose speedup from vectorization
- If less than 2000 training examples --> use batch gradient descent
- else make sure to choose a power of 2 for size of mini-batch and make sure that mini-batch can fit in cpu/gpu memory

Exponentially weighted averages
- lowering beta makes algorithm more susceptible to noise and vice versa
- bias correction will help get a better estimate early on
- allows algorithm to take a more straightforward path to the minimum

Gradient descent with momentum, RMS prop, Adam optimization algorithm

Learning rate decay
- slowly reduce learning rate over time —> as learning approaches convergence then having a slower learning rate allows you to take smaller steps 

1 epoch ==>1 pass through the data

The problem of local optima
- most points of zero gradient are saddle points —> much more likely to run into saddle points than local optima in high dimensional spaces 
- plateaus are a problem as they can slow down learning —> algorithms like gradient descent with momentum, RMS prop, Adam optimization algorithm help 

### Optimization Algorithms Quiz notes

- If batch gradient descent in a deep NN is taking excessively long then try using Adam, tuning the learning rate, better random initializations for the weights and mini-batch gradient-descent.
- for gradient descent	—> increasing momentum —> reduced oscillation 

### Hyperparameter Tuning

Tuning process
- have to with learning rate (alpha), beta, beta1, beta2, epsilon, # layer, # hidden units, learning rate decay, mini-batch size
- most important —> alpha
- 2nd most important —> beta (usually .9), # hidden units, mini-batch size
- 3rd most important —> # layers, learning rate decay
- beta1 is usually .9, beta2 is usually .999 and epsilon is usually 1e-8
- these are not fixed rules :)
- Coarse to fine search

Using an appropriate scale to pick hyperparameters  
- sometimes use log scale 
- for exponentially weighted averages: recall that <i> using beta = .9 is like taking averages of last 10 values whereas using beta = .999 is like averaging over last 1000 values  </i> —> doesn’t make sense to sample on a linear scale —> using log scale here will help sample more densely

Hyperparameter Tuning in practice
- re-evaluate occasionally 
- Panda approach—> if contrainued resources (e.g. less cpus/gpus but lots of data) —> maybe babysit one model
- Caviar approach —> train many models in parallel

### Batch Normalization (more effective than hyperparameter tuning)

Normalizing inputs to speed up learning- normalize values of z (maybe a) of a hidden layer to train next layer’s W or b faster

Fitting Batch Norm in a NN 
- done after z calculation, before a calculation, in every layer
- in practices, batch norm is applied on a mini-batch
- if batch is used then the original bias parameter can be eliminated as batch norm will zero out the mean anyway

Why does Batch Norm Work?
- makes neurons in a deeper layer, say layer 10, more robust to changes than a neuron in an earlier layer, say layer 1
- ensure that no matter how the values of z in a hidden layer shift, their mean and variance remains the same —> this limits the amount to which updating the parameters in an earlier layer changes the values that a deeper layer sees and has to learn on 
- batch norm reduces the problem of input values changing —> causes these values to become more stable —> later layers of the NN have “more firm ground to stand on” —> weakens the coupling between what an earlier layer does and what a later layer does —> allows each layer to work more independently of other layers —> therefore, speeds up learning in the NN
- each mini-batch is scaled by the mean/variance computed on just that mini-batch — this adds some noise (multiplicative noise by multiply by standard deviation and additive noise by subtracting the mean) to z values within a mini-batch — this has a slight regularization effect — not significant — noise can be reduced by making mini-batches bigger — don’t rely on/choose Batch Norm for regularization 

Batch Norm at test time 
- estimate mean and variance using exponentially weighted average across mini-batches seen during training

### Multi-class classification  — softmax…

***

## Course 3: Structured Machine Learning Projects

### Introduction to ML Strategy 

Orthogonalization —> 1st fit training set well —> only then fit dev set well —> only then fit test set well —> only then determine performance in real world

### Setting up the goal 

Single number evaluation metric 
-  have just one real number metric to tell if the new thing is better or worse than the last thing
- to judge classifier, instead of using precision and recall metrics, use one metric that combine precision and recall —> f1 score (harmonic mean of precision and recall)
- dev set + single number metric key to speeding up iterative process of improving the machine learning algorithm

Satisficing and optimizing matrices
- introduce a condition to judge a metric —> the metric has to be just good enough to satisfy the condition —> therefore, called the satisficing metric
- have one metric that must be optimized; all other metrics can be satisficing metrics

Train/dev/test distributions
-  **dev and test sets must be from the same distribution**
- choose dev and test set to reflect data you expect to get in the future and consider important to do well on

Size of dev and test sets
- for big data —> 98/1/1
- **test set must be good enough to high confidence in the overall performance of your system**

When to change dev/test sets and metrics
- if evaluation metric is no longer rank ordering preferences between algorithm —> change evaluation metric or dev and test sets

<i> **Defining a metric is step 1 and doing well on it is step 2 (orthogonalization)** </i>

### Comparing to human-level performance

Why human-level performance 
- accuracy surpasses human-level performance but never Bayes optimal error
- if algorithm doesn’t perform as good humans —> get labeled data from human, gain insight from manual error analysis, better analysis of bias/variance

Avoidable bias
- if training error and dev error are greater than human error % than reduce bias else focus on reducing variance
- think of human-level error as a proxy for Bayes error
- avoidable bias (difference between human error % and training error) acknowledges that error can’t be less than Bayes error
- variance is a measure of the difference in performance on training and dev sets

Understanding human-level performance 
- human-level error should be a proxy for Bayes error —> so it should be as low as possible 

Surpassing human-level performance
- once human-level performance is surpassed —> don’t know whether to focus on bias or variance reduction tactics & can’t rely on human generated labels
- ML has significantly outperformed humans at problems that are not natural perception tasks

***Improving your model performance***
- 2 fundamental assumptions of supervised learning: 1.You can fit the training set pretty well (~ avoidable bias) & 2. The training set performance generalizes pretty well to the dev/test set (~ variance)
- look at difference between human-level and training error as an estimate for avoidable bias
⋅⋅⋅- Train bigger model
⋅⋅⋅- Train longer/better optimization algorithms (~ momentum, RMSprop, Adam)
⋅⋅⋅- NN architecture/hyperparameters search  (RNN, CNN etc)
- look at difference between training error and dev error as an estimate for variance
⋅⋅⋅- More data
⋅⋅⋅- Regularization (L2, dropout, data augmentation)	
⋅⋅⋅- NN architecture/hyperparameters search  (RNN, CNN etc)

### QUIZ notes: Bird recognition in the city of Peacetopia (case study)

- training and dev/test sets can come from different distributions but not dev and test sets
- in case of high avoidable bias and low variance 
⋅⋅⋅- train a bigger model to try to do better on the training set 
⋅⋅⋅- try decreasing regularization. 
