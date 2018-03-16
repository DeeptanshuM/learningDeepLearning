# What I've learnt (in reverse chronological order):
## Course 5: Sequence Models
- [Implement a model that uses an LSTM to generate music and used it to generate music](Improvise+a+Jazz+Solo+with+an+LSTM+Network+-+v3.ipynb)
- [Built a character level language model to generate new names](Dinosaurus+Island+--+Character+level+language+model+final+-+v3.ipynb)
- [Implemented a recurrent neural network - step by step](Building+a+Recurrent+Neural+Network+-+Step+by+Step+-+v3.ipynb)

## Course 4: Convolutional Neural Networks
- [Built and trained a ConvNet in TensorFlow for a classification problem](Convolution+model+-+Application+-+v1.ipynb)
- [Implemented a Convolutional Neural Network using numpy](Convolution+model+-+Step+by+Step+-+v2.ipynb)

## Course 3: Structured Machine Learning Projects
- description coming soon :)

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

### Quiz notes: Bird recognition in the city of Peacetopia (case study)

- training and dev/test sets can come from different distributions but not dev and test sets
- in case of high avoidable bias and low variance 
⋅⋅⋅- train a bigger model to try to do better on the training set 
⋅⋅⋅- try decreasing regularization. 

### Error Analysis

Carrying out error analysis
- Look at mislabelled examples in dev set to evaluate ideas; evaluate multiple ideas in parallel

Cleaning up incorrectly labelled data
- DL algorithms are robust to random errors but less robust to systematically mislabelled data
- look at overall dev set error & error due to incorrect labels & error due to all other causes —> then decide which ones to fix
- remember to apply same process to dev and test sets to make sure that they come from the same distribution

Principles/guidelines:
- apply same process to dev and test sets to make sure they continue to to come from the same distribution
- consider examining examples the algorithm got right as well as the ones it got wrong
- train and dev/test data may now come from slightly different distributions

Build your first system quickly, then iterate
- set up dev/set and metric, build system quickly, use bias/variance analysis and error analysis to prioritize next steps

Mismatched training and dev/test data

Training and testing on different distributions
- **dev and test set should always be comprised of data from distribution that is being optimized for**

**Bias and variance with mismatched data distributions**
- make training-dev set: it comes from same distribution as training set
- don’t train on training-dev set
- use it to judge bias and variance in case of mismatched data distributions
- if high difference between training set error and training-dev set error and high difference between training error and dev error but training-dev error and dev error are similar: high variance
- if training error and training-dev error are both similar but both are a lot lower than dev error: this is a data mismatch problem as your algorithm has learnt to  do well on the wrong data
- if all training, training-dev and dev errors are similar and all higher than Bayes error: high bias
- if training error is a lot higher than Bayes error and training-dev error is similar to training error and dev error is a lot higher than both then: high avoidable bias, low variance and big data mismatch problem
- avoidable bias: difference between human level/Bayes error and training error
- variance: difference between training set and training-dev set errors
- data mismatch: difference between training-dev set and dev set errors
- degree of overfitting to dev set: difference between dev and test set errors

Addressing data mismatch
- carry out manual error analysis to try to understand difference between training adndev/test sets
- make training data more similar to dev/test set or collect more data similar to dev/test sets
- to make training data more similar to dev/test sets: artificial data synthesis

### Learning from multiple tasks

Transfer learning
- use NN trained for x task to perform some other y task (x: cat recognition task, y: radiology image prediction) (both tasks have the same input)
- when does transfer learning make sense — when you have a lot of data for the problem you are transferring from and relatively less data for the problem you are transferring to — if the vice versa is true then transfer learning doesn’t make sense

Multi-task learning
- enables NN to do multiple things at once (e.g., NN starts to try to detect car and person at the same time)
- unlike softmax regression which assigns a single label to a single example, one image can have multiple labels
- if among n tasks, the earlier layers of a NN perform the same function then it may be better to train one NN to perform n tasks rather than to train n NNs to perform n tasks
- training on a set of tasks that could benefit from having shared lower-level features
- also makes sense if data for multiple tasks is “quite similar”
- also makes sense it the NN is so big that it can in fact do well on multiple tasks
- transfer learning is used a lot more to do though

### End-to-end deep learning

When to use and when not to use end-to-end deep learning
- pros: “lets the data speak”, there’s less hand-designing of components needed
- cons: may need lots of data, excludes potentially useful hand-designing of components
- “key question: do you have sufficient data to learn the function of the complexity needed to map from X to Y”

### Quiz notes: Autonomous driving (case study)

- applied ML is a highly iterative process. If you train a basic model and carry out error analysis (see what mistakes it makes) it will help point you in more promising directions.
- deep learning algorithms are quite robust to having slightly different train and dev distributions.

***

## Course 4: Convolutional Neural Networks

### Padding:
- n x n image convolved (not sure if this is a word) with a f x f filter will result in a n - f +1 x n - f + 1 image —> shrinking outputs
- corner pixels aren’t used nearly as much as pixels in the center —> throwing away lots of information from corner of the image
-  so pad the image with 0’s
- p = number of pixels padded on each border
- output is now  n + 2p - f + 1 x n + 2p - f + 1
- valid convolution: no padding
- same convolution: output size (after convolution) is same as input size — p = (f - 1) / 2 — f is usually odd

### Strided convolution:
- if stride is s —> output is a square of dimension floor of [((n + 2*p -f) / s) + 1]

### Convolution on RGB images
- image and filter must have same number of channels (height * width * # channels)
- image n x n x nc; filter f x f x nc; output n - f + 1 x n - f + 1 x number of filters (also equal to number of features you are detecting)
- nc is number of channels; also called depth of the filter

### One layer of a Convolutional Network
- a 3 X 3 X 3 filter yields 27 parameters 
- one such filter will give 27 parameter + 1 bias terms
- ***Number of parameters in one layer***:if you have N filters of dimension n * n * n — then number of parameters will N * (( n * n * n) + 1) — ***this number of parameters is independent of the size of the input image***, ***this property make CNNs less prone to overfitting***
- height or width in layer l = floor(((height or width in layer l - 1 + 2*padding in layer l - filter in layer l) /  stride of layer l) + 1)
- number of channels in the output is just the number of filters in the layer
- each filter is: f X f X number of channels in previous layer
- activation = height * width * number of channels * number of training examples
- weights =  f X f X number of filters in layer l - 1 X number of filters in layer l
- bias = vector of dimension number of channels — convenient to represent it as (1, 1, 1, vector of dimension number of channels )

### Pooling layers
- no padding
- no parameters to learn
- only hyper parameters to be set— f and s — either by hand or cross validation 

### CNN example
- CONV — POOL — CONV — POOL — FC — FC — FC — SOFTMAX

### Why Convolutions?
- parameter sharing: a feature detector thats useful in one part of the image is probably useful in another part of the image 
- sparsity of connections: in each layer, each output value is dependent on only a small number of inputs

### Quiz notes
- Pooling layers do not have parameters, but they do affect the backpropagation (derivatives) calculation

### Classic Networks
- LeNet - 5
	- ![LeNetimg](images/LeNetimg.png?raw=true)
- AlexNet
	- about 60 million parameters vs LeNet’s 60,000
	- easy paper to read
	- ![AlexNetimg](images/AlexNetimg.png?raw=true)
- VGG-16
	- AlexNet has a complicated architecture and too many hyper parameters
	- reduce hyper parameters and have a much simpler network
	- focus on just having conv-layers that are just three-by-three filters with a stride of one and always use same padding and make max pooling layers 2-by-2 with a stride of two
	- still has about 138 million parameters
	- relative uniformity of this architecture made this attractive
	- ![VGG16img](images/VGG16img.png?raw=true)

### ResNets
- very deep NNs fail because of exploding and vanishing gradients
- skip connections allow taking activation from one layer and suddenly feeding it to a layer much deeper in the NN
- Residual blocks like the one in the picture below are stacked together to make very deep networks 
	- ![ResidualBlocks_img](images/ResidualBlocks_img.png?raw=true)
- 5 residual blocks stacked together:
	- ![five_ResidualBlocks_img](images/five_ResidualBlocks_img.png?raw=true)

### Why ResNets Work
- without adding residual blocks: in very deep plain networks learning even the identity function becomes difficult
- main reason ResNet works is that it’s so easy for these extra layers to learn the identity function 
- ![whyResNetswork_img](images/whyResNetswork_img.png?raw=true)
- ![ResNet_img](images/ResNet_img.png?raw=true)	

### Networks in Networks and 1x1 Convolutions
 
- add non-linearity to a network and allows you to change number of channels
- very useful in building the inception neural network
- ![1by1_img1](images/1by1_img1.png?raw=true)
- ![1by1_img2](images/1by1_img2.png?raw=true)

### Inception Network Motivation
- instead of choosing filter size for conv layer or making a choice between conv layer and a pooling layer — do them all
- ![inception_img](images/inception_img.png?raw=true)
- ![computationproblem_img](images/computationproblem_img.png?raw=true)
- ![computationproblemsolution_img](images/computationproblemsolution_img.png?raw=true)

***

## Course 5: Sequence Models

### Why sequence models
- sequence models are useful for mapping sequential inputs to sequential output

### Notation
- input: x and output: y
- element: x<index number> y<index number> 
- element: example, word being looked at at a certain time
- length of sequenceTx Ty
- tth element in the input of ith training example: x(i)<t>
- length of output sequence of ith training example: Tx(i)
- vocabulary: column vector of <some number> words based on which one hot encoded vectors are generated to represent input sentences

### Recurrent Neural Network Model
- Cons of standard network:
	- inputs and outputs can be different lengths in different examples
	- doesn’t share features learned across different positions of text
- RNNs don’t have either of these disadvantages
- RNNs use information from earlier (and later) in the sequence
- forward prop:
![rnn_notation_1](images/rnn_notation_1.png?raw=true)
-  simplified RNN Notation
￼![rnn_notation_2](images/rnn_notation_2.png?raw=true)

### Backpropagation through time
- ![backprop](images/backprop.png?raw=true)

### Different types of RNNs
- many to many (e.g. named entity recognition) , many to one (e.g. sentiment analysis), one to one (standard NN), one to many (e.g. music generation)
- ![types of RNNs](images/RNNTypes.png?raw=true)

### Language model and sequence generation
- language model —> finds P(sentence)
- training set: 1st tokenize sentence
- RNN model example:
![types of RNN models](images/egRNNModel.png?raw=true)

### Vanishing gradients with RNNs
- basic RNNs not good at capturing long-term dependencies
- if exploding gradients: apply gradient clipping

### Gated Recurrent Unit (GRU)
- hidden layer of RNN / RNN unit:
![RNNUnit](images/RNNUnit.png?raw=true)
- GRU unit:
![simplifiedGRUunit](images/simplifiedGRUunit.png?raw=true)
- gamma u (the gate) decides when the value of the memory cell is updated
- the purple curly braces equation in the picture above handles the vanishing gradient problem
- full gamma unit:
![fullGRU](images/fullGRU.png?raw=true)
- gamma r stands for relevance

### Long Short Term Memory (LSTM)
- more powerful and more general version of LSTM
- instead of just gamma u we have 2 gates- forget and output games
- * denotes element-wise multiplication
- ![LSTM](images/LSTM.png?raw=true)
- advantage of GRU: simpler, more scalable
- LSTM have been historically better

### Bidirectional RNN
- BRNN with LSTM — reasonable first thing to try to solve a NLP problem
- ![BRNN](images/BRNN.png?raw=true)

### Deep RNN
- for learning very complex functions it might be useful to stack many RNNs
- ![DeepRNN](images/DeepRNN.png?raw=true)
- even 3 layers are a lot because of temporal dimension

### Quiz: Recurrent Neural Networks
- When using a finished training a language model RNN to sample random sentences: (i) Use the probabilities output by the RNN to randomly sample a chosen word for that time-step as y^<t>. (ii) Then pass this selected word to the next time-step.
