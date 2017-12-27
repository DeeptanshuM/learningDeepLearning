# What I've learnt (in reverse chronological order):

## Course 2: Improving Deep Neural Networks
- [Implemented 1-dimensional and N-dimensional gradient checking](Gradient+Checking+v1.ipynb)

- [Modelled 3-layer NN without regularization, with L2 regularization and with dropout](Regularization.ipynb)

- [Modelled 3-layer NN without zero initializations, with random initializations and with He initializations](Initialization.ipynb)

## Course 1: Neural Networks and Deep Learning
- [Built and applied a deep neural network to supervised learning](Deep+Neural+Network+-+Application+v3.ipynb)

- [implemented all the functions required for building a deep neural network](Building+your+Deep+Neural+Network+-+Step+by+Step+v5.ipynb)

- [Built and trained a 2-class classification neural network with a single hidden layer, and implemented forward propagation and backpropagation](Planar+data+classification+with+one+hidden+layer+v4.ipynb)

- [Built a logistic regression classifier to recognize cats](Logistic+Regression+with+a+Neural+Network+mindset+v4.ipynb)

- [Practised Numpy basics (math functions, broadcasting and vectorization)](Python+Basics+With+Numpy+v3.ipynb)

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
