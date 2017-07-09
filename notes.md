# lessons learned for later lookup

## Multilayer Perceptron

### feature optimization

* try to sketch out significance of features ahead of training
* always make sure that feature gkoptimizations do not bias the network in one direction

### data visualization

* plot frequency of features to identify nodes that might not be necessary
* [t-distributed Stochastic Neighbor Embedding (TSNE)](http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) to visualize multi-dimensional (e.g. neural network weights)

### data preparation

* [sklearn/LabelBinarizer](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelBinarizer.html) for `string` or `int` labels
* [sklearn/OneHotEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) works for `int` labels

### embedding / word2vec

* [embedding](https://en.wikipedia.org/wiki/Word_embedding) are used to speed up hidden layer calculations
* for datasets with huge amount of classes (e.g. text ~50k+ words) instead of matrix multiplication with only a single _1_ in the input vector look up the weights directly with a word2vector index mechanism
* [word2vec](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/) is an embedding technique for text inputs
* w2v also adds semantic information to the vector representation by reproducing the semantic similarity on the vector level (similar words have similar vectors)
* CBOW (continous bag of words): surrounding words are used to define the one in the middle vs Skip-gram: middle word is used to predict surrounding words

## CNN

> Convolutional neural networks generate higher order features by applying kernels over one section of a matrix at a time. Stacking convolutions and multiple kernels increases the _feature depth_.

* https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/
  * premiered in 1990 with LeNet
  * excell at image classification/OCR
  * a convolution uses a filter/kernel/feature detector to find patterns in a part of the image
  * the kernel is [moved across the image](https://ujwlkarn.files.wordpress.com/2016/08/giphy.gif?w=748) where it always looks at a small part of it
  * going deep allows the net to find more complex features
  * the CNN learns which features to connect by adjusting weight and biases
* pooling
  * pooling is used to decrease the image size
  * usually used with a kernel stride of 1 (kernel does not reshape the image)
  * typical implementation is using max pooling
  * another patch is moved over the convoluted image and for a given area (pool kernel size) the maximum value is used for the output value
  * basically looking for local maxima in a given area
* network design
  * each convolutional layer should have an activation function
  * convolutions should decrease image size and increase feature depth
  * end with a few fully connected layers (traditional NN incl. activation fns)
  * convolutions detect higher order features (e.g. ears) and fully connected layer (multi layer perceptrons) create _combinations_ of those features for the final classification

### transfer-learning

* take a pre-learned CNN like vggnet, alexnet and add your own classifier
* super fast as you only need to train the classifier
* HowTo
  * cut off the classification of the existing network
  * run the dataset through the top part of the CNN to calculate activation `codes` (output of the convolutional layers)
  * create new N-dimensional NN >=2 fully connected layers
    * 1+ hidden layer
    * 1 output layer (N=number of classes)
  * train the classification part with the `codes` as input
  * plug the CNN and the classification together to get a model for production

## RNN

> Recurrent neural networks are able to adapt their output depending on the sequence in which inputs were given.

* LSTM cells prevent exploding💥/vanishing👻 gradients (would happen if you multiply weights across multiple executions)
* basic LSTMs have gates to **forget**, **update** and **output** the hidden state using combinations of _sigmoid_ and _tanh_
* depending on the task at hand individual LSTM cells might _solve tasks_ that can actually translate to the way humans solve things e.g. [keep track of how long a line of generated text is getting](https://youtu.be/iX5V1WpxxkY?t=27m23s)
* combining [CNN and RNN](https://youtu.be/iX5V1WpxxkY?t=31m24s) allows to do more advanced things like image captions

## GAN

> Generative adversarial networks (GANs) implement a system of two neural networks competing against each other in a zero-sum game framework. 
This technique can generate photographs and other data in an unsupervised fashion.

* learn probability distributions in given data
* use discriminator to detect _fake_ images and a generator to create them
* the discriminator is used twice during the overall training process
  * it gets fed with real images and a target of _real_
  * it gets trained together with the generator and a target of _fake_
  * to achieve this in tensorflow the discriminator is generated twice but variables are re-used (achieved by `tf.variable_scope(reuse=True)`)
* generator needs to create same output size as real images including value range (take care of `tanh` vs `sigmoid` etc)

## my mistakes
* **always** think twice about activation functions (logits shouldn't have one)
* input placeholders have an additional (first) `None` dimension to represent the batch size which is dynamic