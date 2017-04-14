# bike sharing prediction

## usage

### setup

```sh
conda create --name dlnd python=3
```

### usage

```sh
source activate dlnd
pip install -r requirements.txt
jupyter notebook first_network.ipynb
```

## result

> Training loss: 0.063 ... Validation loss: 0.145

![my result](./prediction.png)

## network implementation

```python
#       forward             .        backprop                     _
#  in1 \                    .                                    /
#       \_ () \             .                             / () _/
#  in2  /      > () output  .   error () err_grad_output <      \_
#       \_ () /      ^      .                             \ () _/
#  in3  /       ^ final_out .                                   \
#            ^ final_in     .                                 ^  \_
#       ^ hidden_outputs    .                            err_grad_hidden
#       ^                   .    
#   hidden_inputs           .    
#
# () = node with f(h) = sigmoid activation function

class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        
        self.activation_function = lambda x : 1 / (1 + np.exp(-x))

    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            # X: (3,)
            ### Forward pass ###
            hidden_inputs = np.dot(X, self.weights_input_to_hidden)
            hidden_outputs = self.activation_function(hidden_inputs)

            final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
            final_outputs = final_inputs
            
            ### Backward pass ###
            error = y - final_outputs
            
            # error gradient output layer
            err_grad_output = error * 1 # f'(x) = 1
            
            # Weight step (hidden to output)
            #err_grad_o: (1,), Who: (2, 1), hidden_outputs: (2,)
            delta_weights_h_o += hidden_outputs[:,None] * err_grad_output # (2,1)
            
            # error gradient hidden layer
            #err_grad_o: (1,), Who: (2, 1), h_o: (2,)
            err_grad_hidden = np.matmul(err_grad_output, self.weights_hidden_to_output.T) * hidden_outputs * (1 - hidden_outputs) # err_grad_output is scalar, (1,2)
            
            # Weight step (input to hidden)
            #delta_weights_i_h: (3, 2), X: (3,) (X[:, None](3, 1)), err_grad_hidden: (1, 2)
            delta_weights_i_h += X[:,None] * err_grad_hidden
            
        # TODO: Update the weights - Replace these values with your calculations.
        self.weights_hidden_to_output += delta_weights_h_o * self.lr / n_records # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += delta_weights_i_h * self.lr / n_records # update input-to-hidden weights with gradient descent step
 
    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        
        #### Implement the forward pass here ####
        hidden_inputs = np.dot(features, self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
        final_outputs = final_inputs
        
        return final_outputs
```
