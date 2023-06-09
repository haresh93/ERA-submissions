# MNIST-CNN

The CNN architecture has been trained on the MNIST Dataset for recognizing Hand written Digits, and the architecture has attained an accuracy of **99.45%** with **19,574** Params and less than **20 epochs**.

After Every Convolution Layer there is a ReLU Activation Layer followed by Batch Normalization and a dropout layer with probability of 0.05. 

Batch Normalization is used to batch normalize the inputs after every layer.
Dropout Layer is used to remove overfitting and introduce a regularization into the network.

The CNN architecture has the following layers:

### Convolution layer 1 
#### Convolution is applied with 8 kernels with size of 3 x 3 to extract initial features
 - No. of Kernels - 8
 - Padding - 1
 - Input Size - 28 x 28 x 1 
 - Output Size - 28 x 28 x 8 
 - Receptive Field - 3 x 3
  
### Convolution Layer 2 
#### Convolution is applied with 16 kernels with size of 3 x 3 taking the receptive field to 5 x 5 which extracts the edges
  - No. of Kernels - 16
  - Input Size - 28 x 28 x 8
  - Output Size - 26 x 26 x 16
  - Receptive Field - 5 x 5

### Max Pooling Layer
#### Max pooling layer reduces the size of the input from 26 x 26 to 13 x 13
  - Input Size - 26 x 26 x 16
  - Output Size - 13 x 13 x 16
  - Receptive Field - 10 x 10

### Convolution Layer 3
#### Convolution is applied with 8 kernels with size 1 x 1 to reduce the no. of output features making them richer 
  - No. of Kernels - 8
  - Input Size - 13 x 13 x 16
  - Output Size - 13 x 13 x 8
  - Receptive Field - 10 x 10

###Convolution Layer 4 
#### Convolution is applied with 16 kernels with size of 3 x 3 taking the receptive field to 12 x 12 which extracts the initial textures
  - No. of Kernels - 16
  - Input Size - 13 x 13 x 8
  - Output Size - 11 x 11 x 16
  - Receptive Field - 12 x 12

### Convolution Layer 5
  - No. of Kernels - 32
  - Input Size - 11 x 11 x 16
  - Output Size - 9 x 9 x 32
  - Receptive Field - 14 x 14

### Convolution Layer 6
  - No. of Kernels - 32
  - Input Size - 9 x 9 x 32
  - Output Size - 7 x 7 x 32
  - Receptive Field - 16 x 16

### Convolution Layer 7
#### Convolution is applied with size of 3 x 3 which extracts the parts of Object 
  - No. of Kernels - 10
  - Input Size - 7 x 7 x 32
  - Output Size - 5 x 5 x 10
  - Receptive Field - 18 x 18

### Average Pooling Layer 
#### Average Pooling converts the 5 x 5 channels to 1 x 1 but taking average of all the elements in the 5 x 5 matrix
  - Input Size - 5 x 5 x 10
  - Output Size - 1 x 1 x 10

### Log Soft Max Layer
#### The Output of the Average Pooling is sent to Log Softmax which gives likelihood of the input image among the 10 classes.