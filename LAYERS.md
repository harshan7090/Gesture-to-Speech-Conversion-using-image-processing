LAYERS:-

Layer 1: Reshape layer
The first layer added is the reshape layer that took an input of (4096, 1) and reshaped it into (64, 64, 1).

Layer 2 - Layer 5: Convolution layer

(i)The function of convolution layer is to find out distinctive features from the given input matrix which is an image of sign which has been already normalized.
(ii)The convolution layer uses a filter matrix which is a combination of zeros and ones.
(iii)The filter matrix slides over the input matrix and keeps the count of elements which matched with the elements of the filter matrix. 
(iv)The matrix so created is called feature map. This convolution layer takes input matrix of shape of (64, 64,1 ). 
(v)This layer uses a filter of size (3, 3) and a stride size of 1. It creates 32 feature maps as output using 32 different filters. It uses ReLU as the activation function.


**Layer 6: Max pooling layer**

(i) Max pooling is done by applying a max filter to (usually) non-overlapping sub regions of the initial representation.
(ii) When the images are too large, it would need to reduce the number of trainable parameters. 
(iii) Pooling is done for the sole purpose of reducing the spatial size of the image. Here it have used a pool size of (2,2) with stride size 2.

Layer 7: Flatten layer

The function of the flatten layer is convert all elements of the feature maps matrices to individual neurons which will serve as input to the next layer.

Layer 8: Dense layer

(i) Dense or the input layer which accepts the output from the flatten layer as input. 
(ii) The value held by a neuron is called activation of that neuron. Every unit of input (neuron) has activation corresponding to intensity of pixel. 
(iii) The output of this layer is determined using the activation function which is ReLU in this case. 
(iv) The function of activation function is to activate the neurons of the dense layer. This layer has 128 units. 
(v) Every neuron has a bias associated with it. 
(vi) The neurons are activated based on the outcome from the activation function, if its activation is greater than 0.5 the neuron is activated else the neuron remains inactive.

Layer 9: Dropout layer

(i) The function of the dropout layer is to remove some of the neurons or the unwanted features that can make the model bulky and increase the training time. 
(ii) It is also helping in avoiding over fitting. A dropout layer which removes 50 percent of neurons was used here.

Layer 10: Dense layer

The last layer of the model is the dense layer which is also called the output layer. The last layer has 20 neurons corresponding to 20 different hand gestures.
