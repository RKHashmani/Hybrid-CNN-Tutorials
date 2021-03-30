# CNN Task 1

Create a simple Convolutional Neural Network. This is similar to the MLP tasks, except this time, you'll initially be using convolutional layers and pooling layers along with activation functions. 

### What are CNNS?
Convolutional neural networks (CNNs) are a type of classical machine learning model often used in computer vision and image processing applications. The structure of CNNs consists of applying alternating convolutional layers (plus an activation function) and pooling layers to an input array, typically followed by some fully connected layers before the output.

Convolutional layers work by sweeping across the input array and applying different filters (often 2x2 or 3x3 matrices) block by block. These are used to detect specific features of the image wherever they might appear. Pooling layers are then used to downsample the results of these convolutions to extract the most relevant features and reduce the size of the data, making it easier to process in subsequent layers. Common pooling methods involve replacing blocks of the data with their maximum or average values.

### Hints:
* Make the first few layers convolutional, then the last few into fully connected MLPs.
* When going from CNN to MLP, make sure to correctly calculate the number of input connections into the MLP layer.
* Feel free to use any and all resources. Even resources that have been shared with you in the past.


Only edit the sections between the `# <<< YOUR CODE HERE >>> #` comments in the `MyCNN.py` file (but feel free to experiment).

To run and evaluate your network:
~~~
python main.py
~~~

Try to get an accuracy of above 90%.