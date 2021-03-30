# CNN Task 2

Once again you'll create a convolutional NN, just like in the last task. However, there is a catch. This time, the images being passed to the network will not always have the same resolution. You must design your network in such a way as to be able to accept images of any shape or resolution. 

### What are CNNS?
Convolutional neural networks (CNNs) are a type of classical machine learning model often used in computer vision and image processing applications. The structure of CNNs consists of applying alternating convolutional layers (plus an activation function) and pooling layers to an input array, typically followed by some fully connected layers before the output.

Convolutional layers work by sweeping across the input array and applying different filters (often 2x2 or 3x3 matrices) block by block. These are used to detect specific features of the image wherever they might appear. Pooling layers are then used to downsample the results of these convolutions to extract the most relevant features and reduce the size of the data, making it easier to process in subsequent layers. Common pooling methods involve replacing blocks of the data with their maximum or average values.

### Hints:
* Make the first few layers convolutional, then the last few into fully connected MLPs.
* When going from CNN to MLP, make sure to correctly calculate the number of input connections into the MLP layer.
* If you finished task CNN1, use that as a base and see how you can modify it to accept images of different resolutions.
* Feel free to use any and all resources. Even resources that have been shared with you in the past.


Only edit the sections between the `# <<< YOUR CODE HERE >>> #` comments in the `MyFlexibleCNN.py` file (but feel free to experiment).

To run and evaluate your network:
~~~
python main.py
~~~

Try to get an accuracy of above 90%.