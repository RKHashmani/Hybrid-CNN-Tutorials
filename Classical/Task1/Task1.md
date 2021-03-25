# Task 1

Create a simple multilayer perceptron with 2 layers. The 1st layer should accept all the pixels of the image (`input_size`) and output `num_hidden_neurons` number of values. The output should then be passed through an activation function (for example, ReLU). Finally, then 2nd layer should take as input `num_hidden_neurons` and output `num_classes` values.

This will create a 2 layer network where the image is passed to the hidden (1st) layer, the outputs of which are passed to the output layer (2nd layer). The output of the 2nd layer tells us the probability of each picture being a certain number in the MNIST dataset. For example, if a image of 3 is passed, the output layer will output an array with 10 values, with the 3rd index's value being the largest.

Only edit the sections below the `# <<< YOUR CODE HERE >>> #` comments in the `MyMLP.py` file (but feel free to experiment).

To run and evaluate your network:
~~~
python main.py
~~~

You should get an accuracy of around 96%.