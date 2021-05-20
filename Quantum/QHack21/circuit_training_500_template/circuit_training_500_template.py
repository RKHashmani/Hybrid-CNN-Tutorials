#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np


def classify_data(X_train, Y_train, X_test):
    """Develop and train your very own variational quantum classifier.

    Use the provided training data to train your classifier. The code you write
    for this challenge should be completely contained within this function
    between the # QHACK # comment markers. The number of qubits, choice of
    variational ansatz, cost function, and optimization method are all to be
    developed by you in this function.

    Args:
        X_train (np.ndarray): An array of floats of size (250, 3) to be used as training data.
        Y_train (np.ndarray): An array of size (250,) which are the categorical labels
            associated to the training data. The categories are labeled by -1, 0, and 1.
        X_test (np.ndarray): An array of floats of (50, 3) to serve as testing data.

    Returns:
        str: The predicted categories of X_test, converted from a list of ints to a
            comma-separated string.
    """

    # Use this array to make a prediction for the labels of the data in X_test
    predictions = []

    # QHACK #

    np.random.seed(0)

    num_classes = 3
    margin = 0.15
    feature_size = 3

    # the number of the required qubits is calculated from the number of features
    num_qubits = int(np.ceil(np.log2(feature_size)))
    num_layers = 2

    dev = qml.device("default.qubit", wires=num_qubits)

    def layer(W):
        for i in range(num_qubits):
            qml.Rot(W[i, 0], W[i, 1], W[i, 2], wires=i)
        for j in range(num_qubits - 1):
            qml.CNOT(wires=[j, j + 1])
        if num_qubits >= 2:
            # Apply additional CNOT to entangle the last with the first qubit
            qml.CNOT(wires=[num_qubits - 1, 0])

    def circuit(weights, feat=None):
        qml.templates.embeddings.AmplitudeEmbedding(feat, range(num_qubits), pad=0.0, normalize=True)
        for W in weights:
            layer(W)

        return qml.expval(qml.PauliZ(0))

    qnodes = []
    for iq in range(num_classes):
        qnode = qml.QNode(circuit, dev)
        qnodes.append(qnode)

    def variational_classifier(q_circuit, params, feat):
        weights = params[0]
        bias = params[1]
        return q_circuit(weights, feat=feat) + bias

    def multiclass_svm_loss(q_circuits, all_params, feature_vecs, true_labels):
        loss = 0
        num_samples = len(true_labels)
        for i, feature_vec in enumerate(feature_vecs):
            # Compute the score given to this sample by the classifier corresponding to the
            # true label. So for a true label of 1, get the score computed by classifer 1,
            # which distinguishes between "class 1" or "not class 1".
            s_true = variational_classifier(
                q_circuits[int(true_labels[i])],
                (all_params[0][int(true_labels[i])], all_params[1][int(true_labels[i])]),
                feature_vec,
            )
            s_true = s_true
            li = 0

            # Get the scores computed for this sample by the other classifiers
            for j in range(num_classes):
                if j != int(true_labels[i]):
                    s_j = variational_classifier(
                        q_circuits[j], (all_params[0][j], all_params[1][j]), feature_vec
                    )
                    s_j = s_j
                    li += max(0, s_j - s_true + margin)
            loss += li

        return loss / num_samples

    def classify(q_circuits, all_params, feature_vecs):
        predicted_labels = []
        for i, feature_vec in enumerate(feature_vecs):
            scores = np.zeros(num_classes)
            for c in range(num_classes):
                score = variational_classifier(
                    q_circuits[c], (all_params[0][c], all_params[1][c]), feature_vec
                )
                scores[c] = float(score)
            pred_class = np.argmax(scores)
            predicted_labels.append(pred_class)
        return predicted_labels


    all_weights = [
        (0.1 * np.random.rand(num_layers, num_qubits, 3))
        for i in range(num_classes)
    ]
    all_bias = [(0.1 * np.ones(1)) for i in range(num_classes)]

    training_params = (all_weights, all_bias)
    q_circuits = qnodes

    opt = qml.AdamOptimizer(stepsize=0.18)

    steps = 12
    cost_tol = 0.008

    Y_train += 1  # To change labels to 0, 1, 2

    for i in range (steps):
        training_params, prev_cost = opt.step_and_cost(lambda v: multiclass_svm_loss(q_circuits, v, X_train, Y_train), training_params)

        if prev_cost <= cost_tol:
            break

    pred = classify(q_circuits, training_params, X_test)
    pred = [x - 1 for x in pred]  # To get original label

    predictions = pred

    # QHACK #

    return array_to_concatenated_string(predictions)


def array_to_concatenated_string(array):
    """DO NOT MODIFY THIS FUNCTION.

    Turns an array of integers into a concatenated string of integers
    separated by commas. (Inverse of concatenated_string_to_array).
    """
    return ",".join(str(x) for x in array)


def concatenated_string_to_array(string):
    """DO NOT MODIFY THIS FUNCTION.

    Turns a concatenated string of integers separated by commas into
    an array of integers. (Inverse of array_to_concatenated_string).
    """
    return np.array([int(x) for x in string.split(",")])


def parse_input(giant_string):
    """DO NOT MODIFY THIS FUNCTION.

    Parse the input data into 3 arrays: the training data, training labels,
    and testing data.

    Dimensions of the input data are:
      - X_train: (250, 3)
      - Y_train: (250,)
      - X_test:  (50, 3)
    """
    X_train_part, Y_train_part, X_test_part = giant_string.split("XXX")

    X_train_row_strings = X_train_part.split("S")
    X_train_rows = [[float(x) for x in row.split(",")] for row in X_train_row_strings]
    X_train = np.array(X_train_rows)

    Y_train = concatenated_string_to_array(Y_train_part)

    X_test_row_strings = X_test_part.split("S")
    X_test_rows = [[float(x) for x in row.split(",")] for row in X_test_row_strings]
    X_test = np.array(X_test_rows)

    return X_train, Y_train, X_test


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    X_train, Y_train, X_test = parse_input(sys.stdin.read())
    output_string = classify_data(X_train, Y_train, X_test)
    print(f"{output_string}")
