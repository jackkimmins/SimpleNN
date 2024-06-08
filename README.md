# Feedforward Neural Network Classifier in C++

This project implements a feedforward neural network classifier from scratch in C++20, without relying on external libraries (other than the standard libray). It provides functionalities for data handling, training, prediction, and evaluation.

This project aims to demonstrate the core principles of neural networks and serves as an educational tool. It is not designed for production deployment.

Usage Example:

```cpp

#include <string>
#include <vector>
#include <iostream>
#include "dataset.hpp"
#include "neural_network.hpp"
#include "evaluator.hpp"

int main() {
    const std::string filename = "datasets/titanic_dataset.csv";
    NeuralNetwork nn(6, 20, 2, 0.001);

    Dataset dataset(filename, true);
    dataset.splitDataset(0.7);

    // Train the neural network
    if (!nn.train(1000, dataset)) return 1;

    // Evaluate the neural network
    double accuracy = Evaluator::evaluate(nn, dataset.data_test, dataset.labels_test);
    std::cout << "Accuracy on test set: " << accuracy * 100 << "%" << std::endl;

    // Predict a new data point
    std::vector<float> dataPoint = {3, 1, 61, 0, 0, 30};
    int predictedLabel = nn.predict(dataPoint);
    std::cout << "Predicted label: " << predictedLabel << std::endl;

    return 0;
}
```

Classification Performance on [Titanic Dataset](https://www.kaggle.com/datasets/vinicius150987/titanic3) on Test Set Split (30%): `68.2836%`
