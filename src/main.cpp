#include <string>
#include <vector>
#include <iostream>

#include "dataset.hpp"
#include "textaware_dataset.hpp"
#include "neural_network.hpp"
#include "evaluator.hpp"

int main() {
    // const std::string filename = "datasets/iris.csv";
    // NeuralNetwork nn(4, 10, 3, 0.01);
    
    // const std::string filename = "datasets/breast_cancer.csv";
    // NeuralNetwork nn(9, 20, 2, 0.001);
    
    // const std::string filename = "datasets/social_network_ads.csv";
    // NeuralNetwork nn(3, 20, 2, 0.001);

    // const std::string filename = "datasets/titanic_dataset.csv";
    // NeuralNetwork nn(6, 20, 2, 0.001);

    // Dataset dataset(filename, true);
    // dataset.splitDataset(0.7);

    const std::string filename = "datasets/emails.csv";
    TextDataset dataset(filename, true);
    dataset.splitDataset(0.7);
    NeuralNetwork nn(2, 20, 2, 0.001);

    // Train the neural network
    if (!nn.train(1000, dataset)) return 1;

    // Evaluate the neural network
    double accuracy = Evaluator::evaluate(nn, dataset.data_test, dataset.labels_test);
    std::cout << "Accuracy on test set: " << accuracy * 100 << "%" << std::endl;

    // Pclass,Sex,Age,SibSp,Parch,Fare
    std::vector<float> dataPoint = {3, 1, 61, 0, 0, 30};
    int predictedLabel = nn.predict(dataPoint);
    std::cout << "Predicted label: " << predictedLabel << std::endl;

    return 0;
}