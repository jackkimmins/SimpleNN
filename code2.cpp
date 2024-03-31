#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <map>
#include <random>

// Sigmoid activation function
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of sigmoid
double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

// Softmax activation function
std::vector<double> softmax(const std::vector<double>& x) {
    double max = *max_element(x.begin(), x.end());
    std::vector<double> exp_values;
    double sum = 0.0;
    for (auto& val : x) {
        exp_values.push_back(exp(val - max)); // Improve numerical stability
        sum += exp_values.back();
    }
    for (auto& val : exp_values) {
        val /= sum;
    }
    return exp_values;
}

// Neural Network class
class NeuralNetwork {
public:
    std::vector<std::vector<double>> weights_input_hidden;
    std::vector<std::vector<double>> weights_hidden_output;
    double learning_rate;

    NeuralNetwork(int input_nodes, int hidden_nodes, int output_nodes, double lr) : learning_rate(lr) {
        // Initialize weights with random values
        srand(time(0)); // Seed for random number generation
        weights_input_hidden.resize(input_nodes, std::vector<double>(hidden_nodes));
        for (auto& w : weights_input_hidden) {
            for (auto& val : w) {
                val = (double)rand() / RAND_MAX * 2.0 - 1.0; // Random values between -1 and 1
            }
        }

        weights_hidden_output.resize(hidden_nodes, std::vector<double>(output_nodes));
        for (auto& w : weights_hidden_output) {
            for (auto& val : w) {
                val = (double)rand() / RAND_MAX * 2.0 - 1.0; // Random values between -1 and 1
            }
        }
    }

    std::vector<double> feedforward(const std::vector<double>& input) {
        // Calculate hidden layer output
        std::vector<double> hidden_output(weights_input_hidden[0].size(), 0.0);
        for (size_t i = 0; i < weights_input_hidden.size(); ++i) {
            for (size_t j = 0; j < weights_input_hidden[0].size(); ++j) {
                hidden_output[j] += input[i] * weights_input_hidden[i][j];
            }
        }
        for (double& val : hidden_output) {
            val = sigmoid(val);
        }

        // Calculate final output
        std::vector<double> final_output(weights_hidden_output[0].size(), 0.0);
        for (size_t i = 0; i < weights_hidden_output.size(); ++i) {
            for (size_t j = 0; j < weights_hidden_output[0].size(); ++j) {
                final_output[j] += hidden_output[i] * weights_hidden_output[i][j];
            }
        }
        final_output = softmax(final_output);

        return final_output;
    }

    void train(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& targets, int epochs) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            double sum_error = 0.0;
            for (size_t i = 0; i < inputs.size(); ++i) {
                // Forward propagation
                std::vector<double> input = inputs[i];
                std::vector<double> hidden_outputs(weights_input_hidden[0].size(), 0.0);
                for (size_t j = 0; j < weights_input_hidden.size(); ++j) {
                    for (size_t k = 0; k < weights_input_hidden[0].size(); ++k) {
                        hidden_outputs[k] += input[j] * weights_input_hidden[j][k];
                    }
                }
                for (double& val : hidden_outputs) {
                    val = sigmoid(val);
                }

                std::vector<double> output = feedforward(input);
                std::vector<double> output_errors(targets[i].size());
                
                // Calculate error (difference between expected and predicted)
                for (size_t j = 0; j < targets[i].size(); ++j) {
                    output_errors[j] = targets[i][j] - output[j];
                    sum_error += output_errors[j] * output_errors[j]; // MSE
                }

                // Calculate gradient for hidden-output weights
                std::vector<double> hidden_errors(weights_hidden_output.size(), 0.0);
                for (size_t j = 0; j < weights_hidden_output.size(); ++j) {
                    for (size_t k = 0; k < weights_hidden_output[0].size(); ++k) {
                        hidden_errors[j] += output_errors[k] * weights_hidden_output[j][k];
                        double delta = learning_rate * output_errors[k] * sigmoid_derivative(hidden_outputs[j]);
                        weights_hidden_output[j][k] -= input[j] * delta;
                    }
                }

                // Update weights for input-hidden layer
                for (size_t j = 0; j < weights_input_hidden.size(); ++j) {
                    for (size_t k = 0; k < weights_input_hidden[0].size(); ++k) {
                        double delta = learning_rate * hidden_errors[k] * sigmoid_derivative(hidden_outputs[k]);
                        weights_input_hidden[j][k] -= input[j] * delta;
                    }
                }
            }
            std::cout << "Epoch " << epoch + 1 << ", MSE: " << sum_error / inputs.size() << std::endl;
        }
    }

    // Predicts class for a given input
    std::vector<double> predict(const std::vector<double>& input) {
        return feedforward(input); // Use feedforward to get the softmax output
    }

    // Evaluates the neural network's performance on a given dataset
    double evaluate(const std::vector<std::vector<double>>& testInputs, const std::vector<std::vector<double>>& testTargets) {
        int correct_predictions = 0;
        for (size_t i = 0; i < testInputs.size(); ++i) {
            std::vector<double> predicted = predict(testInputs[i]);
            int predicted_class = std::distance(predicted.begin(), std::max_element(predicted.begin(), predicted.end()));
            int actual_class = std::distance(testTargets[i].begin(), std::max_element(testTargets[i].begin(), testTargets[i].end()));
            
            if (predicted_class == actual_class) {
                correct_predictions++;
            }
        }
        return static_cast<double>(correct_predictions) / static_cast<double>(testInputs.size());
    }
};

// Function to load the Iris dataset from a CSV file
void loadIrisDataset(const std::string& filename, std::vector<std::vector<double>>& inputs, std::vector<std::vector<double>>& targets) {
    std::ifstream file(filename);
    std::string line;
    // Mapping for the Iris species to integers
    std::map<std::string, int> labelEncoding = {
        {"Iris-setosa", 0},
        {"Iris-versicolor", 1},
        {"Iris-virginica", 2}
    };

    while (std::getline(file, line)) {
        std::vector<double> input;
        std::vector<double> target(3, 0.0); // Initialize target vector for 3 classes
        std::stringstream linestream(line);
        std::string cell;
        int featureIndex = 0;

        while (std::getline(linestream, cell, ',')) {
            if (featureIndex < 4) { // First 4 columns are features
                input.push_back(std::stod(cell));
            } else {
                // Encode the label as a one-hot vector
                std::string label = cell;
                int encodedLabel = labelEncoding[label];
                target[encodedLabel] = 1.0;
            }
            featureIndex++;
        }

        inputs.push_back(input);
        targets.push_back(target);
    }
}

// Function to shuffle and split the dataset
void splitDataset(const std::vector<std::vector<double>>& inputs,
                  const std::vector<std::vector<double>>& targets,
                  std::vector<std::vector<double>>& trainingInputs,
                  std::vector<std::vector<double>>& trainingTargets,
                  std::vector<std::vector<double>>& testingInputs,
                  std::vector<std::vector<double>>& testingTargets,
                  double trainSizeRatio) {
    // Determine the split index
    size_t totalSize = inputs.size();
    size_t trainSize = static_cast<size_t>(totalSize * trainSizeRatio);

    // Create a vector of indices and shuffle it
    std::vector<size_t> indices(totalSize);
    std::iota(indices.begin(), indices.end(), 0); // Fill with 0, 1, ..., totalSize-1
    std::shuffle(indices.begin(), indices.end(), std::default_random_engine(static_cast<unsigned>(time(nullptr))));

    // Split the dataset based on the shuffled indices
    for (size_t i = 0; i < trainSize; ++i) {
        trainingInputs.push_back(inputs[indices[i]]);
        trainingTargets.push_back(targets[indices[i]]);
    }
    for (size_t i = trainSize; i < totalSize; ++i) {
        testingInputs.push_back(inputs[indices[i]]);
        testingTargets.push_back(targets[indices[i]]);
    }
}

int main() {
    NeuralNetwork nn(4, 5, 3, 0.1);

    std::vector<std::vector<double>> inputs;
    std::vector<std::vector<double>> targets;

    // Load the dataset
    loadIrisDataset("iris.csv", inputs, targets);

    // Split dataset into training and testing sets
    std::vector<std::vector<double>> trainingInputs, trainingTargets;
    std::vector<std::vector<double>> testingInputs, testingTargets;
    splitDataset(inputs, targets, trainingInputs, trainingTargets, testingInputs, testingTargets, 0.8);

    // Train the neural network with the training set
    nn.train(trainingInputs, trainingTargets, 1000);

    // Evaluate the neural network with the testing set
    double accuracy = nn.evaluate(testingInputs, testingTargets);
    std::cout << "Accuracy on testing set: " << accuracy * 100 << "%" << std::endl;

    return 0;
}