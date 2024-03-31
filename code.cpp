#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <map>
#include <chrono>
#include <cmath>
#include <numeric>
#include <unordered_map>
#include <limits>

class Dataset {
public:
    std::vector<std::vector<float>> data; // Stores the features of the dataset
    std::vector<int> labels; // Stores the labels of the dataset
    std::vector<std::vector<float>> data_train, data_test;
    std::vector<int> labels_train, labels_test;
    std::unordered_map<std::string, int> label_mapping; // Maps string labels to integers

    Dataset(const std::string& filename, bool skipHeader = false) {
        loadDataset(filename, skipHeader);
    }

    void loadDataset(const std::string& filename, bool skipHeader) {
        std::ifstream file(filename);
        std::string line;
        
        if (skipHeader) {
            std::getline(file, line); // Skip the first line
        }

        while (std::getline(file, line)) {
            std::stringstream lineStream(line);
            std::string cell;
            std::vector<float> dataRow;
            std::string label;
            while (std::getline(lineStream, cell, ',')) {
                if (std::istringstream(cell) >> std::ws && lineStream.peek() == EOF) { // Check if it's the last element
                    label = cell;
                } else {
                    try {
                        dataRow.push_back(std::stof(cell));
                    } catch (const std::exception& e) {
                        std::cerr << "Error converting string to float: " << e.what() << '\n';
                    }
                }
            }
            if (label_mapping.find(label) == label_mapping.end()) { // If label not in map, add it
                int newLabel = label_mapping.size();
                label_mapping[label] = newLabel;
            }
            data.push_back(dataRow);
            labels.push_back(label_mapping[label]);
        }
    }

    void splitDataset(float trainSize = 0.8, unsigned int seed = 42) {
        std::vector<int> indices(data.size());
        std::iota(indices.begin(), indices.end(), 0); // Fill with 0, 1, ..., data.size() - 1

        // unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::shuffle(indices.begin(), indices.end(), std::default_random_engine(seed));

        int splitIndex = static_cast<int>(data.size() * trainSize);
        for (int i = 0; i < splitIndex; ++i) {
            data_train.push_back(data[indices[i]]);
            labels_train.push_back(labels[indices[i]]);
        }
        for (int i = splitIndex; i < data.size(); ++i) {
            data_test.push_back(data[indices[i]]);
            labels_test.push_back(labels[indices[i]]);

            // std::cout << data[indices[i]][0] << " " << data[indices[i]][1] << " " << data[indices[i]][2] << " " << data[indices[i]][3] << " " << labels[indices[i]] << std::endl;
        }

        std::cout << "Training set size: " << data_train.size() << std::endl;
        std::cout << "Testing set size: " << data_test.size() << std::endl;
    }
};

class NeuralNetwork {
private:
    std::vector<std::vector<float>> weights_input_hidden;
    std::vector<float> biases_hidden;
    std::vector<std::vector<float>> weights_hidden_output;
    std::vector<float> biases_output;
    float learning_rate;
    int input_nodes, hidden_nodes, output_nodes;

    // Sigmoid activation function
    float sigmoid(float x) {
        return 1.0f / (1.0f + std::exp(-x));
    }

    // Derivative of sigmoid function
    float sigmoid_derivative(float x) {
        return x * (1 - x);
    }

    // Softmax function for output layer
    std::vector<float> softmax(const std::vector<float>& x) {
        std::vector<float> output(x.size());
        float maxElement = *std::max_element(x.begin(), x.end());
        float sum = 0.0f;
        for (size_t i = 0; i < x.size(); ++i) {
            output[i] = std::exp(x[i] - maxElement); // Improve numerical stability
            sum += output[i];
        }
        for (size_t i = 0; i < x.size(); ++i) {
            output[i] /= sum;
        }
        return output;
    }

    // Cross-Entropy loss function
    float cross_entropy(const std::vector<float>& predicted, int true_index) {
        return -std::log(predicted[true_index]);
    }

public:
    NeuralNetwork(int input, int hidden, int output, float rate, unsigned int seed = 42) : input_nodes(input), hidden_nodes(hidden), output_nodes(output), learning_rate(rate) {
        std::default_random_engine generator(seed);
        std::normal_distribution<float> distribution(0.0, 0.1);

        auto weightInitializer = [&](int rows, int cols) {
            std::vector<std::vector<float>> weights(rows, std::vector<float>(cols));
            for (auto& row : weights)
                for (auto& val : row)
                    val = distribution(generator);
            return weights;
        };

        weights_input_hidden = weightInitializer(input_nodes, hidden_nodes);
        weights_hidden_output = weightInitializer(hidden_nodes, output_nodes);
        biases_hidden = std::vector<float>(hidden_nodes, 0.0f);
        biases_output = std::vector<float>(output_nodes, 0.0f);
    }

    std::pair<std::vector<float>, std::vector<float>> forward(const std::vector<float>& inputs) {
        auto computeLayerOutputs = [&](const std::vector<float>& inputs, const std::vector<std::vector<float>>& weights, const std::vector<float>& biases, bool isOutput = false) {
            std::vector<float> outputs(weights[0].size(), 0.0f);
            for (size_t i = 0; i < outputs.size(); ++i) {
                for (size_t j = 0; j < inputs.size(); ++j) {
                    outputs[i] += inputs[j] * weights[j][i];
                }
                outputs[i] += biases[i];
                outputs[i] = isOutput ? outputs[i] : sigmoid(outputs[i]);
            }
            return outputs;
        };

        std::vector<float> hidden_outputs = computeLayerOutputs(inputs, weights_input_hidden, biases_hidden);
        std::vector<float> final_outputs = computeLayerOutputs(hidden_outputs, weights_hidden_output, biases_output, true);
        final_outputs = softmax(final_outputs);

        return {hidden_outputs, final_outputs};
    }

    void backpropagate(const std::vector<float>& inputs, int true_label) {
        auto [hidden_outputs, outputs] = forward(inputs);

        // Convert true_label to one-hot encoding
        std::vector<float> true_values(output_nodes, 0.0f);
        true_values[true_label] = 1.0f;

        // Calculate error between predicted output and true values
        std::vector<float> output_errors(output_nodes);
        for (int i = 0; i < output_nodes; ++i) {
            output_errors[i] = true_values[i] - outputs[i];
        }

        // Calculate gradients for weights between hidden and output layers
        std::vector<std::vector<float>> hidden_output_gradients(weights_hidden_output.size(), std::vector<float>(weights_hidden_output[0].size()));
        for (int i = 0; i < hidden_nodes; ++i) {
            for (int j = 0; j < output_nodes; ++j) {
                hidden_output_gradients[i][j] = output_errors[j] * outputs[j] * (1 - outputs[j]) * hidden_outputs[i];
            }
        }

        // Calculate gradients for biases in the output layer
        std::vector<float> output_bias_gradients(output_nodes);
        for (int i = 0; i < output_nodes; ++i) {
            output_bias_gradients[i] = output_errors[i] * outputs[i] * (1 - outputs[i]);
        }

        // Calculate the hidden layer errors
        std::vector<float> hidden_errors(hidden_nodes, 0.0f);
        for (int i = 0; i < hidden_nodes; ++i) {
            for (int j = 0; j < output_nodes; ++j) {
                hidden_errors[i] += output_errors[j] * weights_hidden_output[i][j];
            }
        }

        // Calculate gradients for weights between input and hidden layers
        std::vector<std::vector<float>> input_hidden_gradients(weights_input_hidden.size(), std::vector<float>(weights_input_hidden[0].size()));
        for (int i = 0; i < input_nodes; ++i) {
            for (int j = 0; j < hidden_nodes; ++j) {
                input_hidden_gradients[i][j] = hidden_errors[j] * sigmoid_derivative(hidden_outputs[j]) * inputs[i];
            }
        }

        // Calculate gradients for biases in the hidden layer
        std::vector<float> hidden_bias_gradients(hidden_nodes);
        for (int i = 0; i < hidden_nodes; ++i) {
            hidden_bias_gradients[i] = hidden_errors[i] * sigmoid_derivative(hidden_outputs[i]);
        }

        // Update weights and biases using gradient descent
        for (int i = 0; i < input_nodes; ++i) {
            for (int j = 0; j < hidden_nodes; ++j) {
                weights_input_hidden[i][j] += learning_rate * input_hidden_gradients[i][j];
            }
        }

        for (int i = 0; i < hidden_nodes; ++i) {
            biases_hidden[i] += learning_rate * hidden_bias_gradients[i];
        }

        for (int i = 0; i < hidden_nodes; ++i) {
            for (int j = 0; j < output_nodes; ++j) {
                weights_hidden_output[i][j] += learning_rate * hidden_output_gradients[i][j];
            }
        }

        for (int i = 0; i < output_nodes; ++i) {
            biases_output[i] += learning_rate * output_bias_gradients[i];
        }
    }

    void train(int epochs, const Dataset& dataset, float validation_split = 0.1, int patience = 10) {
        int patience_counter = patience; // Initialize patience counter
        float best_loss = std::numeric_limits<float>::max(); // Initialize best loss to maximum float value

        // Determine the split index for training and validation
        int validation_size = static_cast<int>(dataset.data_train.size() * validation_split);
        int training_size = dataset.data_train.size() - validation_size;

        for (int epoch = 0; epoch < epochs; ++epoch) {
            float total_loss = 0.0f;

            // Training loop
            for (int i = 0; i < training_size; ++i) {
                backpropagate(dataset.data_train[i], dataset.labels_train[i]);
                auto [_, outputs] = forward(dataset.data_train[i]);
                total_loss += cross_entropy(outputs, dataset.labels_train[i]);
            }

            float training_loss = total_loss / training_size;

            // Validation loop
            total_loss = 0.0f;
            for (int i = training_size; i < dataset.data_train.size(); ++i) {
                auto [_, outputs] = forward(dataset.data_train[i]);
                total_loss += cross_entropy(outputs, dataset.labels_train[i]);
            }

            float validation_loss = total_loss / validation_size;

            std::cout << "Epoch " << epoch + 1 << " Training Loss: " << training_loss << ", Validation Loss: " << validation_loss << std::endl;

            // Early stopping logic
            if (validation_loss < best_loss) {
                best_loss = validation_loss;
                patience_counter = patience; // Reset patience
            } else {
                patience_counter -= 1; // Decrease patience
                if (patience_counter == 0) {
                    std::cout << "Early stopping triggered at epoch " << epoch + 1 << std::endl;
                    break; // Stop training
                }
            }
        }
    }

    // Predict the class label for a single data point
    int predict(const std::vector<float>& inputs) {
        auto [_, outputs] = forward(inputs);
        return std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end()));
    }
};

class Evaluator {
public:
    // Evaluate the neural network over a given dataset and return the accuracy
    static double evaluate(NeuralNetwork& nn, const std::vector<std::vector<float>>& testData, const std::vector<int>& testLabels) {
        std::vector<int> predictions;
        for (const auto& inputs : testData) {
            auto [hidden, output] = nn.forward(inputs);
            int predicted_label = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
            predictions.push_back(predicted_label);
        }

        return computeAccuracy(predictions, testLabels);
    }

private:
    static double computeAccuracy(const std::vector<int>& predictions, const std::vector<int>& labels) {
        if (predictions.size() != labels.size()) {
            std::cerr << "Error: Size of predictions and labels must be the same." << std::endl;
            return 0.0;
        }

        int correctCount = 0;
        for (size_t i = 0; i < predictions.size(); ++i) {
            if (predictions[i] == labels[i]) {
                ++correctCount;
            }
        }

        return static_cast<double>(correctCount) / predictions.size();
    }
};


int main() {
    // std::string filename = "iris.csv";
    // std::string filename = "breast_cancer.csv";
    // std::string filename = "social_network_ads.csv";
    std::string filename = "titanic_dataset.csv";

    // Load and split the dataset
    Dataset irisDataset(filename, true);
    irisDataset.splitDataset(0.7);

    // Initialize the neural network
    // NeuralNetwork nn(4, 10, 3, 0.01);
    // NeuralNetwork nn(9, 10, 2, 0.001);
    // NeuralNetwork nn(3, 20, 2, 0.001);
    NeuralNetwork nn(6, 20, 2, 0.001);

    // Train the neural network
    nn.train(1000, irisDataset);

    // Evaluate the neural network
    double accuracy = Evaluator::evaluate(nn, irisDataset.data_test, irisDataset.labels_test);
    std::cout << "Accuracy on test set: " << accuracy * 100 << "%" << std::endl;

    // Predict a single data point
    // std::vector<float> dataPoint = {4.8, 3.0, 1.5, 0.3};

    // Pclass,Sex,Age,SibSp,Parch,Fare
    std::vector<float> dataPoint = {3, 1, 61, 0, 0, 30};
    int predictedLabel = nn.predict(dataPoint);
    std::cout << "Predicted label: " << predictedLabel << std::endl;

    return 0;
}