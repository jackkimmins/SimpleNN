#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <iostream>
#include <limits>
#include <utility>

class NeuralNetwork {
private:
    // Matrices for weights and vectors for biases
    std::vector<std::vector<float>> weights_input_hidden, weights_hidden_output;
    std::vector<float> biases_hidden, biases_output;

    // Learning rate for the gradient descent optimisation algorithm
    float learning_rate;

    // Number of nodes in each layer of the network
    int input_nodes, hidden_nodes, output_nodes;

    // Activation function and its derivative
    float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
    float sigmoid_derivative(float x) { return x * (1 - x); }


    // Softmax function for the output layer to normalise outputs to a probability distribution
    std::vector<float> softmax(const std::vector<float>& x) {
        std::vector<float> output(x.size());
        float maxElement = *max_element(x.begin(), x.end()), sum = 0.0f;
        std::transform(x.begin(), x.end(), output.begin(), [&](float val) { return std::exp(val - maxElement); });
        sum = accumulate(output.begin(), output.end(), 0.0f);
        for (auto& val : output) val /= sum;
        return output;
    }

    // Cross-entropy loss function to measure prediction error
    float cross_entropy(const std::vector<float>& predicted, int true_index) {
        float epsilon = 1e-12, predicted_prob = std::clamp(predicted[true_index], epsilon, 1.0f - epsilon);
        return -std::log(predicted_prob);
    }

    // Initialises weights and biases with random values from a normal distribution
    void initialiseWeights(int input, int hidden, int output, unsigned int seed) {
        std::default_random_engine generator(seed);
        std::normal_distribution<float> distribution(0.0, 0.1);

        auto weightInitialiser = [&](int rows, int cols) {
            std::vector<std::vector<float>> weights(rows, std::vector<float>(cols));
            for (auto& row : weights)
                for (auto& val : row)
                    val = distribution(generator);
            return weights;
        };

        weights_input_hidden = weightInitialiser(input, hidden);
        weights_hidden_output = weightInitialiser(hidden, output);
        biases_hidden.resize(hidden, 0.0f);
        biases_output.resize(output, 0.0f);
    }

public:
    // Constructor initialises the network architecture and learning parameters
    NeuralNetwork(int input, int hidden, int output, float rate, unsigned int seed = 42) : input_nodes(input), hidden_nodes(hidden), output_nodes(output), learning_rate(rate) {
        initialiseWeights(input, hidden, output, seed);
    }

    // Forward pass through the network to calculate output values
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

    bool train(int epochs, const Dataset& dataset, float validation_split = 0.1, int patience = 10) {
        // Check if the input size matches the number of input nodes
        if (dataset.getNumInputs() != input_nodes) {
            std::cerr << "Error: Input size does not match the number of input nodes." << std::endl;
            return false;
        }

        int patience_counter = patience;
        float best_loss = std::numeric_limits<float>::max();

        // Determine the split index for training and validation
        int validation_size = static_cast<int>(dataset.data_train.size() * validation_split);
        int training_size = dataset.data_train.size() - validation_size;

        for (int epoch = 0; epoch < epochs; ++epoch) {
            float total_loss = 0.0f;
            std::cout << "Epoch " << epoch + 1 << "/" << epochs << ":" << std::endl;

            // Training loop with progress display
            for (int i = 0; i < training_size; ++i) {
                backpropagate(dataset.data_train[i], dataset.labels_train[i]);
                auto [_, outputs] = forward(dataset.data_train[i]);
                total_loss += cross_entropy(outputs, dataset.labels_train[i]);

                // Progress bar display
                int progress = static_cast<int>(100.0 * (i + 1) / training_size);
                std::cout << "\r[";
                std::cout << std::string(progress / 2, '=') << std::string(50 - progress / 2, ' ');
                std::cout << "] " << progress << "%" << std::flush;
            }
            std::cout << std::endl;

            float training_loss = total_loss / training_size;

            // Validation loop
            total_loss = 0.0f;
            for (int i = training_size; i < dataset.data_train.size(); ++i) {
                auto [_, outputs] = forward(dataset.data_train[i]);
                total_loss += cross_entropy(outputs, dataset.labels_train[i]);
            }

            float validation_loss = total_loss / validation_size;

            std::cout << "Training Loss: " << training_loss << ", Validation Loss: " << validation_loss << std::endl << std::endl;

            // Early stopping if the validation loss does not improve
            if (validation_loss < best_loss) {
                best_loss = validation_loss;
                patience_counter = patience;
            } else {
                patience_counter -= 1;
                if (patience_counter == 0) {
                    std::cout << "Early stopping triggered at epoch " << epoch + 1 << std::endl << std::endl;
                    break;
                }
            }
        }

        return true;
    }

    // Predict the class label for a single data point
    int predict(const std::vector<float>& inputs) {
        if (inputs.size() != input_nodes) {
            std::cerr << "Error: Input size does not match the number of input nodes." << std::endl;
            return -1;
        }

        auto [_, outputs] = forward(inputs);
        return std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end()));
    }
};