#include <vector>
#include <iostream>

class Evaluator {
public:
    // Evaluate the neural network over a given dataset and return the accuracy
    static double evaluate(NeuralNetwork& nn, const std::vector<std::vector<float>>& testData, const std::vector<int>& testLabels) {
        int correctCount = 0;
        for (size_t i = 0; i < testData.size(); ++i) {
            if (nn.predict(testData[i]) == testLabels[i]) ++correctCount;
        }
        return static_cast<double>(correctCount) / testData.size();
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