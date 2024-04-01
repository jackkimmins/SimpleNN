#include <vector>
#include <unordered_map>
#include <fstream>
#include <string>
#include <sstream>
#include <filesystem>
#include <iostream>
#include <algorithm>
#include <random>
#include <numeric>

class Dataset {
public:
    std::vector<std::vector<float>> data, data_train, data_test;
    std::vector<int> labels, labels_train, labels_test;
    std::unordered_map<std::string, int> label_mapping;

    Dataset(const std::string& filename, bool skipHeader = false) {
        // Check if the file exists before attempting to load
        if (!std::filesystem::exists(filename)) {
            std::cerr << "Error: File " << filename << " not found.\n";
            return;
        }
        loadDataset(filename, skipHeader);
    }

    // Load data from a CSV file, optionally skipping the header
    void loadDataset(const std::string& filename, bool skipHeader) {
        std::ifstream file(filename);
        std::string line, cell, label;
        std::vector<float> dataRow;
        
        // If skipping the header, read and discard the first line
        if (skipHeader && !std::getline(file, line)) return;

        // Read each line of the file
        while (std::getline(file, line)) {
            std::stringstream lineStream(line);
            dataRow.clear();
            // Split each line into cells using ',' as a delimiter
            while (std::getline(lineStream, cell, ',')) {
                // Check if the current cell is the label (last element in the row)
                if (lineStream && std::istringstream(cell) >> std::ws && lineStream.peek() == EOF) {
                    label = cell;
                } else {
                    // Convert string to float and add to the current row's data
                    dataRow.push_back(stof(cell));
                }
            }
            // Add the processed row to the dataset and assign a numeric label
            data.push_back(dataRow);
            labels.push_back(label_mapping.try_emplace(label, label_mapping.size()).first->second);
        }
    }

    // Split the dataset into a training set and a testing set based on a specified ratio
    void splitDataset(float trainSize = 0.8, unsigned int seed = 42) {
        std::vector<int> indices(data.size());
        // Generate a sequence of indices for the dataset
        std::iota(indices.begin(), indices.end(), 0);
        // Shuffle the indices to randomise the data before splitting
        std::shuffle(indices.begin(), indices.end(), std::default_random_engine(seed));

        // Determine the index at which to split the data into training and testing sets
        int splitIndex = static_cast<int>(data.size() * trainSize);
        // Assign data to the training set based on the shuffled indices
        for (int i = 0; i < splitIndex; ++i) {
            data_train.push_back(data[indices[i]]);
            labels_train.push_back(labels[indices[i]]);
        }
        // Assign the remaining data to the testing set
        for (int i = splitIndex; i < data.size(); ++i) {
            data_test.push_back(data[indices[i]]);
            labels_test.push_back(labels[indices[i]]);
        }

        // Print the size of the training and testing sets
        std::cout << "Training set size: " << data_train.size() << "\nTesting set size: " << data_test.size() << '\n';
    }

    // Get the number of inputs (features) in the dataset
    int getNumInputs() const { return data[0].size(); }
};