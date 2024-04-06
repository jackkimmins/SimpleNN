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
#include <cctype> // For std::isdigit
#include <iterator> // For std::istream_iterator

class TextDataset : public Dataset {
public:
    // Constructor inherits from Dataset
    TextDataset(const std::string& filename, bool skipHeader = false) : Dataset(filename, skipHeader) {}

protected:
    std::unordered_map<std::string, int> wordIndex; // Maps words to indices in the feature vector
    int totalWords = 0; // Keep track of total number of unique words

    // Override loadDataset to handle text
    void loadDataset(const std::string& filename, bool skipHeader) override {
        std::ifstream file(filename);
        std::string line, cell, label;
        bool isTextColumn = false; // Flag to check if current column is text
        std::vector<float> dataRow;
        std::vector<std::string> textColumns;

        if (skipHeader && !std::getline(file, line)) return;

        // First pass to detect text columns and construct word index
        while (std::getline(file, line)) {
            std::stringstream lineStream(line);
            textColumns.clear();

            while (std::getline(lineStream, cell, ',')) {
                if (lineStream && std::istringstream(cell) >> std::ws && lineStream.peek() == EOF) {
                    label = cell; // Assume last column is label
                } else if (isText(cell)) {
                    textColumns.push_back(cell);
                    isTextColumn = true;
                } else {
                    // If not text, just convert to float and push to row
                    dataRow.push_back(stof(cell));
                }
            }

            if (isTextColumn) {
                for (const auto& text : textColumns) {
                    std::istringstream textStream(text);
                    std::vector<std::string> words{std::istream_iterator<std::string>{textStream},
                                                   std::istream_iterator<std::string>{}};
                    for (const auto& word : words) {
                        wordIndex.try_emplace(word, totalWords);
                        if (wordIndex[word] == totalWords) ++totalWords;
                    }
                }
            }

            // Reset stream to start to process data rows again
            file.clear();
            file.seekg(0);
            if (skipHeader) std::getline(file, line); // Skip header again if necessary

            break; // Break after the first pass
        }

        // Second pass to vectorize text and load data
        while (std::getline(file, line)) {
            std::stringstream lineStream(line);
            std::vector<float> featureVector(totalWords, 0); // Initialize feature vector with zeroes

            while (std::getline(lineStream, cell, ',')) {
                if (lineStream && std::istringstream(cell) >> std::ws && lineStream.peek() == EOF) {
                    label = cell;
                } else if (isTextColumn && isText(cell)) {
                    std::istringstream textStream(cell);
                    std::vector<std::string> words{std::istream_iterator<std::string>{textStream},
                                                   std::istream_iterator<std::string>{}};
                    for (const auto& word : words) {
                        int wordIndexValue = wordIndex[word];
                        featureVector[wordIndexValue] += 1; // Use simple count for word occurrence
                    }
                } else {
                    // Handle numeric data
                }
            }

            data.push_back(featureVector);
            labels.push_back(label_mapping.try_emplace(label, label_mapping.size()).first->second);
        }
    }

    // Function to check if a string is text (not purely numeric)
    bool isText(const std::string& str) {
        return std::any_of(str.begin(), str.end(), [](char c) { return !std::isdigit(c) && c != '.' && c != '-'; });
    }
};