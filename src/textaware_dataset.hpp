#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <string>
#include <sstream>
#include <filesystem>
#include <iostream>
#include <algorithm>
#include <random>
#include <numeric>
#include <cctype>

class TextAwareDataset : public Dataset {
private:
    // Utility function to safely convert string to float
    float safeStof(const std::string& str, float defaultValue = 0.0f) {
        try {
            return std::stof(str);
        } catch (const std::invalid_argument& e) {
            std::cerr << "Invalid argument for conversion: " << str << ". Defaulting to " << defaultValue << ".\n";
        } catch (const std::out_of_range& e) {
            std::cerr << "Out of range for conversion: " << str << ". Defaulting to " << defaultValue << ".\n";
        }
        return defaultValue; // Return a default value if conversion fails
    }

public:
    // Constructor: Use the base class constructor
    TextAwareDataset(const std::string& filename, bool skipHeader = false) : Dataset(filename, skipHeader) {}

protected:
    std::unordered_map<std::string, size_t> wordIndex; // Maps each word to its index in the BoW vector

    // Override the loadDataset method to handle text data
    void loadDataset(const std::string& filename, bool skipHeader) override {
        std::ifstream file(filename);
        std::string line, cell, label;
        std::vector<std::vector<std::string>> rawData; // Temporarily store text data

        if (skipHeader && !std::getline(file, line)) return;

        // Read each line of the file
        while (std::getline(file, line)) {
            std::stringstream lineStream(line);
            std::vector<std::string> row;
            while (std::getline(lineStream, cell, ',')) {
                row.push_back(cell);
            }
            rawData.push_back(row);
        }

        // Determine if a column contains text and should be vectorized
        std::vector<bool> isTextColumn(rawData[0].size(), false); // Assume all columns are numeric initially
        detectTextColumns(rawData, isTextColumn);

        // Build the word index from text columns
        buildWordIndex(rawData, isTextColumn);

        // Convert rawData to numerical data
        convertToNumerical(rawData, isTextColumn);
    }

    // Detect which columns contain text data
    void detectTextColumns(const std::vector<std::vector<std::string>>& rawData, std::vector<bool>& isTextColumn) {
        for (const auto& row : rawData) {
            for (size_t i = 0; i < row.size(); i++) {
                if (!std::all_of(row[i].begin(), row[i].end(), ::isdigit)) {
                    isTextColumn[i] = true;
                }
            }
        }
    }

    // Build word index for text columns
    void buildWordIndex(const std::vector<std::vector<std::string>>& rawData, const std::vector<bool>& isTextColumn) {
        for (const auto& row : rawData) {
            for (size_t i = 0; i < row.size(); i++) {
                if (isTextColumn[i]) {
                    wordIndex[row[i]] = wordIndex.size(); // Assign an index to each unique word
                }
            }
        }
    }

    // Convert text columns to numerical using Bag of Words and integrate with the existing data structure
    void convertToNumerical(const std::vector<std::vector<std::string>>& rawData, const std::vector<bool>& isTextColumn) {
        for (const auto& row : rawData) {
            std::vector<float> dataRow;
            std::string label;
            for (size_t i = 0; i < row.size(); i++) {
                if (!isTextColumn[i]) { // For numeric columns, convert directly
                    std::cout << row[i] << std::endl;
                    // dataRow.push_back(safeStof(row[i]));
                } else { // For text columns, use BoW vector
                    std::vector<float> textVector(wordIndex.size(), 0.0);
                    if (wordIndex.find(row[i]) != wordIndex.end()) {
                        textVector[wordIndex[row[i]]] = 1.0; // Set the corresponding index to 1
                    }
                    dataRow.insert(dataRow.end(), textVector.begin(), textVector.end());
                }
            }
            // Assuming the last column is the label
            label = row.back();
            data.push_back(dataRow);
            labels.push_back(label_mapping.try_emplace(label, label_mapping.size()).first->second);
        }
    }
};
