// Copyright 2020 Graham Bishop. All rights reserved.

#include "bayes/model.h"
#include <vector>
#include <sstream>

#include <nlohmann/json.hpp>

using std::vector;
using std::istringstream;

namespace bayes {

Model::Model() = default;

Model& Model::Train(Image& image, int val) {
  for (int i = 0; i < kImageSize; i++) {
    for (int j = 0; j < kImageSize; j++) {
      int shade = image.value_at(i, j);
      training_data_[i][j][val][shade]++;
    }
  }

  return *this;
}

Model& Model::TrainAll(ifstream &images, ifstream &values) {
  Image current;
  int current_val;

  while (!values.eof()) {
    images >> current;
    values >> current_val;

    Train(current, current_val);
  }

  return *this;
}

double Model::total_size() {
  int sum = 0;
  for (int i = 0; i < kNumClasses; i++)
    for (int j = 0; j < kNumShades; j++)
      sum += training_data_[0][0][i][j];

  return sum;
}

double Model::total_images_of_val(int c) {
  int sum = 0;
  for (int i = 0; i < kNumShades; i++)
    sum += training_data_[0][0][c][i];

  return sum;
}

double Model::FindP(int image_value) {
  int size = (int) total_size();
  if (size == 0) {
    return 0;
  }
  return (total_images_of_val(image_value) / size);
}

double Model::FindP(int x, int y, int shade, int image_value) {
  double n = training_data_[x][y][image_value][shade];
  double total = total_images_of_val(image_value);

  if (total == 0) {
    return 0;
  }

  return (kLaplace + n) / (2 * kLaplace + total);
}

int Model::classify(Image &image) {
  vector<double> probs(kNumClasses);
  for (int i = 0; i < kNumClasses; i++) {
    double prob = log(FindP(i));
    for (int x = 0; x < kImageSize; x++) {
      for (int y = 0; y < kImageSize; y++) {
        int shade = image.value_at(x, y);
        prob += log(FindP(x, y, shade, i));
      }
    }

    if (prob == log(0)) {
      probs.at(i) = 0;
    } else {
      probs.at(i) = prob;
    }
  }

  double max_prob = probs.at(0);
  int max_i = 0;
  for (int i = 0; i < probs.size(); i++) {
    if (probs.at(i) != 0) {
      if (max_prob == 0 || probs.at(i) > max_prob) {
        max_prob = probs.at(i);
        max_i = i;
      }
    }
  }

  if (max_prob == 0) {
    return -1;
  }
  return max_i;
}

ifstream &operator >>(ifstream& input, Model& model) {
  input >> model.kLaplace;

  for (int y = 0; y < kImageSize; y++) {
    for (int x = 0; x < kImageSize; x++) {
      for (int c = 0; c < kNumClasses; c++) {
        for (int s = 0; s < kNumShades; s++) {
          input >> model.training_data_[x][y][c][s];
        }
      }
    }
  }

  return input;
}

ofstream &operator <<(ofstream &output, Model &model) {
  output << model.kLaplace << " ";
  for (int y = 0; y < kImageSize; y++) {
    for (int x = 0; x < kImageSize; x++) {
      for (int c = 0; c < kNumClasses; c++) {
        for (int s = 0; s < kNumShades; s++) {
          output << model.training_data_[x][y][c][s] << " ";
        }
      }
    }
  }
  output << std::endl;

  return output;
}

}  // namespace bayes

