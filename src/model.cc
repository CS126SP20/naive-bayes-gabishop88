// Copyright 2020 Graham Bishop. All rights reserved.

#include <vector>
#include <bayes/model.h>
#include <nlohmann/json.hpp>

using std::vector;

namespace bayes {

Model::Model() = default;

Model& Model::train(Image& image, int val) {
  for (int i = 0; i < kImageSize; i++) {
    for (int j = 0; j < kImageSize; j++) {
      int shade = image.value_at(i, j);
      training_data_[i][j][val][shade]++;
    }
  }

  return *this;
}

Model &Model::train_all(ifstream &images, ifstream &values) {
  Image current;
  int current_val;

  while (!values.eof()) {
    images >> current;
    values >> current_val;

    train(current, current_val);
  }

  return *this;
}

double Model::total_size() {
  int sum = 0;
  for (int i = 0; i < kNumClasses; i++) {
    for (int j = 0; j < kNumShades; j++) {
      sum += training_data_[0][0][i][j];
    }
  }

  return sum;
}

double Model::total_images_of_val(int c) {
  int sum = 0;
  for (int i = 0; i < kNumShades; i++)
    sum += training_data_[0][0][c][i];

  return sum;
}

double Model::find_P(int image_value) {
  int size = (int) total_size();
  if (size == 0) {
    return 0;
  }
  return (total_images_of_val(image_value) / size);
}

double Model::find_P(int x, int y, int shade, int image_value) {
  double n = training_data_[x][y][image_value][shade];
  double total = total_images_of_val(image_value);

  return (kLaplace + n) / (2 * kLaplace + total);
}

int Model::classify(Image &image) {
  vector<double> probs(kNumClasses);
  for (int i = 0; i < kNumClasses; i++) {
    double prob = log(find_P(i));
    for (int x = 0; x < kImageSize; x++) {
      for (int y = 0; y < kImageSize; y++) {
        int shade = image.value_at(x, y);
        prob += log(find_P(x, y, shade, i));
      }
    }

    probs.at(i) = prob;
  }

  double max_prob = probs.at(0);
  int max_i = 0;
  for (int i = 0; i < probs.size(); i++) {
    if (probs.at(i) > max_prob) {
      max_prob = probs.at(i);
      max_i = i;
    }
  }

  return max_i;
}

ifstream &operator>>(ifstream& input, Model& model) {

  return input;
}

}  // namespace bayes

