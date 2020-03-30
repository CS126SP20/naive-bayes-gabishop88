// Copyright (c) 2020 Graham Bishop. All rights reserved.

#ifndef BAYES_MODEL_H_
#define BAYES_MODEL_H_

#include "image.h"
#include <cstdlib>

namespace bayes {

// 0-9 inclusive.
constexpr size_t kNumClasses = 10;
// Shaded or not shaded.
constexpr size_t kNumShades = 2;

/**
 * Represents a Naive Bayes classification model for determining the
 * likelihood that an individual pixel for an individual class is
 * white or black.
 */
class Model {
  /**
   * [a][b][c][d]:
   * Gets the number of training images with color d of classification c at a, b
   */
  int training_data_[kImageSize][kImageSize][kNumClasses][kNumShades] = { 0 };
  double kLaplace = 1.0;

  double total_size();
  double total_images_of_val(int c);

 public:
  Model();

  Model& train(Image& image, int val);
  Model& train_all(ifstream& images, ifstream& values);

  double find_P(int image_value);
  double find_P(int x, int y, int shade, int image_value);

  int classify(Image& image);

  friend ifstream& operator >>(ifstream& input, Model& model);
};

}  // namespace bayes

#endif  // BAYES_MODEL_H_
