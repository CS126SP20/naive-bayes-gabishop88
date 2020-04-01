// Copyright (c) 2020 Graham Bishop. All rights reserved.

#ifndef BAYES_MODEL_H_
#define BAYES_MODEL_H_

#include "image.h"
#include <fstream>
#include <iostream>
using std::ifstream;
using std::ofstream;
using std::ostream;
using bayes::Image;


namespace bayes {

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
  double kLaplace = 0.16;

  double total_size();
  double total_images_of_val(int c);

 public:
  Model();

  double FindP(int image_value);
  double FindP(int x, int y, int shade, int image_value);
  Model& Train(Image& image, int val);
  Model& TrainAll(ifstream& images, ifstream& values);
  int classify(Image& image);

  friend ifstream& operator >>(ifstream& input, Model& model);
  friend ofstream& operator <<(ofstream& output, Model& model);
};

}  // namespace bayes

#endif  // BAYES_MODEL_H_
