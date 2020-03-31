// Copyright (c) 2020 Graham Bishop. All rights reserved.

#ifndef BAYES_IMAGE_H_
#define BAYES_IMAGE_H_

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
using std::ifstream;
using std::string;

namespace bayes {

constexpr size_t kImageSize = 28;

//todo: make sure image is valid

class Image {
  char pixels_[kImageSize][kImageSize] = { ' ' };

 public:
  Image();

  char at(int i, int j);
  int value_at(int i, int j);

  friend ifstream& operator >> (ifstream& input, Image& image);
  friend std::ostream& operator << (std::ostream& ostream, Image& image);
};

}  // namespace bayes

#endif  // BAYES_IMAGE_H_
