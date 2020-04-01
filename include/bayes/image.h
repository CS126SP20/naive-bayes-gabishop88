// Copyright (c) 2020 Graham Bishop. All rights reserved.

#ifndef BAYES_IMAGE_H_
#define BAYES_IMAGE_H_

#include <fstream>
#include <iostream>
#include <string>
#include <map>
using std::ifstream;
using std::string;
using std::map;

namespace bayes {

//pixels on each side of a square image.
constexpr size_t kImageSize = 28;
// 0-9 inclusive.
constexpr size_t kNumClasses = 10;
// Shaded or not shaded.
constexpr size_t kNumShades = 3;

class Image {
  map<char, int> shades = {
      {' ', 0},
      {'+', 1},
      {'#', 2}
  };
  char pixels_[kImageSize][kImageSize] = { ' ' };

  bool verify();

 public:
  Image();

  char at(int i, int j);
  int value_at(int i, int j);

  friend ifstream& operator >> (ifstream& input, Image& image);
  friend std::ostream& operator << (std::ostream& ostream, Image& image);
};

}  // namespace bayes

#endif  // BAYES_IMAGE_H_
