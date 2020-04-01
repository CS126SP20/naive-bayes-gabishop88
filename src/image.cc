// Copyright 2020 Graham Bishop. All rights reserved.

#include <bayes/image.h>


namespace bayes {

bool Image::verify() {
  for (auto & pixel_row : pixels_) {
    for (char c : pixel_row) {
      if (!shades.count(c)) {
        return false;
      }
    }
  }
}

Image::Image() = default;

char Image::at(int i, int j) {
  return pixels_[i][j];
}

int Image::value_at(int i, int j) {
  return shades[pixels_[i][j]];
}

ifstream& operator >>(ifstream &input, Image& image) {
  string line;
  for (auto & pixel_row : image.pixels_) {
    if (!std::getline(input, line)) {
      line = "";
    }
    for (int i = 0; i < kImageSize; i++) {
      if (line.size() <= i) {
        pixel_row[i] = ' ';
      } else {
        pixel_row[i] = line[i];
      }
    }
  }
  return input;
}

std::ostream &operator <<(std::ostream& output, Image &image) {
  for (int i = 0; i < kImageSize + 2; i++) {
    output << "#";
  }
  output << std::endl;
  for (int i = 0; i < kImageSize; i++) {
    output << "#";
    for (int j = 0; j < kImageSize; j++) {
      output << image.at(i, j);
    }
    output << "#" << std::endl;
  }
  for (int i = 0; i < kImageSize + 2; i++) {
    output << "#";
  }

  return output;
}

}  // namespace bayes

