// Copyright 2020 Graham Bishop. All rights reserved.

#include "bayes/classifier.h"
#include <bayes/image.h>
#include <bayes/model.h>
#include <nlohmann/json.hpp>
#include <iomanip>

using namespace bayes;

namespace bayes {

Model& TrainModel(Model& untrained, string image_path, string label_path) {
  ifstream images(image_path);
  ifstream labels(label_path);

  if (!images.good() || !labels.good()) {
    return untrained;
  }

  untrained.TrainAll(images, labels);

  images.close();
  labels.close();

  return untrained;
}

vector<int> ClassifyAll(Model& model, string images) {
  ifstream image_stream(images);
  vector<int> classifications;

  if (image_stream.good()) {
    Image image;
    while(!image_stream.eof()) {
      image_stream >> image;
      classifications.push_back(model.classify(image));
    }
  }

  image_stream.close();

  return classifications;
}

vector<vector<double>> VerifyClassifications(vector<int> &computed,
                                             string path_to_labels) {
  vector<vector<double>> accuracy(kNumClasses, vector<double>(kNumClasses));
  ifstream labels(path_to_labels);

  int label;
  for (int i = 0; !labels.eof(); i++) {
    labels >> label;

    //Making sure there are no problems with labels or computed array.
    if (accuracy.size() <= label || label < 0)
      continue;
    if (computed.size() <= i || computed.at(i) == -1)
      continue;

    accuracy.at(label).at(computed.at(i)) += 1.0;
  }

  labels.close();

  for (vector<double>& row : accuracy) {
    double row_sum = 0.0;
    for (double val : row)
      row_sum += val;

    if (row_sum == 0.0) {
      continue;
    }

    for (double &val : row)
      if (row_sum != 0)
        val /= row_sum;
  }

  return accuracy;
}

double OutputAccuracy(ostream& output, vector<vector<double>> accuracy,
                      bool print) {
  double diagonal_sum = 0;
  std::streamsize output_precision = output.precision();

  if (print) output << std::fixed << std::setprecision(2);
  for (int i = 0; i < accuracy.size(); i++) {
    if (print)
      output << std::setprecision(0) << i << " " << std::setprecision(2);
    for (int j = 0; j < accuracy.at(i).size(); j++) {

      if (i == j) {
        diagonal_sum += accuracy.at(i).at(j);
        if (print) output << "[";
      } else if (j == 0 && print) {
        output << "|";
      }

      if (print) output << accuracy.at(i).at(j);

      if (print) {
        if (i == j) {
          output << "]";
        } else if (j != i - 1) {
          output << "|";
        }
      }
    }
    if (i != accuracy.size() - 1 && print) {
      output << std::endl;
    } else if (print) {
      output << std::endl << "    0    1    2    3    4    5    6    7    8    9"
             << std::endl << "                     Accuracy Chart" << std::endl;

      output << "Average Accuracy: " << diagonal_sum / kNumClasses << std::endl;
    }
  }

  if (print) {
    output.unsetf(std::ios_base::floatfield); // removes fixed precision size
    output.precision(output_precision);
  }

  return diagonal_sum / kNumClasses;
}

}  // namespace bayes

