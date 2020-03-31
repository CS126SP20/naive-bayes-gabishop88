// Copyright 2020 Graham Bishop. All rights reserved.

#include "bayes/classifier.h"
#include <bayes/image.h>
#include <bayes/model.h>
#include <nlohmann/json.hpp>
#include <iomanip>

using namespace bayes;

namespace bayes {

Model& TrainModel(Model& untrained, string images, string labels) {
  ifstream training_images(images);
  ifstream training_labels(labels);

  untrained.train_all(training_images, training_labels);

  training_images.close();
  training_labels.close();

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

  return classifications;
}

vector<vector<double>> VerifyClassifications(vector<int> &computed,
                                             string path_to_labels) {
  vector<vector<double>> accuracy(kNumClasses, vector<double>(kNumClasses));
  ifstream labels(path_to_labels);

  int label;
  for (int i = 0; !labels.eof(); i++) {
    labels >> label;
    accuracy.at(label).at(computed.at(i)) += 1.0;
  }

  for (vector<double>& row : accuracy) {
    double row_sum = 0.0;
    for (double val : row)
      row_sum += val;

    if (row_sum == 0.0) {
      continue;
    }

    for (double &val : row)
      val /= row_sum;
  }

  return accuracy;
}

/*
 * do std::cout << std::fixed << std::setprecision(2);
 *
 * 0 [1.00]0.00|0.00|0.00|0.00|0.00|0.00|0.00|0.00|0.00|
 * 1 |0.00[1.00]0.00|0.00|0.00|0.00|0.00|0.00|0.00|0.00|
 * 2 |0.00|0.00[1.00]0.00|0.00|0.00|0.00|0.00|0.00|0.00|
 * 3 |0.00|0.00|0.00[1.00]0.00|0.00|0.00|0.00|0.00|0.00|
 * 4 |0.00|0.00|0.00|0.00[1.00]0.00|0.00|0.00|0.00|0.00|
 * 5 |0.00|0.00|0.00|0.00|0.00[1.00]0.00|0.00|0.00|0.00|
 * 6 |0.00|0.00|0.00|0.00|0.00|0.00[1.00]0.00|0.00|0.00|
 * 7 |0.00|0.00|0.00|0.00|0.00|0.00|0.00[1.00]0.00|0.00|
 * 8 |0.00|0.00|0.00|0.00|0.00|0.00|0.00|0.00[1.00]0.00|
 * 9 |0.00|0.00|0.00|0.00|0.00|0.00|0.00|0.00|0.00[1.00]
 *     0    1    2    3    4    5    6    7    8    9
 *                  Percent Accuracy
 */

void OutputAccuracy(ostream& output, vector<vector<double>> accuracy) {
  std::streamsize output_precision = output.precision();
  double diagonal_sum = 0;

  output << std::fixed << std::setprecision(2);
  for (int i = 0; i < accuracy.size(); i++) {
    output << std::setprecision(0) << i << " " << std::setprecision(2);
    for (int j = 0; j < accuracy.at(i).size(); j++) {

      if (i == j) {
        diagonal_sum += accuracy.at(i).at(j);
        output << "[";
      } else if (j == 0) {
        output << "|";
      }

      output << accuracy.at(i).at(j);

      if (i == j) {
        output << "]";
      } else if (j != i - 1) {
        output << "|";
      }
    }
    if (i != accuracy.size() - 1) {
      output << std::endl;
    } else {
      output << std::endl
      <<"    0    1    2    3    4    5    6    7    8    9"
      << std::endl << "                     Accuracy Chart" << std::endl;

      output << "Average Accuracy: " << diagonal_sum / kNumClasses << std::endl;
    }
  }

  output.unsetf(std::ios_base::floatfield); // removes fixed precision size
  output.precision(output_precision);
}

}  // namespace bayes

