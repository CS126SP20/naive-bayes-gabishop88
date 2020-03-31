// Copyright (c) 2020 Graham Bishop. All rights reserved.

#ifndef BAYES_CLASSIFIER_H_
#define BAYES_CLASSIFIER_H_

#include "model.h"
#include <vector>

using std::vector;

namespace bayes {

Model& TrainModel(Model& untrained, string images, string labels);
vector<int> ClassifyAll(Model& model, string images);
vector<vector<double>> VerifyClassifications(vector<int>& computed,
    string path_to_labels);
void OutputAccuracy(ostream& output, vector<vector<double>> accuracy);

}  // namespace bayes

#endif  // BAYES_CLASSIFIER_H_
