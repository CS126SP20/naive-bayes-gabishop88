// Copyright (c) 2020 [Your Name]. All rights reserved.

#include <bayes/classifier.h>
#include <bayes/image.h>
#include <bayes/model.h>
#include <gflags/gflags.h>

#include <string>
#include <cstdlib>
#include <iostream>
using std::fstream;
using namespace bayes;

// TODO(you): Change the code below for your project use case.

DEFINE_string(name, "Clarice", "Your first name");
DEFINE_bool(happy, false, "Whether the greeting is a happy greeting");

int main(int argc, char** argv) {

  const string model_storage =
      R"(C:\Users\gabis\CLionProjects\naive-bayes-gabishop88\data\models.txt)";

  gflags::SetUsageMessage(
      "Greets you with your name. Pass --helpshort for options.");

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_name.empty()) {
    std::cerr << "Please provide a name via the --name flag." << std::endl;
    return EXIT_FAILURE;
  }

  const std::string punctuation = FLAGS_happy ? "!" : ".";

  Model model_one;
  Model model_two;

  TrainModel(model_one, "data/trainingimages", "data/traininglabels");

  ifstream model_input("data/models.txt");
  model_input >> model_two;

  std::cout << "Classifying..." << std::endl;

  vector<int> model_one_results(ClassifyAll(model_one, "data/testimages"));
  vector<int> model_two_results(ClassifyAll(model_two, "data/testimages"));

  std::cout << "Model One results:" << std::endl;
  vector<vector<double>> chart_one =
      VerifyClassifications(model_one_results, "data/testlabels");
  OutputAccuracy(std::cout, chart_one);

  std::cout << "Model Two Results" << std::endl;
  vector<vector<double>> chart_two =
      VerifyClassifications(model_two_results, "data/testlabels");
  OutputAccuracy(std::cout, chart_two);

  std::cout << std::endl <<"Hello, " << FLAGS_name << punctuation << std::endl;
  return EXIT_SUCCESS;
}

/*
 * 0 [1][0][0][0][0][0][0][0][0][0]
 * 1 [0][1][0][0][0][0][0][0][0][0]
 * 2 [0][0][1][0][0][0][0][0][0][0]
 * 3 [0][0][0][1][0][0][0][0][0][0]
 * 4 [0][0][0][0][1][0][0][0][0][0]
 * 5 [0][0][0][0][0][1][0][0][0][0]
 * 6 [0][0][0][0][0][0][1][0][0][0]
 * 7 [0][0][0][0][0][0][0][1][0][0]
 * 8 [0][0][0][0][0][0][0][0][1][0]
 * 9 [0][0][0][0][0][0][0][0][0][1]
 *    0  1  2  3  4  5  6  7  8  9
 *
 * score: x/10 where x is the sum of the diagonal.
 */
