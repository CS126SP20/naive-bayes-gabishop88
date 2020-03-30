// Copyright (c) 2020 [Your Name]. All rights reserved.

#include <bayes/classifier.h>
#include <bayes/image.h>
#include <bayes/model.h>
#include <gflags/gflags.h>

#include <string>
#include <cstdlib>
#include <iostream>

// TODO(you): Change the code below for your project use case.

DEFINE_string(name, "Clarice", "Your first name");
DEFINE_bool(happy, false, "Whether the greeting is a happy greeting");

int main(int argc, char** argv) {
  gflags::SetUsageMessage(
      "Greets you with your name. Pass --helpshort for options.");

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_name.empty()) {
    std::cerr << "Please provide a name via the --name flag." << std::endl;
    return EXIT_FAILURE;
  }

  const std::string punctuation = FLAGS_happy ? "!" : ".";

  ifstream training_images("data/trainingimages");
  ifstream training_values("data/traininglabels");
  bayes::Model model;
  model.train_all(training_images, training_values);

  ifstream tests("data/testimages");
  bayes::Image test;

  std::cout << model.classify(test) << std::endl;

  std::cout << "Hello, " << FLAGS_name << punctuation << std::endl;
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
 */
