// Copyright (c) 2020 Graham Bishop. All rights reserved.

#include <gflags/gflags.h>
#include <string>
#include <iostream>

#include "bayes/classifier.h"
#include "bayes/model.h"

using std::fstream;

DEFINE_string(images, "", "File path to images.");
DEFINE_string(labels, "", "File Path to image labels.");
DEFINE_string(usemodel, "", "File path to an existing model in a text file.");
DEFINE_string(savemodel, "", "File path to a .txt where model should be saved"
                             " (absolute path required).");
DEFINE_bool(train, false, "Whether the data should be used for training.");
DEFINE_bool(verbose, false, "Whether to print the uncertainty chart, or just "
                            "the average percent of correct image classifications.");

int main(int argc, char** argv) {

  gflags::SetUsageMessage(
      "Trains a model, or uses a trained one to identify a 28x28 pixel image "
      "of a hand-drawn digit (0-9 inclusive).");

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_images.empty()) {
    std::cerr << "Images must be provided via the --images [filename] flag."
    << std::endl;
    return EXIT_FAILURE;
  }

  if (FLAGS_train && FLAGS_labels.empty()) {
    std::cerr << "In order to train a model, labels must be provided via "
                 "--labels [filename]." << std::endl;
    return EXIT_FAILURE;
  }

  if (FLAGS_verbose && (FLAGS_train || FLAGS_labels.empty()))
    std::cerr << "verbose does not apply, as no validation will be done."
    << std::endl;

  bayes::Model model;

  if (!FLAGS_usemodel.empty()) {
    ifstream get_model(FLAGS_usemodel);
    if (get_model.good()) {
      get_model >> model;
    } else {
      std::cerr << "Could not load a model from the given file." << std::endl;
      return EXIT_FAILURE;
    }
  }

  if (FLAGS_train) {
    std::cout << "Training" << std::endl;
    TrainModel(model, FLAGS_images, FLAGS_labels);
  } else {
    vector<int> results = ClassifyAll(model, FLAGS_images);

    if (!FLAGS_labels.empty()) {
      double avg = bayes::OutputAccuracy(std::cout,
          bayes::VerifyClassifications(results, FLAGS_labels), FLAGS_verbose);
    } else {
      std::cout << "Classification Results: " << std::endl;
      for (int i = 0; i < results.size(); i++)
        std::cout << i << ": " << results[i] << std::endl;
    }
  }

  std::cout << (FLAGS_train ? "Training" : "Validation") << " Complete" <<std::endl;

  if (!FLAGS_savemodel.empty()) {
    ofstream save_model(FLAGS_savemodel, std::ios::app);
    if (save_model.good()) {
      save_model << model;
    } else {
      std::cerr << "Could not save model to file." << std::endl;
      return EXIT_FAILURE;
    }
  }

  return EXIT_SUCCESS;
}
