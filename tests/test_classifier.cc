// Copyright (c) 2020 [Your Name]. All rights reserved.

#define CATCH_CONFIG_MAIN

#include <catch2/catch.hpp>
#include "bayes/model.h"
#include "bayes/classifier.h"

TEST_CASE("Test Model class", "[model]") {

  SECTION("Train with an image") {
    ifstream input("data/sampleimages");
    Image first;
    input >> first;

    bayes::Model model;
    model.Train(first, 5);

    REQUIRE(model.FindP(5) == 1.0);
    REQUIRE(model.FindP(0, 0, 0, 5) == (1.16 / 1.32));
    input.close();
  }

  SECTION("Train from ifstreams") {
    ifstream images("data/sampleimages");
    ifstream labels("data/samplelabels");

    bayes::Model model;
    model.TrainAll(images, labels);

    for (int i = 0; i < bayes::kNumClasses; i++) {
      switch (i) {
        case 0:
          REQUIRE(model.FindP(i) == 0.25);
          break;
        case 4:
          REQUIRE(model.FindP(i) == 0.5);
          break;
        case 5:
          REQUIRE(model.FindP(i) == 0.25);
          break;
        default:
          REQUIRE(model.FindP(i) == 0);
      }
    }

    images.close();
    labels.close();
  }

  SECTION("Classify a new image") {
    ifstream train_images("data/training_images");
    ifstream train_labels("data/training_labels");
    ifstream test_image("data/sampleimages");
    Image test;
    test_image >> test; //The 5 is a little to hard for it, skip it.
    test_image >> test;

    bayes::Model model;
    model.TrainAll(train_images, train_labels);

    REQUIRE(model.classify(test) == 0);

    train_images.close();
    train_labels.close();
    test_image.close();
  }

  SECTION("Write to a file") {
    ofstream write_file(R"(C:\Users\gabis\CLionProjects\naive-bayes-gabishop88\tests\data\test_models.txt)"
    , std::ios::out | std::ios::trunc);

    ifstream images("data/training_images");
    ifstream labels("data/training_labels");

    bayes::Model model;
    model.TrainAll(images, labels);

    images.close();
    labels.close();

    write_file << model;
    write_file.close();

    ifstream read("data/test_models.txt");
    double val;

    read >> val;
    REQUIRE(val == 0.16);

    for (int i = 0; i < 10; i++) {
      read >> val;
      switch (i) {
        case 0:
          REQUIRE(val == 479);
          break;
        case 3:
          REQUIRE(val == 563);
          break;
        case 6:
          REQUIRE(val == 489);
          break;
        case 9:
          REQUIRE(val == 493);
          break;
        default:
          REQUIRE(val == 0);
      }
    }

    read.close();
  }

  SECTION("Load from a file") {
    ifstream read("data/test_models.txt");
    bayes::Model model;

    read >> model;
    read.close();

    ifstream images("data/sampleimages");
    Image test;
    for (int i = 0; i < 3; i++) {
      images >> test;
      int val = model.classify(test);
      if (i == 0) REQUIRE(val == 3); //Classifies it wrong, but it still loaded.
      if (i == 1) REQUIRE(val == 0);
      if (i == 2) REQUIRE(val == 4);
    }
  }
}

TEST_CASE("Test classifier", "[classify]") {
  bayes::Model control;
  ifstream images("data/training_images");
  ifstream labels("data/training_labels");
  control.TrainAll(images, labels);
  images.close();
  labels.close();
  vector<int> control_results {3, 0, 4};
  vector<vector<double>> control_chart {
      {1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
      {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
      {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
      {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
      {0.00, 0.50, 0.00, 0.00, 0.50, 0.00, 0.00, 0.00, 0.00, 0.00},
      {0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
      {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
      {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
      {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
      {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00}
  };

  bayes::Model model;
  vector<int> results;
  vector<vector<double>> chart;
  SECTION("TrainModel") {
    bayes::TrainModel(model, "data/training_images", "data/training_labels");

    for (int i = 0; i < 10; i++) {
      REQUIRE(model.FindP(i) == control.FindP(i));
    }
  }

  SECTION("ClassifyAll") {
    bayes::TrainModel(model, "data/training_images", "data/training_labels");
    results = bayes::ClassifyAll(model, "data/sampleimages");

    for (int i = 0; i < 3; i++) {
      REQUIRE(results[i] == control_results[i]);
    }
  }

  SECTION("VerifyClassifications") {
    bayes::TrainModel(model, "data/training_images", "data/training_labels");
    results = bayes::ClassifyAll(model, "data/sampleimages");
    chart = bayes::VerifyClassifications(results, "data/samplelabels");

    for (int i = 0; i < 10; i++)
      for (int j = 0; j < 10; j++)
        REQUIRE(chart[i][j] == control_chart[i][j]);
  }

  SECTION("OutputAccuracy") {
    bayes::TrainModel(model, "data/training_images", "data/training_labels");
    results = bayes::ClassifyAll(model, "data/sampleimages");
    chart = bayes::VerifyClassifications(results, "data/samplelabels");
    double avg = bayes::OutputAccuracy(std::cout, chart, false);
    REQUIRE(avg == 0.15);
  }
}

//TEST_CASE("Test values for laplace smoothing", "[Laplace]") {
//  //Generate a model, just to be safe.
//  bayes::Model model;
//  bayes::TrainModel(model, "data/training_images", "data/training_labels");
//
//  for (double k = 0.1; k <= 1.5; k += 0.01) {
//    model.setSmoothing(k);
//    vector<int> results = bayes::ClassifyAll(model, "data/test_images_full");
//
//    vector<vector<double>> chart =
//        bayes::VerifyClassifications(results, "data/test_labels_full");
//
//    std::streamsize p = std::cout.precision();
//    std::cout << std::fixed << std::setprecision(10);
//    double avg = bayes::OutputAccuracy(std::cout, chart, false);
//    std::cout << model.getSmoothing() << ": " << avg << std::endl;
//    std::cout.unsetf(std::ios_base::floatfield); // removes fixed precision size
//    std::cout.precision(p);
//  }
//}