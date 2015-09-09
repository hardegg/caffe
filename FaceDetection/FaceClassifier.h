
/* 
 * File:   FaceClassifier.h
 * Author: fanglin
 *
 * Created on August 1, 2015, 10:39 AM
 */
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifndef FACECLASSIFIER_H
#define	FACECLASSIFIER_H

using namespace caffe;  // NOLINT(build/namespaces)
using namespace cv;
using namespace std;
using std::string;

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<int, float> Prediction;

class FaceClassifier {
 public:
  FaceClassifier(const string& model_file,
             const string& trained_file,
             const string& mean_file = "");

  std::vector<Prediction> Classify(const cv::Mat& img, int N = 1);
  std::vector<float> Predict(const cv::Mat& img);
  std::vector<float> Predict(const vector<cv::Mat>& imgs);

 private:
  void SetMean(const string& mean_file);
  
  /* In case of now mean image. Set mean image to zero. */
  void SetMean(void); 

 

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);
  void WrapInputLayer(std::vector<cv::Mat>* input_channels, int batchSize);
  void Preprocess(const vector<cv::Mat>& imgs,
                  std::vector<cv::Mat>* input_channels);    
  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_; 
};

#endif	/* FACECLASSIFIER_H */

