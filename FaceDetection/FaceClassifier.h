
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
  // For multi-resolution
  explicit FaceClassifier(const string& model_file,
             const string& trained_file,
             const vector<string>& mean_files);
  explicit FaceClassifier(const string& model_file,
             const string& trained_file,
             const string& mean_file);

  std::vector<Prediction> Classify(const cv::Mat& img, int N = 1);
  std::vector<float> Predict(const cv::Mat& img);
  std::vector<float> Predict(const vector<cv::Mat>& imgs);
  void ExtractFeature(const vector<cv::Mat>& imgs, const string& featName, vector<vector<float> >& features);
  int GetLabelDim();

 private:
  void Initialize(const string& model_file,
             const string& trained_file,
             const vector<string>& mean_files);
  void SetMean(const string& mean_file, cv::Size input_geometry, Mat& outmean);
  void WrapInputLayer(std::vector<cv::Mat>* input_channels, int batchSize = 1, int iInput = 0);
  void Preprocess(const vector<cv::Mat>& imgs,
                  std::vector<cv::Mat>* input_channels, cv::Size input_geometry, const Mat& meanImage);      

 private:
  shared_ptr<Net<float> > net_;
  // support multi resolution when using multi-resolution
  vector<cv::Mat> means_; 
  vector<cv::Size> input_geometries_;
  vector<int> num_channels_;
  int   num_resos_;
};

#endif	/* FACECLASSIFIER_H */

