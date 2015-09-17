#include "FaceClassifier.h"
#include "utilities_common.h"
#include <algorithm>
//#define CPU_ONLY

FaceClassifier::FaceClassifier(const string& model_file,
                       const string& trained_file,
                       const vector<string>& mean_files)
{
    Initialize(model_file, trained_file, mean_files);
}

void FaceClassifier::Initialize(const string& model_file,
                       const string& trained_file,
                       const vector<string>& mean_files) 
{
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  int nInputs = mean_files.size();
  num_resos_ = nInputs;
  input_geometries_.resize(nInputs);
  means_.resize(nInputs);
  num_channels_.resize(nInputs);
  for (int i = 0; i < nInputs; i++) {
    Blob<float>* input_layer = net_->input_blobs()[i];
    num_channels_[i] = input_layer->channels();
    CHECK(num_channels_[i] == 3 || num_channels_[i] == 1)
      << "Input layer 0 should have 1 or 3 channels.";
    input_geometries_[i] = cv::Size(input_layer->width(), input_layer->height());

    /* Load the binaryproto mean file. */
    SetMean(mean_files[i], input_geometries_[i], means_[i]); 
    input_layer->Reshape(1, num_channels_[i],
                           input_geometries_[i].height, input_geometries_[i].width);      
  }

  /* Forward dimension change to all layers. */
  net_->Reshape(); 
}

 FaceClassifier::FaceClassifier(const string& model_file,
             const string& trained_file,
             const string& mean_file)
 {
     vector<string> mean_files(1);
     mean_files[0] = mean_file;
     num_resos_ = 1;
     Initialize(model_file, trained_file, mean_files);
 }

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
  std::vector<std::pair<float, int> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], i));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);
  return result;
}

/* Return the top N predictions. */
std::vector<Prediction> FaceClassifier::Classify(const cv::Mat& img, int N) {
  std::vector<float> output = Predict(img);

  std::vector<int> maxN = Argmax(output, N);
  std::vector<Prediction> predictions;
  for (int i = 0; i < N; ++i) {
    int idx = maxN[i];
    predictions.push_back(std::make_pair(idx, output[idx]));
  }

  return predictions;
}

/* Load the mean file in binaryproto format. */
void FaceClassifier::SetMean(const string& mean_file, cv::Size input_geometry, Mat& outmean) {
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

  /* Convert from BlobProto to Blob<float> */
  Blob<float> mean_blob;
  mean_blob.FromProto(blob_proto);
  CHECK_EQ(mean_blob.channels(), num_channels_[0])
    << "Number of channels of mean file doesn't match input layer.";

  /* The format of the mean file is planar 32-bit float BGR or grayscale. */
  std::vector<cv::Mat> channels;
  float* data = mean_blob.mutable_cpu_data();
  for (int i = 0; i < num_channels_[0]; ++i) {
    /* Extract an individual channel. */
    cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
    channels.push_back(channel);
    data += mean_blob.height() * mean_blob.width();
  }

  /* Merge the separate channels into a single image. */
  cv::Mat mean;
  cv::merge(channels, mean);

  /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
  cv::Scalar channel_mean = cv::mean(mean);
  outmean = cv::Mat(input_geometry, mean.type(), channel_mean);
}

int FaceClassifier::GetLabelDim()
{
    if (net_)
        return net_->output_blobs()[0]->channels();
    else
        return -1;
}

std::vector<float> FaceClassifier::Predict(const cv::Mat& img) {
    // Automatic conversion from element to vector
    return Predict(img);
}

std::vector<float> FaceClassifier::Predict(const vector<cv::Mat>& imgs) {
  
//  Blob<float>* input_layer = net_->input_blobs()[0];
//  input_layer->Reshape(1, num_channels_,
//                       input_geometry_.height, input_geometry_.width);
//  // Forward dimension change to all layers.
//  net_->Reshape();
    
    for (int i = 0; i < num_resos_; i++) {
        Blob<float>* input_layer = net_->input_blobs()[i];
        input_layer->Reshape(imgs.size(), num_channels_[i],
                           input_geometries_[i].height, input_geometries_[i].width);    
    }
    net_->Reshape();

    int dim = net_->output_blobs()[0]->channels();
    vector<float> probs(imgs.size()*dim);
  
    for (int i = 0; i < num_resos_; i++) {
        std::vector<cv::Mat> input_channels;
        input_channels.clear();
        WrapInputLayer(&input_channels, imgs.size(), i);
        Preprocess(imgs, &input_channels, input_geometries_[i], means_[i]);
    }

    net_->ForwardPrefilled();

    /* Copy the output layer to a std::vector */
    Blob<float>* output_layer = net_->output_blobs()[0];
    const float* begin = output_layer->cpu_data();
    const float* end = begin + imgs.size()*output_layer->channels();
    return vector<float>(begin, end);
}
/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void FaceClassifier::WrapInputLayer(std::vector<cv::Mat>* input_channels, int batchSize, int iInput)
{
  Blob<float>* input_layer = net_->input_blobs()[iInput];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels()*batchSize; ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void FaceClassifier::Preprocess(const vector<cv::Mat>& imgs,
                  std::vector<cv::Mat>* input_channels, cv::Size input_geometry, const Mat& meanImage)
{    
     /* Convert the input image to the input image format of the network. */
    for (int i = 0; i < imgs.size(); i++) {
        Mat img = imgs[i];
        cv::Mat sample;
        if (img.channels() == 3 && num_channels_[0] == 1)
          cv::cvtColor(img, sample, CV_BGR2GRAY);
        else if (img.channels() == 4 && num_channels_[0] == 1)
          cv::cvtColor(img, sample, CV_BGRA2GRAY);
        else if (img.channels() == 4 && num_channels_[0] == 3)
          cv::cvtColor(img, sample, CV_BGRA2BGR);
        else if (img.channels() == 1 && num_channels_[0] == 3)
          cv::cvtColor(img, sample, CV_GRAY2BGR);
        else
          sample = img;

        cv::Mat sample_resized;
        if (sample.size() != input_geometry)
          cv::resize(sample, sample_resized, input_geometry);
        else
          sample_resized = sample;

        cv::Mat sample_float;
        if (num_channels_[0] == 3)
          sample_resized.convertTo(sample_float, CV_32FC3);
        else
          sample_resized.convertTo(sample_float, CV_32FC1);

        cv::Mat sample_normalized;
        cv::subtract(sample_float, meanImage, sample_normalized);  

        /* This operation will write the separate BGR planes directly to the
         * input layer of the network because it is wrapped by the cv::Mat
         * objects in input_channels. */
        
        //cv::split(sample_normalized, *input_channels);
        vector<Mat> channels;
        cv::split(sample_normalized, channels);
        for (int j = 0; j < channels.size(); j++)
            channels[j].copyTo((*input_channels)[i*num_channels_[0]+j]);
        //std::copy(channels.begin(), channels.end(), input_channels->begin() + i*num_channels_);

        /*
        CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
              == net_->input_blobs()[0]->cpu_data())
          << "Input channels are not wrapping the input layer of the network.";
        */ 
    }
}