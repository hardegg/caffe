#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "FaceClassifier.h"

using namespace caffe;  // NOLINT(build/namespaces)
using namespace cv;
using namespace std;
using std::string;

void FaceDetection(const Mat& img, FaceClassifier& classifier, vector<Rect>& resultRects)
{
    if (!img.data)
        return;
    int windowSize = 12;// The detection window size. In this paper, it can be 12, 24, 48
    int spacing = 4;    // 4x4 spacing for 12x12 detection window
    int F = 40;         // acceptable minimum face size
    float scaling = 1.0*windowSize/F;
    
    // detection step
    float scaleStep = 1.18;
    float minScale = 1.0*F/MIN(img.cols, img.rows);
    vector<float> scales;
    float scale = 1.0;
    while(true) {
        scales.push_back(scale*scaling);
        scale /= scaleStep;
        if (scale < minScale)
            break;
    }
    
    int nTotalRects = 0;
    int nDetected = 0;
    for (int f = 0; f < scales.size(); f++) {
        Mat rszImg;
        cv::resize(img, rszImg, cv::Size(), scales[f], scales[f]);
        for (int r = 0; r + windowSize < rszImg.rows; r+= spacing) {
            for (int c = 0; c + windowSize < rszImg.cols; c+= spacing) {
                Rect rect = Rect(c, r, windowSize, windowSize);
                Mat patch = rszImg(rect);
                vector<Prediction> prdct = classifier.Classify(patch);
                ++nTotalRects;
                if( prdct[0].first == 1) {
                    rect.x /= scales[f];
                    rect.y /= scales[f];
                    rect.width /= scales[f];
                    rect.height /= scales[f];
                    resultRects.push_back(rect);
                    ++nDetected;
                }
            }
        }
    }
    
    cout << "Total sliding windows " << nTotalRects << endl;
    cout << "Detected faces " << nDetected << endl;
    
}

int main(int argc, char** argv) {
    ::google::InitGoogleLogging(argv[0]);
    string imgPath = "/home/fanglin/data/FDDB/originalPics/2002/07/20/big/img_425.jpg";
    string model_file, trained_file;
    model_file = "/home/fanglin/caffe/FaceDetection/models/deploy_detection12.prototxt";
    trained_file = "/home/fanglin/caffe/FaceDetection/models/snapshots/facecascade_detection12_train_iter_90000.caffemodel";
    FaceClassifier classifier(model_file, trained_file);
    Mat img = imread(imgPath);
    vector<Rect> resultRects;
    FaceDetection(img, classifier, resultRects);
    for (int i = 0; i < resultRects.size(); i++) {
        cv::rectangle(img, resultRects[i], CV_RGB(255, 0, 0), 2);
    }
    imshow("img", img);
    cv::waitKey(0);
}
