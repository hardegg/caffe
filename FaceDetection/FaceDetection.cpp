#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "FaceDetector.h"
#include "utilities_common.h"

using namespace caffe;  // NOLINT(build/namespaces)
using namespace cv;
using namespace std;
using std::string;

long g_nWindows;
long g_nDetectedWindows;

int main(int argc, char** argv) {
    ::google::InitGoogleLogging(argv[0]);
    GenerateCalibLabelSet();
    Caffe::set_mode(Caffe::CPU);
    // Get files from FDDB
    string folderPath = "/home/fanglin/data/AFW";
    vector<string> imgPaths;
    GetFilePaths(folderPath, ".jpg", imgPaths);
    
    string outputFolder = "output_calib";
    
    string model_file_d12, trained_file_d12;
    model_file_d12 = "/home/fanglin/caffe/FaceDetection/models/deploy_detection12.prototxt";
    trained_file_d12 = "/home/fanglin/caffe/FaceDetection/models/snapshots/facecascade_detection12_train_iter_298000.caffemodel";
    string model_file_c12, trained_file_c12;
    model_file_c12 = "/home/fanglin/caffe/FaceDetection/models/deploy_calibration12.prototxt";
    trained_file_c12 = "/home/fanglin/caffe/FaceDetection/models/snapshots/facecascade_calibration12_train_iter_410000.caffemodel";
    string model_file_d24, trained_file_d24;
    model_file_d24 = "/home/fanglin/caffe/FaceDetection/models/deploy_detection24.prototxt";
    trained_file_d24 = "/home/fanglin/caffe/FaceDetection/models/snapshots/facecascade_detection24_train_iter_500000.caffemodel";
    string model_file_c24, trained_file_c24;
    model_file_c24 = "/home/fanglin/caffe/FaceDetection/models/deploy_calibration24.prototxt";
    trained_file_c24 = "/home/fanglin/caffe/FaceDetection/models/snapshots/facecascade_calibration24_train_iter_450000.caffemodel";
    
    FaceClassifier detector_12(model_file_d12, trained_file_d12);    
    FaceClassifier calibrator_12(model_file_c12, trained_file_c12);
    FaceClassifier detector_24(model_file_d24, trained_file_d24);
    FaceClassifier calibrator_24(model_file_c24, trained_file_c24);
    
    
    g_nWindows = 0;
    g_nDetectedWindows = 0;
    
    for (int i = 0; i < imgPaths.size(); i++) {
        string imgPath = imgPaths[i];
        Mat img = imread(imgPath);
        vector<Rect> resultRects;
        vector<float> scores;
        tic();
        FaceDetection(img, detector_12, detector_24, calibrator_12, calibrator_24, resultRects, scores);
        toc();
        for (int i = 0; i < resultRects.size(); i++) {
            cv::rectangle(img, resultRects[i], CV_RGB(255, 0, 0), 2);
        }
        
        stringstream ss;
        ss << outputFolder << "/" << basename(imgPath.c_str());
        imwrite(ss.str(), img);
        imshow("img", img);
        char key = cv::waitKey(1);
        if (key == 'q')
            break;
    }
    
    cout << "Average total windows " << g_nWindows*1.0/imgPaths.size() << endl;
    cout << "Average detected windows " << g_nDetectedWindows/imgPaths.size() << endl;
}
