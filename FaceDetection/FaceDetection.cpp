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
    
    string model_file_d48, trained_file_d48;
    model_file_d48 = "/home/fanglin/caffe/FaceDetection/models/deploy_detection48.prototxt";
    trained_file_d48 = "/home/fanglin/caffe/FaceDetection/models/snapshots/facecascade_detection48_train_iter_500000.caffemodel";
    string model_file_c48, trained_file_c48;
    model_file_c48 = "/home/fanglin/caffe/FaceDetection/models/deploy_calibration48.prototxt";
    trained_file_c48 = "/home/fanglin/caffe/FaceDetection/models/snapshots/facecascade_calibration48_train_iter_250000.caffemodel";
        
    vector<string> modelFiles_detect, trainedFiles_detect;
    vector<string> modelFiles_calib, trainedFiles_calib;
    
    modelFiles_detect.push_back(model_file_d12);
    modelFiles_detect.push_back(model_file_d24);
    modelFiles_detect.push_back(model_file_d48);

    trainedFiles_detect.push_back(trained_file_d12);
    trainedFiles_detect.push_back(trained_file_d24);
    trainedFiles_detect.push_back(trained_file_d48);

    
    modelFiles_calib.push_back(model_file_c12);
    modelFiles_calib.push_back(model_file_c24);
    //modelFiles_calib.push_back(model_file_c48);

    trainedFiles_calib.push_back(trained_file_c12);
    trainedFiles_calib.push_back(trained_file_c24);
    //trainedFiles_calib.push_back(trained_file_c48);

    
    
    FaceDetector facedetector(80, 1.414, 4);
    facedetector.SetDetectors(modelFiles_detect, trainedFiles_detect);
    facedetector.SetCalibrators(modelFiles_calib, trainedFiles_calib);
    
    
    g_nWindows = 0;
    g_nDetectedWindows = 0;
    
    for (int i = 0; i < imgPaths.size(); i++) {
        string imgPath = imgPaths[i];
        Mat img = imread(imgPath);
        vector<Rect> resultRects;
        vector<float> scores;
        tic();
        facedetector.Detect(img, resultRects, scores);
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
