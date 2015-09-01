#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <fstream>

#include "FaceClassifier.h"
#include "FaceDetector.h"
#include "utilities_common.h"
#include "config.pb.h"

using namespace caffe;  // NOLINT(build/namespaces)
using namespace cv;
using namespace std;
using std::string;

int main(int argc, char** argv) {
    ::google::InitGoogleLogging(argv[0]);
        
    
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
    trained_file_c48 = "/home/fanglin/caffe/FaceDetection/models/snapshots/facecascade_calibration48_train_iter_450000.caffemodel";
        
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
  //  modelFiles_calib.push_back(model_file_c48);

    trainedFiles_calib.push_back(trained_file_c12);
    trainedFiles_calib.push_back(trained_file_c24);
    //trainedFiles_calib.push_back(trained_file_c48);

    
    int min_FaceSize; float scaleStep; int spacing;
    min_FaceSize = 32; scaleStep = 1.118; spacing = 4;
    FaceDetector facedetector(min_FaceSize, scaleStep, spacing);
    facedetector.LoadConfigs("/home/fanglin/caffe/FaceDetection/faceConfig.txt");
//    facedetector.SetDetectors(modelFiles_detect, trainedFiles_detect);
//    facedetector.SetCalibrators(modelFiles_calib, trainedFiles_calib);

    

    string baseDir = "/media/ssd/data/FDDB";
    string listFilepath = baseDir + "/FDDB-folds/imgList.txt";
    string detFilepath = baseDir + "/FDDB-folds/detection_d12.txt";
    string annoFile = baseDir + "/FDDB-folds/annotation.txt";
    string imgDir = baseDir + "/originalPics";
    
    ifstream pathin;
    pathin.open(listFilepath.c_str());
    string t;
    
    ofstream of;
    of.open(detFilepath.c_str());
    long long nTotalWindows = 0;
    long long nDetected = 0;
    int nImages = 0;
    while(pathin >> t) {
        tic();
        ++nImages;
        Mat img = imread(imgDir + "/" + t + ".jpg");
        cout << t << endl;
        vector<Rect> rects;
        vector<float> scores;
        int nWs = facedetector.Detect(img, rects, scores);
        
        cout << "Total sliding windows " << nWs << endl;
        cout << "Detected faces " << rects.size() << endl; 
        //int nWs = GetAllWindows(img, rects);
        nTotalWindows += nWs;
        nDetected += rects.size();
        
        for (int i = 0; i < rects.size(); i++) {
            // Expand by 20% vertically
            rects[i].y -= rects[i].height*0.1;
            rects[i].height *= 1.2;
            cv::rectangle(img, rects[i], CV_RGB(255, 0, 0), 2);            
        }
        
        of << t << endl;
        of << rects.size() << endl;
        for (int i = 0; i < rects.size(); i++) {
            of << rects[i].x << " " << rects[i].y << " " << rects[i].width << " "
                    << rects[i].height << " " << scores[i] << endl;
        }

        
        imshow("img", img);
        char c = cv::waitKey(1);
        if (c == 'q')
            break;
        toc();
    }
    
    cout << "Average sliding windows " << nTotalWindows/nImages << endl;
    cout << "Average detected " << nDetected/nImages << endl;
    
    
    of.close();
    pathin.close();
    return 0;
}   


