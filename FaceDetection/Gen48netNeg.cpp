#include <opencv2/opencv.hpp>
#include <iostream>
#include <caffe/caffe.hpp>
#include "FaceClassifier.h"
#include "FaceDetector.h"
#include "utilities_common.h"
#include <fstream>


using namespace std;
using namespace cv;

int main(int argc, const char* argv[])
{
    GenerateCalibLabelSet();
    string folderName = "/media/ssd/data/VOC2007/nonPerson";
    string oFolder = "/media/ssd/data/aflw/data/neg48x48";
    string listFolder = "/media/ssd/data/aflw/data";
    
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
    
    
    vector<string> filePaths;
    GetFilePaths(folderName, ".jpg", filePaths);
    vector<Mat> imgs(filePaths.size());
    for (size_t i = 0; i < filePaths.size(); i++) {
        imgs[i] = imread(filePaths[i]);
    }
    cout << imgs.size() << " images!" << endl;
    
    ofstream negTrListPath(string(listFolder+"/neg48_train.txt").c_str());
    ofstream negValListPath(string(listFolder+"/neg48_val.txt").c_str());
    
    long long nTotalWindows = 0;
    long long nDetected = 0;
    for (int i = 0; i < imgs.size(); i++) {
        Mat img = imgs[i];
        vector<Rect> rects;
        vector<float> scores;
        tic();
        int nWs = FaceDetection(img, detector_12, detector_24, calibrator_12, calibrator_24, rects, scores);
                     
        for (int j = 0; j < rects.size(); j++) {
            Mat rsz, patch;
            if (rects[j].x < 0 || rects[j].y < 0 || 
                    rects[j].br().x > img.cols -1 || rects[j].br().y > img.rows -1)
                continue;
            patch = img(rects[j]);

            cv::resize(patch, rsz, cv::Size(48, 48));
            stringstream ss;
            ss << oFolder << "/neg_24x24_" << i << "_" << j << ".bmp";
            imwrite(ss.str(), rsz);
            //if (imgs.size() -i > 500)
                negTrListPath << ss.str() << " " << 0 << endl;
            //else
                //negValListPath << ss.str() << " " << 0 << endl;
        }
        
        for (int j = 0; j < rects.size(); j++) {
            // Expand by 20% vertically            
            cv::rectangle(img, rects[j], CV_RGB(255, 0, 0), 2);            
        }
        nTotalWindows += nWs;
        nDetected += rects.size();
        imshow("img", img);
        
        char c = cv::waitKey(1);
        if (c == 'q')
            break;
    }
    negTrListPath.close();
    negValListPath.close();
    
    cout << "Total negatives: " << nDetected << endl;
    return 0;
}
