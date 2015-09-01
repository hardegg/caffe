#include <opencv2/opencv.hpp>
#include <iostream>
#include <caffe/caffe.hpp>
#include "FaceClassifier.h"
#include "FaceDetector.h"
#include "utilities_common.h"
#include <fstream>
#include "ThreadManager/ThreadManager.h"


using namespace std;
using namespace cv;

boost::mutex io_mutex;

long g_nImages = 0;

using namespace BoatDetection;

void GenerateNegSamples(const Mat& img, FaceDetector& facedetector, 
        ofstream& negTrListPath, ofstream& negValListPath, const string& oFolder)
{
    vector<Rect> rects;
    vector<float> scores;
    tic();
    int nWs = facedetector.Detect(img, rects, scores);         
    static int i = 0;
 
    for (int j = 0; j < rects.size(); j++) {
        Mat rsz, patch;
        if (rects[j].x < 0 || rects[j].y < 0 || 
                rects[j].br().x > img.cols -1 || rects[j].br().y > img.rows -1)
            continue;
        patch = img(rects[j]);

        cv::resize(patch, rsz, cv::Size(48, 48));
        stringstream ss;
        ss << oFolder << "/neg_48x48_" << i << "_" << j << ".bmp";     
 
        //if (imgs.size() -i > 500)
        boost::unique_lock<boost::mutex> scoped_lock(io_mutex);
        negTrListPath << ss.str() << " " << 0 << endl;
        imwrite(ss.str(), rsz);        
            //else
                //negValListPath << ss.str() << " " << 0 << endl;
    }
    
    boost::unique_lock<boost::mutex> scoped_lock(io_mutex);
    ++i;
    cout << "Total sliding windows " << nWs << endl;
    cout << "Detected FPs " << rects.size() << endl;
    cout << i << " / " << g_nImages << " has been finished.\n";
}

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
  
    vector<string> modelFiles_detect, trainedFiles_detect;
    vector<string> modelFiles_calib, trainedFiles_calib;
    
    modelFiles_detect.push_back(model_file_d12);
    modelFiles_detect.push_back(model_file_d24);  
    trainedFiles_detect.push_back(trained_file_d12);
    trainedFiles_detect.push_back(trained_file_d24);
    
    modelFiles_calib.push_back(model_file_c12);
    //modelFiles_calib.push_back(model_file_c24);  
    trainedFiles_calib.push_back(trained_file_c12);
    //trainedFiles_calib.push_back(trained_file_c24);
    
    int min_FaceSize; float scaleStep; int spacing;
    min_FaceSize = 24; scaleStep = 1.1; spacing = 2;
    FaceDetector facedetector(min_FaceSize, scaleStep, spacing);
    facedetector.SetDetectors(modelFiles_detect, trainedFiles_detect);
    facedetector.SetCalibrators(modelFiles_calib, trainedFiles_calib);
    
    
    vector<string> filePaths;
    
    /*
    folderName = "/media/ssd/data/VOC2007/nonfaces_pico";
    GetFilePaths(folderName, ".JPG", filePaths);
    vector<Mat> imgs(filePaths.size());
    
    for (size_t i = 0; i < filePaths.size(); i++) {
        imgs[i] = imread(filePaths[i]);
    }
    cout << imgs.size() << " images!" << endl;
    */

    folderName = "/media/ssd/data/VOC2007/nonPerson";
    //folderName = "/media/ssd/data/WuJX/SkinColor/nonFaces";

    GetFilePaths(folderName, ".jpg", filePaths);
    vector<Mat> imgs;    
    int oldSize = imgs.size();
    //imgs.resize(oldSize+filePaths.size());
    for (size_t i = 0; i < filePaths.size(); i++) {
        Mat img = imread(filePaths[i]);
        if (!img.empty())
            imgs.push_back(img);
    }
    cout << imgs.size() << " images!" << endl;
    
    ofstream negTrListPath(string(listFolder+"/neg48_train.txt").c_str());
    ofstream negValListPath(string(listFolder+"/neg48_val.txt").c_str());
    
    long long nTotalWindows = 0;
    long long nDetected = 0;
    g_nImages = imgs.size();
    
    ThreadManager threadManager;
    tic();
    for (int i = 0; i < g_nImages; i++) {
        toc();
        tic();
        Mat img = imgs[i];
        if(!img.data)
            continue;
        vector<Rect> rects;
        vector<float> scores;
        int nWs = facedetector.Detect(img, rects, scores);         
        
        for (int j = 0; j < rects.size(); j++) {
            Mat rsz, patch;
            if (rects[j].x < 0 || rects[j].y < 0 || 
                    rects[j].br().x > img.cols -1 || rects[j].br().y > img.rows -1)
                continue;
            patch = img(rects[j]);

            cv::resize(patch, rsz, cv::Size(48, 48));
            stringstream ss;
            ss << oFolder << "/neg_48x48_" << i << "_" << j << ".bmp";
            imwrite(ss.str(), rsz);
            
            //if (imgs.size() -i > 500)
            negTrListPath << ss.str() << " " << 0 << endl;
            //else
                //negValListPath << ss.str() << " " << 0 << endl;
        }       
        
        cout << "Total sliding windows " << nWs << endl;
        cout << "Detected FPs " << rects.size() << endl;
        cout << i << " / " << imgs.size() << " has been finished.\n";
        
        for (int i = 0; i < rects.size(); i++) {            
            cv::rectangle(img, rects[i], CV_RGB(255, 0, 0), 2);            
        }
        
        nTotalWindows += nWs;
        nDetected += rects.size();
        imshow("img", img);
        
        char c = cv::waitKey(1);
        if (c == 'q')
            break;
        /*
        threadManager.LaunchNewThread(
            boost::bind(
                &GenerateNegSamples, boost::ref(imgs[i]), boost::ref(facedetector), 
                boost::ref(negTrListPath), boost::ref(negValListPath), boost::ref(oFolder))
        );
         */
        
    }
    //threadManager.JoinAll();
    negTrListPath.close();
    negValListPath.close();
    
    cout << "Total negatives: " << nDetected << endl;
    return 0;
}
