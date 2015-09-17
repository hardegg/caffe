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
    
    int min_FaceSize; float scaleStep; int spacing;
    min_FaceSize = 28; scaleStep = 1.1; spacing = 4;
    FaceDetector facedetector(min_FaceSize, scaleStep, spacing);
    facedetector.LoadConfigs("/home/fanglin/caffe/FaceDetection/faceConfig_2nd.txt");
    
    
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

    vector<string> folderNames;
    folderNames.push_back("/media/ssd/data/VOC2007/nonPerson");
    folderNames.push_back("/media/ssd/data/WuJX/SkinColor/nonFaces");
    folderNames.push_back("/media/ssd/data/WuJX/SkinColor/personHeadMasked");
    
    vector<Mat> imgs;    
    for (int k = 0; k < folderNames.size(); k++) {
        folderName = folderNames[k];
        GetFilePaths(folderName, ".jpg|.JPG", filePaths);       
        for (size_t i = 0; i < filePaths.size(); i++) {
            Mat img = imread(filePaths[i]);
            if (!img.empty())
                imgs.push_back(img);
        }
    }
    cout << imgs.size() << " images!" << endl;
    
    ofstream negTrListPath(string(listFolder+"/neg48_train.txt").c_str());
    ofstream negValListPath(string(listFolder+"/neg48_val.txt").c_str());
    
    long long nTotalWindows = 0;
    long long nDetected = 0;
    g_nImages = imgs.size();
    
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
