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
    string oFolder = "/media/ssd/data/aflw/data/neg24x24";
    string listFolder = "/media/ssd/data/aflw/data";
    
    int min_FaceSize; float scaleStep; int spacing;
    min_FaceSize = 40; scaleStep = 1.18; spacing = 4;
    FaceDetector facedetector(min_FaceSize, scaleStep, spacing);
    facedetector.LoadConfigs("/home/fanglin/caffe/FaceDetection/faceConfig_2nd.txt");
    
    vector<string> folderNames;
    folderNames.push_back("/media/ssd/data/VOC2007/nonPerson");
    //folderNames.push_back("/media/ssd/data/VOC2007/personHeadMasked");
    
    vector<Mat> imgs;    

    for (int k = 0; k < folderNames.size(); k++) {
        string folderName = folderNames[k];
        vector<string> filePaths;
        GetFilePaths(folderName, ".jpg|.JPG", filePaths);
        //imgs.resize(oldSize+filePaths.size());
        for (size_t i = 0; i < filePaths.size(); i++) {
            Mat img = imread(filePaths[i]);
            if (!img.empty())
                imgs.push_back(img);
        }
    }
    
    ofstream negTrListPath(string(listFolder+"/neg24_train.txt").c_str());
    ofstream negValListPath(string(listFolder+"/neg24_val.txt").c_str());
    
    long long nTotalWindows = 0;
    long long nDetected = 0;
    for (int i = 0; i < imgs.size(); i++) {
        Mat img = imgs[i];
        vector<Rect> rects;
        vector<float> scores;
        tic();
        int nWs = facedetector.Detect(img, rects, scores);
        
        cout << "Total sliding windows " << nWs << endl;
        cout << "Detected faces " << rects.size() << endl; 
        
        
        Mat img_ext;
        Size extSize(img.cols/2, img.rows/2);        
        cv::copyMakeBorder(img, img_ext, extSize.height, extSize.height, 
        extSize.width, extSize.width, BORDER_REPLICATE, CV_RGB(0,0,0));
    
        for (int j = 0; j < rects.size(); j++) {
            Mat rsz, patch;
            Rect rect_ext = rects[j];
            rect_ext.x += extSize.width;
            rect_ext.y += extSize.height;      
            /*
            if (rects[j].x < 0 || rects[j].y < 0 || 
                    rects[j].br().x > img.cols -1 || rects[j].br().y > img.rows -1)
                continue;
             */
            patch = img_ext(rect_ext);

            cv::resize(patch, rsz, cv::Size(24, 24));
            stringstream ss;
            ss << oFolder << "/neg_24x24_" << i << "_" << j << ".bmp";
            imwrite(ss.str(), rsz);
            if (imgs.size() -i > 100)
                negTrListPath << ss.str() << " " << 0 << endl;
            else
                negValListPath << ss.str() << " " << 0 << endl;
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
    cout << "Average faces per image " << 1.0*nDetected/imgs.size() << endl;
    return 0;
}
