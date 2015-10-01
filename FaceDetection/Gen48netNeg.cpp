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

vector<string> split(const string &s, char delim) {
    stringstream ss(s);
    string item;
    vector<string> tokens;
    while (getline(ss, item, delim)) {
        tokens.push_back(item);
    }
    return tokens;
}

float getIOU(Rect rect1, Rect rect2)
{
    Rect inter = rect1 & rect2;
    return 1.0*inter.area()/(rect1.area()+rect2.area()-inter.area());
}

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
    min_FaceSize = 24; scaleStep = 1.05; spacing = 4;
    FaceDetector facedetector(min_FaceSize, scaleStep, spacing);
    facedetector.LoadConfigs("/home/fanglin/caffe/FaceDetection/faceConfig_2nd.txt");
    
    ofstream negTrListPath(string(listFolder+"/neg48_train_addFP.txt").c_str());
    ofstream negValListPath(string(listFolder+"/neg48_val_addFP.txt").c_str());
    
    vector<string> filePaths;    
    string afwPath = "/media/ssd/data/AFW";
    string annoFile = afwPath + "/annotation.txt";
    ifstream afwf(annoFile.c_str());
    const int MAX_LENGTH = 4096;
    char line[MAX_LENGTH];
    int iImage = 0;
    long long nTotalWindows = 0;
    long long nDetected = 0;
    tic();
    while (afwf.getline(line, MAX_LENGTH)) {
        toc();
        tic();
        string filepath = afwPath + "/" + line;
        string filename = line;
        afwf.getline(line, MAX_LENGTH);
        int nFaces = atoi(line);
        vector<Rect> annoRects(nFaces);
        for (int i = 0; i < nFaces; i++) {
            afwf.getline(line, MAX_LENGTH);
            vector<string> strRect = split(line, ' ');
            annoRects[i].x = atoi(strRect[0].c_str());
            annoRects[i].y = atoi(strRect[1].c_str());
            annoRects[i].width = atoi(strRect[2].c_str()) - annoRects[i].x;
            annoRects[i].height = atoi(strRect[3].c_str()) - annoRects[i].y;          
        }
        Mat img = imread(filepath);
        
        Mat img_ext;
        Size extSize(img.cols/2, img.rows/2);        
        cv::copyMakeBorder(img, img_ext, extSize.height, extSize.height, 
        extSize.width, extSize.width, BORDER_REPLICATE, CV_RGB(0,0,0));
        
        //for (int i = 0; i < nFaces; i++) {
            //cv::rectangle(img, annoRects[i], CV_RGB(0, 255, 0), 2);
        //}
        //imwrite(string("output/")+filename, img);

        
        
        vector<Rect> rects;
        vector<float> scores;
        vector<float> IOU;
        int nWs = facedetector.Detect(img, rects, scores);    
        float iouThresh = 0.1;
        for (int i = 0; i < rects.size(); i++) {
            bool detected = false;
            for (int j = 0; j < annoRects.size(); j++) {
                //if (getIOU(rects[i], annoRects[j])>0.3 && getIOU(rects[i], annoRects[j])<0.35) {

                if (getIOU(rects[i], annoRects[j])>iouThresh) {
                    detected = true;
                    break;
                }
            }
            /*
            if (detected)
                cv::rectangle(img, rects[i], CV_RGB(255, 0, 0), 2);
            else
                cv::rectangle(img, rects[i], CV_RGB(255, 255, 0), 2);
             * */
            
            if (!detected) {
                Rect rect_ext = rects[i];
                rect_ext.x += extSize.width;
                rect_ext.y += extSize.height;   
                Mat patch = img_ext(rect_ext);
                Mat rsz;
                cv::resize(patch, rsz, cv::Size(48, 48));
                stringstream ss;
                ss << oFolder << "/neg_48x48_AFW_" << iImage << "_" << i << ".bmp";
                imwrite(ss.str(), rsz);
            
                negTrListPath << ss.str() << " " << 0 << endl;
            }             
        }
        nTotalWindows += nWs;
        nDetected += rects.size();
        cout << "Total sliding windows " << nWs << endl;
        cout << "Detected FPs " << rects.size() << endl;
        cout << iImage + 1 << " has been finished.\n";
        ++iImage;        
        imshow("img", img);
        cv::waitKey(1);
    }
    GetFilePaths(folderName, ".jpg|.JPG", filePaths); 

    vector<string> folderNames;    
    folderNames.push_back("/media/ssd/data/VOC2007/personHeadMasked");
    folderNames.push_back("/media/ssd/data/VOC2007/nonPerson");
    folderNames.push_back("/media/ssd/data/WuJX/SkinColor/nonFaces");
    
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
    
    g_nImages = imgs.size();    
    
    for (int i = 0; i < g_nImages; i++) {
        toc();
        tic();
        Mat img = imgs[i];
        Mat img_ext;
        Size extSize(img.cols/2, img.rows/2);        
        cv::copyMakeBorder(img, img_ext, extSize.height, extSize.height, 
        extSize.width, extSize.width, BORDER_REPLICATE, CV_RGB(0,0,0));
        if(!img.data)
            continue;
        vector<Rect> rects;
        vector<float> scores;
        int nWs = facedetector.Detect(img, rects, scores);         
        
        for (int j = 0; j < rects.size(); j++) {
            Mat rsz, patch;
            Rect rect_ext = rects[j];
            rect_ext.x += extSize.width;
            rect_ext.y += extSize.height;   
            patch = img_ext(rect_ext);

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
        cout << i+1 << " / " << imgs.size() << " has been finished.\n";
        
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
