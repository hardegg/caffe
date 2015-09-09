#include <opencv2/opencv.hpp>
#include <iostream>
#include "utilities_common.h"
#include <fstream>

using namespace std;
using namespace cv;

int main(int argc, const char* argv[])
{
    string folderName = "/media/ssd/data/VOC2007/nonPerson";
    string oFolder = "/media/ssd/data/aflw/data/neg12x12";
    string listFolder = "/media/ssd/data/aflw/data";
    
    vector<string> filePaths;
    GetFilePaths(folderName, ".jpg", filePaths);
    vector<Mat> imgs(filePaths.size());
    for (size_t i = 0; i < filePaths.size(); i++) {
        imgs[i] = imread(filePaths[i]);
    }
    cout << imgs.size() << " images!" << endl;
    
    RNG rng( 0xFFFFFFFF );
    rng.uniform( 0, 100 );
    int nNegs = 200000 + 10000; // 10000 for validation
    int nDone = 0;
    
    ofstream negTrListPath(string(listFolder+"/neg12_train.txt").c_str());
    ofstream negValListPath(string(listFolder+"/neg12_val.txt").c_str());
    while(1) {
        int idx = rng.uniform(0, imgs.size());
        int r = rng.uniform(0, imgs[idx].rows);
        int c = rng.uniform(0, imgs[idx].cols);
        int s = rng.uniform(0, 2*MIN(MIN(r, imgs[idx].rows-r), MIN(c, imgs[idx].cols-c)) + 1 );
        
        if (s < 28)
            continue;
        
        Rect rect = Rect(c-s/2, r-s/2, s, s);
        Mat patch = imgs[idx](rect);
        Mat rsz;
        cv::resize(patch, rsz, cv::Size(12, 12));
        stringstream ss;
        ss << oFolder << "/neg_" << nDone << ".bmp";
        imwrite(ss.str(), rsz);
        if (nDone <= 200000)
            negTrListPath << ss.str() << " " << 0 << endl;
        else
            negValListPath << ss.str() << " " << 0 << endl;
        ++nDone;
        if (nDone == nNegs)
            break;        
        if (nDone % 1000 == 0)
            cout << nDone << " neg generated!" << endl;
    }
    
    negTrListPath.close();
    negValListPath.close();
    return 0;
}
