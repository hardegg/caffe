#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>

#include "FaceClassifier.h"
#include "utilities_common.h"
#include "FaceDetector.h"

static double s_set[5] = {0.83, 0.91, 1.0, 1.10, 1.21};
static double x_set[3] = {-0.17, 0, 0.17};
static double y_set[3] = {-0.17, 0, 0.17};

static double gs_s[45];
static double gs_x[45];
static double gs_y[45];

static int gs_sizes[3] = {12, 24, 48};
static float gs_nms_overlaps[3] = {0.75, 0.75, 0.5};
static float gs_detect_thr[3] = {0.2, 0.0001, 0.2};
static float gs_calibrate_thr[3] = {0.01, 0.1, 0.1};

static void	nms(vector<Rect>& inRects, vector<float>& scores, float overlap)
//% Non-maximum suppression.
//%   pick = nms(boxes, overlap) 
//% 
//%   Greedily select high-scoring detections and skip detections that are 
//%   significantly covered by a previously selected detection.
//%
//% Return value
//%   pick      Indices of locally maximal detections
//%
//% Arguments
//%   boxes     Detection bounding boxes (see pascal_test.m)
//%   overlap   Overlap threshold for suppression
//%             For a selected box Bi, all boxes Bj that are covered by 
//%             more than overlap are suppressed. Note that 'covered' is
//%             is |Bi \cap Bj| / |Bj|, not the PASCAL intersection over 
//%             union measure.
{
    Mat boxes(Size(4, inRects.size()), CV_32FC1);
    Mat s(Size(1, inRects.size()), CV_32FC1);

    for (int i = 0; i < inRects.size(); i++) {
        boxes.at<float>(i, 0) = inRects[i].x;
        boxes.at<float>(i, 1) = inRects[i].y;
        boxes.at<float>(i, 2) = inRects[i].x + inRects[i].width - 1;
        boxes.at<float>(i, 3) = inRects[i].y + inRects[i].height - 1;
        s.at<float>(i, 0) = scores[i];
    }
    
    Mat parts;
    if( boxes.empty() )
        return;

    Mat	x1 = boxes.col(0);
    Mat	y1 = boxes.col(1);
    Mat	x2 = boxes.col(2);
    Mat	y2 = boxes.col(3);

    Mat	area = x2-x1+1;
    area = area.mul(y2-y1+1);

    vector<int>	Ind( s.rows, 0 );
    Mat	Idx( s.rows, 1, CV_32SC1, &Ind[0] );
    sortIdx( s, Idx, CV_SORT_EVERY_COLUMN+CV_SORT_ASCENDING );   
   
    vector<int>	pick;
    while( !Ind.empty() ){
        int	last = Ind.size() - 1;
        int	i = Ind[last];
        pick.push_back(i);

        vector<int>	suppress( 1, last );
        for( int pos=0; pos<last; pos++ ){
            int		j = Ind[pos];
            float		xx1 = std::max(x1.at<float>(i), x1.at<float>(j));
            float		yy1 = std::max(y1.at<float>(i), y1.at<float>(j));
            float		xx2 = std::min(x2.at<float>(i), x2.at<float>(j));
            float		yy2 = std::min(y2.at<float>(i), y2.at<float>(j));
            float		w = xx2-xx1+1;
            float		h = yy2-yy1+1;
            if( w>0 && h>0 ){
                // compute overlap 
                float	area_intersection = w * h;
                //float	o1 = area_intersection / area.at<float>(j);
                //float	o2 = area_intersection / area.at<float>(i);
//                float	o = std::max(o1,o2);
                float	o = area_intersection / (area.at<float>(j) + area.at<float>(i) - area_intersection);

                if( o>overlap )
                    suppress.push_back(pos);
            }
        }

        std::set<int>	supp( suppress.begin(), suppress.end() );
        vector<int>		Ind2;
        for( int i=0; i!=Ind.size(); i++ ){
            if( supp.find(i)==supp.end() )
                Ind2.push_back( Ind[i] );
        }
        Ind = Ind2;

    }

    Mat	tmp( pick.size(), boxes.cols, boxes.type() );
    vector<float> tmpScores = scores;
    scores.resize(pick.size());
    for( int i=0; i<pick.size(); i++ ) {
        boxes.row( pick[i] ).copyTo( tmp.row(i) );
        scores[i] = tmpScores[pick[i]];
    }
    boxes.create( tmp.rows, tmp.cols, tmp.type() );
    tmp.copyTo( boxes );   
    
    inRects.resize(boxes.rows);
    for (int i = 0; i < boxes.rows; i++) {
        inRects[i].x = boxes.at<float>(i, 0);
        inRects[i].y = boxes.at<float>(i, 1);
        inRects[i].width = boxes.at<float>(i, 2) - inRects[i].x + 1;
        inRects[i].height = boxes.at<float>(i, 3)- inRects[i].y + 1;
    }

}

void FaceDetector::SetClassifiers(const vector<string>& modelFiles, const vector<string>& trainedFiles, vector<FaceClassifier*>& classifiers)
{
    if (classifiers.size() > 0) {
        for (int i = 0; i < classifiers.size(); i++)
            delete classifiers[i];
        classifiers.clear();
    }
        
    for (int i = 0; i < modelFiles.size(); i++) {
        FaceClassifier* classifier = new FaceClassifier(modelFiles[i], trainedFiles[i]);
        classifiers.push_back(classifier);
    }
}

int FaceDetector::GetSlidingWindows(const Mat& img, vector<Rect>& rects)
{
    if (!img.data)
        return 0;
    
    float scaling = 1.0*m_rootWindowSize/m_minFaceSize;
    
    // detection step
    float scaleStep = 1.118; //1.118 for FDDB, 1.414 for AFW
    // Todo: Modify it to MAX(img.cols, img.rows) to include some margin
    float minScale = 1.0*m_minFaceSize/MIN(img.cols, img.rows);
    vector<float> scales;
    float scale = 1.0;
    while(true) {
        scales.push_back(scale*scaling);
        scale /= scaleStep;
        if (scale < minScale)
            break;
    }
        
    for (int f = 0; f < scales.size(); f++) {
        Mat rszImg;
        cv::resize(img, rszImg, cv::Size(), scales[f], scales[f]);
        for (int r = 0; r + m_rootWindowSize < rszImg.rows; r+= m_rootSpacing) {
            for (int c = 0; c + m_rootWindowSize < rszImg.cols; c+= m_rootSpacing) {
                Rect rect = Rect(c, r, m_rootWindowSize, m_rootWindowSize);
                rect.x /= scales[f];
                rect.y /= scales[f];
                rect.width /= scales[f];
                rect.height /= scales[f];
                rects.push_back(rect);
            }
        }
    }
    return rects.size();
}

void FaceDetector::Detect(const Mat& img, vector<Rect>& rects, vector<float>& scores)
{    
    if (m_detectors.size() != m_calibrators.size()) {
        cout << "The number of detectors and calibrators should be same!" << endl;
        return;
    }
    
    if (m_detectors.size() < 1) {
        cout << "There should be at least one detector." << endl;
        return;
    }
    
    // Generate sliding windows first
    rects.clear();
    GetSlidingWindows(img, rects); 
    int nTotalRects = rects.size();
    
    // Root detector+calibrator
    for (int i = 0; i < m_detectors.size(); i++) {
        vector<Rect> tmpRects;
        vector<float> tmpScores;
        for (int j = 0; j < rects.size(); j++) {
            Rect rect = rects[j];
            Mat patch = img(rects[j]);
            Mat rsz;
            cv::resize(patch, rsz, cv::Size(gs_sizes[i], gs_sizes[i]));
            vector<float> outputs = m_detectors[i]->Predict(rsz);
            // Calibrate
            
            float fs, fx, fy;
            fs = fx = fy = 0.0f;
            int nEffectives = 0;
            //if( prdct[0].first == 1) {
            if (outputs[1] > gs_detect_thr[i]) {
                if (true) {
                    vector<float> label_scores = m_calibrators[i]->Predict(patch);
                    for (int k = 0; k < label_scores.size(); k++) {
                        if (label_scores[k]> gs_calibrate_thr[i]) {
                            fs += gs_s[k];
                            fx += gs_x[k];
                            fy += gs_y[k];
                            ++nEffectives;
                            //cout << label_scores[i] << ", ";
                        }
                    }
                    //cout << endl;
                    //cout << "nEffectives " << nEffectives << endl;

                } 
                
                if (nEffectives > 0) {
                    fs /= nEffectives;
                    fx /= nEffectives;
                    fy /= nEffectives;

                    rect = CalibrateRect(rect, -fx, -fy, 1/fs);
                }                   

//                    scores.push_back(prdct[0].second);
                tmpScores.push_back(outputs[1]);
                tmpRects.push_back(rect);
            }
        }
        rects = tmpRects;
        scores = tmpScores;
        nms(rects, scores, gs_nms_overlaps[i]);
    }   
    cout << "Total sliding windows " << nTotalRects << endl;
}

void GenerateCalibLabelSet()
{
    for (int is = 0; is < 5; is++) {
        for (int ix = 0; ix < 3; ix++) {
            for (int iy = 0; iy < 3; iy++) {
                int calibLabel = is*3*3 + ix*3 + iy;
                gs_s[calibLabel] = s_set[is];
                gs_x[calibLabel] = x_set[ix];
                gs_y[calibLabel] = y_set[iy];
            }
        }
    }                
}

Rect CalibrateRect(const Rect rect, float ex, float ey, float es)
{
    int rx = cvRound(rect.x - ex*rect.width/es);
    int ry = cvRound(rect.y - ey*rect.height/es);
    int rw = cvRound(rect.width/es);
    int rh = cvRound(rect.height/es);
    
    cv::Rect eRect = cv::Rect(rx, ry, rw, rh);
    return eRect;
}

int GetAllWindows(const Mat& img, vector<Rect>& rects)
{
    if (!img.data)
        return 0;
    int windowSize = 12;// The detection window size. In this paper, it can be 12, 24, 48
    int spacing = 4;    // 4x4 spacing for 12x12 detection window
    int F = 32;         // acceptable minimum face size. 40 for FDDB, 80 for AFW.
    float scaling = 1.0*windowSize/F;
    
    // detection step
    float scaleStep = 1.118; //1.118 for FDDB, 1.414 for AFW
    float minScale = 1.0*F/MIN(img.cols, img.rows);
    vector<float> scales;
    float scale = 1.0;
    while(true) {
        scales.push_back(scale*scaling);
        scale /= scaleStep;
        if (scale < minScale)
            break;
    }
        
    for (int f = 0; f < scales.size(); f++) {
        Mat rszImg;
        cv::resize(img, rszImg, cv::Size(), scales[f], scales[f]);
        for (int r = 0; r + windowSize < rszImg.rows; r+= spacing) {
            for (int c = 0; c + windowSize < rszImg.cols; c+= spacing) {
                Rect rect = Rect(c, r, windowSize, windowSize);
                rect.x /= scales[f];
                rect.y /= scales[f];
                rect.width /= scales[f];
                rect.height /= scales[f];
                rects.push_back(rect);
            }
        }
    }
    return rects.size();
}

int FaceDetection(const Mat& img, FaceClassifier& detector12, FaceClassifier& detector24, FaceClassifier& calibrator12, FaceClassifier& calibrator24, vector<Rect>& resultRects, vector<float>& scores)
{
    if (!img.data)
        return 0;
    int windowSize = 12;// The detection window size. In this paper, it can be 12, 24, 48
    int spacing = 4;    // 4x4 spacing for 12x12 detection window
    int F = 32;         // acceptable minimum face size. 40 for FDDB, 80 for AFW.
    float scaling = 1.0*windowSize/F;
    
    // detection step
    float scaleStep = 1.118; //1.118 for FDDB, 1.414 for AFW
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
    for (int f = 0; f < scales.size(); f++) {
        Mat rszImg;
        cv::resize(img, rszImg, cv::Size(), scales[f], scales[f]);
        for (int r = 0; r + windowSize < rszImg.rows; r+= spacing) {
            for (int c = 0; c + windowSize < rszImg.cols; c+= spacing) {
                Rect rect = Rect(c, r, windowSize, windowSize);
                Mat patch = rszImg(rect);
                //vector<Prediction> prdct = detector.Classify(patch);
            
                vector<float> outputs = detector12.Predict(patch);
                float threshold = 0.2;

                ++nTotalRects;
                float fs, fx, fy;
                fs = fx = fy = 0.0f;
                int nEffectives = 0;
                //if( prdct[0].first == 1) {
                if (outputs[1] > threshold) {
                    if (true) {
                        vector<float> label_scores = calibrator12.Predict(patch);
                        float threshold_c= 0.01;                        
                        
                        for (int i = 0; i < label_scores.size(); i++) {
                            if (label_scores[i]> threshold_c) {
                                fs += gs_s[i];
                                fx += gs_x[i];
                                fy += gs_y[i];
                                ++nEffectives;
                                //cout << label_scores[i] << ", ";
                            }
                        }
                        //cout << endl;
                        //cout << "nEffectives " << nEffectives << endl;
                        
                    } 
                    
                    rect.x /= scales[f];
                    rect.y /= scales[f];
                    rect.width /= scales[f];
                    rect.height /= scales[f];
                    if (nEffectives > 0) {
                        fs /= nEffectives;
                        fx /= nEffectives;
                        fy /= nEffectives;

                        rect = CalibrateRect(rect, -fx, -fy, 1/fs);
                    }                   
                    
//                    scores.push_back(prdct[0].second);
                    scores.push_back(outputs[1]);

                    resultRects.push_back(rect);
                }
            }
        }
    }
    vector<float> tmpScores = scores;
    
    nms(resultRects, scores, 0.75);
        
    vector<Rect> tmpRects = resultRects;
    
    scores.clear();
    resultRects.clear();
    for (int i = 0; i < tmpRects.size(); i++) {
        Mat patch;
        tmpRects[i] &= Rect(0, 0, img.cols, img.rows);
        cv::resize(img(tmpRects[i]), patch, Size(24, 24));
        
        vector<float> outputs = detector24.Predict(patch);
        if (outputs[1] > 0.0001) {
            Rect rect = tmpRects[i];
            float threshold_c= 0.1;                        
            float fs, fx, fy;
            fs = fx = fy = 0.0f;
            int nEffectives = 0;
            
            if (true) {
                vector<float> label_scores = calibrator24.Predict(patch);

                for (int l = 0; l < label_scores.size(); l++) {
                    //cout << label_scores[l] << ", ";
                    if (label_scores[l]> threshold_c) {
                        fs += gs_s[l];
                        fx += gs_x[l];
                        fy += gs_y[l];
                        ++nEffectives;                        
                    }
                }

                if (nEffectives > 0) {
                    fs /= nEffectives;
                    fx /= nEffectives;
                    fy /= nEffectives;

                    rect = CalibrateRect(rect, -fx, -fy, 1/fs);
                } 
                //cout << "\nnEffectives " << nEffectives << endl;
            }
            resultRects.push_back(rect);
            scores.push_back(tmpScores[i]);
        }
        
    }
    
    nms(resultRects, scores, 0.75);
//  std::sort(scores.begin(), scores.end());
    cout << "Total sliding windows " << nTotalRects << endl;
    cout << "Detected faces " << resultRects.size() << endl;    
    return nTotalRects;
}
