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
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include "FaceClassifier.h"
#include "utilities_common.h"
#include "FaceDetector.h"
#include <fcntl.h>


static double s_set[5] = {0.83, 0.91, 1.0, 1.10, 1.21};
static double x_set[3] = {-0.17, 0, 0.17};
static double y_set[3] = {-0.17, 0, 0.17};

static double sg_s[45];
static double sg_x[45];
static double sg_y[45];

static void nms(vector<Rect>& inRects, vector<float>& scores, float overlap)
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
        int last = Ind.size() - 1;
        int i = Ind[last];
        pick.push_back(i);

        vector<int>	suppress( 1, last );
        for( int pos=0; pos<last; pos++ ){
            int	j = Ind[pos];
            float xx1 = std::max(x1.at<float>(i), x1.at<float>(j));
            float yy1 = std::max(y1.at<float>(i), y1.at<float>(j));
            float xx2 = std::min(x2.at<float>(i), x2.at<float>(j));
            float yy2 = std::min(y2.at<float>(i), y2.at<float>(j));
            float w = xx2-xx1+1;
            float h = yy2-yy1+1;
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

FaceDetector::~FaceDetector()
{
    if (m_detectors.size() > 0) {
        for (int i = 0; i < m_detectors.size(); i++)
            delete m_detectors[i];
        m_detectors.clear();
    }
    if (m_calibrators.size() > 0) {
        for (int i = 0; i < m_calibrators.size(); i++)
            delete m_calibrators[i];
        m_calibrators.clear();
    }
}

using google::protobuf::io::FileInputStream;
void FaceDetector::LoadConfigs(const string& protoFilepath)
{
    int fd = open(protoFilepath.c_str(), O_RDONLY);
    CHECK_NE(fd, -1) << "File not found: " << protoFilepath;
    FileInputStream* input = new FileInputStream(fd);
    
    bool success = google::protobuf::TextFormat::Parse(input, &m_configs);    

    SetDetectors(m_configs);
    SetCalibrators(m_configs);
    
}

void FaceDetector::SetDetectors(const NetConfigs& configs)
{
    vector<string> modelFiles, trainedFiles;
    vector<vector<string> > allMeanFiles;
   
    for (int i = 0; i < configs.detectnet_size(); i++) {
        modelFiles.push_back(m_configs.modelfolder()+"/"+configs.detectnet(i).modelfile());
        trainedFiles.push_back(m_configs.modelfolder()+"/"+configs.detectnet(i).trainedfile());
        vector<string> meanFiles;
        if (configs.detectnet(i).has_meanimagefile_1())
            meanFiles.push_back(m_configs.modelfolder()+"/"+configs.detectnet(i).meanimagefile_1());
        if (configs.detectnet(i).has_meanimagefile_2())
            meanFiles.push_back(m_configs.modelfolder()+"/"+configs.detectnet(i).meanimagefile_2());
        meanFiles.push_back(m_configs.modelfolder()+"/"+configs.detectnet(i).meanimagefile());
        allMeanFiles.push_back(meanFiles);
    }
    SetClassifiers(modelFiles, trainedFiles, allMeanFiles, m_detectors);
}

void FaceDetector::SetCalibrators(const NetConfigs& configs)
{
    vector<string> modelFiles, trainedFiles;
    vector<vector<string> > allMeanFiles;
   
    for (int i = 0; i < configs.calibnet_size(); i++) {
        vector<string> meanFiles;
        modelFiles.push_back(m_configs.modelfolder()+"/"+configs.calibnet(i).modelfile());
        trainedFiles.push_back(m_configs.modelfolder()+"/"+configs.calibnet(i).trainedfile());
        meanFiles.push_back(m_configs.modelfolder()+"/"+configs.calibnet(i).meanimagefile());
        allMeanFiles.push_back(meanFiles);
    }
    SetClassifiers(modelFiles, trainedFiles, allMeanFiles, m_calibrators);
}

void FaceDetector::SetClassifiers(const vector<string>& modelFiles, const vector<string>& trainedFiles, const vector<vector<string> >& meanImageFiles, vector<FaceClassifier*>& classifiers)
{
    assert(modelFiles.size() == trainedFiles.size());
    if (classifiers.size() > 0) {
        for (int i = 0; i < classifiers.size(); i++)
            delete classifiers[i];
        classifiers.clear();
    }
        
    for (int i = 0; i < modelFiles.size(); i++) {
        FaceClassifier* classifier = new FaceClassifier(modelFiles[i], trainedFiles[i], meanImageFiles[i]);
        classifiers.push_back(classifier);
    }
}

int FaceDetector::GetSlidingWindows(const Mat& img, vector<Rect>& rects, vector<float>& scores)
{
    if (!img.data)
        return 0;
    
    float scaling = 1.0*m_rootWindowSize/m_minFaceSize;
    
    // detection step
    // Todo: Modify it to MAX(img.cols, img.rows) to include some margin
    float minScale = 1.0*m_minFaceSize/MIN(img.cols, img.rows);
    vector<float> scales;
    float scale = 1.0;
    while(true) {
        scales.push_back(scale*scaling);
        scale /= m_scaleStep;
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
                scores.push_back(-100);
            }
        }
    }
    return rects.size();
}

void FaceDetector::DetectRects(const Mat& img_ext, const Size& extSize, FaceClassifier* detector, const NetConfig_detect& config, vector<Rect>& rects, vector<float>& scores)
{
    vector<Rect> tmpRects = rects;  
    scores.clear();
    rects.clear();
    for (int i = 0; i < tmpRects.size(); i++) {
        Mat patch;
        Rect rect = tmpRects[i];
        // TODO: remove bound check since we should add some margin if face is out of image
        Rect rect_ext = rect;
        rect_ext.x += extSize.width;
        rect_ext.y += extSize.height;
        Mat patch_ext = img_ext(rect_ext);
        cv::resize(patch_ext, patch, Size(config.size(), config.size()));
        
        vector<float> outputs = detector->Predict(patch);
        if (outputs[1] > config.threshold()) {
            scores.push_back(outputs[1]);
            rects.push_back(rect);
        }
    }
}

void FaceDetector::DetectRects_batch(const Mat& img_ext, const Size& extSize, FaceClassifier* detector, const NetConfig_detect& config, vector<Rect>& rects, vector<float>& scores)
{
    vector<Rect> tmpRects = rects;  
    scores.clear();
    rects.clear();
    
    vector<Mat> patches;
    vector<Rect> rects_batch;
    for (int i = 0; i < tmpRects.size(); i++) {
        Mat patch;
        Rect rect = tmpRects[i];
        // TODO: remove bound check since we should add some margin if face is out of image
        Rect rect_ext = rect;
        rect_ext.x += extSize.width;
        rect_ext.y += extSize.height;
        Mat patch_ext = img_ext(rect_ext);
        cv::resize(patch_ext, patch, Size(config.size(), config.size()));
        patches.push_back(patch);
        rects_batch.push_back(rect);
        if ((i + 1)%m_batchSize == 0 || i == tmpRects.size() - 1) {
            vector<float> outputs = detector->Predict(patches);
            
            for (int ip = 0; ip < patches.size(); ip++) {
                if (outputs[ip*2+1] > config.threshold()) {
                    scores.push_back(outputs[ip*2+1]);
                    rects.push_back(rects_batch[ip]);
                }
            }            
            patches.clear();
            rects_batch.clear();
        }     
    }
}

void FaceDetector::CalibrateRects(const Mat& img_ext, const Size& extSize, FaceClassifier* calibrator, FaceClassifier* detector, const NetConfig_calib& config_c, const NetConfig_detect& config_d, vector<Rect>& rects, vector<float>& scores)
{
    vector<Rect> resultRects;
    scores.clear();
    for (int f = 0; f < rects.size(); f++) {   
        Rect rect = rects[f];
        Rect rect_ext = rect;
        rect_ext.x += extSize.width;
        rect_ext.y += extSize.height;            
        Mat patch;
        cv::resize(img_ext(rect_ext), patch, Size(config_c.size(), config_c.size()));
        vector<float> label_scores = calibrator->Predict(patch);         
        float fs, fx, fy;
        fs = fx = fy = 0.0f;
        int nEffectives = 0;
        for (int i = 0; i < label_scores.size(); i++) {
            if (label_scores[i]> config_c.threshold()) {
                fs += sg_s[i];
                fx += sg_x[i];
                fy += sg_y[i];
                ++nEffectives;
                //cout << label_scores[i] << ", ";
            }
        }

        if (nEffectives > 0) {
            fs /= nEffectives;
            fx /= nEffectives;
            fy /= nEffectives;

            rect = CalibrateRect(rect, fx, fy, fs);
        }                   
        rect_ext = rect;
        rect_ext.x += extSize.width;
        rect_ext.y += extSize.height;
        Mat patch_c;
        cv::resize(img_ext(rect_ext), patch_c, Size(config_c.size(), config_c.size()));

        vector<float> outputs_c = detector->Predict(patch_c);
        if (outputs_c[1] > config_d.threshold()) {
            scores.push_back(outputs_c[1]);
            resultRects.push_back(rect);
        }
    }
    rects = resultRects;
}

void FaceDetector::CalibrateRects_batch(const Mat& img_ext, const Size& extSize, FaceClassifier* calibrator, FaceClassifier* detector, const NetConfig_calib& config_c, const NetConfig_detect& config_d, vector<Rect>& rects, vector<float>& scores)
{
    vector<Mat> patches;
    vector<Rect> rects_all;
    scores.clear();
    vector<Rect> resultRects;
    for (int f = 0; f < rects.size(); f++) {   
        Rect rect_ext = rects[f];
        rect_ext.x += extSize.width;
        rect_ext.y += extSize.height;            
        Mat patch;
        cv::resize(img_ext(rect_ext), patch, Size(config_c.size(), config_c.size()));
        patches.push_back(patch);
        rects_all.push_back(rects[f]);
        if ((f + 1)%m_batchSize == 0 || f == rects.size() - 1) {
            vector<float> label_scores_all = calibrator->Predict(patches); 
            int dim = calibrator->GetLabelDim();
            vector<Mat> patches_c;
            for (int ip = 0; ip < patches.size(); ip++) {
                vector<float> label_scores(label_scores_all.begin()+ip*dim, label_scores_all.begin()+(ip+1)*dim);
                Rect rect = rects_all[ip];
                float fs, fx, fy;
                fs = fx = fy = 0.0f;
                int nEffectives = 0;
                for (int i = 0; i < label_scores.size(); i++) {
                    if (label_scores[i]> config_c.threshold()) {                        
                        fs += sg_s[i];
                        fx += sg_x[i];
                        fy += sg_y[i];
                        ++nEffectives;
                        //cout << i << " " << label_scores[i] << ", ";
                    }
                }
                //cout << endl;
                if (nEffectives > 0) {
                    fs /= nEffectives;
                    fx /= nEffectives;
                    fy /= nEffectives;

                    Rect rct = CalibrateRect(rect, fx, fy, fs);
                    rect = rct;
                }                   
                Rect rect_ext1 = rect;
                rect_ext1.x += extSize.width;
                rect_ext1.y += extSize.height;                         
                Mat patch_c;
                cv::resize(img_ext(rect_ext1), patch_c, Size(config_c.size(), config_c.size()));
                patches_c.push_back(patch_c);  
                rects_all[ip] = rect;
            }
            vector<float> outputs_c = detector->Predict(patches_c);
            for (int ip = 0; ip < patches.size(); ip++) {
                if (outputs_c[ip*2 + 1] > config_d.threshold()) {
                    scores.push_back(outputs_c[ip*2+1]);
                    resultRects.push_back(rects_all[ip]);
                    
                }
            }
            patches.clear();
            rects_all.clear();
            patches_c.clear();
        }
    }
    rects = resultRects;
}

int FaceDetector::Detect(const Mat& img, vector<Rect>& rects, vector<float>& scores)
{   
    rects.clear();
    scores.clear();
    /*
     * 2015-08-19, Fanglin
     * Previous plan is to make this detect method as generalize as possible.
     * Now I make it more specific to the method described in [A Convolutional Neural Network Cascade for Face Detection] 
     */    
    Mat img_extended;
    Size extSize(img.cols, img.rows);
        
    cv::copyMakeBorder(img, img_extended, extSize.height, extSize.height, 
            extSize.width, extSize.width, BORDER_REPLICATE, CV_RGB(0,0,0));
    //--------------------------------------------------
    // Step 1: First cascade, 12-net and its calibration
    //--------------------------------------------------
    float minScale = 1.0*m_minFaceSize/MIN(img.cols, img.rows);
    float scaling = 1.0*m_rootWindowSize/m_minFaceSize;
    vector<float> scales;
    float scale = 1.0;
        
    while(true) {
        scales.push_back(scale*scaling);
        scale /= m_scaleStep;
        if (scale < minScale)
            break;
    }
    FaceClassifier *detector12 = 0, *calibrator12 = 0;
    NetConfig_detect config_d12;
    if (m_detectors.size() > 0) {
        detector12 = m_detectors[0];
        config_d12 = m_configs.detectnet(0);
    }
    int nTotalRects = 0;
    
    vector<Mat> patches(m_batchSize);
    
    vector<Rect> rects_all;
    vector<float> scales_all;
    for (int f = 0; f < scales.size(); f++) {
        Mat rszImg;
        cv::resize(img, rszImg, cv::Size(), scales[f], scales[f]);
        
        for (int r = 0; r + m_rootWindowSize < rszImg.rows; r+= m_rootSpacing) {
            for (int c = 0; c + m_rootWindowSize < rszImg.cols; c+= m_rootSpacing) {
                Rect rect = Rect(c, r, m_rootWindowSize, m_rootWindowSize);
                //Mat patch = rszImg(rect);  
                patches[nTotalRects%m_batchSize] = rszImg(rect);
                rects_all.push_back(rect);
                scales_all.push_back(scales[f]);
                if (detector12 && (nTotalRects+1) % m_batchSize == 0) {
                    vector<float> outputs = detector12->Predict(patches);
                    for (int i = 0; i < patches.size(); i++) {
                        Rect rct = rects_all[i];
                        if (outputs[2*i+1] > config_d12.threshold()) {
                            rct.x = rct.x / scales_all[i] + 0.5;
                            rct.y = rct.y / scales_all[i] + 0.5;
                            rct.width = rct.width / scales_all[i] + 0.5;
                            rct.height = rct.height / scales_all[i] + 0.5;
                            scores.push_back(outputs[2*i+1]);
                            rects.push_back(rct);
                        }
                    }
                    //patches.clear();
                    rects_all.clear();
                    scales_all.clear();
                }
                
               
                ++nTotalRects;               
//                vector<float> outputs = detector12->Predict(patch);
//                if (outputs[1] > config_d12.threshold()) {
//                    rect.x = rect.x / scales[f] + 0.5;
//                    rect.y = rect.y / scales[f] + 0.5;
//                    rect.width = rect.width / scales[f] + 0.5;
//                    rect.height = rect.height / scales[f] + 0.5;
//                    scores.push_back(outputs[1]);
//                    rects.push_back(rect);
//                }
            }
        }        
    }  
    int nLeft = nTotalRects%m_batchSize;
    if (detector12 && nLeft > 0) {        
        vector<Mat> patches_left(patches.begin(), patches.begin()+nLeft);
        vector<float> outputs = detector12->Predict(patches_left);
        for (int i = 0; i < patches_left.size(); i++) {
            Rect rct = rects_all[i];
            if (outputs[2*i+1] > config_d12.threshold()) {
                rct.x = rct.x / scales_all[i] + 0.5;
                rct.y = rct.y / scales_all[i] + 0.5;
                rct.width = rct.width / scales_all[i] + 0.5;
                rct.height = rct.height / scales_all[i] + 0.5;
                scores.push_back(outputs[2*i+1]);
                rects.push_back(rct);
            }
        }        
    }
    patches.clear();
    rects_all.clear();
    scales_all.clear();
    
    if (m_calibrators.size() >= 1) {
        NetConfig_calib config_c12 = m_configs.calibnet(0);
        calibrator12 = m_calibrators[0];
        CalibrateRects_batch(img_extended, extSize, calibrator12, detector12, 
                config_c12, config_d12, rects, scores);
    }
    // First NMS after calibration. Only do nms when calibration is done.
    if (m_calibrators.size() >= 1)
        nms(rects, scores, m_configs.calibnet(0).nmsoverlap());     
    
    if (m_detectors.size() < 2)
        return nTotalRects;
    
    //----------------------------------------------------------------
    // Step 2: 24-net detection and calibration
    //----------------------------------------------------------------
    FaceClassifier *detector24, *calibrator24;
    detector24 = m_detectors[1];
    NetConfig_detect config_d24 = m_configs.detectnet(1);
    DetectRects_batch(img_extended, extSize, detector24, config_d24, rects, scores);
    if (m_calibrators.size() >= 2) {
        NetConfig_calib config_c24 = m_configs.calibnet(1);
        calibrator24 = m_calibrators[1];
        CalibrateRects_batch(img_extended, extSize, calibrator24, detector24, 
                config_c24, config_d24, rects, scores);
    }
    
    // Second NMS after calibration. Only do nms when calibration is done.
    if (m_calibrators.size() >= 2)
        nms(rects, scores, 0.9);
    
    if (m_detectors.size() < 3)
        return nTotalRects;
    
    //----------------------------------------------------------------
    // Step 3: 48-net detection and calibration
    // Detection-> NMS -> Calibration
    //----------------------------------------------------------------
    FaceClassifier *detector48, *calibrator48;
    detector48 = m_detectors[2];
    NetConfig_detect config_d48 = m_configs.detectnet(2);
    DetectRects_batch(img_extended, extSize, detector48, config_d48, rects, scores);
    // Global NMS.
    //nms(rects, scores, 0.5); 
    //nms(rects, scores, 0.1);    

    //nms(rects, scores, 0.1);    
    
    // Third calibration
    if (m_calibrators.size() >= 3) {
        NetConfig_calib config_c48 = m_configs.calibnet(1);
        calibrator48 = m_calibrators[2];
        CalibrateRects(img_extended, extSize, calibrator48, detector48, 
                config_c48, config_d48, rects, scores);
    }         
    return nTotalRects;
}

void GenerateCalibLabelSet()
{
    for (int is = 0; is < 5; is++) {
        for (int ix = 0; ix < 3; ix++) {
            for (int iy = 0; iy < 3; iy++) {
                int calibLabel = is*3*3 + ix*3 + iy;
                sg_s[calibLabel] = s_set[is];
                sg_x[calibLabel] = x_set[ix];
                sg_y[calibLabel] = y_set[iy];
            }
        }
    }                
}

Rect CalibrateRect(const Rect rect, float ex, float ey, float es)
{
    int rx = cvRound(rect.x + ex*rect.width);
    int ry = cvRound(rect.y + ey*rect.height);
    int rw = cvRound(rect.width*es);
    int rh = cvRound(rect.height*es);
    
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
                                fs += sg_s[i];
                                fx += sg_x[i];
                                fy += sg_y[i];
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
    
    
    nms(resultRects, scores, 0.75);
        
    vector<Rect> tmpRects = resultRects;
    vector<float> tmpScores = scores;
    
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
                        fs += sg_s[l];
                        fx += sg_x[l];
                        fy += sg_y[l];
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
