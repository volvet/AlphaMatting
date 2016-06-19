

#ifndef SHAREDMSTTING_H
#define SHAREDMSTTING_H

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cmath>
#include <vector>

using namespace std;

struct labelPoint
{
    int x;
    int y;
    int label;
};

struct Tuple
{
    cv::Scalar f;
    cv::Scalar b;
    double   sigmaf;
    double   sigmab;
    
    int flag;
    
};

struct Ftuple
{
    cv::Scalar f;
    cv::Scalar b;
    double   alphar;
    double   confidence;
};

/*程序中认定cv::Point中 x为行，y为列，可能错误，但对程序结果没有影响*/
class SharedMatting
{
public:
    SharedMatting();
    ~SharedMatting();
    
    void loadImage(const char * filename);
    void loadTrimap(const char * filename);
    void expandKnown();
    void sample(cv::Point p, vector<cv::Point>& f, vector<cv::Point>& b);
    void gathering();
    void refineSample();
    void localSmooth();
    void solveAlpha();
    void save(const char * filename);
    void Sample(vector<vector<cv::Point> > &F, vector<vector<cv::Point> > &B);
    void getMatte();
    void release();
    
    double mP(int i, int j, cv::Scalar f, cv::Scalar b);
    double nP(int i, int j, cv::Scalar f, cv::Scalar b);
    double eP(int i1, int j1, int i2, int j2);
    double pfP(cv::Point p, vector<cv::Point>& f, vector<cv::Point>& b);
    double aP(int i, int j, double pf, cv::Scalar f, cv::Scalar b);
    double gP(cv::Point p, cv::Point fp, cv::Point bp, double pf);
    double gP(cv::Point p, cv::Point fp, cv::Point bp, double dpf, double pf);
    double dP(cv::Point s, cv::Point d);
    double sigma2(cv::Point p);
    double distanceColor2(cv::Scalar cs1, cv::Scalar cs2);
    double comalpha(cv::Scalar c, cv::Scalar f, cv::Scalar b);
    
private:
    cv::Mat pImg;
    cv::Mat trimap;
    cv::Mat matte;
    
    vector<cv::Point> uT;
    vector<struct Tuple> tuples;
    vector<struct Ftuple> ftuples;
    
    int height;
    int width;
    int ** unknownIndex;//Unknown的索引信息；
    int ** tri;
    int ** alpha;
    
    size_t step;
    int channels;
    uchar* data;
    
};

#endif

