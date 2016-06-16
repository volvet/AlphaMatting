//
//  main.cpp
//  AlphaMatting
//
//  Created by Volvet Zhang on 16/6/15.
//  Copyright © 2016年 Volvet Zhang. All rights reserved.
//

#include <iostream>
#include <string>

#include "SharedMatting.h"

using namespace cv;

int compareResult(const char * lResultName, const char * rResultName)
{
    cv::Mat  l = imread(lResultName);
    cv::Mat  r = imread(rResultName);
    
    if (l.total() != r.total()) return 1;
    if (l.rows != r.rows ) return 1;
    if (l.cols != r.cols ) return 1;
    if( l.channels() != r.channels() ) return 1;
    int i, j;
    unsigned char * l_ptr, * r_ptr;
    
    for ( j=0;j<r.rows;j++ ){
        l_ptr = l.ptr<uchar>(j);
        r_ptr = r.ptr<uchar>(j);
        for( i=0;i<r.cols * r.channels();i++ ){
            if( l_ptr[i] != r_ptr[i] )
            {
                cout << "Inconsist found: j="<<j<<",i="<<i<<std::endl;
                return 1;
            }
        }
    }
    
    return 0;
}


int main(int argc, const char * argv[]) {
    
    SharedMatting sm;
    
    std::string inMap = "../TestSet/input.png";
    std::string triMap = "../TestSet/trimap.png";
    std::string resultMap = "../TestSet/resultMap.png";
    std::string banchMark = "../TestSet/result.png";
    
    sm.loadImage(inMap.c_str());
    
    sm.loadTrimap(triMap.c_str());
    
    sm.solveAlpha();
    
    sm.save(resultMap.c_str());
    
    
    std::string error;
    if( compareResult(banchMark.c_str(), resultMap.c_str()) != 0 ){
        error = "Test Fail";
    } else {
        error = "Test Pass";
    }
    
    std::cout << error <<std::endl;
    
    return 0;
}
