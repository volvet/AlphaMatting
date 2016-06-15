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

int main(int argc, const char * argv[]) {
    
    SharedMatting sm;
    
    std::string inMap = "../TestSet/input.png";
    std::string triMap = "../TestSet/trimap.png";
    std::string resultMap = "../TestSet/resultMap.png";
    
    sm.loadImage(inMap.c_str());
    
    sm.loadTrimap(triMap.c_str());
    
    sm.solveAlpha();
    
    sm.save(resultMap.c_str());
    
    std::cout << "end" <<std::endl;
    
    return 0;
}
