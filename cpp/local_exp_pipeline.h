/**
  ******************************************************************************
  * Copyright (C), 2021-2025, Tsinghua Unv. & Pengcheng Lab.
  * Author             : Richard Jiang
  * Email              : jyc14@tsinghua.org.cn
  * History            :
        <author>       <version>       <time>       <desc>
      Richard Jiang      v0.0        2022/11/21     Init
  ******************************************************************************
  */

#pragma once

#include <gmp.h> 
#define __gmp_const const
#include <math.h>
#include <cmath>
// #include "ap_int.h"
// #include "ap_fixed.h"
// #include "hls_math.h"
// #include "hls_streamofblocks.h"
#include "local_exp_pipeline_hls.h"

#ifdef __unix
#define OPENCV_TRAITS_ENABLE_DEPRECATED
#endif

// opencv library
#include <opencv2/opencv.hpp>
#ifdef _DEBUG
#pragma comment(lib,"opencv_world310d.lib")
#else
#pragma comment(lib,"opencv_world310.lib")
#endif

void initBinaryCost(int width, int height, cv::Mat imL, cv::Mat imR, std::vector<cv::Mat>& smoothnessCoeff, std::vector<cv::Point> neighbors, const int M, float omega, float epsilon);
void localExpPipeline(const std::string inputDir, const std::string outputDir, std::string volDir, cv::Mat imL, cv::Mat imR, cv::Mat& dispL, cv::Mat& dispR, std::string KITTI_img);
