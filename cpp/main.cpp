/**
  ******************************************************************************
  * Copyright (C), 2021-2025, Tsinghua Unv. & Pengcheng Lab.
  * Author             : Richard Jiang
  * Email              : jyc14@tsinghua.org.cn
  * History            :
        <author>       <version>       <time>       <desc>
      Richard Jiang      v0.0        2022/08/30     Init
  ******************************************************************************
  */

#include <iostream>
#include "common.h"
#include "local_exp_opcv.h"
#include "local_exp_pipeline.h"
#include "Utilities.hpp"
#include "TimeStamper.h"

TimeStamper timer;

// opencv library
#include <opencv2/opencv.hpp>
#ifdef _DEBUG
#pragma comment(lib,"opencv_world310d.lib")
#else
#pragma comment(lib,"opencv_world310.lib")
#endif

// 
int main(int argc, char** argv) {
    std::string input_dir, output_dir;
    std::string KITTI_str = "0";
    std::string KITTI_str_padded = std::string(6 - KITTI_str.length(), '0') + KITTI_str;
    if (!(argc == 1 || argc == 7)) {
        printf("command format: \n1) ./run ../input_dir/ ../output_dir/ im0.png im1.png im0.bin im_gt.png\n2) ./run\n");
        return 0;
    }
    if (argc == 1) {
#ifdef __unix
        input_dir = "../../../KITTI/";
        output_dir = "../../../KITTI/est_default/" + KITTI_str_padded + "/";
#else
        input_dir = "..\\..\\..\\clone_config\\KITTI\\";
        output_dir = "..\\..\\..\\clone_config\\KITTI\\est_default\\" + KITTI_str_padded + "\\";
        //input_dir = "..\\..\\KITTI\\";
        //output_dir = "..\\..\\KITTI\\est_default\\" + KITTI_str_padded + "\\";
#endif
    }
    else {
        input_dir = argv[1];
        output_dir = argv[2];
    }
    std::string img_left, img_right;
    if (argc == 1) {
#ifdef __unix
        img_left =  "training/image_2/" + KITTI_str_padded + "_10.png";
        img_right = "training/image_3/" + KITTI_str_padded + "_10.png";
#else
        img_left =  "training\\image_2\\" + KITTI_str_padded + "_10.png";
        img_right = "training\\image_3\\" + KITTI_str_padded + "_10.png";
#endif
    } 
    else {
        img_left = argv[3];
        img_right = argv[4];
    }
    std::string img_l_dir = input_dir + img_left;
    std::string img_r_dir = input_dir + img_right;
    std::string disp_l_dir = output_dir + "disp0.pfm";
    std::string disp_r_dir = output_dir + "disp1.pfm";
    std::string cost_vol_dir;
    if (argc == 1) {
#ifdef __unix
        cost_vol_dir = "cost_volume_rtstereo/" + KITTI_str_padded + "_4.bin";
#else
        cost_vol_dir = "cost_volume\\" + KITTI_str_padded + ".bin";
#endif
    }
    else {
        cost_vol_dir = argv[5];
    }
    std::string ground_truth_dir; 
    if(argc == 1){
#ifdef __unix
        ground_truth_dir = "training/disp_noc_0/" + KITTI_str_padded + "_10.png";
#else
        ground_truth_dir = "training\\disp_noc_0\\" + KITTI_str_padded + "_10.png";
#endif
    }
    else{
        ground_truth_dir = argv[6];
    }
    cv::Mat img_l_mat = cv::imread(img_l_dir, cv::IMREAD_UNCHANGED);
    cv::Mat img_r_mat = cv::imread(img_r_dir, cv::IMREAD_UNCHANGED);
    cv::Mat disp_l_mat;
    cv::Mat disp_r_mat;
    if (img_l_mat.empty() == true || img_r_mat.empty() == true) {
        std::cout << "\nimage reading failed!\n" << std::endl;
    }
    std::cout << "left image: " << img_l_dir << std::endl;
    std::cout << "right image: " << img_r_dir << std::endl;
    std::cout << "cost volume: " << cost_vol_dir << std::endl;
    // image parameter
    const int width = static_cast<uint>(img_l_mat.cols);
    const int height = static_cast<uint>(img_l_mat.rows);
    // 
    assert(width == WIDTH && height == HEIGHT);
    printf("image size: %d x %d\n", WIDTH, HEIGHT);
    printf("disparity range: %d - %d\n", 0, DISP_MAX - 1);
    //
#if 0
    localExpOpcv(input_dir, cost_vol_dir, img_l_mat, img_r_mat, disp_l_mat, disp_r_mat, DISP_MAX, 0);
#else
    localExpPipeline(input_dir, output_dir, cost_vol_dir, img_l_mat, img_r_mat, disp_l_mat, disp_r_mat, ground_truth_dir);
#endif
    // 
#if 0
    cvutils::io::save_pfm_file(disp_l_dir, disp_l_mat);
    cvutils::io::save_pfm_file(disp_r_dir, disp_r_mat);
#endif
    return 0;
}