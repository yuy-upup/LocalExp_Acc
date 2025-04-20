#pragma once

#ifdef __unix
#define OPENCV_TRAITS_ENABLE_DEPRECATED
#define fopen_s(pFile,filename,mode) ((*(pFile))=fopen((filename),  (mode)))==NULL
typedef int errno_t;
#endif

// opencv library
#include <opencv2/opencv.hpp>
#ifdef _DEBUG
#pragma comment(lib,"opencv_world310d.lib")
#else
#pragma comment(lib,"opencv_world310.lib")
#endif

constexpr auto Invalid_Float = std::numeric_limits<float>::infinity();

uint hammingDistance(uint& x, uint& y);

void cvImshowBool(int width, int height, bool* gray, cv::String win_name);

void cvImshowGray(int width, int height, uint* gray, cv::String win_name);

void cvImshowRgb(int width, int height, uint* rgb, cv::String win_name);

void showDisparity(const float* disp_map, const int& width, const int& height);

void subPixelEnhancement(int width, int height, uint disp_max, uint disp_min, float* disp_l, float* cost);

void onMouse(int event, int x, int y, int flags, void* param);

void writeFilePFM(float* data, int width, int height, char* filename, float scalefactor = 1 / 255.0);

void cvImWriteRgb(int width, int height, uint* rgb, cv::String dir);

void cvImWriteGray(int width, int height, uint* gray, cv::String dir);

void cvImWriteBool(int width, int height, bool* gray, cv::String dir);

int saveFileDat(float* data, int data_num, std::string file_name);

void genRightCostVolume(int width, int height, uint disp_max, uint disp_min, float* cost_volume_l, float* cost_volume_r);

int saveCostVolume(float* data, int width, int height, int disp_max, int disp_min, std::string file_name);

void readCostVolume(float* data, int width, int height, int disp_max, int disp_min, std::string file_name);

void ShowDisparityMap(const float* disp_map, const int& width, const int& height, const std::string& name);



