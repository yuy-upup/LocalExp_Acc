#include "common.h"

/**
  * \brief calculate Hamming distance
  * \param dist: Hanmming distance
  */
uint hammingDistance(uint& x, uint& y) {
	uint dist = 0;
	uint hamming = x ^ y;
	while (hamming != 0) {
		dist++;
		hamming &= hamming - 1;
	}
	return dist;
}

/**
  * \brief show bool array as gray image
  */
void cvImshowBool(int width, int height, bool* gray, cv::String win_name) {
	cv::Mat tmp(height, width, CV_8UC1);
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			uint& tmp_pix = tmp.at<uint>(y, x);
			tmp_pix = 255 * gray[y * width + x];
		}
	}
	cv::namedWindow(win_name, CV_WINDOW_NORMAL);
	cv::imshow(win_name, tmp);
	char c = (char)cv::waitKey();
}

/**
  * \brief show uint array as gray image
  */
void cvImshowGray(int width, int height, uint* gray, cv::String win_name) {
    cv::Mat tmp(height, width, CV_8UC1);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            uint& tmp_pix = tmp.at<uint>(y, x);
            tmp_pix = gray[y * width + x];
        }
    }
    cv::namedWindow(win_name, CV_WINDOW_NORMAL);
    cv::imshow(win_name, tmp);
    char c = (char)cv::waitKey();
}

void cvImWriteGray(int width, int height, uint* gray, cv::String dir) {
	cv::Mat tmp(height, width, CV_8UC1);
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			uint& tmp_pix = tmp.at<uint>(y, x);
			tmp_pix = gray[y * width + x];
		}
	}
	cv::imwrite(dir, tmp);
}

void cvImWriteBool(int width, int height, bool* gray, cv::String dir) {
	cv::Mat tmp(height, width, CV_8UC1);
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			uchar& tmp_pix = tmp.at<uchar>(y, x);
			tmp_pix = gray[y * width + x] * 255;
		}
	}
	cv::imwrite(dir, tmp);
}

/**
  * \brief show rgb array as color image
  */
void cvImshowRgb(int width, int height, uint* rgb, cv::String win_name) {
    cv::Mat tmp(height, width, CV_8UC3);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            cv::Vec3b& tmp_pix = tmp.at<cv::Vec3b>(y, x);
            tmp_pix[0] = cv::saturate_cast<uint>(rgb[y * width * 3 + x * 3 + 0]);
            tmp_pix[1] = cv::saturate_cast<uint>(rgb[y * width * 3 + x * 3 + 1]);
            tmp_pix[2] = cv::saturate_cast<uint>(rgb[y * width * 3 + x * 3 + 2]);
        }
    }
    cv::namedWindow(win_name, CV_WINDOW_NORMAL);
	cv::setMouseCallback(win_name, onMouse, reinterpret_cast<void*> (&tmp));
    cv::imshow(win_name, tmp);
    char c = (char)cv::waitKey();
} 

void cvImWriteRgb(int width, int height, uint* rgb, cv::String dir) {
	cv::Mat tmp(height, width, CV_8UC3);
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			cv::Vec3b& tmp_pix = tmp.at<cv::Vec3b>(y, x);
			tmp_pix[0] = cv::saturate_cast<uint>(rgb[y * width * 3 + x * 3 + 0]);
			tmp_pix[1] = cv::saturate_cast<uint>(rgb[y * width * 3 + x * 3 + 1]);
			tmp_pix[2] = cv::saturate_cast<uint>(rgb[y * width * 3 + x * 3 + 2]);
		}
	}
	cv::imwrite(dir, tmp);
}

/**
  * \brief show disparity
  */
void showDisparity(const float* disp_map, const int& width, const int& height)
{
	// show disparity
	const cv::Mat disp_mat = cv::Mat(height, width, CV_8UC1);
	cv::Mat disp_mat_s = cv::Mat(height, width, CV_32F);
	float min_disp = float(width), max_disp = -float(width);
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			const float disp = abs(disp_map[i * width + j]);
			if (disp != Invalid_Float) {
				min_disp = std::min(min_disp, disp);
				max_disp = std::max(max_disp, disp);
			}
		}
	}
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			const float disp = abs(disp_map[i * width + j]);
			if (disp == Invalid_Float) {
				disp_mat.data[i * width + j] = 0;
			}
			else {
				disp_mat.data[i * width + j] = static_cast<uchar>((disp - min_disp) / (max_disp - min_disp) * 255);
			}
		}
	}
	cv::namedWindow("left disparity", CV_WINDOW_NORMAL);
	cv::imshow("left disparity", disp_mat);
	// cv::imwrite("D:\\git\\personal\\stereo_project\\tmp.png", disp_mat);
	// cv::Mat disp_color;
	// applyColorMap(disp_mat, disp_color, cv::COLORMAP_JET);
	// cv::namedWindow(name + "-color", CV_WINDOW_NORMAL);
	// cv::imshow(name + "-color", disp_color);
	char c = (char)cv::waitKey();
}


/**
  * \brief sub pixel enhancement
  * \param input
  * \param output
  */
void subPixelEnhancement(int width, int height, uint disp_max, uint disp_min, float* disp_l, float* cost) {
	// 
	uint disp_range = disp_max - disp_min;
	float disp_tmp;
	// 
	for (uint y = 0; y < height; y++) {
		for (uint x = 0; x < width; x++) {
			if (disp_l[y * width + x] == Invalid_Float) continue;
			auto d = lround(disp_l[y * width + x]);
			if (d != disp_min && d != disp_max) {
				float cost_c = cost[y * width * disp_range + x * disp_range + d];
				float cost_l = cost[y * width * disp_range + x * disp_range + d - 1];
				float cost_r = cost[y * width * disp_range + x * disp_range + d + 1];
				float denom = cost_l + cost_r - 2 * cost_c;
				if (denom != 0.0f) {
					disp_tmp = disp_l[y * width + x] + (cost_l - cost_r) / (denom * 2.0f);
				}
				else {
					// disp_l[y * width + x] = cost_c;
					disp_tmp = disp_l[y * width + x];
				}
				if (disp_tmp < disp_max)
					disp_l[y * width + x] = disp_tmp;
			}
		}
	}
}

void onMouse(int event, int x, int y, int flags, void* param)
{
	cv::Mat* im = reinterpret_cast<cv::Mat*>(param);
	switch (event)
	{
	case 1:     // left button down: return coordinates
		std::cout << "(" << x << "," << y << ") R:"
			<< static_cast<int>(im->at<cv::Vec3b>(y, x)[0]) << " G:"
			<< static_cast<int>(im->at<cv::Vec3b>(y, x)[1]) << " B:"
			<< static_cast<int>(im->at<cv::Vec3b>(y, x)[2]) << std::endl;
		break; 
	// case 2:    // right button down: return gray 
	// 	std::cout << "input(x,y)" << std::endl;
	// 	std::cout << "x =" << std::endl;
	// 	std::cin >> x;
	// 	std::cout << "y =" << std::endl;
	// 	std::cin >> y;
	// 	std::cout << "at(" << x << "," << y << ")value is:"
	// 		<< static_cast<int>(im->at<uchar>(cv::Point(x, y))) << std::endl;
	// 	break;
	}
}


/**
  * \brief check whether machine is little endian
  */
int littleendian()
{
	int intval = 1;
	uchar* uval = (uchar*)&intval;
	return uval[0] == 1;
}

/**
  * \brief save disparity to pfm format
  * 1-band PFM image, see http://netpbm.sourceforge.net/doc/pfm.html
  */
void writeFilePFM(float* data, int width, int height, char* filename, float scalefactor)
{
	// Open the file
	FILE* stream = fopen(filename, "wb");
	if (stream == 0) {
		fprintf(stderr, "WriteFilePFM: could not open %s\n", filename);
		exit(1);
	}

	// sign of scalefact indicates endianness, see pfms specs
	if (littleendian())
		scalefactor = -scalefactor;

	// write the header: 3 lines: Pf, dimensions, scale factor (negative val == little endian)
	fprintf(stream, "Pf\n%d %d\n%f\n", width, height, scalefactor);

	int n = width;
	// write rows -- pfm stores rows in inverse order!
	for (int y = height - 1; y >= 0; y--) {
		float* ptr = data + y * width;
		// change invalid pixels (which seem to be represented as -10) to INF
		for (int x = 0; x < width; x++) {
			if (ptr[x] < 0)
				ptr[x] = INFINITY;
		}
		if ((int)fwrite(ptr, sizeof(float), n, stream) != n) {
			fprintf(stderr, "WriteFilePFM: problem writing data\n");
			exit(1);
		}
	}

	// close file
	fclose(stream);
}

int saveFileDat(float* data, int data_num, std::string file_name) {
	FILE* file_dat = NULL;
	errno_t err;
	err = fopen_s(&file_dat, file_name.c_str(), "wb");
	if (err != 0) {
		printf(" Dat file open failed\n");
		return 1;
	}
	// else
	// 	printf(" Open success\n");
	fwrite(data, sizeof(float), data_num, file_dat);
	fclose(file_dat);
}

void genRightCostVolume(int width, int height, uint disp_max, uint disp_min, float* cost_volume_l, float* cost_volume_r) {
	const uint disp_range = disp_max - disp_min;
	const float cost_default = cost_volume_l[0 * width * disp_range + 0 * disp_range + 1];
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			for (int d = disp_min; d < disp_max; d++) {
				float& cost = cost_volume_r[y * width * disp_range + x * disp_range + d - disp_min];
				int x_l = x + d;
				if (x_l < 0 || x_l >= width) { cost = cost_default; continue; }
				cost = cost_volume_l[y * width * disp_range + x_l * disp_range + d - disp_min];
			}
		}
	}
}

int saveCostVolume(float* data, int width, int height, int disp_max, int disp_min, std::string file_name) {
	const uint disp_range = disp_max - disp_min;
	FILE* file_dat = NULL;
	errno_t err;
	// 
	auto*** volume0 = new float** [disp_range];
	for (int i = 0; i < disp_range; i++) {
		volume0[i] = new float* [height];
		for (int j = 0; j < height; j++)
			volume0[i][j] = new float[width];
	}
	// 
	for (int i = 0; i < disp_range; i++)
		for (int j = 0; j < height; j++)
			for (int k = 0; k < width; k++)
				volume0[i][j][k] = data[j * width * disp_range + k * disp_range + i];
	// auto volume0 = new float[disp_range * height * width];
	// for (int y = 0; y < height; y++)
	// 	for (int x = 0; x < width; x++)
	// 		for (int d = 0; d < disp_range; d++)
	// 			volume0[y * width * disp_range + x * disp_range + d] = 0.0;
	err = fopen_s(&file_dat, file_name.c_str(), "wb");
	if (err != 0) {
		printf(" Dat file open failed\n");
		return 1;
	}
	// fwrite(volume0, sizeof(float), disp_range * height * width, file_dat);
	for (int i = 0; i < disp_range; i++)
		for (int j = 0; j < height; j++)
			fwrite(volume0[i][j], sizeof(float), width, file_dat);
	fclose(file_dat);
	delete[] volume0; volume0 = nullptr;
	return 0;
}

void readCostVolume(float* data, int width, int height, int disp_max, int disp_min, std::string file_name) {
	const uint disp_range = disp_max - disp_min;
	// 
	auto*** volume0 = new float** [disp_range];
	for (int i = 0; i < disp_range; i++) {
		volume0[i] = new float* [height];
		for (int j = 0; j < height; j++)
			volume0[i][j] = new float[width];
	}
	// 
	FILE* file_dat = fopen(file_name.c_str(), "rb");
	for (int i = 0; i < disp_range; i++)
		for (int j = 0; j < height; j++)
			fread(volume0[i][j], sizeof(float), width, file_dat);
	// fread(volume0, sizeof(float), disp_range * height * width, file_dat);
	// 
	for (int i = 0; i < disp_range; i++)
		for (int j = 0; j < height; j++)
			for (int k = 0; k < width; k++)
				data[j * width * disp_range + k * disp_range + i] = volume0[i][j][k];
	//
	fclose(file_dat);
	delete[] volume0; volume0 = nullptr;
}

// 
void ShowDisparityMap(const float* disp_map, const int& width, const int& height, const std::string& name)
{
	// ÏÔÊ¾ÊÓ²îÍ¼
	const cv::Mat disp_mat = cv::Mat(height, width, CV_8UC1);
	float min_disp = float(width), max_disp = -float(width);
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			float disp = abs(disp_map[i * width + j]);
			if (disp != Invalid_Float) {
				if (disp >= 64) disp = 64;
				min_disp = std::min(min_disp, disp);
				max_disp = std::max(max_disp, disp);
			}
		}
	}
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			const float disp = abs(disp_map[i * width + j]);
			if (disp == Invalid_Float) {
				disp_mat.data[i * width + j] = 0;
			}
			else {
				disp_mat.data[i * width + j] = static_cast<uchar>((disp - min_disp) / (max_disp - min_disp) * 255);
			}
		}
	}
	cv::namedWindow(name, CV_WINDOW_NORMAL);
	cv::imshow(name, disp_mat);
	char c = (char)cv::waitKey();
	// cv::Mat disp_color;
	// applyColorMap(disp_mat, disp_color, cv::COLORMAP_JET);
	// cv::imshow(name + "-color", disp_color);
}

