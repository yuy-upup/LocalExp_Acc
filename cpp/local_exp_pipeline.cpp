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
#include "local_exp_pipeline.h"
#include "local_exp_opcv.h"
#include "TimeStamper.h"
#include "Utilities.hpp"

extern TimeStamper timer;

void initBinaryCost(int width, int height, cv::Mat imL, cv::Mat imR, std::vector<cv::Mat>& smoothnessCoeff, std::vector<cv::Point> neighbors, const int M, float omega = 10.0f, float epsilon = 0.01f)
{
	cv::Mat I_m;
	cv::copyMakeBorder(imL, I_m, M, M, M, M, cv::BORDER_CONSTANT, cv::Scalar::all(0));
	cv::Rect rect_ee = cv::Rect(M, M, imL.cols, imL.rows);
	cv::Mat IL_ee = I_m(rect_ee);
	smoothnessCoeff.resize(neighbors.size());
	for (int i = 0; i < neighbors.size(); i++)
	{
		cv::Mat IL_nb = I_m(rect_ee + neighbors[i]).clone();
		absdiff(IL_nb, IL_ee, IL_nb);
		cv::exp(-cvutils::channelSum(IL_nb) / omega, smoothnessCoeff[i]);
		smoothnessCoeff[i] = cv::max(epsilon, smoothnessCoeff[i]);
		// set invalid pairwise terms to zero
		if (neighbors[i].x < 0)
			smoothnessCoeff[i].colRange(0, -neighbors[i].x) = 0;
		if (neighbors[i].x > 0)
			smoothnessCoeff[i].colRange(width - neighbors[i].x, width) = 0;
		if (neighbors[i].y < 0)
			smoothnessCoeff[i].rowRange(0, -neighbors[i].y) = 0;
		if (neighbors[i].y > 0)
			smoothnessCoeff[i].rowRange(height - neighbors[i].y, height) = 0;
		cv::Mat tmp;
		cv::copyMakeBorder(smoothnessCoeff[i], tmp, M, M, M, M, cv::BORDER_CONSTANT, cv::Scalar::all(0));
		smoothnessCoeff[i] = tmp;
	}
}

cv::Mat localExpPipelineComputeNormalMap(const cv::Mat& labeling)
{
	cv::Mat normalMap;
	std::vector<cv::Mat> abc;
	cv::split(labeling, abc);
	cv::Mat nzMap = abc[0].mul(abc[0]) + abc[1].mul(abc[1]) + 1.0;
	cv::sqrt(nzMap, nzMap);
	cv::divide(1.0, nzMap, nzMap);
	abc[0] = (abc[0].mul(nzMap, -1.0) + 1.0) / 2.0;
	abc[1] = (abc[1].mul(nzMap, -1.0) + 1.0) / 2.0;
	abc[2] = abc[0];
	abc[0] = nzMap;
	abc.resize(3);
	cv::merge(abc, normalMap);
	return normalMap;
}

void localExpPipeline(const std::string inputDir, const std::string outputDir, std::string volDir, cv::Mat imL, cv::Mat imR, cv::Mat& dispL, cv::Mat& dispR, std::string KITTI_ground_truth) {
    /* output data */
	static disp_range ddr_disp[WIDTH * HEIGHT];
    static ap_uint<DISP_DDR_BIT>  ddr_cost[WIDTH * HEIGHT];
    static label_range ddr_label[WIDTH * HEIGHT * 3];
	/* load two view images */
    cv::Mat grayL, grayR;
	static uchar ddr_im0_gray[WIDTH * HEIGHT];
    static uchar ddr_im1_gray[WIDTH * HEIGHT];
    cv::cvtColor(imL, grayL, cv::COLOR_BGR2GRAY);
    cv::cvtColor(imR, grayR, cv::COLOR_BGR2GRAY);
    for (int y = 0; y < HEIGHT; y++) {
		for (int x = 0; x < WIDTH; x++) {
			ddr_im0_gray[y * WIDTH + x] = grayL.at<uchar>(y, x);
            ddr_im1_gray[y * WIDTH + x] = grayR.at<uchar>(y, x);
		}
	}
	// printf("finish image loading\n");
#if COST_CALC_MODE == 1
    /* calculate census transform */
	static uint64_t ddr_im0_census[WIDTH * HEIGHT];
    static uint64_t ddr_im1_census[WIDTH * HEIGHT];
	for (int y = 0; y < HEIGHT; y++) {
		for (int x = 0; x < WIDTH; x++) {
	        ddr_im0_census[y * WIDTH + x] = unaryCostCalcCensus(x, y, 5, 5, ddr_im0_gray);
			ddr_im1_census[y * WIDTH + x] = unaryCostCalcCensus(x, y, 5, 5, ddr_im1_gray);
		}
	}
	// printf("finish census loading\n");
#endif // COST_CALC_MODE
#if COST_LOAD_MODE == 1
	/* load unary cost */
	/* cost value must between 0 ~ 1 */
    cv::Mat I[2];
	static unary_cost_t0 ddr_unary_cost[WIDTH * HEIGHT * DISP_DDR_MAX0];
    I[0] = cv::Mat(cv::Size(imL.cols, imL.rows), CV_MAKE_TYPE(CV_32F, 3));
	I[1] = cv::Mat(cv::Size(imL.cols, imL.rows), CV_MAKE_TYPE(CV_32F, 3));
	imL.convertTo(I[0], I[0].type());
	imR.convertTo(I[1], I[1].type());
    int sizes[] = { int(DISP_MAX - DISP_MIN), imL.rows, imL.cols };
    cv::Mat vol = cv::Mat_<float>(3, sizes);
	if (cvutils::io::loadMatBinary(inputDir + volDir, vol, false) == false) {
		printf("Cost volume file im0.acrt not found\n");
		return;
	}

	for (int y = 0; y < HEIGHT; y++) {
		for (int x = 0; x < WIDTH; x++) {
//			int min_cost = 300; // 初始化最小代价值为最大整数
//            int min_disp = 0; 
            for (int d_ddr = 0; d_ddr < DISP_DDR_MAX0; d_ddr++) { // DISP_DDR_MAX
				for (int d = 0; d < DISP_DDR_NUM0; d++) { // DISP_DDR_NUM
					ddr_unary_cost[(y * WIDTH + x) * DISP_DDR_MAX0 + d_ddr](d * DISP_DDR_BIT + DISP_DDR_BIT -1, d * DISP_DDR_BIT) = ((d_ddr * DISP_DDR_NUM0 + d) < DISP_MAX) ? round(vol.at<float>(d_ddr * DISP_DDR_NUM0 + d, y, x) * DISP_DDR_VAL_MAX) : DISP_DDR_VAL_MAX;
//				    if (ddr_unary_cost[(y * WIDTH + x) * DISP_DDR_MAX0 + d_ddr](d * DISP_DDR_BIT + DISP_DDR_BIT -1, d * DISP_DDR_BIT)< min_cost) {
//                    min_cost = ddr_unary_cost[(y * WIDTH + x) * DISP_DDR_MAX0 + d_ddr](d * DISP_DDR_BIT + DISP_DDR_BIT -1, d * DISP_DDR_BIT);
//                    min_disp = d_ddr * DISP_DDR_NUM0 + d;
//                    }
			    }
			}
//			ddr_disp[y * WIDTH + x] = min_disp * 256;
		}
	}
    // printf("finish unary cost loading\n");
#endif // COST_LOAD_MODE
    /*  */
#if PROPOSER_RANDOM_INIT == 1
	ap_uint<32> l0_random_num[LAYER0_PROP_NUM][3];
	ap_uint<32> l1_random_num[LAYER1_PROP_NUM][3];
	// std::random_device rd;
	// int random_seed = static_cast<int>(rd());
	// 4236977840
    std::mt19937 gen(1423320506);
    std::uniform_int_distribution<uint32_t> dis(0, UINT32_MAX);
	for (int i = 0; i < LAYER0_PROP_NUM; i++) {
		for (int j = 0; j < 3; j++) {
			l0_random_num[i][j] = dis(gen);
		}
	}
	for (int i = 0; i < LAYER1_PROP_NUM; i++) {
		for (int j = 0; j < 3; j++) {
			l1_random_num[i][j] = dis(gen);
		}
	}
#endif // PROPOSER_RANDOM_INIT == 1
#if COST_LOAD_MODE == 1
    localExpLayers(
#if PROPOSER_RANDOM_INIT == 1
		l0_random_num, l1_random_num, 
#endif // PROPOSER_RANDOM_INIT == 1
	    ddr_unary_cost, ddr_im0_gray, 
		ddr_unary_cost, ddr_im0_gray, 
#if LAYER2_ENABLE == 1
		ddr_unary_cost, ddr_im0_gray, 
#endif
		ddr_cost, ddr_label, ddr_disp);
#elif COST_CALC_MODE == 1
    localExpLayers(ddr_im0_gray, ddr_im1_gray, ddr_im0_census, ddr_im1_census, ddr_im0_gray, ddr_im1_gray, ddr_im0_census, ddr_im1_census, ddr_im0_gray, ddr_im1_gray, ddr_im0_census, ddr_im1_census, ddr_cost, ddr_label, ddr_disp);
#endif
    /* write disparity to binary file */
	// int disp_size = sizeof(ddr_disp)/sizeof(ddr_disp[0]);
	// std::ofstream disp_out(outputDir + "disp.bin", std::ios::binary);
	// if (disp_out.is_open()) {
    //     disp_out.write((char*)ddr_disp, sizeof(ddr_disp));
    //     disp_out.close();
    //     // std::cout << "File was written successfully." << std::endl;
    // } else {
    //     std::cout << "unable to open disp.bin." << std::endl;
    // }
	/* result analysis */
	std::vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(0);
	// 
	int para=pow(2,8);
    cv::Mat labeling = cv::Mat::zeros(HEIGHT, WIDTH, CV_MAKE_TYPE(cv::DataType<float>::depth, 3));
    for (int y = 0; y < HEIGHT; y++) {
		for (int x = 0; x < WIDTH; x++) {
			float ddr_label0=ddr_label[(y * WIDTH + x) * 3 + 0];
			float ddr_label1=ddr_label[(y * WIDTH + x) * 3 + 1];
			float ddr_label2=ddr_label[(y * WIDTH + x) * 3 + 2];
			labeling.at<cv::Vec<float, 3>>(y, x)(0) = ddr_label0/para;
			labeling.at<cv::Vec<float, 3>>(y, x)(1) = ddr_label1/para;
			labeling.at<cv::Vec<float, 3>>(y, x)(2) = ddr_label2/para;
		}
	}
    cv::Mat normalMapVis = localExpPipelineComputeNormalMap(labeling);
    cv::imwrite(outputDir + "normal_map.png", normalMapVis * 255);
    //
    cv::Mat disp = cv::Mat::zeros(HEIGHT, WIDTH, CV_MAKE_TYPE(cv::DataType<float>::depth, 1));
    for (int y = 0; y < HEIGHT; y++) {
		for (int x = 0; x < WIDTH; x++) {
			float ddr_disp_f=ddr_disp[y * WIDTH + x];
			disp.at<float>(y, x) = ddr_disp_f/para;
		}
	}
	// cv::imwrite(outputDir + "disp.png", disp, compression_params);
#if KITTI
	cv::Mat disp_16b(disp.rows, disp.cols, CV_16U);
	disp = (disp * 256.0);
	disp.convertTo(disp_16b, disp_16b.type());
    cv::imwrite(outputDir + "disp.png", disp_16b, compression_params);
	// Calculate error rate
	cv::Mat D_est(disp.rows, disp.cols, CV_32FC1);
	disp_16b.convertTo(D_est, CV_32FC1);
#ifdef __unix
	cv::Mat D_gt = cv::imread(inputDir + KITTI_ground_truth, cv::IMREAD_ANYDEPTH);
#else
	cv::Mat D_gt = cv::imread(inputDir + KITTI_ground_truth, cv::IMREAD_ANYDEPTH); 
#endif
	D_gt.convertTo(D_gt, CV_32FC1);
	D_est /= 256.0; D_gt /= 256.0;
	int n_err = 0, n_total = 0;
	cv::Mat E = cv::abs(D_gt - D_est);
	n_total = cv::countNonZero(D_gt > 0);
	D_gt.setTo(1e7, D_gt == 0);
	n_err = cv::countNonZero((D_gt > 0) & (D_gt < 1e7) & (E > 3) & (E / abs(D_gt) > 0.05));
	float D_err = static_cast<float>(n_err) / static_cast<float>(n_total);
	std::cout << KITTI_ground_truth << " error rate: " << D_err * 100 << " %" << std::endl;
#else
    // 
    cv::Mat disp_16b(disp.rows, disp.cols, CV_16U);
	disp = (disp * 256.0);
	disp.convertTo(disp_16b, disp_16b.type());
    cv::imwrite(outputDir + "disp.png", disp_16b, compression_params);
# if 0
    // 
    cv::Mat dispGT = cv::imread(inputDir + "groundtruth.png", cv::IMREAD_GRAYSCALE);
    if (!dispGT.empty())
	{
		// if (calib.gt_prec > 0)
		dispGT.convertTo(dispGT, CV_32F, 0.25);
		dispGT.setTo(cv::Scalar(INFINITY), dispGT == 0);
	}
	else
	{
		dispGT = cvutils::io::read_pfm_file(inputDir + "disp0GT.pfm");
	}
	if (dispGT.empty())
		dispGT = cv::Mat_<float>::zeros(imL.size());
	cv::Mat nonoccMask = cv::imread(inputDir + "nonocc.png", cv::IMREAD_GRAYSCALE);
	if (nonoccMask.empty())
		nonoccMask = cv::imread(inputDir + "mask0nocc.png", cv::IMREAD_GRAYSCALE);

	if (!nonoccMask.empty())
		nonoccMask = nonoccMask == 255;
	else
		nonoccMask = cv::Mat_<uchar>(imL.size(), 255);
	float errorThreshold = 1.0f;
	cv::Mat validMask= (dispGT > 0.0f) & (dispGT != INFINITY);
	cv::Mat occMask = ~nonoccMask & validMask;
	int validPixels = cv::countNonZero(validMask);
	int nonoccPixels = cv::countNonZero(nonoccMask);
    cv::Mat errorMap = cv::abs(disp - dispGT) <= errorThreshold;
	cv::Mat errorMapVis = errorMap | (~validMask);
	errorMapVis.setTo(cv::Scalar(200), occMask & (~errorMapVis));
	double all = 1.0 - (double)cv::countNonZero(errorMap & validMask) / validPixels;
	double nonocc = 1.0 - (double)cv::countNonZero(errorMap & nonoccMask) / nonoccPixels;
	all *= 100.0;
	nonocc *= 100.0;
    cv::imwrite(outputDir + "error.png", errorMapVis);
    // 
    printf("error rate: %f\n", nonocc);
#endif
#if 0
    std::cout << "ddr_cost address: " << ddr_cost << std::endl;
    std::cout << "ddr_label address: " << ddr_label << std::endl;
    std::cout << "ddr_disp address: " << ddr_disp << std::endl;
    printf("ddr store finished\n");
#endif
#if 0
	// dispL.create(HEIGHT, WIDTH, CV_32FC1);
	// output disp result
	// for (int y = 0; y < HEIGHT; y++) {
	// 	for (int x = 0; x < WIDTH; x++) {
	// 		dispL.at<float>(y, x) = ddr_disp[y * WIDTH + x];
	// 	}
	// }
	cv::imwrite(outputDir + "disp.png", disp, compression_params);
	cvutils::io::save_pfm_file(outputDir + "disp0HLS.pfm", disp);
#endif
#endif
}