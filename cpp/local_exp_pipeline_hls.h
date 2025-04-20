/**
  ******************************************************************************
  * Copyright (C), 2021-2025, Tsinghua Unv. & Pengcheng Lab.
  * Author             : Richard Jiang
  * Email              : jyc14@tsinghua.org.cn
  * History            :
        <author>       <version>       <time>       <desc>
      Richard Jiang      v0.0        2022/11/25     Init
  ******************************************************************************
  */
#pragma once

#if defined(__SYNTHESIS__)
typedef unsigned char uchar;
#else
// opencv library
#include <opencv2/opencv.hpp> 
#ifdef _DEBUG
#pragma comment(lib,"opencv_world310d.lib")
#else
#pragma comment(lib,"opencv_world310.lib")
#endif
#endif

//#include <gmp.h> 
#define __gmp_const const

#define AP_INT_MAX_W 4096
#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_math.h"
#include "hls_streamofblocks.h"
#include <random>

#define PRAGMA_SUB(x) _Pragma (#x)
#define DO_PRAGMA(x) PRAGMA_SUB(x)

#include "PLPPR_hls.h"

#define TMP_HLS 1

#ifndef KITTI
#define KITTI          1 // weight of KITTI and Middlebury is different
#endif
#ifndef LAYER1_GC
#define LAYER1_GC      0 // control whether do GC in layer1
#endif
#ifndef LAYER2_GC
#define LAYER2_GC      0 // control whether do GC in layer2
#endif

#ifndef COST_CALC_MODE
#define COST_CALC_MODE 0 // 
#endif
#ifndef LAYER0_PARALL_DEGREE
#define LAYER0_PARALL_DEGREE 6 // only in COST_CALC_MODE
#endif
#ifndef LAYER1_PARALL_DEGREE
#define LAYER1_PARALL_DEGREE 3 // only in COST_CALC_MODE
#endif
#ifndef LAYER2_PARALL_DEGREE
#define LAYER2_PARALL_DEGREE 1 // only in COST_CALC_MODE
#endif

#ifndef COST_LOAD_MODE
#define COST_LOAD_MODE 1 // 
#endif
#ifndef SCAN_DIRECTION
#define SCAN_DIRECTION 1 // only in COST_LOAD_MODE; 0: horizontal; 1: vertical
#endif

#define PROPOSER_RANDOM_INIT 1 // 0: fix proposer init data; input proposer init data from top

#ifndef WIDTH
#define WIDTH 1242  // 105
#endif
#ifndef HEIGHT
#define HEIGHT 375 // 75
#endif
#ifndef DISP_MAX
#define DISP_MAX 128 // to be reduced to 192
#endif
#define DISP_MIN 0 // fixed to 0

#define DISP_DDR_BIT 8 // bits of 1 cost value; to be reduced to 8
#define DISP_DDR_NUM 16 // number of cost values in 1 DDR address
#define DISP_DDR_NUM0 32
typedef ap_uint<DISP_DDR_BIT*DISP_DDR_NUM*2> unary_cost_t0;
#define DISP_DDR_MAX0 ((DISP_MAX % DISP_DDR_NUM0 == 0) ? (DISP_MAX / DISP_DDR_NUM0) : (DISP_MAX / DISP_DDR_NUM0 + 1))
#define DISP_DDR_NUM_LOG2 4
#define DISP_DDR_MAX ((DISP_MAX % DISP_DDR_NUM == 0) ? (DISP_MAX / DISP_DDR_NUM) : (DISP_MAX / DISP_DDR_NUM + 1)) // number of DDR address to store the cost volume of 1 pixel 
#define DISP_DDR_VAL_MAX ((1 << (DISP_DDR_BIT)) - 1)

#ifndef LAYER0_SIZE
#define LAYER0_SIZE 9
#endif
#ifndef LAYER1_SIZE
#define LAYER1_SIZE 18
#endif
#ifndef LAYER2_SIZE
#define LAYER2_SIZE 36 
#endif
#if LAYER0_SIZE == 5 && LAYER1_SIZE == 15 && LAYER2_SIZE == 30
#include "local_exp_pipeline_hls_ram_mapping_macro_5_15_30.h"
#elif LAYER0_SIZE == 9 && LAYER1_SIZE == 18 && LAYER2_SIZE == 36
#include "local_exp_pipeline_hls_ram_mapping_macro_9_18_36.h"
#else
#error "LAYER_SZIE settings not supported!"
#endif
#ifndef LAYER0_ENABLE
#define LAYER0_ENABLE 1 
#endif
#ifndef LAYER1_ENABLE
#define LAYER1_ENABLE 1 
#endif
#ifndef LAYER2_ENABLE
#define LAYER2_ENABLE 0
#endif
#define LAYER0_HOR_REGION_NUM  ((WIDTH % LAYER0_SIZE == 0) ? (WIDTH / LAYER0_SIZE) : (WIDTH / LAYER0_SIZE + 1))
#define LAYER0_VER_REGION_NUM  ((HEIGHT % LAYER0_SIZE == 0) ? (HEIGHT / LAYER0_SIZE) : (HEIGHT / LAYER0_SIZE + 1))
#define LAYER1_HOR_REGION_NUM  ((WIDTH % LAYER1_SIZE == 0) ? (WIDTH / LAYER1_SIZE) : (WIDTH / LAYER1_SIZE + 1))
#define LAYER1_VER_REGION_NUM  ((HEIGHT % LAYER1_SIZE == 0) ? (HEIGHT / LAYER1_SIZE) : (HEIGHT / LAYER1_SIZE + 1))
#define LAYER2_HOR_REGION_NUM  ((WIDTH % LAYER2_SIZE == 0) ? (WIDTH / LAYER2_SIZE) : (WIDTH / LAYER2_SIZE + 1))
#define LAYER2_VER_REGION_NUM  ((HEIGHT % LAYER2_SIZE == 0) ? (HEIGHT / LAYER2_SIZE) : (HEIGHT / LAYER2_SIZE + 1))
#define LAYER0_HOR_REGION_NUM_PARALL      ((LAYER0_HOR_REGION_NUM % LAYER0_PARALL_DEGREE == 0) ? (LAYER0_HOR_REGION_NUM / LAYER0_PARALL_DEGREE) : (LAYER0_HOR_REGION_NUM / LAYER0_PARALL_DEGREE + 1))
#define LAYER0_HOR_REGION_NUM_PARALL_LAST LAYER0_HOR_REGION_NUM - (LAYER0_PARALL_DEGREE - 1) * LAYER0_HOR_REGION_NUM_PARALL
#define LAYER1_HOR_REGION_NUM_PARALL      ((LAYER1_HOR_REGION_NUM % LAYER1_PARALL_DEGREE == 0) ? (LAYER1_HOR_REGION_NUM / LAYER1_PARALL_DEGREE) : (LAYER1_HOR_REGION_NUM / LAYER1_PARALL_DEGREE + 1))
#define LAYER1_HOR_REGION_NUM_PARALL_LAST LAYER1_HOR_REGION_NUM - (LAYER1_PARALL_DEGREE - 1) * LAYER1_HOR_REGION_NUM_PARALL
#define LAYER2_HOR_REGION_NUM_PARALL      ((LAYER2_HOR_REGION_NUM % LAYER2_PARALL_DEGREE == 0) ? (LAYER2_HOR_REGION_NUM / LAYER2_PARALL_DEGREE) : (LAYER2_HOR_REGION_NUM / LAYER2_PARALL_DEGREE + 1))
#define LAYER2_HOR_REGION_NUM_PARALL_LAST LAYER2_HOR_REGION_NUM - (LAYER2_PARALL_DEGREE - 1) * LAYER2_HOR_REGION_NUM_PARALL

#define HLS_LAYER0_ONLY 0
#define HLS_LAYER1_ONLY 0
#define HLS_LAYER2_ONLY 0

#define LAYER0_PROP_NUM 3
#define LAYER1_PROP_NUM 2

#define LAYER0_PROP_LOOP 2
#define LAYER1_PROP_LOOP 8

#define PRINT_STATUS 0

typedef ap_uint<DISP_DDR_BIT*DISP_DDR_NUM> unary_cost_t;

// typedef ap_uint<108> unary_cost_t;
typedef ap_int<17> label_range;
typedef ap_int<17> disp_range;
typedef ap_fixed<17, 9, AP_RND, AP_SAT> fixed_yy;
typedef int fixed_min;
typedef ap_uint<11> ap_uint11;
typedef ap_uint<10> ap_uint10;
typedef ap_uint<8> ap_uint8;

#if COST_LOAD_MODE == 1 
typedef ap_uint<DISP_DDR_BIT> layer0_cost_blk[LAYER0_SIZE * LAYER0_SIZE];
typedef ap_uint<DISP_DDR_BIT> layer1_cost_blk[LAYER1_SIZE * LAYER1_SIZE];
typedef label_range layer0_label_blk[LAYER0_SIZE * LAYER0_SIZE * 3];
typedef label_range layer1_label_blk[LAYER1_SIZE * LAYER1_SIZE * 3];
typedef disp_range layer0_disp_blk[LAYER0_SIZE * LAYER0_SIZE];
typedef disp_range layer1_disp_blk[LAYER1_SIZE * LAYER1_SIZE];
#elif COST_CALC_MODE == 1
typedef float layer0_cost_blk[LAYER0_PARALL_DEGREE * LAYER0_SIZE * LAYER0_SIZE];
typedef float layer1_cost_blk[LAYER1_PARALL_DEGREE * LAYER1_SIZE * LAYER1_SIZE];
typedef float layer0_label_blk[LAYER0_PARALL_DEGREE * LAYER0_SIZE * LAYER0_SIZE * 3];
typedef float layer1_label_blk[LAYER1_PARALL_DEGREE * LAYER1_SIZE * LAYER1_SIZE * 3];
typedef float layer0_disp_blk[LAYER0_PARALL_DEGREE * LAYER0_SIZE * LAYER0_SIZE];
typedef float layer1_disp_blk[LAYER1_PARALL_DEGREE * LAYER1_SIZE * LAYER1_SIZE];
#endif

#if defined(_DEBUG) || defined(__SYNTHESIS__) || defined(__unix)
const ap_ufixed<31, 7> HLS_PI = 3.1415926535897932384626433832795;
const ap_ufixed<31, 7> HLS_PI_DIV3 = 1.0471975511965977461542144610932;
const ap_ufixed<31, 7> HLS_PI_MUL2 = 6.283185307179586476925286766559;
#else
const float HLS_PI = 3.1415926535897932384626433832795;
#endif

#define PHI_ 0x9e3779b9
const ap_ufixed<31, 7> MINMIN  = 0.0001;
const ap_fixed<32, 7> FIXED_SMALL_VAL = 0.0001;
const ap_fixed<32, 15, AP_RND, AP_SAT> FIXED_SMALL_VAL_16B = 0.0001;
// const ap_fixed<32, 15, AP_RND, AP_SAT> FIXED_ONE = 1;
const ap_uint<17> FIXED_ONE = 0x10000;

uint64_t unaryCostCalcCensus(int x, int y, int win_w, int win_h, uchar gray[WIDTH * HEIGHT]);

// template<size_t WIN_W, size_t WIN_H> 
// uint64_t unaryCostCalcCensus(int x, int y, bool is_sparse, bool sparse_flag[WIN_W * WIN_H], uchar gray[WIDTH * HEIGHT]);

#if COST_LOAD_MODE == 1 
void localExpLayers(
#if PROPOSER_RANDOM_INIT == 1
    ap_uint<32> l0_random_num[LAYER0_PROP_NUM][3], ap_uint<32> l1_random_num[LAYER1_PROP_NUM][3],
#endif // PROPOSER_RANDOM_INIT == 1
    unary_cost_t0 ddr_unary_cost_layer0[WIDTH * HEIGHT * DISP_DDR_MAX0], uchar ddr_im0_gray_layer0[WIDTH * HEIGHT], 
    unary_cost_t0 ddr_unary_cost_layer1[WIDTH * HEIGHT * DISP_DDR_MAX0], uchar ddr_im0_gray_layer1[WIDTH * HEIGHT], 
#if LAYER2_ENABLE == 1
    unary_cost_t ddr_unary_cost_layer2[WIDTH * HEIGHT * DISP_DDR_MAX], uchar ddr_im0_gray_layer2[WIDTH * HEIGHT], 
#endif
    ap_uint<DISP_DDR_BIT> ddr_cost[WIDTH * HEIGHT], label_range ddr_label[WIDTH * HEIGHT * 3], disp_range ddr_disp[WIDTH * HEIGHT]);
    /*
#elif COST_LOAD_MODE == 1 && SCAN_DIRECTION == 1
void localExpLayers(float ddr_unary_cost_layer0[WIDTH * HEIGHT * DISP_MAX], uchar ddr_im0_gray_layer0[WIDTH * HEIGHT], 
    float ddr_unary_cost_layer1[WIDTH * HEIGHT * DISP_MAX], uchar ddr_im0_gray_layer1[WIDTH * HEIGHT], 
    float ddr_unary_cost_layer2[WIDTH * HEIGHT * DISP_MAX], uchar ddr_im0_gray_layer2[WIDTH * HEIGHT], 
    float ddr_cost[WIDTH * HEIGHT], float ddr_label[WIDTH * HEIGHT * 3], float ddr_disp[WIDTH * HEIGHT]);
    */
#elif COST_CALC_MODE == 1
void localExpLayers(uchar ddr_im0_gray_layer0[WIDTH * HEIGHT], uchar ddr_im1_gray_layer0[WIDTH * HEIGHT], uint64_t ddr_im0_census_layer0[WIDTH * HEIGHT], uint64_t ddr_im1_census_layer0[WIDTH * HEIGHT], 
    uchar ddr_im0_gray_layer1[WIDTH * HEIGHT], uchar ddr_im1_gray_layer1[WIDTH * HEIGHT], uint64_t ddr_im0_census_layer1[WIDTH * HEIGHT], uint64_t ddr_im1_census_layer1[WIDTH * HEIGHT],
    uchar ddr_im0_gray_layer2[WIDTH * HEIGHT], uchar ddr_im1_gray_layer2[WIDTH * HEIGHT], uint64_t ddr_im0_census_layer2[WIDTH * HEIGHT], uint64_t ddr_im1_census_layer2[WIDTH * HEIGHT],
    float ddr_cost[WIDTH * HEIGHT], float ddr_label[WIDTH * HEIGHT * 3], float ddr_disp[WIDTH * HEIGHT]);
#endif