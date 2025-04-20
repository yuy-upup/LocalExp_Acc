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
#include "local_exp_pipeline_hls.h"
#include "sin_lut.h"
#if 1
float unaryCostCalcHammDist(uint64_t& x, uint64_t& y) {
	int dist = 0;
	uint64_t hamming = x ^ y;
	while (hamming != 0) {
		dist++;
		hamming &= hamming - 1;
	}
	return float(dist);
}

uint64_t unaryCostCalcCensus(int x, int y, int win_w, int win_h, uchar gray[WIDTH * HEIGHT]) {
	int half_win_h = (win_h - 1) / 2;
    int half_win_w = (win_w - 1) / 2;
	uint64_t census = 0u;
	uchar pixel_c = gray[y * WIDTH + x];
	uchar pixel_n;
	for (int i = -half_win_h; i <= half_win_h; i++) {
		for (int j = -half_win_w; j <= half_win_w; j++) {
            census <<= 1;
			if ((y + i) < 0 || (y + i) >= HEIGHT || (x + j) < 0 || (x + j) >= WIDTH) 
                pixel_n = pixel_c;
			else 
                pixel_n = gray[(y + i) * WIDTH + (x + j)];
			if (pixel_n < pixel_c) {
                census += 1;
            }
		}
	}
	return census;
}

float unaryCostCalc(int x, int y, int d, uchar ddr_im0_gray[WIDTH * HEIGHT], uchar ddr_im1_gray[WIDTH * HEIGHT], 
    uint64_t ddr_im0_census[WIDTH * HEIGHT], uint64_t ddr_im1_census[WIDTH * HEIGHT]) {
    float cost = 0.0;
    int x_r = ((x - d) < 0) ? 0 : (x - d);
    /* ad */
    uchar gray_l = ddr_im0_gray[y * WIDTH + x];
    uchar gray_r = ddr_im1_gray[y * WIDTH + x_r];
    cost += std::abs((float)gray_l - (float)gray_r) / 64.0;
    /* census */
    uint64_t census_l = ddr_im0_census[y * WIDTH + x];
    uint64_t census_r = ddr_im1_census[y * WIDTH + x_r];
    cost += (unaryCostCalcHammDist(census_l, census_r) / 32);
    /* threshold */
    return std::fmin(cost, (float)1.0);
}

float unaryCostCalc(uchar gray_l, uchar gray_r, uint64_t census_l, uint64_t census_r) {
    /* ad + census */
    float cost = 0.0;
    cost += std::abs((float)gray_l - (float)gray_r) / 64.0;
    cost += (unaryCostCalcHammDist(census_l, census_r) / 32);
    /* threshold */
    return std::fmin(cost, (float)1.0);
}

ap_uint<32> randMwc(ap_uint<32>* q, ap_uint<32>& i, ap_uint<32>& c)
{
#pragma HLS inline off
    ap_uint<64> t, a = 18782LL;
    // static ap_uint<32> i = 4095;
    ap_uint<32> x, r = 0xfffffffe;
    i = (i + 1) & 4095;
    t = a * q[i] + c;
    c = (t >> 32);
    x = t + c;
    if (x < c) {
        x++;
        c++;
    }
    return (q[i] = r - x);
}

template<size_t CUR_SIZE>
void layerExpnProposal(int ver_region_i, ap_uint<32>* q, ap_uint<32>& i, ap_uint<32>& c, 
    int expn_region_x, int expn_region_y, int unit_region_x, int unit_region_y, 
    int unit_region_w, int unit_region_h, float local_label[CUR_SIZE * 3][CUR_SIZE * 3][3], ap_fixed<32, 15, AP_RND, AP_SAT>& p_a, ap_fixed<32, 15, AP_RND, AP_SAT>& p_b, ap_fixed<32, 15, AP_RND, AP_SAT>& p_c) {
#pragma HLS inline
    // select a pixel randomly
    int p_x = randMwc(q, i, c) % unit_region_w;
    int p_y = randMwc(q, i, c) % unit_region_h;
    //
    int uram_x = p_x + unit_region_x - expn_region_x;
    int uram_y = p_y + unit_region_y - expn_region_y;
    // output expansion label
    p_a = local_label[uram_y][uram_x][0];
    p_b = local_label[uram_y][uram_x][1];
    p_c = local_label[uram_y][uram_x][2];
    // printf("p_x=%s, p_y=%s\n", p_x.to_string(10).c_str(), p_y.to_string(10).c_str());
}

void layer0ExpnProposal(int ver_region_i, ap_uint<32>* q, ap_uint<32>& i, ap_uint<32>& c, 
    int expn_region_x, int expn_region_y, int unit_region_x, int unit_region_y, int unit_region_w, int unit_region_h, /* int base_h, */
    float local_label[LAYER0_SIZE * 3][LAYER0_SIZE * 3][3], ap_fixed<32, 15, AP_RND, AP_SAT>& p_a, ap_fixed<32, 15, AP_RND, AP_SAT>& p_b, ap_fixed<32, 15, AP_RND, AP_SAT>& p_c) {
#pragma HLS inline off
    // select a pixel randomly
    int p_x = randMwc(q, i, c) % unit_region_w;
    int p_y = randMwc(q, i, c) % unit_region_h;
    // int p_y = randMwc(q, i, c) % (unit_region_h/3);
    //
    int uram_x = p_x + unit_region_x - expn_region_x;
    int uram_y = p_y + unit_region_y - expn_region_y;
    // int uram_y = p_y + unit_region_y - expn_region_y + base_h;
    // output expansion label
    p_a = local_label[uram_y][uram_x][0];
    p_b = local_label[uram_y][uram_x][1];
    p_c = local_label[uram_y][uram_x][2];
    // printf("p_x=%s, p_y=%s\n", p_x.to_string(10).c_str(), p_y.to_string(10).c_str());
}

template<size_t CUR_SIZE>
void layerRandProposal(int ver_region_i, ap_uint<32>* q, ap_uint<32>& i, ap_uint<32>& c, 
    int outer_iter, int inner_iter, int expn_region_x, int expn_region_y, 
    int unit_region_x, int unit_region_y, int unit_region_w, int unit_region_h,
    float local_label[CUR_SIZE * 3][CUR_SIZE * 3][3], ap_fixed<32, 15, AP_RND, AP_SAT>& p_a, ap_fixed<32, 15, AP_RND, AP_SAT>& p_b, ap_fixed<32, 15, AP_RND, AP_SAT>& p_c) {
#pragma HLS inline
    //
    // ap_fixed<32, 15, AP_RND, AP_SAT> p_a_fixed, p_b_fixed, p_c_fixed;
    // 
    ap_ufixed<16, 1> half_fixed = 0.5;
    ap_ufixed<16, 1> one_fixed = 1.0;
    ap_fixed<32, 15, AP_RND, AP_SAT> disp_max = DISP_MAX;
    ap_fixed<32, 15, AP_RND, AP_SAT> disp_min = DISP_MIN;
    ap_uint<8> power;
    // select a pixel randomly
    ap_uint<12> p_x = randMwc(q, i, c) % unit_region_w;
    ap_uint<12> p_y = randMwc(q, i, c) % unit_region_h;
    // printf("p_x=%s, p_y=%s, ", p_x.to_string(10).c_str(), p_y.to_string(10).c_str());
    //
    int uram_x = p_x + unit_region_x - expn_region_x;
    int uram_y = p_y + unit_region_y - expn_region_y;
    // int uram_x, uram_y;
    // get label of selected pixel
    ap_fixed<32, 15, AP_RND, AP_SAT> init_a = local_label[uram_y][uram_x][0];
    ap_fixed<32, 15, AP_RND, AP_SAT> init_b = local_label[uram_y][uram_x][1];
    ap_fixed<32, 15, AP_RND, AP_SAT> init_c = local_label[uram_y][uram_x][2];
    //
    ap_uint<8> total_iter = outer_iter + inner_iter;
    // calculate initial disp
    ap_uint<12> p_x_abs = unit_region_x + p_x; // absolute coordinate
    ap_uint<12> p_y_abs = unit_region_y + p_y;
    ap_fixed<32, 15, AP_RND, AP_SAT> zs = init_a * p_x_abs + init_b * p_y_abs + init_c;
    // printf("zs=%s \n", zs.to_string(10).c_str());
    // get random z
    power = total_iter + 1;
    // float_fixed dz = (disp_max - disp_min) * hls::pow(half_fixed, power);
    ap_fixed<32, 15, AP_RND, AP_SAT> dz = (disp_max - disp_min) * (half_fixed >> power);
    // printf("dz=%s \n", dz.to_string(10).c_str());
    ap_fixed<32, 15, AP_RND, AP_SAT> zs_min = zs - dz;
    ap_fixed<32, 15, AP_RND, AP_SAT> zs_max = zs + dz;
    ap_fixed<32, 15, AP_RND, AP_SAT> minz = hls::max(disp_min, zs_min);
    ap_fixed<32, 15, AP_RND, AP_SAT> maxz = hls::min(disp_max, zs_max);
    zs.range() = randMwc(q, i, c)(31, 0);
    // printf("zs=%s \n", zs.to_string(10).c_str());
    // printf("iter=%s, Q=%s \n", iter.to_string(10).c_str(), Q[iter].to_string(10).c_str());
    zs = zs - hls::floor(zs / (maxz - minz)) * (maxz - minz) + minz;
    // printf("out=%s, in=%s, min=%s, max=%s, zs=%s \n", outer_iter.to_string(10).c_str(), inner_iter.to_string(10).c_str(), minz.to_string(10).c_str(), maxz.to_string(10).c_str(), zs.to_string(10).c_str());
    // get initial normal vector
    power = total_iter;
    ap_fixed<32, 15, AP_RND, AP_SAT> nr = one_fixed * (half_fixed >> power);
    ap_ufixed<32, 7> nz_dec = one_fixed + init_a * init_a + init_b * init_b;
    // printf("nz_dec^2=%s \n", nz_dec.to_string(10).c_str());
    nz_dec = hls::sqrt(nz_dec);
    // printf("nz_dec=%s \n", nz_dec.to_string(10).c_str());
    ap_fixed<32, 15, AP_RND, AP_SAT> nz = one_fixed / nz_dec;
    ap_fixed<32, 15, AP_RND, AP_SAT> nx = -init_a * nz;
    ap_fixed<32, 15, AP_RND, AP_SAT> ny = -init_b * nz;
    // printf("nx=%s, ny=%s, nz=%s\n", nx.to_string(10).c_str(), ny.to_string(10).c_str(), nz.to_string(10).c_str());
    // get random unit vetor
    ap_ufixed<32, 7> theta;
    theta.range() = randMwc(q, i, c)(31, 0);
    theta = theta - hls::floor(theta / (HLS_PI)) * (HLS_PI);
    ap_ufixed<32, 7> phi;
    phi.range() = randMwc(q, i, c)(31, 0);
    phi = phi - hls::floor(phi / (2*HLS_PI)) * (2*HLS_PI);
    ap_fixed<32, 7> cosT = cosf(theta), sinT = sinf(theta);
    ap_fixed<32, 7> cosP = cosf(phi), sinP = sinf(phi);
    ap_fixed<32, 7> unit_n[3];
    unit_n[0] = sinT * cosP;
    unit_n[1] = sinT * sinP;
    unit_n[2] = cosT;
    // printf("unit_n[0]=%s, unit_n[1]=%s, unit_n[2]=%s\n", unit_n[0].to_string(10).c_str(), unit_n[1].to_string(10).c_str(), unit_n[2].to_string(10).c_str());
    // get new normal vector
    ap_fixed<32, 15, AP_RND, AP_SAT> nv[3];
    nv[0] = nx + unit_n[0] * nr;
    nv[1] = ny + unit_n[1] * nr;
    nv[2] = nz + unit_n[2] * nr;
    ap_ufixed<32, 7> nv_norm_dec = nv[0] * nv[0] + nv[1] * nv[1] + nv[2] * nv[2];
    ap_fixed<32, 15, AP_RND, AP_SAT> nv_norm = hls::sqrt(nv_norm_dec);
    // float_fixed nv_norm = hls::sqrt(nv[0] * nv[0] + nv[1] * nv[1] + nv[2] * nv[2]);
    nv_norm = (nv_norm != 0) ? nv_norm : FIXED_SMALL_VAL_16B;
    nv[0] = nv[0] / nv_norm;
    nv[1] = nv[1] / nv_norm;
    nv[2] = nv[2] / nv_norm;
    // create label based on new normal vector & random z
    nv[2] = (nv[2] != 0) ? nv[2] : FIXED_SMALL_VAL_16B;
    p_a = -nv[0] / nv[2];
    p_b = -nv[1] / nv[2];
    p_c = zs - p_a * p_x_abs - p_b * p_y_abs;
    // printf("p_a=%s, p_b=%s, p_c=%s\n", p_a.to_string(10).c_str(), p_b.to_string(10).c_str(), p_c.to_string(10).c_str());
    // inner_iter++;
    // p_a = p_a_fixed.to_float();
    // p_b = p_b_fixed.to_float();
    // p_c = p_c_fixed.to_float();
}
#define LUT_SIZE 4096
#define LUT_SCALE (LUT_SIZE / (2 * HLS_PI))  // 弧度到索引的缩放因子

// 通过LUT实现sin和cos
template<typename T>
T lut_sin(T radian) {
    // 将弧度映射到0~2π
    T normalized_rad = radian - hls::floor(radian / (2*HLS_PI)) * (2*HLS_PI);
    // 计算LUT索引（转换为定点数后取模）
    ap_ufixed<32, 12> index_float = normalized_rad * LUT_SCALE;
    int index = index_float.to_int() % LUT_SIZE;
    return sin_lut[index];
}

template<typename T>
T lut_cos(T radian) {
    // cos(theta) = sin(theta + π/2)
    return lut_sin(radian + HLS_PI / 2);
}


void layer0RandProposal(ap_uint<32>* q, ap_uint<32>& i, ap_uint<32>& c, int outer_iter, int inner_iter, ap_uint<12> expn_region_x, ap_uint<12> expn_region_y, ap_uint<12> unit_region_x, ap_uint<12> unit_region_y, ap_uint<7> unit_region_w, ap_uint<7> unit_region_h, 
    ap_uint<12> p_x, ap_uint<12> p_y, label_range init_a, label_range init_b, label_range init_c, label_range& p_a, label_range& p_b, label_range & p_c) {
#pragma HLS inline off
    int para=pow(2,8);
    fixed_yy init_a_fixed; init_a_fixed.range()=init_a;
    fixed_yy init_b_fixed; init_b_fixed.range()=init_b;
    fixed_yy init_c_fixed; init_c_fixed.range()=init_c;
    fixed_yy p_a_fixed,p_b_fixed,p_c_fixed;
    //
    // ap_fixed<32, 15, AP_RND, AP_SAT> p_a_fixed, p_b_fixed, p_c_fixed;
    // 
    // if (is_init == true) {
    //     p_a = init_a; p_b = init_b; p_c = init_c; 
    //     return;
    // }
    ap_ufixed<16, 1> half_fixed = 0.5;
    ap_ufixed<16, 1> one_fixed = 1.0;
    ap_fixed<32, 15, AP_RND, AP_SAT> disp_max = DISP_MAX;
    ap_fixed<32, 15, AP_RND, AP_SAT> disp_min = DISP_MIN;
    ap_uint<8> power;
    /*  */
    ap_uint<8> total_iter = outer_iter + inner_iter;
    // calculate initial disp
    ap_uint<12> p_x_abs = unit_region_x + p_x; // absolute coordinate
    ap_uint<12> p_y_abs = unit_region_y + p_y;
    ap_fixed<17, 9, AP_RND, AP_SAT> zs = init_a_fixed * p_x_abs + init_b_fixed * p_y_abs + init_c_fixed;
    // printf("zs=%s \n", zs.to_string(10).c_str());
    // get random z
    power = total_iter + 1;
    // float_fixed dz = (disp_max - disp_min) * hls::pow(half_fixed, power);
    ap_fixed<32, 15, AP_RND, AP_SAT> dz = (disp_max - disp_min) * (half_fixed >> power);
    // printf("dz=%s \n", dz.to_string(10).c_str());
    ap_fixed<32, 15, AP_RND, AP_SAT> zs_min = zs - dz;
    ap_fixed<32, 15, AP_RND, AP_SAT> zs_max = zs + dz;
    ap_fixed<32, 15, AP_RND, AP_SAT> minz = hls::max(disp_min, zs_min);
    ap_fixed<32, 15, AP_RND, AP_SAT> maxz = hls::min(disp_max, zs_max);
      zs.range() = randMwc(q, i, c)(31, 0);
    // printf("zs=%s \n", zs.to_string(10).c_str());
    // printf("iter=%s, Q=%s \n", iter.to_string(10).c_str(), Q[iter].to_string(10).c_str());
    if(maxz==minz) maxz=maxz+FIXED_SMALL_VAL_16B;
assert((maxz - minz) != 0);
    // ap_fixed<32, 15, AP_RND, AP_SAT> d_z = ((maxz - minz) != 0) ? (ap_fixed<32, 15, AP_RND, AP_SAT>)(maxz - minz) : FIXED_SMALL_VAL_16B;
    zs = zs - hls::floor(zs / (maxz - minz)) * (maxz - minz) + minz; 
    // printf("out=%s, in=%s, min=%s, max=%s, zs=%s \n", outer_iter.to_string(10).c_str(), inner_iter.to_string(10).c_str(), minz.to_string(10).c_str(), maxz.to_string(10).c_str(), zs.to_string(10).c_str());
    // get initial normal vector
    power = total_iter;
    ap_fixed<32, 15, AP_RND, AP_SAT> nr = one_fixed * (half_fixed >> power);
    ap_ufixed<31, 7> nz_dec = one_fixed + init_a_fixed * init_a_fixed + init_b_fixed * init_b_fixed;
    // printf("nz_dec^2=%s \n", nz_dec.to_string(10).c_str());
assert(nz_dec != 0);
    ap_uint<31> nz_int;
    nz_int = nz_dec.range(30, 0);
    nz_int = hls::sqrt(nz_int);
    ap_ufixed<31, 15> nz_fixed = nz_int;
    nz_dec = nz_fixed >> 12;
    if(nz_dec==0) nz_dec = MINMIN;
assert(nz_dec != 0);
    ap_fixed<32, 15, AP_RND, AP_SAT> nz = one_fixed / nz_dec;
    ap_fixed<32, 15, AP_RND, AP_SAT> nx = -init_a_fixed * nz;
    ap_fixed<32, 15, AP_RND, AP_SAT> ny = -init_b_fixed * nz;
    // printf("nx=%s, ny=%s, nz=%s\n", nx.to_string(10).c_str(), ny.to_string(10).c_str(), nz.to_string(10).c_str());
    // get random unit vetor
    ap_ufixed<32, 7> theta;
    theta.range() = randMwc(q, i, c)(31, 0);
//    theta = theta - hls::floor(theta / (HLS_PI)) * (HLS_PI);
    
    ap_ufixed<32, 7> phi;
    phi.range() = randMwc(q, i, c)(31, 0);
//    phi = phi - hls::floor(phi / (2*HLS_PI)) * (2*HLS_PI);
//    ap_fixed<32, 7> cosT = cosf(theta), sinT = sinf(theta);
//    ap_fixed<32, 7> cosP = cosf(phi), sinP = sinf(phi);
    ap_fixed<32, 7> cosT = lut_cos(theta);
    ap_fixed<32, 7> sinT = lut_sin(theta);
    ap_fixed<32, 7> cosP = lut_cos(phi);
    ap_fixed<32, 7> sinP = lut_sin(phi);
    // ap_fixed<32, 7> cosT = hls::cosf(theta), sinT = hls::sinf(theta);
    // ap_fixed<32, 7> cosP = hls::cosf(phi), sinP = hls::sinf(phi);
    ap_fixed<32, 7> unit_n[3];
    unit_n[0] = sinT * cosP; 
    unit_n[1] = sinT * sinP;
    unit_n[2] = cosT;
    // printf("unit_n[0]=%s, unit_n[1]=%s, unit_n[2]=%s\n", unit_n[0].to_string(10).c_str(), unit_n[1].to_string(10).c_str(), unit_n[2].to_string(10).c_str());
    // get new normal vector
    ap_fixed<32, 15, AP_RND, AP_SAT> nv[3];
    nv[0] = nx + unit_n[0] * nr;
    nv[1] = ny + unit_n[1] * nr;
    nv[2] = nz + unit_n[2] * nr;
    ap_ufixed<31, 7> nv_norm_dec = nv[0] * nv[0] + nv[1] * nv[1] + nv[2] * nv[2];
    assert(nv_norm_dec != 0);
    ap_uint<31> nv_int;
    nv_int = nv_norm_dec.range(30, 0);
    nv_int = hls::sqrt(nv_int);
    ap_ufixed<31, 15> nv_fixed = nv_int;
    ap_fixed<32, 15, AP_RND, AP_SAT> nv_norm = nv_fixed >> 12;
    //ap_fixed<32, 15, AP_RND, AP_SAT> nv_norm = hls::sqrt(nv_norm_dec);
    // float_fixed nv_norm = hls::sqrt(nv[0] * nv[0] + nv[1] * nv[1] + nv[2] * nv[2]);
    nv_norm = (nv_norm != 0) ? nv_norm : FIXED_SMALL_VAL_16B;
    assert(nv_norm != 0);
    nv[0] = nv[0] / nv_norm;
    nv[1] = nv[1] / nv_norm;
    nv[2] = nv[2] / nv_norm;
    // create label based on new normal vector & random z
    nv[2] = (nv[2] != 0) ? nv[2] : FIXED_SMALL_VAL_16B;
    assert(nv[2] != 0);
    p_a_fixed = -nv[0] / nv[2];
    p_b_fixed = -nv[1] / nv[2];
    p_c_fixed = zs - p_a_fixed * p_x_abs - p_b_fixed * p_y_abs;
    p_a=(p_a_fixed*para).to_ap_int(); p_b=(p_b_fixed*para).to_ap_int(); p_c=(p_c_fixed*para).to_ap_int();
    // printf("p_a=%s, p_b=%s, p_c=%s\n", p_a.to_string(10).c_str(), p_b.to_string(10).c_str(), p_c.to_string(10).c_str());
    // inner_iter++;
    // p_a = p_a_fixed.to_float();
    // p_b = p_b_fixed.to_float();
    // p_c = p_c_fixed.to_float();
}

void layer0RandProposal(ap_uint<32> pp[3], int outer_iter, int inner_iter, ap_uint<12> expn_region_x, ap_uint<12> expn_region_y, ap_uint<12> unit_region_x, ap_uint<12> unit_region_y, ap_uint<7> unit_region_w, ap_uint<7> unit_region_h, 
    ap_uint<12> p_x, ap_uint<12> p_y, label_range init_a, label_range init_b, label_range init_c, label_range& p_a, label_range& p_b, label_range & p_c) {
#pragma HLS inline off
    int para=pow(2,8);
    fixed_yy init_a_fixed; init_a_fixed.range()=init_a;
    fixed_yy init_b_fixed; init_b_fixed.range()=init_b;
    fixed_yy init_c_fixed; init_c_fixed.range()=init_c;
    fixed_yy p_a_fixed,p_b_fixed,p_c_fixed;
    //
    // ap_fixed<32, 15, AP_RND, AP_SAT> p_a_fixed, p_b_fixed, p_c_fixed;
    // 
    // if (is_init == true) {
    //     p_a = init_a; p_b = init_b; p_c = init_c; 
    //     return;
    // }
    ap_ufixed<16, 1> half_fixed = 0.5;
    ap_ufixed<16, 1> one_fixed = 1.0;
    ap_fixed<32, 15, AP_RND, AP_SAT> disp_max = DISP_MAX;
    ap_fixed<32, 15, AP_RND, AP_SAT> disp_min = DISP_MIN;
    ap_uint<8> power;
    /*  */
    ap_uint<8> total_iter = outer_iter + inner_iter;
    // calculate initial disp
    ap_uint<12> p_x_abs = unit_region_x + p_x; // absolute coordinate
    ap_uint<12> p_y_abs = unit_region_y + p_y;
    ap_fixed<17, 9, AP_RND, AP_SAT> zs = init_a_fixed * p_x_abs + init_b_fixed * p_y_abs + init_c_fixed;
    // printf("zs=%s \n", zs.to_string(10).c_str());
    // get random z
    power = total_iter + 1;
    // float_fixed dz = (disp_max - disp_min) * hls::pow(half_fixed, power);
    ap_fixed<32, 15, AP_RND, AP_SAT> dz = (disp_max - disp_min) * (half_fixed >> power);
    // printf("dz=%s \n", dz.to_string(10).c_str());
    ap_fixed<32, 15, AP_RND, AP_SAT> zs_min = zs - dz;
    ap_fixed<32, 15, AP_RND, AP_SAT> zs_max = zs + dz;
    ap_fixed<32, 15, AP_RND, AP_SAT> minz = hls::max(disp_min, zs_min);
    ap_fixed<32, 15, AP_RND, AP_SAT> maxz = hls::min(disp_max, zs_max);
//      zs.range() = randMwc(q, i, c)(31, 0);
      zs.range() = pp[0];
    // printf("zs=%s \n", zs.to_string(10).c_str());
    // printf("iter=%s, Q=%s \n", iter.to_string(10).c_str(), Q[iter].to_string(10).c_str());
    if(maxz==minz) maxz=maxz+FIXED_SMALL_VAL_16B;
assert((maxz - minz) != 0);
    // ap_fixed<32, 15, AP_RND, AP_SAT> d_z = ((maxz - minz) != 0) ? (ap_fixed<32, 15, AP_RND, AP_SAT>)(maxz - minz) : FIXED_SMALL_VAL_16B;
    zs = zs - hls::floor(zs / (maxz - minz)) * (maxz - minz) + minz; 
    // printf("out=%s, in=%s, min=%s, max=%s, zs=%s \n", outer_iter.to_string(10).c_str(), inner_iter.to_string(10).c_str(), minz.to_string(10).c_str(), maxz.to_string(10).c_str(), zs.to_string(10).c_str());
    // get initial normal vector
    power = total_iter;
    ap_fixed<32, 15, AP_RND, AP_SAT> nr = one_fixed * (half_fixed >> power);
    ap_ufixed<31, 7> nz_dec = one_fixed + init_a_fixed * init_a_fixed + init_b_fixed * init_b_fixed;
    // printf("nz_dec^2=%s \n", nz_dec.to_string(10).c_str());
    assert(nz_dec != 0);
    ap_uint<31> nz_int;
    nz_int = nz_dec.range(30, 0);
    nz_int = hls::sqrt(nz_int);
    ap_ufixed<31, 15> nz_fixed = nz_int;
    nz_dec = nz_fixed >> 12;
    if(nz_dec==0) nz_dec = MINMIN;
    assert(nz_dec != 0);
    ap_fixed<32, 15, AP_RND, AP_SAT> nz = one_fixed / nz_dec;
    ap_fixed<32, 15, AP_RND, AP_SAT> nx = -init_a_fixed * nz;
    ap_fixed<32, 15, AP_RND, AP_SAT> ny = -init_b_fixed * nz;
    // printf("nx=%s, ny=%s, nz=%s\n", nx.to_string(10).c_str(), ny.to_string(10).c_str(), nz.to_string(10).c_str());
    // get random unit vetor
    ap_ufixed<32, 7> theta;
    theta.range() = pp[1];
//    theta = theta - hls::floor(theta / (HLS_PI)) * (HLS_PI);
    ap_ufixed<32, 7> phi;
    phi.range() = pp[2];
//    phi = phi - hls::floor(phi / (2*HLS_PI)) * (2*HLS_PI);
/*
    ap_fixed<32, 7> cosT = cosf(theta), sinT = sinf(theta);
    ap_fixed<32, 7> cosP = cosf(phi), sinP = sinf(phi);
*/
    ap_fixed<32, 7> cosT = lut_cos(theta);
    ap_fixed<32, 7> sinT = lut_sin(theta);
    ap_fixed<32, 7> cosP = lut_cos(phi);
    ap_fixed<32, 7> sinP = lut_sin(phi);

    // ap_fixed<32, 7> cosT = hls::cosf(theta), sinT = hls::sinf(theta);
    // ap_fixed<32, 7> cosP = hls::cosf(phi), sinP = hls::sinf(phi);
    ap_fixed<32, 7> unit_n[3];
    unit_n[0] = sinT * cosP; 
    unit_n[1] = sinT * sinP;
    unit_n[2] = cosT;
    // printf("unit_n[0]=%s, unit_n[1]=%s, unit_n[2]=%s\n", unit_n[0].to_string(10).c_str(), unit_n[1].to_string(10).c_str(), unit_n[2].to_string(10).c_str());
    // get new normal vector
    ap_fixed<32, 15, AP_RND, AP_SAT> nv[3];
    nv[0] = nx + unit_n[0] * nr;
    nv[1] = ny + unit_n[1] * nr;
    nv[2] = nz + unit_n[2] * nr;
    ap_ufixed<31, 7> nv_norm_dec = nv[0] * nv[0] + nv[1] * nv[1] + nv[2] * nv[2];
    assert(nv_norm_dec != 0);
    ap_uint<31> nv_int;
    nv_int = nv_norm_dec.range(30, 0);
    nv_int = hls::sqrt(nv_int);
    ap_ufixed<31, 15> nv_fixed = nv_int;
    ap_fixed<32, 15, AP_RND, AP_SAT> nv_norm = nv_fixed >> 12;
    //ap_fixed<32, 15, AP_RND, AP_SAT> nv_norm = hls::sqrt(nv_norm_dec);
    // float_fixed nv_norm = hls::sqrt(nv[0] * nv[0] + nv[1] * nv[1] + nv[2] * nv[2]);
    nv_norm = (nv_norm != 0) ? nv_norm : FIXED_SMALL_VAL_16B;
    assert(nv_norm != 0);
    nv[0] = nv[0] / nv_norm;
    nv[1] = nv[1] / nv_norm;
    nv[2] = nv[2] / nv_norm;
    // create label based on new normal vector & random z
    nv[2] = (nv[2] != 0) ? nv[2] : FIXED_SMALL_VAL_16B;
    assert(nv[2] != 0);
    p_a_fixed = -nv[0] / nv[2];
    p_b_fixed = -nv[1] / nv[2];
    p_c_fixed = zs - p_a_fixed * p_x_abs - p_b_fixed * p_y_abs;
    p_a=(p_a_fixed*para).to_ap_int(); p_b=(p_b_fixed*para).to_ap_int(); p_c=(p_c_fixed*para).to_ap_int();
    // printf("p_a=%s, p_b=%s, p_c=%s\n", p_a.to_string(10).c_str(), p_b.to_string(10).c_str(), p_c.to_string(10).c_str());
    // inner_iter++;
    // p_a = p_a_fixed.to_float();
    // p_b = p_b_fixed.to_float();
    // p_c = p_c_fixed.to_float();
}

void layer0RandProposalOpt(ap_uint<32> q[4096], ap_uint<32>& i, ap_uint<32>& c, int outer_iter, int inner_iter, ap_uint<12> expn_region_x, ap_uint<12> expn_region_y, ap_uint<12> unit_region_x, ap_uint<12> unit_region_y, ap_uint<7> unit_region_w, ap_uint<7> unit_region_h, 
    ap_uint<12> p_x, ap_uint<12> p_y, ap_fixed<32, 15, AP_RND, AP_SAT> init_a, ap_fixed<32, 15, AP_RND, AP_SAT> init_b, ap_fixed<32, 15, AP_RND, AP_SAT> init_c, ap_fixed<32, 15, AP_RND, AP_SAT>& p_a, ap_fixed<32, 15, AP_RND, AP_SAT>& p_b, ap_fixed<32, 15, AP_RND, AP_SAT>& p_c) {
#pragma HLS inline off
    //
    // ap_fixed<32, 15, AP_RND, AP_SAT> p_a_fixed, p_b_fixed, p_c_fixed;
    // 
    // if (is_init == true) {
    //     p_a = init_a; p_b = init_b; p_c = init_c; 
    //     return;
    // }
    ap_ufixed<16, 1> half_fixed = 0.5;
    ap_ufixed<16, 1> one_fixed = 1.0;
    ap_fixed<32, 15, AP_RND, AP_SAT> disp_max = DISP_MAX;
    ap_fixed<32, 15, AP_RND, AP_SAT> disp_min = DISP_MIN;
    ap_uint<8> power;
    /*  */
    ap_uint<8> total_iter = outer_iter + inner_iter;
    // calculate initial disp
    ap_uint<12> p_x_abs = unit_region_x + p_x; // absolute coordinate
    ap_uint<12> p_y_abs = unit_region_y + p_y;
    ap_fixed<32, 15, AP_RND, AP_SAT> zs = init_a * p_x_abs + init_b * p_y_abs + init_c;
    // printf("zs=%s \n", zs.to_string(10).c_str());
    // get random z
    power = total_iter + 1;
    // float_fixed dz = (disp_max - disp_min) * hls::pow(half_fixed, power);
    ap_fixed<32, 15, AP_RND, AP_SAT> dz = (disp_max - disp_min) * (half_fixed >> power);
    // printf("dz=%s \n", dz.to_string(10).c_str());
    ap_fixed<32, 15, AP_RND, AP_SAT> zs_min = zs - dz;
    ap_fixed<32, 15, AP_RND, AP_SAT> zs_max = zs + dz;
    ap_fixed<32, 15, AP_RND, AP_SAT> minz = hls::max(disp_min, zs_min);
    ap_fixed<32, 15, AP_RND, AP_SAT> maxz = hls::min(disp_max, zs_max);
    zs.range() = randMwc(q, i, c)(31, 0);
    // printf("zs=%s \n", zs.to_string(10).c_str());
    // printf("iter=%s, Q=%s \n", iter.to_string(10).c_str(), Q[iter].to_string(10).c_str());
assert((maxz - minz) != 0);
    zs = zs - hls::floor(zs / (maxz - minz)) * (maxz - minz) + minz;
    // printf("out=%s, in=%s, min=%s, max=%s, zs=%s \n", outer_iter.to_string(10).c_str(), inner_iter.to_string(10).c_str(), minz.to_string(10).c_str(), maxz.to_string(10).c_str(), zs.to_string(10).c_str());
    // get initial normal vector
    power = total_iter;
    ap_fixed<32, 15, AP_RND, AP_SAT> nr = one_fixed * (half_fixed >> power);
    ap_ufixed<32, 7> nz_dec = one_fixed + init_a * init_a + init_b * init_b;
    // printf("nz_dec^2=%s \n", nz_dec.to_string(10).c_str());
    assert(nz_dec != 0);
    nz_dec = hls::sqrt(nz_dec);
    // printf("nz_dec=%s \n", nz_dec.to_string(10).c_str());
    assert(nz_dec != 0);
    ap_fixed<32, 15, AP_RND, AP_SAT> nz = one_fixed / nz_dec;
    ap_fixed<32, 15, AP_RND, AP_SAT> nx = -init_a * nz;
    ap_fixed<32, 15, AP_RND, AP_SAT> ny = -init_b * nz;
    // printf("nx=%s, ny=%s, nz=%s\n", nx.to_string(10).c_str(), ny.to_string(10).c_str(), nz.to_string(10).c_str());
    // get random unit vetor
    ap_ufixed<32, 7> theta;
    theta.range() = randMwc(q, i, c)(31, 0);
    theta = theta - hls::floor(theta / (HLS_PI)) * (HLS_PI);
    ap_ufixed<32, 7> phi;
    phi.range() = randMwc(q, i, c)(31, 0);
    phi = phi - hls::floor(phi / (2*HLS_PI)) * (2*HLS_PI);
    ap_fixed<32, 7> cosT = cosf(theta), sinT = sinf(theta);
    ap_fixed<32, 7> cosP = cosf(phi), sinP = sinf(phi);
    // ap_fixed<32, 7> cosT = hls::cosf(theta), sinT = hls::sinf(theta);
    // ap_fixed<32, 7> cosP = hls::cosf(phi), sinP = hls::sinf(phi);
    ap_fixed<32, 7> unit_n[3];
    unit_n[0] = sinT * cosP;
    unit_n[1] = sinT * sinP;
    unit_n[2] = cosT;
    // printf("unit_n[0]=%s, unit_n[1]=%s, unit_n[2]=%s\n", unit_n[0].to_string(10).c_str(), unit_n[1].to_string(10).c_str(), unit_n[2].to_string(10).c_str());
    // get new normal vector
    ap_fixed<32, 15, AP_RND, AP_SAT> nv[3];
    nv[0] = nx + unit_n[0] * nr;
    nv[1] = ny + unit_n[1] * nr;
    nv[2] = nz + unit_n[2] * nr;
    ap_ufixed<32, 7> nv_norm_dec = nv[0] * nv[0] + nv[1] * nv[1] + nv[2] * nv[2];
    assert(nv_norm_dec != 0);
    ap_fixed<32, 15, AP_RND, AP_SAT> nv_norm = hls::sqrt(nv_norm_dec);
    // float_fixed nv_norm = hls::sqrt(nv[0] * nv[0] + nv[1] * nv[1] + nv[2] * nv[2]);
    nv_norm = (nv_norm != 0) ? nv_norm : FIXED_SMALL_VAL_16B;
    assert(nv_norm != 0);
    nv[0] = nv[0] / nv_norm;
    nv[1] = nv[1] / nv_norm;
    nv[2] = nv[2] / nv_norm;
    // create label based on new normal vector & random z
    nv[2] = (nv[2] != 0) ? nv[2] : FIXED_SMALL_VAL_16B;
    assert(nv[2] != 0);
    p_a = -nv[0] / nv[2];
    p_b = -nv[1] / nv[2];
    p_c = zs - p_a * p_x_abs - p_b * p_y_abs;
    // printf("p_a=%s, p_b=%s, p_c=%s\n", p_a.to_string(10).c_str(), p_b.to_string(10).c_str(), p_c.to_string(10).c_str());
    // inner_iter++;
    // p_a = p_a_fixed.to_float();
    // p_b = p_b_fixed.to_float();
    // p_c = p_c_fixed.to_float();
}

void layer1RandProposal(ap_uint<32>* q, ap_uint<32>& i, ap_uint<32>& c, int outer_iter, int inner_iter, ap_uint<12> expn_region_x, ap_uint<12> expn_region_y, ap_uint<12> unit_region_x, ap_uint<12> unit_region_y, ap_uint<7> unit_region_w, ap_uint<7> unit_region_h, 
    ap_uint<12> p_x, ap_uint<12> p_y, label_range init_a, label_range init_b, label_range init_c, label_range& p_a, label_range& p_b, label_range& p_c) {
#pragma HLS inline off
    int para=pow(2,8);
    fixed_yy init_a_fixed; init_a_fixed.range()=init_a;
    fixed_yy init_b_fixed; init_b_fixed.range()=init_b;
    fixed_yy init_c_fixed; init_c_fixed.range()=init_c;
    fixed_yy p_a_fixed,p_b_fixed,p_c_fixed;
    //
    // ap_fixed<32, 15, AP_RND, AP_SAT> p_a_fixed, p_b_fixed, p_c_fixed;
    // 
    if (inner_iter == 0) {
        p_a = init_a; p_b = init_b; p_c = init_c; 
        return;
    }
    ap_ufixed<16, 1> half_fixed = 0.5;
    ap_ufixed<16, 1> one_fixed = 1.0;
    ap_fixed<32, 15, AP_RND, AP_SAT> disp_max = DISP_MAX;
    ap_fixed<32, 15, AP_RND, AP_SAT> disp_min = DISP_MIN;
    ap_uint<8> power;
    /*  */
    ap_uint<8> total_iter = outer_iter + inner_iter;
    // calculate initial disp
    ap_uint<12> p_x_abs = unit_region_x + p_x; // absolute coordinate
    ap_uint<12> p_y_abs = unit_region_y + p_y;
    ap_fixed<17, 9, AP_RND, AP_SAT> zs = init_a_fixed * p_x_abs + init_b_fixed * p_y_abs + init_c_fixed;
    // printf("zs=%s \n", zs.to_string(10).c_str());
    // get random z
    power = total_iter + 1;
    // float_fixed dz = (disp_max - disp_min) * hls::pow(half_fixed, power);
    ap_fixed<32, 15, AP_RND, AP_SAT> dz = (disp_max - disp_min) * (half_fixed >> power);
    // printf("dz=%s \n", dz.to_string(10).c_str());
    ap_fixed<32, 15, AP_RND, AP_SAT> zs_min = zs - dz;
    ap_fixed<32, 15, AP_RND, AP_SAT> zs_max = zs + dz;
    ap_fixed<32, 15, AP_RND, AP_SAT> minz = hls::max(disp_min, zs_min);
    ap_fixed<32, 15, AP_RND, AP_SAT> maxz = hls::min(disp_max, zs_max);
      zs.range() = randMwc(q, i, c)(31, 0);
    // printf("zs=%s \n", zs.to_string(10).c_str());
    // printf("iter=%s, Q=%s \n", iter.to_string(10).c_str(), Q[iter].to_string(10).c_str());
    if(maxz==minz)  maxz=maxz+FIXED_SMALL_VAL_16B;
assert((maxz - minz) != 0);
    zs = zs - hls::floor(zs / (maxz - minz)) * (maxz - minz) + minz;
    // printf("out=%s, in=%s, min=%s, max=%s, zs=%s \n", outer_iter.to_string(10).c_str(), inner_iter.to_string(10).c_str(), minz.to_string(10).c_str(), maxz.to_string(10).c_str(), zs.to_string(10).c_str());
    // get initial normal vector
    power = total_iter;
    ap_fixed<32, 15, AP_RND, AP_SAT> nr = one_fixed * (half_fixed >> power);
    ap_ufixed<32, 7> nz_dec = one_fixed + init_a_fixed * init_a_fixed + init_b_fixed * init_b_fixed;
    // printf("nz_dec^2=%s \n", nz_dec.to_string(10).c_str());
    assert(nz_dec != 0);
    nz_dec = hls::sqrt(nz_dec);
    // printf("nz_dec=%s \n", nz_dec.to_string(10).c_str());
    assert(nz_dec != 0);
    ap_fixed<32, 15, AP_RND, AP_SAT> nz = one_fixed / nz_dec;
    ap_fixed<32, 15, AP_RND, AP_SAT> nx = -init_a_fixed * nz;
    ap_fixed<32, 15, AP_RND, AP_SAT> ny = -init_b_fixed * nz;
    // printf("nx=%s, ny=%s, nz=%s\n", nx.to_string(10).c_str(), ny.to_string(10).c_str(), nz.to_string(10).c_str());
    // get random unit vetor
    ap_ufixed<32, 7> theta;
    theta.range() = randMwc(q, i, c)(31, 0);
    theta = theta - hls::floor(theta / (HLS_PI)) * (HLS_PI);
    ap_ufixed<32, 7> phi;
    phi.range() = randMwc(q, i, c)(31, 0);
    phi = phi - hls::floor(phi / (2*HLS_PI)) * (2*HLS_PI);
    ap_fixed<32, 7> cosT = cosf(theta), sinT = sinf(theta);
    ap_fixed<32, 7> cosP = cosf(phi), sinP = sinf(phi);
    // ap_fixed<32, 7> cosT = hls::cosf(theta), sinT = hls::sinf(theta);
    // ap_fixed<32, 7> cosP = hls::cosf(phi), sinP = hls::sinf(phi);
    ap_fixed<32, 7> unit_n[3];
    unit_n[0] = sinT * cosP;
    unit_n[1] = sinT * sinP;
    unit_n[2] = cosT;
    // printf("unit_n[0]=%s, unit_n[1]=%s, unit_n[2]=%s\n", unit_n[0].to_string(10).c_str(), unit_n[1].to_string(10).c_str(), unit_n[2].to_string(10).c_str());
    // get new normal vector
    ap_fixed<32, 15, AP_RND, AP_SAT> nv[3];
    nv[0] = nx + unit_n[0] * nr;
    nv[1] = ny + unit_n[1] * nr;
    nv[2] = nz + unit_n[2] * nr;
    ap_ufixed<32, 7> nv_norm_dec = nv[0] * nv[0] + nv[1] * nv[1] + nv[2] * nv[2];
    assert(nv_norm_dec != 0);
    ap_fixed<32, 15, AP_RND, AP_SAT> nv_norm = hls::sqrt(nv_norm_dec);
    // float_fixed nv_norm = hls::sqrt(nv[0] * nv[0] + nv[1] * nv[1] + nv[2] * nv[2]);
    nv_norm = (nv_norm != 0) ? nv_norm : FIXED_SMALL_VAL_16B;
    assert(nv_norm != 0);
    nv[0] = nv[0] / nv_norm;
    nv[1] = nv[1] / nv_norm;
    nv[2] = nv[2] / nv_norm;
    // create label based on new normal vector & random z
    nv[2] = (nv[2] != 0) ? nv[2] : FIXED_SMALL_VAL_16B;
    assert(nv[2] != 0);
    p_a_fixed = -nv[0] / nv[2];
    p_b_fixed = -nv[1] / nv[2];
    p_c_fixed = zs - p_a_fixed * p_x_abs - p_b_fixed * p_y_abs;
    p_a=(p_a_fixed*para).to_ap_int(); p_b=(p_b_fixed*para).to_ap_int(); p_c=(p_c_fixed*para).to_ap_int();
    // printf("p_a=%s, p_b=%s, p_c=%s\n", p_a.to_string(10).c_str(), p_b.to_string(10).c_str(), p_c.to_string(10).c_str());
    // inner_iter++;
    // p_a = p_a_fixed.to_float();
    // p_b = p_b_fixed.to_float();
    // p_c = p_c_fixed.to_float();
}

template<size_t CUR_SIZE>
void layerProposal(int ver_region_i, ap_uint<32>* q, ap_uint<32>& i, ap_uint<32>& c, 
    int outer_iter, int inner_iter, int expn_region_x, int expn_region_y, 
    int unit_region_x, int unit_region_y, int unit_region_w, int unit_region_h,
    float local_label[CUR_SIZE * 3][CUR_SIZE * 3][3], ap_fixed<32, 15, AP_RND, AP_SAT>& p_a, ap_fixed<32, 15, AP_RND, AP_SAT>& p_b, ap_fixed<32, 15, AP_RND, AP_SAT>& p_c) {
#pragma HLS inline off
    if (inner_iter == 0)
        layerExpnProposal<CUR_SIZE>(ver_region_i, q, i, c, expn_region_x, expn_region_y, unit_region_x, unit_region_y, unit_region_w, unit_region_h, 
            local_label, p_a, p_b, p_c);
    else
        layerRandProposal<CUR_SIZE>(ver_region_i, q, i, c, 0, inner_iter -1, expn_region_x, expn_region_y, unit_region_x, unit_region_y, unit_region_w, unit_region_h, 
            local_label, p_a, p_b, p_c);
}

void layer0ProposalRandLoad(ap_uint<32>* Q, ap_uint<32>& I, ap_uint<32>& C, int expn_region_x, int expn_region_y, int unit_region_x, int unit_region_y, int unit_region_w, int unit_region_h,
    float local_label[LAYER0_SIZE * 3][LAYER0_SIZE * 3][3], ap_uint<12>& p_x, ap_uint<12>& p_y, ap_fixed<32, 15, AP_RND, AP_SAT>& init_a, ap_fixed<32, 15, AP_RND, AP_SAT>& init_b, ap_fixed<32, 15, AP_RND, AP_SAT>& init_c) {
#pragma HLS inline off
    /* select a pixel randomly */
    p_x = randMwc(Q, I, C) % unit_region_w;
    p_y = randMwc(Q, I, C) % unit_region_h;
    /*  */
    int uram_x = p_x + unit_region_x - expn_region_x;
    int uram_y = p_y + unit_region_y - expn_region_y;
    /* get label of selected pixel */
    init_a = local_label[uram_y][uram_x][0];
    init_b = local_label[uram_y][uram_x][1];
    init_c = local_label[uram_y][uram_x][2];
}

void layer0ProposalRandLoad(
    ap_uint<12> expn_region_x, ap_uint<12> expn_region_y, ap_uint<12> unit_region_x, ap_uint<12> unit_region_y, ap_uint<7> unit_region_w, ap_uint<7> unit_region_h, 
    label_range local_label[LAYER0_SIZE * 3][LAYER0_SIZE * 3][3], ap_uint<4> p_x[LAYER0_PROP_NUM], ap_uint<4> p_y[LAYER0_PROP_NUM], 
    label_range init_a[LAYER0_PROP_NUM], label_range init_b[LAYER0_PROP_NUM], label_range init_c[LAYER0_PROP_NUM]) {
#pragma HLS inline off
    for (int i = 0; i < LAYER0_PROP_NUM; i++) {
#pragma HLS unroll
        /* select a pixel randomly */
//        p_x[i] = randMwc(Q[i], I[i], C[i]) % unit_region_w;
//        p_y[i] = randMwc(Q[i], I[i], C[i]) % unit_region_h;
        /* relative coordinate */
        ap_uint<7> uram_x = p_x[i] + unit_region_x - expn_region_x;
        ap_uint<7> uram_y = p_y[i] + unit_region_y - expn_region_y;
        /* get label of selected pixel */
        init_a[i] = local_label[uram_y][uram_x][0];
        init_b[i] = local_label[uram_y][uram_x][1];
        init_c[i] = local_label[uram_y][uram_x][2];
    }
}

void layer0ProposalRandLoad(ap_uint<32> Q[LAYER0_PROP_NUM][4096], ap_uint<32> I[LAYER0_PROP_NUM], ap_uint<32> C[LAYER0_PROP_NUM],
    ap_uint<12> expn_region_x, ap_uint<12> expn_region_y, ap_uint<12> unit_region_x, ap_uint<12> unit_region_y, ap_uint<7> unit_region_w, ap_uint<7> unit_region_h, 
    label_range local_label[LAYER0_SIZE * 3][LAYER0_SIZE * 3][3], ap_uint<12> p_x[LAYER0_PROP_NUM], ap_uint<12> p_y[LAYER0_PROP_NUM], 
    label_range init_a[LAYER0_PROP_NUM], label_range init_b[LAYER0_PROP_NUM], label_range init_c[LAYER0_PROP_NUM]) {
#pragma HLS inline off
    for (int i = 0; i < LAYER0_PROP_NUM; i++) {
#pragma HLS unroll
        /* select a pixel randomly */
        p_x[i] = randMwc(Q[i], I[i], C[i]) % unit_region_w;
        p_y[i] = randMwc(Q[i], I[i], C[i]) % unit_region_h;
        /* relative coordinate */
        ap_uint<7> uram_x = p_x[i] + unit_region_x - expn_region_x;
        ap_uint<7> uram_y = p_y[i] + unit_region_y - expn_region_y;
        /* get label of selected pixel */
        init_a[i] = local_label[uram_y][uram_x][0];
        init_b[i] = local_label[uram_y][uram_x][1];
        init_c[i] = local_label[uram_y][uram_x][2];
    }
}

void layer0ProposalRandLoadOpt(ap_uint<32> Q[LAYER0_PROP_NUM][4096], ap_uint<32> I[LAYER0_PROP_NUM], ap_uint<32> C[LAYER0_PROP_NUM],
    ap_uint<11> expn_region_x, ap_uint<11> expn_region_y, ap_uint<11> unit_region_x, ap_uint<11> unit_region_y, ap_uint<4> unit_region_w, ap_uint<4> unit_region_h, 
    ap_int<24> local_label[LAYER0_SIZE * 3][LAYER0_SIZE * 3][3], ap_uint<4> p_x[LAYER0_PROP_NUM], ap_uint<4> p_y[LAYER0_PROP_NUM], 
    ap_int<24> init_a[LAYER0_PROP_NUM], ap_int<24> init_b[LAYER0_PROP_NUM], ap_int<24> init_c[LAYER0_PROP_NUM]) {
#pragma HLS inline off
#pragma HLS array_partition variable=p_x type=complete dim=1
#pragma HLS array_partition variable=p_y type=complete dim=1
#pragma HLS array_partition variable=Q type=complete dim=1
#pragma HLS array_partition variable=I type=complete dim=1
#pragma HLS array_partition variable=C type=complete dim=1
#pragma HLS array_partition variable=local_label type=complete dim=1
#pragma HLS array_partition variable=local_label type=complete dim=3
#pragma HLS array_partition variable=init_a type=complete dim=1
#pragma HLS array_partition variable=init_b type=complete dim=1
#pragma HLS array_partition variable=init_c type=complete dim=1
    ap_uint<7> uram_x[LAYER0_PROP_NUM];
    ap_uint<7> uram_y[LAYER0_PROP_NUM];
#pragma HLS array_partition variable=uram_x type=complete dim=1
#pragma HLS array_partition variable=uram_y type=complete dim=1
    for (ap_uint<4> i = 0; i < LAYER0_PROP_NUM; i++) {
#pragma HLS unroll
        /* select a pixel randomly */
        p_x[i] = randMwc(Q[i], I[i], C[i]) % unit_region_w;
        p_y[i] = randMwc(Q[i], I[i], C[i]) % unit_region_h;
        /* relative coordinate */
        uram_x[i] = p_x[i] + unit_region_x - expn_region_x;
        uram_y[i] = p_y[i] + unit_region_y - expn_region_y;
    }
    for (ap_uint<4> i = 0; i < LAYER0_PROP_NUM; i++) {
#pragma HLS pipeline II=1    
        /* get label of selected pixel */
        init_a[i] = local_label[uram_y[i]][uram_x[i]][0];
        init_b[i] = local_label[uram_y[i]][uram_x[i]][1];
        init_c[i] = local_label[uram_y[i]][uram_x[i]][2];    
    }
}

void layer1ProposalRandLoad(ap_uint<32> Q[LAYER1_PROP_NUM][4096], ap_uint<32> I[LAYER1_PROP_NUM], ap_uint<32> C[LAYER1_PROP_NUM],
    ap_uint<12> expn_region_x, ap_uint<12> expn_region_y, ap_uint<12> unit_region_x, ap_uint<12> unit_region_y, ap_uint<7> unit_region_w, ap_uint<7> unit_region_h, 
    label_range local_label[LAYER1_SIZE * 3][LAYER1_SIZE * 3][3], ap_uint<12> p_x[LAYER1_PROP_NUM], ap_uint<12> p_y[LAYER1_PROP_NUM], 
    label_range init_a[LAYER1_PROP_NUM], label_range init_b[LAYER1_PROP_NUM], label_range init_c[LAYER1_PROP_NUM]) {
#pragma HLS inline off
    for (int i = 0; i < LAYER1_PROP_NUM; i++) {
#pragma HLS unroll
        /* select a pixel randomly */
        p_x[i] = randMwc(Q[i], I[i], C[i]) % unit_region_w;
        p_y[i] = randMwc(Q[i], I[i], C[i]) % unit_region_h;
        /* relative coordinate */
        ap_uint<7> uram_x = p_x[i] + unit_region_x - expn_region_x;
        ap_uint<7> uram_y = p_y[i] + unit_region_y - expn_region_y;
        /* get label of selected pixel */
        init_a[i] = local_label[uram_y][uram_x][0];
        init_b[i] = local_label[uram_y][uram_x][1];
        init_c[i] = local_label[uram_y][uram_x][2];
    }
}

void layer0ProposalParall(int ver_region_i, ap_uint<32> Q[LAYER0_PROP_NUM][4096], ap_uint<32> I[LAYER0_PROP_NUM], ap_uint<32> C[LAYER0_PROP_NUM],
    int inner_iter, ap_uint<12> expn_region_x, ap_uint<12> expn_region_y, ap_uint<12> unit_region_x, ap_uint<12> unit_region_y, ap_uint<7> unit_region_w, ap_uint<7> unit_region_h, 
    label_range local_label[LAYER0_SIZE * 3][LAYER0_SIZE * 3][3], label_range p_a[LAYER0_PROP_NUM], label_range p_b[LAYER0_PROP_NUM], label_range p_c[LAYER0_PROP_NUM]) {
#pragma HLS inline off
#if COST_LOAD_MODE == 1 && SCAN_DIRECTION == 1
#pragma HLS array_partition variable=local_label type=complete dim=2
#elif COST_LOAD_MODE == 1 && SCAN_DIRECTION == 0
#pragma HLS array_partition variable=local_label type=complete dim=1
#endif
#pragma HLS array_partition variable=local_label type=complete dim=3
#pragma HLS array_partition variable=p_a type=complete dim=1
#pragma HLS array_partition variable=p_b type=complete dim=1
#pragma HLS array_partition variable=p_c type=complete dim=1
#pragma HLS array_partition variable=Q type=complete dim=1
#pragma HLS array_partition variable=I type=complete dim=1
#pragma HLS array_partition variable=C type=complete dim=1
    label_range init_a[LAYER0_PROP_NUM], init_b[LAYER0_PROP_NUM], init_c[LAYER0_PROP_NUM];
#pragma HLS array_partition variable=init_a type=complete dim=1
#pragma HLS array_partition variable=init_b type=complete dim=1
#pragma HLS array_partition variable=init_c type=complete dim=1
    ap_uint<12> p_x[LAYER0_PROP_NUM], p_y[LAYER0_PROP_NUM];
#pragma HLS array_partition variable=p_x type=complete dim=1
#pragma HLS array_partition variable=p_y type=complete dim=1
    /*  */
    layer0ProposalRandLoad(Q, I, C, expn_region_x, expn_region_y, unit_region_x, unit_region_y, unit_region_w, unit_region_h, local_label,
        p_x, p_y, init_a, init_b, init_c);
    /*  */
    LOOP_PROP:
    for (int i = 0; i < LAYER0_PROP_NUM; i++) {
#pragma HLS unroll
        layer0RandProposal(Q[i], I[i], C[i], 0, inner_iter, expn_region_x, expn_region_y, unit_region_x, unit_region_y, unit_region_w, unit_region_h, 
            p_x[i], p_y[i], init_a[i], init_b[i], init_c[i], p_a[i], p_b[i], p_c[i]);
    }
}


void layer0ProposalParall(int iter, int ver_region_i, ap_uint<4> p_x[LAYER0_PROP_NUM], ap_uint<4> p_y[LAYER0_PROP_NUM], ap_uint<32> pp[LAYER0_PROP_NUM][3],
    int inner_iter, ap_uint<12> expn_region_x, ap_uint<12> expn_region_y, ap_uint<12> unit_region_x, ap_uint<12> unit_region_y, ap_uint<7> unit_region_w, ap_uint<7> unit_region_h, 
    label_range local_label[LAYER0_SIZE * 3][LAYER0_SIZE * 3][3], label_range p_a[LAYER0_PROP_NUM], label_range p_b[LAYER0_PROP_NUM], label_range p_c[LAYER0_PROP_NUM]) {
#pragma HLS inline off
#if COST_LOAD_MODE == 1 && SCAN_DIRECTION == 1
#pragma HLS array_partition variable=local_label type=complete dim=2
#elif COST_LOAD_MODE == 1 && SCAN_DIRECTION == 0
#pragma HLS array_partition variable=local_label type=complete dim=1
#endif
#pragma HLS array_partition variable=local_label type=complete dim=3
#pragma HLS array_partition variable=p_a type=complete dim=1
#pragma HLS array_partition variable=p_b type=complete dim=1
#pragma HLS array_partition variable=p_c type=complete dim=1
    label_range init_a[LAYER0_PROP_NUM], init_b[LAYER0_PROP_NUM], init_c[LAYER0_PROP_NUM];
#pragma HLS array_partition variable=init_a type=complete dim=1
#pragma HLS array_partition variable=init_b type=complete dim=1
#pragma HLS array_partition variable=init_c type=complete dim=1
    /*  */
    layer0ProposalRandLoad(expn_region_x, expn_region_y, unit_region_x, unit_region_y, unit_region_w, unit_region_h, local_label,
        p_x, p_y, init_a, init_b, init_c);
    
    /*  */
    LOOP_PROP:
    for (int i = 0; i < LAYER0_PROP_NUM; i++) {
#pragma HLS unroll
        layer0RandProposal(pp[i], 0, inner_iter, expn_region_x, expn_region_y, unit_region_x, unit_region_y, unit_region_w, unit_region_h, 
            p_x[i], p_y[i], init_a[i], init_b[i], init_c[i], p_a[i], p_b[i], p_c[i]);
    }
}

void layer1ProposalParall(int ver_region_i, ap_uint<32> Q[LAYER1_PROP_NUM][4096], ap_uint<32> I[LAYER1_PROP_NUM], ap_uint<32> C[LAYER1_PROP_NUM],  
    int inner_iter, ap_uint<12> expn_region_x, ap_uint<12> expn_region_y, ap_uint<12> unit_region_x, ap_uint<12> unit_region_y, ap_uint<7> unit_region_w, ap_uint<7> unit_region_h, 
    label_range local_label[LAYER1_SIZE * 3][LAYER1_SIZE * 3][3], label_range p_a[LAYER1_PROP_NUM], label_range p_b[LAYER1_PROP_NUM], label_range p_c[LAYER1_PROP_NUM]) {
#pragma HLS inline off
#if COST_LOAD_MODE == 1 && SCAN_DIRECTION == 1
#pragma HLS array_partition variable=local_label type=complete dim=2
#elif COST_LOAD_MODE == 1 && SCAN_DIRECTION == 0
#pragma HLS array_partition variable=local_label type=complete dim=1
#endif
#pragma HLS array_partition variable=local_label type=complete dim=3
#pragma HLS array_partition variable=p_a type=complete dim=1
#pragma HLS array_partition variable=p_b type=complete dim=1
#pragma HLS array_partition variable=p_c type=complete dim=1
#pragma HLS array_partition variable=Q type=complete dim=1
#pragma HLS array_partition variable=I type=complete dim=1
#pragma HLS array_partition variable=C type=complete dim=1
    label_range init_a[LAYER1_PROP_NUM], init_b[LAYER1_PROP_NUM], init_c[LAYER1_PROP_NUM];
#pragma HLS array_partition variable=init_a type=complete dim=1
#pragma HLS array_partition variable=init_b type=complete dim=1
#pragma HLS array_partition variable=init_c type=complete dim=1
    ap_uint<12> p_x[LAYER1_PROP_NUM], p_y[LAYER1_PROP_NUM];
#pragma HLS array_partition variable=p_x type=complete dim=1
#pragma HLS array_partition variable=p_y type=complete dim=1
    /*  */
    layer1ProposalRandLoad(Q, I, C, expn_region_x, expn_region_y, unit_region_x, unit_region_y, unit_region_w, unit_region_h, local_label,
        p_x, p_y, init_a, init_b, init_c);
    /*  */
    for (int i = 0; i < LAYER1_PROP_NUM; i++) {
#pragma HLS unroll
        layer1RandProposal(Q[i], I[i], C[i], 0, inner_iter, expn_region_x, expn_region_y, unit_region_x, unit_region_y, unit_region_w, unit_region_h, 
            p_x[i], p_y[i], init_a[i], init_b[i], init_c[i], p_a[i], p_b[i], p_c[i]);
    }
}

void calcSinCos(ap_ufixed<32, 7> theta, ap_fixed<32, 7>& cos_theta, ap_fixed<32, 7>& sin_theta, 
                ap_ufixed<32, 7> phi, ap_fixed<32, 7>& cos_phi, ap_fixed<32, 7>& sin_phi) {
#pragma HLS allocation function instances=cosf limit=2
#pragma HLS allocation function instances=sinf limit=2
    cos_theta = cosf(theta);
    sin_theta = sinf(theta);
    cos_phi = cosf(phi);
    sin_phi = sinf(phi);
}

ap_uint<32> randomNumGen(ap_uint<32>* q, ap_uint<32>& i, ap_uint<32>& c, ap_uint<12> dividend, ap_uint<12> bias) {
#pragma HLS inline off
    ap_uint<32> random_int;
    random_int = randMwc(q, i, c) % dividend;
    return random_int + bias;
}

ap_ufixed<32, 7> randomAngleGen(ap_uint<32>* q, ap_uint<32>& i, ap_uint<32>& c) {
#pragma HLS inline off
    ap_ufixed<32, 7> theta;
    theta.range() = randMwc(q, i, c)(31, 0);
    theta = theta - hls::floor(theta / (HLS_PI/3)) * (HLS_PI/3);
    return theta;
}

void randomPlaneInit(ap_uint<32>* q, ap_uint<32>& i, ap_uint<32>& c, ap_uint<32>* q1, ap_uint<32>& i1, ap_uint<32>& c1, 
    ap_uint<32>* q2, ap_uint<32>& i2, ap_uint<32>& c2, ap_uint<32>* q3, ap_uint<32>& i3, ap_uint<32>& c3, 
    ap_uint<32>* q4, ap_uint<32>& i4, ap_uint<32>& c4, ap_uint<32>* q5, ap_uint<32>& i5, ap_uint<32>& c5,
    int unit_region_x, int unit_region_y, int unit_region_w, int unit_region_h, float& p_a, float& p_b, float& p_c) {
    // 
    ap_uint<12> p_x, p_y;
    ap_uint<32> random_int;
    ap_uint<32> random_dec;
    ap_fixed<32, 15, AP_RND, AP_SAT> n[3], p_a_fixed, p_b_fixed, p_c_fixed;
    ap_ufixed<32, 7> zs;
    ap_ufixed<32, 7> theta;
    ap_ufixed<32, 7> phi;
    ap_fixed<32, 7> cosT, sinT, cosP, sinP;
    random_int = randMwc(q, i, c) % unit_region_w;
    p_x = random_int + unit_region_x;
    random_int = randMwc(q1, i1, c1) % unit_region_h;
    p_y = random_int + unit_region_y;
    // p_x = randomNumGen(q, i, c, unit_region_w, unit_region_x);
    // p_y = randomNumGen(q1, i1, c1, unit_region_h, unit_region_y);
    // zs  = randomNumGen(q2, i2, c2, DISP_MAX, 0);
    random_int = randMwc(q2, i2, c2) % DISP_MAX;
    random_dec = randMwc(q3, i3, c3);
    zs.range(31, 25) = random_int(6, 0);
    zs.range(24, 0) = random_dec(24, 0);
    // theta = randomAngleGen(q4, i4, c4);
    // phi   = randomAngleGen(q5, i5, c5);
    theta.range() = randMwc(q4, i4, c4)(31, 0);
    theta = theta - hls::floor(theta / HLS_PI_DIV3) * HLS_PI_DIV3;
    phi.range() = randMwc(q5, i5, c5)(31, 0);
    phi = phi - hls::floor(phi / HLS_PI_MUL2) * HLS_PI_MUL2;
    // calcSinCos(theta, cosT, sinT, phi, cosP, sinP);
    cosT = cosf(theta);
    sinT = sinf(theta);
    cosP = cosf(phi);
    sinP = sinf(phi);
    n[0] = sinT * cosP; 
    n[1] = sinT * sinP; 
    n[2] = (cosT != 0) ? cosT : FIXED_SMALL_VAL;
    p_a_fixed = -n[0] / n[2]; 
    p_b_fixed = -n[1] / n[2]; 
    p_c_fixed = zs - p_a_fixed * p_x - p_b_fixed * p_y;
    p_a = p_a_fixed.to_float(); 
    p_b = p_b_fixed.to_float(); 
    p_c = p_c_fixed.to_float();
    // printf("(%s, %s) zs=%s \n", p_x.to_string(10).c_str(), p_y.to_string(10).c_str(), zs.to_string(10).c_str());
    // printf("\npa=%s pb=%s pc=%s \n", p_a.to_string(10).c_str(), p_b.to_string(10).c_str(), p_c.to_string(10).c_str());
}

void initRandGen(ap_uint<32> x, ap_uint<32>* q, ap_uint<32>& i, ap_uint<32>& c)
{
    q[0] = x;
    q[1] = x + PHI_;
    q[2] = x + PHI_ + PHI_;
    for (int k = 3; k < 4096; k++)
        q[k] = q[k - 3] ^ q[k - 2] ^ PHI_ ^ k;
    for (int k = 0; k < 4096; k++)
        randMwc(q, i, c);
}

#if 0
template<size_t CUR_SIZE>
void costLabelUpdateBinary(int expn_region_x, int expn_region_y, int expn_region_w, int expn_region_h, float plane_a, float plane_b, float plane_c, 
                     float local_unary_cost[CUR_SIZE * 3][CUR_SIZE * 3][DISP_MAX], float local_binary_cost[CUR_SIZE * 3][CUR_SIZE * 3][8], 
                     float local_cost[CUR_SIZE * 3][CUR_SIZE * 3], float local_label[CUR_SIZE * 3][CUR_SIZE * 3][3]) {
    const int nb_dx_dy[8][2] = {{-1, 0}, {+1, 0}, {0, -1}, {0, 1}, {-1, -1}, {1, -1}, {-1, 1}, {1, 1}};
    for (int y = 0; y < CUR_SIZE * 3; y++) {
        for (int x = 0; x < CUR_SIZE * 3; x++) {
            if ((x >= expn_region_w) || (y >= expn_region_h))
                continue;
            // unary term
            float d = plane_a * (expn_region_x + x) + plane_b * (expn_region_y + y) + plane_c;
            float cost_unary_prop;
            if (d < DISP_MIN) cost_unary_prop = local_unary_cost[y][x][0];
            else if (d >= DISP_MAX - 1) cost_unary_prop = local_unary_cost[y][x][DISP_MAX - 1];
            else {
                int d_l = std::floor(d);
                int d_h = d_l + 1;
                float f_h = d - d_l;
                float f_l = 1.0 - f_h;
                cost_unary_prop = f_l * local_unary_cost[y][x][d_l] + f_h * local_unary_cost[y][x][d_h];
            }
            float cost_unary_orig = local_cost[y][x];
            float plane_a_orig = local_label[y][x][0]; float plane_b_orig = local_label[y][x][1]; float plane_c_orig = local_label[y][x][2]; 
            // binary term
            int valid_nb_num = 0;
            float cost_binary_prop = 0.0;
            float cost_binary_orig = 0.0;
            for(int nb = 0; nb < 8; nb++) {
                int nb_x = x + nb_dx_dy[nb][0]; 
                int nb_y = y + nb_dx_dy[nb][1];
                if (nb_x < 0 || nb_x >= expn_region_w || nb_y < 0 || nb_y >= expn_region_h)
                    continue;
                valid_nb_num++;
                float plane_a_nb = local_label[nb_y][nb_x][0]; float plane_b_nb = local_label[nb_y][nb_x][1]; float plane_c_nb = local_label[nb_y][nb_x][2];
                float dp_fp, dp_fq, dq_fq, dq_fp;
                dq_fq = plane_a_nb * (expn_region_x + nb_x) + plane_b_nb * (expn_region_y + nb_y) + plane_c_nb;
                dp_fq = plane_a_nb * (expn_region_x +  x  ) + plane_b_nb * (expn_region_y +  y  ) + plane_c_nb;
                // orig
                dp_fp = plane_a_orig * (expn_region_x +  x  ) + plane_b_orig * (expn_region_y +  y  ) + plane_c_orig;
                dq_fp = plane_a_orig * (expn_region_x + nb_x) + plane_b_orig * (expn_region_y + nb_y) + plane_c_orig;
                cost_binary_orig += local_binary_cost[y][x][nb] * std::fmin((std::abs(dp_fp - dp_fq) + std::abs(dq_fq - dq_fp)), float(1.0));
                // prop
                dp_fp = plane_a * (expn_region_x +  x  ) + plane_b * (expn_region_y +  y  ) + plane_c;
                dq_fp = plane_a * (expn_region_x + nb_x) + plane_b * (expn_region_y + nb_y) + plane_c;
                cost_binary_prop += local_binary_cost[y][x][nb] * std::fmin((std::abs(dp_fp - dp_fq) + std::abs(dq_fq - dq_fp)), float(1.0));
            }
            float cost_total_orig = cost_unary_orig + 1 * cost_binary_orig;
            float cost_total_prop = cost_unary_prop + 1 * cost_binary_prop;
            // cost comparision
            if (cost_total_prop < cost_total_orig) {
                local_cost[y][x] = cost_unary_prop;
                local_label[y][x][0] = plane_a; local_label[y][x][1] = plane_b; local_label[y][x][2] = plane_c;
            }
        }
    }
}
#endif

void genRegionInfo(int width, int height, int hor_region_i, int ver_region_i, int region_size, 
                   int& unit_region_x, int& unit_region_y, int& unit_region_w, int& unit_region_h, 
                   int& expn_region_x, int& expn_region_y, int& expn_region_w, int& expn_region_h) {
    unit_region_x = hor_region_i * region_size;
    unit_region_y = ver_region_i * region_size;
    unit_region_w = ((width - unit_region_x) > region_size) ? region_size : (width - unit_region_x);
    unit_region_h = ((height - unit_region_y) > region_size) ? region_size : (height - unit_region_y);
    expn_region_x = ((unit_region_x - region_size) < 0) ? unit_region_x : (unit_region_x - region_size);
    expn_region_y = ((unit_region_y - region_size) < 0) ? unit_region_y : (unit_region_y - region_size);
    expn_region_w = ((unit_region_x + unit_region_w + region_size) >= width) ? (width - expn_region_x) : (unit_region_x + unit_region_w + region_size - expn_region_x);
    expn_region_h = ((unit_region_y + unit_region_h + region_size) >= height) ? (height - expn_region_y) : (unit_region_y + unit_region_h + region_size - expn_region_y);
}
#endif

#if COST_CALC_MODE == 1
template <int layer_size>
void genCacheRegion(int iter, int parall_i, int hor_region_num, int hor_region_num_parall,
    int& buff0_region_x, int& buff0_region_y, int& buff0_region_w, int& buff0_region_h, 
    int& buff1_region_x, int& buff1_region_y, int& buff1_region_w, int& buff1_region_h) {
    int ver_region_i, hor_region_i; // absolute coordinates
    ver_region_i = iter / hor_region_num_parall;
    hor_region_i = iter % hor_region_num_parall + parall_i * hor_region_num_parall;
    if (hor_region_i >= hor_region_num) {
        buff0_region_x = buff0_region_y = buff0_region_w = buff0_region_h = 0;
        buff1_region_x = buff1_region_y = buff1_region_w = buff1_region_h = 0;
        return; 
    }
    // 
    int unit_region_x, unit_region_y, unit_region_w, unit_region_h;
    int expn_region_x, expn_region_y, expn_region_w, expn_region_h;
    genRegionInfo(WIDTH, HEIGHT, hor_region_i, ver_region_i, layer_size, unit_region_x, unit_region_y, unit_region_w, unit_region_h, 
        expn_region_x, expn_region_y, expn_region_w, expn_region_h);
    buff0_region_x = expn_region_x;
    buff0_region_y = expn_region_y; 
    buff0_region_w = expn_region_w; 
    buff0_region_h = ((expn_region_y + expn_region_h + layer_size) > (HEIGHT - 1)) ? (HEIGHT - expn_region_y) : (expn_region_h + layer_size);
    // printf("0(%d, %d, %d, %d) ", buff0_region_x, buff0_region_y, buff0_region_w, buff0_region_h);
    buff1_region_x = ((expn_region_x - DISP_MAX) >= 0) ? (expn_region_x - DISP_MAX) : 0;
    buff1_region_y = expn_region_y;
    buff1_region_w = ((expn_region_x - DISP_MAX) >= 0) ? (expn_region_w + DISP_MAX) : (expn_region_w + expn_region_x);
    buff1_region_h = ((expn_region_y + expn_region_h + layer_size) > (HEIGHT - 1)) ? (HEIGHT - expn_region_y) : (expn_region_h + layer_size); 
    // printf("1(%d, %d, %d, %d)    ", buff1_region_x, buff1_region_y, buff1_region_w, buff1_region_h);
}

template<int layer_size, int layer_parall_degree>
void genInitRegion(int iter, int parall_i, int hor_region_num, int ver_region_num, int hor_region_num_parall, int hor_region_num_parall_last,
    int& ver_region_i, int& hor_region_i, int& init_region_x, int& init_region_y, int& init_region_w, int& init_region_h) {
    // int ver_region_i, hor_region_i; // absolute coordinates
    int hor_region_relative_i; // relative coordinates
    ver_region_i = iter / hor_region_num_parall;
    hor_region_i = iter % hor_region_num_parall + parall_i * hor_region_num_parall;
    hor_region_relative_i = iter % hor_region_num_parall;
    if (hor_region_i >= hor_region_num) {
        init_region_x = init_region_y = init_region_w = init_region_h = 0;
        return;
    }
    if (parall_i < (layer_parall_degree - 1)) {
        if (hor_region_relative_i < (hor_region_num_parall - 2)) {
            ver_region_i++; hor_region_i++;
        }
        else {
            ver_region_i+=2; hor_region_i++;
        }
    }
    else {
        if (hor_region_relative_i < (hor_region_num_parall_last - 1)) {
            ver_region_i++; hor_region_i++;
        }
        else {
            ver_region_i+=2; hor_region_i = 0;
        }
    }
    // if (ver_region_i >= ver_region_num) 
    //     continue;
    if ((ver_region_i >= ver_region_num) || (hor_region_i >= hor_region_num)) {
        init_region_x = init_region_y = init_region_w = init_region_h = 0;
        return;
    }
    // 
    int unit_region_x, unit_region_y, unit_region_w, unit_region_h;
    int expn_region_x, expn_region_y, expn_region_w, expn_region_h;
    genRegionInfo(WIDTH, HEIGHT, hor_region_i, ver_region_i, layer_size, unit_region_x, unit_region_y, unit_region_w, unit_region_h, 
        expn_region_x, expn_region_y, expn_region_w, expn_region_h);
    init_region_x = unit_region_x;
    init_region_y = unit_region_y;
    init_region_w = unit_region_w;
    init_region_h = unit_region_h;
    // printf("(%d, %d, %d, %d)    ", unit_region_x, unit_region_y, unit_region_w, unit_region_h);
}

template<int layer_size, int layer_parall_degree>
void genTransRegion(int iter, int parall_i, int hor_region_num, int hor_region_num_parall, int hor_region_num_parall_last, 
    int& ver_region_i, int& hor_region_i, int& trans_region_x, int& trans_region_y, int& trans_region_w, int& trans_region_h) {
    int hor_region_relative_i; // relative coordinates
    ver_region_i = iter / hor_region_num_parall;
    hor_region_i = iter % hor_region_num_parall + parall_i * hor_region_num_parall;
    hor_region_relative_i = iter % hor_region_num_parall;
    // 
    if ((parall_i == (layer_parall_degree - 1)) && (hor_region_relative_i >= hor_region_num_parall_last)) {
        trans_region_x = trans_region_y = trans_region_w = trans_region_h = 0;
        return;
    }
    if (hor_region_relative_i > 0) {
        ver_region_i--; hor_region_i--;
    }
    else {
        ver_region_i-=2; hor_region_i = ((parall_i == (layer_parall_degree - 1)) ? (hor_region_num_parall_last - 1) : (hor_region_num_parall - 1)) + parall_i * hor_region_num_parall;
    }
    // assert((ver_region_i >= 0) && (ver_region_i < ver_region_num) && (hor_region_i >= 0) && (hor_region_i < hor_region_num));
    // 
    int unit_region_x, unit_region_y, unit_region_w, unit_region_h;
    int expn_region_x, expn_region_y, expn_region_w, expn_region_h;
    genRegionInfo(WIDTH, HEIGHT, hor_region_i, ver_region_i, layer_size, unit_region_x, unit_region_y, unit_region_w, unit_region_h, 
        expn_region_x, expn_region_y, expn_region_w, expn_region_h);
    trans_region_x = unit_region_x;
    trans_region_y = unit_region_y;
    trans_region_w = unit_region_w;
    trans_region_h = unit_region_h;
    // printf("(%d, %d, %d, %d)    ", unit_region_x, unit_region_y, unit_region_w, unit_region_h);
}

template<int layer_size>
void genCalcRegion(int iter, int parall_i, int hor_region_num, int hor_region_num_parall, int& ver_region_i, int& hor_region_i, 
    int& unit_region_x, int& unit_region_y, int& unit_region_w, int& unit_region_h, 
    int& expn_region_x, int& expn_region_y, int& expn_region_w, int& expn_region_h) {
    ver_region_i = iter / hor_region_num_parall;
    hor_region_i = iter % hor_region_num_parall + parall_i * hor_region_num_parall;
    if (hor_region_i >= hor_region_num) {
        unit_region_x = unit_region_y = unit_region_w = unit_region_h = 0;
        expn_region_x = expn_region_y = expn_region_w = expn_region_h = 0;
        return;
    }
    // printf("(%d, %d)-", ver_region_i, hor_region_i);
    // int unit_region_x, unit_region_y, unit_region_w, unit_region_h;
    // int expn_region_x, expn_region_y, expn_region_w, expn_region_h;
    genRegionInfo(WIDTH, HEIGHT, hor_region_i, ver_region_i, layer_size, unit_region_x, unit_region_y, unit_region_w, unit_region_h, 
        expn_region_x, expn_region_y, expn_region_w, expn_region_h);
    // printf("(%d, %d, %d, %d)    ", expn_region_x, expn_region_y, expn_region_w, expn_region_h);
}

template<size_t CUR_SIZE, int layer_size>
void costLabelUpdate(int parall_i, bool buff_id, int expn_region_x, int expn_region_y, int expn_region_w, int expn_region_h, int unit_region_x, int buff1_region_w, float plane_a, float plane_b, float plane_c, 
    uchar local_im0_gray[2][CUR_SIZE * 4][CUR_SIZE * 3], uchar local_im1_gray[2][CUR_SIZE * 4][CUR_SIZE * 3 + DISP_MAX],
    uint64_t local_im0_census[2][CUR_SIZE * 4][CUR_SIZE * 3], uint64_t local_im1_census[2][CUR_SIZE * 4][CUR_SIZE * 3 + DISP_MAX], 
    float local_cost[CUR_SIZE * 3][CUR_SIZE * 3], float local_label[CUR_SIZE * 3][CUR_SIZE * 3][3], float local_disp[CUR_SIZE * 3][CUR_SIZE * 3],
    bool is_binary_enable, bool is_debug = false) {
    /* neighbour offsets */
    const int nb_dx_dy[8][2] = {{-1, 0}, {+1, 0}, {0, -1}, {0, 1}, {-1, -1}, {1, -1}, {-1, 1}, {1, 1}};
    /* parallel in column */
    for (int x = 0; x < layer_size * 3; x++) {
        for (int y = 0; y < layer_size * 3; y++) {
            if ((x >= expn_region_w) || (y >= expn_region_h))
                continue;
            /* calculate disp */
            float d_prop = plane_a * (expn_region_x + x) + plane_b * (expn_region_y + y) + plane_c;
            float cost_unary_prop = 0.0;
            float cost_binary_prop = 0.0;
            float d_orig = local_disp[y][x];
            float cost_unary_orig = local_cost[y][x];
            float cost_binary_orig = 0.0;
            /* calculate unary cost */
            if (d_prop < DISP_MIN) cost_unary_prop = 1.0f;
            else if (d_prop >= DISP_MAX - 1) cost_unary_prop = 1.0f;
            else {
                /* calculate disparities for interpolation */
                int d_l = std::floor(d_prop);
                int d_h = d_l + 1;
                float f_h = d_prop - d_l;
                float f_l = 1.0 - f_h;
                /* get gray value & census transform for each pixel */
                int x_l = (buff1_region_w - expn_region_w + x - d_l) > 0 ? (buff1_region_w - expn_region_w + x - d_l) : 0;
                int x_h = (buff1_region_w - expn_region_w + x - d_h) > 0 ? (buff1_region_w - expn_region_w + x - d_h) : 0;
                uchar gray_l   = local_im0_gray[buff_id][y][x];
                uchar gray_r_l = local_im1_gray[buff_id][y][x_l];
                uchar gray_r_h = local_im1_gray[buff_id][y][x_h];
                uint64_t census_l   = local_im0_census[buff_id][y][x];
                uint64_t census_r_l = local_im1_census[buff_id][y][x_l];
                uint64_t census_r_h = local_im1_census[buff_id][y][x_h];
                cost_unary_prop = f_l * unaryCostCalc(gray_l, gray_r_l, census_l, census_r_l) + f_h * unaryCostCalc(gray_l, gray_r_h, census_l, census_r_h);
                // uchar glb_gray_l   = global_im0_gray[(expn_region_y + y) * WIDTH + expn_region_x + x];
                // uchar glb_gray_r_l = global_im1_gray[(expn_region_y + y) * WIDTH + ((expn_region_x + x - d_l) > 0 ? (expn_region_x + x - d_l) : 0)];
                // uchar glb_gray_r_h = global_im1_gray[(expn_region_y + y) * WIDTH + ((expn_region_x + x - d_h) > 0 ? (expn_region_x + x - d_h) : 0)];
                // uint64_t glb_census_l   = global_im0_census[(expn_region_y + y) * WIDTH + expn_region_x + x];
                // uint64_t glb_census_r_l = global_im1_census[(expn_region_y + y) * WIDTH + ((expn_region_x + x - d_l) > 0 ? (expn_region_x + x - d_l) : 0)];
                // uint64_t glb_census_r_h = global_im1_census[(expn_region_y + y) * WIDTH + ((expn_region_x + x - d_h) > 0 ? (expn_region_x + x - d_h) : 0)];
                // assert(gray_l == glb_gray_l);
                // cost_unary_prop = f_l * unaryCostCalc(glb_gray_l, glb_gray_r_l, glb_census_l, glb_census_r_l) + f_h * unaryCostCalc(glb_gray_l, glb_gray_r_h, glb_census_l, glb_census_r_h);
            }
            /* binary cost */
            if (is_binary_enable == true) {
                for(int nb = 0; nb < 8; nb++) {
                    int x_nb = x + nb_dx_dy[nb][0]; 
                    int y_nb = y + nb_dx_dy[nb][1];
                    if (x_nb < 0 || x_nb >= expn_region_w || y_nb < 0 || y_nb >= expn_region_h)
                        continue;
                    float d_nb = local_disp[y_nb][x_nb];
                    uchar p_gray = local_im0_gray[buff_id][y][x];
                    uchar p_gray_nb = local_im0_gray[buff_id][y_nb][x_nb];
                    float weight = (std::abs(p_gray_nb - p_gray) < 10) ? 0.3 : 0.05;
                    cost_binary_prop += weight * std::abs(d_prop - d_nb);
                    cost_binary_orig += weight * std::abs(d_orig - d_nb);
                }
            }
            /* update local buffer */
            if ((cost_unary_prop + cost_binary_prop) < (cost_unary_orig + cost_binary_orig)) {
                local_cost[y][x] = cost_unary_prop;
                local_label[y][x][0] = plane_a; local_label[y][x][1] = plane_b; local_label[y][x][2] = plane_c;
                local_disp[y][x] = d_prop;
            }
        }
    }
}

void layer0OutloopInit(int hor_region_num, int hor_region_num_parall,
    ap_uint<32>* q, ap_uint<32>& i, ap_uint<32>& c, ap_uint<32>* q1, ap_uint<32>& i1, ap_uint<32>& c1, 
    ap_uint<32>* q2, ap_uint<32>& i2, ap_uint<32>& c2, ap_uint<32>* q3, ap_uint<32>& i3, ap_uint<32>& c3, 
    ap_uint<32>* q4, ap_uint<32>& i4, ap_uint<32>& c4, ap_uint<32>* q5, ap_uint<32>& i5, ap_uint<32>& c5,
    uchar ddr_im0_gray[WIDTH * HEIGHT], uchar ddr_im1_gray[WIDTH * HEIGHT], uint64_t ddr_im0_census[WIDTH * HEIGHT], uint64_t ddr_im1_census[WIDTH * HEIGHT], 
    float uram_cost[LAYER0_SIZE * 4][WIDTH], float uram_label[LAYER0_SIZE * 4][WIDTH][3], float uram_disp[LAYER0_SIZE * 4][WIDTH]) {
    /* init 1 row & (2 * (PARALL_DEGREE - 1) + 1) block */
    static bool skip_flag;
    for (int iter = 0; iter < (hor_region_num * 2); iter++) {
        // printf("%d\n", iter);
        int ver_region_i = iter / hor_region_num;
        int hor_region_i = iter % hor_region_num;
        int unit_region_x, unit_region_y, unit_region_w, unit_region_h;
        int expn_region_x, expn_region_y, expn_region_w, expn_region_h;
        genRegionInfo(WIDTH, HEIGHT, hor_region_i, ver_region_i, LAYER0_SIZE, unit_region_x, unit_region_y, unit_region_w, unit_region_h, 
            expn_region_x, expn_region_y, expn_region_w, expn_region_h);
        // printf("init: (%d, %d, %d, %d)\n", unit_region_x, unit_region_y, unit_region_x + unit_region_w, unit_region_y + unit_region_h);
        /* generate plane label */
        float plane_a, plane_b, plane_c;
        randomPlaneInit(q, i, c, q1, i1, c1, q2, i2, c2, q3, i3, c3, q4, i4, c4, q5, i5, c5, 
            unit_region_x, unit_region_y, unit_region_w, unit_region_h, plane_a, plane_b, plane_c);
        /* compute cost of each pixel */
        for (int y = 0; y < LAYER0_SIZE; y++) {
            for (int x = 0; x < LAYER0_SIZE; x++) {
                if (((unit_region_x + x) >= WIDTH) || (unit_region_y + y) >= HEIGHT)
                    continue;
                float d = plane_a * (unit_region_x + x) + plane_b * (unit_region_y + y) + plane_c;
                float cost_tmp;
                /* ad-census cost */
                if (d < DISP_MIN) cost_tmp = unaryCostCalc((unit_region_x + x), (unit_region_y + y), 0, ddr_im0_gray, ddr_im1_gray, ddr_im0_census, ddr_im1_census);
                else if (d >= DISP_MAX - 1) cost_tmp = unaryCostCalc((unit_region_x + x), (unit_region_y + y), DISP_MAX - 1, ddr_im0_gray, ddr_im1_gray, ddr_im0_census, ddr_im1_census);
                else {
                    int d_l = std::floor(d);
                    int d_h = d_l + 1;
                    float f_h = d - d_l;
                    float f_l = 1.0 - f_h;
                    cost_tmp = f_l * unaryCostCalc((unit_region_x + x), (unit_region_y + y), d_l, ddr_im0_gray, ddr_im1_gray, ddr_im0_census, ddr_im1_census) + 
                               f_h * unaryCostCalc((unit_region_x + x), (unit_region_y + y), d_h, ddr_im0_gray, ddr_im1_gray, ddr_im0_census, ddr_im1_census);
                }
                /* set corresponding region */
                uram_cost[ver_region_i * LAYER0_SIZE + y][(unit_region_x + x)] = cost_tmp;
                uram_label[ver_region_i * LAYER0_SIZE + y][(unit_region_x + x)][0] = plane_a;
                uram_label[ver_region_i * LAYER0_SIZE + y][(unit_region_x + x)][1] = plane_b;
                uram_label[ver_region_i * LAYER0_SIZE + y][(unit_region_x + x)][2] = plane_c;
                uram_disp[ver_region_i * LAYER0_SIZE + y][(unit_region_x + x)] = d;
            }
        }
        /* ref to ppt */
        if(iter >= hor_region_num) {
            if (skip_flag == true)
                iter += (hor_region_num_parall - 2);
            skip_flag = !skip_flag;
        }
        else {
            skip_flag = true;
        }
    }
}

template<size_t CUR_PARALL_DEGREE, size_t CUR_SIZE, int layer_size>
void layerCache(int iter, int parall_i, int hor_region_num, int hor_region_num_parall, bool buf_id, 
    uchar ddr_im0_gray[WIDTH * HEIGHT], uchar ddr_im1_gray[WIDTH * HEIGHT], uint64_t ddr_im0_census[WIDTH * HEIGHT], uint64_t ddr_im1_census[WIDTH * HEIGHT],
    uchar local_im0_gray[2][CUR_SIZE * 4][CUR_SIZE * 3], uchar local_im1_gray[2][CUR_SIZE * 4][CUR_SIZE * 3 + DISP_MAX],
    uint64_t local_im0_census[2][CUR_SIZE * 4][CUR_SIZE * 3], uint64_t local_im1_census[2][CUR_SIZE * 4][CUR_SIZE * 3 + DISP_MAX],
    bool is_debug = false) {
    /*  */
    int buff0_region_x, buff0_region_y, buff0_region_w, buff0_region_h;
    int buff1_region_x, buff1_region_y, buff1_region_w, buff1_region_h;
    genCacheRegion<layer_size>(iter, parall_i, hor_region_num, hor_region_num_parall, buff0_region_x, buff0_region_y, buff0_region_w, buff0_region_h, 
        buff1_region_x, buff1_region_y, buff1_region_w, buff1_region_h);
    if (is_debug == true)
        printf("iter=%d, parall_i=%d): (%d, %d, %d, %d)\n", iter, parall_i, buff1_region_x, buff1_region_y, buff1_region_x + buff1_region_w, buff1_region_y + buff1_region_h);
    for (int y = 0; y < buff0_region_h; y++) {
        for (int x = 0; x < buff0_region_w; x++) {
            local_im0_gray[buf_id][y][x] = ddr_im0_gray[(buff0_region_y + y) * WIDTH + (buff0_region_x + x)];
            local_im0_census[buf_id][y][x] = ddr_im0_census[(buff0_region_y + y) * WIDTH + (buff0_region_x + x)];
        }
    }
    for (int y = 0; y < buff1_region_h; y++) {
        for (int x = 0; x < buff1_region_w; x++) {
            local_im1_gray[buf_id][y][x] = ddr_im1_gray[(buff1_region_y + y) * WIDTH + (buff1_region_x + x)];
            local_im1_census[buf_id][y][x] = ddr_im1_census[(buff1_region_y + y) * WIDTH + (buff1_region_x + x)];
        }
    }
}

void layer0InloopInit(int iter, int parall_i, int ver_region_num, int hor_region_num, int hor_region_num_parall, int hor_region_num_parall_last, bool buff_id, 
    ap_uint<32>* q, ap_uint<32>& i, ap_uint<32>& c, ap_uint<32>* q1, ap_uint<32>& i1, ap_uint<32>& c1, 
    ap_uint<32>* q2, ap_uint<32>& i2, ap_uint<32>& c2, ap_uint<32>* q3, ap_uint<32>& i3, ap_uint<32>& c3, 
    ap_uint<32>* q4, ap_uint<32>& i4, ap_uint<32>& c4, ap_uint<32>* q5, ap_uint<32>& i5, ap_uint<32>& c5, 
    uchar local_im0_gray[2][LAYER0_SIZE * 4][LAYER0_SIZE * 3], uchar local_im1_gray[2][LAYER0_SIZE * 4][LAYER0_SIZE * 3 + DISP_MAX],
    uint64_t local_im0_census[2][LAYER0_SIZE * 4][LAYER0_SIZE * 3], uint64_t local_im1_census[2][LAYER0_SIZE * 4][LAYER0_SIZE * 3 + DISP_MAX], 
    float block_cost[LAYER0_SIZE][LAYER0_SIZE], float block_label[LAYER0_SIZE][LAYER0_SIZE][3], 
    float block_disp[LAYER0_SIZE][LAYER0_SIZE], bool& block_vld, int block_info[4]) {
    /* generate init region info */
    int init_ver_region_i, init_hor_region_i; // absolute coordinates
    int init_region_x, init_region_y, init_region_w, init_region_h;
    genInitRegion<LAYER0_SIZE, LAYER0_PARALL_DEGREE>(iter, parall_i, hor_region_num, ver_region_num, hor_region_num_parall, hor_region_num_parall_last,
        init_ver_region_i, init_hor_region_i, init_region_x, init_region_y, init_region_w, init_region_h);
    // printf("(%d, %d, %d, %d)    ", init_region_x, init_region_y, init_region_x + init_region_w, init_region_y + init_region_h);
    /* get coordinates of up-right corner pixel in local_im1_* */
    int buff0_region_x, buff0_region_y, buff0_region_w, buff0_region_h;
    int buff1_region_x, buff1_region_y, buff1_region_w, buff1_region_h;
    genCacheRegion<LAYER0_SIZE>(iter, parall_i, hor_region_num, hor_region_num_parall, buff0_region_x, buff0_region_y, buff0_region_w, buff0_region_h, 
        buff1_region_x, buff1_region_y, buff1_region_w, buff1_region_h);
    /* get relative position of init region in local_im0_* */
    int calc_ver_region_i, calc_hor_region_i; // absolute coordinates
    int unit_region_x, unit_region_y, unit_region_w, unit_region_h;
    int calc_region_x, calc_region_y, calc_region_w, calc_region_h;
    genCalcRegion<LAYER0_SIZE>(iter, parall_i, hor_region_num, hor_region_num_parall, calc_ver_region_i, calc_hor_region_i, unit_region_x, unit_region_y, unit_region_w, unit_region_h,
        calc_region_x, calc_region_y, calc_region_w, calc_region_h);
    // printf("init: (%d, %d, %d, %d)\n", init_region_x, init_region_y, init_region_x + init_region_w, init_region_y + init_region_h);
    // printf("calc: (%d, %d, %d, %d)\n", calc_region_x, calc_region_y, calc_region_x + calc_region_w, calc_region_y + calc_region_h);
    // printf("buff: (%d, %d, %d, %d)\n", buff1_region_x, buff1_region_y, buff1_region_x + buff1_region_w, buff1_region_y + buff1_region_h);
    /* exit */
    if (init_region_w == 0) {
        block_vld = false; return;
    }
    else {
        block_vld = true;
        block_info[0] = init_region_x;
        block_info[1] = (init_ver_region_i % 4) * LAYER0_SIZE;
        block_info[2] = init_region_w;
        block_info[3] = init_region_h;
    }
    /* generate init plane label */
    float plane_a, plane_b, plane_c;
    randomPlaneInit(q, i, c, q1, i1, c1, q2, i2, c2, q3, i3, c3, q4, i4, c4, q5, i5, c5, 
        init_region_x, init_region_y, init_region_w, init_region_h, plane_a, plane_b, plane_c);
    /* calculate disp & cost for each pixel & update uram */
    for (int y = 0; y < LAYER0_SIZE; y++) {
        for (int x = 0; x < LAYER0_SIZE; x++) {
            if (((init_region_x + x) >= WIDTH) || (init_region_y + y) >= HEIGHT)
                continue;
            /* calculate disp */
            float d = plane_a * (init_region_x + x) + plane_b * (init_region_y + y) + plane_c;
            int x_r = ((x - d) < 0) ? 0 : (x - d);
            /* calculate cost */
            float cost_tmp;
            if (init_hor_region_i == 0) {
                cost_tmp = 1.0f;
            }
            else {
                // unaryCostCalc(uchar gray_l, uchar gray_r, uint64_t census_l, uint64_t census_r)
                if (d < DISP_MIN) cost_tmp = 1.0f;
                else if (d >= DISP_MAX - 1) cost_tmp = 1.0f;
                else {
                    /* calculate disparities for interpolation */
                    int d_l = std::floor(d);
                    int d_h = d_l + 1;
                    float f_h = d - d_l;
                    float f_l = 1.0 - f_h;
                    /* get gray value & census transform for each pixel */
                    uchar gray_l   = local_im0_gray[buff_id][init_region_y - calc_region_y + y][init_region_x - calc_region_x + x];
                    uchar gray_r_l = local_im1_gray[buff_id][init_region_y - calc_region_y + y][buff1_region_w - calc_region_w + init_region_x - calc_region_x + x - d_l];
                    uchar gray_r_h = local_im1_gray[buff_id][init_region_y - calc_region_y + y][buff1_region_w - calc_region_w + init_region_x - calc_region_x + x - d_h];
                    uint64_t census_l   = local_im0_census[buff_id][init_region_y - calc_region_y + y][init_region_x - calc_region_x + x];
                    uint64_t census_r_l = local_im1_census[buff_id][init_region_y - calc_region_y + y][buff1_region_w - calc_region_w + init_region_x - calc_region_x + x - d_l];
                    uint64_t census_r_h = local_im1_census[buff_id][init_region_y - calc_region_y + y][buff1_region_w - calc_region_w + init_region_x - calc_region_x + x - d_h];
                    cost_tmp = f_l * unaryCostCalc(gray_l, gray_r_l, census_l, census_r_l) + f_h * unaryCostCalc(gray_l, gray_r_h, census_l, census_r_h);
                }
            }
            /* update uram */
            block_cost[y][x] = cost_tmp;
            block_label[y][x][0] = plane_a;
            block_label[y][x][1] = plane_b;
            block_label[y][x][2] = plane_c;
            block_disp[y][x] = d;
        }
    }
}

void layer0InloopInitStore(float block_cost[LAYER0_PARALL_DEGREE][LAYER0_SIZE][LAYER0_SIZE], float block_label[LAYER0_PARALL_DEGREE][LAYER0_SIZE][LAYER0_SIZE][3], 
    float block_disp[LAYER0_PARALL_DEGREE][LAYER0_SIZE][LAYER0_SIZE], bool block_vld[LAYER0_PARALL_DEGREE], int block_info[LAYER0_PARALL_DEGREE][4], // x, y, w, h
    float uram_cost[LAYER0_SIZE * 4][WIDTH], float uram_label[LAYER0_SIZE * 4][WIDTH][3], float uram_disp[LAYER0_SIZE * 4][WIDTH]) {
    for (int parall_i = 0; parall_i < LAYER0_PARALL_DEGREE; parall_i++) {
        if (block_vld[parall_i] == false)
            continue;
        for (int y = 0; y < block_info[parall_i][3]; y++) {
            for (int x = 0; x < block_info[parall_i][2]; x++) {
                uram_cost[(block_info[parall_i][1] + y)][(block_info[parall_i][0] + x)] = block_cost[parall_i][y][x];
                uram_label[(block_info[parall_i][1] + y)][(block_info[parall_i][0] + x)][0] = block_label[parall_i][y][x][0];
                uram_label[(block_info[parall_i][1] + y)][(block_info[parall_i][0] + x)][1] = block_label[parall_i][y][x][1];
                uram_label[(block_info[parall_i][1] + y)][(block_info[parall_i][0] + x)][2] = block_label[parall_i][y][x][2];
                uram_disp[(block_info[parall_i][1] + y)][(block_info[parall_i][0] + x)] = block_disp[parall_i][y][x];
            }
        }
    }
}

void layer0CalcParallLoad(int iter, int parall_i, int ver_region_num, int hor_region_num, int hor_region_num_parall,
    float uram_cost[LAYER0_SIZE * 4][WIDTH], float uram_label[LAYER0_SIZE * 4][WIDTH][3], float uram_disp[LAYER0_SIZE * 4][WIDTH],
    float local_cost[LAYER0_SIZE * 3][LAYER0_SIZE * 3], float local_label[LAYER0_SIZE * 3][LAYER0_SIZE * 3][3], float local_disp[LAYER0_SIZE * 3][LAYER0_SIZE * 3]) {
    /* generate region info */
    int ver_region_i, hor_region_i; // absolute coordinates
    int unit_region_x, unit_region_y, unit_region_w, unit_region_h;
    int expn_region_x, expn_region_y, expn_region_w, expn_region_h;
    genCalcRegion<LAYER0_SIZE>(iter, parall_i, hor_region_num, hor_region_num_parall, ver_region_i, hor_region_i, unit_region_x, unit_region_y, unit_region_w, unit_region_h, 
        expn_region_x, expn_region_y, expn_region_w, expn_region_h);
    /* global ram -> local ram */
    int uram_y_addr_id = ver_region_i % 4;
    for (int x = 0; x < LAYER0_SIZE * 3; x++) {
        if ((expn_region_x + x) >= WIDTH)
            continue;
        if (ver_region_i == 0) {
            L0_GLOBAL_LOCAL_COST_CALC_MODE_ADDR1
        }
        else {
            if (uram_y_addr_id == 0) {
                L0_GLOBAL_LOCAL_COST_CALC_MODE_ADDR0
            }
            else if (uram_y_addr_id == 1) {
                L0_GLOBAL_LOCAL_COST_CALC_MODE_ADDR1
            }
            else if (uram_y_addr_id == 2) {
                L0_GLOBAL_LOCAL_COST_CALC_MODE_ADDR2
            }
            else if (uram_y_addr_id == 3) {
                L0_GLOBAL_LOCAL_COST_CALC_MODE_ADDR3
            }
        }
    }
}

void layer0CalcParallStore(int iter, int parall_i, int ver_region_num, int hor_region_num, int hor_region_num_parall,
    float local_cost[LAYER0_SIZE * 3][LAYER0_SIZE * 3], float local_label[LAYER0_SIZE * 3][LAYER0_SIZE * 3][3], float local_disp[LAYER0_SIZE * 3][LAYER0_SIZE * 3],
    float uram_cost[LAYER0_SIZE * 4][WIDTH], float uram_label[LAYER0_SIZE * 4][WIDTH][3], float uram_disp[LAYER0_SIZE * 4][WIDTH]) {
    /* generate region info */
    int ver_region_i, hor_region_i; // absolute coordinates
    int unit_region_x, unit_region_y, unit_region_w, unit_region_h;
    int expn_region_x, expn_region_y, expn_region_w, expn_region_h;
    genCalcRegion<LAYER0_SIZE>(iter, parall_i, hor_region_num, hor_region_num_parall, ver_region_i, hor_region_i, unit_region_x, unit_region_y, unit_region_w, unit_region_h, 
        expn_region_x, expn_region_y, expn_region_w, expn_region_h);
    /* local ram -> global uram */
    int uram_y_addr_id = ver_region_i % 4;
    for (int x = 0; x < LAYER0_SIZE * 3; x++) {
#pragma HLS pipeline II=1
        if ((expn_region_x + x) >= WIDTH)
            continue;
        if (ver_region_i == 0) {
            L0_LOCAL_GLOBAL_COST_CALC_MODE_ADDR1
        }
        else {
            if (uram_y_addr_id == 0) {
                L0_LOCAL_GLOBAL_COST_CALC_MODE_ADDR0
            }
            else if (uram_y_addr_id == 1) {
                L0_LOCAL_GLOBAL_COST_CALC_MODE_ADDR1
            }
            else if (uram_y_addr_id == 2) {
                L0_LOCAL_GLOBAL_COST_CALC_MODE_ADDR2
            }
            else if (uram_y_addr_id == 3) {
                L0_LOCAL_GLOBAL_COST_CALC_MODE_ADDR3
            }
        }
    }
}

template<size_t CUR_SIZE, int layer_size, bool is_binary_enable>
void layerCalc(int iter, int parall_i, int ver_region_num, int hor_region_num, int hor_region_num_parall, bool buff_id,
    ap_uint<32>* Q, ap_uint<32>& I, ap_uint<32>& C,
    uchar local_im0_gray[2][CUR_SIZE * 4][CUR_SIZE * 3], uchar local_im1_gray[2][CUR_SIZE * 4][CUR_SIZE * 3 + DISP_MAX],
    uint64_t local_im0_census[2][CUR_SIZE * 4][CUR_SIZE * 3], uint64_t local_im1_census[2][CUR_SIZE * 4][CUR_SIZE * 3 + DISP_MAX], 
    float local_cost[CUR_SIZE * 3][CUR_SIZE * 3], float local_label[CUR_SIZE * 3][CUR_SIZE * 3][3], float local_disp[CUR_SIZE * 3][CUR_SIZE * 3]) {
    /* plane label */
    float plane_a, plane_b, plane_c;
    /* generate region info */
    int ver_region_i, hor_region_i; // absolute coordinates
    int unit_region_x, unit_region_y, unit_region_w, unit_region_h;
    int expn_region_x, expn_region_y, expn_region_w, expn_region_h;
    genCalcRegion<layer_size>(iter, parall_i, hor_region_num, hor_region_num_parall, ver_region_i, hor_region_i, unit_region_x, unit_region_y, unit_region_w, unit_region_h, 
        expn_region_x, expn_region_y, expn_region_w, expn_region_h);
    int buff0_region_x, buff0_region_y, buff0_region_w, buff0_region_h;
    int buff1_region_x, buff1_region_y, buff1_region_w, buff1_region_h;
    genCacheRegion<layer_size>(iter, parall_i, hor_region_num, hor_region_num_parall, buff0_region_x, buff0_region_y, buff0_region_w, buff0_region_h, 
        buff1_region_x, buff1_region_y, buff1_region_w, buff1_region_h);
    if (unit_region_w == 0)
        return;
    // printf("(%d, %d, %d, %d)    ", calc_region_x, calc_region_y, calc_region_x + calc_region_w, calc_region_y + calc_region_h);
    /* process local ram */
    for (int inner_iter = 0; inner_iter < 8; inner_iter++) {
#pragma HLS pipeline off
        if (inner_iter == 0)
            layerExpnProposal<layer_size>(ver_region_i, Q, I, C, expn_region_x, expn_region_y, unit_region_x, unit_region_y, unit_region_w, unit_region_h, 
                local_label, plane_a, plane_b, plane_c);
        else
            layerRandProposal<layer_size>(ver_region_i, Q, I, C, 0, inner_iter -1, expn_region_x, expn_region_y, unit_region_x, unit_region_y, unit_region_w, unit_region_h, 
                local_label, plane_a, plane_b, plane_c);
        costLabelUpdate<CUR_SIZE, layer_size>(parall_i, buff_id, expn_region_x, expn_region_y, expn_region_w, expn_region_h, unit_region_x, buff1_region_w, plane_a, plane_b, plane_c, 
            local_im0_gray, local_im1_gray, local_im0_census, local_im1_census, local_cost, local_label, local_disp, is_binary_enable, (inner_iter == 0) && (iter == 287) && 0);
    }
}

void layer0Trans(int iter, int ver_region_num, int hor_region_num, int hor_region_num_parall, int hor_region_num_parall_last, 
    hls::stream_of_blocks<layer0_cost_blk> &cost_blk_out, hls::stream_of_blocks<layer0_label_blk> &label_blk_out, hls::stream_of_blocks<layer0_disp_blk> &disp_blk_out, 
    float uram_cost[LAYER0_SIZE * 4][WIDTH], float uram_label[LAYER0_SIZE * 4][WIDTH][3], float uram_disp[LAYER0_SIZE * 4][WIDTH]) {
    /*  */
    hls::write_lock<layer0_cost_blk> cost_bout(cost_blk_out);
    hls::write_lock<layer0_label_blk> label_bout(label_blk_out);
    hls::write_lock<layer0_disp_blk> disp_bout(disp_blk_out);
    /*  */
    for (int parall_i = 0; parall_i < LAYER0_PARALL_DEGREE; parall_i++) {
        /* generate region info */
        int ver_region_i, hor_region_i; // absolute coordinates
        int trans_region_x, trans_region_y, trans_region_w, trans_region_h;
        genTransRegion<LAYER0_SIZE, LAYER0_PARALL_DEGREE>(iter, parall_i, hor_region_num, hor_region_num_parall, hor_region_num_parall_last, ver_region_i, hor_region_i, trans_region_x, trans_region_y, trans_region_w, trans_region_h);
        // printf("(%d, %d, %d, %d)    ", trans_region_x, trans_region_y, trans_region_x + trans_region_w, trans_region_y + trans_region_h);
        for (int x = 0; x < LAYER0_SIZE; x++) {
            for (int y = 0; y < LAYER0_SIZE; y++) {
                if ((x >= trans_region_w) || (y >= trans_region_h))
                    continue;
                cost_bout[y * (LAYER0_PARALL_DEGREE * LAYER0_SIZE) + parall_i * LAYER0_SIZE + x] = uram_cost[(ver_region_i % 4) * LAYER0_SIZE + y][trans_region_x + x];
                label_bout[(y * (LAYER0_PARALL_DEGREE * LAYER0_SIZE) + parall_i * LAYER0_SIZE + x) * 3 + 0] = uram_label[(ver_region_i % 4) * LAYER0_SIZE + y][trans_region_x + x][0];
                label_bout[(y * (LAYER0_PARALL_DEGREE * LAYER0_SIZE) + parall_i * LAYER0_SIZE + x) * 3 + 1] = uram_label[(ver_region_i % 4) * LAYER0_SIZE + y][trans_region_x + x][1];
                label_bout[(y * (LAYER0_PARALL_DEGREE * LAYER0_SIZE) + parall_i * LAYER0_SIZE + x) * 3 + 2] = uram_label[(ver_region_i % 4) * LAYER0_SIZE + y][trans_region_x + x][2];
                disp_bout[y * (LAYER0_PARALL_DEGREE * LAYER0_SIZE) + parall_i * LAYER0_SIZE + x] = uram_disp[(ver_region_i % 4) * LAYER0_SIZE + y][trans_region_x + x];
            }
        }
    }
}

void layer1Init(int& l0_iter, int l1_iter, 
    int l0_hor_region_num, int l0_ver_region_num, int l0_hor_region_num_parall, int l0_hor_region_num_parall_last, int l1_hor_region_num, int l1_ver_region_num, int l1_hor_region_num_parall, 
    hls::stream_of_blocks<layer0_cost_blk> &cost_blk_in, hls::stream_of_blocks<layer0_label_blk> &label_blk_in, hls::stream_of_blocks<layer0_disp_blk> &disp_blk_in,
    float uram_cost[LAYER1_SIZE * 4][WIDTH], float uram_label[LAYER1_SIZE * 4][WIDTH][3], float uram_disp[LAYER1_SIZE * 4][WIDTH],
    bool is_debug = false) {
    /* get the height of initialization region */
    int l1_ver_region_i = (l1_iter + l1_hor_region_num + 1) / l1_hor_region_num;
    int l1_hor_region_i = (l1_iter + l1_hor_region_num + 1) % l1_hor_region_num;
    int unit_region_x, unit_region_y, unit_region_w, unit_region_h;
    int expn_region_x, expn_region_y, expn_region_w, expn_region_h;
    genRegionInfo(WIDTH, HEIGHT, l1_hor_region_i, l1_ver_region_i, LAYER1_SIZE, unit_region_x, unit_region_y, unit_region_w, unit_region_h, expn_region_x, expn_region_y, expn_region_w, expn_region_h);
    /* generate receiving times */
    int rcv_time;
    if (l1_iter == -1) {
        rcv_time = l0_hor_region_num_parall * (LAYER1_SIZE / LAYER0_SIZE) * 2;
    }
    else if (((l1_iter + 1) % l1_hor_region_num_parall) == 0){
        rcv_time = l0_hor_region_num_parall * (unit_region_h / LAYER0_SIZE) * 1;
    }
    else {
        rcv_time = 0;
    }
    /* receive blocks */
    for (int i = 0; i < rcv_time; i++) {
        /*  */
        hls::read_lock<layer0_cost_blk> cost_bin(cost_blk_in);
        hls::read_lock<layer0_label_blk> label_bin(label_blk_in);
        hls::read_lock<layer0_disp_blk> disp_bin(disp_blk_in);
        /* save buffer data separately */
        for (int parall_i = 0; parall_i < LAYER0_PARALL_DEGREE; parall_i++) {
            /* generate region info */
            int ver_region_i, hor_region_i; // absolute coordinates
            int trans_region_x, trans_region_y, trans_region_w, trans_region_h;
            genTransRegion<LAYER0_SIZE, LAYER0_PARALL_DEGREE>((l0_iter + (l0_hor_region_num_parall + 1) + i), parall_i, l0_hor_region_num, l0_hor_region_num_parall, l0_hor_region_num_parall_last, ver_region_i, hor_region_i, trans_region_x, trans_region_y, trans_region_w, trans_region_h);
            if (trans_region_w == 0)
                continue;
            if (is_debug == true)
                printf("(%d, %d): (%d, %d, %d, %d)\n", ver_region_i, hor_region_i, trans_region_x, trans_region_y, trans_region_x + trans_region_w, trans_region_y + trans_region_h);
            int uram_y = (ver_region_i % ((LAYER1_SIZE / LAYER0_SIZE) * 4)) * LAYER0_SIZE;
            for (int x = 0; x < LAYER0_SIZE; x++) {
                for (int y = 0; y < LAYER0_SIZE; y++) {
                    if ((x >= trans_region_w) || (y >= trans_region_h))
                        continue;
                    uram_cost[uram_y + y][trans_region_x + x]     = cost_bin[y * (LAYER0_PARALL_DEGREE * LAYER0_SIZE) + parall_i * LAYER0_SIZE + x];
                    uram_label[uram_y + y][trans_region_x + x][0] = label_bin[(y * (LAYER0_PARALL_DEGREE * LAYER0_SIZE) + parall_i * LAYER0_SIZE + x) * 3 + 0];
                    uram_label[uram_y + y][trans_region_x + x][1] = label_bin[(y * (LAYER0_PARALL_DEGREE * LAYER0_SIZE) + parall_i * LAYER0_SIZE + x) * 3 + 1];
                    uram_label[uram_y + y][trans_region_x + x][2] = label_bin[(y * (LAYER0_PARALL_DEGREE * LAYER0_SIZE) + parall_i * LAYER0_SIZE + x) * 3 + 2];
                    uram_disp[uram_y + y][trans_region_x + x]     = disp_bin[y * (LAYER0_PARALL_DEGREE * LAYER0_SIZE) + parall_i * LAYER0_SIZE + x];
                }
            }
        }
    }
    /* update number of received blocks of layer0_size */
    l0_iter += rcv_time;
}

void layer1CalcParallLoad(int iter, int parall_i, int ver_region_num, int hor_region_num, int hor_region_num_parall,
    float uram_cost[LAYER1_SIZE * 4][WIDTH], float uram_label[LAYER1_SIZE * 4][WIDTH][3], float uram_disp[LAYER1_SIZE * 4][WIDTH],
    float local_cost[LAYER1_SIZE * 3][LAYER1_SIZE * 3], float local_label[LAYER1_SIZE * 3][LAYER1_SIZE * 3][3], float local_disp[LAYER1_SIZE * 3][LAYER1_SIZE * 3]) {
    /* generate region info */
    int ver_region_i, hor_region_i; // absolute coordinates
    int unit_region_x, unit_region_y, unit_region_w, unit_region_h;
    int expn_region_x, expn_region_y, expn_region_w, expn_region_h;
    genCalcRegion<LAYER1_SIZE>(iter, parall_i, hor_region_num, hor_region_num_parall, ver_region_i, hor_region_i, unit_region_x, unit_region_y, unit_region_w, unit_region_h, 
        expn_region_x, expn_region_y, expn_region_w, expn_region_h);
    /* global ram -> local ram */
    int uram_y_addr_id = ver_region_i % 4;
    for (int x = 0; x < LAYER1_SIZE * 3; x++) {
        if ((expn_region_x + x) >= WIDTH)
            continue;
        if (ver_region_i == 0) {
            L1_GLOBAL_LOCAL_ADDR1
        }
        else {
            if (uram_y_addr_id == 0) {
                L1_GLOBAL_LOCAL_ADDR0
            }
            else if (uram_y_addr_id == 1) {
                L1_GLOBAL_LOCAL_ADDR1
            }
            else if (uram_y_addr_id == 2) {
                L1_GLOBAL_LOCAL_ADDR2
            }
            else if (uram_y_addr_id == 3) {
                L1_GLOBAL_LOCAL_ADDR3
            }
        }
    }
}

void layer1CalcParallStore(int iter, int parall_i, int ver_region_num, int hor_region_num, int hor_region_num_parall,
    float local_cost[LAYER1_SIZE * 3][LAYER1_SIZE * 3], float local_label[LAYER1_SIZE * 3][LAYER1_SIZE * 3][3], float local_disp[LAYER1_SIZE * 3][LAYER1_SIZE * 3],
    float uram_cost[LAYER1_SIZE * 4][WIDTH], float uram_label[LAYER1_SIZE * 4][WIDTH][3], float uram_disp[LAYER1_SIZE * 4][WIDTH]) {
    /* generate region info */
    int ver_region_i, hor_region_i; // absolute coordinates
    int unit_region_x, unit_region_y, unit_region_w, unit_region_h;
    int expn_region_x, expn_region_y, expn_region_w, expn_region_h;
    genCalcRegion<LAYER1_SIZE>(iter, parall_i, hor_region_num, hor_region_num_parall, ver_region_i, hor_region_i, unit_region_x, unit_region_y, unit_region_w, unit_region_h, 
        expn_region_x, expn_region_y, expn_region_w, expn_region_h);
    /* local ram -> global uram */
    int uram_y_addr_id = ver_region_i % 4;
    for (int x = 0; x < LAYER1_SIZE * 3; x++) {
#pragma HLS pipeline II=1
        if ((expn_region_x + x) >= WIDTH)
            continue;
        if (ver_region_i == 0) {
            L1_LOCAL_GLOBAL_ADDR1
        }
        else {
            if (uram_y_addr_id == 0) {
                L1_LOCAL_GLOBAL_ADDR0
            }
            else if (uram_y_addr_id == 1) {
                L1_LOCAL_GLOBAL_ADDR1
            }
            else if (uram_y_addr_id == 2) {
                L1_LOCAL_GLOBAL_ADDR2
            }
            else if (uram_y_addr_id == 3) {
                L1_LOCAL_GLOBAL_ADDR3
            }
        }
    }
}

void layer1Trans(int iter, int ver_region_num, int hor_region_num, int hor_region_num_parall, int hor_region_num_parall_last, 
    hls::stream_of_blocks<layer1_cost_blk> &cost_blk_out, hls::stream_of_blocks<layer1_label_blk> &label_blk_out, hls::stream_of_blocks<layer1_disp_blk> &disp_blk_out, 
    float uram_cost[LAYER1_SIZE * 4][WIDTH], float uram_label[LAYER1_SIZE * 4][WIDTH][3], float uram_disp[LAYER1_SIZE * 4][WIDTH]) {
    /*  */
    hls::write_lock<layer1_cost_blk> cost_bout(cost_blk_out);
    hls::write_lock<layer1_label_blk> label_bout(label_blk_out);
    hls::write_lock<layer1_disp_blk> disp_bout(disp_blk_out);
    /*  */
    for (int parall_i = 0; parall_i < LAYER1_PARALL_DEGREE; parall_i++) {
        /* generate region info */
        int ver_region_i, hor_region_i; // absolute coordinates
        int trans_region_x, trans_region_y, trans_region_w, trans_region_h;
        genTransRegion<LAYER1_SIZE, LAYER1_PARALL_DEGREE>(iter, parall_i, hor_region_num, hor_region_num_parall, hor_region_num_parall_last, ver_region_i, hor_region_i, trans_region_x, trans_region_y, trans_region_w, trans_region_h);
        // printf("(%d, %d, %d, %d)    ", trans_region_x, trans_region_y, trans_region_x + trans_region_w, trans_region_y + trans_region_h);
        for (int x = 0; x < LAYER1_SIZE; x++) {
            for (int y = 0; y < LAYER1_SIZE; y++) {
                if ((x >= trans_region_w) || (y >= trans_region_h))
                    continue;
                cost_bout[y * (LAYER1_PARALL_DEGREE * LAYER1_SIZE) + parall_i * LAYER1_SIZE + x] = uram_cost[(ver_region_i % 4) * LAYER1_SIZE + y][trans_region_x + x];
                label_bout[(y * (LAYER1_PARALL_DEGREE * LAYER1_SIZE) + parall_i * LAYER1_SIZE + x) * 3 + 0] = uram_label[(ver_region_i % 4) * LAYER1_SIZE + y][trans_region_x + x][0];
                label_bout[(y * (LAYER1_PARALL_DEGREE * LAYER1_SIZE) + parall_i * LAYER1_SIZE + x) * 3 + 1] = uram_label[(ver_region_i % 4) * LAYER1_SIZE + y][trans_region_x + x][1];
                label_bout[(y * (LAYER1_PARALL_DEGREE * LAYER1_SIZE) + parall_i * LAYER1_SIZE + x) * 3 + 2] = uram_label[(ver_region_i % 4) * LAYER1_SIZE + y][trans_region_x + x][2];
                disp_bout[y * (LAYER1_PARALL_DEGREE * LAYER1_SIZE) + parall_i * LAYER1_SIZE + x] = uram_disp[(ver_region_i % 4) * LAYER1_SIZE + y][trans_region_x + x];
            }
        }
    }
}

void layer2Init(int& l1_iter, int l2_iter, 
    int l1_hor_region_num, int l1_ver_region_num, int l1_hor_region_num_parall, int l1_hor_region_num_parall_last, int l2_hor_region_num, int l2_ver_region_num, int l2_hor_region_num_parall, 
    hls::stream_of_blocks<layer1_cost_blk> &cost_blk_in, hls::stream_of_blocks<layer1_label_blk> &label_blk_in, hls::stream_of_blocks<layer1_disp_blk> &disp_blk_in,
    float uram_cost[LAYER2_SIZE * 4][WIDTH], float uram_label[LAYER2_SIZE * 4][WIDTH][3], float uram_disp[LAYER2_SIZE * 4][WIDTH],
    bool is_debug = false) {
    /* get the height of initialization region */
    int l2_ver_region_i = (l2_iter + l2_hor_region_num + 1) / l2_hor_region_num;
    int l2_hor_region_i = (l2_iter + l2_hor_region_num + 1) % l2_hor_region_num;
    int unit_region_x, unit_region_y, unit_region_w, unit_region_h;
    int expn_region_x, expn_region_y, expn_region_w, expn_region_h;
    genRegionInfo(WIDTH, HEIGHT, l2_hor_region_i, l2_ver_region_i, LAYER2_SIZE, unit_region_x, unit_region_y, unit_region_w, unit_region_h, expn_region_x, expn_region_y, expn_region_w, expn_region_h);
    /* generate receiving times */
    int rcv_time;
    if (l2_iter == -1) {
        rcv_time = l1_hor_region_num_parall * (LAYER2_SIZE / LAYER1_SIZE) * 2;
    }
    else if ((((l2_iter + 1) % l2_hor_region_num_parall) == 0)){
        rcv_time = l1_hor_region_num_parall * (unit_region_h / LAYER1_SIZE) * 1;
    }
    else {
        rcv_time = 0;
    }
    /* receive blocks */
    for (int i = 0; i < rcv_time; i++) {
        /*  */
        hls::read_lock<layer1_cost_blk> cost_bin(cost_blk_in);
        hls::read_lock<layer1_label_blk> label_bin(label_blk_in);
        hls::read_lock<layer1_disp_blk> disp_bin(disp_blk_in);
        /* save buffer data separately */
        for (int parall_i = 0; parall_i < LAYER1_PARALL_DEGREE; parall_i++) {
            /* generate region info */
            int ver_region_i, hor_region_i; // absolute coordinates
            int trans_region_x, trans_region_y, trans_region_w, trans_region_h;
            genTransRegion<LAYER1_SIZE, LAYER1_PARALL_DEGREE>((l1_iter + (l1_hor_region_num_parall + 1) + i), parall_i, l1_hor_region_num, l1_hor_region_num_parall, l1_hor_region_num_parall_last, ver_region_i, hor_region_i, trans_region_x, trans_region_y, trans_region_w, trans_region_h);
            if ((trans_region_w == 0))
                continue;
            if (is_debug == true)
                printf("(%d, %d): (%d, %d, %d, %d)\n", ver_region_i, hor_region_i, trans_region_x, trans_region_y, trans_region_x + trans_region_w, trans_region_y + trans_region_h);
            int uram_y = (ver_region_i % ((LAYER2_SIZE / LAYER1_SIZE) * 4)) * LAYER1_SIZE;
            for (int x = 0; x < LAYER1_SIZE; x++) {
                for (int y = 0; y < LAYER1_SIZE; y++) {
                    if ((x >= trans_region_w) || (y >= trans_region_h))
                        continue;
                    uram_cost[uram_y + y][trans_region_x + x]     = cost_bin[y * (LAYER1_PARALL_DEGREE * LAYER1_SIZE) + parall_i * LAYER1_SIZE + x];
                    uram_label[uram_y + y][trans_region_x + x][0] = label_bin[(y * (LAYER1_PARALL_DEGREE * LAYER1_SIZE) + parall_i * LAYER1_SIZE) * 3 + 0];
                    uram_label[uram_y + y][trans_region_x + x][1] = label_bin[(y * (LAYER1_PARALL_DEGREE * LAYER1_SIZE) + parall_i * LAYER1_SIZE) * 3 + 1];
                    uram_label[uram_y + y][trans_region_x + x][2] = label_bin[(y * (LAYER1_PARALL_DEGREE * LAYER1_SIZE) + parall_i * LAYER1_SIZE) * 3 + 2];
                    uram_disp[uram_y + y][trans_region_x + x]     = disp_bin[y * (LAYER1_PARALL_DEGREE * LAYER1_SIZE) + parall_i * LAYER1_SIZE + x];
                }
            }
        }
    }
    /* update number of received blocks of layer0_size */
    l1_iter += rcv_time;
}

void layer2CalcParallLoad(int iter, int parall_i, int ver_region_num, int hor_region_num, int hor_region_num_parall,
    float uram_cost[LAYER2_SIZE * 4][WIDTH], float uram_label[LAYER2_SIZE * 4][WIDTH][3], float uram_disp[LAYER2_SIZE * 4][WIDTH],
    float local_cost[LAYER2_SIZE * 3][LAYER2_SIZE * 3], float local_label[LAYER2_SIZE * 3][LAYER2_SIZE * 3][3], float local_disp[LAYER2_SIZE * 3][LAYER2_SIZE * 3]) {
    /* generate region info */
    int ver_region_i, hor_region_i; // absolute coordinates
    int unit_region_x, unit_region_y, unit_region_w, unit_region_h;
    int expn_region_x, expn_region_y, expn_region_w, expn_region_h;
    genCalcRegion<LAYER2_SIZE>(iter, parall_i, hor_region_num, hor_region_num_parall, ver_region_i, hor_region_i, unit_region_x, unit_region_y, unit_region_w, unit_region_h, 
        expn_region_x, expn_region_y, expn_region_w, expn_region_h);
    /* global ram -> local ram */
    int uram_y_addr_id = ver_region_i % 4;
    for (int x = 0; x < LAYER2_SIZE * 3; x++) {
        if ((expn_region_x + x) >= WIDTH)
            continue;
        if (ver_region_i == 0) {
            L2_GLOBAL_LOCAL_ADDR1
        }
        else {
            if (uram_y_addr_id == 0) {
                L2_GLOBAL_LOCAL_ADDR0
            }
            else if (uram_y_addr_id == 1) {
                L2_GLOBAL_LOCAL_ADDR1
            }
            else if (uram_y_addr_id == 2) {
                L2_GLOBAL_LOCAL_ADDR2
            }
            else if (uram_y_addr_id == 3) {
                L2_GLOBAL_LOCAL_ADDR3
            }
        }
    }
}

void layer2CalcParallStore(int iter, int parall_i, int ver_region_num, int hor_region_num, int hor_region_num_parall,
    float local_cost[LAYER2_SIZE * 3][LAYER2_SIZE * 3], float local_label[LAYER2_SIZE * 3][LAYER2_SIZE * 3][3], float local_disp[LAYER2_SIZE * 3][LAYER2_SIZE * 3],
    float uram_cost[LAYER2_SIZE * 4][WIDTH], float uram_label[LAYER2_SIZE * 4][WIDTH][3], float uram_disp[LAYER2_SIZE * 4][WIDTH]) {
    /* generate region info */
    int ver_region_i, hor_region_i; // absolute coordinates
    int unit_region_x, unit_region_y, unit_region_w, unit_region_h;
    int expn_region_x, expn_region_y, expn_region_w, expn_region_h;
    genCalcRegion<LAYER2_SIZE>(iter, parall_i, hor_region_num, hor_region_num_parall, ver_region_i, hor_region_i, unit_region_x, unit_region_y, unit_region_w, unit_region_h, 
        expn_region_x, expn_region_y, expn_region_w, expn_region_h);
    /* local ram -> global uram */
    int uram_y_addr_id = ver_region_i % 4;
    for (int x = 0; x < LAYER2_SIZE * 3; x++) {
#pragma HLS pipeline II=1
        if ((expn_region_x + x) >= WIDTH)
            continue;
        if (ver_region_i == 0) {
            L2_LOCAL_GLOBAL_ADDR1
        }
        else {
            if (uram_y_addr_id == 0) {
                L2_LOCAL_GLOBAL_ADDR0
            }
            else if (uram_y_addr_id == 1) {
                L2_LOCAL_GLOBAL_ADDR1
            }
            else if (uram_y_addr_id == 2) {
                L2_LOCAL_GLOBAL_ADDR2
            }
            else if (uram_y_addr_id == 3) {
                L2_LOCAL_GLOBAL_ADDR3
            }
        }
    }
}

template<size_t CUR_SIZE, int layer_size, int layer_parall_degree>
void layerTransDDR(int iter, int ver_region_num, int hor_region_num, int hor_region_num_parall, int hor_region_num_parall_last, 
    float ddr_cost[WIDTH * HEIGHT], float ddr_label[WIDTH * HEIGHT * 3], float ddr_disp[WIDTH * HEIGHT], 
    float uram_cost[CUR_SIZE * 4][WIDTH], float uram_label[CUR_SIZE * 4][WIDTH][3], float uram_disp[CUR_SIZE * 4][WIDTH], 
    bool is_debug = false) {
    for (int parall_i = 0; parall_i < layer_parall_degree; parall_i++) {
        /* generate region info */
        int ver_region_i, hor_region_i; // absolute coordinates
        int trans_region_x, trans_region_y, trans_region_w, trans_region_h;
        genTransRegion<layer_size, layer_parall_degree>(iter, parall_i, hor_region_num, hor_region_num_parall, hor_region_num_parall_last, ver_region_i, hor_region_i, trans_region_x, trans_region_y, trans_region_w, trans_region_h);
        if (trans_region_w == 0)
            continue;
        if (is_debug ==true)
            printf("(%d, %d): (%d, %d, %d, %d)\n", ver_region_i, hor_region_i, trans_region_x, trans_region_y, trans_region_x + trans_region_w, trans_region_y + trans_region_h);
        for (int x = 0; x < layer_size; x++) {
            for (int y = 0; y < layer_size; y++) {
                if ((x >= trans_region_w) || (y >= trans_region_h))
                    continue;
                ddr_cost[(trans_region_y + y) * WIDTH + trans_region_x + x] = uram_cost[(ver_region_i % 4) * layer_size + y][trans_region_x + x];
                ddr_label[((trans_region_y + y) * WIDTH + trans_region_x + x) * 3 + 0] = uram_label[(ver_region_i % 4) * layer_size + y][trans_region_x + x][0];
                ddr_label[((trans_region_y + y) * WIDTH + trans_region_x + x) * 3 + 1] = uram_label[(ver_region_i % 4) * layer_size + y][trans_region_x + x][1];
                ddr_label[((trans_region_y + y) * WIDTH + trans_region_x + x) * 3 + 2] = uram_label[(ver_region_i % 4) * layer_size + y][trans_region_x + x][2];
                ddr_disp[(trans_region_y + y) * WIDTH + trans_region_x + x] = uram_disp[(ver_region_i % 4) * layer_size + y][trans_region_x + x];
            }
        }
    }
}

void localExpLayer0(uchar ddr_im0_gray[WIDTH * HEIGHT], uchar ddr_im1_gray[WIDTH * HEIGHT], uint64_t ddr_im0_census[WIDTH * HEIGHT], uint64_t ddr_im1_census[WIDTH * HEIGHT],
    hls::stream_of_blocks<layer0_cost_blk> &cost_blk_out, hls::stream_of_blocks<layer0_label_blk> &label_blk_out, hls::stream_of_blocks<layer0_disp_blk> &disp_blk_out) {
    // float ddr_cost[WIDTH * HEIGHT], float ddr_label[WIDTH * HEIGHT * 3], float ddr_disp[WIDTH * HEIGHT]) {
    // 
    static float uram_cost[LAYER0_SIZE * 4][WIDTH]; // row partition
#pragma HLS array_partition variable=uram_cost type=complete dim=1
    static float uram_label[LAYER0_SIZE * 4][WIDTH][3]; 
#pragma HLS array_partition variable=uram_label type=complete dim=1
#pragma HLS array_partition variable=uram_label type=complete dim=3
    static float uram_disp[LAYER0_SIZE * 4][WIDTH]; 
#pragma HLS array_partition variable=uram_disp type=complete dim=1
    // 
    static uchar local_im0_gray[LAYER0_PARALL_DEGREE][2][LAYER0_SIZE * 4][LAYER0_SIZE * 3];
    static uchar local_im1_gray[LAYER0_PARALL_DEGREE][2][LAYER0_SIZE * 4][LAYER0_SIZE * 3 + DISP_MAX];
    static uint64_t local_im0_census[LAYER0_PARALL_DEGREE][2][LAYER0_SIZE * 4][LAYER0_SIZE * 3]; // TODO: change dim1 & dim2 ?
    static uint64_t local_im1_census[LAYER0_PARALL_DEGREE][2][LAYER0_SIZE * 4][LAYER0_SIZE * 3 + DISP_MAX];
    float block_cost[LAYER0_PARALL_DEGREE][LAYER0_SIZE][LAYER0_SIZE];
    float block_label[LAYER0_PARALL_DEGREE][LAYER0_SIZE][LAYER0_SIZE][3];
    float block_disp[LAYER0_PARALL_DEGREE][LAYER0_SIZE][LAYER0_SIZE]; 
    bool block_vld[LAYER0_PARALL_DEGREE];
    int block_info[LAYER0_PARALL_DEGREE][4];
    float local_cost[LAYER0_PARALL_DEGREE][LAYER0_SIZE * 3][LAYER0_SIZE * 3];
    float local_label[LAYER0_PARALL_DEGREE][LAYER0_SIZE * 3][LAYER0_SIZE * 3][3];
    float local_disp[LAYER0_PARALL_DEGREE][LAYER0_SIZE * 3][LAYER0_SIZE * 3];
    // 
    static bool random_generator_init = false;
    static ap_uint<32> Q[4096];  static ap_uint<32> I  = 4095;   static ap_uint<32> C  = 362436;
    static ap_uint<32> Q1[4096]; static ap_uint<32> I1 = 963512; static ap_uint<32> C1 = 781265;
    static ap_uint<32> Q2[4096]; static ap_uint<32> I2 = 124125; static ap_uint<32> C2 = 84812;
    static ap_uint<32> Q3[4096]; static ap_uint<32> I3 = 6321;   static ap_uint<32> C3 = 98411;
    static ap_uint<32> Q4[4096]; static ap_uint<32> I4 = 320541; static ap_uint<32> C4 = 3334;
    static ap_uint<32> Q5[4096]; static ap_uint<32> I5 = 94;     static ap_uint<32> C5 = 515954;
    static ap_uint<32> Q6[4096]; static ap_uint<32> I6 = 756411; static ap_uint<32> C6 = 913354;
    if (random_generator_init == false) {
        initRandGen(0, Q, I, C); initRandGen(6413, Q1, I1, C1); initRandGen(67, Q2, I2, C2);
        initRandGen(13359, Q3, I3, C3); initRandGen(7541, Q4, I4, C4); initRandGen(986548, Q5, I5, C5);
        initRandGen(156411, Q6, I6, C6);
        random_generator_init = true;
    }
    /*  */
    const int hor_region_num = LAYER0_HOR_REGION_NUM;
    const int ver_region_num = LAYER0_VER_REGION_NUM;
    const int hor_region_num_parall = LAYER0_HOR_REGION_NUM_PARALL;
    const int hor_region_num_parall_last = LAYER0_HOR_REGION_NUM_PARALL_LAST;
    /* index of ping-pong buffer */
    static bool buff_id = false;
    /* pre initialization */
    layer0OutloopInit(hor_region_num, hor_region_num_parall, Q, I, C, Q1, I1, C1, Q2, I2, C2, Q3, I3, C3, Q4, I4, C4, Q5, I5, C5, 
        ddr_im0_gray, ddr_im1_gray, ddr_im0_census, ddr_im1_census, uram_cost, uram_label, uram_disp);
    /* pre cache LAYER0_PARALL_DEGREE im0 & im1 buffer1 */
    for (int parall_i = 0; parall_i < LAYER0_PARALL_DEGREE; parall_i++) {
        layerCache<LAYER0_PARALL_DEGREE, LAYER0_SIZE, LAYER0_SIZE>(0, parall_i, hor_region_num, hor_region_num_parall, buff_id, ddr_im0_gray, ddr_im1_gray, ddr_im0_census, ddr_im1_census, local_im0_gray[parall_i], local_im1_gray[parall_i], local_im0_census[parall_i], local_im1_census[parall_i]);
    }
    /* iteration for (ver_region_num * hor_region_num_parall + (hor_region_num_parall + 1)) times */
    for (int iter = 0; iter < (ver_region_num * hor_region_num_parall + (hor_region_num_parall + 1)); iter++) {    
        /* cache */
        for (int parall_i = 0; parall_i < LAYER0_PARALL_DEGREE; parall_i++) {
            if (iter < ver_region_num * hor_region_num_parall - 1) {
                layerCache<LAYER0_PARALL_DEGREE, LAYER0_SIZE, LAYER0_SIZE>(iter + 1, parall_i, hor_region_num, hor_region_num_parall, !buff_id, ddr_im0_gray, ddr_im1_gray, ddr_im0_census, ddr_im1_census, local_im0_gray[parall_i], local_im1_gray[parall_i], local_im0_census[parall_i], local_im1_census[parall_i]);
            }
        }
        /* initialization */
        /* --calculate four blocks in parallel */
        for (int parall_i = 0; parall_i < LAYER0_PARALL_DEGREE; parall_i++) {
            if (iter < (ver_region_num * hor_region_num_parall - hor_region_num_parall - 2)) {
                layer0InloopInit(iter, parall_i, ver_region_num, hor_region_num, hor_region_num_parall, hor_region_num_parall_last, buff_id, Q, I, C, Q1, I1, C1, Q2, I2, C2, Q3, I3, C3, Q4, I4, C4, Q5, I5, C5, local_im0_gray[parall_i], local_im1_gray[parall_i], local_im0_census[parall_i], local_im1_census[parall_i], block_cost[parall_i], block_label[parall_i], block_disp[parall_i], block_vld[parall_i], block_info[parall_i]);
            }
        }
        /* --update uram in serial */
        layer0InloopInitStore(block_cost, block_label, block_disp, block_vld, block_info, uram_cost, uram_label, uram_disp);
        /* calculation */
        /* --copy uram to LAYER0_PARALL_DEGREE local rams*/
        for (int parall_i = 0; parall_i < LAYER0_PARALL_DEGREE; parall_i++) {
            if (iter < ver_region_num * hor_region_num_parall) {
                layer0CalcParallLoad(iter, parall_i, ver_region_num, hor_region_num, hor_region_num_parall, uram_cost, uram_label, uram_disp, local_cost[parall_i], local_label[parall_i], local_disp[parall_i]);
            }
        }
        /* --calculation in parallel */
        for (int parall_i = 0; parall_i < LAYER0_PARALL_DEGREE; parall_i++) {
            if (iter < ver_region_num * hor_region_num_parall) {
                layerCalc<LAYER0_SIZE, LAYER0_SIZE, false>(iter, parall_i, ver_region_num, hor_region_num, hor_region_num_parall, buff_id, Q6, I6, C6, local_im0_gray[parall_i], local_im1_gray[parall_i], local_im0_census[parall_i], local_im1_census[parall_i], local_cost[parall_i], local_label[parall_i], local_disp[parall_i]);
            }
        }
        /* --copy LAYER0_PARALL_DEGREE local rams to uram */
        for (int parall_i = 0; parall_i < LAYER0_PARALL_DEGREE; parall_i++) {
            if (iter < ver_region_num * hor_region_num_parall) {
                layer0CalcParallStore(iter, parall_i, ver_region_num, hor_region_num, hor_region_num_parall, local_cost[parall_i], local_label[parall_i], local_disp[parall_i], uram_cost, uram_label, uram_disp);
            }
        }
        /* transmission */
        if (iter >= (hor_region_num_parall + 1)){
            layer0Trans(iter, ver_region_num, hor_region_num, hor_region_num_parall, hor_region_num_parall_last, cost_blk_out, label_blk_out, disp_blk_out, uram_cost, uram_label, uram_disp);
        }
        /* toggle ping-pong buffer index */
        buff_id = !buff_id;
    }
}

void localExpLayer1(uchar ddr_im0_gray[WIDTH * HEIGHT], uchar ddr_im1_gray[WIDTH * HEIGHT], uint64_t ddr_im0_census[WIDTH * HEIGHT], uint64_t ddr_im1_census[WIDTH * HEIGHT],
    hls::stream_of_blocks<layer0_cost_blk> &cost_blk_in, hls::stream_of_blocks<layer0_label_blk> &label_blk_in, hls::stream_of_blocks<layer0_disp_blk> &disp_blk_in,
    hls::stream_of_blocks<layer1_cost_blk> &cost_blk_out, hls::stream_of_blocks<layer1_label_blk> &label_blk_out, hls::stream_of_blocks<layer1_disp_blk> &disp_blk_out) {
    // 
    static float uram_cost[LAYER1_SIZE * 4][WIDTH]; // row partition
#pragma HLS array_partition variable=uram_cost type=complete dim=1
    static float uram_label[LAYER1_SIZE * 4][WIDTH][3]; 
#pragma HLS array_partition variable=uram_label type=complete dim=1
#pragma HLS array_partition variable=uram_label type=complete dim=3
    static float uram_disp[LAYER1_SIZE * 4][WIDTH]; 
#pragma HLS array_partition variable=uram_disp type=complete dim=1
    static uchar local_im0_gray[LAYER1_PARALL_DEGREE][2][LAYER1_SIZE * 4][LAYER1_SIZE * 3];
    static uchar local_im1_gray[LAYER1_PARALL_DEGREE][2][LAYER1_SIZE * 4][LAYER1_SIZE * 3 + DISP_MAX];
    static uint64_t local_im0_census[LAYER1_PARALL_DEGREE][2][LAYER1_SIZE * 4][LAYER1_SIZE * 3];
    static uint64_t local_im1_census[LAYER1_PARALL_DEGREE][2][LAYER1_SIZE * 4][LAYER1_SIZE * 3 + DISP_MAX];
    float block_cost[LAYER1_PARALL_DEGREE][LAYER1_SIZE][LAYER1_SIZE];
    float block_label[LAYER1_PARALL_DEGREE][LAYER1_SIZE][LAYER1_SIZE][3];
    float block_disp[LAYER1_PARALL_DEGREE][LAYER1_SIZE][LAYER1_SIZE]; 
    bool block_vld[LAYER1_PARALL_DEGREE];
    int block_info[LAYER1_PARALL_DEGREE][4];
    float local_cost[LAYER1_PARALL_DEGREE][LAYER1_SIZE * 3][LAYER1_SIZE * 3];
    float local_label[LAYER1_PARALL_DEGREE][LAYER1_SIZE * 3][LAYER1_SIZE * 3][3];
    float local_disp[LAYER1_PARALL_DEGREE][LAYER1_SIZE * 3][LAYER1_SIZE * 3];
    // 
    static bool random_generator_init = false;
    static ap_uint<32> Q[4096];
    static ap_uint<32> I = 1731;
    static ap_uint<32> C = 793451;
    if (random_generator_init == false) {
        initRandGen(0, Q, I, C); 
        random_generator_init = true;
    }
    const int l0_hor_region_num = LAYER0_HOR_REGION_NUM;
    const int l0_ver_region_num = LAYER0_VER_REGION_NUM;
    const int l0_hor_region_num_parall = LAYER0_HOR_REGION_NUM_PARALL;
    const int l0_hor_region_num_parall_last = LAYER0_HOR_REGION_NUM_PARALL_LAST;
    const int l1_hor_region_num = LAYER1_HOR_REGION_NUM;
    const int l1_ver_region_num = LAYER1_VER_REGION_NUM;
    const int l1_hor_region_num_parall = LAYER1_HOR_REGION_NUM_PARALL;
    const int l1_hor_region_num_parall_last = LAYER1_HOR_REGION_NUM_PARALL_LAST;
    /* index of ping-pong buffer */
    static bool buff_id = false;
    /* pre cache LAYER1_PARALL_DEGREE im0 & im1 buffer */
    for (int parall_i = 0; parall_i < LAYER1_PARALL_DEGREE; parall_i++) {
        layerCache<LAYER1_PARALL_DEGREE, LAYER1_SIZE, LAYER1_SIZE>(0, parall_i, l1_hor_region_num, l1_hor_region_num_parall, buff_id, ddr_im0_gray, ddr_im1_gray, ddr_im0_census, ddr_im1_census, local_im0_gray[parall_i], local_im1_gray[parall_i], local_im0_census[parall_i], local_im1_census[parall_i]);
    }
    /* iteration */
    int l0_iter = 0;
    for (int l1_iter = -1; l1_iter < (l1_ver_region_num * l1_hor_region_num_parall + (l1_hor_region_num_parall + 1)); l1_iter++) {
        /* cache */
        for (int parall_i = 0; parall_i < LAYER1_PARALL_DEGREE; parall_i++) {
            if (l1_iter < l1_ver_region_num * l1_hor_region_num_parall - 1) {
                layerCache<LAYER1_PARALL_DEGREE, LAYER1_SIZE, LAYER1_SIZE>(l1_iter + 1, parall_i, l1_hor_region_num, l1_hor_region_num_parall, !buff_id, ddr_im0_gray, ddr_im1_gray, ddr_im0_census, ddr_im1_census, local_im0_gray[parall_i], local_im1_gray[parall_i], local_im0_census[parall_i], local_im1_census[parall_i]);
            }
        }
        /* initialization; pre initialization 2 rows of layer1_size for l1_iter=-1 */
        if (l1_iter < (l1_ver_region_num * l1_hor_region_num_parall - 2 * l1_hor_region_num_parall)) {
            layer1Init(l0_iter, l1_iter, l0_hor_region_num, l0_ver_region_num, l0_hor_region_num_parall, l0_hor_region_num_parall_last, l1_hor_region_num, l1_ver_region_num, l1_hor_region_num_parall, 
                cost_blk_in, label_blk_in, disp_blk_in, uram_cost, uram_label, uram_disp);
        }
        /* calculation */
        /* --copy uram to LAYER1_PARALL_DEGREE local rams*/
        for (int parall_i = 0; parall_i < LAYER1_PARALL_DEGREE; parall_i++) {
            if (l1_iter < l1_ver_region_num * l1_hor_region_num_parall) {
                layer1CalcParallLoad(l1_iter, parall_i, l1_ver_region_num, l1_hor_region_num, l1_hor_region_num_parall, uram_cost, uram_label, uram_disp, local_cost[parall_i], local_label[parall_i], local_disp[parall_i]);
            }
        }
        /* --calculation in parallel */
        for (int parall_i = 0; parall_i < LAYER1_PARALL_DEGREE; parall_i++) {
            if (l1_iter < l1_ver_region_num * l1_hor_region_num_parall) {
                layerCalc<LAYER1_SIZE, LAYER1_SIZE, true>(l1_iter, parall_i, l1_ver_region_num, l1_hor_region_num, l1_hor_region_num_parall, buff_id, Q, I, C, local_im0_gray[parall_i], local_im1_gray[parall_i], local_im0_census[parall_i], local_im1_census[parall_i], local_cost[parall_i], local_label[parall_i], local_disp[parall_i]);
            }
        }
        /* --copy LAYER1_PARALL_DEGREE local rams to uram */
        for (int parall_i = 0; parall_i < LAYER1_PARALL_DEGREE; parall_i++) {
            if (l1_iter < l1_ver_region_num * l1_hor_region_num_parall) {
                layer1CalcParallStore(l1_iter, parall_i, l1_ver_region_num, l1_hor_region_num, l1_hor_region_num_parall, local_cost[parall_i], local_label[parall_i], local_disp[parall_i], uram_cost, uram_label, uram_disp);
            }
        }
        /* transmission */
        if (l1_iter >= (l1_hor_region_num_parall + 1)){
            layer1Trans(l1_iter, l1_ver_region_num, l1_hor_region_num, l1_hor_region_num_parall, l1_hor_region_num_parall_last, 
                cost_blk_out, label_blk_out, disp_blk_out, uram_cost, uram_label, uram_disp);
        }
        /* toggle ping-pong buffer index */
        buff_id = !buff_id;
    } 
}

void localExpLayer2(uchar ddr_im0_gray[WIDTH * HEIGHT], uchar ddr_im1_gray[WIDTH * HEIGHT], uint64_t ddr_im0_census[WIDTH * HEIGHT], uint64_t ddr_im1_census[WIDTH * HEIGHT],
    hls::stream_of_blocks<layer1_cost_blk> &cost_blk_in, hls::stream_of_blocks<layer1_label_blk> &label_blk_in, hls::stream_of_blocks<layer1_disp_blk> &disp_blk_in,
    float ddr_cost[WIDTH * HEIGHT], float ddr_label[WIDTH * HEIGHT * 3], float ddr_disp[WIDTH * HEIGHT]) {
    // 
    static float uram_cost[LAYER2_SIZE * 4][WIDTH]; // row partition
#pragma HLS array_partition variable=uram_cost type=complete dim=1
    static float uram_label[LAYER2_SIZE * 4][WIDTH][3]; 
#pragma HLS array_partition variable=uram_label type=complete dim=1
#pragma HLS array_partition variable=uram_label type=complete dim=3
    static float uram_disp[LAYER2_SIZE * 4][WIDTH]; 
#pragma HLS array_partition variable=uram_disp type=complete dim=1
    static uchar local_im0_gray[LAYER2_PARALL_DEGREE][2][LAYER2_SIZE * 4][LAYER2_SIZE * 3];
    static uchar local_im1_gray[LAYER2_PARALL_DEGREE][2][LAYER2_SIZE * 4][LAYER2_SIZE * 3 + DISP_MAX];
    static uint64_t local_im0_census[LAYER2_PARALL_DEGREE][2][LAYER2_SIZE * 4][LAYER2_SIZE * 3];
    static uint64_t local_im1_census[LAYER2_PARALL_DEGREE][2][LAYER2_SIZE * 4][LAYER2_SIZE * 3 + DISP_MAX];
    float block_cost[LAYER2_PARALL_DEGREE][LAYER2_SIZE][LAYER2_SIZE];
    float block_label[LAYER2_PARALL_DEGREE][LAYER2_SIZE][LAYER2_SIZE][3];
    float block_disp[LAYER2_PARALL_DEGREE][LAYER2_SIZE][LAYER2_SIZE]; 
    bool block_vld[LAYER2_PARALL_DEGREE];
    int block_info[LAYER2_PARALL_DEGREE][4];
    float local_cost[LAYER2_PARALL_DEGREE][LAYER2_SIZE * 3][LAYER2_SIZE * 3];
    float local_label[LAYER2_PARALL_DEGREE][LAYER2_SIZE * 3][LAYER2_SIZE * 3][3];
    float local_disp[LAYER2_PARALL_DEGREE][LAYER2_SIZE * 3][LAYER2_SIZE * 3];
    // 
    static bool random_generator_init = false;
    static ap_uint<32> Q[4096];
    static ap_uint<32> I = 1731;
    static ap_uint<32> C = 793451;
    if (random_generator_init == false) {
        initRandGen(0, Q, I, C); 
        random_generator_init = true;
    }
    const int l1_hor_region_num = LAYER1_HOR_REGION_NUM;
    const int l1_ver_region_num = LAYER1_VER_REGION_NUM;
    const int l1_hor_region_num_parall = LAYER1_HOR_REGION_NUM_PARALL;
    const int l1_hor_region_num_parall_last = LAYER1_HOR_REGION_NUM_PARALL_LAST;
    const int l2_hor_region_num = LAYER2_HOR_REGION_NUM;
    const int l2_ver_region_num = LAYER2_VER_REGION_NUM;
    const int l2_hor_region_num_parall = LAYER2_HOR_REGION_NUM_PARALL;
    const int l2_hor_region_num_parall_last = LAYER2_HOR_REGION_NUM_PARALL_LAST;
    /* index of ping-pong buffer */
    static bool buff_id = false;
    /* pre cache LAYER2_PARALL_DEGREE im0 & im1 buffer */
    for (int parall_i = 0; parall_i < LAYER2_PARALL_DEGREE; parall_i++) {
        layerCache<LAYER2_PARALL_DEGREE, LAYER2_SIZE, LAYER2_SIZE>(0, parall_i, l2_hor_region_num, l2_hor_region_num_parall, buff_id, ddr_im0_gray, ddr_im1_gray, ddr_im0_census, ddr_im1_census, local_im0_gray[parall_i], local_im1_gray[parall_i], local_im0_census[parall_i], local_im1_census[parall_i]);
    }
    /* iteration */
    int l1_iter = 0;
    for (int l2_iter = -1; l2_iter < (l2_ver_region_num * l2_hor_region_num_parall + (l2_hor_region_num_parall + 1)); l2_iter++) {
        /* cache */
        for (int parall_i = 0; parall_i < LAYER2_PARALL_DEGREE; parall_i++) {
            if (l2_iter < l2_ver_region_num * l2_hor_region_num_parall - 1) {
                layerCache<LAYER2_PARALL_DEGREE, LAYER2_SIZE, LAYER2_SIZE>(l2_iter + 1, parall_i, l2_hor_region_num, l2_hor_region_num_parall, !buff_id, ddr_im0_gray, ddr_im1_gray, ddr_im0_census, ddr_im1_census, local_im0_gray[parall_i], local_im1_gray[parall_i], local_im0_census[parall_i], local_im1_census[parall_i]);
            }
        }
        /* initialization; pre initialization 2 rows of layer2_size for l2_iter=-1 */
        if (l2_iter < (l2_ver_region_num * l2_hor_region_num_parall - 2 * l2_hor_region_num_parall)) {
            layer2Init(l1_iter, l2_iter, l1_hor_region_num, l1_ver_region_num, l1_hor_region_num_parall, l1_hor_region_num_parall_last, l2_hor_region_num, l2_ver_region_num, l2_hor_region_num_parall, 
                cost_blk_in, label_blk_in, disp_blk_in, uram_cost, uram_label, uram_disp, false);
        }
        /* calculation */
        /* --copy uram to LAYER2_PARALL_DEGREE local rams*/
        for (int parall_i = 0; parall_i < LAYER2_PARALL_DEGREE; parall_i++) {
            if (l2_iter < l2_ver_region_num * l2_hor_region_num_parall) {
                layer2CalcParallLoad(l2_iter, parall_i, l2_ver_region_num, l2_hor_region_num, l2_hor_region_num_parall, uram_cost, uram_label, uram_disp, local_cost[parall_i], local_label[parall_i], local_disp[parall_i]);
            }
        }
        /* --calculation in parallel */
        for (int parall_i = 0; parall_i < LAYER2_PARALL_DEGREE; parall_i++) {
            if (l2_iter < l2_ver_region_num * l2_hor_region_num_parall) {
                layerCalc<LAYER2_SIZE, LAYER2_SIZE, true>(l2_iter, parall_i, l2_ver_region_num, l2_hor_region_num, l2_hor_region_num_parall, buff_id, Q, I, C, local_im0_gray[parall_i], local_im1_gray[parall_i], local_im0_census[parall_i], local_im1_census[parall_i], local_cost[parall_i], local_label[parall_i], local_disp[parall_i]);
            }
        }
        /* --copy LAYER2_PARALL_DEGREE local rams to uram */
        for (int parall_i = 0; parall_i < LAYER2_PARALL_DEGREE; parall_i++) {
            if (l2_iter < l2_ver_region_num * l2_hor_region_num_parall) {
                layer2CalcParallStore(l2_iter, parall_i, l2_ver_region_num, l2_hor_region_num, l2_hor_region_num_parall, local_cost[parall_i], local_label[parall_i], local_disp[parall_i], uram_cost, uram_label, uram_disp);
            }
        }
        /* transmission */
        if (l2_iter >= (l2_hor_region_num_parall + 1)){
            layerTransDDR<LAYER2_SIZE, LAYER2_SIZE, LAYER2_PARALL_DEGREE>(l2_iter, l2_ver_region_num, l2_hor_region_num, l2_hor_region_num_parall, l2_hor_region_num_parall_last, 
                ddr_cost, ddr_label, ddr_disp, uram_cost, uram_label, uram_disp);
        }
        /* toggle ping-pong buffer index */
        buff_id = !buff_id;
    }
}
#endif // COST_CALC_MODE == 1

#if COST_LOAD_MODE == 1 && SCAN_DIRECTION == 0
template<size_t CUR_SIZE>
void costLabelUpdate(int unary_cost_x0, int unary_cost_x1, int unary_cost_x2, 
    int expn_region_x, int expn_region_y, int expn_region_w, int expn_region_h, int unit_region_x,
    float plane_a, float plane_b, float plane_c, 
    float local_unary_cost[CUR_SIZE * 3][4][CUR_SIZE][DISP_MAX], uchar local_im0_gray[CUR_SIZE * 3][CUR_SIZE * 3], 
    float local_cost[CUR_SIZE * 3][CUR_SIZE * 3], float local_label[CUR_SIZE * 3][CUR_SIZE * 3][3], float local_disp[CUR_SIZE * 3][CUR_SIZE * 3],
    bool is_binary_enable, bool do_gc, bool is_debug = false) {
// #pragma HLS array_partition variable=local_unary_cost type=complete dim=1
#pragma HLS inline off
    //
    uchar gray_reg[CUR_SIZE * 3][3];
	#pragma HLS array_partition variable=gray_reg type=complete dim=0
	float disp_reg[CUR_SIZE * 3][3];
	#pragma HLS array_partition variable=disp_reg type=complete dim=0	
    // cost for graph cut
    float cost00[4][CUR_SIZE * 3][CUR_SIZE * 3], cost01[4][CUR_SIZE * 3][CUR_SIZE * 3], cost10[4][CUR_SIZE * 3][CUR_SIZE * 3];
    float proposalCosts[CUR_SIZE * 3][CUR_SIZE * 3];
    bool updateMask[CUR_SIZE * 3][CUR_SIZE * 3];
    const float th_smooth = 1.0f;
    const float epsilon = 0.01f;
#if !KITTI
    const float omega = 10.0f;
    const float lambda = 0.5f;
#else
    const float omega = 30.0f;
    const float lambda = 1.0f;
#endif
#if !defined(__SYNTHESIS__)
    // show disp & gray image
    if (is_debug == true) {
        printf("left-up corner:(%d, %d); w:%d, h:%d\n", expn_region_x, expn_region_y, expn_region_w, expn_region_h);
        //
        cv::Mat disp = cv::Mat::zeros(expn_region_h, expn_region_w, CV_MAKE_TYPE(cv::DataType<float>::depth, 1));
        for (int y = 0; y < expn_region_h; y++) {
	    	for (int x = 0; x < expn_region_w; x++) {
	    		disp.at<float>(y, x) = local_disp[y][x];
	    	}
	    }
        cv::imwrite("disp_test_pre.png", disp);
    }
#endif // !defined(__SYNTHESIS__)
    /* load to last colume of data registers*/
    LOOP_2ND_CACHE0:
    for (int i = 0; i < CUR_SIZE * 3; i++) {
#pragma HLS unroll
        gray_reg[i][2] = local_im0_gray[i][0];
        disp_reg[i][2] = local_disp[i][0];
    }
    // 
    LOOP_CALC_HOR:
    for (int x = 0; x < CUR_SIZE * 3; x++) {
#pragma HLS pipeline II=1
        if (x >= expn_region_w)
            continue;
        int x_remainder;
        if (x >= CUR_SIZE * 2)
            x_remainder = x - CUR_SIZE * 2;
        else if (x >= CUR_SIZE)
            x_remainder = x - CUR_SIZE;
        else
            x_remainder = x;
        int local_x_id = 0;
        if (expn_region_x == unit_region_x) {
            if (x < CUR_SIZE)
                local_x_id = unary_cost_x1;
            else if (x < CUR_SIZE * 2)
                local_x_id = unary_cost_x2;
        }
        else {
            if (x < CUR_SIZE)
                local_x_id = unary_cost_x0;
            else if (x < CUR_SIZE * 2)
                local_x_id = unary_cost_x1;
            else
                local_x_id = unary_cost_x2;
        }
        /* update data registers for binary cost calculation */
        /* shift first 2 columns */
        LOOP_2ND_CACHE1:
        for (int j = 0; j < 2; j++) {
#pragma HLS unroll
            for (int i = 0; i < CUR_SIZE * 3; i++) {
#pragma HLS unroll
                gray_reg[i][j] = gray_reg[i][j + 1];
                disp_reg[i][j] = disp_reg[i][j + 1];
            }
        }
        /* load to last column */
        if (x < (expn_region_w - 1)) {
            LOOP_2ND_CACHE2:
            for (int i = 0; i < CUR_SIZE * 3; i++) {
#pragma HLS unroll
                gray_reg[i][2] = local_im0_gray[i][x + 1];
                disp_reg[i][2] = local_disp[i][x + 1];
            }
        }
        float d_prop_x = plane_a * (expn_region_x + x) + plane_c;
        LOOP_CALC_VER:
        for (int y = 0; y < CUR_SIZE * 3; y++) {
#pragma HLS unroll
// #pragma HLS dependence variable=local_cost type=inter false
// #pragma HLS dependence variable=local_label type=inter false
// #pragma HLS pipeline II=1
            if (y >= expn_region_h)
                continue;
            //
            float d_prop = d_prop_x + plane_b * (expn_region_y + y);
            float cost_unary_prop = 0.0;
            float cost_binary_prop = 0.0;
            float cost_unary_orig = local_cost[y][x];
            float cost_binary_orig = 0.0;
            float d_orig = disp_reg[y][1];
            uchar p_gray = gray_reg[y][1];
            /* unary cost */
            if (d_prop < DISP_MIN) cost_unary_prop = local_unary_cost[y][local_x_id][x_remainder][0];
            else if (d_prop >= DISP_MAX - 1) cost_unary_prop = local_unary_cost[y][local_x_id][x_remainder][DISP_MAX - 1];
            else {
                int d_l = std::floor(d_prop);
                int d_h = d_l + 1;
                float f_h = d_prop - d_l;
                float f_l = 1.0 - f_h;
                cost_unary_prop = f_l * local_unary_cost[y][local_x_id][x_remainder][d_l] + f_h * local_unary_cost[y][local_x_id][x_remainder][d_h];
            }
            proposalCosts[y][x] = cost_unary_prop;
            /* binary cost */
            if (is_binary_enable == true) {
                if (do_gc) {
                    for (int i = 0; i < 3; i++) {
                        for (int j = 0; j < 3; j++) {
                            int index = i * 3 + j;
                            if (index > 4) {
                                int x_q = x + j - 1;
                                int y_q = y + i - 1;

                                int e_x_q = x_q + expn_region_x;
                                int e_y_q = y_q + expn_region_y;
                                bool q_e_flag = (e_x_q < 0 || e_x_q >= WIDTH || e_y_q < 0 || e_y_q >= HEIGHT);
                                bool q_flag = (x_q < 0 || x_q >= expn_region_w || y_q < 0 || y_q >= expn_region_h);
                                // dp(fp), dp(fq), dp(alpha), dq(fq), dq(fp), dq(alpha)
                                float dq_fq = q_flag ? 0 : disp_reg[y_q][j];
                                // float dp_fp = d_orig;
                                // float dp_alpha = d_prop;
                                float dq_alpha = q_e_flag ? 0 : (plane_a * (e_x_q) + plane_b * (e_y_q) + plane_c);
                                float dp_fq = q_flag ? 0 : (local_label[y_q][x_q][0] * (expn_region_x + x) + local_label[y_q][x_q][1] * (expn_region_y + y) + local_label[y_q][x_q][2]);
                                float dq_fp = q_e_flag ? 0 : (local_label[y][x][0] * (e_x_q) + local_label[y][x][1] * (e_y_q) + local_label[y][x][2]);

                                uchar p_gray_nb = gray_reg[y_q][j];
                                float weight = std::exp(-3 * std::abs(p_gray_nb - p_gray) / omega);
                                weight = std::fmax(epsilon, weight);
                                int nb = index - 5;
                                float tmp = 0.;
                                // | dp(fp) - dp(fq) | + | dq(fp) - dq(fq) |
                                tmp = std::abs(d_orig - dp_fq) + std::abs(dq_fp - dq_fq);
                                tmp = (tmp > th_smooth) ? th_smooth : tmp;
                                cost00[nb][y][x] = tmp * weight * lambda * DISP_DDR_VAL_MAX;
                                // | dp(fp) - dp(alpha) | + | dq(fp) - dq(alpha) |
                                tmp = std::abs(d_orig - d_prop) + std::abs(dq_fp - dq_alpha);
                                tmp = (tmp > th_smooth) ? th_smooth : tmp;
                                cost01[nb][y][x] = tmp * weight * lambda * DISP_DDR_VAL_MAX;
                                // | dp(alpha) - dp(fq) | + | dq(alpha) - dq(fq) |
                                tmp = std::abs(d_prop - dp_fq) + std::abs(dq_alpha - dq_fq);
                                tmp = (tmp > th_smooth) ? th_smooth : tmp;
                                cost10[nb][y][x] = tmp * weight * lambda * DISP_DDR_VAL_MAX;
                            }
                        }
                    }
                }
                else {
                LOOP_BINARY_CALC:
                    for (int i = 0; i < 3; i++) {
#pragma HLS unroll
                        for (int j = 0; j < 3; j++) {
#pragma HLS unroll
                            if (((i == 1) && (j == 1)) || ((x + j - 1) < 0) || ((x + j - 1) >= expn_region_w) || ((y + i - 1) < 0) || ((y + i - 1) >= expn_region_h) || ((y + i - 1) >= CUR_SIZE * 3))
                                continue;
                            float d_nb = disp_reg[y + i - 1][j];
                            uchar p_gray_nb = gray_reg[y + i - 1][j];
                            float weight = (std::abs(p_gray_nb - p_gray) < 10) ? 0.32 : 0.08;
                            cost_binary_prop += weight * std::abs(d_prop - d_nb) * DISP_DDR_VAL_MAX;
                            cost_binary_orig += weight * std::abs(d_orig - d_nb) * DISP_DDR_VAL_MAX;
                        }
                    }
                }
            }
            if (do_gc == false) {
                /* update local buffer */
                if ((cost_unary_prop + cost_binary_prop) < (cost_unary_orig + cost_binary_orig)) {
                    local_cost[y][x] = cost_unary_prop;
                    local_label[y][x][0] = plane_a; local_label[y][x][1] = plane_b; local_label[y][x][2] = plane_c;
                    local_disp[y][x] = d_prop;
                }
            }
        }
    }
#if !defined(__SYNTHESIS__)
    // show disp & gray image
    if (is_debug == true) {
        //
        cv::Mat disp = cv::Mat::zeros(expn_region_h, expn_region_w, CV_MAKE_TYPE(cv::DataType<float>::depth, 1));
        for (int y = 0; y < expn_region_h; y++) {
	    	for (int x = 0; x < expn_region_w; x++) {
	    		disp.at<float>(y, x) = local_disp[y][x];
	    	}
	    }
        cv::imwrite("disp_test_post.png", disp);
        // cv::waitKey();
        cv::Mat gray = cv::Mat::zeros(expn_region_h, expn_region_w, CV_MAKE_TYPE(cv::DataType<uchar>::depth, 1));
        for (int y = 0; y < expn_region_h; y++) {
	    	for (int x = 0; x < expn_region_w; x++) {
	    		gray.at<uchar>(y, x) = local_im0_gray[y][x];
	    	}
	    }
        cv::imwrite("gray_test.png", gray);
    }
#endif // !defined(__SYNTHESIS__)
    if (is_binary_enable == true && do_gc == true) {
        expansionMoveBK<CUR_SIZE>(expn_region_w, expn_region_h, local_cost, proposalCosts, cost00, cost01, cost10, updateMask);
        /* update local buffer */
        updateLocalBuffer<CUR_SIZE>(expn_region_x, expn_region_y, expn_region_w, expn_region_h, proposalCosts, updateMask, local_cost, local_disp, local_label, plane_a, plane_b, plane_c);
    }
}

ap_uint<16> bit_select_512b_16b(ap_uint<6> d_l, ap_uint<512> c_l_in) {
#pragma HLS inline off
    ap_uint<256> c_l_0 = (d_l[5] == 0) ? c_l_in.range(255, 0) : c_l_in.range(511, 256);
    ap_uint<128> c_l_1 = (d_l[4] == 0) ? c_l_0.range(127, 0) : c_l_0.range(255, 128);
    ap_uint<64>  c_l_2 = (d_l[3] == 0) ? c_l_1.range(63, 0) : c_l_1.range(127, 64);
    ap_uint<32>  c_l_3 = (d_l[2] == 0) ? c_l_2.range(31, 0) : c_l_2.range(63, 32);
    ap_uint<16>  c_l_4 = (d_l[1] == 0) ? c_l_3.range(15, 0) : c_l_3.range(31, 16);
    ap_uint<8>   c_l_5 = (d_l[0] == 0) ? c_l_4.range(7, 0)  : c_l_4.range(15, 8);
    return c_l_5;
}

ap_uint<16> bit_select_128b_8b(ap_uint<4> d_l, ap_uint<128> c_l_in) {
#pragma HLS inline off
    ap_uint<64>  c_l_2 = (d_l[3] == 0) ? c_l_in.range(63, 0) : c_l_in.range(127, 64);
    ap_uint<32>  c_l_3 = (d_l[2] == 0) ? c_l_2.range(31, 0) : c_l_2.range(63, 32);
    ap_uint<16>  c_l_4 = (d_l[1] == 0) ? c_l_3.range(15, 0) : c_l_3.range(31, 16);
    ap_uint<8>   c_l_5 = (d_l[0] == 0) ? c_l_4.range(7, 0)  : c_l_4.range(15, 8);
    return c_l_5;
}

ap_uint<8> bit_select_96b_6b(ap_uint<4> d_l, ap_uint<96> c_l_in) {
#pragma HLS inline off
    ap_uint<48>  c_l_2 = (d_l[3] == 0) ? c_l_in.range(47, 0) : c_l_in.range(95, 48);
    ap_uint<24>  c_l_3 = (d_l[2] == 0) ? c_l_2.range(23, 0) : c_l_2.range(47, 24);
    ap_uint<12>  c_l_4 = (d_l[1] == 0) ? c_l_3.range(11, 0) : c_l_3.range(23, 12);
    ap_uint<6>   c_l_5 = (d_l[0] == 0) ? c_l_4.range(5, 0)  : c_l_4.range(11, 6);
    return c_l_5;
}

ap_uint<8> bit_select_108b(ap_uint<4> d_l, ap_uint<108> c_l_in) {
#pragma HLS inline off
    ap_uint<54>  c_l_2 = (d_l[3] == 0) ? c_l_in.range(53, 0) : c_l_in.range(107, 54);
    ap_uint<27>  c_l_3 = (d_l[2] == 0) ? c_l_2.range(26, 0) : c_l_2.range(53, 27);
    ap_uint<28>  c_l_3_tmp = c_l_3 << 1;
    ap_uint<14>  c_l_4 = (d_l[1] == 0) ? c_l_3_tmp.range(13, 0) : c_l_3_tmp.range(27, 14);
    ap_uint<7>   c_l_5 = (d_l[0] == 0) ? c_l_4.range(6, 0)  : c_l_4.range(13, 7);
    return c_l_5;
}

ap_uint<6> bit_select_96b(ap_uint<4> d_l, ap_uint<96> c_l_in) {
#pragma HLS inline off
    ap_uint<48>  c_l_2 = (d_l[3] == 0) ? c_l_in.range(47, 0) : c_l_in.range(95, 48);
    ap_uint<24>  c_l_3 = (d_l[2] == 0) ? c_l_2.range(23, 0) : c_l_2.range(47, 24);
    ap_uint<12>  c_l_4 = (d_l[1] == 0) ? c_l_3.range(11, 0) : c_l_3.range(23, 12);
    ap_uint<6>   c_l_5 = (d_l[0] == 0) ? c_l_4.range(5, 0)  : c_l_4.range(11, 6);
    return c_l_5;
}

ap_uint<16> bit_select_256b(ap_uint<4> d_l, ap_uint<256> c_l_in) {
#pragma HLS inline off
    ap_uint<128> c_l_2 = (d_l[3] == 0) ? c_l_in.range(127, 0) : c_l_in.range(255, 128);
    ap_uint<64>  c_l_3 = (d_l[2] == 0) ? c_l_2.range(63, 0) : c_l_2.range(127, 64);
    ap_uint<32>  c_l_4 = (d_l[1] == 0) ? c_l_3.range(31, 0) : c_l_3.range(63, 32);
    ap_uint<16>  c_l_5 = (d_l[0] == 0) ? c_l_4.range(15, 0)  : c_l_4.range(31, 16);
    return c_l_5;
}

ap_uint<8> bit_select_128b(ap_uint<4> d_l, ap_uint<128> c_l_in) {
#pragma HLS inline off
    ap_uint<64>  c_l_2 = (d_l[3] == 0) ? c_l_in.range(63, 0) : c_l_in.range(127, 64);
    ap_uint<32>  c_l_3 = (d_l[2] == 0) ? c_l_2.range(31, 0) : c_l_2.range(63, 32);
    ap_uint<16>  c_l_4 = (d_l[1] == 0) ? c_l_3.range(15, 0) : c_l_3.range(31, 16);
    ap_uint<8>   c_l_5 = (d_l[0] == 0) ? c_l_4.range(7, 0)  : c_l_4.range(15, 8);
    return c_l_5;
}

void layer0CostLabelUpdate(ap_uint<3> unary_cost_x0, ap_uint<3> unary_cost_x1, ap_uint<3> unary_cost_x2, ap_uint<11> expn_region_x, ap_uint<11> expn_region_y, 
    ap_uint<5> expn_region_w, ap_uint<5> expn_region_h, ap_uint<11> unit_region_x,
    label_range plane_a[LAYER0_PROP_NUM], label_range plane_b[LAYER0_PROP_NUM], label_range plane_c[LAYER0_PROP_NUM], 
    label_range plane_a1[LAYER0_PROP_NUM], label_range plane_b1[LAYER0_PROP_NUM], label_range plane_c1[LAYER0_PROP_NUM], 
    unary_cost_t local_unary_cost[LAYER0_SIZE * 3][4][LAYER0_SIZE][DISP_DDR_MAX], 
    ap_uint<DISP_DDR_BIT> local_cost[LAYER0_SIZE * 3][LAYER0_SIZE * 3], label_range local_label[LAYER0_SIZE * 3][LAYER0_SIZE * 3][3], disp_range local_disp[LAYER0_SIZE * 3][LAYER0_SIZE * 3]) {
#pragma HLS array_partition variable=plane_a type=complete dim=1
#pragma HLS array_partition variable=plane_b type=complete dim=1
#pragma HLS array_partition variable=plane_c type=complete dim=1
#pragma HLS array_partition variable=local_unary_cost type=complete dim=1
#pragma HLS array_partition variable=local_unary_cost type=complete dim=2
//DO_PRAGMA(HLS array_reshape variable=local_unary_cost type=cyclic factor=DISP_DDR_NUM dim=4) 
// #pragma HLS INTERFACE mode=ap_memory port=local_cost storage_type=ram_2p
#pragma HLS array_partition variable=local_cost type=complete dim=1
#pragma HLS array_partition variable=local_label type=complete dim=1
#pragma HLS array_partition variable=local_label type=complete dim=3
#pragma HLS array_partition variable=local_disp type=complete dim=1
#pragma HLS inline off
    assert((unary_cost_x0 != unary_cost_x1) && (unary_cost_x0 != unary_cost_x2) && (unary_cost_x1 != unary_cost_x2));
    assert(unary_cost_x0 >= 0 && unary_cost_x0 < 4 && unary_cost_x1 >= 0 && unary_cost_x1 < 4 && unary_cost_x2 >= 0 && unary_cost_x2 < 4);
    // 
    bool is_left_region = (expn_region_x == unit_region_x);
    //
    for (int px = 0; px < LAYER0_PROP_NUM * LAYER0_SIZE * 3 * 2; px++) {
    // for (int p = 0; p < 3; p++) {
    //     // LOOP_CALC_HOR:
    //     for (int x = 0; x < LAYER0_SIZE * 3; x++) {
#pragma HLS loop_flatten
#pragma HLS pipeline II=1
            // LOOP_CALC_VER:
            // int x_base = buf_x[x/LAYER0_SIZE];
            // int x_ofst = x%LAYER0_SIZE;
            int index = (px / (3 * LAYER0_SIZE)) / LAYER0_PROP_NUM;
            ap_uint<4> p = (px / (3 * LAYER0_SIZE)) % LAYER0_PROP_NUM; // (4, 0)
            plane_a[p] = (index == 0) ? plane_a[p] : plane_a1[p];
            plane_b[p] = (index == 0) ? plane_b[p] : plane_b1[p];
            plane_c[p] = (index == 0) ? plane_c[p] : plane_c1[p];
            int x = px % (3 * LAYER0_SIZE); // (5, 0)
            ap_uint<4> x_ofst = x % LAYER0_SIZE; // (4, 0)
            ap_uint<3> x_base; // (3, 0)
            if (is_left_region) {
                if (x < LAYER0_SIZE)
                    x_base = unary_cost_x1;
                else if (x < LAYER0_SIZE * 2)
                    x_base = unary_cost_x2;
            }
            else {
                if (x < LAYER0_SIZE)
                    x_base = unary_cost_x0;
                else if (x < LAYER0_SIZE * 2)
                    x_base = unary_cost_x1;
                else
                    x_base = unary_cost_x2;
            }
            ap_int<17> d_prop_base = plane_a[p] * (expn_region_x + x) + plane_c[p];
            for (int y = 0; y < LAYER0_SIZE * 3; y++) {
#pragma HLS unroll
                ap_int<17> d_prop = d_prop_base + plane_b[p] * (expn_region_y + y);
                ap_uint<DISP_DDR_BIT> cost_unary_prop = 0;
                ap_uint<DISP_DDR_BIT> cost_unary_orig = local_cost[y][x];
                /* unary cost */
                // if (d_prop < DISP_MIN) cost_unary_prop = local_unary_cost[y][x_base][x_ofst][0].range(6-1,0) << 1;
                // else if (d_prop.range(16,8) >= DISP_MAX - 1) cost_unary_prop = local_unary_cost[y][x_base][x_ofst][DISP_DDR_MAX-1].range(107,101);
                if (d_prop< DISP_MIN) cost_unary_prop = local_unary_cost[y][x_base][x_ofst][0].range(DISP_DDR_BIT-1,0);
                else if (d_prop.range(16,8) >= DISP_MAX - 1) cost_unary_prop = local_unary_cost[y][x_base][x_ofst][DISP_DDR_MAX-1].range(DISP_DDR_BIT*DISP_DDR_NUM-1,DISP_DDR_BIT*DISP_DDR_NUM-DISP_DDR_BIT);
                else {
                    // int d_l = d_prop.to_int();
                    ap_uint<8> d_l_int = d_prop.range(15,8);
                    if (d_prop[7]==1)
                        d_l_int = d_l_int + 1;
                    unary_cost_t c_l_int = local_unary_cost[y][x_base][x_ofst][d_l_int.range(7, DISP_DDR_NUM_LOG2)];
                    // ap_uint<8>  c_l_5_int =  bit_select_108b(d_l_int.range(DISP_DDR_NUM_LOG2-1, 0), c_l_512b_int);
#if DISP_DDR_BIT == 6
                    ap_uint<DISP_DDR_BIT>  c_l_5_int =  bit_select_96b(d_l_int.range(DISP_DDR_NUM_LOG2-1, 0), c_l_int);
#elif DISP_DDR_BIT == 8
                    ap_uint<DISP_DDR_BIT>  c_l_5_int =  bit_select_128b(d_l_int.range(DISP_DDR_NUM_LOG2-1, 0), c_l_int);
#elif DISP_DDR_BIT == 16
                    ap_uint<DISP_DDR_BIT>  c_l_5_int =  bit_select_256b(d_l_int.range(DISP_DDR_NUM_LOG2-1, 0), c_l_int);
#endif
                    cost_unary_prop = c_l_5_int;
                }             
                /* update local buffer */
                if (cost_unary_prop < cost_unary_orig && ((x < expn_region_w) && (y < expn_region_h))) {
                    local_cost[y][x] = cost_unary_prop;
                    local_label[y][x][0] = plane_a[p]; local_label[y][x][1] = plane_b[p]; local_label[y][x][2] = plane_c[p];
                    local_disp[y][x] = d_prop;
                }
            }
        // }
    }
}

void layer0CostLabelUpdate(ap_uint<3> unary_cost_x0, ap_uint<3> unary_cost_x1, ap_uint<3> unary_cost_x2, ap_uint<11> expn_region_x, ap_uint<11> expn_region_y, 
    ap_uint<5> expn_region_w, ap_uint<5> expn_region_h, ap_uint<11> unit_region_x,
    label_range plane_a[LAYER0_PROP_NUM], label_range plane_b[LAYER0_PROP_NUM], label_range plane_c[LAYER0_PROP_NUM], 
    unary_cost_t local_unary_cost[LAYER0_SIZE * 3][4][LAYER0_SIZE][DISP_DDR_MAX], 
    ap_uint<DISP_DDR_BIT> local_cost[LAYER0_SIZE * 3][LAYER0_SIZE * 3], label_range local_label[LAYER0_SIZE * 3][LAYER0_SIZE * 3][3], disp_range local_disp[LAYER0_SIZE * 3][LAYER0_SIZE * 3]) {
#pragma HLS array_partition variable=plane_a type=complete dim=1
#pragma HLS array_partition variable=plane_b type=complete dim=1
#pragma HLS array_partition variable=plane_c type=complete dim=1
#pragma HLS array_partition variable=local_unary_cost type=complete dim=1
#pragma HLS array_partition variable=local_unary_cost type=complete dim=2
//DO_PRAGMA(HLS array_reshape variable=local_unary_cost type=cyclic factor=DISP_DDR_NUM dim=4) 
// #pragma HLS INTERFACE mode=ap_memory port=local_cost storage_type=ram_2p
#pragma HLS array_partition variable=local_cost type=complete dim=1
#pragma HLS array_partition variable=local_label type=complete dim=1
#pragma HLS array_partition variable=local_label type=complete dim=3
#pragma HLS array_partition variable=local_disp type=complete dim=1
#pragma HLS inline off
    assert((unary_cost_x0 != unary_cost_x1) && (unary_cost_x0 != unary_cost_x2) && (unary_cost_x1 != unary_cost_x2));
    assert(unary_cost_x0 >= 0 && unary_cost_x0 < 4 && unary_cost_x1 >= 0 && unary_cost_x1 < 4 && unary_cost_x2 >= 0 && unary_cost_x2 < 4);
    // 
    bool is_left_region = (expn_region_x == unit_region_x);
    //
    for (int px = 0; px < LAYER0_PROP_NUM * LAYER0_SIZE * 3; px++) {
    // for (int p = 0; p < 3; p++) {
    //     // LOOP_CALC_HOR:
    //     for (int x = 0; x < LAYER0_SIZE * 3; x++) {
#pragma HLS loop_flatten
#pragma HLS pipeline II=1
            // LOOP_CALC_VER:
            // int x_base = buf_x[x/LAYER0_SIZE];
            // int x_ofst = x%LAYER0_SIZE;
            ap_uint<4> p = px / (3 * LAYER0_SIZE); // (4, 0)
            int x = px % (3 * LAYER0_SIZE); // (5, 0)
            ap_uint<4> x_ofst = x % LAYER0_SIZE; // (4, 0)
            ap_uint<3> x_base; // (3, 0)
            if (is_left_region) {
                if (x < LAYER0_SIZE)
                    x_base = unary_cost_x1;
                else if (x < LAYER0_SIZE * 2)
                    x_base = unary_cost_x2;
            }
            else {
                if (x < LAYER0_SIZE)
                    x_base = unary_cost_x0;
                else if (x < LAYER0_SIZE * 2)
                    x_base = unary_cost_x1;
                else
                    x_base = unary_cost_x2;
            }
            ap_int<17> d_prop_base = plane_a[p] * (expn_region_x + x) + plane_c[p];
            for (int y = 0; y < LAYER0_SIZE * 3; y++) {
#pragma HLS unroll
                ap_int<17> d_prop = d_prop_base + plane_b[p] * (expn_region_y + y);
                ap_uint<DISP_DDR_BIT> cost_unary_prop = 0;
                ap_uint<DISP_DDR_BIT> cost_unary_orig = local_cost[y][x];
                /* unary cost */
                // if (d_prop < DISP_MIN) cost_unary_prop = local_unary_cost[y][x_base][x_ofst][0].range(6-1,0) << 1;
                // else if (d_prop.range(16,8) >= DISP_MAX - 1) cost_unary_prop = local_unary_cost[y][x_base][x_ofst][DISP_DDR_MAX-1].range(107,101);
                if (d_prop< DISP_MIN) cost_unary_prop = local_unary_cost[y][x_base][x_ofst][0].range(DISP_DDR_BIT-1,0);
                else if (d_prop.range(16,8) >= DISP_MAX - 1) cost_unary_prop = local_unary_cost[y][x_base][x_ofst][DISP_DDR_MAX-1].range(DISP_DDR_BIT*DISP_DDR_NUM-1,DISP_DDR_BIT*DISP_DDR_NUM-DISP_DDR_BIT);
                else {
                    // int d_l = d_prop.to_int();
                    ap_uint<8> d_l_int = d_prop.range(15,8);
                    if (d_prop[7]==1)
                        d_l_int = d_l_int + 1;
                    unary_cost_t c_l_int = local_unary_cost[y][x_base][x_ofst][d_l_int.range(7, DISP_DDR_NUM_LOG2)];
                    // ap_uint<8>  c_l_5_int =  bit_select_108b(d_l_int.range(DISP_DDR_NUM_LOG2-1, 0), c_l_512b_int);
#if DISP_DDR_BIT == 6
                    ap_uint<DISP_DDR_BIT>  c_l_5_int =  bit_select_96b(d_l_int.range(DISP_DDR_NUM_LOG2-1, 0), c_l_int);
#elif DISP_DDR_BIT == 8
                    ap_uint<DISP_DDR_BIT>  c_l_5_int =  bit_select_128b(d_l_int.range(DISP_DDR_NUM_LOG2-1, 0), c_l_int);
#elif DISP_DDR_BIT == 16
                    ap_uint<DISP_DDR_BIT>  c_l_5_int =  bit_select_256b(d_l_int.range(DISP_DDR_NUM_LOG2-1, 0), c_l_int);
#endif
                    cost_unary_prop = c_l_5_int;
                }             
                /* update local buffer */
                if (cost_unary_prop < cost_unary_orig && ((x < expn_region_w) && (y < expn_region_h))) {
                    local_cost[y][x] = cost_unary_prop;
                    local_label[y][x][0] = plane_a[p]; local_label[y][x][1] = plane_b[p]; local_label[y][x][2] = plane_c[p];
                    local_disp[y][x] = d_prop;
                }
            }
        // }
    }
}


void layer0CostLabelUpdate(ap_uint<3> unary_cost_x0, ap_uint<3> unary_cost_x1, ap_uint<3> unary_cost_x2, 
    ap_uint<11> expn_region_x, ap_uint<11> expn_region_y, ap_uint<6> expn_region_w, ap_uint<6> expn_region_h, ap_uint<11> unit_region_x,
    label_range plane_a[LAYER0_PROP_NUM], label_range plane_b[LAYER0_PROP_NUM], label_range plane_c[LAYER0_PROP_NUM], 
    unary_cost_t local_unary_cost[LAYER0_SIZE * 3][4][LAYER0_SIZE][DISP_DDR_MAX], uchar local_im0_gray[LAYER0_SIZE * 3][LAYER0_SIZE * 3], 
    ap_uint<DISP_DDR_BIT> local_cost[LAYER0_SIZE * 3][LAYER0_SIZE * 3], label_range local_label[LAYER0_SIZE * 3][LAYER0_SIZE * 3][3], disp_range local_disp[LAYER0_SIZE * 3][LAYER0_SIZE * 3]) {
#pragma HLS inline off
#pragma HLS array_partition variable=local_unary_cost type=complete dim=1
#pragma HLS array_partition variable=local_unary_cost type=complete dim=2
//DO_PRAGMA(HLS array_reshape variable=local_unary_cost type=cyclic factor=DISP_DDR_NUM dim=4) 
#pragma HLS array_partition variable=local_im0_gray type=complete dim=1
#pragma HLS array_partition variable=local_cost type=complete dim=1
#pragma HLS array_partition variable=local_label type=complete dim=1
#pragma HLS array_partition variable=local_label type=complete dim=3
#pragma HLS array_partition variable=local_disp type=complete dim=1
#pragma HLS array_partition variable=plane_a type=complete dim=1
#pragma HLS array_partition variable=plane_b type=complete dim=1
#pragma HLS array_partition variable=plane_c type=complete dim=1
    assert((unary_cost_x0 != unary_cost_x1) && (unary_cost_x0 != unary_cost_x2) && (unary_cost_x1 != unary_cost_x2));
    assert(unary_cost_x0 >= 0 && unary_cost_x0 < 4 && unary_cost_x1 >= 0 && unary_cost_x1 < 4 && unary_cost_x2 >= 0 && unary_cost_x2 < 4);
    assert(expn_region_h <= (LAYER0_SIZE * 3));
    assert(expn_region_w <= (LAYER0_SIZE * 3));
    //
    bool is_left_region = (expn_region_x == unit_region_x);
    //
    ap_uint<8> gray_reg[LAYER0_SIZE * 3][3];
#pragma HLS array_partition variable=gray_reg type=complete dim=0
	disp_range disp_reg[LAYER0_SIZE * 3][3];
#pragma HLS array_partition variable=disp_reg type=complete dim=0	
    /* load to last colume of data registers*/
    LOOP_2ND_CACHE0:
    for (int i = 0; i < LAYER0_SIZE * 3; i++) {
#pragma HLS unroll
        gray_reg[i][2] = local_im0_gray[i][0];
        disp_reg[i][2] = local_disp[i][0];
    }
    // 
//     for (int p = 0; p < LAYER0_PROP_NUM; p++) {
//         for (int x = 0; x < LAYER0_SIZE * 3; x++) {
// #pragma HLS loop_flatten
// #pragma HLS pipeline II=1
    for (int px = 0; px < LAYER0_PROP_NUM * LAYER0_SIZE * 3; px++) {
#pragma HLS pipeline II=2
            int p = px / (3 * LAYER0_SIZE);
            int x = px % (3 * LAYER0_SIZE);
            ap_uint<5> x_ofst;
            if (x >= LAYER0_SIZE * 2)
                x_ofst = x - LAYER0_SIZE * 2;
            else if (x >= LAYER0_SIZE)
                x_ofst = x - LAYER0_SIZE;
            else
                x_ofst = x;
            ap_uint<3> x_base = 0;
            if (is_left_region) {
                if (x < LAYER0_SIZE)
                    x_base = unary_cost_x1;
                else if (x < LAYER0_SIZE * 2)
                    x_base = unary_cost_x2;
            }
            else {
                if (x < LAYER0_SIZE)
                    x_base = unary_cost_x0;
                else if (x < LAYER0_SIZE * 2)
                    x_base = unary_cost_x1;
                else
                    x_base = unary_cost_x2;
            }
            /* update data registers for binary cost calculation */
            /* shift first 2 columns */
            LOOP_2ND_CACHE1:
            for (int j = 0; j < 2; j++) {
#pragma HLS unroll
                for (int i = 0; i < LAYER0_SIZE * 3; i++) {
#pragma HLS unroll
                    gray_reg[i][j] = gray_reg[i][j + 1];
                    disp_reg[i][j] = disp_reg[i][j + 1];
                }
            }
            /* load to last column */
            if (x < (expn_region_w - 1)) {
                LOOP_2ND_CACHE2:
                for (int i = 0; i < LAYER0_SIZE * 3; i++) {
#pragma HLS unroll
                    gray_reg[i][2] = local_im0_gray[i][x + 1];
                    disp_reg[i][2] = local_disp[i][x + 1];
                }
            }
            ap_int<17> d_prop_x = plane_a[p] * (expn_region_x + x) + plane_c[p];
            LOOP_CALC_VER:
            for (int y = 0; y < LAYER0_SIZE * 3; y++) {
#pragma HLS unroll
// #pragma HLS dependence variable=local_cost type=inter false
// #pragma HLS dependence variable=local_label type=inter false
// #pragma HLS pipeline II=1
                //
                disp_range d_prop = d_prop_x + plane_b[p] * (expn_region_y + y);
                ap_uint<DISP_DDR_BIT> cost_unary_prop = 0;
                ap_uint<19> cost_binary_prop = 0;
                ap_uint<DISP_DDR_BIT> cost_unary_orig = local_cost[y][x];
                ap_uint<19> cost_binary_orig = 0;
                disp_range d_orig = disp_reg[y][1];
                ap_uint<8> p_gray = gray_reg[y][1];
                /* unary cost */
                // if (d_prop< DISP_MIN) cost_unary_prop = local_unary_cost[y][x_base][x_ofst][0].range(6-1,0) << 1;
                // else if (d_prop.range(16,8) >= DISP_MAX - 1) cost_unary_prop = local_unary_cost[y][x_base][x_ofst][DISP_DDR_MAX-1].range(107,101);
                if (d_prop< DISP_MIN) cost_unary_prop = local_unary_cost[y][x_base][x_ofst][0].range(DISP_DDR_BIT-1,0);
                else if (d_prop.range(16,8) >= DISP_MAX - 1) cost_unary_prop = local_unary_cost[y][x_base][x_ofst][DISP_DDR_MAX-1].range(DISP_DDR_BIT*DISP_DDR_NUM-1,DISP_DDR_BIT*DISP_DDR_NUM-DISP_DDR_BIT);
                else {
                    ap_uint<8> d_l_int = d_prop.range(15,8);
                    if (d_prop[7]==1)
                        d_l_int = d_l_int + 1;
                    unary_cost_t c_l_int = local_unary_cost[y][x_base][x_ofst][d_l_int.range(7, DISP_DDR_NUM_LOG2)];
                    // ap_uint<8>  c_l_5_int =  bit_select_108b(d_l_int.range(DISP_DDR_NUM_LOG2-1, 0), c_l_512b_int);
#if DISP_DDR_BIT == 6
                    ap_uint<DISP_DDR_BIT>  c_l_5_int =  bit_select_96b(d_l_int.range(DISP_DDR_NUM_LOG2-1, 0), c_l_int);
#elif DISP_DDR_BIT == 8
                    ap_uint<DISP_DDR_BIT>  c_l_5_int =  bit_select_128b(d_l_int.range(DISP_DDR_NUM_LOG2-1, 0), c_l_int);
#elif DISP_DDR_BIT == 16
                    ap_uint<DISP_DDR_BIT>  c_l_5_int =  bit_select_256b(d_l_int.range(DISP_DDR_NUM_LOG2-1, 0), c_l_int);
#endif
                    cost_unary_prop = c_l_5_int;
                }
                /* binary cost */
                ap_uint<15> cost_binary_prop_array[9];
#pragma HLS array_partition variable=cost_binary_prop_array type=complete dim=1
                ap_uint<15> cost_binary_orig_array[9];
#pragma HLS array_partition variable=cost_binary_orig_array type=complete dim=1
                LOOP_BINARY_CALC:
                for (int i = 0; i < 3; i++) {
#pragma HLS unroll
                    for (int j = 0; j < 3; j++) {
#pragma HLS unroll
                        if (((i == 1) && (j == 1)) || ((x + j - 1) < 0) || ((x + j - 1) >= expn_region_w) || ((y + i - 1) < 0) || ((y + i - 1) >= expn_region_h)){
                            cost_binary_prop_array[i * 3 + j] = 0;
                            cost_binary_orig_array[i * 3 + j] = 0;
                        }
                        else {
                            assert((y + i - 1) < LAYER0_SIZE * 3);
                            disp_range d_nb = disp_reg[y + i - 1][j];
                            ap_uint<8> p_gray_nb = gray_reg[y + i - 1][j];
                            bool flag = ((p_gray_nb - p_gray) < 16) && ((p_gray_nb - p_gray) > -16);
                            disp_range d_prop_diff = (d_prop > d_nb) ? (d_prop - d_nb) : (d_nb - d_prop);
                            disp_range d_orig_diff = (d_orig > d_nb) ? (d_orig - d_nb) : (d_nb - d_orig);
                            cost_binary_prop_array[i * 3 + j] = (flag == true) ? d_prop_diff.range(16, 0) : d_prop_diff.range(16, 2);
                            cost_binary_orig_array[i * 3 + j] = (flag == true) ? d_orig_diff.range(16, 0) : d_orig_diff.range(16, 2);
                        }
                    }
                }
                for (int i = 0; i < 3; i++) {
#pragma HLS unroll
                    for (int j = 0; j < 3; j++) {
#pragma HLS unroll
                        cost_binary_prop += cost_binary_prop_array[i * 3 + j];
                        cost_binary_orig += cost_binary_orig_array[i * 3 + j];
                    }
                }
                if(y<18){
                //if (cost_unary_prop < cost_unary_orig && ((x < expn_region_w) && (y < expn_region_h))) {
                if ((cost_unary_prop + cost_binary_prop) < (cost_unary_orig + cost_binary_orig)) {
                    local_cost[y][x] = cost_unary_prop;
                    local_label[y][x][0] = plane_a[p]; local_label[y][x][1] = plane_b[p]; local_label[y][x][2] = plane_c[p];
                    local_disp[y][x] = d_prop;
                }
                }
            }  
        }
    // }
}

// 16bit frac
void layer0CostLabelUpdateOpt(ap_uint<3> unary_cost_x0, ap_uint<3> unary_cost_x1, ap_uint<3> unary_cost_x2, ap_uint<11> expn_region_x, ap_uint<11> expn_region_y, ap_uint<5> expn_region_w, ap_uint<5> expn_region_h, ap_uint<11> unit_region_x,
    ap_int<17> plane_a[LAYER0_PROP_NUM], ap_int<17> plane_b[LAYER0_PROP_NUM], ap_int<24> plane_c[LAYER0_PROP_NUM], 
    ap_uint<512> local_unary_cost[LAYER0_SIZE * 3][4][LAYER0_SIZE][3], // 8bit * 64
    ap_uint<8> local_cost[LAYER0_SIZE * 3][LAYER0_SIZE * 3], ap_int<24> local_label[LAYER0_SIZE * 3][LAYER0_SIZE * 3][3], ap_uint<24> local_disp[LAYER0_SIZE * 3][LAYER0_SIZE * 3]) {
#pragma HLS array_partition variable=plane_a type=complete dim=1
#pragma HLS array_partition variable=plane_b type=complete dim=1
#pragma HLS array_partition variable=plane_c type=complete dim=1
#pragma HLS array_partition variable=local_unary_cost type=complete dim=1
#pragma HLS array_partition variable=local_unary_cost type=complete dim=2
#pragma HLS array_partition variable=local_cost type=complete dim=1
#pragma HLS array_partition variable=local_label type=complete dim=1
#pragma HLS array_partition variable=local_label type=complete dim=3
#pragma HLS array_partition variable=local_disp type=complete dim=1
#pragma HLS inline off
    assert((unary_cost_x0 != unary_cost_x1) && (unary_cost_x0 != unary_cost_x2) && (unary_cost_x1 != unary_cost_x2));
    assert(unary_cost_x0 >= 0 && unary_cost_x0 < 4 && unary_cost_x1 >= 0 && unary_cost_x1 < 4 && unary_cost_x2 >= 0 && unary_cost_x2 < 4);
    bool is_left_region = (expn_region_x == unit_region_x);
    // 
    for (ap_uint<8> px = 0; px < LAYER0_PROP_NUM * LAYER0_SIZE * 3; px++) {
#pragma HLS pipeline II=1
        ap_uint<4> p = px / (3 * LAYER0_SIZE); // (4, 0)
        ap_uint<5> x = px % (3 * LAYER0_SIZE); // (5, 0)
        ap_uint<4> x_ofst = x % LAYER0_SIZE; // (4, 0)
        ap_uint<3> x_base; // (3, 0)
        if (is_left_region) {
            if (x < LAYER0_SIZE)
                x_base = unary_cost_x1;
            else if (x < LAYER0_SIZE * 2)
                x_base = unary_cost_x2;
        }
        else {
            if (x < LAYER0_SIZE)
                x_base = unary_cost_x0;
            else if (x < LAYER0_SIZE * 2)
                x_base = unary_cost_x1;
            else
                x_base = unary_cost_x2;
        }
        ap_int<28> d_prop_base = plane_a[p] * (expn_region_x + x) + plane_c[p]; // (12, 16)
        for (ap_uint<5> y = 0; y < LAYER0_SIZE * 3; y++) {
#pragma HLS unroll
            ap_int<28> d_prop = d_prop_base + plane_b[p] * (expn_region_y + y); // (12, 16)
            ap_uint<8> cost_unary_prop; // (0, 16)
            ap_uint<8> cost_unary_orig = local_cost[y][x]; // (0, 16)
            /* unary cost */  
            if (d_prop < 0) cost_unary_prop = local_unary_cost[y][x_base][x_ofst][0].range(7, 0);
            else if (d_prop.range(26, 16) >= DISP_MAX - 1) cost_unary_prop = local_unary_cost[y][x_base][x_ofst][2].range(511, 504); // 228 % 64 = 36; 36*8 = 288
            else {
                // int d_l = d_prop.to_int();
                ap_uint<11> d_l = d_prop.range(26, 16); // (11, 0)
                // ap_uint<11> d_h = d_prop.range(26, 16) + 1; // (11, 0)
                // ap_uint<16> f_h = d_prop.range(15, 0); // (0, 16)
                // ap_uint<16> f_l = FIXED_ONE - f_h; // (0, 16)
                // cost_unary_prop = (f_l * local_unary_cost[y][x_base][x_ofst][(d_l>>6)]/* .range((d_l.range(5, 0)<<4) + 15, (d_l.range(5, 0)<<4)) */ + 
                //                    f_h * local_unary_cost[y][x_base][x_ofst][(d_h>>6)]/* .range((d_h.range(5, 0)<<4) + 15, (d_h.range(5, 0)<<4)) */) >> 16;
                // ap_uint<512> c_l_0 = (d_l[5] == 0) ? local_unary_cost[y][x_base][x_ofst][(d_l>>6)].range(511, 0) : local_unary_cost[y][x_base][x_ofst][(d_l>>6)].range(1023, 512);
                // ap_uint<256> c_l_1 = (d_l[4] == 0) ? c_l_0.range(255, 0) : c_l_0.range(511, 256);
                // ap_uint<128> c_l_2 = (d_l[3] == 0) ? c_l_1.range(127, 0) : c_l_1.range(255, 128);
                // ap_uint<64>  c_l_3 = (d_l[2] == 0) ? c_l_2.range(63, 0)  : c_l_2.range(127, 64);
                // ap_uint<32>  c_l_4 = (d_l[1] == 0) ? c_l_3.range(31, 0)  : c_l_3.range(63, 32);
                // ap_uint<16>  c_l_5 = (d_l[0] == 0) ? c_l_4.range(15, 0)  : c_l_4.range(31, 16);
                ap_uint<512> c_l_512b = local_unary_cost[y][x_base][x_ofst][d_l.range(7, 6)];
                // ap_uint<512> c_l_512b = load_512b(y, x_base, x_ofst, d_l.range(7, 6), local_unary_cost);
                ap_uint<8>  c_l_5 =  bit_select_512b_16b(d_l.range(5, 0), c_l_512b);
                // cost_unary_prop = local_unary_cost[y][x_base][x_ofst][(d_l>>6)].range((d_l.range(5, 0)<<4) + 15, (d_l.range(5, 0)<<4));
                cost_unary_prop = c_l_5;
            }
            /* update local buffer */
            if (cost_unary_prop < cost_unary_orig && ((x < expn_region_w) && (y < expn_region_h))) {
                local_cost[y][x] = cost_unary_prop;
                local_label[y][x][0] = plane_a[p]; local_label[y][x][1] = plane_b[p]; local_label[y][x][2] = plane_c[p];
                local_disp[y][x] = d_prop;
            }
        }
    }
}

void layer1CostLabelUpdate(ap_uint<3> unary_cost_x0, ap_uint<3> unary_cost_x1, ap_uint<3> unary_cost_x2, 
    ap_uint<11> expn_region_x, ap_uint<11> expn_region_y, ap_uint<6> expn_region_w, ap_uint<6> expn_region_h, ap_uint<11> unit_region_x,
    label_range plane_a[LAYER1_PROP_NUM], label_range plane_b[LAYER1_PROP_NUM], label_range plane_c[LAYER1_PROP_NUM], 
    unary_cost_t local_unary_cost[LAYER1_SIZE * 3][4][LAYER1_SIZE][DISP_DDR_MAX], uchar local_im0_gray[LAYER1_SIZE * 3][LAYER1_SIZE * 3], 
    ap_uint<DISP_DDR_BIT> local_cost[LAYER1_SIZE * 3][LAYER1_SIZE * 3], label_range local_label[LAYER1_SIZE * 3][LAYER1_SIZE * 3][3], disp_range local_disp[LAYER1_SIZE * 3][LAYER1_SIZE * 3]) {
#pragma HLS inline off
#pragma HLS array_partition variable=local_unary_cost type=complete dim=1
#pragma HLS array_partition variable=local_unary_cost type=complete dim=2
//DO_PRAGMA(HLS array_reshape variable=local_unary_cost type=cyclic factor=DISP_DDR_NUM dim=4) 
#pragma HLS array_partition variable=local_im0_gray type=complete dim=1
#pragma HLS array_partition variable=local_cost type=complete dim=1
#pragma HLS array_partition variable=local_label type=complete dim=1
#pragma HLS array_partition variable=local_label type=complete dim=3
#pragma HLS array_partition variable=local_disp type=complete dim=1
#pragma HLS array_partition variable=plane_a type=complete dim=1
#pragma HLS array_partition variable=plane_b type=complete dim=1
#pragma HLS array_partition variable=plane_c type=complete dim=1
    assert((unary_cost_x0 != unary_cost_x1) && (unary_cost_x0 != unary_cost_x2) && (unary_cost_x1 != unary_cost_x2));
    assert(unary_cost_x0 >= 0 && unary_cost_x0 < 4 && unary_cost_x1 >= 0 && unary_cost_x1 < 4 && unary_cost_x2 >= 0 && unary_cost_x2 < 4);
    assert(expn_region_h <= (LAYER1_SIZE * 3));
    assert(expn_region_w <= (LAYER1_SIZE * 3));
    //
    bool is_left_region = (expn_region_x == unit_region_x);
    //
    ap_uint<8> gray_reg[LAYER1_SIZE * 3][3];
#pragma HLS array_partition variable=gray_reg type=complete dim=0
	disp_range disp_reg[LAYER1_SIZE * 3][3];
#pragma HLS array_partition variable=disp_reg type=complete dim=0	
    /* load to last colume of data registers*/
    LOOP_2ND_CACHE0:
    for (int i = 0; i < LAYER1_SIZE * 3; i++) {
#pragma HLS unroll
        gray_reg[i][2] = local_im0_gray[i][0];
        disp_reg[i][2] = local_disp[i][0];
    }
    // 
//     for (int p = 0; p < LAYER1_PROP_NUM; p++) {
//         for (int x = 0; x < LAYER1_SIZE * 3; x++) {
// #pragma HLS loop_flatten
// #pragma HLS pipeline II=1
    for (int px = 0; px < LAYER1_PROP_NUM * LAYER1_SIZE * 3; px++) {
#pragma HLS pipeline II=2
            int p = px / (3 * LAYER1_SIZE);
            int x = px % (3 * LAYER1_SIZE);
            ap_uint<5> x_ofst;
            if (x >= LAYER1_SIZE * 2)
                x_ofst = x - LAYER1_SIZE * 2;
            else if (x >= LAYER1_SIZE)
                x_ofst = x - LAYER1_SIZE;
            else
                x_ofst = x;
            ap_uint<3> x_base = 0;
            if (is_left_region) {
                if (x < LAYER1_SIZE)
                    x_base = unary_cost_x1;
                else if (x < LAYER1_SIZE * 2)
                    x_base = unary_cost_x2;
            }
            else {
                if (x < LAYER1_SIZE)
                    x_base = unary_cost_x0;
                else if (x < LAYER1_SIZE * 2)
                    x_base = unary_cost_x1;
                else
                    x_base = unary_cost_x2;
            }
            /* update data registers for binary cost calculation */
            /* shift first 2 columns */
            LOOP_2ND_CACHE1:
            for (int j = 0; j < 2; j++) {
#pragma HLS unroll
                for (int i = 0; i < LAYER1_SIZE * 3; i++) {
#pragma HLS unroll
                    gray_reg[i][j] = gray_reg[i][j + 1];
                    disp_reg[i][j] = disp_reg[i][j + 1];
                }
            }
            /* load to last column */
            if (x < (expn_region_w - 1)) {
                LOOP_2ND_CACHE2:
                for (int i = 0; i < LAYER1_SIZE * 3; i++) {
#pragma HLS unroll
                    gray_reg[i][2] = local_im0_gray[i][x + 1];
                    disp_reg[i][2] = local_disp[i][x + 1];
                }
            }
            ap_int<17> d_prop_x = plane_a[p] * (expn_region_x + x) + plane_c[p];
            LOOP_CALC_VER:
            for (int y = 0; y < LAYER1_SIZE * 3; y++) {
#pragma HLS unroll
// #pragma HLS dependence variable=local_cost type=inter false
// #pragma HLS dependence variable=local_label type=inter false
// #pragma HLS pipeline II=1
                //
                disp_range d_prop = d_prop_x + plane_b[p] * (expn_region_y + y);
                ap_uint<DISP_DDR_BIT> cost_unary_prop = 0;
                ap_uint<19> cost_binary_prop = 0;
                ap_uint<DISP_DDR_BIT> cost_unary_orig = local_cost[y][x];
                ap_uint<19> cost_binary_orig = 0;
                disp_range d_orig = disp_reg[y][1];
                ap_uint<8> p_gray = gray_reg[y][1];
                /* unary cost */
                // if (d_prop< DISP_MIN) cost_unary_prop = local_unary_cost[y][x_base][x_ofst][0].range(6-1,0) << 1;
                // else if (d_prop.range(16,8) >= DISP_MAX - 1) cost_unary_prop = local_unary_cost[y][x_base][x_ofst][DISP_DDR_MAX-1].range(107,101);
                if (d_prop< DISP_MIN) cost_unary_prop = local_unary_cost[y][x_base][x_ofst][0].range(DISP_DDR_BIT-1,0);
                else if (d_prop.range(16,8) >= DISP_MAX - 1) cost_unary_prop = local_unary_cost[y][x_base][x_ofst][DISP_DDR_MAX-1].range(DISP_DDR_BIT*DISP_DDR_NUM-1,DISP_DDR_BIT*DISP_DDR_NUM-DISP_DDR_BIT);
                else {
                    ap_uint<8> d_l_int = d_prop.range(15,8);
                    if (d_prop[7]==1)
                        d_l_int = d_l_int + 1;
                    unary_cost_t c_l_int = local_unary_cost[y][x_base][x_ofst][d_l_int.range(7, DISP_DDR_NUM_LOG2)];
                    // ap_uint<8>  c_l_5_int =  bit_select_108b(d_l_int.range(DISP_DDR_NUM_LOG2-1, 0), c_l_512b_int);
#if DISP_DDR_BIT == 6
                    ap_uint<DISP_DDR_BIT>  c_l_5_int =  bit_select_96b(d_l_int.range(DISP_DDR_NUM_LOG2-1, 0), c_l_int);
#elif DISP_DDR_BIT == 8
                    ap_uint<DISP_DDR_BIT>  c_l_5_int =  bit_select_128b(d_l_int.range(DISP_DDR_NUM_LOG2-1, 0), c_l_int);
#elif DISP_DDR_BIT == 16
                    ap_uint<DISP_DDR_BIT>  c_l_5_int =  bit_select_256b(d_l_int.range(DISP_DDR_NUM_LOG2-1, 0), c_l_int);
#endif
                    cost_unary_prop = c_l_5_int;
                }
                /* binary cost */
                ap_uint<15> cost_binary_prop_array[9];
#pragma HLS array_partition variable=cost_binary_prop_array type=complete dim=1
                ap_uint<15> cost_binary_orig_array[9];
#pragma HLS array_partition variable=cost_binary_orig_array type=complete dim=1
                LOOP_BINARY_CALC:
                for (int i = 0; i < 3; i++) {
#pragma HLS unroll
                    for (int j = 0; j < 3; j++) {
#pragma HLS unroll
                        if (((i == 1) && (j == 1)) || ((x + j - 1) < 0) || ((x + j - 1) >= expn_region_w) || ((y + i - 1) < 0) || ((y + i - 1) >= expn_region_h)){
                            cost_binary_prop_array[i * 3 + j] = 0;
                            cost_binary_orig_array[i * 3 + j] = 0;
                        }
                        else {
                            assert((y + i - 1) < LAYER1_SIZE * 3);
                            disp_range d_nb = disp_reg[y + i - 1][j];
                            ap_uint<8> p_gray_nb = gray_reg[y + i - 1][j];
                            bool flag = ((p_gray_nb - p_gray) < 16) && ((p_gray_nb - p_gray) > -16);
                            disp_range d_prop_diff = (d_prop > d_nb) ? (d_prop - d_nb) : (d_nb - d_prop);
                            disp_range d_orig_diff = (d_orig > d_nb) ? (d_orig - d_nb) : (d_nb - d_orig);
                            cost_binary_prop_array[i * 3 + j] = (flag == true) ? d_prop_diff.range(16, 0) : d_prop_diff.range(16, 2);
                            cost_binary_orig_array[i * 3 + j] = (flag == true) ? d_orig_diff.range(16, 0) : d_orig_diff.range(16, 2);
                        }
                    }
                }
                for (int i = 0; i < 3; i++) {
#pragma HLS unroll
                    for (int j = 0; j < 3; j++) {
#pragma HLS unroll
                        cost_binary_prop += cost_binary_prop_array[i * 3 + j];
                        cost_binary_orig += cost_binary_orig_array[i * 3 + j];
                    }
                }
                if ((cost_unary_prop + cost_binary_prop) < (cost_unary_orig + cost_binary_orig)) {
                    local_cost[y][x] = cost_unary_prop;
                    local_label[y][x][0] = plane_a[p]; local_label[y][x][1] = plane_b[p]; local_label[y][x][2] = plane_c[p];
                    local_disp[y][x] = d_prop;
                }
            }  
        }
    // }
}

// 16bit frac
void layer1CostLabelUpdateOpt(ap_uint<3> unary_cost_x0, ap_uint<3> unary_cost_x1, ap_uint<3> unary_cost_x2, 
    ap_uint<11> expn_region_x, ap_uint<11> expn_region_y, ap_uint<6> expn_region_w, ap_uint<6> expn_region_h, ap_uint<11> unit_region_x,
    ap_int<17> plane_a[LAYER1_PROP_NUM], ap_int<17> plane_b[LAYER1_PROP_NUM], ap_int<17> plane_c[LAYER1_PROP_NUM], 
    ap_uint<512> local_unary_cost[LAYER1_SIZE * 3][4][LAYER1_SIZE][3], ap_uint<8> local_im0_gray[LAYER1_SIZE * 3][LAYER1_SIZE * 3], 
    ap_uint<8> local_cost[LAYER1_SIZE * 3][LAYER1_SIZE * 3], ap_int<24> local_label[LAYER1_SIZE * 3][LAYER1_SIZE * 3][3], ap_uint<24> local_disp[LAYER1_SIZE * 3][LAYER1_SIZE * 3]) {
#pragma HLS inline off
#pragma HLS array_partition variable=local_unary_cost type=complete dim=1
#pragma HLS array_partition variable=local_unary_cost type=complete dim=2
#pragma HLS array_partition variable=local_im0_gray type=complete dim=1
#pragma HLS array_partition variable=local_cost type=complete dim=1
#pragma HLS array_partition variable=local_label type=complete dim=1
#pragma HLS array_partition variable=local_label type=complete dim=3
#pragma HLS array_partition variable=local_disp type=complete dim=1
#pragma HLS array_partition variable=plane_a type=complete dim=1
#pragma HLS array_partition variable=plane_b type=complete dim=1
#pragma HLS array_partition variable=plane_c type=complete dim=1
    assert((unary_cost_x0 != unary_cost_x1) && (unary_cost_x0 != unary_cost_x2) && (unary_cost_x1 != unary_cost_x2));
    assert(unary_cost_x0 >= 0 && unary_cost_x0 < 4 && unary_cost_x1 >= 0 && unary_cost_x1 < 4 && unary_cost_x2 >= 0 && unary_cost_x2 < 4);
    //
    bool is_left_region = (expn_region_x == unit_region_x);
    // 
    ap_uint<8> gray_reg[LAYER1_SIZE * 3][3];
#pragma HLS array_partition variable=gray_reg type=complete dim=0
	ap_uint<24> disp_reg[LAYER1_SIZE * 3][3];
#pragma HLS array_partition variable=disp_reg type=complete dim=0	
    /* load to last colume of data registers*/
    LOOP_2ND_CACHE0:
    for (ap_uint<6>  i = 0; i < LAYER1_SIZE * 3; i++) {
#pragma HLS unroll
        gray_reg[i][2] = local_im0_gray[i][0];
        disp_reg[i][2] = local_disp[i][0];
    }
    // 
    for (ap_uint<4> p = 0; p < LAYER1_PROP_NUM; p++) {
        for (ap_uint<6> x = 0; x < LAYER1_SIZE * 3; x++) {
#pragma HLS loop_flatten
#pragma HLS pipeline II=1
//     for (int px = 0; px < LAYER1_PROP_NUM * LAYER1_SIZE * 3; px++) {
// // #pragma HLS pipeline II=1
//             int p = px / (3 * LAYER1_SIZE);
//             int x = px % (3 * LAYER1_SIZE);
            ap_uint<5> x_ofst;
            if (x >= LAYER1_SIZE * 2)
                x_ofst = x - LAYER1_SIZE * 2;
            else if (x >= LAYER1_SIZE)
                x_ofst = x - LAYER1_SIZE;
            else
                x_ofst = x;
            ap_uint<3> x_base = 0;
            if (is_left_region) {
                if (x < LAYER1_SIZE)
                    x_base = unary_cost_x1;
                else if (x < LAYER1_SIZE * 2)
                    x_base = unary_cost_x2;
            }
            else {
                if (x < LAYER1_SIZE)
                    x_base = unary_cost_x0;
                else if (x < LAYER1_SIZE * 2)
                    x_base = unary_cost_x1;
                else
                    x_base = unary_cost_x2;
            }
            /* update data registers for binary cost calculation */
            /* shift first 2 columns */
            LOOP_2ND_CACHE1:
            for (ap_uint<2> j = 0; j < 2; j++) {
#pragma HLS unroll
                for (ap_uint<6> i = 0; i < LAYER1_SIZE * 3; i++) {
#pragma HLS unroll
                    gray_reg[i][j] = gray_reg[i][j + 1];
                    disp_reg[i][j] = disp_reg[i][j + 1];
                }
            }
            /* load to last column */
            if (x < (expn_region_w - 1)) {
                LOOP_2ND_CACHE2:
                for (ap_uint<6> i = 0; i < LAYER1_SIZE * 3; i++) {
#pragma HLS unroll
                    gray_reg[i][2] = local_im0_gray[i][x + 1];
                    disp_reg[i][2] = local_disp[i][x + 1];
                }
            }
            ap_int<28> d_prop_base = plane_a[p] * (expn_region_x + x) + plane_c[p];
            LOOP_CALC_VER:
            for (ap_uint<6> y = 0; y < LAYER1_SIZE * 3; y++) {
#pragma HLS unroll
// #pragma HLS dependence variable=local_cost type=inter false
// #pragma HLS dependence variable=local_label type=inter false
// #pragma HLS pipeline II=1
                //
                ap_int<28> d_prop = d_prop_base + plane_b[p] * (expn_region_y + y);
                ap_uint<8> cost_unary_prop = 0;
                ap_uint<8> cost_binary_prop = 0;
                ap_uint<8> cost_unary_orig = local_cost[y][x];
                ap_uint<8> cost_binary_orig = 0;
                ap_uint<24> d_orig = disp_reg[y][1];
                ap_uint<8> p_gray = gray_reg[y][1];
                /* unary cost */
                if (d_prop < DISP_MIN) cost_unary_prop = local_unary_cost[y][x_base][x_ofst][0].range(7, 0);
                else if (d_prop.range(26, 16) >= DISP_MAX - 1) cost_unary_prop = local_unary_cost[y][x_base][x_ofst][2].range(511, 496);
                else {
                    ap_uint<11> d_l = d_prop.range(26, 16);
                    ap_uint<512> c_l_512b = local_unary_cost[y][x_base][x_ofst][d_l.range(7, 6)];
                    // ap_uint<512> c_l_512b = load_512b(y, x_base, x_ofst, d_l.range(7, 6), local_unary_cost);
                    ap_uint<8>  c_l_5 =  bit_select_512b_16b(d_l.range(5, 0), c_l_512b);
                    // cost_unary_prop = local_unary_cost[y][x_base][x_ofst][(d_l>>6)].range((d_l.range(5, 0)<<4) + 15, (d_l.range(5, 0)<<4));
                    cost_unary_prop = c_l_5;
                }
                /* binary cost */
                LOOP_BINARY_CALC:
                for (ap_uint<3> i = 0; i < 3; i++) {
#pragma HLS unroll
                    for (ap_uint<3> j = 0; j < 3; j++) {
#pragma HLS unroll
                        if (((i == 1) && (j == 1)) || ((x + j - 1) < 0) || ((x + j - 1) >= expn_region_w) || ((y + i - 1) < 0) || ((y + i - 1) >= expn_region_h) || ((y + i - 1) >= LAYER1_SIZE * 3))
                            continue;
                        ap_uint<24> d_nb = disp_reg[y + i - 1][j];
                        ap_uint<8> p_gray_nb = gray_reg[y + i - 1][j];
                        ap_uint<8> weight = (std::abs(p_gray_nb - p_gray) < 10) ? 0.32 : 0.08;
                        cost_binary_prop += (weight * std::abs(d_prop - d_nb) * DISP_DDR_VAL_MAX) >> 8;
                        cost_binary_orig += (weight * std::abs(d_orig - d_nb) * DISP_DDR_VAL_MAX) >> 8;
                    }
                }
                if ((cost_unary_prop + cost_binary_prop) < (cost_unary_orig + cost_binary_orig) && ((x < expn_region_w) && (y < expn_region_h))) {
                    local_cost[y][x] = cost_unary_prop;
                    local_label[y][x][0] = plane_a[p]; local_label[y][x][1] = plane_b[p]; local_label[y][x][2] = plane_c[p];
                    local_disp[y][x] = d_prop;
                }
            }
        }
    }
}

struct ABC {
    ap_fixed<32, 17, AP_RND, AP_SAT> a, b, c;
};

// 计算3x3矩阵的行列式
fixed_min determinant3x3(ap_uint11 a11, ap_uint10 a12, ap_uint8 a13,
                        ap_uint11 a21, ap_uint10 a22, ap_uint8 a23,
                        ap_uint11 a31, ap_uint10 a32, ap_uint8 a33) {
    return a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31) + a13 * (a21 * a32 - a22 * a31);
}

// 计算a, b, c的函数
ABC calculateABC(ap_uint11 x0, ap_uint10 y0, ap_uint8 d0,
                 ap_uint11 x1, ap_uint10 y1, ap_uint8 d1,
                 ap_uint11 x2, ap_uint10 y2, ap_uint8 d2
                 ) {
//#pragma HLS inline off
    // 计算系数矩阵A的行列式D
    fixed_min D = determinant3x3(x0, y0, 1, x1, y1, 1, x2, y2, 1);

    
    if (D == 0) {
        D = 1;
    }

    // 计算Da, Db, Dc，并行处理
    fixed_min Da, Db, Dc;
    Da = determinant3x3(d0, y0, 1, d1, y1, 1, d2, y2, 1);
    Db = determinant3x3(x0, d0, 1, x1, d1, 1, x2, d2, 1);
    Dc = determinant3x3(x0, y0, d0, x1, y1, d1, x2, y2, d2);
    ap_fixed<32, 17, AP_RND, AP_SAT> a, b, c;          
    a = Da / D;
    b = Db / D;
    c = Dc / D;    
    ABC outcome;
    outcome.a=a;
    outcome.b=b;
    outcome.c=c;
    return outcome; 
}

void choose_further(ap_uint<8> best_byte[DISP_DDR_MAX], ap_uint<8> best_index[DISP_DDR_MAX], ap_uint<8>& index){
ap_uint<8> min_byte = 255;  // 假设字节的最大值为255
    ap_uint<8> min_index = 0;

    for(int j=0; j<DISP_DDR_MAX; j++){
        #pragma HLS unroll
        if(best_byte[j] < min_byte){
            min_byte = best_byte[j];
            min_index = best_index[j];
        }
    }
    index = min_index;
/*   
//#pragma HLS inline off
//ap_uint<8> cost_one[6];
ap_uint<8> cost_min0[6][2];
ap_uint<8> cost_min1[3][2];
ap_uint<8> byte;
//ap_uint<8> cost_min3[2];
//#pragma HLS array_partition variable=cost_one type=complete dim=0
#pragma HLS array_partition variable=cost_min0 type=complete dim=0
#pragma HLS array_partition variable=cost_min1 type=complete dim=0
//#pragma HLS array_partition variable=cost_min2 type=complete dim=0
//#pragma HLS array_partition variable=cost_min3 type=complete dim=0

for(int j=0; j<6; j++){
    #pragma HLS unroll
    cost_min0[j][0] = (best_byte[j]<best_byte[j+6])? best_byte[j]:best_byte[j+6];
    cost_min0[j][1] = (best_byte[j]<best_byte[j+6])? best_index[j]:best_index[j+6];
}

for(int j=0; j<3; j++){
    #pragma HLS unroll
    cost_min1[j][0] = (cost_min0[j][0]<cost_min0[j+3][0])? cost_min0[j][0]:cost_min0[j+3][0];
    cost_min1[j][1] = (cost_min0[j][0]<cost_min0[j+3][0])? cost_min0[j][1]:cost_min0[j+3][1];
}

    byte = (cost_min1[0][0]<cost_min1[1][0])? cost_min1[0][0]:cost_min1[1][0];
    index = (cost_min1[0][0]<cost_min1[1][0])? cost_min1[0][1]:cost_min1[1][1];
    index = (byte<cost_min1[2][0])? index:cost_min1[2][1];
    */ 
}

void choose_min_cost(int base, ap_uint<128> cost_all, ap_uint<8>& byte, ap_uint<8>& index){
//#pragma HLS inline off
ap_uint<8> cost_one[16];
ap_uint<8> cost_min0[8][2];
ap_uint<8> cost_min1[4][2];
ap_uint<8> cost_min2[2][2];
//ap_uint<8> cost_min3[2];
#pragma HLS array_partition variable=cost_one type=complete dim=0
#pragma HLS array_partition variable=cost_min0 type=complete dim=0
#pragma HLS array_partition variable=cost_min1 type=complete dim=0
#pragma HLS array_partition variable=cost_min2 type=complete dim=0
//#pragma HLS array_partition variable=cost_min3 type=complete dim=0

for(int i=0; i<16; i++){
    #pragma HLS unroll
    cost_one[i] = cost_all(8*i, 8*i + 7);
}
for(int j=0; j<8; j++){
    #pragma HLS unroll

    cost_min0[j][0] = (cost_one[j]<cost_one[j+8])? cost_one[j]:cost_one[j+8];
    cost_min0[j][1] = (cost_one[j]<cost_one[j+8])? j:j+8;
}
for(int j=0; j<4; j++){
    #pragma HLS unroll

    cost_min1[j][0] = (cost_min0[j][0]<cost_min0[j+4][0])? cost_min0[j][0]:cost_min0[j+4][0];
    cost_min1[j][1] = (cost_min0[j][0]<cost_min0[j+4][0])? cost_min0[j][1]:cost_min0[j+4][1];
}
for(int j=0; j<2; j++){
    #pragma HLS unroll

    cost_min2[j][0] = (cost_min1[j][0]<cost_min1[j+2][0])? cost_min1[j][0]:cost_min1[j+2][0];
    cost_min2[j][1] = (cost_min1[j][0]<cost_min1[j+2][0])? cost_min1[j][1]:cost_min1[j+2][1];
}
byte = (cost_min2[0][0]<cost_min2[1][0])? cost_min2[0][0]:cost_min2[1][0];
index = (cost_min2[0][0]<cost_min2[1][0])? base + cost_min2[0][1]: base + cost_min2[1][1];
}


void new_abc_hls(ap_uint<4> p_xx[LAYER0_PROP_NUM], ap_uint<4> p_yy[LAYER0_PROP_NUM], int unary_cost_x1, int unary_cost_x2,
ap_uint<12> expn_region_x, ap_uint<12> expn_region_y, ap_uint<12> unit_region_x, ap_uint<12> unit_region_y, ap_uint<7> unit_region_w, ap_uint<7> unit_region_h, 
unary_cost_t local_unary_cost[LAYER0_SIZE * 3][4][LAYER0_SIZE][DISP_DDR_MAX], label_range p_a[LAYER0_PROP_NUM], label_range p_b[LAYER0_PROP_NUM], label_range p_c[LAYER0_PROP_NUM]){
#pragma HLS inline off
//#pragma HLS array_partition variable=local_unary_cost type=complete dim=1
//#pragma HLS array_partition variable=local_unary_cost type=complete dim=2
#pragma HLS array_partition variable=p_a type=complete dim=1
//#pragma HLS array_partition variable=p_a type=complete dim=2
#pragma HLS array_partition variable=p_b type=complete dim=1
//#pragma HLS array_partition variable=p_b type=complete dim=2
#pragma HLS array_partition variable=p_c type=complete dim=1
//#pragma HLS array_partition variable=p_c type=complete dim=2
ap_uint<8> best_byte0[LAYER0_PROP_NUM][DISP_DDR_MAX]; ap_uint<8> best_index0[LAYER0_PROP_NUM][DISP_DDR_MAX];
ap_uint<8> best_byte1[LAYER0_PROP_NUM][DISP_DDR_MAX]; ap_uint<8> best_index1[LAYER0_PROP_NUM][DISP_DDR_MAX];
ap_uint<8> best_byte2[LAYER0_PROP_NUM][DISP_DDR_MAX]; ap_uint<8> best_index2[LAYER0_PROP_NUM][DISP_DDR_MAX];
#pragma HLS array_partition variable=best_byte0 type=complete dim=0
#pragma HLS array_partition variable=best_byte1 type=complete dim=0
#pragma HLS array_partition variable=best_byte2 type=complete dim=0
#pragma HLS array_partition variable=best_index0 type=complete dim=0
#pragma HLS array_partition variable=best_index1 type=complete dim=0
#pragma HLS array_partition variable=best_index2 type=complete dim=0
unary_cost_t bank0[DISP_DDR_MAX*DISP_DDR_NUM];
unary_cost_t bank1[DISP_DDR_MAX*DISP_DDR_NUM];
unary_cost_t bank2[DISP_DDR_MAX*DISP_DDR_NUM];
#pragma HLS array_partition variable=bank0 type=complete dim=0
#pragma HLS array_partition variable=bank1 type=complete dim=0
#pragma HLS array_partition variable=bank2 type=complete dim=0
ap_uint<8> d0, d1, d2;
ap_uint<4> p_x[LAYER0_PROP_NUM][3];
ap_uint<4> p_y[LAYER0_PROP_NUM][3]; 
#pragma HLS array_partition variable=p_x type=complete dim=0
#pragma HLS array_partition variable=p_y type=complete dim=0
ap_uint<4> x_base = unary_cost_x1;
ap_uint<4> y_base = (expn_region_y == unit_region_y)? 0 : 9;
ap_uint<4> para1 = 2;
ap_uint<4> para2 = 6;
for (int i = 0; i < LAYER0_PROP_NUM; i++) {
#pragma HLS unroll
 p_x[i][0] = p_xx[i];//随机选三个像素找最小cost
 p_y[i][0] = p_yy[i];
 //p_x[i][1] = randMwc(Q[i], I[i], C[i]) % unit_region_w;
 //p_y[i][1] = randMwc(Q[i], I[i], C[i]) % unit_region_h; 
 //p_x[i][2] = randMwc(Q[i], I[i], C[i]) % unit_region_w;
 //p_y[i][2] = randMwc(Q[i], I[i], C[i]) % unit_region_h;  
 ap_uint<4> para11 = p_y[i][0]-1;
 ap_uint<4> para22 = p_y[i][0]+1;
 ap_uint<4> para3 = 7;
 ap_uint<4> para33 = p_x[i][0] + 1;
 ap_uint<4> para44 = p_x[i][0] + 2;
 p_x[i][1] = p_y[i][0];
 // p_x[i][1] = ((p_x[i][0]==8)? para3: para33);
 p_y[i][1] = ((p_y[i][0]==0)? para1: para11);
 p_x[i][2] = p_y[i][1];
 // p_x[i][2] = ((p_x[i][0]>=7)? p_x[i][1]: para44);
 p_y[i][2] = ((p_y[i][0]==8)? para2: para22);
}
for (int k = 0; k < LAYER0_PROP_NUM * DISP_DDR_MAX; k++) {
//for (int i = 0; i < LAYER0_PROP_NUM; i++) {
 //for (int j = 0; j < DISP_DDR_MAX; j++) { 
 #pragma HLS pipeline II=1
    int i = k / DISP_DDR_MAX;
    int j = k % DISP_DDR_MAX;
    int base = j << 4;
    bank0[j] = local_unary_cost[y_base + p_y[i][0]][x_base][p_x[i][0]][j];
    choose_min_cost(base, bank0[j], best_byte0[i][j], best_index0[i][j]);
    bank1[j] = local_unary_cost[y_base + p_y[i][1]][x_base][p_x[i][1]][j];
    choose_min_cost(base, bank1[j], best_byte1[i][j], best_index1[i][j]);
    bank2[j] = local_unary_cost[y_base + p_y[i][2]][x_base][p_x[i][2]][j];
    choose_min_cost(base, bank2[j], best_byte2[i][j], best_index2[i][j]);
 //}
}

for (int i = 0; i < LAYER0_PROP_NUM; i++) {
#pragma HLS unroll
choose_further(best_byte0[i], best_index0[i], d0);
choose_further(best_byte1[i], best_index1[i], d1);
choose_further(best_byte2[i], best_index2[i], d2);

ABC label = calculateABC(unit_region_x+p_x[i][0], unit_region_y+p_y[i][0], d0, 
                         unit_region_x+p_x[i][1], unit_region_y+p_y[i][1], d1,
                         unit_region_x+p_x[i][2], unit_region_y+p_y[i][2], d2);
    p_a[i]=((label.a) << 8).to_ap_int();
    p_b[i]=((label.b) << 8).to_ap_int();
    p_c[i]=((label.c) << 8).to_ap_int();
  }
/*
for (int i = 0; i < LAYER0_PROP_NUM; i++) {
   p_a[index][i]=864;
    p_b[index][i]=723;
    p_c[index][i]=1024;
}
*/
}

void new_abc(ap_uint<32> Q[LAYER0_PROP_NUM][4096], ap_uint<32> I[LAYER0_PROP_NUM], ap_uint<32> C[LAYER0_PROP_NUM], int unary_cost_x1, int unary_cost_x2,
ap_uint<12> expn_region_x, ap_uint<12> expn_region_y, ap_uint<12> unit_region_x, ap_uint<12> unit_region_y, ap_uint<7> unit_region_w, ap_uint<7> unit_region_h, 
label_range local_label[LAYER0_SIZE * 3][LAYER0_SIZE * 3][3], 
unary_cost_t local_unary_cost[LAYER0_SIZE * 3][4][LAYER0_SIZE][DISP_DDR_MAX], label_range p_a[LAYER0_PROP_NUM], label_range p_b[LAYER0_PROP_NUM], label_range p_c[LAYER0_PROP_NUM]){
#pragma HLS inline off
ap_uint<8> best_byte[12][12];
ap_uint<8> bank0[DISP_DDR_MAX*DISP_DDR_NUM];
ap_uint<8> bank1[DISP_DDR_MAX*DISP_DDR_NUM];
ap_uint<8> bank2[DISP_DDR_MAX*DISP_DDR_NUM];
ap_uint<12> p_x[LAYER0_PROP_NUM][3];
ap_uint<12> p_y[LAYER0_PROP_NUM][3]; 
int x_base = unary_cost_x1;
int y_base = (expn_region_y == unit_region_y)? 0 : 9;
ap_uint<4> para1 = 2;
ap_uint<4> para2 = 6;
for (int i = 0; i < LAYER0_PROP_NUM; i++) {
#pragma HLS unroll
 p_x[i][0] = randMwc(Q[i], I[i], C[i]) % unit_region_w;//随机选三个像素找最小cost
 p_y[i][0] = randMwc(Q[i], I[i], C[i]) % unit_region_h;
 //p_x[i][1] = randMwc(Q[i], I[i], C[i]) % unit_region_w;
 //p_y[i][1] = randMwc(Q[i], I[i], C[i]) % unit_region_h; 
 //p_x[i][2] = randMwc(Q[i], I[i], C[i]) % unit_region_w;
 //p_y[i][2] = randMwc(Q[i], I[i], C[i]) % unit_region_h;  
 ap_uint<4> para11 = p_y[i][0]-1;
 ap_uint<4> para22 = p_y[i][0]+1;
 p_x[i][1] = p_y[i][0];
 p_y[i][1] = ((p_y[i][0]==0)? para1: para11);
 p_x[i][2] = p_y[i][1];
 p_y[i][2] = ((p_y[i][0]==8)? para2: para22);
 for (int j = 0; j < DISP_DDR_MAX; j++) { 
    for (int d = 0; d < DISP_DDR_NUM; d++) {
    bank0[j*16+d] = local_unary_cost[y_base + p_y[i][0]][x_base][p_x[i][0]][j](d * DISP_DDR_BIT + DISP_DDR_BIT -1, d * DISP_DDR_BIT);
    bank1[j*16+d] = local_unary_cost[y_base + p_y[i][1]][x_base][p_x[i][1]][j](d * DISP_DDR_BIT + DISP_DDR_BIT -1, d * DISP_DDR_BIT);
    bank2[j*16+d] = local_unary_cost[y_base + p_y[i][2]][x_base][p_x[i][2]][j](d * DISP_DDR_BIT + DISP_DDR_BIT -1, d * DISP_DDR_BIT);
    }
}

ap_uint<8> min_index = 0;
ap_uint<8> min_value = 255; // 使用INT_MAX作为初始最小值

    // 遍历best_byte数组，找到最小值和对应的索引
    for (int k = 0; k < DISP_MAX; ++k) {      
            if ((bank0[k] <= min_value)) {
                min_value = bank0[k];
                min_index = k; // 存储最小值对应的第一维索引
                float c0 = min_value;
                float d0 = min_index;
            }       
    }

ap_uint<8> min_index1 = 0;
ap_uint<8> min_value1 = 255; // 使用INT_MAX作为初始最小值

    // 遍历best_byte数组，找到最小值和对应的索引
    for (int k = 0; k < DISP_MAX; ++k) {      
            if ((bank1[k] <= min_value1)) {
                min_value1 = bank1[k];
                min_index1 = k; // 存储最小值对应的第一维索引
                float c1 = min_value1;
                float d1 = min_index1;
            }       
    }

ap_uint<8> min_index2 = 0;
ap_uint<8> min_value2 = 255; // 使用INT_MAX作为初始最小值

    // 遍历best_byte数组，找到最小值和对应的索引
    for (int k = 0; k < DISP_MAX; ++k) {      
            if ((bank2[k] <= min_value2)) {
                min_value2 = bank2[k];
                min_index2 = k; // 存储最小值对应的第一维索引
                float c2 = min_value2;
                float d2 = min_index2;
            }       
    }
ABC label = calculateABC(unit_region_x+p_x[i][0], unit_region_y+p_y[i][0], min_index, 
                         unit_region_x+p_x[i][1], unit_region_y+p_y[i][1], min_index1,
                         unit_region_x+p_x[i][2], unit_region_y+p_y[i][2], min_index2);
    p_a[i]=((label.a)*256).to_ap_int();
    p_b[i]=((label.b)*256).to_ap_int();
    p_c[i]=((label.c)*256).to_ap_int();
    float aaa = p_a[i];
    float bbb = p_b[i];
    float ccc = p_c[i];
    float ddd = aaa + bbb;
}
}

void random_xy(int iter, int inner_iter, ap_uint<32> Q[LAYER0_PROP_NUM][4096], ap_uint<32> I[LAYER0_PROP_NUM], ap_uint<32> C[LAYER0_PROP_NUM], ap_uint<4> p_x[LAYER0_PROP_NUM], ap_uint<4> p_y[LAYER0_PROP_NUM], ap_uint<32> pp[LAYER0_PROP_NUM][3], ap_uint<4> p_xx[LAYER0_PROP_NUM], ap_uint<4> p_yy[LAYER0_PROP_NUM]){
#pragma HLS array_partition variable=p_x type=complete dim=0
#pragma HLS array_partition variable=p_y type=complete dim=0
#pragma HLS array_partition variable=pp type=complete dim=0
#pragma HLS array_partition variable=p_xx type=complete dim=0
#pragma HLS array_partition variable=p_yy type=complete dim=0 
#pragma HLS array_partition variable=Q type=complete dim=1
#pragma HLS array_partition variable=I type=complete dim=0
#pragma HLS array_partition variable=C type=complete dim=0   
    int ver_region_i = iter / LAYER0_HOR_REGION_NUM;
    int hor_region_i = iter % LAYER0_HOR_REGION_NUM;
    int unit_region_x, unit_region_y, unit_region_w, unit_region_h;
    int expn_region_x, expn_region_y, expn_region_w, expn_region_h;
    genRegionInfo(WIDTH, HEIGHT, hor_region_i, ver_region_i, LAYER0_SIZE, unit_region_x, unit_region_y, unit_region_w, unit_region_h, 
        expn_region_x, expn_region_y, expn_region_w, expn_region_h);
    for (int i = 0; i < LAYER0_PROP_NUM; i++) {
    #pragma HLS unroll
    if((unit_region_w==0)||(unit_region_h==0)){
        printf("iter = %d, inner_iter = %d, w = %d, h = %d\n", iter,inner_iter,unit_region_w, unit_region_h);
    }
    p_x[i] = randMwc(Q[i], I[i], C[i]) % unit_region_w;//随机选三个像素找最小cost
    p_y[i] = randMwc(Q[i], I[i], C[i]) % unit_region_h;
    p_xx[i] = randMwc(Q[i], I[i], C[i]) % unit_region_w;//随机选三个像素找最小cost
    p_yy[i] = randMwc(Q[i], I[i], C[i]) % unit_region_h;
    }
    for (int i = 0; i < LAYER0_PROP_NUM * 3; i++) {
    #pragma HLS unroll
    int j = i / 3;
    int k = i % 3;
    pp[j][k] = randMwc(Q[k], I[k], C[k]);//随机选三个像素找最小cost 
    }
    /*
    for (int i = 0; i < LAYER0_PROP_NUM; i++) {
    #pragma HLS unroll
    p_xx[i] = randMwc(Q[i], I[i], C[i]) % unit_region_w;//随机选三个像素找最小cost
    p_yy[i] = randMwc(Q[i], I[i], C[i]) % unit_region_h;
    }
    */
}

void layer0Calc(int iter, ap_uint<4> p_x[LAYER0_PROP_NUM], ap_uint<4> p_y[LAYER0_PROP_NUM], ap_uint<32> pp[LAYER0_PROP_NUM][3], ap_uint<4> p_xx[LAYER0_PROP_NUM], ap_uint<4> p_yy[LAYER0_PROP_NUM], ap_uint<32> Q[LAYER0_PROP_NUM][4096], ap_uint<32> I[LAYER0_PROP_NUM], ap_uint<32> C[LAYER0_PROP_NUM], 
    int unary_cost_x0, int unary_cost_x1, int unary_cost_x2,  unary_cost_t local_unary_cost[LAYER0_SIZE * 3][4][LAYER0_SIZE][DISP_DDR_MAX], 
    ap_uint<DISP_DDR_BIT> local_cost[LAYER0_SIZE * 3][LAYER0_SIZE * 3], label_range local_label[LAYER0_SIZE * 3][LAYER0_SIZE * 3][3], disp_range local_disp[LAYER0_SIZE * 3][LAYER0_SIZE * 3]) {
#pragma HLS inline off
// #pragma HLS array_partition variable=local_unary_cost type=complete dim=1
// #pragma HLS array_partition variable=local_unary_cost type=complete dim=2
// DO_PRAGMA(HLS array_reshape variable=local_unary_cost type=cyclic factor=DISP_DDR_NUM dim=4) 
// #pragma HLS INTERFACE mode=bram port=local_cost storage_impl=bram storage_type=ram_2p
#pragma HLS array_partition variable=local_cost type=complete dim=1
#pragma HLS array_partition variable=local_label type=complete dim=1
#pragma HLS array_partition variable=local_label type=complete dim=3
#pragma HLS array_partition variable=local_disp type=complete dim=1

    // float plane_a, plane_b, plane_c;
    label_range plane_a[LAYER0_PROP_NUM], plane_b[LAYER0_PROP_NUM], plane_c[LAYER0_PROP_NUM];
    label_range plane_a1[LAYER0_PROP_NUM], plane_b1[LAYER0_PROP_NUM], plane_c1[LAYER0_PROP_NUM];
#pragma HLS array_partition variable=plane_a type=complete dim=1
#pragma HLS array_partition variable=plane_b type=complete dim=1
#pragma HLS array_partition variable=plane_c type=complete dim=1
    //
    assert((iter >= 0) && (iter < LAYER0_VER_REGION_NUM * LAYER0_HOR_REGION_NUM));
    int ver_region_i = iter / LAYER0_HOR_REGION_NUM;
    int hor_region_i = iter % LAYER0_HOR_REGION_NUM;
    int unit_region_x, unit_region_y, unit_region_w, unit_region_h;
    int expn_region_x, expn_region_y, expn_region_w, expn_region_h;
    genRegionInfo(WIDTH, HEIGHT, hor_region_i, ver_region_i, LAYER0_SIZE, unit_region_x, unit_region_y, unit_region_w, unit_region_h, 
        expn_region_x, expn_region_y, expn_region_w, expn_region_h);
    /* process local ram */
#if LAYER0_ENABLE 
#if LAYER0_PROP_NUM==3 
    for (int inner_iter = 0; inner_iter < LAYER0_PROP_LOOP; inner_iter++) {
#pragma HLS pipeline off
//////////    random_xy(iter, Q, I, C, p_x, p_y, pp, p_xx, p_yy);
        
        layer0ProposalParall(iter, ver_region_i, p_x, p_y, pp, inner_iter, expn_region_x, expn_region_y, unit_region_x, unit_region_y, unit_region_w, unit_region_h,
           local_label, plane_a, plane_b, plane_c);
            
        new_abc_hls(p_xx, p_yy, unary_cost_x1, unary_cost_x2, expn_region_x, expn_region_y, unit_region_x, unit_region_y, unit_region_w, unit_region_h, local_unary_cost, plane_a1, plane_b1, plane_c1);
        // 调用 layer0updatewrapper 包含 layer0CostLabelUpdate 和 random_xy
        layer0CostLabelUpdate(unary_cost_x0, unary_cost_x1, unary_cost_x2, expn_region_x, expn_region_y, expn_region_w, expn_region_h, unit_region_x, 
            plane_a, plane_b, plane_c, plane_a1, plane_b1, plane_c1, local_unary_cost, local_cost, local_label, local_disp);  
        int iter0 = (inner_iter==0)? iter : ((iter==LAYER0_HOR_REGION_NUM*LAYER0_VER_REGION_NUM - 1) ? iter : (iter + 1));
        random_xy(iter0, inner_iter, Q, I, C, p_x, p_y, pp, p_xx, p_yy);
        

        }       
//    }
 #else    
for (int inner_iter = 0; inner_iter < LAYER0_PROP_LOOP; inner_iter++) {
#pragma HLS pipeline off
        layer0ProposalParall(ver_region_i, Q, I, C, inner_iter, expn_region_x, expn_region_y, unit_region_x, unit_region_y, unit_region_w, unit_region_h,
            local_label, plane_a, plane_b, plane_c);
        layer0CostLabelUpdate(unary_cost_x0, unary_cost_x1, unary_cost_x2, expn_region_x, expn_region_y, expn_region_w, expn_region_h, unit_region_x, 
            plane_a, plane_b, plane_c, local_unary_cost, local_cost, local_label, local_disp);
    }
#endif
#endif
}

void layer0OutloopInit(int hor_region_num, /* ap_uint<32>* q, ap_uint<32>& i, ap_uint<32>& c, 
    ap_uint<32>* q1, ap_uint<32>& i1, ap_uint<32>& c1, ap_uint<32>* q2, ap_uint<32>& i2, ap_uint<32>& c2, 
    ap_uint<32>* q3, ap_uint<32>& i3, ap_uint<32>& c3, ap_uint<32>* q4, ap_uint<32>& i4, ap_uint<32>& c4, 
    ap_uint<32>* q5, ap_uint<32>& i5, ap_uint<32>& c5, */ 
    ap_uint<DISP_DDR_BIT> uram_cost[LAYER0_SIZE * 3][WIDTH], label_range uram_label[LAYER0_SIZE * 3][WIDTH][3], disp_range uram_disp[LAYER0_SIZE * 3][WIDTH]) {
#pragma HLS inline off
// #pragma HLS array_partition variable=uram_cost type=complete dim=1
// #pragma HLS array_partition variable=uram_label type=complete dim=1
// #pragma HLS array_partition variable=uram_label type=complete dim=3
// #pragma HLS array_partition variable=uram_disp type=complete dim=1
// #pragma HLS interface mode=m_axi bundle=BUS_LAYER0 port=ddr_unary_cost offset=slave
    for (int iter = 0; iter < (LAYER0_HOR_REGION_NUM + 2); iter++) {
        int ver_region_i = iter / LAYER0_HOR_REGION_NUM;
        int hor_region_i = iter % LAYER0_HOR_REGION_NUM;
        // 
        int unit_region_x, unit_region_y, unit_region_w, unit_region_h;
        int expn_region_x, expn_region_y, expn_region_w, expn_region_h;
        genRegionInfo(WIDTH, HEIGHT, hor_region_i, ver_region_i, LAYER0_SIZE, unit_region_x, unit_region_y, unit_region_w, unit_region_h, expn_region_x, expn_region_y, expn_region_w, expn_region_h);
#if 0
        // 
        float plane_a, plane_b, plane_c;
        randomPlaneInit(q, i, c, q1, i1, c1, q2, i2, c2, q3, i3, c3, q4, i4, c4, q5, i5, c5, 
            unit_region_x, unit_region_y, unit_region_w, unit_region_h, plane_a, plane_b, plane_c);
        // 
        for (int x = 0; x < LAYER0_SIZE; x++) {
            for (int y = 0; y < LAYER0_SIZE; y++) {
#pragma HLS pipeline II=1
                if (((unit_region_x + x) >= WIDTH) || (unit_region_y + y) >= HEIGHT)
                    continue;
                float d = plane_a * (unit_region_x + x) + plane_b * (unit_region_y + y) + plane_c;
                float cost_tmp;
                if (d < DISP_MIN) {
                    cost_tmp = ddr_unary_cost[((unit_region_y + y) * WIDTH + (unit_region_x + x)) * DISP_DDR_MAX + 0](DISP_DDR_BIT -1, 0).to_int();
                }
                else if (d >= DISP_MAX - 1) {
                    int d_dec = (DISP_MAX - 1) % DISP_DDR_NUM;
                    cost_tmp = ddr_unary_cost[((unit_region_y + y) * WIDTH + (unit_region_x + x)) * DISP_DDR_MAX + DISP_DDR_MAX - 1](d_dec * DISP_DDR_BIT + DISP_DDR_BIT -1, d_dec * DISP_DDR_BIT).to_int();
                }
                else {
                    int d_l = std::floor(d);
                    int d_h = d_l + 1;
                    float f_h = d - d_l;
                    float f_l = 1.0 - f_h;
                    int d_l_int = d_l / DISP_DDR_NUM;
                    int d_l_dec = d_l % DISP_DDR_NUM;
                    int d_h_int = d_h / DISP_DDR_NUM;
                    int d_h_dec = d_h % DISP_DDR_NUM;
                    cost_tmp = f_l * ddr_unary_cost[((unit_region_y + y) * WIDTH + (unit_region_x + x)) * DISP_DDR_MAX + d_l_int](d_l_dec * DISP_DDR_BIT + DISP_DDR_BIT -1, d_l_dec * DISP_DDR_BIT).to_int() + 
                               f_h * ddr_unary_cost[((unit_region_y + y) * WIDTH + (unit_region_x + x)) * DISP_DDR_MAX + d_h_int](d_h_dec * DISP_DDR_BIT + DISP_DDR_BIT -1, d_h_dec * DISP_DDR_BIT).to_int();
                }
                uram_cost[((ver_region_i % 3) * LAYER0_SIZE + y)][(unit_region_x + x)] = cost_tmp;
                uram_label[((ver_region_i % 3) * LAYER0_SIZE + y)][(unit_region_x + x)][0] = plane_a;
                uram_label[((ver_region_i % 3) * LAYER0_SIZE + y)][(unit_region_x + x)][1] = plane_b;
                uram_label[((ver_region_i % 3) * LAYER0_SIZE + y)][(unit_region_x + x)][2] = plane_c;
                uram_disp[((ver_region_i % 3) * LAYER0_SIZE + y)][(unit_region_x + x)] = d;
            }
        }
#else
        for (int x = 0; x < LAYER0_SIZE; x++) {
#pragma HLS pipeline II=1
            for (int y = 0; y < LAYER0_SIZE; y++) {
#pragma HLS unroll
                if (((unit_region_x + x) >= WIDTH) || (unit_region_y + y) >= HEIGHT)
                    continue;
                uram_cost[((ver_region_i % 3) * LAYER0_SIZE + y)][(unit_region_x + x)] = ((1 << (DISP_DDR_BIT)) - 1);
                uram_label[((ver_region_i % 3) * LAYER0_SIZE + y)][(unit_region_x + x)][0] = 0;
                uram_label[((ver_region_i % 3) * LAYER0_SIZE + y)][(unit_region_x + x)][1] = 0;
                uram_label[((ver_region_i % 3) * LAYER0_SIZE + y)][(unit_region_x + x)][2] = 0;
                uram_disp[((ver_region_i % 3) * LAYER0_SIZE + y)][(unit_region_x + x)] = 0;
            }
        }
#endif
    }
}

void layer0InloopInit(int iter, int ver_region_num, int hor_region_num, int unary_cost_x, 
    ap_uint<32>* q, ap_uint<32>& i, ap_uint<32>& c, ap_uint<32>* q1, ap_uint<32>& i1, ap_uint<32>& c1, 
    ap_uint<32>* q2, ap_uint<32>& i2, ap_uint<32>& c2, ap_uint<32>* q3, ap_uint<32>& i3, ap_uint<32>& c3, 
    ap_uint<32>* q4, ap_uint<32>& i4, ap_uint<32>& c4, ap_uint<32>* q5, ap_uint<32>& i5, ap_uint<32>& c5, 
    float local_unary_cost[LAYER0_SIZE][LAYER0_SIZE][DISP_MAX],
    float local_init_cost[LAYER0_SIZE][LAYER0_SIZE], float local_init_label[LAYER0_SIZE][LAYER0_SIZE][3], float local_init_disp[LAYER0_SIZE][LAYER0_SIZE]) {
#pragma HLS inline off
// #pragma HLS array_partition variable=uram_cost type=complete dim=1
// #pragma HLS array_partition variable=uram_label type=complete dim=1
// #pragma HLS array_partition variable=uram_label type=complete dim=3
// #pragma HLS array_partition variable=local_unary_cost type=complete dim=1
    // if (iter < (ver_region_num * hor_region_num - hor_region_num - 1)) {
        //
        int init_ver_region_i = (iter + hor_region_num + 2) / hor_region_num;
        int init_hor_region_i = (iter + hor_region_num + 2) % hor_region_num;
        int calc_ver_region_i = iter / hor_region_num;
        int calc_hor_region_i = iter % hor_region_num;
        int init_unit_region_x, init_unit_region_y, init_unit_region_w, init_unit_region_h;
        int init_expn_region_x, init_expn_region_y, init_expn_region_w, init_expn_region_h;
        genRegionInfo(WIDTH, HEIGHT, init_hor_region_i, init_ver_region_i, LAYER0_SIZE, init_unit_region_x, init_unit_region_y, init_unit_region_w, init_unit_region_h, 
            init_expn_region_x, init_expn_region_y, init_expn_region_w, init_expn_region_h);
        int calc_unit_region_x, calc_unit_region_y, calc_unit_region_w, calc_unit_region_h;
        int calc_expn_region_x, calc_expn_region_y, calc_expn_region_w, calc_expn_region_h;
        genRegionInfo(WIDTH, HEIGHT, calc_hor_region_i, calc_ver_region_i, LAYER0_SIZE, calc_unit_region_x, calc_unit_region_y, calc_unit_region_w, calc_unit_region_h, 
            calc_expn_region_x, calc_expn_region_y, calc_expn_region_w, calc_expn_region_h);
        // 
        // int cach_x = (iter + 1) % 4;
        //
        float plane_a, plane_b, plane_c;
        randomPlaneInit(q, i, c, q1, i1, c1, q2, i2, c2, q3, i3, c3, q4, i4, c4, q5, i5, c5, 
            init_unit_region_x, init_unit_region_y, init_unit_region_w, init_unit_region_h, plane_a, plane_b, plane_c);
        // 
        for (int y = 0; y < LAYER0_SIZE; y++) {
            for (int x = 0; x < LAYER0_SIZE; x++) {
#pragma HLS pipeline II=1
                if (((init_unit_region_x + x) >= WIDTH) || (init_unit_region_y + y) >= HEIGHT)
                    continue;
                float d = plane_a * (init_unit_region_x + x) + plane_b * (init_unit_region_y + y) + plane_c;
                float cost_tmp;
                if (init_hor_region_i != 0) {
                    if (d < DISP_MIN) cost_tmp = local_unary_cost[y][x][0];
                    else if (d >= DISP_MAX - 1) cost_tmp = local_unary_cost[y][x][DISP_MAX - 1];
                    else {
                        int d_l = std::floor(d);
                        int d_h = d_l + 1;
                        float f_h = d - d_l;
                        float f_l = 1.0 - f_h;
                        cost_tmp = f_l * local_unary_cost[y][x][d_l] + 
                                   f_h * local_unary_cost[y][x][d_h];
                    }
                }
                else {
                    if (d < DISP_MIN) cost_tmp = local_unary_cost[y][x][0];
                    else if (d >= DISP_MAX - 1) cost_tmp = local_unary_cost[y][x][DISP_MAX - 1];
                    else {
                        int d_l = std::floor(d);
                        int d_h = d_l + 1;
                        float f_h = d - d_l;
                        float f_l = 1.0 - f_h;
                        cost_tmp = f_l * local_unary_cost[y][x][d_l] + 
                                   f_h * local_unary_cost[y][x][d_h];
                    }
                }
                local_init_cost[y][x] = cost_tmp;
                local_init_label[y][x][0] = plane_a;
                local_init_label[y][x][1] = plane_b;
                local_init_label[y][x][2] = plane_c;
                local_init_disp[y][x] = d;
            }
        }
    // }
}

void layer0InloopInit(int iter, int ver_region_num, int hor_region_num, int buff_id, 
    /* ap_uint<32>* q, ap_uint<32>& i, ap_uint<32>& c, ap_uint<32>* q1, ap_uint<32>& i1, ap_uint<32>& c1, 
    ap_uint<32>* q2, ap_uint<32>& i2, ap_uint<32>& c2, ap_uint<32>* q3, ap_uint<32>& i3, ap_uint<32>& c3, 
    ap_uint<32>* q4, ap_uint<32>& i4, ap_uint<32>& c4, ap_uint<32>* q5, ap_uint<32>& i5, ap_uint<32>& c5,  */
    ap_uint<DISP_DDR_BIT> local_init_cost[LAYER0_SIZE][LAYER0_SIZE], label_range local_init_label[LAYER0_SIZE][LAYER0_SIZE][3], disp_range local_init_disp[LAYER0_SIZE][LAYER0_SIZE]) {
#pragma HLS inline off
// #pragma HLS array_partition variable=local_init_cost type=complete dim=1
// #pragma HLS array_partition variable=local_init_label type=complete dim=1
// #pragma HLS array_partition variable=local_init_label type=complete dim=3
// #pragma HLS array_partition variable=local_init_disp type=complete dim=1
    // if (iter < (ver_region_num * hor_region_num - hor_region_num - 1)) {
        //
        int init_ver_region_i = (iter + hor_region_num + 2) / hor_region_num;
        int init_hor_region_i = (iter + hor_region_num + 2) % hor_region_num;
        int calc_ver_region_i = iter / hor_region_num;
        int calc_hor_region_i = iter % hor_region_num;
        int init_unit_region_x, init_unit_region_y, init_unit_region_w, init_unit_region_h;
        int init_expn_region_x, init_expn_region_y, init_expn_region_w, init_expn_region_h;
        genRegionInfo(WIDTH, HEIGHT, init_hor_region_i, init_ver_region_i, LAYER0_SIZE, init_unit_region_x, init_unit_region_y, init_unit_region_w, init_unit_region_h, 
            init_expn_region_x, init_expn_region_y, init_expn_region_w, init_expn_region_h);
        int calc_unit_region_x, calc_unit_region_y, calc_unit_region_w, calc_unit_region_h;
        int calc_expn_region_x, calc_expn_region_y, calc_expn_region_w, calc_expn_region_h;
        genRegionInfo(WIDTH, HEIGHT, calc_hor_region_i, calc_ver_region_i, LAYER0_SIZE, calc_unit_region_x, calc_unit_region_y, calc_unit_region_w, calc_unit_region_h, 
            calc_expn_region_x, calc_expn_region_y, calc_expn_region_w, calc_expn_region_h);
        // 
        // int cach_x = (iter + 1) % 4;
        //
#if 0
        float plane_a, plane_b, plane_c;
        randomPlaneInit(q, i, c, q1, i1, c1, q2, i2, c2, q3, i3, c3, q4, i4, c4, q5, i5, c5, 
            init_unit_region_x, init_unit_region_y, init_unit_region_w, init_unit_region_h, plane_a, plane_b, plane_c);
        // 
        for (int x = 0; x < LAYER0_SIZE; x++) { 
#pragma HLS pipeline II=1
            for (int y = 0; y < LAYER0_SIZE; y++) {
#pragma HLS unroll
                if (((init_unit_region_x + x) >= WIDTH) || (init_unit_region_y + y) >= HEIGHT)
                    continue;
                float d = plane_a * (init_unit_region_x + x) + plane_b * (init_unit_region_y + y) + plane_c;
                float cost_tmp;
                if (init_hor_region_i != 0) {
                    if (d < DISP_MIN) cost_tmp = local_unary_cost[buff_id][y][x][0];
                    else if (d >= DISP_MAX - 1) cost_tmp = local_unary_cost[buff_id][y][x][DISP_MAX - 1];
                    else {
                        int d_l = std::floor(d);
                        int d_h = d_l + 1;
                        float f_h = d - d_l;
                        float f_l = 1.0 - f_h;
                        cost_tmp = f_l * local_unary_cost[buff_id][y][x][d_l] + 
                                   f_h * local_unary_cost[buff_id][y][x][d_h];
                    }
                }
                else {
                    if (d < DISP_MIN) cost_tmp = local_unary_cost[buff_id][y][x][0];
                    else if (d >= DISP_MAX - 1) cost_tmp = local_unary_cost[buff_id][y][x][DISP_MAX - 1];
                    else {
                        int d_l = std::floor(d);
                        int d_h = d_l + 1;
                        float f_h = d - d_l;
                        float f_l = 1.0 - f_h;
                        cost_tmp = f_l * local_unary_cost[buff_id][y][x][d_l] + 
                                   f_h * local_unary_cost[buff_id][y][x][d_h];
                    }
                }
                local_init_cost[y][x] = cost_tmp;
                local_init_label[y][x][0] = plane_a;
                local_init_label[y][x][1] = plane_b;
                local_init_label[y][x][2] = plane_c;
                local_init_disp[y][x] = d;
            }
        }
#else
    for (int x = 0; x < LAYER0_SIZE; x++) { 
#pragma HLS pipeline II=1
        for (int y = 0; y < LAYER0_SIZE; y++) {
#pragma HLS unroll
            if (((init_unit_region_x + x) >= WIDTH) || (init_unit_region_y + y) >= HEIGHT)
                continue;
            local_init_cost[y][x] = DISP_DDR_VAL_MAX;
            local_init_label[y][x][0] = 0;
            local_init_label[y][x][1] = 0;
            local_init_label[y][x][2] = 0;
            local_init_disp[y][x] = 0;
        }
    }
#endif
    // }
}

void layer0InloopCache(unary_cost_t local_buffer[2][LAYER0_SIZE][DISP_DDR_MAX0], int iter, int ver_region_num, int hor_region_num, int unary_cost_x, 
    unary_cost_t local_unary_cost[LAYER0_SIZE * 3][4][LAYER0_SIZE][DISP_DDR_MAX], unary_cost_t0 ddr_unary_cost[WIDTH * HEIGHT * DISP_DDR_MAX0]
    ) {
#pragma HLS inline
#pragma HLS array_partition variable=local_unary_cost type=complete dim=1
//#pragma HLS array_partition variable=local_unary_cost type=complete dim=2
// DO_PRAGMA(HLS array_reshape variable=local_unary_cost type=cyclic factor=DISP_DDR_NUM dim=4)
// #pragma HLS interface mode=m_axi bundle=BUS_LAYER0 port=ddr_unary_cost offset=slave
    //
    assert(iter >= 0 && iter < (LAYER0_VER_REGION_NUM * LAYER0_HOR_REGION_NUM - 2));
    assert(unary_cost_x >= 0 && unary_cost_x < 4);
    int ver_region_i = (iter + 2) / LAYER0_HOR_REGION_NUM;
    int hor_region_i = (iter + 2) % LAYER0_HOR_REGION_NUM;
    // 
    int unit_region_x, unit_region_y, unit_region_w, unit_region_h;
    int expn_region_x, expn_region_y, expn_region_w, expn_region_h;
    genRegionInfo(WIDTH, HEIGHT, hor_region_i, ver_region_i, LAYER0_SIZE, unit_region_x, unit_region_y, unit_region_w, unit_region_h, 
        expn_region_x, expn_region_y, expn_region_w, expn_region_h);
    // int x = 0, y = 0;
    //
     for (int xyd = 0; xyd <  LAYER0_SIZE * DISP_DDR_MAX0; xyd++) {
#pragma HLS pipeline II=1
        int y = 0;
        int x = xyd % (LAYER0_SIZE * DISP_DDR_MAX0) / DISP_DDR_MAX0;
        int d = xyd % (LAYER0_SIZE * DISP_DDR_MAX0) % DISP_DDR_MAX0;
//        if ((x >= unit_region_w) || (expn_region_y + y >= HEIGHT))
//                continue;
        bool index = y % 2;
        int dd = d << 1;
        local_unary_cost[y][unary_cost_x][x][dd] = ddr_unary_cost[((expn_region_y + y) * WIDTH + (unit_region_x + x)) * DISP_DDR_MAX0 + d](127,0); 
        local_buffer[index][x][d] = ddr_unary_cost[((expn_region_y + y) * WIDTH + (unit_region_x + x)) * DISP_DDR_MAX0 + d](255,128);
    }
    LOOP_COST1:
    for (int xyd = 0; xyd < LAYER0_SIZE * DISP_DDR_MAX0 * (LAYER0_SIZE * 3 - 1); xyd++) {
#pragma HLS pipeline II=1
        int y = 1 + xyd / (LAYER0_SIZE * DISP_DDR_MAX0);
        int x = xyd % (LAYER0_SIZE * DISP_DDR_MAX0) / DISP_DDR_MAX0;
        int d = xyd % (LAYER0_SIZE * DISP_DDR_MAX0) % DISP_DDR_MAX0;
        bool index = y % 2;
        int dd = d << 1;
//         if ((x >= unit_region_w) || (expn_region_y + y >= HEIGHT))
//                continue;
        local_unary_cost[y][unary_cost_x][x][dd] = ddr_unary_cost[((expn_region_y + y) * WIDTH + (unit_region_x + x)) * DISP_DDR_MAX0 + d](127,0); 
        local_buffer[index][x][d] = ddr_unary_cost[((expn_region_y + y) * WIDTH + (unit_region_x + x)) * DISP_DDR_MAX0 + d](255,128);
        if (dd >= DISP_DDR_MAX - 1)
            continue; 
        local_unary_cost[y - 1][unary_cost_x][x][dd + 1] = local_buffer[!index][x][d];
    }
    LOOP_COST2:
    for (int xyd = 0; xyd <  LAYER0_SIZE * DISP_DDR_MAX0; xyd++) {
#pragma HLS pipeline II=1
        int y = 27;
        int x = xyd % (LAYER0_SIZE * DISP_DDR_MAX0) / DISP_DDR_MAX0;
        int d = xyd % (LAYER0_SIZE * DISP_DDR_MAX0) % DISP_DDR_MAX0;
        bool index = y % 2;
        int dd = d << 1;
//        if ((x >= unit_region_w) || (expn_region_y + y >= HEIGHT) || (dd >= DISP_DDR_MAX - 1))
//                continue;
        if (dd >= DISP_DDR_MAX - 1)
            continue;
        local_unary_cost[y - 1][unary_cost_x][x][dd + 1] = local_buffer[!index][x][d];
    }
    /*
    LOOP_IMAGE: 
    for (int xy = 0; xy < LAYER0_SIZE * LAYER0_SIZE * 3; xy++) {
#pragma HLS pipeline II=1
        int x = xy / (LAYER0_SIZE * 3);
        int y = xy % (LAYER0_SIZE * 3);
        if ((x >= unit_region_w) || (expn_region_y + y >= HEIGHT))
                continue;
        local_im0_gray[y][unary_cost_x][x] = ddr_im0_gray[(expn_region_y + y) * WIDTH + (unit_region_x + x)]; 
    }
    
    for (int xyd = 0; xyd < LAYER0_SIZE * LAYER0_SIZE * 2 * DISP_DDR_MAX; xyd++) {
#pragma HLS pipeline II=1
        int y = xyd / (LAYER0_SIZE * DISP_DDR_MAX);
        int x = xyd % (LAYER0_SIZE * DISP_DDR_MAX) / DISP_DDR_MAX;
        int d = xyd % (LAYER0_SIZE * DISP_DDR_MAX) % DISP_DDR_MAX;
        if ((x >= unit_region_w) || (expn_region_y + y >= HEIGHT))
                continue;
        local_unary_cost[y][unary_cost_x][x][d] = ddr_unary_cost[((expn_region_y + y) * WIDTH + (unit_region_x + x)) * DISP_DDR_MAX + d]; 
    }
    */ 
}

void layer0InloopCache(int iter, int ver_region_num, int hor_region_num, int unary_cost_x, 
    unary_cost_t local_unary_cost[LAYER0_SIZE * 3][4][LAYER0_SIZE][DISP_DDR_MAX], unary_cost_t ddr_unary_cost[WIDTH * HEIGHT * DISP_DDR_MAX],
    uchar local_im0_gray[LAYER0_SIZE * 3][4][LAYER0_SIZE], uchar ddr_im0_gray[WIDTH * HEIGHT]) {
#pragma HLS inline
#pragma HLS array_partition variable=local_unary_cost type=complete dim=1
#pragma HLS array_partition variable=local_unary_cost type=complete dim=2
// DO_PRAGMA(HLS array_reshape variable=local_unary_cost type=cyclic factor=DISP_DDR_NUM dim=4)
// #pragma HLS interface mode=m_axi bundle=BUS_LAYER0 port=ddr_unary_cost offset=slave
#pragma HLS array_partition variable=local_im0_gray type=complete dim=1
#pragma HLS array_partition variable=local_im0_gray type=complete dim=2
    //
    assert(iter >= 0 && iter < (LAYER0_VER_REGION_NUM * LAYER0_HOR_REGION_NUM - 2));
    assert(unary_cost_x >= 0 && unary_cost_x < 4);
    int ver_region_i = (iter + 2) / LAYER0_HOR_REGION_NUM;
    int hor_region_i = (iter + 2) % LAYER0_HOR_REGION_NUM;
    // 
    int unit_region_x, unit_region_y, unit_region_w, unit_region_h;
    int expn_region_x, expn_region_y, expn_region_w, expn_region_h;
    genRegionInfo(WIDTH, HEIGHT, hor_region_i, ver_region_i, LAYER0_SIZE, unit_region_x, unit_region_y, unit_region_w, unit_region_h, 
        expn_region_x, expn_region_y, expn_region_w, expn_region_h);
    // int x = 0, y = 0;
    // 
    for (int xyd = 0; xyd < LAYER0_SIZE * LAYER0_SIZE * 3 * DISP_DDR_MAX; xyd++) {
#pragma HLS pipeline II=1
        int x = xyd / (LAYER0_SIZE * 3 * DISP_DDR_MAX);
        int y = xyd % (LAYER0_SIZE * 3 * DISP_DDR_MAX) / DISP_DDR_MAX;
        int d = xyd % (LAYER0_SIZE * 3 * DISP_DDR_MAX) % DISP_DDR_MAX;
        local_unary_cost[y][unary_cost_x][x][d] = ddr_unary_cost[((expn_region_y + y) * WIDTH + (unit_region_x + x)) * DISP_DDR_MAX + d]; 
    }

    LOOP_IMAGE: 
    for (int xy = 0; xy < LAYER0_SIZE * LAYER0_SIZE * 3; xy++) {
#pragma HLS pipeline II=1
        int x = xy / (LAYER0_SIZE * 3);
        int y = xy % (LAYER0_SIZE * 3);
        local_im0_gray[y][unary_cost_x][x] = ddr_im0_gray[(expn_region_y + y) * WIDTH + (unit_region_x + x)]; 
    }
}

void layer0InloopCacheInit(int iter, int ver_region_num, int hor_region_num, int buff_id, 
    float local_unary_cost_init[2][LAYER0_SIZE][LAYER0_SIZE][DISP_DDR_MAX * DISP_DDR_NUM], unary_cost_t ddr_unary_cost[WIDTH * HEIGHT * DISP_DDR_MAX]) {
#pragma HLS inline
// #pragma HLS array_partition variable=local_unary_cost_init type=complete dim=1
// #pragma HLS array_partition variable=local_unary_cost_init type=complete dim=2
// DO_PRAGMA(HLS array_reshape variable=local_unary_cost_init type=cyclic factor=DISP_DDR_NUM dim=4)
// #pragma HLS interface mode=m_axi bundle=BUS_LAYER0 port=ddr_unary_cost offset=slave
    int init_ver_region_i = (iter + LAYER0_HOR_REGION_NUM + 3) / LAYER0_HOR_REGION_NUM;
    int init_hor_region_i = (iter + LAYER0_HOR_REGION_NUM + 3) % LAYER0_HOR_REGION_NUM;
    int init_unit_region_x, init_unit_region_y, init_unit_region_w, init_unit_region_h;
    int init_expn_region_x, init_expn_region_y, init_expn_region_w, init_expn_region_h;
    genRegionInfo(WIDTH, HEIGHT, init_hor_region_i, init_ver_region_i, LAYER0_SIZE, init_unit_region_x, init_unit_region_y, init_unit_region_w, init_unit_region_h, 
        init_expn_region_x, init_expn_region_y, init_expn_region_w, init_expn_region_h);
    float local_unary_cost_tmp[DISP_DDR_MAX * DISP_DDR_NUM];
DO_PRAGMA(HLS array_reshape variable=local_unary_cost_tmp type=cyclic factor=DISP_DDR_NUM dim=1)    
    LOOP0:
    for (int xyd = 0; xyd < LAYER0_SIZE * LAYER0_SIZE * DISP_DDR_MAX; xyd++) {
#pragma HLS pipeline II=1
		        // unary_cost_t unary_cost = ddr_unary_cost[((expn_region_y + y) * WIDTH + (unit_region_x + x)) * DISP_DDR_MAX + d_ddr];
                int y = (xyd / DISP_DDR_MAX) % LAYER0_SIZE;
                int x = (xyd / DISP_DDR_MAX) / LAYER0_SIZE;
                int d_base = (xyd % DISP_DDR_MAX);
                unary_cost_t unary_cost = ddr_unary_cost[((init_unit_region_y + y) * WIDTH + (init_unit_region_x + x)) * DISP_DDR_MAX + d_base];  
                LOOP1:
                for (int d = 0; d < DISP_DDR_NUM; d++) {
#pragma HLS unroll
                    local_unary_cost_tmp[d_base * DISP_DDR_NUM + d] = unary_cost(d * DISP_DDR_BIT + DISP_DDR_BIT -1, d * DISP_DDR_BIT); 
                }
                LOOP2:
                for (int d = 0; d < DISP_DDR_NUM; d++) {
#pragma HLS unroll
                    local_unary_cost_init[buff_id][y][x][d_base * DISP_DDR_NUM + d] = local_unary_cost_tmp[d_base * DISP_DDR_NUM + d];   
                }      
    }
}

void layer0OutloopCache(unary_cost_t local_unary_cost[LAYER0_SIZE * 3][4][LAYER0_SIZE][DISP_DDR_MAX],unary_cost_t ddr_unary_cost[WIDTH * HEIGHT * DISP_DDR_MAX],
     uchar local_im0_gray[LAYER0_SIZE * 3][4][LAYER0_SIZE], uchar ddr_im0_gray[WIDTH * HEIGHT]) {
#pragma HLS inline off
// #pragma HLS array_partition variable=local_unary_cost type=complete dim=1
// #pragma HLS array_partition variable=local_unary_cost type=complete dim=2
// DO_PRAGMA(HLS array_reshape variable=local_unary_cost type=cyclic factor=DISP_DDR_NUM dim=4)
// #pragma HLS array_partition variable=local_unary_cost_init type=complete dim=1
// #pragma HLS array_partition variable=local_unary_cost_init type=complete dim=2
// DO_PRAGMA(HLS array_reshape variable=local_unary_cost_init type=cyclic factor=DISP_DDR_NUM dim=4)   
// #pragma HLS interface mode=m_axi bundle=BUS_LAYER0 port=ddr_unary_cost offset=slave
    LOOP_COST0:
    for (int xyd = 0; xyd < LAYER0_SIZE * LAYER0_SIZE * DISP_DDR_MAX; xyd++) {
#pragma HLS pipeline II=1
        int y = xyd / (LAYER0_SIZE * DISP_DDR_MAX);
        int x = xyd % (LAYER0_SIZE * DISP_DDR_MAX) / DISP_DDR_MAX;
        int d = xyd % (LAYER0_SIZE * DISP_DDR_MAX) % DISP_DDR_MAX;
        local_unary_cost[y][0][x][d] = ddr_unary_cost[(y * WIDTH + x) * DISP_DDR_MAX + d]; 
    }
    
    LOOP_COST1:
    for (int xyd = 0; xyd < LAYER0_SIZE * LAYER0_SIZE * DISP_DDR_MAX; xyd++) {
#pragma HLS pipeline II=1
        int y = xyd / (LAYER0_SIZE * DISP_DDR_MAX);
        int x = xyd % (LAYER0_SIZE * DISP_DDR_MAX) / DISP_DDR_MAX;
        int d = xyd % (LAYER0_SIZE * DISP_DDR_MAX) % DISP_DDR_MAX;
        local_unary_cost[y + LAYER0_SIZE][0][x][d] = ddr_unary_cost[((y + LAYER0_SIZE) * WIDTH + x) * DISP_DDR_MAX + d];
    }

    LOOP_COST2:
    for (int xyd = 0; xyd < LAYER0_SIZE * LAYER0_SIZE * DISP_DDR_MAX; xyd++) {
#pragma HLS pipeline II=1
        int y = xyd / (LAYER0_SIZE * DISP_DDR_MAX);
        int x = xyd % (LAYER0_SIZE * DISP_DDR_MAX) / DISP_DDR_MAX;
        int d = xyd % (LAYER0_SIZE * DISP_DDR_MAX) % DISP_DDR_MAX;
        local_unary_cost[y][1][x][d] = ddr_unary_cost[(y * WIDTH + x + LAYER0_SIZE) * DISP_DDR_MAX + d];
    }

    LOOP_COST3:
    for (int xyd = 0; xyd < LAYER0_SIZE * LAYER0_SIZE * DISP_DDR_MAX; xyd++) {
#pragma HLS pipeline II=1
        int y = xyd / (LAYER0_SIZE * DISP_DDR_MAX);
        int x = xyd % (LAYER0_SIZE * DISP_DDR_MAX) / DISP_DDR_MAX;
        int d = xyd % (LAYER0_SIZE * DISP_DDR_MAX) % DISP_DDR_MAX;
        local_unary_cost[y + LAYER0_SIZE][1][x][d] = ddr_unary_cost[((y + LAYER0_SIZE) * WIDTH + x + LAYER0_SIZE) * DISP_DDR_MAX + d];
    }

    LOOP_IMG:
    for (int y = 0; y < LAYER0_SIZE; y++) {
        for (int x = 0; x < LAYER0_SIZE; x++) {
#pragma HLS loop_flatten
#pragma HLS pipeline II=1
            local_im0_gray[y][0][x] = ddr_im0_gray[y * WIDTH + x];
            local_im0_gray[y + LAYER0_SIZE][0][x] = ddr_im0_gray[(y + LAYER0_SIZE) * WIDTH + x];
            local_im0_gray[y][1][x] = ddr_im0_gray[y * WIDTH + x + LAYER0_SIZE];
            local_im0_gray[y + LAYER0_SIZE][1][x] = ddr_im0_gray[(y + LAYER0_SIZE) * WIDTH + x + LAYER0_SIZE];
        }
    }
}

void layer0OutloopCache(unary_cost_t local_buffer[2][LAYER0_SIZE][DISP_DDR_MAX0], unary_cost_t local_unary_cost[LAYER0_SIZE * 3][4][LAYER0_SIZE][DISP_DDR_MAX], unary_cost_t0 ddr_unary_cost[WIDTH * HEIGHT * DISP_DDR_MAX0]) {
#pragma HLS inline off
// #pragma HLS array_partition variable=local_unary_cost type=complete dim=1
// #pragma HLS array_partition variable=local_unary_cost type=complete dim=2
// DO_PRAGMA(HLS array_reshape variable=local_unary_cost type=cyclic factor=DISP_DDR_NUM dim=4)
// #pragma HLS array_partition variable=local_unary_cost_init type=complete dim=1
// #pragma HLS array_partition variable=local_unary_cost_init type=complete dim=2
// DO_PRAGMA(HLS array_reshape variable=local_unary_cost_init type=cyclic factor=DISP_DDR_NUM dim=4)   
// #pragma HLS interface mode=m_axi bundle=BUS_LAYER0 port=ddr_unary_cost offset=slave
#pragma HLS array_partition variable=local_buffer type=complete dim=1
    //bool index = 0;
    LOOP_COST0:
    for (int xyd = 0; xyd <  LAYER0_SIZE * DISP_DDR_MAX0; xyd++) {
#pragma HLS pipeline II=1
        int y = 0;
        int x = xyd % (LAYER0_SIZE * DISP_DDR_MAX0) / DISP_DDR_MAX0;
        int d = xyd % (LAYER0_SIZE * DISP_DDR_MAX0) % DISP_DDR_MAX0;
        bool index = y % 2;
        int dd = d << 1;
        local_unary_cost[y][0][x][dd] = ddr_unary_cost[(y * WIDTH + x) * DISP_DDR_MAX0 + d](127,0); 
        local_buffer[index][x][d] = ddr_unary_cost[(y * WIDTH + x) * DISP_DDR_MAX0 + d](255,128);
    }
    LOOP_COST1:
    for (int xyd = 0; xyd < LAYER0_SIZE * DISP_DDR_MAX0 * (LAYER0_SIZE * 2 - 1); xyd++) {
#pragma HLS pipeline II=1
        int y = 1 + xyd / (LAYER0_SIZE * DISP_DDR_MAX0);
        int x = xyd % (LAYER0_SIZE * DISP_DDR_MAX0) / DISP_DDR_MAX0;
        int d = xyd % (LAYER0_SIZE * DISP_DDR_MAX0) % DISP_DDR_MAX0;
        bool index = y % 2;
        int dd = d << 1;
        local_unary_cost[y][0][x][dd] = ddr_unary_cost[(y * WIDTH + x) * DISP_DDR_MAX0 + d](127,0); 
        local_buffer[index][x][d] = ddr_unary_cost[(y * WIDTH + x) * DISP_DDR_MAX0 + d](255,128);
        if (dd >= DISP_DDR_MAX - 1)
            continue;  
        local_unary_cost[y - 1][0][x][dd + 1] = local_buffer[!index][x][d];
    }
    LOOP_COST2:
    for (int xyd = 0; xyd <  LAYER0_SIZE * DISP_DDR_MAX0; xyd++) {
#pragma HLS pipeline II=1
        int y = 18;
        int x = xyd % (LAYER0_SIZE * DISP_DDR_MAX0) / DISP_DDR_MAX0;
        int d = xyd % (LAYER0_SIZE * DISP_DDR_MAX0) % DISP_DDR_MAX0;
        bool index = y % 2;
        int dd = d << 1;
        if (dd >= DISP_DDR_MAX - 1)
            continue; 
        local_unary_cost[y - 1][0][x][dd + 1] = local_buffer[!index][x][d];
    }
    
    LOOP_COST3:
for (int xyd = 0; xyd <  LAYER0_SIZE * DISP_DDR_MAX0; xyd++) {
#pragma HLS pipeline II=1
        int y = 0;
        int x = xyd % (LAYER0_SIZE * DISP_DDR_MAX0) / DISP_DDR_MAX0;
        int d = xyd % (LAYER0_SIZE * DISP_DDR_MAX0) % DISP_DDR_MAX0;
        bool index = y % 2;
        int dd = d << 1;
        local_unary_cost[y][1][x][dd] = ddr_unary_cost[(y * WIDTH + x + LAYER0_SIZE)* DISP_DDR_MAX0 + d](127,0); 
        local_buffer[index][x][d] = ddr_unary_cost[(y * WIDTH + x + LAYER0_SIZE)* DISP_DDR_MAX0 + d](255,128);
    }
    LOOP_COST4:
    for (int xyd = 0; xyd < LAYER0_SIZE * DISP_DDR_MAX0 * (LAYER0_SIZE * 2 - 1); xyd++) {
#pragma HLS pipeline II=1
        int y = 1 + xyd / (LAYER0_SIZE * DISP_DDR_MAX0);
        int x = xyd % (LAYER0_SIZE * DISP_DDR_MAX0) / DISP_DDR_MAX0;
        int d = xyd % (LAYER0_SIZE * DISP_DDR_MAX0) % DISP_DDR_MAX0;
        bool index = y % 2;
        int dd = d << 1;
        local_unary_cost[y][1][x][dd] = ddr_unary_cost[(y * WIDTH + x + LAYER0_SIZE)* DISP_DDR_MAX0 + d](127,0); 
        local_buffer[index][x][d] = ddr_unary_cost[(y * WIDTH + x + LAYER0_SIZE)* DISP_DDR_MAX0 + d](255,128);
        if (dd >= DISP_DDR_MAX - 1)
            continue; 
        local_unary_cost[y - 1][1][x][dd + 1] = local_buffer[!index][x][d];
    }
    LOOP_COST5:
    for (int xyd = 0; xyd <  LAYER0_SIZE * DISP_DDR_MAX0; xyd++) {
#pragma HLS pipeline II=1
        int y = 18;
        int x = xyd % (LAYER0_SIZE * DISP_DDR_MAX0) / DISP_DDR_MAX0;
        int d = xyd % (LAYER0_SIZE * DISP_DDR_MAX0) % DISP_DDR_MAX0;
        bool index = y % 2;
        int dd = d << 1;
        if (dd >= DISP_DDR_MAX - 1)
            continue; 
        local_unary_cost[y - 1][1][x][dd + 1] = local_buffer[!index][x][d];
    }
    /*
    LOOP_IMG:
    for (int y = 0; y < LAYER0_SIZE; y++) {
        for (int x = 0; x < LAYER0_SIZE; x++) {
#pragma HLS loop_flatten
#pragma HLS pipeline II=1
            local_im0_gray[y][0][x] = ddr_im0_gray[y * WIDTH + x];
            local_im0_gray[y + LAYER0_SIZE][0][x] = ddr_im0_gray[(y + LAYER0_SIZE) * WIDTH + x];
            local_im0_gray[y][1][x] = ddr_im0_gray[y * WIDTH + x + LAYER0_SIZE];
            local_im0_gray[y + LAYER0_SIZE][1][x] = ddr_im0_gray[(y + LAYER0_SIZE) * WIDTH + x + LAYER0_SIZE];
        }
    }
   
    LOOP_COST1:
    for (int xyd = 0; xyd < LAYER0_SIZE * LAYER0_SIZE * DISP_DDR_MAX; xyd++) {
#pragma HLS pipeline II=1
        int y = xyd / (LAYER0_SIZE * DISP_DDR_MAX);
        int x = xyd % (LAYER0_SIZE * DISP_DDR_MAX) / DISP_DDR_MAX;
        int d = xyd % (LAYER0_SIZE * DISP_DDR_MAX) % DISP_DDR_MAX;
        local_unary_cost[y + LAYER0_SIZE][0][x][d] = ddr_unary_cost[((y + LAYER0_SIZE) * WIDTH + x) * DISP_DDR_MAX + d];
    }
    
    LOOP_COST2:
    for (int xyd = 0; xyd < LAYER0_SIZE * LAYER0_SIZE * DISP_DDR_MAX; xyd++) {
#pragma HLS pipeline II=1
        int y = xyd / (LAYER0_SIZE * DISP_DDR_MAX);
        int x = xyd % (LAYER0_SIZE * DISP_DDR_MAX) / DISP_DDR_MAX;
        int d = xyd % (LAYER0_SIZE * DISP_DDR_MAX) % DISP_DDR_MAX;
        local_unary_cost[y][1][x][d] = ddr_unary_cost[(y * WIDTH + x + LAYER0_SIZE) * DISP_DDR_MAX + d];
    }

    LOOP_COST3:
    for (int xyd = 0; xyd < LAYER0_SIZE * LAYER0_SIZE * DISP_DDR_MAX; xyd++) {
#pragma HLS pipeline II=1
        int y = xyd / (LAYER0_SIZE * DISP_DDR_MAX);
        int x = xyd % (LAYER0_SIZE * DISP_DDR_MAX) / DISP_DDR_MAX;
        int d = xyd % (LAYER0_SIZE * DISP_DDR_MAX) % DISP_DDR_MAX;
        local_unary_cost[y + LAYER0_SIZE][1][x][d] = ddr_unary_cost[((y + LAYER0_SIZE) * WIDTH + x + LAYER0_SIZE) * DISP_DDR_MAX + d];
    }
    */
}

template<size_t CUR_SIZE, int row_num, int layer_size, typename cost_blk, typename label_blk, typename disp_blk>
void layerTransLayer(int iter, int ver_region_num, int hor_region_num, 
#if HLS_LAYER1_ONLY == 0
    hls::stream_of_blocks<cost_blk> &cost_blk_out, hls::stream_of_blocks<label_blk> &label_blk_out, hls::stream_of_blocks<disp_blk> &disp_blk_out, 
#else
    float cost_bout[LAYER1_SIZE * LAYER1_SIZE], float label_bout[LAYER1_SIZE * LAYER1_SIZE * 3], float disp_bout[LAYER1_SIZE * LAYER1_SIZE],
#endif
    float uram_cost[CUR_SIZE][WIDTH], float uram_label[CUR_SIZE][WIDTH][3], float uram_disp[CUR_SIZE][WIDTH]) {
#pragma HLS inline off
#if HLS_LAYER1_ONLY == 0
    hls::write_lock<cost_blk> cost_bout(cost_blk_out);
    hls::write_lock<label_blk> label_bout(label_blk_out);
    hls::write_lock<disp_blk> disp_bout(disp_blk_out);
#endif
    // #pragma HLS array_partition variable=bout type=block factor=5 dim=1
    int tran_ver_region_i = (iter - (hor_region_num + 1)) / hor_region_num;
    int tran_hor_region_i = (iter - (hor_region_num + 1)) % hor_region_num;
    int tran_unit_region_x, tran_unit_region_y, tran_unit_region_w, tran_unit_region_h;
    int tran_expn_region_x, tran_expn_region_y, tran_expn_region_w, tran_expn_region_h;
    genRegionInfo(WIDTH, HEIGHT, tran_hor_region_i, tran_ver_region_i, layer_size, tran_unit_region_x, tran_unit_region_y, tran_unit_region_w, tran_unit_region_h, 
        tran_expn_region_x, tran_expn_region_y, tran_expn_region_w, tran_expn_region_h);
    // 
    LOOP7:
    for (int x = 0; x < layer_size; x++) {
        LOOP8:
        for (int y = 0; y < layer_size; y++) {
#pragma HLS pipeline II=1
            if ((x >= tran_unit_region_w) || (y >= tran_unit_region_h))
                continue;
            cost_bout[y * layer_size + x] = uram_cost[(tran_ver_region_i % row_num) * layer_size + y][tran_unit_region_x + x];
            label_bout[(y * layer_size + x) * 3 + 0] = uram_label[(tran_ver_region_i % row_num) * layer_size + y][tran_unit_region_x + x][0];
            label_bout[(y * layer_size + x) * 3 + 1] = uram_label[(tran_ver_region_i % row_num) * layer_size + y][tran_unit_region_x + x][1];
            label_bout[(y * layer_size + x) * 3 + 2] = uram_label[(tran_ver_region_i % row_num) * layer_size + y][tran_unit_region_x + x][2];
            disp_bout[y * layer_size + x] = uram_disp[(tran_ver_region_i % row_num) * layer_size + y][tran_unit_region_x + x];
        }
    }
}

void layer0Trans(int iter, int ver_region_num, int hor_region_num, int blk_num,
#if HLS_LAYER0_ONLY == 0
    hls::stream_of_blocks<layer0_cost_blk> &cost_blk_out, hls::stream_of_blocks<layer0_label_blk> &label_blk_out, hls::stream_of_blocks<layer0_disp_blk> &disp_blk_out,
    hls::stream_of_blocks<layer0_cost_blk> &init_cost_blk_out, hls::stream_of_blocks<layer0_label_blk> &init_label_blk_out, hls::stream_of_blocks<layer0_disp_blk> &init_disp_blk_out,
#else
    float cost_bout[LAYER0_SIZE * LAYER0_SIZE], float label_bout[LAYER0_SIZE * LAYER0_SIZE * 3], float disp_bout[LAYER0_SIZE * LAYER0_SIZE], 
#endif
    ap_uint<DISP_DDR_BIT> local_trans_cost[LAYER0_SIZE][LAYER0_SIZE], label_range local_trans_label[LAYER0_SIZE][LAYER0_SIZE][3], disp_range local_trans_disp[LAYER0_SIZE][LAYER0_SIZE]) {
#pragma HLS inline off
    // 
    if (blk_num < LAYER0_HOR_REGION_NUM * (LAYER1_SIZE / LAYER0_SIZE) * 2) {
        hls::write_lock<layer0_cost_blk> init_cost_bout(init_cost_blk_out);
        hls::write_lock<layer0_label_blk> init_label_bout(init_label_blk_out);
        hls::write_lock<layer0_disp_blk> init_disp_bout(init_disp_blk_out);
        for (int x = 0; x < LAYER0_SIZE; x++) {
            for (int y = 0; y < LAYER0_SIZE; y++) {
#pragma HLS pipeline II=1
                init_cost_bout[y * LAYER0_SIZE + x] = local_trans_cost[y][x];
                init_label_bout[(y * LAYER0_SIZE + x) * 3 + 0] = local_trans_label[y][x][0];
                init_label_bout[(y * LAYER0_SIZE + x) * 3 + 1] = local_trans_label[y][x][1];
                init_label_bout[(y * LAYER0_SIZE + x) * 3 + 2] = local_trans_label[y][x][2];
                init_disp_bout[y * LAYER0_SIZE + x] = local_trans_disp[y][x];
            }
        }
    }
    else {
        hls::write_lock<layer0_cost_blk> cost_bout(cost_blk_out);
        hls::write_lock<layer0_label_blk> label_bout(label_blk_out);
        hls::write_lock<layer0_disp_blk> disp_bout(disp_blk_out);
        LOOP7:
        for (int x = 0; x < LAYER0_SIZE; x++) {
            LOOP8:
            for (int y = 0; y < LAYER0_SIZE; y++) {
#pragma HLS pipeline II=1
                cost_bout[y * LAYER0_SIZE + x] = local_trans_cost[y][x];
                label_bout[(y * LAYER0_SIZE + x) * 3 + 0] = local_trans_label[y][x][0];
                label_bout[(y * LAYER0_SIZE + x) * 3 + 1] = local_trans_label[y][x][1];
                label_bout[(y * LAYER0_SIZE + x) * 3 + 2] = local_trans_label[y][x][2];
                disp_bout[y * LAYER0_SIZE + x] = local_trans_disp[y][x];
            }
        }
    }
}

void layer0Transnew(int iter, int ver_region_num, int hor_region_num, int blk_num,
#if HLS_LAYER0_ONLY == 0
    ap_uint<DISP_DDR_BIT> ddr_cost[WIDTH * HEIGHT], label_range ddr_label[WIDTH * HEIGHT * 3], disp_range ddr_disp[WIDTH * HEIGHT],
#else
    float cost_bout[LAYER0_SIZE * LAYER0_SIZE], float label_bout[LAYER0_SIZE * LAYER0_SIZE * 3], float disp_bout[LAYER0_SIZE * LAYER0_SIZE], 
#endif
      ap_uint<DISP_DDR_BIT> local_trans_cost[LAYER0_SIZE][LAYER0_SIZE], label_range local_trans_label[LAYER0_SIZE][LAYER0_SIZE][3], disp_range local_trans_disp[LAYER0_SIZE][LAYER0_SIZE]
//      ap_uint<DISP_DDR_BIT> local_cost[LAYER0_SIZE * 3][LAYER0_SIZE * 3], label_range local_label[LAYER0_SIZE * 3][LAYER0_SIZE * 3][3], disp_range local_disp[LAYER0_SIZE * 3][LAYER0_SIZE * 3]
) {
#pragma HLS inline off
#pragma HLS array_partition variable=local_trans_cost type=complete dim=1
#pragma HLS array_partition variable=local_trans_label type=complete dim=1
#pragma HLS array_partition variable=local_trans_label type=complete dim=3
#pragma HLS array_partition variable=local_trans_disp type=complete dim=1
    // 
    int tran_ver_region_i = (iter - (hor_region_num + 2)) / hor_region_num;
    int tran_hor_region_i = (iter - (hor_region_num + 2)) % hor_region_num;
    int tran_unit_region_x, tran_unit_region_y, tran_unit_region_w, tran_unit_region_h;
    int tran_expn_region_x, tran_expn_region_y, tran_expn_region_w, tran_expn_region_h;
    genRegionInfo(WIDTH, HEIGHT, tran_hor_region_i, tran_ver_region_i, LAYER0_SIZE, tran_unit_region_x, tran_unit_region_y, tran_unit_region_w, tran_unit_region_h, 
        tran_expn_region_x, tran_expn_region_y, tran_expn_region_w, tran_expn_region_h);

        for (int x = 0; x < LAYER0_SIZE; x++) {
            for (int y = 0; y < LAYER0_SIZE; y++) {
#pragma HLS pipeline II=1
if ((x >= tran_unit_region_w) || (y >= tran_unit_region_h))
                continue;
                ddr_cost[(tran_unit_region_y + y) * WIDTH + tran_unit_region_x + x] = local_trans_cost[y][x];
                ddr_label[((tran_unit_region_y + y) * WIDTH + tran_unit_region_x + x) * 3 + 0] = local_trans_label[y][x][0];
                ddr_label[((tran_unit_region_y + y) * WIDTH + tran_unit_region_x + x) * 3 + 1] = local_trans_label[y][x][1];
                ddr_label[((tran_unit_region_y + y) * WIDTH + tran_unit_region_x + x) * 3 + 2] = local_trans_label[y][x][2];
                ddr_disp[(tran_unit_region_y + y) * WIDTH + tran_unit_region_x + x] = local_trans_disp[y][x];
            }
        }
    }

void layer1Trans(int iter, int ver_region_num, int hor_region_num, 
#if LAYER2_ENABLE == 0
    float ddr_cost[WIDTH * HEIGHT], float ddr_label[WIDTH * HEIGHT * 3], float ddr_disp[WIDTH * HEIGHT],
#elif HLS_LAYER1_ONLY == 0
    hls::stream_of_blocks<layer1_cost_blk> &cost_blk_out, hls::stream_of_blocks<layer1_label_blk> &label_blk_out, hls::stream_of_blocks<layer1_disp_blk> &disp_blk_out, 
#else
    float cost_bout[LAYER1_SIZE * LAYER1_SIZE], float label_bout[LAYER1_SIZE * LAYER1_SIZE * 3], float disp_bout[LAYER1_SIZE * LAYER1_SIZE],
#endif
    float uram_cost[LAYER1_SIZE * 4][WIDTH], float uram_label[LAYER1_SIZE * 4][WIDTH][3], float uram_disp[LAYER1_SIZE * 4][WIDTH]) {
#pragma HLS inline off
#if HLS_LAYER1_ONLY == 0 && LAYER2_ENABLE == 1
    hls::write_lock<layer1_cost_blk> cost_bout(cost_blk_out);
    hls::write_lock<layer1_label_blk> label_bout(label_blk_out);
    hls::write_lock<layer1_disp_blk> disp_bout(disp_blk_out);
#endif
    // #pragma HLS array_partition variable=bout type=block factor=5 dim=1
    int tran_ver_region_i = (iter - (hor_region_num + 1)) / hor_region_num;
    int tran_hor_region_i = (iter - (hor_region_num + 1)) % hor_region_num;
    int tran_unit_region_x, tran_unit_region_y, tran_unit_region_w, tran_unit_region_h;
    int tran_expn_region_x, tran_expn_region_y, tran_expn_region_w, tran_expn_region_h;
    genRegionInfo(WIDTH, HEIGHT, tran_hor_region_i, tran_ver_region_i, LAYER1_SIZE, tran_unit_region_x, tran_unit_region_y, tran_unit_region_w, tran_unit_region_h, 
        tran_expn_region_x, tran_expn_region_y, tran_expn_region_w, tran_expn_region_h);
    // 
    LOOP7:
    for (int x = 0; x < LAYER1_SIZE; x++) {
        LOOP8:
        for (int y = 0; y < LAYER1_SIZE; y++) {
#pragma HLS pipeline II=1
            if ((x >= tran_unit_region_w) || (y >= tran_unit_region_h))
                continue;
#if LAYER2_ENABLE == 1
            cost_bout[y * LAYER1_SIZE + x] = uram_cost[(tran_ver_region_i % 4) * LAYER1_SIZE + y][tran_unit_region_x + x];
            label_bout[(y * LAYER1_SIZE + x) * 3 + 0] = uram_label[(tran_ver_region_i % 4) * LAYER1_SIZE + y][tran_unit_region_x + x][0];
            label_bout[(y * LAYER1_SIZE + x) * 3 + 1] = uram_label[(tran_ver_region_i % 4) * LAYER1_SIZE + y][tran_unit_region_x + x][1];
            label_bout[(y * LAYER1_SIZE + x) * 3 + 2] = uram_label[(tran_ver_region_i % 4) * LAYER1_SIZE + y][tran_unit_region_x + x][2];
            disp_bout[y * LAYER1_SIZE + x] = uram_disp[(tran_ver_region_i % 4) * LAYER1_SIZE + y][tran_unit_region_x + x];
#else
            ddr_cost[(tran_unit_region_y + y) * WIDTH + tran_unit_region_x + x] = uram_cost[(tran_ver_region_i % 4) * LAYER2_SIZE + y][tran_unit_region_x + x];
            ddr_label[((tran_unit_region_y + y) * WIDTH + tran_unit_region_x + x) * 3 + 0] = uram_label[(tran_ver_region_i % 4) * LAYER2_SIZE + y][tran_unit_region_x + x][0];
            ddr_label[((tran_unit_region_y + y) * WIDTH + tran_unit_region_x + x) * 3 + 1] = uram_label[(tran_ver_region_i % 4) * LAYER2_SIZE + y][tran_unit_region_x + x][1];
            ddr_label[((tran_unit_region_y + y) * WIDTH + tran_unit_region_x + x) * 3 + 2] = uram_label[(tran_ver_region_i % 4) * LAYER2_SIZE + y][tran_unit_region_x + x][2];
            ddr_disp[(tran_unit_region_y + y) * WIDTH + tran_unit_region_x + x] = uram_disp[(tran_ver_region_i % 4) * LAYER2_SIZE + y][tran_unit_region_x + x];
#endif
        }
    }
}

void layer1Trans(int iter, 
#if LAYER2_ENABLE == 0
    int tran_unit_region_x, int tran_unit_region_y, int tran_unit_region_w, int tran_unit_region_h,
    ap_uint<DISP_DDR_BIT> ddr_cost[WIDTH * HEIGHT], label_range ddr_label[WIDTH * HEIGHT * 3], disp_range ddr_disp[WIDTH * HEIGHT],
#elif HLS_LAYER1_ONLY == 0
    hls::stream_of_blocks<layer1_cost_blk> &cost_blk_out, hls::stream_of_blocks<layer1_label_blk> &label_blk_out, hls::stream_of_blocks<layer1_disp_blk> &disp_blk_out,
#else
    float cost_bout[LAYER1_SIZE * LAYER1_SIZE], float label_bout[LAYER1_SIZE * LAYER1_SIZE * 3], float disp_bout[LAYER1_SIZE * LAYER1_SIZE], 
#endif
    ap_uint<DISP_DDR_BIT> local_trans_cost[LAYER1_SIZE][LAYER1_SIZE], label_range local_trans_label[LAYER1_SIZE][LAYER1_SIZE][3], disp_range local_trans_disp[LAYER1_SIZE][LAYER1_SIZE]) {
#pragma HLS inline off
#pragma HLS array_partition variable=local_trans_cost type=complete dim=1
#pragma HLS array_partition variable=local_trans_label type=complete dim=1
#pragma HLS array_partition variable=local_trans_label type=complete dim=3
#pragma HLS array_partition variable=local_trans_disp type=complete dim=1
#if HLS_LAYER1_ONLY == 0 && LAYER2_ENABLE == 1
    hls::write_lock<layer1_cost_blk> cost_bout(cost_blk_out);
    hls::write_lock<layer1_label_blk> label_bout(label_blk_out);
    hls::write_lock<layer1_disp_blk> disp_bout(disp_blk_out);
#endif
    // 
    LOOP7:
    for (int x = 0; x < LAYER1_SIZE; x++) {
        LOOP8:
        for (int y = 0; y < LAYER1_SIZE; y++) {
#pragma HLS pipeline II=1
            if ((x >= tran_unit_region_w) || (y >= tran_unit_region_h))
                continue;
#if LAYER2_ENABLE == 1
            cost_bout[y * LAYER1_SIZE + x] = local_trans_cost[y][x];
            label_bout[(y * LAYER1_SIZE + x) * 3 + 0] = local_trans_label[y][x][0];
            label_bout[(y * LAYER1_SIZE + x) * 3 + 1] = local_trans_label[y][x][1];
            label_bout[(y * LAYER1_SIZE + x) * 3 + 2] = local_trans_label[y][x][2];
            disp_bout[y * LAYER1_SIZE + x] = local_trans_disp[y][x];
#else
            ddr_cost[(tran_unit_region_y + y) * WIDTH + tran_unit_region_x + x] = local_trans_cost[y][x];
            ddr_label[((tran_unit_region_y + y) * WIDTH + tran_unit_region_x + x) * 3 + 0] = local_trans_label[y][x][0];
            ddr_label[((tran_unit_region_y + y) * WIDTH + tran_unit_region_x + x) * 3 + 1] = local_trans_label[y][x][1];
            ddr_label[((tran_unit_region_y + y) * WIDTH + tran_unit_region_x + x) * 3 + 2] = local_trans_label[y][x][2];
            ddr_disp[(tran_unit_region_y + y) * WIDTH + tran_unit_region_x + x] = local_trans_disp[y][x];
#endif
        }
    }
}

void layer1TransWrapper(int l1_iter, 
#if LAYER2_ENABLE == 0
    int tran_unit_region_x, int tran_unit_region_y, int tran_unit_region_w, int tran_unit_region_h,
    ap_uint<DISP_DDR_BIT> ddr_cost[WIDTH * HEIGHT], label_range ddr_label[WIDTH * HEIGHT * 3], disp_range ddr_disp[WIDTH * HEIGHT],
#elif HLS_LAYER1_ONLY == 0
    hls::stream_of_blocks<layer1_cost_blk> &cost_blk_out, hls::stream_of_blocks<layer1_label_blk> &label_blk_out, hls::stream_of_blocks<layer1_disp_blk> &disp_blk_out,
#else
    float cost_blk_out[LAYER1_SIZE * LAYER1_SIZE], float label_blk_out[LAYER1_SIZE * LAYER1_SIZE * 3], float disp_blk_out[LAYER1_SIZE * LAYER1_SIZE], 
#endif
    ap_uint<DISP_DDR_BIT> local_trans_cost[LAYER1_SIZE][LAYER1_SIZE], label_range local_trans_label[LAYER1_SIZE][LAYER1_SIZE][3], disp_range local_trans_disp[LAYER1_SIZE][LAYER1_SIZE]) {
#pragma HLS inline off
    if (l1_iter >= (LAYER1_HOR_REGION_NUM + 2)){
#if LAYER2_ENABLE == 1
        layer1Trans(l1_iter, cost_blk_out, label_blk_out, disp_blk_out, local_trans_cost, local_trans_label, local_trans_disp);
#else
        layer1Trans(l1_iter,  tran_unit_region_x, tran_unit_region_y, tran_unit_region_w, tran_unit_region_h, ddr_cost, ddr_label, ddr_disp, local_trans_cost, local_trans_label, local_trans_disp);
#endif
    }
}

void layer0CacheWrapper(unary_cost_t local_buffer[2][LAYER0_SIZE][DISP_DDR_MAX0], int iter, int unary_cost_x3, int buff_id, 
    unary_cost_t local_unary_cost[LAYER0_SIZE * 3][4][LAYER0_SIZE][DISP_DDR_MAX], 
    unary_cost_t0 ddr_unary_cost[WIDTH * HEIGHT * DISP_DDR_MAX0]) {
#pragma HLS inline off
// #pragma HLS array_partition variable=local_unary_cost type=complete dim=1
// #pragma HLS array_partition variable=local_unary_cost type=complete dim=2
// DO_PRAGMA(HLS array_reshape variable=local_unary_cost type=cyclic factor=DISP_DDR_NUM dim=4)   
// #pragma HLS interface mode=m_axi bundle=BUS_LAYER0 port=ddr_unary_cost offset=slave
    assert(buff_id == 0 || buff_id == 1);
    /* cache unary cost from DDR */
    if (iter < (LAYER0_VER_REGION_NUM * LAYER0_HOR_REGION_NUM - 2)) {
        //layer0InloopCache(iter, LAYER0_VER_REGION_NUM, LAYER0_HOR_REGION_NUM, unary_cost_x3, local_unary_cost, ddr_unary_cost, local_im0_gray, ddr_im0_gray);
        layer0InloopCache(local_buffer, iter, LAYER0_VER_REGION_NUM, LAYER0_HOR_REGION_NUM, unary_cost_x3, local_unary_cost, ddr_unary_cost);
    }
    // if (iter < (LAYER0_VER_REGION_NUM * LAYER0_HOR_REGION_NUM - LAYER0_HOR_REGION_NUM - 3)) {
    //     layer0InloopCacheInit(iter, LAYER0_VER_REGION_NUM, LAYER0_HOR_REGION_NUM, buff_id, local_unary_cost_init, ddr_unary_cost);
    // }
}

void layer0UramLoad(int iter, int ver_region_num, int hor_region_num, ap_uint<DISP_DDR_BIT> uram_cost[LAYER0_SIZE * 3][WIDTH], label_range uram_label[LAYER0_SIZE * 3][WIDTH][3], disp_range uram_disp[LAYER0_SIZE * 3][WIDTH],
    ap_uint<DISP_DDR_BIT> local_cost[LAYER0_SIZE * 3][LAYER0_SIZE * 3], label_range local_label[LAYER0_SIZE * 3][LAYER0_SIZE * 3][3], disp_range local_disp[LAYER0_SIZE * 3][LAYER0_SIZE * 3],
    ap_uint<DISP_DDR_BIT> local_trans_cost[LAYER0_SIZE][LAYER0_SIZE], label_range local_trans_label[LAYER0_SIZE][LAYER0_SIZE][3], disp_range local_trans_disp[LAYER0_SIZE][LAYER0_SIZE]
) {
#pragma HLS array_partition variable=uram_cost type=complete dim=1      
#pragma HLS array_partition variable=uram_label type=complete dim=1
#pragma HLS array_partition variable=uram_label type=complete dim=3
#pragma HLS array_partition variable=uram_disp type=complete dim=1
#pragma HLS array_partition variable=local_cost type=complete dim=1
#pragma HLS array_partition variable=local_label type=complete dim=1
#pragma HLS array_partition variable=local_label type=complete dim=3
#pragma HLS array_partition variable=local_disp type=complete dim=1
#pragma HLS array_partition variable=local_trans_cost type=complete dim=1
#pragma HLS array_partition variable=local_trans_label type=complete dim=1
#pragma HLS array_partition variable=local_trans_label type=complete dim=3
#pragma HLS array_partition variable=local_trans_disp type=complete dim=1
    /*  */
    int ver_region_i = iter / hor_region_num;
    int hor_region_i = iter % hor_region_num;
    int unit_region_x, unit_region_y, unit_region_w, unit_region_h;
    int expn_region_x, expn_region_y, expn_region_w, expn_region_h;
    genRegionInfo(WIDTH, HEIGHT, hor_region_i, ver_region_i, LAYER0_SIZE, unit_region_x, unit_region_y, unit_region_w, unit_region_h, 
        expn_region_x, expn_region_y, expn_region_w, expn_region_h);
    /* address mapping table for uram */
    int uram_y_addr_id = ver_region_i % 3;
    /* uram -> local ram */
    LOOP_LOAD:
    for (int x = 0; x < LAYER0_SIZE * 3; x++) {
#pragma HLS pipeline II=1
        if ((expn_region_x + x) >= WIDTH)   
            continue;
        if (ver_region_i == 0) {
            L0_GLOBAL_LOCAL_ADDR1
        }
        else {
            if (uram_y_addr_id == 0) {
                L0_GLOBAL_LOCAL_ADDR0
            }
            else if (uram_y_addr_id == 2) {
                L0_GLOBAL_LOCAL_ADDR2
            }
            else {
                L0_GLOBAL_LOCAL_ADDR1
            }
        }
    }

    int tran_ver_region_i = (iter - (hor_region_num + 2)) / hor_region_num;
    int tran_hor_region_i = (iter - (hor_region_num + 2)) % hor_region_num;
    int tran_unit_region_x, tran_unit_region_y, tran_unit_region_w, tran_unit_region_h;
    int tran_expn_region_x, tran_expn_region_y, tran_expn_region_w, tran_expn_region_h;
    genRegionInfo(WIDTH, HEIGHT, tran_hor_region_i, tran_ver_region_i, LAYER0_SIZE, tran_unit_region_x, tran_unit_region_y, tran_unit_region_w, tran_unit_region_h, 
        tran_expn_region_x, tran_expn_region_y, tran_expn_region_w, tran_expn_region_h);
    if (tran_ver_region_i < 0 || tran_hor_region_i < 0)
        return;
    for (int x = 0; x < LAYER0_SIZE; x++) {
#pragma HLS pipeline II=1
        // for (int y = 0; y < LAYER0_SIZE; y++) {
// #pragma HLS unroll
            // if ((x >= tran_unit_region_w) || (y >= tran_unit_region_h)) continue;
            if ((tran_unit_region_x + x) >= WIDTH)   
                continue;
            if ((tran_ver_region_i % 3) == 0) {
                L0_GLOBAL_LOCAL_TRANS_ADDR0
            }
            else if ((tran_ver_region_i % 3) == 1) {
                L0_GLOBAL_LOCAL_TRANS_ADDR1
            }
            else {
                L0_GLOBAL_LOCAL_TRANS_ADDR2
            }
            // local_trans_cost[y][x] = uram_cost[(tran_ver_region_i % 3) * LAYER0_SIZE + y][tran_unit_region_x + x];
            // local_trans_label[y][x][0] = uram_label[(tran_ver_region_i % 3) * LAYER0_SIZE + y][tran_unit_region_x + x][0];
            // local_trans_label[y][x][1] = uram_label[(tran_ver_region_i % 3) * LAYER0_SIZE + y][tran_unit_region_x + x][1];
            // local_trans_label[y][x][2] = uram_label[(tran_ver_region_i % 3) * LAYER0_SIZE + y][tran_unit_region_x + x][2];
            // local_trans_disp[y][x] = uram_disp[(tran_ver_region_i % 3) * LAYER0_SIZE + y][tran_unit_region_x + x];
        // }
    }
    

}

void layer1UramLoad(int iter, int uram_y0, int uram_y1, int uram_y2, int& tran_unit_region_x, int& tran_unit_region_y, int& tran_unit_region_w, int& tran_unit_region_h,
    ap_uint<DISP_DDR_BIT> uram_cost[4][LAYER1_SIZE][WIDTH], label_range uram_label[4][LAYER1_SIZE][WIDTH][3], disp_range uram_disp[4][LAYER1_SIZE][WIDTH],
    ap_uint<DISP_DDR_BIT> local_cost[LAYER1_SIZE * 3][LAYER1_SIZE * 3], label_range local_label[LAYER1_SIZE * 3][LAYER1_SIZE * 3][3], disp_range local_disp[LAYER1_SIZE * 3][LAYER1_SIZE * 3],
    ap_uint<DISP_DDR_BIT> local_trans_cost[LAYER1_SIZE][LAYER1_SIZE], label_range local_trans_label[LAYER1_SIZE][LAYER1_SIZE][3], disp_range local_trans_disp[LAYER1_SIZE][LAYER1_SIZE],
    ap_uint<DISP_DDR_BIT> local_trans_cost_tmp[LAYER1_SIZE][LAYER1_SIZE * 2], label_range local_trans_label_tmp[LAYER1_SIZE][LAYER1_SIZE * 2][3], disp_range local_trans_disp_tmp[LAYER1_SIZE][LAYER1_SIZE * 2]) {
// #pragma HLS array_partition variable=uram_cost type=complete dim=1
// #pragma HLS array_partition variable=uram_label type=complete dim=1
// #pragma HLS array_partition variable=uram_label type=complete dim=3 
// #pragma HLS array_partition variable=uram_disp type=complete dim=1
#pragma HLS array_partition variable=local_cost type=complete dim=1
#pragma HLS array_partition variable=local_label type=complete dim=1
#pragma HLS array_partition variable=local_label type=complete dim=3
#pragma HLS array_partition variable=local_disp type=complete dim=1
#pragma HLS array_partition variable=local_trans_cost type=complete dim=1
#pragma HLS array_partition variable=local_trans_label type=complete dim=1
#pragma HLS array_partition variable=local_trans_label type=complete dim=3
#pragma HLS array_partition variable=local_trans_disp type=complete dim=1
    // FILE* log_file = fopen("/home/jyc/prj/0GIT/stereo_project/prj_local_expansion_hls/prj_linux/new.log", "a");
    // if (log_file == NULL) {
    //     printf("Error: failed to create log file.\n");
    //     return;
    // }
    // fprintf(log_file, "iter=%d, %d, %d, %d\n", iter, uram_y0, uram_y1, uram_y2);
    /*  */
    int ver_region_i = iter / LAYER1_HOR_REGION_NUM;
    int hor_region_i = iter % LAYER1_HOR_REGION_NUM;
    int unit_region_x, unit_region_y, unit_region_w, unit_region_h;
    int expn_region_x, expn_region_y, expn_region_w, expn_region_h;
    genRegionInfo(WIDTH, HEIGHT, hor_region_i, ver_region_i, LAYER1_SIZE, unit_region_x, unit_region_y, unit_region_w, unit_region_h, 
        expn_region_x, expn_region_y, expn_region_w, expn_region_h);
    //
    int uram_y_addr_id = ver_region_i % 4;
    /* global ram -> local ram */
    LOOP_LOAD:
    for (int x = 0; x < LAYER1_SIZE * 3; x++) {
#pragma HLS pipeline II=1
        if ((expn_region_x + x) >= WIDTH)
            continue;
        L1_GLOBAL_LOCAL_URAM_Y
    }
    /*  */
    int tran_ver_region_i = (iter - (LAYER1_HOR_REGION_NUM + 2)) / LAYER1_HOR_REGION_NUM;
    int tran_hor_region_i = (iter - (LAYER1_HOR_REGION_NUM + 2)) % LAYER1_HOR_REGION_NUM;
    // int tran_unit_region_x, tran_unit_region_y, tran_unit_region_w, tran_unit_region_h;
    int tran_expn_region_x, tran_expn_region_y, tran_expn_region_w, tran_expn_region_h;
    genRegionInfo(WIDTH, HEIGHT, tran_hor_region_i, tran_ver_region_i, LAYER1_SIZE, tran_unit_region_x, tran_unit_region_y, tran_unit_region_w, tran_unit_region_h, 
        tran_expn_region_x, tran_expn_region_y, tran_expn_region_w, tran_expn_region_h);
    if (tran_ver_region_i < 0 || tran_hor_region_i < 0)
        return;
    // fprintf(log_file, "iter=%d, %d\n", iter, uram_y0);
    LOOP_TRANS_LOAD:
    if ((iter % LAYER1_HOR_REGION_NUM) == 0) {
        for (int x = 0; x < LAYER1_SIZE; x++) {
#pragma HLS pipeline II=1
            LOCAL_TRANS_TMP_LOAD0
        }
    }
    else if (((iter - 1) % LAYER1_HOR_REGION_NUM) == 0) {
        for (int x = 0; x < LAYER1_SIZE; x++) {
#pragma HLS pipeline II=1
            LOCAL_TRANS_TMP_LOAD1
        }
    }
    else {
        for (int x = 0; x < LAYER1_SIZE; x++) {
#pragma HLS pipeline II=1
            if ((tran_unit_region_x + x) >= WIDTH)
                continue;
            L1_GLOBAL_LOCAL_TRANS_URAM_Y
        }
    }
    // fclose(log_file);
}

void layer0UramStore(int iter, int ver_region_num, int hor_region_num, ap_uint<DISP_DDR_BIT> uram_cost[LAYER0_SIZE * 3][WIDTH], label_range uram_label[LAYER0_SIZE * 3][WIDTH][3], disp_range uram_disp[LAYER0_SIZE * 3][WIDTH],
    ap_uint<DISP_DDR_BIT> local_cost[LAYER0_SIZE * 3][LAYER0_SIZE * 3], label_range local_label[LAYER0_SIZE * 3][LAYER0_SIZE * 3][3], disp_range local_disp[LAYER0_SIZE * 3][LAYER0_SIZE * 3]
//    ap_uint<DISP_DDR_BIT> local_init_cost[LAYER0_SIZE][LAYER0_SIZE], label_range local_init_label[LAYER0_SIZE][LAYER0_SIZE][3], disp_range local_init_disp[LAYER0_SIZE][LAYER0_SIZE]
) {
    /*  */
    int ver_region_i = iter / hor_region_num;
    int hor_region_i = iter % hor_region_num;
    int unit_region_x, unit_region_y, unit_region_w, unit_region_h;
    int expn_region_x, expn_region_y, expn_region_w, expn_region_h;
    genRegionInfo(WIDTH, HEIGHT, hor_region_i, ver_region_i, LAYER0_SIZE, unit_region_x, unit_region_y, unit_region_w, unit_region_h, 
        expn_region_x, expn_region_y, expn_region_w, expn_region_h);
    /* address mapping table for uram */
    int uram_y_addr_id = ver_region_i % 3;
    int init_ver_region_i = (iter + hor_region_num + 2) / hor_region_num;
    int init_hor_region_i = (iter + hor_region_num + 2) % hor_region_num;
    int calc_ver_region_i = iter / hor_region_num;
    int calc_hor_region_i = iter % hor_region_num;
    int init_unit_region_x, init_unit_region_y, init_unit_region_w, init_unit_region_h;
    int init_expn_region_x, init_expn_region_y, init_expn_region_w, init_expn_region_h;
    genRegionInfo(WIDTH, HEIGHT, init_hor_region_i, init_ver_region_i, LAYER0_SIZE, init_unit_region_x, init_unit_region_y, init_unit_region_w, init_unit_region_h, 
        init_expn_region_x, init_expn_region_y, init_expn_region_w, init_expn_region_h);
    /*  */
    LOOP_STORE:
    for (int x = 0; x < LAYER0_SIZE * 3; x++) {
#pragma HLS pipeline II=1
        if ((expn_region_x + x) >= WIDTH)
            continue;
        if (ver_region_i == 0) {
            L0_LOCAL_GLOBAL_ADDR1
        }
        else {
            if (uram_y_addr_id == 0) {
                L0_LOCAL_GLOBAL_ADDR0
            }
            else if (uram_y_addr_id == 2) {
                L0_LOCAL_GLOBAL_ADDR2
            }
            else {
                L0_LOCAL_GLOBAL_ADDR1
            }
        }
    }
    /*  */
    for (int x = 0; x < LAYER0_SIZE; x++) {
#pragma HLS pipeline II=1
        for (int y = 0; y < LAYER0_SIZE; y++) {
#pragma HLS unroll
            if ((init_unit_region_x + x) >= WIDTH)
                continue;
            uram_cost[((init_ver_region_i % 3) * LAYER0_SIZE + y)][(init_unit_region_x + x)] = DISP_DDR_VAL_MAX;
            uram_label[((init_ver_region_i % 3) * LAYER0_SIZE + y)][(init_unit_region_x + x)][0] = 0;
            uram_label[((init_ver_region_i % 3) * LAYER0_SIZE + y)][(init_unit_region_x + x)][1] = 0;
            uram_label[((init_ver_region_i % 3) * LAYER0_SIZE + y)][(init_unit_region_x + x)][2] = 0;
            uram_disp[((init_ver_region_i % 3) * LAYER0_SIZE + y)][(init_unit_region_x + x)] = 0;
        }
    }
}

void layer0UramStoreOpt(int iter, ap_uint<8> uram_cost[LAYER0_SIZE * 3][WIDTH], ap_int<24> uram_label[LAYER0_SIZE * 3][WIDTH][3], ap_int<24> uram_disp[LAYER0_SIZE * 3][WIDTH],
    ap_uint<8> local_cost[LAYER0_SIZE * 3][LAYER0_SIZE * 3], ap_int<24> local_label[LAYER0_SIZE * 3][LAYER0_SIZE * 3][3], ap_int<24> local_disp[LAYER0_SIZE * 3][LAYER0_SIZE * 3],
    ap_uint<8> local_init_cost[LAYER0_SIZE][LAYER0_SIZE], ap_int<24> local_init_label[LAYER0_SIZE][LAYER0_SIZE][3], ap_int<24> local_init_disp[LAYER0_SIZE][LAYER0_SIZE]) {
#pragma HLS array_partition variable=uram_cost type=complete dim=1
#pragma HLS array_partition variable=uram_label type=complete dim=1
#pragma HLS array_partition variable=uram_label type=complete dim=3
#pragma HLS array_partition variable=uram_disp type=complete dim=1
#pragma HLS array_partition variable=local_cost type=complete dim=1
#pragma HLS array_partition variable=local_label type=complete dim=1
#pragma HLS array_partition variable=local_label type=complete dim=3
#pragma HLS array_partition variable=local_disp type=complete dim=1
#pragma HLS array_partition variable=local_init_cost type=complete dim=1
#pragma HLS array_partition variable=local_init_label type=complete dim=1
#pragma HLS array_partition variable=local_init_label type=complete dim=3
#pragma HLS array_partition variable=local_init_disp type=complete dim=1
    /*  */
    int ver_region_i = iter / LAYER0_HOR_REGION_NUM;
    int hor_region_i = iter % LAYER0_HOR_REGION_NUM;
    int unit_region_x, unit_region_y, unit_region_w, unit_region_h;
    int expn_region_x, expn_region_y, expn_region_w, expn_region_h;
    genRegionInfo(WIDTH, HEIGHT, hor_region_i, ver_region_i, LAYER0_SIZE, unit_region_x, unit_region_y, unit_region_w, unit_region_h, 
        expn_region_x, expn_region_y, expn_region_w, expn_region_h);
    /* address mapping table for uram */
    int uram_y_addr_id = ver_region_i % 3;
    int init_ver_region_i = (iter + LAYER0_HOR_REGION_NUM + 2) / LAYER0_HOR_REGION_NUM;
    int init_hor_region_i = (iter + LAYER0_HOR_REGION_NUM + 2) % LAYER0_HOR_REGION_NUM;
    int calc_ver_region_i = iter / LAYER0_HOR_REGION_NUM;
    int calc_hor_region_i = iter % LAYER0_HOR_REGION_NUM;
    int init_unit_region_x, init_unit_region_y, init_unit_region_w, init_unit_region_h;
    int init_expn_region_x, init_expn_region_y, init_expn_region_w, init_expn_region_h;
    genRegionInfo(WIDTH, HEIGHT, init_hor_region_i, init_ver_region_i, LAYER0_SIZE, init_unit_region_x, init_unit_region_y, init_unit_region_w, init_unit_region_h, 
        init_expn_region_x, init_expn_region_y, init_expn_region_w, init_expn_region_h);
    /*  */
    LOOP_STORE:
    for (int x = 0; x < LAYER0_SIZE * 3; x++) {
#pragma HLS pipeline II=1
        if ((expn_region_x + x) >= WIDTH)
            continue;
        if (ver_region_i == 0) {
            L0_LOCAL_GLOBAL_ADDR1
        }
        else {
            if (uram_y_addr_id == 0) {
                L0_LOCAL_GLOBAL_ADDR0
            }
            else if (uram_y_addr_id == 2) {
                L0_LOCAL_GLOBAL_ADDR2
            }
            else {
                L0_LOCAL_GLOBAL_ADDR1
            }
        }
    }
    /*  */
    for (int x = 0; x < LAYER0_SIZE; x++) {
#pragma HLS pipeline II=1
        for (int y = 0; y < LAYER0_SIZE; y++) {
#pragma HLS unroll
            if ((init_unit_region_x + x) >= WIDTH)
                continue;
            uram_cost[((init_ver_region_i % 3) * LAYER0_SIZE + y)][(init_unit_region_x + x)] = local_init_cost[y][x];
            uram_label[((init_ver_region_i % 3) * LAYER0_SIZE + y)][(init_unit_region_x + x)][0] = local_init_label[y][x][0];
            uram_label[((init_ver_region_i % 3) * LAYER0_SIZE + y)][(init_unit_region_x + x)][1] = local_init_label[y][x][1];
            uram_label[((init_ver_region_i % 3) * LAYER0_SIZE + y)][(init_unit_region_x + x)][2] = local_init_label[y][x][2];
            uram_disp[((init_ver_region_i % 3) * LAYER0_SIZE + y)][(init_unit_region_x + x)] = local_init_disp[y][x];
        }
    }
}

void layer1UramStore(int iter, int uram_y0, int uram_y1, int uram_y2, 
    ap_uint<DISP_DDR_BIT> uram_cost[4][LAYER1_SIZE][WIDTH], label_range uram_label[4][LAYER1_SIZE][WIDTH][3], disp_range uram_disp[4][LAYER1_SIZE][WIDTH],
    ap_uint<DISP_DDR_BIT> local_cost[LAYER1_SIZE * 3][LAYER1_SIZE * 3], label_range local_label[LAYER1_SIZE * 3][LAYER1_SIZE * 3][3], disp_range local_disp[LAYER1_SIZE * 3][LAYER1_SIZE * 3],
    ap_uint<DISP_DDR_BIT> local_trans_cost_tmp[LAYER1_SIZE][LAYER1_SIZE * 2], label_range local_trans_label_tmp[LAYER1_SIZE][LAYER1_SIZE * 2][3], disp_range local_trans_disp_tmp[LAYER1_SIZE][LAYER1_SIZE * 2]) {
// #pragma HLS array_partition variable=uram_cost type=complete dim=1
// #pragma HLS array_partition variable=uram_label type=complete dim=1
// #pragma HLS array_partition variable=uram_label type=complete dim=3 
// #pragma HLS array_partition variable=uram_disp type=complete dim=1
#pragma HLS array_partition variable=local_cost type=complete dim=1
#pragma HLS array_partition variable=local_label type=complete dim=1
#pragma HLS array_partition variable=local_label type=complete dim=3
#pragma HLS array_partition variable=local_disp type=complete dim=1
    //
    int ver_region_i = iter / LAYER1_HOR_REGION_NUM;
    int hor_region_i = iter % LAYER1_HOR_REGION_NUM;
    int unit_region_x, unit_region_y, unit_region_w, unit_region_h;
    int expn_region_x, expn_region_y, expn_region_w, expn_region_h;
    genRegionInfo(WIDTH, HEIGHT, hor_region_i, ver_region_i, LAYER1_SIZE, unit_region_x, unit_region_y, unit_region_w, unit_region_h, 
        expn_region_x, expn_region_y, expn_region_w, expn_region_h);
    //
    int uram_y_addr_id = ver_region_i % 4;
    /* local ram -> uram */
    LOOP_STORE:
    for (int x = 0; x < LAYER1_SIZE * 3; x++) {
#pragma HLS pipeline II=1
        if ((expn_region_x + x) >= WIDTH)
            continue;
        L1_LOCAL_GLOBAL_URAM_Y
    }
    if ((iter + 1) % LAYER1_HOR_REGION_NUM == 0) {
        for (int x = 0; x < LAYER1_SIZE * 2; x++) {
#pragma HLS pipeline II=1
            LOCAL_TRANS_TMP_STORE
        }
    }
}

void layer0InitWrapper(int iter, int ver_region_num, int hor_region_num, int buff_id, 
    /* ap_uint<32>* Q,  ap_uint<32>& I,  ap_uint<32>& C,  ap_uint<32>* Q1, ap_uint<32>& I1, ap_uint<32>& C1, 
    ap_uint<32>* Q2, ap_uint<32>& I2, ap_uint<32>& C2, ap_uint<32>* Q3, ap_uint<32>& I3, ap_uint<32>& C3, 
    ap_uint<32>* Q4, ap_uint<32>& I4, ap_uint<32>& C4, ap_uint<32>* Q5, ap_uint<32>& I5, ap_uint<32>& C5,  */
    ap_uint<DISP_DDR_BIT> local_init_cost[LAYER0_SIZE][LAYER0_SIZE], label_range local_init_label[LAYER0_SIZE][LAYER0_SIZE][3], disp_range local_init_disp[LAYER0_SIZE][LAYER0_SIZE]) {
#pragma HLS inline off
    if (iter < (ver_region_num * hor_region_num - hor_region_num - 2)) {
        layer0InloopInit(iter, ver_region_num, hor_region_num, buff_id, /* Q, I, C, Q1, I1, C1, Q2, I2, C2, Q3, I3, C3, Q4, I4, C4, Q5, I5, C5, */ 
            local_init_cost, local_init_label, local_init_disp);
    }     
}

void layer0CalcWrapper(int iter, ap_uint<4> p_x[LAYER0_PROP_NUM], ap_uint<4> p_y[LAYER0_PROP_NUM], ap_uint<32> pp[LAYER0_PROP_NUM][3], ap_uint<4> p_xx[LAYER0_PROP_NUM], ap_uint<4> p_yy[LAYER0_PROP_NUM], int ver_region_num, int hor_region_num, int unary_cost_x0, int unary_cost_x1, int unary_cost_x2,
    ap_uint<32> Q[LAYER0_PROP_NUM][4096], ap_uint<32> I[LAYER0_PROP_NUM], ap_uint<32> C[LAYER0_PROP_NUM],
    unary_cost_t local_unary_cost[LAYER0_SIZE * 3][4][LAYER0_SIZE][DISP_DDR_MAX],
    ap_uint<DISP_DDR_BIT> local_cost[LAYER0_SIZE * 3][LAYER0_SIZE * 3], label_range local_label[LAYER0_SIZE * 3][LAYER0_SIZE * 3][3], disp_range local_disp[LAYER0_SIZE * 3][LAYER0_SIZE * 3]) {
#pragma HLS inline off
    if (iter < ver_region_num * hor_region_num) {
        layer0Calc(iter, p_x, p_y, pp, p_xx, p_yy, Q, I, C, unary_cost_x0, unary_cost_x1, unary_cost_x2, local_unary_cost, local_cost, local_label, local_disp);
    }
}

void layer0TransWrapper(int iter, int ver_region_num, int hor_region_num, 
#if HLS_LAYER0_ONLY == 0
    ap_uint<DISP_DDR_BIT> ddr_cost[WIDTH * HEIGHT], label_range ddr_label[WIDTH * HEIGHT * 3], disp_range ddr_disp[WIDTH * HEIGHT],
#else
    float cost_blk_out[LAYER0_SIZE * LAYER0_SIZE], float label_blk_out[LAYER0_SIZE * LAYER0_SIZE * 3], float disp_blk_out[LAYER0_SIZE * LAYER0_SIZE], 
#endif
    ap_uint<DISP_DDR_BIT> local_trans_cost[LAYER0_SIZE][LAYER0_SIZE], label_range local_trans_label[LAYER0_SIZE][LAYER0_SIZE][3], disp_range local_trans_disp[LAYER0_SIZE][LAYER0_SIZE]
//      ap_uint<DISP_DDR_BIT> local_cost[LAYER0_SIZE * 3][LAYER0_SIZE * 3], label_range local_label[LAYER0_SIZE * 3][LAYER0_SIZE * 3][3], disp_range local_disp[LAYER0_SIZE * 3][LAYER0_SIZE * 3]
) {
#pragma HLS inline off
    static int blk_num;
    if (iter >= (hor_region_num + 2)){
        layer0Transnew(iter, ver_region_num, hor_region_num, blk_num, ddr_cost, ddr_label, ddr_disp, local_trans_cost, local_trans_label, local_trans_disp);
        blk_num++;
    }
    else {
        blk_num = 0;
    }
}

void layer0GrayReorder(int iter, int ver_region_num, int hor_region_num, int unary_cost_x0, int unary_cost_x1, int unary_cost_x2,
    uchar local_im0_gray[LAYER0_SIZE * 3][4][LAYER0_SIZE], uchar local_im0_gray_reorder[LAYER0_SIZE * 3][LAYER0_SIZE * 3]) {
    int ver_region_i = iter / hor_region_num;
    int hor_region_i = iter % hor_region_num;
    int unit_region_x, unit_region_y, unit_region_w, unit_region_h;
    int expn_region_x, expn_region_y, expn_region_w, expn_region_h;
    genRegionInfo(WIDTH, HEIGHT, hor_region_i, ver_region_i, LAYER0_SIZE, unit_region_x, unit_region_y, unit_region_w, unit_region_h, 
        expn_region_x, expn_region_y, expn_region_w, expn_region_h);
    LOOP_GRAY_REORDER:
    for (int x = 0; x < LAYER0_SIZE * 3; x++) {
#pragma HLS pipeline II=1
        int x_remainder = x % LAYER0_SIZE;
        int local_x_id = 0;
        if (expn_region_x == unit_region_x) {
            if (x < LAYER0_SIZE)
                local_x_id = unary_cost_x1;
            else if (x < LAYER0_SIZE * 2)
                local_x_id = unary_cost_x2;
        }
        else {
            if (x < LAYER0_SIZE)
                local_x_id = unary_cost_x0;
            else if (x < LAYER0_SIZE * 2)
                local_x_id = unary_cost_x1;
            else
                local_x_id = unary_cost_x2;
        }
        for (int y = 0; y < LAYER0_SIZE * 3; y++) {
#pragma HLS unroll
            local_im0_gray_reorder[y][x] = local_im0_gray[y][local_x_id][x_remainder];
        }
    }
}

void layer0LoadCalcTransWrapper(int iter, ap_uint<4> p_x[LAYER0_PROP_NUM], ap_uint<4> p_y[LAYER0_PROP_NUM], ap_uint<32> pp[LAYER0_PROP_NUM][3], ap_uint<4> p_xx[LAYER0_PROP_NUM],  ap_uint<4> p_yy[LAYER0_PROP_NUM], int ver_region_num, int hor_region_num, ap_uint<DISP_DDR_BIT> uram_cost[LAYER0_SIZE * 3][WIDTH], label_range uram_label[LAYER0_SIZE * 3][WIDTH][3], disp_range uram_disp[LAYER0_SIZE * 3][WIDTH],
    ap_uint<DISP_DDR_BIT> local_cost[LAYER0_SIZE * 3][LAYER0_SIZE * 3], label_range local_label[LAYER0_SIZE * 3][LAYER0_SIZE * 3][3], disp_range local_disp[LAYER0_SIZE * 3][LAYER0_SIZE * 3],
    ap_uint<DISP_DDR_BIT> local_trans_cost[LAYER0_SIZE][LAYER0_SIZE], label_range local_trans_label[LAYER0_SIZE][LAYER0_SIZE][3], disp_range local_trans_disp[LAYER0_SIZE][LAYER0_SIZE],
    int unary_cost_x0, int unary_cost_x1, int unary_cost_x2, ap_uint<32> Q[LAYER0_PROP_NUM][4096], ap_uint<32> I[LAYER0_PROP_NUM], ap_uint<32> C[LAYER0_PROP_NUM], 
#if HLS_LAYER0_ONLY == 0
    ap_uint<DISP_DDR_BIT> ddr_cost[WIDTH * HEIGHT], label_range ddr_label[WIDTH * HEIGHT * 3], disp_range ddr_disp[WIDTH * HEIGHT],
#else
    float cost_blk_out[LAYER0_SIZE * LAYER0_SIZE], float label_blk_out[LAYER0_SIZE * LAYER0_SIZE * 3], float disp_blk_out[LAYER0_SIZE * LAYER0_SIZE], 
#endif
    unary_cost_t local_unary_cost[LAYER0_SIZE * 3][4][LAYER0_SIZE][DISP_DDR_MAX]) {
#pragma HLS inline off
    /* load from uram */
    layer0UramLoad(iter, LAYER0_VER_REGION_NUM, LAYER0_HOR_REGION_NUM, uram_cost, uram_label, uram_disp, local_cost, local_label, local_disp, local_trans_cost, local_trans_label, local_trans_disp);
    /* calculation */
    layer0CalcWrapper(iter, p_x, p_y, pp, p_xx, p_yy, LAYER0_VER_REGION_NUM, LAYER0_HOR_REGION_NUM, unary_cost_x0, unary_cost_x1, unary_cost_x2, Q, I, C, local_unary_cost, local_cost, local_label, local_disp);
    /* transmission */
    layer0TransWrapper(iter, LAYER0_VER_REGION_NUM, LAYER0_HOR_REGION_NUM, ddr_cost, ddr_label, ddr_disp, local_trans_cost, local_trans_label, local_trans_disp);
}

void layer0InitCalcTransWrapper(int iter,  ap_uint<4> p_x[LAYER0_PROP_NUM], ap_uint<4> p_y[LAYER0_PROP_NUM], ap_uint<32> pp[LAYER0_PROP_NUM][3], ap_uint<4> p_xx[LAYER0_PROP_NUM], ap_uint<4> p_yy[LAYER0_PROP_NUM], int unary_cost_x0, int unary_cost_x1, int unary_cost_x2, int buff_id, 
    ap_uint<32> Q[LAYER0_PROP_NUM][4096], ap_uint<32> I[LAYER0_PROP_NUM], ap_uint<32> C[LAYER0_PROP_NUM], 
    unary_cost_t local_unary_cost[LAYER0_SIZE * 3][4][LAYER0_SIZE][DISP_DDR_MAX],
    ap_uint<DISP_DDR_BIT> uram_cost[LAYER0_SIZE * 3][WIDTH], label_range uram_label[LAYER0_SIZE * 3][WIDTH][3], disp_range uram_disp[LAYER0_SIZE * 3][WIDTH], 
#if HLS_LAYER0_ONLY == 0
    ap_uint<DISP_DDR_BIT> ddr_cost[WIDTH * HEIGHT], label_range ddr_label[WIDTH * HEIGHT * 3], disp_range ddr_disp[WIDTH * HEIGHT]) {
#else
    float cost_blk_out[LAYER0_SIZE * LAYER0_SIZE], float label_blk_out[LAYER0_SIZE * LAYER0_SIZE * 3], float disp_blk_out[LAYER0_SIZE * LAYER0_SIZE]) {
#endif
#pragma HLS inline off
#pragma HLS array_partition variable=uram_cost type=complete dim=1
#pragma HLS array_partition variable=uram_label type=complete dim=1
#pragma HLS array_partition variable=uram_label type=complete dim=3
#pragma HLS array_partition variable=uram_disp type=complete dim=1
// #pragma HLS array_partition variable=local_unary_cost type=complete dim=1
// #pragma HLS array_partition variable=local_unary_cost type=complete dim=2
// // DO_PRAGMA(HLS array_partition variable=local_unary_cost type=cyclic factor=DISP_DDR_NUM dim=4) 
// DO_PRAGMA(HLS array_reshape variable=local_unary_cost type=cyclic factor=DISP_DDR_NUM dim=4) 
    assert((iter >= 0) && (iter < LAYER0_VER_REGION_NUM * LAYER0_HOR_REGION_NUM + LAYER0_HOR_REGION_NUM + 2));
    // int unary_cost_x0 = (iter - 1) % 4; // calc pre 
    // int unary_cost_x1 = (iter + 0) % 4; // calc curr
    // int unary_cost_x2 = (iter + 1) % 4; // calc post
    /* local buffers */
    static ap_uint<DISP_DDR_BIT> local_trans_cost[LAYER0_SIZE][LAYER0_SIZE]; 
    static label_range local_trans_label[LAYER0_SIZE][LAYER0_SIZE][3];
    static disp_range local_trans_disp[LAYER0_SIZE][LAYER0_SIZE];
#pragma HLS array_partition variable=local_trans_cost type=complete dim=1
#pragma HLS array_partition variable=local_trans_label type=complete dim=1
#pragma HLS array_partition variable=local_trans_label type=complete dim=3
#pragma HLS array_partition variable=local_trans_disp type=complete dim=1
    static ap_uint<DISP_DDR_BIT> local_cost[LAYER0_SIZE * 3][LAYER0_SIZE * 3];
// #pragma HLS bind_storage variable=local_cost type=RAM_T2P impl=bram
    static label_range local_label[LAYER0_SIZE * 3][LAYER0_SIZE * 3][3];
    static disp_range local_disp[LAYER0_SIZE * 3][LAYER0_SIZE * 3];
#pragma HLS array_partition variable=local_cost type=complete dim=1
#pragma HLS array_partition variable=local_label type=complete dim=1
#pragma HLS array_partition variable=local_label type=complete dim=3
#pragma HLS array_partition variable=local_disp type=complete dim=1
//     uchar local_im0_gray_reorder[LAYER0_SIZE * 3][LAYER0_SIZE * 3];
// #pragma HLS array_partition variable=local_im0_gray_reorder type=complete dim=1
    // /* load from uram */
    // layer0UramLoad(iter, LAYER0_VER_REGION_NUM, LAYER0_HOR_REGION_NUM, uram_cost, uram_label, uram_disp, local_cost, local_label, local_disp, local_trans_cost, local_trans_label, local_trans_disp);
    // /* calculation */
    // layer0CalcWrapper(iter, LAYER0_VER_REGION_NUM, LAYER0_HOR_REGION_NUM, unary_cost_x0, unary_cost_x1, unary_cost_x2, Q6, I6, C6, Q7, I7, C7, Q8, I8, C8, local_unary_cost, local_cost, local_label, local_disp);
    // /* transmission */
    // layer0TransWrapper(iter, LAYER0_VER_REGION_NUM, LAYER0_HOR_REGION_NUM, cost_blk_out, label_blk_out, disp_blk_out, local_trans_cost, local_trans_label, local_trans_disp);
    layer0LoadCalcTransWrapper(iter, p_x, p_y, pp, p_xx, p_yy, LAYER0_VER_REGION_NUM, LAYER0_HOR_REGION_NUM, uram_cost, uram_label, uram_disp, local_cost, local_label, local_disp, local_trans_cost, local_trans_label, local_trans_disp,
        unary_cost_x0, unary_cost_x1, unary_cost_x2, Q, I, C, ddr_cost, ddr_label, ddr_disp, local_unary_cost);
    /* init bottom-right unit region */
//    layer0InitWrapper(iter, LAYER0_VER_REGION_NUM, LAYER0_HOR_REGION_NUM, buff_id, /* Q, I, C, Q1, I1, C1, Q2, I2, C2, Q3, I3, C3, Q4, I4, C4, Q5, I5, C5, */local_init_cost, local_init_label, local_init_disp);
    /* save to uram */
    layer0UramStore(iter, LAYER0_VER_REGION_NUM, LAYER0_HOR_REGION_NUM, uram_cost, uram_label, uram_disp, local_cost, local_label, local_disp/*, local_init_cost, local_init_label, local_init_disp*/);
}

void layer1OutloopCache(unary_cost_t local_unary_cost[LAYER1_SIZE * 3][4][LAYER1_SIZE][DISP_DDR_MAX], unary_cost_t ddr_unary_cost[WIDTH * HEIGHT * DISP_DDR_MAX],
    uchar local_im0_gray[LAYER1_SIZE * 3][4][LAYER1_SIZE], uchar ddr_im0_gray[WIDTH * HEIGHT]) {
#pragma HLS inline off
#pragma HLS array_partition variable=local_unary_cost type=complete dim=1
#pragma HLS array_partition variable=local_unary_cost type=complete dim=2
//DO_PRAGMA(HLS array_reshape variable=local_unary_cost type=cyclic factor=DISP_DDR_NUM dim=4)
#pragma HLS array_partition variable=local_im0_gray type=complete dim=1
#pragma HLS array_partition variable=local_im0_gray type=complete dim=2
// #pragma HLS interface mode=m_axi bundle=BUS_LAYER1 port=ddr_unary_cost offset=slave
// #pragma HLS interface mode=m_axi bundle=BUS_LAYER1 port=ddr_im0_gray offset=slave
    LOOP_COST0:
    for (int xyd = 0; xyd < LAYER1_SIZE * LAYER1_SIZE * DISP_DDR_MAX; xyd++) {
#pragma HLS pipeline II=1
        int y = xyd / (LAYER1_SIZE * DISP_DDR_MAX);
        int x = xyd % (LAYER1_SIZE * DISP_DDR_MAX) / DISP_DDR_MAX;
        int d = xyd % (LAYER1_SIZE * DISP_DDR_MAX) % DISP_DDR_MAX;
        local_unary_cost[y][0][x][d] = ddr_unary_cost[(y * WIDTH + x) * DISP_DDR_MAX + d];
    }

    LOOP_COST1:
    for (int xyd = 0; xyd < LAYER1_SIZE * LAYER1_SIZE * DISP_DDR_MAX; xyd++) {
#pragma HLS pipeline II=1
        int y = xyd / (LAYER1_SIZE * DISP_DDR_MAX);
        int x = xyd % (LAYER1_SIZE * DISP_DDR_MAX) / DISP_DDR_MAX;
        int d = xyd % (LAYER1_SIZE * DISP_DDR_MAX) % DISP_DDR_MAX;
        local_unary_cost[y + LAYER1_SIZE][0][x][d] = ddr_unary_cost[((y + LAYER1_SIZE) * WIDTH + x) * DISP_DDR_MAX + d];
    }

    LOOP_COST2:
    for (int xyd = 0; xyd < LAYER1_SIZE * LAYER1_SIZE * DISP_DDR_MAX; xyd++) {
#pragma HLS pipeline II=1
        int y = xyd / (LAYER1_SIZE * DISP_DDR_MAX);
        int x = xyd % (LAYER1_SIZE * DISP_DDR_MAX) / DISP_DDR_MAX;
        int d = xyd % (LAYER1_SIZE * DISP_DDR_MAX) % DISP_DDR_MAX;
        local_unary_cost[y][1][x][d] = ddr_unary_cost[(y * WIDTH + x + LAYER1_SIZE) * DISP_DDR_MAX + d];
    }

    LOOP_COST3:
    for (int xyd = 0; xyd < LAYER1_SIZE * LAYER1_SIZE * DISP_DDR_MAX; xyd++) {
#pragma HLS pipeline II=1
        int y = xyd / (LAYER1_SIZE * DISP_DDR_MAX);
        int x = xyd % (LAYER1_SIZE * DISP_DDR_MAX) / DISP_DDR_MAX;
        int d = xyd % (LAYER1_SIZE * DISP_DDR_MAX) % DISP_DDR_MAX;
        local_unary_cost[y + LAYER1_SIZE][1][x][d] = ddr_unary_cost[((y + LAYER1_SIZE) * WIDTH + x + LAYER1_SIZE) * DISP_DDR_MAX + d];
    }

    LOOP_IMG:
    for (int y = 0; y < LAYER1_SIZE; y++) {
        for (int x = 0; x < LAYER1_SIZE; x++) {
#pragma HLS loop_flatten
#pragma HLS pipeline II=1
            local_im0_gray[y][0][x] = ddr_im0_gray[y * WIDTH + x];
            local_im0_gray[y + LAYER1_SIZE][0][x] = ddr_im0_gray[(y + LAYER1_SIZE) * WIDTH + x];
            local_im0_gray[y][1][x] = ddr_im0_gray[y * WIDTH + x + LAYER1_SIZE];
            local_im0_gray[y + LAYER1_SIZE][1][x] = ddr_im0_gray[(y + LAYER1_SIZE) * WIDTH + x + LAYER1_SIZE];
        }
    }
}

void layer1InloopCache(int iter, int unary_cost_x, 
    unary_cost_t local_unary_cost[LAYER1_SIZE * 3][4][LAYER1_SIZE][DISP_DDR_MAX], unary_cost_t ddr_unary_cost[WIDTH * HEIGHT * DISP_DDR_MAX],
    uchar local_im0_gray[LAYER1_SIZE * 3][4][LAYER1_SIZE], uchar ddr_im0_gray[WIDTH * HEIGHT]) {
#pragma HLS inline off
#pragma HLS array_partition variable=local_unary_cost type=complete dim=1
#pragma HLS array_partition variable=local_unary_cost type=complete dim=2
//DO_PRAGMA(HLS array_reshape variable=local_unary_cost type=cyclic factor=DISP_DDR_NUM dim=4)
#pragma HLS array_partition variable=local_im0_gray type=complete dim=1
#pragma HLS array_partition variable=local_im0_gray type=complete dim=2
// #pragma HLS interface mode=m_axi bundle=BUS_LAYER1 port=ddr_unary_cost offset=slave
// #pragma HLS interface mode=m_axi bundle=BUS_LAYER1 port=ddr_im0_gray offset=slave
    /*  */
    assert(iter >= 0 && iter < (LAYER1_VER_REGION_NUM * LAYER1_HOR_REGION_NUM - 2));
    assert(unary_cost_x >= 0 && unary_cost_x < 4);
    //
    int ver_region_i = (iter + 2) / LAYER1_HOR_REGION_NUM;
    int hor_region_i = (iter + 2) % LAYER1_HOR_REGION_NUM;
    // 
    int unit_region_x, unit_region_y, unit_region_w, unit_region_h;
    int expn_region_x, expn_region_y, expn_region_w, expn_region_h;
    genRegionInfo(WIDTH, HEIGHT, hor_region_i, ver_region_i, LAYER1_SIZE, unit_region_x, unit_region_y, unit_region_w, unit_region_h, 
        expn_region_x, expn_region_y, expn_region_w, expn_region_h);
    // 
    // int cach_x = (iter + 2) % 4;
    // 
    LOOP_COST:
    for (int xyd = 0; xyd < LAYER1_SIZE * LAYER1_SIZE * 3 * DISP_DDR_MAX; xyd++) {
#pragma HLS pipeline II=1
        int x = xyd / (LAYER1_SIZE * 3 * DISP_DDR_MAX);
        int y = xyd % (LAYER1_SIZE * 3 * DISP_DDR_MAX) / DISP_DDR_MAX;
        int d = xyd % (LAYER1_SIZE * 3 * DISP_DDR_MAX) % DISP_DDR_MAX;
        local_unary_cost[y][unary_cost_x][x][d] = ddr_unary_cost[((expn_region_y + y) * WIDTH + (unit_region_x + x)) * DISP_DDR_MAX + d]; 
    }
    LOOP_IMAGE: 
    for (int xy = 0; xy < LAYER1_SIZE * LAYER1_SIZE * 3; xy++) {
#pragma HLS pipeline II=1
        int x = xy / (LAYER1_SIZE * 3);
        int y = xy % (LAYER1_SIZE * 3);
        local_im0_gray[y][unary_cost_x][x] = ddr_im0_gray[(expn_region_y + y) * WIDTH + (unit_region_x + x)]; 
    }
}

int layer1InitOutloop(int l0_iter, hls::stream_of_blocks<layer0_cost_blk> &cost_blk_in, hls::stream_of_blocks<layer0_label_blk> &label_blk_in, hls::stream_of_blocks<layer0_disp_blk> &disp_blk_in,
    ap_uint<DISP_DDR_BIT> uram_cost[4][LAYER1_SIZE][WIDTH], label_range uram_label[4][LAYER1_SIZE][WIDTH][3], disp_range uram_disp[4][LAYER1_SIZE][WIDTH]) {
    /*  */
    int blk_num = LAYER0_HOR_REGION_NUM * (LAYER1_SIZE / LAYER0_SIZE) * 2; // init 2 LAYER1_SIZE rows before loop starts
    // receive blocks
    LOOP_CACHE_LAYER0_OUTLOOP:
    for (int i = 0; i < blk_num; i++) {
// #pragma HLS loop_tripcount max=4
#if HLS_LAYER1_ONLY == 0
        hls::read_lock<layer0_cost_blk> cost_bin(cost_blk_in);
        hls::read_lock<layer0_label_blk> label_bin(label_blk_in);
        hls::read_lock<layer0_disp_blk> disp_bin(disp_blk_in);
#endif
        int l0_ver_region_i = (l0_iter + i) / LAYER0_HOR_REGION_NUM;
        int l0_hor_region_i = (l0_iter + i) % LAYER0_HOR_REGION_NUM;
        int l0_unit_region_x, l0_unit_region_y, l0_unit_region_w, l0_unit_region_h;
        int l0_expn_region_x, l0_expn_region_y, l0_expn_region_w, l0_expn_region_h;
        genRegionInfo(WIDTH, HEIGHT, l0_hor_region_i, l0_ver_region_i, LAYER0_SIZE, l0_unit_region_x, l0_unit_region_y, 
            l0_unit_region_w, l0_unit_region_h, l0_expn_region_x, l0_expn_region_y, l0_expn_region_w, l0_expn_region_h);
        // l0_ver_region_i % (3 * 4)
        int base_y = l0_unit_region_y % LAYER1_SIZE;
        int uram_y = l0_unit_region_y / LAYER1_SIZE;
        // fprintf(log_file, "l1_iter=%d, l0_iter=%d, i=%d, y=%d, x=%d\n", l1_iter, l0_iter, i, uram_y * 18 + base_y, l0_unit_region_x);
        for (int x = 0; x < LAYER0_SIZE; x++) {
            if ((l0_unit_region_x + x) >= WIDTH)
                continue;
            for (int y = 0; y < LAYER0_SIZE; y++) {
#pragma HLS pipeline II=1
                uram_cost[uram_y][base_y + y][(l0_unit_region_x + x)]    = cost_bin[y * LAYER0_SIZE + x];
                uram_label[uram_y][base_y + y][(l0_unit_region_x + x)][0] = label_bin[(y * LAYER0_SIZE + x) * 3 + 0];
                uram_label[uram_y][base_y + y][(l0_unit_region_x + x)][1] = label_bin[(y * LAYER0_SIZE + x) * 3 + 1];
                uram_label[uram_y][base_y + y][(l0_unit_region_x + x)][2] = label_bin[(y * LAYER0_SIZE + x) * 3 + 2];
                uram_disp[uram_y][base_y  + y][(l0_unit_region_x + x)]    = disp_bin[y * LAYER0_SIZE + x];
            }
        }
    }
    return blk_num;
}

int layer1InitInloop(int l1_iter, int l0_iter, int uram_y3, hls::stream_of_blocks<layer0_cost_blk> &cost_blk_in, hls::stream_of_blocks<layer0_label_blk> &label_blk_in, hls::stream_of_blocks<layer0_disp_blk> &disp_blk_in,
    ap_uint<DISP_DDR_BIT> uram_cost[4][LAYER1_SIZE][WIDTH], label_range uram_label[4][LAYER1_SIZE][WIDTH][3], disp_range uram_disp[4][LAYER1_SIZE][WIDTH]) {
    /*  */
    int l1_init_ver_region_i = (l1_iter + LAYER1_HOR_REGION_NUM * 2) / LAYER1_HOR_REGION_NUM;
    int l1_init_hor_region_i = (l1_iter + LAYER1_HOR_REGION_NUM * 2) % LAYER1_HOR_REGION_NUM;
    int l1_init_unit_region_x, l1_init_unit_region_y, l1_init_unit_region_w, l1_init_unit_region_h;
    int l1_init_expn_region_x, l1_init_expn_region_y, l1_init_expn_region_w, l1_init_expn_region_h;
    genRegionInfo(WIDTH, HEIGHT, l1_init_hor_region_i, l1_init_ver_region_i, LAYER1_SIZE, l1_init_unit_region_x, l1_init_unit_region_y, 
        l1_init_unit_region_w, l1_init_unit_region_h, l1_init_expn_region_x, l1_init_expn_region_y, l1_init_expn_region_w, l1_init_expn_region_h);
    int blk_num_h = ((l1_init_unit_region_h % LAYER0_SIZE) == 0) ? (l1_init_unit_region_h / LAYER0_SIZE) : (l1_init_unit_region_h / LAYER0_SIZE) + 1;
    int blk_num_w = ((l1_init_unit_region_w % LAYER0_SIZE) == 0) ? (l1_init_unit_region_w / LAYER0_SIZE) : (l1_init_unit_region_w / LAYER0_SIZE) + 1;
    int blk_num = blk_num_h * blk_num_w;
    /* receive blocks */
    if (l1_iter < (LAYER1_VER_REGION_NUM * LAYER1_HOR_REGION_NUM - 2 * LAYER1_HOR_REGION_NUM)) {
        LOOP_CACHE_LAYER0_INLOOP:
        for (int i = 0; i < blk_num; i++) {
#pragma HLS loop_tripcount max=4
#if HLS_LAYER1_ONLY == 0
            hls::read_lock<layer0_cost_blk> cost_bin(cost_blk_in);
            hls::read_lock<layer0_label_blk> label_bin(label_blk_in);
            hls::read_lock<layer0_disp_blk> disp_bin(disp_blk_in);
#endif
            int l0_ver_region_i = (l0_iter + i) / LAYER0_HOR_REGION_NUM;
            int l0_hor_region_i = (l0_iter + i) % LAYER0_HOR_REGION_NUM;
            int l0_unit_region_x, l0_unit_region_y, l0_unit_region_w, l0_unit_region_h;
            int l0_expn_region_x, l0_expn_region_y, l0_expn_region_w, l0_expn_region_h;
            genRegionInfo(WIDTH, HEIGHT, l0_hor_region_i, l0_ver_region_i, LAYER0_SIZE, l0_unit_region_x, l0_unit_region_y, 
                l0_unit_region_w, l0_unit_region_h, l0_expn_region_x, l0_expn_region_y, l0_expn_region_w, l0_expn_region_h);
            // l0_ver_region_i % (3 * 4)
            int base_y = l0_unit_region_y % LAYER1_SIZE;
            // fprintf(log_file, "l1_iter=%d, l0_iter=%d, i=%d, y=%d, x=%d\n", l1_iter, l0_iter, i, uram_y3 * 18 + base_y, l0_unit_region_x);
            // LOOP_CACHE_LAYER0:
            for (int x = 0; x < LAYER0_SIZE; x++) {
                if ((l0_unit_region_x + x) >= WIDTH)
                    continue;
                for (int y = 0; y < LAYER0_SIZE; y++) {
#pragma HLS pipeline II=1
                    uram_cost[uram_y3][base_y + y][(l0_unit_region_x + x)]    = cost_bin[y * LAYER0_SIZE + x];
                    uram_label[uram_y3][base_y + y][(l0_unit_region_x + x)][0] = label_bin[(y * LAYER0_SIZE + x) * 3 + 0];
                    uram_label[uram_y3][base_y + y][(l0_unit_region_x + x)][1] = label_bin[(y * LAYER0_SIZE + x) * 3 + 1];
                    uram_label[uram_y3][base_y + y][(l0_unit_region_x + x)][2] = label_bin[(y * LAYER0_SIZE + x) * 3 + 2];
                    uram_disp[uram_y3][base_y  + y][(l0_unit_region_x + x)]    = disp_bin[y * LAYER0_SIZE + x];
                }
            }
        }
    }
    return blk_num;
}

void layer1Calc(int iter, int ver_region_num, int hor_region_num, int unary_cost_x0, int unary_cost_x1, int unary_cost_x2,
    ap_uint<32> Q[LAYER1_PROP_NUM][4096], ap_uint<32> I[LAYER1_PROP_NUM], ap_uint<32> C[LAYER1_PROP_NUM],
    unary_cost_t local_unary_cost[LAYER1_SIZE * 3][4][LAYER1_SIZE][DISP_DDR_MAX], uchar local_im0_gray[LAYER1_SIZE * 3][4][LAYER1_SIZE], 
    ap_uint<DISP_DDR_BIT> local_cost[LAYER1_SIZE * 3][LAYER1_SIZE * 3], label_range local_label[LAYER1_SIZE * 3][LAYER1_SIZE * 3][3], disp_range local_disp[LAYER1_SIZE * 3][LAYER1_SIZE * 3]) {
#pragma HLS inline off
#pragma HLS array_partition variable=local_cost type=complete dim=1
#pragma HLS array_partition variable=local_label type=complete dim=1
#pragma HLS array_partition variable=local_label type=complete dim=3
#pragma HLS array_partition variable=local_disp type=complete dim=1
#pragma HLS array_partition variable=local_unary_cost type=complete dim=1
#pragma HLS array_partition variable=local_unary_cost type=complete dim=2
//DO_PRAGMA(HLS array_reshape variable=local_unary_cost type=cyclic factor=DISP_DDR_NUM dim=4) 
#pragma HLS array_partition variable=local_im0_gray type=complete dim=1
#pragma HLS array_partition variable=local_im0_gray type=complete dim=2
    //
    label_range plane_a[LAYER1_PROP_NUM], plane_b[LAYER1_PROP_NUM], plane_c[LAYER1_PROP_NUM];
#pragma HLS array_partition variable=plane_a type=complete dim=1
#pragma HLS array_partition variable=plane_b type=complete dim=1
#pragma HLS array_partition variable=plane_c type=complete dim=1
    //
    int ver_region_i = iter / LAYER1_HOR_REGION_NUM;
    int hor_region_i = iter % LAYER1_HOR_REGION_NUM;
    int unit_region_x, unit_region_y, unit_region_w, unit_region_h;
    int expn_region_x, expn_region_y, expn_region_w, expn_region_h;
    genRegionInfo(WIDTH, HEIGHT, hor_region_i, ver_region_i, LAYER1_SIZE, unit_region_x, unit_region_y, unit_region_w, unit_region_h, 
        expn_region_x, expn_region_y, expn_region_w, expn_region_h);
#if LAYER1_ENABLE 
    /* gray image re-order */
    uchar local_im0_gray_reorder[LAYER1_SIZE * 3][LAYER1_SIZE * 3];
#pragma HLS array_partition variable=local_im0_gray type=complete dim=1
    LOOP_IMG_REORDER:
    for (int x = 0; x < LAYER1_SIZE * 3; x++) {
#pragma HLS pipeline II=1
        int x_remainder = x % LAYER1_SIZE;
        int local_x_id = 0;
        if (expn_region_x == unit_region_x) {
            if (x < LAYER1_SIZE)
                local_x_id = unary_cost_x1;
            else if (x < LAYER1_SIZE * 2)
                local_x_id = unary_cost_x2;
        }
        else {
            if (x < LAYER1_SIZE)
                local_x_id = unary_cost_x0;
            else if (x < LAYER1_SIZE * 2)
                local_x_id = unary_cost_x1;
            else
                local_x_id = unary_cost_x2;
        }
        for (int y = 0; y < LAYER1_SIZE * 3; y++) {
#pragma HLS unroll
            local_im0_gray_reorder[y][x] = local_im0_gray[y][local_x_id][x_remainder];
        }
    }
    /* process local ram */
    LOOP_PROPOSAL:
    for (int inner_iter = 0; inner_iter < LAYER1_PROP_LOOP; inner_iter++) {
#pragma HLS pipeline off
        layer1ProposalParall(ver_region_i, Q, I, C, inner_iter, expn_region_x, expn_region_y, unit_region_x, unit_region_y, unit_region_w, unit_region_h,
            local_label, plane_a, plane_b, plane_c);
        layer1CostLabelUpdate(unary_cost_x0, unary_cost_x1, unary_cost_x2, expn_region_x, expn_region_y, expn_region_w, expn_region_h, unit_region_x, plane_a, plane_b, plane_c, 
            local_unary_cost, local_im0_gray_reorder, local_cost, local_label, local_disp);
    }
#endif
}

void layer1CalcWrapper(int l1_iter, int unary_cost_x0, int unary_cost_x1, int unary_cost_x2, 
    ap_uint<32> Q[LAYER1_PROP_NUM][4096], ap_uint<32> I[LAYER1_PROP_NUM], ap_uint<32> C[LAYER1_PROP_NUM],
    unary_cost_t local_unary_cost[LAYER1_SIZE * 3][4][LAYER1_SIZE][DISP_DDR_MAX], uchar local_im0_gray[LAYER1_SIZE * 3][4][LAYER1_SIZE], 
    ap_uint<DISP_DDR_BIT> local_cost[LAYER1_SIZE * 3][LAYER1_SIZE * 3], label_range local_label[LAYER1_SIZE * 3][LAYER1_SIZE * 3][3], disp_range local_disp[LAYER1_SIZE * 3][LAYER1_SIZE * 3]) {
#pragma HLS inline off
    /*  */
    if ((l1_iter >= 0) && (l1_iter < LAYER1_VER_REGION_NUM * LAYER1_HOR_REGION_NUM)) {
        layer1Calc(l1_iter, LAYER1_VER_REGION_NUM, LAYER1_HOR_REGION_NUM, unary_cost_x0, unary_cost_x1, unary_cost_x2, Q, I, C, local_unary_cost, local_im0_gray, local_cost, local_label, local_disp);
    }
}

void layer2OutloopCache(float local_unary_cost[LAYER2_SIZE * 3][4][LAYER2_SIZE][DISP_MAX], unary_cost_t ddr_unary_cost[WIDTH * HEIGHT * DISP_DDR_MAX],
    uchar local_im0_gray[LAYER2_SIZE * 3][4][LAYER2_SIZE], uchar ddr_im0_gray[WIDTH * HEIGHT]) {
#pragma HLS inline off
// #pragma HLS array_partition variable=local_unary_cost type=complete dim=1
// #pragma HLS interface mode=m_axi bundle=BUS_LAYER0 port=ddr_unary_cost offset=slave
    for (int y = 0; y < LAYER2_SIZE; y++) {
        for (int x = 0; x < LAYER2_SIZE; x++) {
            for (int d_ddr = 0; d_ddr < DISP_DDR_MAX; d_ddr++) {
#pragma HLS pipeline II=1
                for (int d = 0; d < DISP_DDR_NUM; d++) {
                    if (d_ddr * DISP_DDR_NUM + d >= DISP_MAX)
                        continue;
                    local_unary_cost[y][0][x][d_ddr * DISP_DDR_NUM + d] = ddr_unary_cost[(y * WIDTH + x) * DISP_DDR_MAX + d_ddr](d * DISP_DDR_BIT + DISP_DDR_BIT -1, d * DISP_DDR_BIT);
                    local_unary_cost[y + LAYER2_SIZE][0][x][d_ddr * DISP_DDR_NUM + d] = ddr_unary_cost[((y + LAYER2_SIZE) * WIDTH + x) * DISP_DDR_MAX + d_ddr](d * DISP_DDR_BIT + DISP_DDR_BIT -1, d * DISP_DDR_BIT);
                    local_unary_cost[y][1][x][d_ddr * DISP_DDR_NUM + d] = ddr_unary_cost[(y * WIDTH + x + LAYER2_SIZE) * DISP_DDR_MAX + d_ddr](d * DISP_DDR_BIT + DISP_DDR_BIT -1, d * DISP_DDR_BIT);
                    local_unary_cost[y + LAYER2_SIZE][1][x][d_ddr * DISP_DDR_NUM + d] = ddr_unary_cost[((y + LAYER2_SIZE) * WIDTH + x + LAYER2_SIZE) * DISP_DDR_MAX + d_ddr](d * DISP_DDR_BIT + DISP_DDR_BIT -1, d * DISP_DDR_BIT);
                }
            }
        }
    }
    for (int y = 0; y < LAYER2_SIZE; y++) {
        for (int x = 0; x < LAYER2_SIZE; x++) {
            local_im0_gray[y][0][x] = ddr_im0_gray[y * WIDTH + x];
            local_im0_gray[y + LAYER2_SIZE][0][x] = ddr_im0_gray[(y + LAYER2_SIZE) * WIDTH + x];
            local_im0_gray[y][1][x] = ddr_im0_gray[y * WIDTH + x + LAYER2_SIZE];
            local_im0_gray[y + LAYER2_SIZE][1][x] = ddr_im0_gray[(y + LAYER2_SIZE) * WIDTH + x + LAYER2_SIZE];
        }
    }
}

void layer2InloopCache(int iter, int ver_region_num, int hor_region_num, int unary_cost_x, 
    float local_unary_cost[LAYER2_SIZE * 3][4][LAYER2_SIZE][DISP_MAX], unary_cost_t ddr_unary_cost[WIDTH * HEIGHT * DISP_DDR_MAX],
    uchar local_im0_gray[LAYER2_SIZE * 3][4][LAYER2_SIZE], uchar ddr_im0_gray[WIDTH * HEIGHT]) {
#pragma HLS inline off
#pragma HLS array_partition variable=local_unary_cost type=complete dim=1
#pragma HLS array_partition variable=local_unary_cost type=complete dim=2
DO_PRAGMA(HLS array_reshape variable=local_unary_cost type=cyclic factor=DISP_DDR_NUM dim=4)
#pragma HLS array_partition variable=local_im0_gray type=complete dim=1
#pragma HLS array_partition variable=local_im0_gray type=complete dim=2
#pragma HLS interface mode=m_axi bundle=BUS_LAYER2 port=ddr_unary_cost offset=slave
#pragma HLS interface mode=m_axi bundle=BUS_LAYER2 port=ddr_im0_gray offset=slave
    assert(iter >= 0 && iter < (LAYER2_VER_REGION_NUM * LAYER2_HOR_REGION_NUM - 2));
    assert(unary_cost_x >= 0 && unary_cost_x < 4);
    //
    int ver_region_i = (iter + 2) / LAYER2_HOR_REGION_NUM;
    int hor_region_i = (iter + 2) % LAYER2_HOR_REGION_NUM;
    // 
    int unit_region_x, unit_region_y, unit_region_w, unit_region_h;
    int expn_region_x, expn_region_y, expn_region_w, expn_region_h;
    genRegionInfo(WIDTH, HEIGHT, hor_region_i, ver_region_i, LAYER2_SIZE, unit_region_x, unit_region_y, unit_region_w, unit_region_h, 
        expn_region_x, expn_region_y, expn_region_w, expn_region_h);
    // 
    // int cach_x = (iter + 2) % 4;
    // 
    for (int y = 0; y < LAYER2_SIZE * 3; y++) {
        for (int x = 0; x < LAYER2_SIZE; x++) {
            // if ((x >= unit_region_w) || (y >= expn_region_h))
            //     continue;
            for (int d_ddr = 0; d_ddr < DISP_DDR_MAX; d_ddr++) {
#pragma HLS pipeline II=1
			    for (int d = 0; d < DISP_DDR_NUM; d++) {
                    if ((x < unit_region_w) && (y < expn_region_h) && (d_ddr * DISP_DDR_NUM + d < DISP_MAX))
                        local_unary_cost[y][unary_cost_x][x][d_ddr * DISP_DDR_NUM + d] = ddr_unary_cost[((expn_region_y + y) * WIDTH + (unit_region_x + x)) * DISP_DDR_MAX + d_ddr](d * DISP_DDR_BIT + DISP_DDR_BIT -1, d * DISP_DDR_BIT);            
                }
            }
        }
    }
    for (int y = 0; y < LAYER2_SIZE * 3; y++) {
        for (int x = 0; x < LAYER2_SIZE; x++) {
#pragma HLS pipeline II=1
            // if ((x >= unit_region_w) || (y >= expn_region_h))
            //     continue;
            if ((x < unit_region_w) && (y < expn_region_h))
                local_im0_gray[y][unary_cost_x][x] = ddr_im0_gray[(expn_region_y + y) * WIDTH + (unit_region_x + x)];
        }
    }
}

void layer2InloopCacheHls(int iter, int unary_cost_x, 
    float local_unary_cost[LAYER2_SIZE * 3][4][LAYER2_SIZE][DISP_DDR_MAX * DISP_DDR_NUM], unary_cost_t ddr_unary_cost[WIDTH * HEIGHT * DISP_DDR_MAX],
    uchar local_im0_gray[LAYER2_SIZE * 3][4][LAYER2_SIZE], uchar ddr_im0_gray[WIDTH * HEIGHT]) {
#pragma HLS inline off
#pragma HLS array_partition variable=local_unary_cost type=block factor=54 dim=1
#pragma HLS array_partition variable=local_unary_cost type=complete dim=2
DO_PRAGMA(HLS array_reshape variable=local_unary_cost type=cyclic factor=DISP_DDR_NUM dim=4)
#pragma HLS array_partition variable=local_im0_gray type=block factor=54 dim=1
#pragma HLS array_partition variable=local_im0_gray type=complete dim=2
#pragma HLS interface mode=m_axi bundle=BUS_LAYER2 port=ddr_unary_cost offset=slave
#pragma HLS interface mode=m_axi bundle=BUS_LAYER2 port=ddr_im0_gray offset=slave
    assert(iter >= 0 && iter < (LAYER2_VER_REGION_NUM * LAYER2_HOR_REGION_NUM - 2));
    assert(unary_cost_x >= 0 && unary_cost_x < 4);
    //
    int ver_region_i = (iter + 2) / LAYER2_HOR_REGION_NUM;
    int hor_region_i = (iter + 2) % LAYER2_HOR_REGION_NUM;
    // 
    int unit_region_x, unit_region_y, unit_region_w, unit_region_h;
    int expn_region_x, expn_region_y, expn_region_w, expn_region_h;
    genRegionInfo(WIDTH, HEIGHT, hor_region_i, ver_region_i, LAYER2_SIZE, unit_region_x, unit_region_y, unit_region_w, unit_region_h, 
        expn_region_x, expn_region_y, expn_region_w, expn_region_h);
    // 
    // int cach_x = (iter + 2) % 4;
    // 
    LOOP_COST_EVEN:
    for (int x = 0; x < LAYER2_SIZE; x++) {
        for (int y = 0; y < LAYER2_SIZE * 3; y = y + 2) {
            // if ((x >= unit_region_w) || (y >= expn_region_h))
            //     continue;
            for (int d_ddr = 0; d_ddr < DISP_DDR_MAX; d_ddr++) {
#pragma HLS loop_flatten
#pragma HLS pipeline II=1
			    for (int d = 0; d < DISP_DDR_NUM; d++) {
#pragma HLS unroll
                    local_unary_cost[y][unary_cost_x][x][d_ddr * DISP_DDR_NUM + d] = ddr_unary_cost[((expn_region_y + y) * WIDTH + (unit_region_x + x)) * DISP_DDR_MAX + d_ddr](d * DISP_DDR_BIT + DISP_DDR_BIT -1, d * DISP_DDR_BIT);            
                }
            }
        }
    }
    LOOP_COST_ODD:
    for (int x = 0; x < LAYER2_SIZE; x++) {
        for (int y = 1; y < LAYER2_SIZE * 3; y = y + 2) {
            // if ((x >= unit_region_w) || (y >= expn_region_h))
            //     continue;
            for (int d_ddr = 0; d_ddr < DISP_DDR_MAX; d_ddr++) {
#pragma HLS loop_flatten
#pragma HLS pipeline II=1
			    for (int d = 0; d < DISP_DDR_NUM; d++) {
#pragma HLS unroll
                    local_unary_cost[y][unary_cost_x][x][d_ddr * DISP_DDR_NUM + d] = ddr_unary_cost[((expn_region_y + y) * WIDTH + (unit_region_x + x)) * DISP_DDR_MAX + d_ddr](d * DISP_DDR_BIT + DISP_DDR_BIT -1, d * DISP_DDR_BIT);            
                }
            }
        }
    }
    LOOP_IMAGE: 
    for (int x = 0; x < LAYER2_SIZE; x++) {
        for (int y = 0; y < LAYER2_SIZE * 3; y = y + 2) {
#pragma HLS loop_flatten
#pragma HLS pipeline II=1
            if ((x < unit_region_w) && (y < expn_region_h))
                local_im0_gray[y][unary_cost_x][x] = ddr_im0_gray[(expn_region_y + y) * WIDTH + (unit_region_x + x)];
        }
    }
}

void layer2Init(int& l1_iter, int l2_iter, int l1_hor_region_num, int l1_ver_region_num, int l2_hor_region_num, int l2_ver_region_num,
#if HLS_LAYER2_ONLY == 0
    hls::stream_of_blocks<layer1_cost_blk> &cost_blk_in, hls::stream_of_blocks<layer1_label_blk> &label_blk_in, hls::stream_of_blocks<layer1_disp_blk> &disp_blk_in,
#else
    float cost_bin[LAYER1_SIZE * LAYER1_SIZE], float label_bin[LAYER1_SIZE * LAYER1_SIZE * 3], float disp_bin[LAYER1_SIZE * LAYER1_SIZE],
#endif
    float uram_cost[LAYER2_SIZE * 4][WIDTH], float uram_label[LAYER2_SIZE * 4][WIDTH][3], float uram_disp[LAYER2_SIZE * 4][WIDTH]) {
#pragma HLS inline off

    int blk_num;
    // generate receiving block number
    if (l2_iter == -1) {
        blk_num = LAYER1_HOR_REGION_NUM * (LAYER2_SIZE / LAYER1_SIZE) * 2; // init 2 LAYER2_SIZE rows before loop starts
    }
    else {
        int l2_init_ver_region_i = (l2_iter + LAYER2_HOR_REGION_NUM * 2) / LAYER2_HOR_REGION_NUM;
        int l2_init_hor_region_i = (l2_iter + LAYER2_HOR_REGION_NUM * 2) % LAYER2_HOR_REGION_NUM;
        int l2_init_unit_region_x, l2_init_unit_region_y, l2_init_unit_region_w, l2_init_unit_region_h;
        int l2_init_expn_region_x, l2_init_expn_region_y, l2_init_expn_region_w, l2_init_expn_region_h;
        genRegionInfo(WIDTH, HEIGHT, l2_init_hor_region_i, l2_init_ver_region_i, LAYER2_SIZE, l2_init_unit_region_x, l2_init_unit_region_y, 
            l2_init_unit_region_w, l2_init_unit_region_h, l2_init_expn_region_x, l2_init_expn_region_y, l2_init_expn_region_w, l2_init_expn_region_h);
        int blk_num_h = ((l2_init_unit_region_h % LAYER1_SIZE) == 0) ? (l2_init_unit_region_h / LAYER1_SIZE) : (l2_init_unit_region_h / LAYER1_SIZE) + 1;
        int blk_num_w = ((l2_init_unit_region_w % LAYER1_SIZE) == 0) ? (l2_init_unit_region_w / LAYER1_SIZE) : (l2_init_unit_region_w / LAYER1_SIZE) + 1;
        blk_num = blk_num_h * blk_num_w;
    }
    // receive blocks
    for (int i = 0; i < blk_num; i++) {
#pragma HLS loop_tripcount max=4
#if HLS_LAYER2_ONLY == 0
        hls::read_lock<layer1_cost_blk> cost_bin(cost_blk_in);
        hls::read_lock<layer1_label_blk> label_bin(label_blk_in);
        hls::read_lock<layer1_disp_blk> disp_bin(disp_blk_in);
#endif
        int l1_ver_region_i = (l1_iter + i) / LAYER1_HOR_REGION_NUM;
        int l1_hor_region_i = (l1_iter + i) % LAYER1_HOR_REGION_NUM;
        int l1_unit_region_x, l1_unit_region_y, l1_unit_region_w, l1_unit_region_h;
        int l1_expn_region_x, l1_expn_region_y, l1_expn_region_w, l1_expn_region_h;
        genRegionInfo(WIDTH, HEIGHT, l1_hor_region_i, l1_ver_region_i, LAYER1_SIZE, l1_unit_region_x, l1_unit_region_y, 
            l1_unit_region_w, l1_unit_region_h, l1_expn_region_x, l1_expn_region_y, l1_expn_region_w, l1_expn_region_h);
        // l0_ver_region_i % (3 * 4)
        int uram_y = (l1_ver_region_i % ((LAYER2_SIZE / LAYER1_SIZE) * 4)) * LAYER1_SIZE;
        for (int x = 0; x < LAYER1_SIZE; x++) {
            if ((l1_unit_region_x + x) >= WIDTH)
                continue;
            for (int y = 0; y < LAYER1_SIZE; y++) {
#pragma HLS pipeline II=1
                uram_cost[uram_y  + y][(l1_unit_region_x + x)]    = cost_bin[y * LAYER1_SIZE + x];
                uram_label[uram_y + y][(l1_unit_region_x + x)][0] = label_bin[(y * LAYER1_SIZE + x) * 3 + 0];
                uram_label[uram_y + y][(l1_unit_region_x + x)][1] = label_bin[(y * LAYER1_SIZE + x) * 3 + 1];
                uram_label[uram_y + y][(l1_unit_region_x + x)][2] = label_bin[(y * LAYER1_SIZE + x) * 3 + 2];
                uram_disp[uram_y  + y][(l1_unit_region_x + x)]    = disp_bin[y * LAYER1_SIZE + x];
            }
        }
    }
    l1_iter += blk_num;
}

void layer2Calc(int iter, int ver_region_num, int hor_region_num, int unary_cost_x0, int unary_cost_x1, int unary_cost_x2,
    ap_uint<32>* Q, ap_uint<32>& I, ap_uint<32>& C,
    float local_unary_cost[LAYER2_SIZE * 3][4][LAYER2_SIZE][DISP_MAX], uchar local_im0_gray[LAYER2_SIZE * 3][4][LAYER2_SIZE],
    float uram_cost[LAYER2_SIZE * 4][WIDTH], float uram_label[LAYER2_SIZE * 4][WIDTH][3], float uram_disp[LAYER2_SIZE * 4][WIDTH]){
#pragma HLS inline off
// #pragma HLS array_partition variable=local_unary_cost type=complete dim=1
// #pragma HLS array_partition variable=local_unary_cost type=complete dim=2
// #pragma HLS array_partition variable=uram_cost type=complete dim=1
// #pragma HLS array_partition variable=uram_label type=complete dim=1
// #pragma HLS array_partition variable=uram_label type=complete dim=3
    //
    float local_cost[LAYER2_SIZE * 3][LAYER2_SIZE * 3];
    float local_label[LAYER2_SIZE * 3][LAYER2_SIZE * 3][3];
    float local_disp[LAYER2_SIZE * 3][LAYER2_SIZE * 3];
// #pragma HLS array_partition variable=local_cost type=complete dim=1
// #pragma HLS array_partition variable=local_label type=complete dim=1
// #pragma HLS array_partition variable=local_label type=complete dim=3
// #pragma HLS array_partition variable=local_disp type=complete dim=1
    ap_fixed<32, 15, AP_RND, AP_SAT> plane_a, plane_b, plane_c;
    //
    int ver_region_i = iter / hor_region_num;
    int hor_region_i = iter % hor_region_num;
    int unit_region_x, unit_region_y, unit_region_w, unit_region_h;
    int expn_region_x, expn_region_y, expn_region_w, expn_region_h;
    genRegionInfo(WIDTH, HEIGHT, hor_region_i, ver_region_i, LAYER2_SIZE, unit_region_x, unit_region_y, unit_region_w, unit_region_h, 
        expn_region_x, expn_region_y, expn_region_w, expn_region_h);
    //
    int uram_y_addr_id = ver_region_i % 4;
    // global ram -> local ram
    LOOP_LOAD:
    for (int x = 0; x < LAYER2_SIZE * 3; x++) {
#pragma HLS pipeline II=1
        if ((expn_region_x + x) >= WIDTH)
            continue;
        if (ver_region_i == 0) {
            L2_GLOBAL_LOCAL_ADDR1
        }
        else {
            if (uram_y_addr_id == 0) {
                L2_GLOBAL_LOCAL_ADDR0
            }
            else if (uram_y_addr_id == 1) {
                L2_GLOBAL_LOCAL_ADDR1
            }
            else if (uram_y_addr_id == 2) {
                L2_GLOBAL_LOCAL_ADDR2
            }
            else if (uram_y_addr_id == 3) {
                L2_GLOBAL_LOCAL_ADDR3
            }
        }
    }
#if LAYER2_ENABLE
    /* gray image re-order */
    uchar local_im0_gray_reorder[LAYER2_SIZE * 3][LAYER2_SIZE * 3];
#pragma HLS array_partition variable=local_im0_gray type=complete dim=1
    LOOP_GRAY_REORDER:
    for (int x = 0; x < LAYER2_SIZE * 3; x++) {
#pragma HLS pipeline II=1
        int x_remainder = x % LAYER2_SIZE;
        int local_x_id = 0;
        if (expn_region_x == unit_region_x) {
            if (x < LAYER2_SIZE)
                local_x_id = unary_cost_x1;
            else if (x < LAYER2_SIZE * 2)
                local_x_id = unary_cost_x2;
        }
        else {
            if (x < LAYER2_SIZE)
                local_x_id = unary_cost_x0;
            else if (x < LAYER2_SIZE * 2)
                local_x_id = unary_cost_x1;
            else
                local_x_id = unary_cost_x2;
        }
        for (int y = 0; y < LAYER2_SIZE * 3; y++) {
#pragma HLS unroll
            local_im0_gray_reorder[y][x] = local_im0_gray[y][local_x_id][x_remainder];
        }
    }
    // process local ram
    LOOP_PROPOSAL:
    for (int inner_iter = 0; inner_iter < 8; inner_iter++) {
#pragma HLS pipeline off
        layerProposal<LAYER2_SIZE>(ver_region_i, Q, I, C, 0, inner_iter, expn_region_x, expn_region_y, unit_region_x, unit_region_y, unit_region_w, unit_region_h, 
            local_label, plane_a, plane_b, plane_c);     
        costLabelUpdate<LAYER2_SIZE>(unary_cost_x0, unary_cost_x1, unary_cost_x2, expn_region_x, expn_region_y, expn_region_w, expn_region_h, unit_region_x, plane_a, plane_b, plane_c, 
            local_unary_cost, local_im0_gray_reorder, local_cost, local_label, local_disp, true, LAYER2_GC, (inner_iter == 0) && (iter == 52) && 0);
    }
#endif
    // local ram -> uram
    LOOP_STORE:
    for (int x = 0; x < LAYER2_SIZE * 3; x++) {
#pragma HLS pipeline II=1
        if ((expn_region_x + x) >= WIDTH)
            continue;
        if (ver_region_i == 0) {
            L2_LOCAL_GLOBAL_ADDR1
        }
        else {
            if (uram_y_addr_id == 0) {
                L2_LOCAL_GLOBAL_ADDR0
            }
            else if (uram_y_addr_id == 1) {
                L2_LOCAL_GLOBAL_ADDR1
            }
            else if (uram_y_addr_id == 2) {
                L2_LOCAL_GLOBAL_ADDR2
            }
            else if (uram_y_addr_id == 3) {
                L2_LOCAL_GLOBAL_ADDR3
            }
        }
    }
}

void layer2Trans(int l2_iter, int l2_ver_region_num, int l2_hor_region_num, 
    float uram_cost[LAYER2_SIZE * 4][WIDTH], float uram_label[LAYER2_SIZE * 4][WIDTH][3], float uram_disp[LAYER2_SIZE * 4][WIDTH],
    float ddr_cost[WIDTH * HEIGHT], float ddr_label[WIDTH * HEIGHT * 3], float ddr_disp[WIDTH * HEIGHT]) {
#pragma HLS inline off
    // 
    int tran_ver_region_i = (l2_iter - (l2_hor_region_num + 1)) / l2_hor_region_num;
    int tran_hor_region_i = (l2_iter - (l2_hor_region_num + 1)) % l2_hor_region_num;
    int tran_unit_region_x, tran_unit_region_y, tran_unit_region_w, tran_unit_region_h;
    int tran_expn_region_x, tran_expn_region_y, tran_expn_region_w, tran_expn_region_h;
    genRegionInfo(WIDTH, HEIGHT, tran_hor_region_i, tran_ver_region_i, LAYER2_SIZE, tran_unit_region_x, tran_unit_region_y, tran_unit_region_w, tran_unit_region_h, 
        tran_expn_region_x, tran_expn_region_y, tran_expn_region_w, tran_expn_region_h);
    // 
    LOOP7:
    for (int x = 0; x < LAYER2_SIZE; x++) {
        LOOP8:
        for (int y = 0; y < LAYER2_SIZE; y++) {
#pragma HLS pipeline II=1
            if ((x >= tran_unit_region_w) || (y >= tran_unit_region_h))
                continue;
            ddr_cost[(tran_unit_region_y + y) * WIDTH + tran_unit_region_x + x] = uram_cost[(tran_ver_region_i % 4) * LAYER2_SIZE + y][tran_unit_region_x + x];
            ddr_label[((tran_unit_region_y + y) * WIDTH + tran_unit_region_x + x) * 3 + 0] = uram_label[(tran_ver_region_i % 4) * LAYER2_SIZE + y][tran_unit_region_x + x][0];
            ddr_label[((tran_unit_region_y + y) * WIDTH + tran_unit_region_x + x) * 3 + 1] = uram_label[(tran_ver_region_i % 4) * LAYER2_SIZE + y][tran_unit_region_x + x][1];
            ddr_label[((tran_unit_region_y + y) * WIDTH + tran_unit_region_x + x) * 3 + 2] = uram_label[(tran_ver_region_i % 4) * LAYER2_SIZE + y][tran_unit_region_x + x][2];
            // ddr_disp[(tran_unit_region_y + y) * WIDTH + tran_unit_region_x + x] = 
            //     uram_label[(tran_ver_region_i % 4) * LAYER2_SIZE + y][tran_unit_region_x + x][0] * (tran_unit_region_x + x) + 
            //     uram_label[(tran_ver_region_i % 4) * LAYER2_SIZE + y][tran_unit_region_x + x][1] * (tran_unit_region_y + y) + 
            //     uram_label[(tran_ver_region_i % 4) * LAYER2_SIZE + y][tran_unit_region_x + x][2];
            ddr_disp[(tran_unit_region_y + y) * WIDTH + tran_unit_region_x + x] = uram_disp[(tran_ver_region_i % 4) * LAYER2_SIZE + y][tran_unit_region_x + x];
        }
    }
}

void layer1CacheWrapper(int l1_iter, int unary_cost_x, 
    unary_cost_t local_unary_cost[LAYER1_SIZE * 3][4][LAYER1_SIZE][DISP_DDR_MAX], unary_cost_t ddr_unary_cost[WIDTH * HEIGHT * DISP_DDR_MAX],
    uchar local_im0_gray[LAYER1_SIZE * 3][4][LAYER1_SIZE], uchar ddr_im0_gray[WIDTH * HEIGHT]) {
#pragma HLS inline off
    // cache unary & binary cost from DDR
    if ((l1_iter >= 0) && (l1_iter < (LAYER1_VER_REGION_NUM * LAYER1_HOR_REGION_NUM - 2))) {
        layer1InloopCache(l1_iter, unary_cost_x, local_unary_cost, ddr_unary_cost, local_im0_gray, ddr_im0_gray);
    }
}

void layer1CalcTransWrapper(int& l0_iter, int l1_iter, int unary_cost_x0, int unary_cost_x1, int unary_cost_x2, int uram_y0, int uram_y1, int uram_y2, int uram_y3, 
    ap_uint<32> Q[LAYER1_PROP_NUM][4096], ap_uint<32> I[LAYER1_PROP_NUM], ap_uint<32> C[LAYER1_PROP_NUM],
    unary_cost_t local_unary_cost[LAYER1_SIZE * 3][4][LAYER1_SIZE][DISP_DDR_MAX], uchar local_im0_gray[LAYER1_SIZE * 3][4][LAYER1_SIZE], 
    ap_uint<DISP_DDR_BIT> uram_cost[4][LAYER1_SIZE][WIDTH], label_range uram_label[4][LAYER1_SIZE][WIDTH][3], disp_range uram_disp[4][LAYER1_SIZE][WIDTH],
#if HLS_LAYER1_ONLY == 0
    hls::stream_of_blocks<layer0_cost_blk> &cost_blk_in, hls::stream_of_blocks<layer0_label_blk> &label_blk_in, hls::stream_of_blocks<layer0_disp_blk> &disp_blk_in,
    /* hls::stream_of_blocks<layer0_cost_blk> &init_cost_blk_in, hls::stream_of_blocks<layer0_label_blk> &init_label_blk_in, hls::stream_of_blocks<layer0_disp_blk> &init_disp_blk_in, */
#else
    float cost_blk_in[LAYER0_SIZE * LAYER0_SIZE], float label_blk_in[LAYER0_SIZE * LAYER0_SIZE * 3], float disp_blk_in[LAYER0_SIZE * LAYER0_SIZE],
#endif
#if LAYER2_ENABLE == 0
    ap_uint<DISP_DDR_BIT> ddr_cost[WIDTH * HEIGHT], label_range ddr_label[WIDTH * HEIGHT * 3], disp_range ddr_disp[WIDTH * HEIGHT]) {
#elif HLS_LAYER1_ONLY == 0
    hls::stream_of_blocks<layer1_cost_blk> &cost_blk_out, hls::stream_of_blocks<layer1_label_blk> &label_blk_out, hls::stream_of_blocks<layer1_disp_blk> &disp_blk_out) {
#else
    float cost_blk_out[LAYER1_SIZE * LAYER1_SIZE], float label_blk_out[LAYER1_SIZE * LAYER1_SIZE * 3], float disp_blk_out[LAYER1_SIZE * LAYER1_SIZE]) {
#endif
#pragma HLS inline off
    int tran_unit_region_x, tran_unit_region_y, tran_unit_region_w, tran_unit_region_h;
    /* local buffer */
    static ap_uint<DISP_DDR_BIT> local_cost[LAYER1_SIZE * 3][LAYER1_SIZE * 3];
// #pragma HLS bind_storage variable=local_cost type=RAM_T2P impl=bram
    static label_range local_label[LAYER1_SIZE * 3][LAYER1_SIZE * 3][3];
    static disp_range local_disp[LAYER1_SIZE * 3][LAYER1_SIZE * 3];
#pragma HLS array_partition variable=local_cost type=complete dim=1
#pragma HLS array_partition variable=local_label type=complete dim=1
#pragma HLS array_partition variable=local_label type=complete dim=3
#pragma HLS array_partition variable=local_disp type=complete dim=1
    static ap_uint<DISP_DDR_BIT> local_trans_cost[LAYER1_SIZE][LAYER1_SIZE]; 
    static label_range local_trans_label[LAYER1_SIZE][LAYER1_SIZE][3];
    static disp_range local_trans_disp[LAYER1_SIZE][LAYER1_SIZE];
#pragma HLS array_partition variable=local_trans_cost type=complete dim=1
#pragma HLS array_partition variable=local_trans_label type=complete dim=1
#pragma HLS array_partition variable=local_trans_label type=complete dim=3
#pragma HLS array_partition variable=local_trans_disp type=complete dim=1
    static ap_uint<DISP_DDR_BIT> local_trans_cost_tmp[LAYER1_SIZE][LAYER1_SIZE * 2]; 
    static label_range local_trans_label_tmp[LAYER1_SIZE][LAYER1_SIZE * 2][3];
    static disp_range local_trans_disp_tmp[LAYER1_SIZE][LAYER1_SIZE * 2];
#pragma HLS array_partition variable=local_trans_cost type=complete dim=1
#pragma HLS array_partition variable=local_trans_label type=complete dim=1
#pragma HLS array_partition variable=local_trans_label type=complete dim=3
#pragma HLS array_partition variable=local_trans_disp type=complete dim=1
    /* load to local buffer */
    layer1UramLoad(l1_iter, uram_y0, uram_y1, uram_y2, tran_unit_region_x, tran_unit_region_y, tran_unit_region_w, tran_unit_region_h, uram_cost, uram_label, uram_disp, 
        local_cost, local_label, local_disp, local_trans_cost, local_trans_label, local_trans_disp, local_trans_cost_tmp, local_trans_label_tmp, local_trans_disp_tmp);
    /* transmit data blocks to layer2 */
#if LAYER2_ENABLE == 1
    layer1TransWrapper(l1_iter, cost_blk_out, label_blk_out, disp_blk_out, local_trans_cost, local_trans_label, local_trans_disp);
#else
    layer1TransWrapper(l1_iter, tran_unit_region_x, tran_unit_region_y, tran_unit_region_w, tran_unit_region_h, ddr_cost, ddr_label, ddr_disp, local_trans_cost, local_trans_label, local_trans_disp);
#endif
    /* calculation */
    layer1CalcWrapper(l1_iter, unary_cost_x0, unary_cost_x1, unary_cost_x2, Q, I, C, local_unary_cost, local_im0_gray, local_cost, local_label, local_disp);
    /* cache layer0 output */
    l0_iter += layer1InitInloop(l1_iter, l0_iter, uram_y3, cost_blk_in, label_blk_in, disp_blk_in, uram_cost, uram_label, uram_disp);
    /* store to global buffer & trans_tmp buffer */
    layer1UramStore(l1_iter, uram_y0, uram_y1, uram_y2, uram_cost, uram_label, uram_disp, local_cost, local_label, local_disp, local_trans_cost_tmp, local_trans_label_tmp, local_trans_disp_tmp);
    // }
}

void layer2InitWrapper(int& l1_iter, int l2_iter, int l1_hor_region_num, int l1_ver_region_num, int l2_hor_region_num, int l2_ver_region_num,
#if HLS_LAYER2_ONLY == 0
    hls::stream_of_blocks<layer1_cost_blk> &cost_blk_in, hls::stream_of_blocks<layer1_label_blk> &label_blk_in, hls::stream_of_blocks<layer1_disp_blk> &disp_blk_in,
#else
    float cost_blk_in[LAYER1_SIZE * LAYER1_SIZE], float label_blk_in[LAYER1_SIZE * LAYER1_SIZE * 3], float disp_blk_in[LAYER1_SIZE * LAYER1_SIZE],
#endif
    float uram_cost[LAYER2_SIZE * 4][WIDTH], float uram_label[LAYER2_SIZE * 4][WIDTH][3], float uram_disp[LAYER2_SIZE * 4][WIDTH]) {
#pragma HLS inline off
    // init using data blocks from layer1
    if (l2_iter < (l2_ver_region_num * l2_hor_region_num - 2 * l2_hor_region_num)) {
        layer2Init(l1_iter, l2_iter, l1_hor_region_num, l1_ver_region_num, l2_hor_region_num, l2_ver_region_num, cost_blk_in, label_blk_in, disp_blk_in, uram_cost, uram_label, uram_disp);
    }
}

void layer2CacheWrapper(int l2_iter, int l2_ver_region_num, int l2_hor_region_num, int unary_cost_x, 
    float local_unary_cost[LAYER2_SIZE * 3][4][LAYER2_SIZE][DISP_MAX], unary_cost_t ddr_unary_cost[WIDTH * HEIGHT * DISP_DDR_MAX],
    uchar local_im0_gray[LAYER2_SIZE * 3][4][LAYER2_SIZE], uchar ddr_im0_gray[WIDTH * HEIGHT]) {
#pragma HLS inline off
    // cache unary & binary cost from DDR
    if ((l2_iter >= 0) && (l2_iter < (l2_ver_region_num * l2_hor_region_num - 2))) {
        layer2InloopCache(l2_iter, l2_ver_region_num, l2_hor_region_num, unary_cost_x, local_unary_cost, ddr_unary_cost, local_im0_gray, ddr_im0_gray);
    }
}

void layer2CalcWrapper(int l2_iter, int l2_ver_region_num, int l2_hor_region_num, int unary_cost_x0, int unary_cost_x1, int unary_cost_x2,
    ap_uint<32>* Q, ap_uint<32>& I, ap_uint<32>& C,
    float local_unary_cost[LAYER2_SIZE * 3][4][LAYER2_SIZE][DISP_MAX], uchar local_im0_gray[LAYER2_SIZE * 3][4][LAYER2_SIZE], 
    float uram_cost[LAYER2_SIZE * 4][WIDTH], float uram_label[LAYER2_SIZE * 4][WIDTH][3], float uram_disp[LAYER2_SIZE * 4][WIDTH],
    float ddr_cost[WIDTH * HEIGHT], float ddr_label[WIDTH * HEIGHT * 3], float ddr_disp[WIDTH * HEIGHT]) {
#pragma HLS inline off
    // calculation
    if ((l2_iter >= 0) && (l2_iter < l2_ver_region_num * l2_hor_region_num)) {
        layer2Calc(l2_iter, l2_ver_region_num, l2_hor_region_num, unary_cost_x0, unary_cost_x1, unary_cost_x2, Q, I, C, local_unary_cost, local_im0_gray, uram_cost, uram_label, uram_disp);
    }
    // transmit disp/cost/label to ddr
    if (l2_iter >= (l2_hor_region_num + 1)){
        layer2Trans(l2_iter, l2_ver_region_num, l2_hor_region_num, uram_cost, uram_label, uram_disp, ddr_cost, ddr_label, ddr_disp);
    }
}

#if HLS_LAYER0_ONLY == 0
void localExpLayer0(
#if PROPOSER_RANDOM_INIT == 1
    ap_uint<32> random_num[LAYER0_PROP_NUM][3], 
#endif // PROPOSER_RANDOM_INIT == 1
    unary_cost_t0 ddr_unary_cost[WIDTH * HEIGHT * DISP_DDR_MAX0], 
    ap_uint<DISP_DDR_BIT> ddr_cost[WIDTH * HEIGHT], label_range ddr_label[WIDTH * HEIGHT * 3], disp_range ddr_disp[WIDTH * HEIGHT]) {
#else
void localExpLayer0(ap_uint<32> random_num[LAYER0_PROP_NUM][3], unary_cost_t ddr_unary_cost[WIDTH * HEIGHT * DISP_DDR_MAX], uchar ddr_im0_gray[WIDTH * HEIGHT],
    hls::stream_of_blocks<layer0_cost_blk> &cost_blk_out, hls::stream_of_blocks<layer0_label_blk> &label_blk_out, hls::stream_of_blocks<layer0_disp_blk> &disp_blk_out) {}
void localExpLayer0Hls(ap_uint<32> random_num[LAYER0_PROP_NUM][3], unary_cost_t ddr_unary_cost[WIDTH * HEIGHT * DISP_DDR_MAX], uchar ddr_im0_gray[WIDTH * HEIGHT],
    float cost_blk_out[LAYER0_SIZE * LAYER0_SIZE], float label_blk_out[LAYER0_SIZE * LAYER0_SIZE * 3], float disp_blk_out[LAYER0_SIZE * LAYER0_SIZE]) {
#endif
// #pragma HLS interface mode=m_axi bundle=BUS_LAYER0 port=ddr_unary_cost offset=slave
    // 
    static ap_uint<DISP_DDR_BIT> uram_cost[LAYER0_SIZE * 3][WIDTH]; // row partition
#pragma HLS array_partition variable=uram_cost type=complete dim=1
    static label_range uram_label[LAYER0_SIZE * 3][WIDTH][3]; 
#pragma HLS array_partition variable=uram_label type=complete dim=1
#pragma HLS array_partition variable=uram_label type=complete dim=3
    static disp_range uram_disp[LAYER0_SIZE * 3][WIDTH]; 
#pragma HLS array_partition variable=uram_disp type=complete dim=1
    // 
    //static float local_unary_cost[LAYER0_SIZE * 3][4][LAYER0_SIZE][DISP_DDR_MAX * DISP_DDR_NUM];
    static unary_cost_t local_unary_cost[LAYER0_SIZE * 3][4][LAYER0_SIZE][DISP_DDR_MAX];
#pragma HLS array_partition variable=local_unary_cost type=complete dim=1
#pragma HLS array_partition variable=local_unary_cost type=complete dim=2
//DO_PRAGMA(HLS array_reshape variable=local_unary_cost type=cyclic factor=DISP_DDR_NUM dim=4) 
    // 
    static bool random_generator_init = false;
    static ap_uint<32> Q[LAYER0_PROP_NUM][4096]; static ap_uint<32> I[LAYER0_PROP_NUM]; static ap_uint<32> C[LAYER0_PROP_NUM];
#pragma HLS array_partition variable=Q type=complete dim=1
#pragma HLS array_partition variable=I type=complete dim=0
#pragma HLS array_partition variable=C type=complete dim=0
    #if LAYER0_PROP_NUM == 3
    ap_uint<4> p_x[LAYER0_PROP_NUM] = {0, 1, 2};
    ap_uint<4> p_y[LAYER0_PROP_NUM] = {4, 5, 6}; 
    ap_uint<32> pp[LAYER0_PROP_NUM][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    ap_uint<4> p_xx[LAYER0_PROP_NUM] = {0, 1, 2};
    ap_uint<4> p_yy[LAYER0_PROP_NUM] = {4, 5, 6};
#else 
    ap_uint<4> p_x[LAYER0_PROP_NUM];
    ap_uint<4> p_y[LAYER0_PROP_NUM]; 
    ap_uint<32> pp[LAYER0_PROP_NUM][3];
    ap_uint<4> p_xx[LAYER0_PROP_NUM];
    ap_uint<4> p_yy[LAYER0_PROP_NUM]; 
#endif
#pragma HLS array_partition variable=p_x type=complete dim=0
#pragma HLS array_partition variable=p_y type=complete dim=0
#pragma HLS array_partition variable=pp type=complete dim=0
#pragma HLS array_partition variable=p_xx type=complete dim=0
#pragma HLS array_partition variable=p_yy type=complete dim=0
#if PROPOSER_RANDOM_INIT == 0
    const ap_uint<32> random_num[2][3] = {
        {2357136044, 2546248239, 3071714933},
        {3626093760, 2588848963, 3684848379}
    };
#endif // PROPOSER_RANDOM_INIT == 0
#pragma HLS array_partition variable=random_num type=complete dim=0
    if (random_generator_init == false) {
        LOOP_INIT:
        for (int i = 0; i < LAYER0_PROP_NUM; i++) {
#pragma HLS unroll
            I[i] = random_num[i][0];
            C[i] = random_num[i][1];
            initRandGen(random_num[i][2], Q[i], I[i], C[i]);
        }
        random_generator_init = true;
    }
    const int hor_region_num = LAYER0_HOR_REGION_NUM;
    const int ver_region_num = LAYER0_VER_REGION_NUM;
    /* pre initialization 1 row & 1 unit region */
    layer0OutloopInit(hor_region_num, uram_cost, uram_label, uram_disp);
    /* pre cache 4 unit region (2 columns) */
    bool buff_id = false;
    //layer0OutloopCache(local_unary_cost, ddr_unary_cost ,local_im0_gray, ddr_im0_gray);
    static unary_cost_t local_buffer[2][LAYER0_SIZE][DISP_DDR_MAX0];
#pragma HLS array_partition variable=local_buffer type=complete dim=1

    layer0OutloopCache(local_buffer, local_unary_cost, ddr_unary_cost);
    // int ver_region_i = 0;
    LOOP0:
    for (int iter = 0; iter < (ver_region_num * hor_region_num + (hor_region_num + 2)); iter++) {
// #pragma HLS DATAFLOW
#pragma HLS dependence variable=local_unary_cost type=intra false
        //
        int unary_cost_x0 = (((iter - 1) % 4) == -1) ? 3 : ((iter - 1) % 4); // calc pre 
        int unary_cost_x1 = (iter + 0) % 4; // calc curr
        int unary_cost_x2 = (iter + 1) % 4; // calc post
        int unary_cost_x3 = (iter + 2) % 4; // cache
        assert((unary_cost_x0 != unary_cost_x1) && (unary_cost_x0 != unary_cost_x2) && (unary_cost_x0 != unary_cost_x3) && 
               (unary_cost_x1 != unary_cost_x2) && (unary_cost_x1 != unary_cost_x3) && (unary_cost_x2 != unary_cost_x3));
        // 
        layer0CacheWrapper(local_buffer, iter, unary_cost_x3, !buff_id, local_unary_cost, ddr_unary_cost);
        // 
        layer0InitCalcTransWrapper(iter, p_x, p_y, pp, p_xx, p_yy, unary_cost_x0, unary_cost_x1, unary_cost_x2, buff_id, Q, I, C, local_unary_cost, uram_cost, uram_label, uram_disp, ddr_cost, ddr_label, ddr_disp);
        buff_id = !buff_id;
#if PRINT_STATUS == 1 && !defined(__SYNTHESIS__)
        printf("\rlayer0 process: %4d / %d", iter, (ver_region_num * hor_region_num + (hor_region_num + 2)) - 1);  
        fflush(stdout);
    }
    printf("\n");
#else
    }
#endif
}

#if HLS_LAYER1_ONLY == 0
void localExpLayer1(
#if PROPOSER_RANDOM_INIT == 1
    ap_uint<32> random_num[LAYER1_PROP_NUM][3], 
#endif // PROPOSER_RANDOM_INIT == 1
    unary_cost_t ddr_unary_cost[WIDTH * HEIGHT * DISP_DDR_MAX], uchar ddr_im0_gray[WIDTH * HEIGHT], 
    hls::stream_of_blocks<layer0_cost_blk> &cost_blk_in, hls::stream_of_blocks<layer0_label_blk> &label_blk_in, hls::stream_of_blocks<layer0_disp_blk> &disp_blk_in,
    hls::stream_of_blocks<layer0_cost_blk> &init_cost_blk_in, hls::stream_of_blocks<layer0_label_blk> &init_label_blk_in, hls::stream_of_blocks<layer0_disp_blk> &init_disp_blk_in,
#if LAYER2_ENABLE == 1
    hls::stream_of_blocks<layer1_cost_blk> &cost_blk_out, hls::stream_of_blocks<layer1_label_blk> &label_blk_out, hls::stream_of_blocks<layer1_disp_blk> &disp_blk_out) {
#else
    ap_uint<DISP_DDR_BIT> ddr_cost[WIDTH * HEIGHT], label_range ddr_label[WIDTH * HEIGHT * 3], disp_range ddr_disp[WIDTH * HEIGHT]) {
#endif
#else
void localExpLayer1(ap_uint<32> random_num[LAYER1_PROP_NUM][3], unary_cost_t ddr_unary_cost[WIDTH * HEIGHT * DISP_DDR_MAX], uchar ddr_im0_gray[WIDTH * HEIGHT], 
    hls::stream_of_blocks<layer0_cost_blk> &cost_blk_in, hls::stream_of_blocks<layer0_label_blk> &label_blk_in, hls::stream_of_blocks<layer0_disp_blk> &disp_blk_in,
#if LAYER2_ENABLE == 1
    hls::stream_of_blocks<layer1_cost_blk> &cost_blk_out, hls::stream_of_blocks<layer1_label_blk> &label_blk_out, hls::stream_of_blocks<layer1_disp_blk> &disp_blk_out) {}
#else
    float ddr_cost[WIDTH * HEIGHT], float ddr_label[WIDTH * HEIGHT * 3], float ddr_disp[WIDTH * HEIGHT]) {}
#endif
void localExpLayer1Hls(ap_uint<32> random_num[LAYER1_PROP_NUM][3], unary_cost_t ddr_unary_cost[WIDTH * HEIGHT * DISP_DDR_MAX], uchar ddr_im0_gray[WIDTH * HEIGHT], 
    float cost_blk_in[LAYER0_SIZE * LAYER0_SIZE], float label_blk_in[LAYER0_SIZE * LAYER0_SIZE * 3], float disp_blk_in[LAYER0_SIZE * LAYER0_SIZE],
#if LAYER2_ENABLE == 1
    float cost_blk_out[LAYER1_SIZE * LAYER1_SIZE], float label_blk_out[LAYER1_SIZE * LAYER1_SIZE * 3], float disp_blk_out[LAYER1_SIZE * LAYER1_SIZE]) {
#else
    float ddr_cost[WIDTH * HEIGHT], float ddr_label[WIDTH * HEIGHT * 3], float ddr_disp[WIDTH * HEIGHT]) {
#endif
#endif
// #pragma HLS interface mode=m_axi bundle=BUS_LAYER1 port=ddr_unary_cost offset=slave
// #pragma HLS interface mode=m_axi bundle=BUS_LAYER1 port=ddr_im0_gray offset=slave
#if LAYER2_ENABLE == 0
// #pragma HLS interface mode=m_axi bundle=BUS_OUT port=ddr_cost offset=slave
// #pragma HLS interface mode=m_axi bundle=BUS_OUT port=ddr_label offset=slave
// #pragma HLS interface mode=m_axi bundle=BUS_OUT port=ddr_disp offset=slave
#endif
    // 
    static ap_uint<DISP_DDR_BIT> uram_cost[4][LAYER1_SIZE][WIDTH]; // row partition
#pragma HLS array_partition variable=uram_cost type=complete dim=1
#pragma HLS array_partition variable=uram_cost type=complete dim=2
    static label_range uram_label[4][LAYER1_SIZE][WIDTH][3]; 
#pragma HLS array_partition variable=uram_label type=complete dim=1
#pragma HLS array_partition variable=uram_label type=complete dim=2
#pragma HLS array_partition variable=uram_label type=complete dim=4
    static disp_range uram_disp[4][LAYER1_SIZE][WIDTH]; 
#pragma HLS array_partition variable=uram_disp type=complete dim=1
#pragma HLS array_partition variable=uram_disp type=complete dim=2
// #pragma HLS bind_storage variable=uram_disp type=RAM_1P impl=uram
    static unary_cost_t local_unary_cost[LAYER1_SIZE * 3][4][LAYER1_SIZE][DISP_DDR_MAX];
#pragma HLS array_partition variable=local_unary_cost type=complete dim=1
#pragma HLS array_partition variable=local_unary_cost type=complete dim=2
//DO_PRAGMA(HLS array_reshape variable=local_unary_cost type=cyclic factor=DISP_DDR_NUM dim=4) 
    static uchar local_im0_gray[LAYER1_SIZE * 3][4][LAYER1_SIZE];
#pragma HLS array_partition variable=local_im0_gray type=complete dim=1
#pragma HLS array_partition variable=local_im0_gray type=complete dim=2
    // 
    static bool random_generator_init = false;
    static ap_uint<32> Q[LAYER1_PROP_NUM][4096]; static ap_uint<32> I[LAYER1_PROP_NUM]; static ap_uint<32> C[LAYER1_PROP_NUM];
#pragma HLS array_partition variable=Q type=complete dim=1
#pragma HLS array_partition variable=I type=complete dim=0
#pragma HLS array_partition variable=C type=complete dim=0
#if PROPOSER_RANDOM_INIT == 0
    const ap_uint<32> random_num[2][3] = {
        {2340255427, 3638918503, 1819583497},
        {2678185683, 2774094101, 1650906866}
    };
#endif // PROPOSER_RANDOM_INIT == 0
#pragma HLS array_partition variable=random_num type=complete dim=0
    if (random_generator_init == false) {
        for (int i = 0; i < LAYER1_PROP_NUM; i++) {
#pragma HLS unroll
            I[i] = random_num[i][0];
            C[i] = random_num[i][1];
            initRandGen(random_num[i][2], Q[i], I[i], C[i]);
        }
        random_generator_init = true;
    }
    //
    const int l0_hor_region_num = LAYER0_HOR_REGION_NUM;
    const int l0_ver_region_num = LAYER0_VER_REGION_NUM;
    const int l1_hor_region_num = LAYER1_HOR_REGION_NUM;
    const int l1_ver_region_num = LAYER1_VER_REGION_NUM;
    // pre cache 4 unit region (2 columns)
    layer1OutloopCache(local_unary_cost, ddr_unary_cost, local_im0_gray, ddr_im0_gray);
    // 
    // int l0_iter = 0;
    int l0_iter = layer1InitOutloop(0, init_cost_blk_in, init_label_blk_in, init_disp_blk_in, uram_cost, uram_label, uram_disp);
    // 
    for (int l1_iter = 0; l1_iter < (l1_ver_region_num * l1_hor_region_num + (l1_hor_region_num + 2)); l1_iter++) {
// #pragma HLS dependence variable=uram_cost type=intra false
// #pragma HLS dependence variable=uram_disp type=intra false
// #pragma HLS dependence variable=uram_label type=intra false
#pragma HLS dependence variable=local_im0_gray type=intra false
#pragma HLS dependence variable=local_unary_cost type=intra false
        // 
        int unary_cost_x0 = ((l1_iter - 1) % 4) < 0 ? 3 : ((l1_iter - 1) % 4); // calc pre
        int unary_cost_x1 = (l1_iter + 0) % 4; // calc curr
        int unary_cost_x2 = (l1_iter + 1) % 4; // calc post
        int unary_cost_x3 = (l1_iter + 2) % 4; // cache
        assert((unary_cost_x0 != unary_cost_x1) && (unary_cost_x0 != unary_cost_x2) && (unary_cost_x0 != unary_cost_x3) && 
               (unary_cost_x1 != unary_cost_x2) && (unary_cost_x1 != unary_cost_x3) && (unary_cost_x2 != unary_cost_x3));
        int uram_y0 = (l1_iter < l1_hor_region_num) ? 0 : (l1_iter / l1_hor_region_num - 1) % 4;
        int uram_y1 = (l1_iter < l1_hor_region_num) ? 1 : (l1_iter / l1_hor_region_num + 0) % 4;
        int uram_y2 = (l1_iter < l1_hor_region_num) ? 3 : (l1_iter / l1_hor_region_num + 1) % 4;
        int uram_y3 = (l1_iter < l1_hor_region_num) ? 2 : (l1_iter / l1_hor_region_num + 2) % 4;
        assert((uram_y0 != uram_y1) && (uram_y0 != uram_y2) && (uram_y0 != uram_y3) && 
               (uram_y1 != uram_y2) && (uram_y1 != uram_y3) && (uram_y2 != uram_y3));
        // 
        // layer1InitWrapper(l0_iter, l1_iter, uram_y3, cost_blk_in, label_blk_in, disp_blk_in, uram_cost, uram_label, uram_disp);
        // 
        layer1CacheWrapper(l1_iter, unary_cost_x3, local_unary_cost, ddr_unary_cost, local_im0_gray, ddr_im0_gray);
        // 
#if LAYER2_ENABLE == 1
        layer1CalcTransWrapper(l1_iter, unary_cost_x0, unary_cost_x1, unary_cost_x2, Q, I, C, local_unary_cost, local_im0_gray, uram_cost, uram_label, uram_disp, cost_blk_out, label_blk_out, disp_blk_out);
#else
        layer1CalcTransWrapper(l0_iter, l1_iter, unary_cost_x0, unary_cost_x1, unary_cost_x2, uram_y0, uram_y1, uram_y2, uram_y3, Q, I, C, local_unary_cost, local_im0_gray, 
            uram_cost, uram_label, uram_disp, cost_blk_in, label_blk_in, disp_blk_in, /* init_cost_blk_in, init_label_blk_in, init_disp_blk_in, */ ddr_cost, ddr_label, ddr_disp);
#endif
#if PRINT_STATUS == 1 && !defined(__SYNTHESIS__)
        printf("\rlayer1 process: %4d / %d", l1_iter, (l1_ver_region_num * l1_hor_region_num + (l1_hor_region_num + 2)) - 1);  
        fflush(stdout);
    }
    printf("\n");
#else
    }
#endif
}

#if HLS_LAYER2_ONLY == 0
void localExpLayer2(unary_cost_t ddr_unary_cost[WIDTH * HEIGHT * DISP_DDR_MAX], uchar ddr_im0_gray[WIDTH * HEIGHT], 
    hls::stream_of_blocks<layer1_cost_blk> &cost_blk_in, hls::stream_of_blocks<layer1_label_blk> &label_blk_in, hls::stream_of_blocks<layer1_disp_blk> &disp_blk_in,
    float ddr_cost[WIDTH * HEIGHT], float ddr_label[WIDTH * HEIGHT * 3], float ddr_disp[WIDTH * HEIGHT]) {
#else
void localExpLayer2(unary_cost_t ddr_unary_cost[WIDTH * HEIGHT * DISP_DDR_MAX], uchar ddr_im0_gray[WIDTH * HEIGHT], 
    hls::stream_of_blocks<layer1_cost_blk> &cost_blk_in, hls::stream_of_blocks<layer1_label_blk> &label_blk_in, hls::stream_of_blocks<layer1_disp_blk> &disp_blk_in,
    float ddr_cost[WIDTH * HEIGHT], float ddr_label[WIDTH * HEIGHT * 3], float ddr_disp[WIDTH * HEIGHT]) {}
void localExpLayer2Hls(unary_cost_t ddr_unary_cost[WIDTH * HEIGHT * DISP_DDR_MAX], uchar ddr_im0_gray[WIDTH * HEIGHT], 
    float cost_blk_in[LAYER1_SIZE * LAYER1_SIZE], float label_blk_in[LAYER1_SIZE * LAYER1_SIZE * 3], float disp_blk_in[LAYER1_SIZE * LAYER1_SIZE],
    float ddr_cost[WIDTH * HEIGHT], float ddr_label[WIDTH * HEIGHT * 3], float ddr_disp[WIDTH * HEIGHT]) {
#endif
    // 
    static float uram_cost[LAYER2_SIZE * 4][WIDTH]; // row partition
#pragma HLS array_partition variable=uram_cost type=complete dim=1
    static float uram_label[LAYER2_SIZE * 4][WIDTH][3]; 
#pragma HLS array_partition variable=uram_label type=complete dim=1
#pragma HLS array_partition variable=uram_label type=complete dim=3
    static float uram_disp[LAYER2_SIZE * 4][WIDTH]; 
#pragma HLS array_partition variable=uram_disp type=complete dim=1
    static float local_unary_cost[LAYER2_SIZE * 3][4][LAYER2_SIZE][DISP_MAX];
#pragma HLS array_partition variable=local_unary_cost type=complete dim=1
#pragma HLS array_partition variable=local_unary_cost type=complete dim=2
DO_PRAGMA(HLS array_reshape variable=local_unary_cost type=cyclic factor=DISP_DDR_NUM dim=4) 
    static uchar local_im0_gray[LAYER2_SIZE * 3][4][LAYER2_SIZE];
#pragma HLS array_partition variable=local_im0_gray type=complete dim=1
#pragma HLS array_partition variable=local_im0_gray type=complete dim=2
    // 
    static bool random_generator_init = false;
    static ap_uint<32> Q[4096];
    static ap_uint<32> I = 1731;
    static ap_uint<32> C = 793451;
    if (random_generator_init == false) {
        initRandGen(0, Q, I, C); 
        random_generator_init = true;
    }
    //
    const int l1_hor_region_num = LAYER1_HOR_REGION_NUM;
    const int l1_ver_region_num = LAYER1_VER_REGION_NUM;
    const int l2_hor_region_num = LAYER2_HOR_REGION_NUM;
    const int l2_ver_region_num = LAYER2_VER_REGION_NUM;
    // pre cache 4 unit region (2 columns)
    layer2OutloopCache(local_unary_cost, ddr_unary_cost, local_im0_gray, ddr_im0_gray);
    // 
    int l1_iter = 0;
    // 
    for (int l2_iter = -1; l2_iter < (l2_ver_region_num * l2_hor_region_num + (l2_hor_region_num + 1)); l2_iter++) {
#pragma HLS dependence variable=uram_cost type=intra false
#pragma HLS dependence variable=uram_label type=intra false
#pragma HLS dependence variable=uram_disp type=intra false
#pragma HLS dependence variable=local_unary_cost type=intra false
#pragma HLS dependence variable=local_im0_gray type=intra false
        // 
        int unary_cost_x0 = (l2_iter - 1) % 4; // calc pre
        int unary_cost_x1 = (l2_iter + 0) % 4; // calc curr
        int unary_cost_x2 = (l2_iter + 1) % 4; // calc post
        int unary_cost_x3 = (l2_iter + 2) % 4; // cache
        assert((unary_cost_x0 != unary_cost_x1) && (unary_cost_x0 != unary_cost_x2) && (unary_cost_x0 != unary_cost_x3) && 
               (unary_cost_x1 != unary_cost_x2) && (unary_cost_x1 != unary_cost_x3) && (unary_cost_x2 != unary_cost_x3));
        // 
        layer2InitWrapper(l1_iter, l2_iter, l1_hor_region_num, l1_ver_region_num, l2_hor_region_num, l2_ver_region_num, cost_blk_in, label_blk_in, disp_blk_in, uram_cost, uram_label, uram_disp);
        // 
        layer2CacheWrapper(l2_iter, l2_ver_region_num, l2_hor_region_num, unary_cost_x3, local_unary_cost, ddr_unary_cost, local_im0_gray, ddr_im0_gray);
        // 
        layer2CalcWrapper(l2_iter, l2_ver_region_num, l2_hor_region_num, unary_cost_x0, unary_cost_x1, unary_cost_x2, Q, I, C, local_unary_cost, local_im0_gray, uram_cost, uram_label, uram_disp, ddr_cost, ddr_label, ddr_disp);
    }
}
#endif // COST_LOAD_MODE == 1 && SCAN_DIRECTION == 0

#if COST_LOAD_MODE == 1 && SCAN_DIRECTION == 1

/*
void layer0OutloopCache(float local_unary_cost[4][LAYER0_SIZE][LAYER0_SIZE * 3][DISP_MAX], float ddr_unary_cost[WIDTH * HEIGHT * DISP_MAX]) {
#pragma HLS inline off
#pragma HLS array_partition variable=local_unary_cost type=complete dim=1
#pragma HLS interface mode=m_axi bundle=BUS_LAYER0 port=ddr_unary_cost offset=slave
    for (int y = 0; y < LAYER0_SIZE; y++) {
        for (int x = 0; x < LAYER0_SIZE; x++) {
            for (int d = 0; d < DISP_MAX; d++) {
#pragma HLS pipeline II=1
                local_unary_cost[0][y][x][d] = ddr_unary_cost[(y * WIDTH + x) * DISP_MAX + d];
                local_unary_cost[1][y][x][d] = ddr_unary_cost[((y + LAYER0_SIZE) * WIDTH + x) * DISP_MAX + d];
                local_unary_cost[0][y][x + LAYER0_SIZE][d] = ddr_unary_cost[(y * WIDTH + x + LAYER0_SIZE) * DISP_MAX + d];
                local_unary_cost[1][y][x + LAYER0_SIZE][d] = ddr_unary_cost[((y + LAYER0_SIZE) * WIDTH + x + LAYER0_SIZE) * DISP_MAX + d];
            }
        }
    }
}
*/

void layer0OutloopCache(unary_cost_t local_buffer[2][DISP_DDR_MAX0], unary_cost_t local_unary_cost[4][LAYER0_SIZE][LAYER0_SIZE * 3][DISP_DDR_MAX],unary_cost_t0 ddr_unary_cost[WIDTH * HEIGHT * DISP_DDR_MAX0]) {
#pragma HLS inline off
// #pragma HLS array_partition variable=local_unary_cost type=complete dim=1
// #pragma HLS array_partition variable=local_unary_cost type=complete dim=2
// DO_PRAGMA(HLS array_reshape variable=local_unary_cost type=cyclic factor=DISP_DDR_NUM dim=4)
// #pragma HLS array_partition variable=local_unary_cost_init type=complete dim=1
// #pragma HLS array_partition variable=local_unary_cost_init type=complete dim=2
// DO_PRAGMA(HLS array_reshape variable=local_unary_cost_init type=cyclic factor=DISP_DDR_NUM dim=4)   
// #pragma HLS interface mode=m_axi bundle=BUS_LAYER0 port=ddr_unary_cost offset=slave
    /*
    LOOP_COST1:
    for (int xyd = 0; xyd < DISP_DDR_MAX0 * (LAYER0_SIZE * LAYER0_SIZE * 2); xyd++) {
#pragma HLS pipeline II=1
        int y = (xyd) / (2 * LAYER0_SIZE * DISP_DDR_MAX0);
        int x = (xyd) % (2 * LAYER0_SIZE * DISP_DDR_MAX0) / DISP_DDR_MAX0;
        int d = (xyd) % (2 * LAYER0_SIZE * DISP_DDR_MAX0) % DISP_DDR_MAX0;
        bool index = x % 2;
        int dd = d << 1;
        local_unary_cost[0][y][x][dd] = ddr_unary_cost[(y * WIDTH + x) * DISP_DDR_MAX0 + d](127,0); 
        if (dd >= DISP_DDR_MAX - 1)
            continue;
        local_unary_cost[0][y][x][dd + 1] = ddr_unary_cost[(y * WIDTH + x) * DISP_DDR_MAX0 + d](255,128);
    }
    LOOP_COST2:
    for (int xyd = 0; xyd < DISP_DDR_MAX0 * (LAYER0_SIZE * LAYER0_SIZE * 2); xyd++) {
#pragma HLS pipeline II=1
        int y = (xyd) / (2 * LAYER0_SIZE * DISP_DDR_MAX0);
        int x = (xyd) % (2 * LAYER0_SIZE * DISP_DDR_MAX0) / DISP_DDR_MAX0;
        int d = (xyd) % (2 * LAYER0_SIZE * DISP_DDR_MAX0) % DISP_DDR_MAX0;
        bool index = x % 2;
        int dd = d << 1;
        local_unary_cost[1][y][x][dd] = ddr_unary_cost[((y + LAYER0_SIZE) * WIDTH + x) * DISP_DDR_MAX0 + d](127,0); 
        if (dd >= DISP_DDR_MAX - 1)
            continue;
        local_unary_cost[1][y][x][dd + 1] = ddr_unary_cost[((y + LAYER0_SIZE) * WIDTH + x) * DISP_DDR_MAX0 + d](255,128);
    }
    */
    LOOP_COST1:
    for (int xyd = 0; xyd < DISP_DDR_MAX0 * (LAYER0_SIZE * LAYER0_SIZE * 2); xyd++) {
#pragma HLS pipeline II=1
        int y = (xyd) / (2 * LAYER0_SIZE * DISP_DDR_MAX0);
        int x = (xyd) % (2 * LAYER0_SIZE * DISP_DDR_MAX0) / DISP_DDR_MAX0;
        int d = (xyd) % (2 * LAYER0_SIZE * DISP_DDR_MAX0) % DISP_DDR_MAX0;
        bool index = x % 2;
        int dd = d << 1;
        local_unary_cost[0][y][x][dd] = ddr_unary_cost[(y * WIDTH + x) * DISP_DDR_MAX0 + d](127,0); 
        local_buffer[index][d] = ddr_unary_cost[(y * WIDTH + x) * DISP_DDR_MAX0 + d](255,128);
        if ((dd >= DISP_DDR_MAX - 1) || (xyd < DISP_DDR_MAX0))
            continue;  
        int yy = (x == 0) ? (y - 1) : y;
        int xx = (x == 0) ? 17 : (x - 1);
        local_unary_cost[0][yy][xx][dd + 1] = local_buffer[!index][d];
    }
    LOOP_COST2:
    for (int xyd = 0; xyd <  DISP_DDR_MAX0; xyd++) {
#pragma HLS pipeline II=1
        int y = 9;
        int x = 0;
        int d = xyd;
        bool index = 0;
        int dd = d << 1;
        if (dd >= DISP_DDR_MAX - 1)
            continue; 
        int yy = (x == 0) ? (y - 1) : y;
        int xx = (x == 0) ? 17 : (x - 1);
        local_unary_cost[0][yy][xx][dd + 1] = local_buffer[!index][d];
    }
    LOOP_COST4:
    for (int xyd = 0; xyd < DISP_DDR_MAX0 * (LAYER0_SIZE * LAYER0_SIZE * 2); xyd++) {
#pragma HLS pipeline II=1
        int y = (xyd) / (2 * LAYER0_SIZE * DISP_DDR_MAX0);
        int x = (xyd) % (2 * LAYER0_SIZE * DISP_DDR_MAX0) / DISP_DDR_MAX0;
        int d = (xyd) % (2 * LAYER0_SIZE * DISP_DDR_MAX0) % DISP_DDR_MAX0;
        bool index = x % 2;
        int dd = d << 1;
        local_unary_cost[1][y][x][dd] = ddr_unary_cost[((y + LAYER0_SIZE) * WIDTH + x) * DISP_DDR_MAX0 + d](127,0); 
        local_buffer[index][d] = ddr_unary_cost[((y + LAYER0_SIZE) * WIDTH + x) * DISP_DDR_MAX0 + d](255,128);
        if  ((dd >= DISP_DDR_MAX - 1) || (xyd < DISP_DDR_MAX0))
            continue; 
        int yy = (x == 0) ? (y - 1) : y;
        int xx = (x == 0) ? 17 : (x - 1);
        local_unary_cost[1][yy][xx][dd + 1] = local_buffer[!index][d];
    }
    LOOP_COST5:
    for (int xyd = 0; xyd < DISP_DDR_MAX0; xyd++) {
#pragma HLS pipeline II=1
        int y = 9;
        int x = 0;
        int d = xyd;
        bool index = 0;
        int dd = d << 1;
        if (dd >= DISP_DDR_MAX - 1)
            continue; 
        int yy = (x == 0) ? (y - 1) : y;
        int xx = (x == 0) ? 17 : (x - 1);
        local_unary_cost[1][yy][xx][dd + 1] = local_buffer[!index][d];
    }
    
    /*
    LOOP_COST0:
    for (int xyd = 0; xyd < LAYER0_SIZE * LAYER0_SIZE * DISP_DDR_MAX; xyd++) {
#pragma HLS pipeline II=1
        int y = xyd / (LAYER0_SIZE * DISP_DDR_MAX);
        int x = xyd % (LAYER0_SIZE * DISP_DDR_MAX) / DISP_DDR_MAX;
        int d = xyd % (LAYER0_SIZE * DISP_DDR_MAX) % DISP_DDR_MAX;
        local_unary_cost[0][y][x][d] = ddr_unary_cost[(y * WIDTH + x) * DISP_DDR_MAX + d]; 
    }
    
    LOOP_COST1:
    for (int xyd = 0; xyd < LAYER0_SIZE * LAYER0_SIZE * DISP_DDR_MAX; xyd++) {
#pragma HLS pipeline II=1
        int y = xyd / (LAYER0_SIZE * DISP_DDR_MAX);
        int x = xyd % (LAYER0_SIZE * DISP_DDR_MAX) / DISP_DDR_MAX;
        int d = xyd % (LAYER0_SIZE * DISP_DDR_MAX) % DISP_DDR_MAX;
        local_unary_cost[1][y][x][d] = ddr_unary_cost[((y + LAYER0_SIZE) * WIDTH + x) * DISP_DDR_MAX + d];
    }

    LOOP_COST2:
    for (int xyd = 0; xyd < LAYER0_SIZE * LAYER0_SIZE * DISP_DDR_MAX; xyd++) {
#pragma HLS pipeline II=1
        int y = xyd / (LAYER0_SIZE * DISP_DDR_MAX);
        int x = xyd % (LAYER0_SIZE * DISP_DDR_MAX) / DISP_DDR_MAX;
        int d = xyd % (LAYER0_SIZE * DISP_DDR_MAX) % DISP_DDR_MAX;
        local_unary_cost[0][y][x + LAYER0_SIZE][d] = ddr_unary_cost[(y * WIDTH + x + LAYER0_SIZE) * DISP_DDR_MAX + d];
    }

    LOOP_COST3:
    for (int xyd = 0; xyd < LAYER0_SIZE * LAYER0_SIZE * DISP_DDR_MAX; xyd++) {
#pragma HLS pipeline II=1
        int y = xyd / (LAYER0_SIZE * DISP_DDR_MAX);
        int x = xyd % (LAYER0_SIZE * DISP_DDR_MAX) / DISP_DDR_MAX;
        int d = xyd % (LAYER0_SIZE * DISP_DDR_MAX) % DISP_DDR_MAX;
        local_unary_cost[1][y][x + LAYER0_SIZE][d] = ddr_unary_cost[((y + LAYER0_SIZE) * WIDTH + x + LAYER0_SIZE) * DISP_DDR_MAX + d];
    }
    */
}

/*
void layer0OutloopInit(int ver_region_num, ap_uint<32>* q, ap_uint<32>& i, ap_uint<32>& c, 
    ap_uint<32>* q1, ap_uint<32>& i1, ap_uint<32>& c1, ap_uint<32>* q2, ap_uint<32>& i2, ap_uint<32>& c2, 
    ap_uint<32>* q3, ap_uint<32>& i3, ap_uint<32>& c3, ap_uint<32>* q4, ap_uint<32>& i4, ap_uint<32>& c4, 
    ap_uint<32>* q5, ap_uint<32>& i5, ap_uint<32>& c5, 
    float uram_cost[HEIGHT][LAYER0_SIZE * 3], float uram_label[HEIGHT][LAYER0_SIZE * 3][3], float uram_disp[HEIGHT][LAYER0_SIZE * 3],
    float ddr_unary_cost[WIDTH * HEIGHT * DISP_MAX]) {
#pragma HLS inline off
    for (int iter = 0; iter < (ver_region_num + 1); iter++) {
        int ver_region_i = iter % ver_region_num;
        int hor_region_i = iter / ver_region_num;
        // 
        int unit_region_x, unit_region_y, unit_region_w, unit_region_h;
        int expn_region_x, expn_region_y, expn_region_w, expn_region_h;
        genRegionInfo(WIDTH, HEIGHT, hor_region_i, ver_region_i, LAYER0_SIZE, unit_region_x, unit_region_y, unit_region_w, unit_region_h, expn_region_x, expn_region_y, expn_region_w, expn_region_h);
        // 
        float plane_a, plane_b, plane_c;
        randomPlaneInit(q, i, c, q1, i1, c1, q2, i2, c2, q3, i3, c3, q4, i4, c4, q5, i5, c5, 
            unit_region_x, unit_region_y, unit_region_w, unit_region_h, plane_a, plane_b, plane_c);
        // 
        for (int y = 0; y < LAYER0_SIZE; y++) {
            for (int x = 0; x < LAYER0_SIZE; x++) {
                if (((unit_region_x + x) >= WIDTH) || (unit_region_y + y) >= HEIGHT)
                    continue;
                float d = plane_a * (unit_region_x + x) + plane_b * (unit_region_y + y) + plane_c;
                float cost_tmp;
                if (d < DISP_MIN) cost_tmp = ddr_unary_cost[((unit_region_y + y) * WIDTH + (unit_region_x + x)) * DISP_MAX + 0];
                else if (d >= DISP_MAX - 1) cost_tmp = ddr_unary_cost[((unit_region_y + y) * WIDTH + (unit_region_x + x)) * DISP_MAX + DISP_MAX - 1];
                else {
                    int d_l = std::floor(d);
                    int d_h = d_l + 1;
                    float f_h = d - d_l;
                    float f_l = 1.0 - f_h;
                    cost_tmp = f_l * ddr_unary_cost[((unit_region_y + y) * WIDTH + (unit_region_x + x)) * DISP_MAX + d_l] + 
                               f_h * ddr_unary_cost[((unit_region_y + y) * WIDTH + (unit_region_x + x)) * DISP_MAX + d_h];
                }
                uram_cost[(unit_region_y + y)][((hor_region_i % 3) * LAYER0_SIZE + x)] = cost_tmp;
                uram_label[(unit_region_y + y)][((hor_region_i % 3) * LAYER0_SIZE + x)][0] = plane_a;
                uram_label[(unit_region_y + y)][((hor_region_i % 3) * LAYER0_SIZE + x)][1] = plane_b;
                uram_label[(unit_region_y + y)][((hor_region_i % 3) * LAYER0_SIZE + x)][2] = plane_c;
                uram_disp[(unit_region_y + y)][((hor_region_i % 3) * LAYER0_SIZE + x)] = d;
            }
        }
    }
}
*/

void layer0OutloopInit(int ver_region_num, /* ap_uint<32>* q, ap_uint<32>& i, ap_uint<32>& c, 
    ap_uint<32>* q1, ap_uint<32>& i1, ap_uint<32>& c1, ap_uint<32>* q2, ap_uint<32>& i2, ap_uint<32>& c2, 
    ap_uint<32>* q3, ap_uint<32>& i3, ap_uint<32>& c3, ap_uint<32>* q4, ap_uint<32>& i4, ap_uint<32>& c4, 
    ap_uint<32>* q5, ap_uint<32>& i5, ap_uint<32>& c5, */ 
    ap_uint<DISP_DDR_BIT> uram_cost[HEIGHT][LAYER0_SIZE * 3], label_range uram_label[HEIGHT][LAYER0_SIZE * 3][3], disp_range uram_disp[HEIGHT][LAYER0_SIZE * 3]) {
#pragma HLS inline off
// #pragma HLS array_partition variable=uram_cost type=complete dim=1
// #pragma HLS array_partition variable=uram_label type=complete dim=1
// #pragma HLS array_partition variable=uram_label type=complete dim=3
// #pragma HLS array_partition variable=uram_disp type=complete dim=1
// #pragma HLS interface mode=m_axi bundle=BUS_LAYER0 port=ddr_unary_cost offset=slave
    for (int iter = 0; iter < (LAYER0_VER_REGION_NUM + 2); iter++) {
        int ver_region_i = iter % LAYER0_VER_REGION_NUM;
        int hor_region_i = iter / LAYER0_VER_REGION_NUM;
        // 
        int unit_region_x, unit_region_y, unit_region_w, unit_region_h;
        int expn_region_x, expn_region_y, expn_region_w, expn_region_h;
        genRegionInfo(WIDTH, HEIGHT, hor_region_i, ver_region_i, LAYER0_SIZE, unit_region_x, unit_region_y, unit_region_w, unit_region_h, expn_region_x, expn_region_y, expn_region_w, expn_region_h);
#if 0
        // 
        float plane_a, plane_b, plane_c;
        randomPlaneInit(q, i, c, q1, i1, c1, q2, i2, c2, q3, i3, c3, q4, i4, c4, q5, i5, c5, 
            unit_region_x, unit_region_y, unit_region_w, unit_region_h, plane_a, plane_b, plane_c);
        // 
        for (int x = 0; x < LAYER0_SIZE; x++) {
            for (int y = 0; y < LAYER0_SIZE; y++) {
#pragma HLS pipeline II=1
                if (((unit_region_x + x) >= WIDTH) || (unit_region_y + y) >= HEIGHT)
                    continue;
                float d = plane_a * (unit_region_x + x) + plane_b * (unit_region_y + y) + plane_c;
                float cost_tmp;
                if (d < DISP_MIN) {
                    cost_tmp = ddr_unary_cost[((unit_region_y + y) * WIDTH + (unit_region_x + x)) * DISP_DDR_MAX + 0](DISP_DDR_BIT -1, 0).to_int();
                }
                else if (d >= DISP_MAX - 1) {
                    int d_dec = (DISP_MAX - 1) % DISP_DDR_NUM;
                    cost_tmp = ddr_unary_cost[((unit_region_y + y) * WIDTH + (unit_region_x + x)) * DISP_DDR_MAX + DISP_DDR_MAX - 1](d_dec * DISP_DDR_BIT + DISP_DDR_BIT -1, d_dec * DISP_DDR_BIT).to_int();
                }
                else {
                    int d_l = std::floor(d);
                    int d_h = d_l + 1;
                    float f_h = d - d_l;
                    float f_l = 1.0 - f_h;
                    int d_l_int = d_l / DISP_DDR_NUM;
                    int d_l_dec = d_l % DISP_DDR_NUM;
                    int d_h_int = d_h / DISP_DDR_NUM;
                    int d_h_dec = d_h % DISP_DDR_NUM;
                    cost_tmp = f_l * ddr_unary_cost[((unit_region_y + y) * WIDTH + (unit_region_x + x)) * DISP_DDR_MAX + d_l_int](d_l_dec * DISP_DDR_BIT + DISP_DDR_BIT -1, d_l_dec * DISP_DDR_BIT).to_int() + 
                               f_h * ddr_unary_cost[((unit_region_y + y) * WIDTH + (unit_region_x + x)) * DISP_DDR_MAX + d_h_int](d_h_dec * DISP_DDR_BIT + DISP_DDR_BIT -1, d_h_dec * DISP_DDR_BIT).to_int();
                }
                uram_cost[((ver_region_i % 3) * LAYER0_SIZE + y)][(unit_region_x + x)] = cost_tmp;
                uram_label[((ver_region_i % 3) * LAYER0_SIZE + y)][(unit_region_x + x)][0] = plane_a;
                uram_label[((ver_region_i % 3) * LAYER0_SIZE + y)][(unit_region_x + x)][1] = plane_b;
                uram_label[((ver_region_i % 3) * LAYER0_SIZE + y)][(unit_region_x + x)][2] = plane_c;
                uram_disp[((ver_region_i % 3) * LAYER0_SIZE + y)][(unit_region_x + x)] = d;
            }
        }
#else
        for (int y = 0; y < LAYER0_SIZE; y++) {
#pragma HLS pipeline II=1
            for (int x = 0; x < LAYER0_SIZE; x++) {
#pragma HLS unroll
                if (((unit_region_x + x) >= WIDTH) || (unit_region_y + y) >= HEIGHT)
                    continue;
                uram_cost[(unit_region_y + y)][((hor_region_i % 3) * LAYER0_SIZE + x)] = ((1 << (DISP_DDR_BIT)) - 1);
                uram_label[(unit_region_y + y)][((hor_region_i % 3) * LAYER0_SIZE + x)][0] = 0;
                uram_label[(unit_region_y + y)][((hor_region_i % 3) * LAYER0_SIZE + x)][1] = 0;
                uram_label[(unit_region_y + y)][((hor_region_i % 3) * LAYER0_SIZE + x)][2] = 0;
                uram_disp[(unit_region_y + y)][((hor_region_i % 3) * LAYER0_SIZE + x)] = 0;
            }
        }
#endif
    }
}

/*
void layer0InloopInit(int iter, int ver_region_num, int hor_region_num, int unary_cost_y, 
    ap_uint<32>* q, ap_uint<32>& i, ap_uint<32>& c, ap_uint<32>* q1, ap_uint<32>& i1, ap_uint<32>& c1, 
    ap_uint<32>* q2, ap_uint<32>& i2, ap_uint<32>& c2, ap_uint<32>* q3, ap_uint<32>& i3, ap_uint<32>& c3, 
    ap_uint<32>* q4, ap_uint<32>& i4, ap_uint<32>& c4, ap_uint<32>* q5, ap_uint<32>& i5, ap_uint<32>& c5, 
    float local_unary_cost[4][LAYER0_SIZE][LAYER0_SIZE * 3][DISP_MAX],
    float uram_cost[HEIGHT][LAYER0_SIZE * 3], float uram_label[HEIGHT][LAYER0_SIZE * 3][3], float uram_disp[HEIGHT][LAYER0_SIZE * 3]) {
    //
    int init_ver_region_i = (iter + ver_region_num + 1) % ver_region_num;
    int init_hor_region_i = (iter + ver_region_num + 1) / ver_region_num;
    int calc_ver_region_i = iter % ver_region_num;
    int calc_hor_region_i = iter / ver_region_num;
    int init_unit_region_x, init_unit_region_y, init_unit_region_w, init_unit_region_h;
    int init_expn_region_x, init_expn_region_y, init_expn_region_w, init_expn_region_h;
    genRegionInfo(WIDTH, HEIGHT, init_hor_region_i, init_ver_region_i, LAYER0_SIZE, init_unit_region_x, init_unit_region_y, init_unit_region_w, init_unit_region_h, 
        init_expn_region_x, init_expn_region_y, init_expn_region_w, init_expn_region_h);
    int calc_unit_region_x, calc_unit_region_y, calc_unit_region_w, calc_unit_region_h;
    int calc_expn_region_x, calc_expn_region_y, calc_expn_region_w, calc_expn_region_h;
    genRegionInfo(WIDTH, HEIGHT, calc_hor_region_i, calc_ver_region_i, LAYER0_SIZE, calc_unit_region_x, calc_unit_region_y, calc_unit_region_w, calc_unit_region_h, 
        calc_expn_region_x, calc_expn_region_y, calc_expn_region_w, calc_expn_region_h);
    //
    float plane_a, plane_b, plane_c;
    randomPlaneInit(q, i, c, q1, i1, c1, q2, i2, c2, q3, i3, c3, q4, i4, c4, q5, i5, c5, 
        init_unit_region_x, init_unit_region_y, init_unit_region_w, init_unit_region_h, plane_a, plane_b, plane_c);
    // 
    for (int y = 0; y < LAYER0_SIZE; y++) {
        for (int x = 0; x < LAYER0_SIZE; x++) {
#pragma HLS pipeline II=1
            if (((init_unit_region_x + x) >= WIDTH) || (init_unit_region_y + y) >= HEIGHT)
                continue;
            float d = plane_a * (init_unit_region_x + x) + plane_b * (init_unit_region_y + y) + plane_c;
            float cost_tmp;
            if (init_hor_region_i != 0) {
                if (d < DISP_MIN) cost_tmp = local_unary_cost[unary_cost_y][y][init_unit_region_x - calc_expn_region_x + x][0];
                else if (d >= DISP_MAX - 1) cost_tmp = local_unary_cost[unary_cost_y][y][init_unit_region_x - calc_expn_region_x + x][DISP_MAX - 1];
                else {
                    int d_l = std::floor(d);
                    int d_h = d_l + 1;
                    float f_h = d - d_l;
                    float f_l = 1.0 - f_h;
                    cost_tmp = f_l * local_unary_cost[unary_cost_y][y][init_unit_region_x - calc_expn_region_x + x][d_l] + 
                               f_h * local_unary_cost[unary_cost_y][y][init_unit_region_x - calc_expn_region_x + x][d_h];
                }
            }
            else {
                if (d < DISP_MIN) cost_tmp = local_unary_cost[unary_cost_y][y][LAYER0_SIZE * 2 + x][0];
                else if (d >= DISP_MAX - 1) cost_tmp = local_unary_cost[unary_cost_y][y][LAYER0_SIZE * 2 + x][DISP_MAX - 1];
                else {
                    int d_l = std::floor(d);
                    int d_h = d_l + 1;
                    float f_h = d - d_l;
                    float f_l = 1.0 - f_h;
                    cost_tmp = f_l * local_unary_cost[unary_cost_y][y][LAYER0_SIZE * 2 + x][d_l] + 
                               f_h * local_unary_cost[unary_cost_y][y][LAYER0_SIZE * 2 + x][d_h];
                }
            }
            uram_cost[(init_unit_region_y + y)][((init_hor_region_i % 3) * LAYER0_SIZE + x)] = cost_tmp;
            uram_label[(init_unit_region_y + y)][((init_hor_region_i % 3) * LAYER0_SIZE + x)] [0] = plane_a;
            uram_label[(init_unit_region_y + y)][((init_hor_region_i % 3) * LAYER0_SIZE + x)] [1] = plane_b;
            uram_label[(init_unit_region_y + y)][((init_hor_region_i % 3) * LAYER0_SIZE + x)] [2] = plane_c;
            uram_disp[(init_unit_region_y + y)][((init_hor_region_i % 3) * LAYER0_SIZE + x)]  = d;
        }
    }
}
*/

void layer0InloopInit(int iter, int ver_region_num, int hor_region_num, int buff_id, 
    /* ap_uint<32>* q, ap_uint<32>& i, ap_uint<32>& c, ap_uint<32>* q1, ap_uint<32>& i1, ap_uint<32>& c1, 
    ap_uint<32>* q2, ap_uint<32>& i2, ap_uint<32>& c2, ap_uint<32>* q3, ap_uint<32>& i3, ap_uint<32>& c3, 
    ap_uint<32>* q4, ap_uint<32>& i4, ap_uint<32>& c4, ap_uint<32>* q5, ap_uint<32>& i5, ap_uint<32>& c5,  */
    ap_uint<DISP_DDR_BIT> local_init_cost[LAYER0_SIZE][LAYER0_SIZE], label_range local_init_label[LAYER0_SIZE][LAYER0_SIZE][3], disp_range local_init_disp[LAYER0_SIZE][LAYER0_SIZE]) {
#pragma HLS inline off
// #pragma HLS array_partition variable=local_init_cost type=complete dim=1
// #pragma HLS array_partition variable=local_init_label type=complete dim=1
// #pragma HLS array_partition variable=local_init_label type=complete dim=3
// #pragma HLS array_partition variable=local_init_disp type=complete dim=1
    // if (iter < (ver_region_num * hor_region_num - hor_region_num - 1)) {
        //
        int init_ver_region_i = (iter + ver_region_num + 2) % ver_region_num;
        int init_hor_region_i = (iter + ver_region_num + 2) / ver_region_num;
        int calc_ver_region_i = iter % ver_region_num;
        int calc_hor_region_i = iter / ver_region_num;
        int init_unit_region_x, init_unit_region_y, init_unit_region_w, init_unit_region_h;
        int init_expn_region_x, init_expn_region_y, init_expn_region_w, init_expn_region_h;
        genRegionInfo(WIDTH, HEIGHT, init_hor_region_i, init_ver_region_i, LAYER0_SIZE, init_unit_region_x, init_unit_region_y, init_unit_region_w, init_unit_region_h, 
            init_expn_region_x, init_expn_region_y, init_expn_region_w, init_expn_region_h);
        int calc_unit_region_x, calc_unit_region_y, calc_unit_region_w, calc_unit_region_h;
        int calc_expn_region_x, calc_expn_region_y, calc_expn_region_w, calc_expn_region_h;
        genRegionInfo(WIDTH, HEIGHT, calc_hor_region_i, calc_ver_region_i, LAYER0_SIZE, calc_unit_region_x, calc_unit_region_y, calc_unit_region_w, calc_unit_region_h, 
            calc_expn_region_x, calc_expn_region_y, calc_expn_region_w, calc_expn_region_h);
        // 
        // int cach_x = (iter + 1) % 4;
        //
#if 0
        float plane_a, plane_b, plane_c;
        randomPlaneInit(q, i, c, q1, i1, c1, q2, i2, c2, q3, i3, c3, q4, i4, c4, q5, i5, c5, 
            init_unit_region_x, init_unit_region_y, init_unit_region_w, init_unit_region_h, plane_a, plane_b, plane_c);
        // 
        for (int x = 0; x < LAYER0_SIZE; x++) { 
#pragma HLS pipeline II=1
            for (int y = 0; y < LAYER0_SIZE; y++) {
#pragma HLS unroll
                if (((init_unit_region_x + x) >= WIDTH) || (init_unit_region_y + y) >= HEIGHT)
                    continue;
                float d = plane_a * (init_unit_region_x + x) + plane_b * (init_unit_region_y + y) + plane_c;
                float cost_tmp;
                if (init_hor_region_i != 0) {
                    if (d < DISP_MIN) cost_tmp = local_unary_cost[buff_id][y][x][0];
                    else if (d >= DISP_MAX - 1) cost_tmp = local_unary_cost[buff_id][y][x][DISP_MAX - 1];
                    else {
                        int d_l = std::floor(d);
                        int d_h = d_l + 1;
                        float f_h = d - d_l;
                        float f_l = 1.0 - f_h;
                        cost_tmp = f_l * local_unary_cost[buff_id][y][x][d_l] + 
                                   f_h * local_unary_cost[buff_id][y][x][d_h];
                    }
                }
                else {
                    if (d < DISP_MIN) cost_tmp = local_unary_cost[buff_id][y][x][0];
                    else if (d >= DISP_MAX - 1) cost_tmp = local_unary_cost[buff_id][y][x][DISP_MAX - 1];
                    else {
                        int d_l = std::floor(d);
                        int d_h = d_l + 1;
                        float f_h = d - d_l;
                        float f_l = 1.0 - f_h;
                        cost_tmp = f_l * local_unary_cost[buff_id][y][x][d_l] + 
                                   f_h * local_unary_cost[buff_id][y][x][d_h];
                    }
                }
                local_init_cost[y][x] = cost_tmp;
                local_init_label[y][x][0] = plane_a;
                local_init_label[y][x][1] = plane_b;
                local_init_label[y][x][2] = plane_c;
                local_init_disp[y][x] = d;
            }
        }
#else
    for (int y = 0; y < LAYER0_SIZE; y++) { 
#pragma HLS pipeline II=1
        for (int x = 0; x < LAYER0_SIZE; x++) {
#pragma HLS unroll
            if (((init_unit_region_x + x) >= WIDTH) || (init_unit_region_y + y) >= HEIGHT)
                continue;
            local_init_cost[y][x] = DISP_DDR_VAL_MAX;
            local_init_label[y][x][0] = 0;
            local_init_label[y][x][1] = 0;
            local_init_label[y][x][2] = 0;
            local_init_disp[y][x] = 0;
        }
    }
#endif
    // }
}

void layer0InitWrapper(int iter, int ver_region_num, int hor_region_num, int buff_id, 
    /* ap_uint<32>* Q,  ap_uint<32>& I,  ap_uint<32>& C,  ap_uint<32>* Q1, ap_uint<32>& I1, ap_uint<32>& C1, 
    ap_uint<32>* Q2, ap_uint<32>& I2, ap_uint<32>& C2, ap_uint<32>* Q3, ap_uint<32>& I3, ap_uint<32>& C3, 
    ap_uint<32>* Q4, ap_uint<32>& I4, ap_uint<32>& C4, ap_uint<32>* Q5, ap_uint<32>& I5, ap_uint<32>& C5,  */
    ap_uint<DISP_DDR_BIT> local_init_cost[LAYER0_SIZE][LAYER0_SIZE], label_range local_init_label[LAYER0_SIZE][LAYER0_SIZE][3], disp_range local_init_disp[LAYER0_SIZE][LAYER0_SIZE]) {
#pragma HLS inline off
    if (iter < (ver_region_num * hor_region_num - ver_region_num - 2)) {
        layer0InloopInit(iter, ver_region_num, hor_region_num, buff_id, /* Q, I, C, Q1, I1, C1, Q2, I2, C2, Q3, I3, C3, Q4, I4, C4, Q5, I5, C5, */ 
            local_init_cost, local_init_label, local_init_disp);
    }     
}

/*
template<size_t CUR_SIZE, int row_num, int layer_size>
void layerTransDDR(int iter, int ver_region_num, 
    float uram_cost[HEIGHT][CUR_SIZE], float uram_label[HEIGHT][CUR_SIZE][3], float uram_disp[HEIGHT][CUR_SIZE],
    float ddr_cost[WIDTH * HEIGHT], float ddr_label[WIDTH * HEIGHT * 3], float ddr_disp[WIDTH * HEIGHT]) {
#pragma HLS inline off
    // 
    int tran_ver_region_i = (iter - (ver_region_num + 1)) % ver_region_num;
    int tran_hor_region_i = (iter - (ver_region_num + 1)) / ver_region_num;
    int tran_unit_region_x, tran_unit_region_y, tran_unit_region_w, tran_unit_region_h;
    int tran_expn_region_x, tran_expn_region_y, tran_expn_region_w, tran_expn_region_h;
    genRegionInfo(WIDTH, HEIGHT, tran_hor_region_i, tran_ver_region_i, layer_size, tran_unit_region_x, tran_unit_region_y, tran_unit_region_w, tran_unit_region_h, 
        tran_expn_region_x, tran_expn_region_y, tran_expn_region_w, tran_expn_region_h);
    // 
    LOOP7:
    for (int x = 0; x < layer_size; x++) {
        LOOP8:
        for (int y = 0; y < layer_size; y++) {
#pragma HLS pipeline II=1
            if ((x >= tran_unit_region_w) || (y >= tran_unit_region_h))
                continue;
            ddr_cost[(tran_unit_region_y + y) * WIDTH + tran_unit_region_x + x] = uram_cost[tran_unit_region_y + y][(tran_hor_region_i % row_num) * layer_size + x];
            ddr_label[((tran_unit_region_y + y) * WIDTH + tran_unit_region_x + x) * 3 + 0] = uram_label[tran_unit_region_y + y][(tran_hor_region_i % row_num) * layer_size + x][0];
            ddr_label[((tran_unit_region_y + y) * WIDTH + tran_unit_region_x + x) * 3 + 1] = uram_label[tran_unit_region_y + y][(tran_hor_region_i % row_num) * layer_size + x][1];
            ddr_label[((tran_unit_region_y + y) * WIDTH + tran_unit_region_x + x) * 3 + 2] = uram_label[tran_unit_region_y + y][(tran_hor_region_i % row_num) * layer_size + x][2];
            ddr_disp[(tran_unit_region_y + y) * WIDTH + tran_unit_region_x + x] = uram_disp[tran_unit_region_y + y][(tran_hor_region_i % row_num) * layer_size + x];
        }
    }
}

template<size_t CUR_SIZE, int row_num, int layer_size, typename cost_blk, typename label_blk, typename disp_blk>
void layerTransLayer(int iter, int ver_region_num, 
    hls::stream_of_blocks<cost_blk> &cost_blk_out, hls::stream_of_blocks<label_blk> &label_blk_out, hls::stream_of_blocks<disp_blk> &disp_blk_out, 
    float uram_cost[HEIGHT][CUR_SIZE], float uram_label[HEIGHT][CUR_SIZE][3], float uram_disp[HEIGHT][CUR_SIZE]) {
#pragma HLS inline off
    hls::write_lock<cost_blk> cost_bout(cost_blk_out);
    hls::write_lock<label_blk> label_bout(label_blk_out);
    hls::write_lock<disp_blk> disp_bout(disp_blk_out);
    // #pragma HLS array_partition variable=bout type=block factor=5 dim=1
    int tran_ver_region_i = (iter - (ver_region_num + 1)) % ver_region_num;
    int tran_hor_region_i = (iter - (ver_region_num + 1)) / ver_region_num;
    int tran_unit_region_x, tran_unit_region_y, tran_unit_region_w, tran_unit_region_h;
    int tran_expn_region_x, tran_expn_region_y, tran_expn_region_w, tran_expn_region_h;
    genRegionInfo(WIDTH, HEIGHT, tran_hor_region_i, tran_ver_region_i, layer_size, tran_unit_region_x, tran_unit_region_y, tran_unit_region_w, tran_unit_region_h, 
        tran_expn_region_x, tran_expn_region_y, tran_expn_region_w, tran_expn_region_h);
    // 
    LOOP7:
    for (int y = 0; y < layer_size; y++) {
        LOOP8:
        for (int x = 0; x < layer_size; x++) {
#pragma HLS pipeline II=1
            if ((x >= tran_unit_region_w) || (y >= tran_unit_region_h))
                continue;
            cost_bout[y * layer_size + x] = uram_cost[tran_unit_region_y + y][(tran_hor_region_i % row_num) * layer_size + x];
            label_bout[(y * layer_size + x) * 3 + 0] = uram_label[tran_unit_region_y + y][(tran_hor_region_i % row_num) * layer_size + x][0];
            label_bout[(y * layer_size + x) * 3 + 1] = uram_label[tran_unit_region_y + y][(tran_hor_region_i % row_num) * layer_size + x][1];
            label_bout[(y * layer_size + x) * 3 + 2] = uram_label[tran_unit_region_y + y][(tran_hor_region_i % row_num) * layer_size + x][2];
            disp_bout[y * layer_size + x] = uram_disp[tran_unit_region_y + y][(tran_hor_region_i % row_num) * layer_size + x];
        }
    }
}
*/

void layer0Trans(int iter, int ver_region_num, int hor_region_num, int blk_num,
#if HLS_LAYER0_ONLY == 0
    hls::stream_of_blocks<layer0_cost_blk> &cost_blk_out, hls::stream_of_blocks<layer0_label_blk> &label_blk_out, hls::stream_of_blocks<layer0_disp_blk> &disp_blk_out,
    hls::stream_of_blocks<layer0_cost_blk> &init_cost_blk_out, hls::stream_of_blocks<layer0_label_blk> &init_label_blk_out, hls::stream_of_blocks<layer0_disp_blk> &init_disp_blk_out,
#else
    float cost_bout[LAYER0_SIZE * LAYER0_SIZE], float label_bout[LAYER0_SIZE * LAYER0_SIZE * 3], float disp_bout[LAYER0_SIZE * LAYER0_SIZE], 
#endif
    ap_uint<DISP_DDR_BIT> local_trans_cost[LAYER0_SIZE][LAYER0_SIZE], label_range local_trans_label[LAYER0_SIZE][LAYER0_SIZE][3], disp_range local_trans_disp[LAYER0_SIZE][LAYER0_SIZE]) {
#pragma HLS inline off
    // 
    if (blk_num < LAYER0_VER_REGION_NUM * (LAYER1_SIZE / LAYER0_SIZE) * 2) {
        hls::write_lock<layer0_cost_blk> init_cost_bout(init_cost_blk_out);
        hls::write_lock<layer0_label_blk> init_label_bout(init_label_blk_out);
        hls::write_lock<layer0_disp_blk> init_disp_bout(init_disp_blk_out);
        for (int y = 0; y < LAYER0_SIZE; y++) {
            for (int x = 0; x < LAYER0_SIZE; x++) {
#pragma HLS pipeline II=1
                init_cost_bout[y * LAYER0_SIZE + x] = local_trans_cost[y][x];
                init_label_bout[(y * LAYER0_SIZE + x) * 3 + 0] = local_trans_label[y][x][0];
                init_label_bout[(y * LAYER0_SIZE + x) * 3 + 1] = local_trans_label[y][x][1];
                init_label_bout[(y * LAYER0_SIZE + x) * 3 + 2] = local_trans_label[y][x][2];
                init_disp_bout[y * LAYER0_SIZE + x] = local_trans_disp[y][x];
            }
        }
    }
    else {
        hls::write_lock<layer0_cost_blk> cost_bout(cost_blk_out);
        hls::write_lock<layer0_label_blk> label_bout(label_blk_out);
        hls::write_lock<layer0_disp_blk> disp_bout(disp_blk_out);
        LOOP7:
        for (int y = 0; y < LAYER0_SIZE; y++) {
            LOOP8:
            for (int x = 0; x < LAYER0_SIZE; x++) {
#pragma HLS pipeline II=1
                cost_bout[y * LAYER0_SIZE + x] = local_trans_cost[y][x];
                label_bout[(y * LAYER0_SIZE + x) * 3 + 0] = local_trans_label[y][x][0];
                label_bout[(y * LAYER0_SIZE + x) * 3 + 1] = local_trans_label[y][x][1];
                label_bout[(y * LAYER0_SIZE + x) * 3 + 2] = local_trans_label[y][x][2];
                disp_bout[y * LAYER0_SIZE + x] = local_trans_disp[y][x];
            }
        }
    }
}

void layer0Transnew(int iter, int ver_region_num, int hor_region_num, int blk_num,
#if HLS_LAYER0_ONLY == 0
    /*ap_uint<DISP_DDR_BIT> ddr_cost[WIDTH * HEIGHT], label_range ddr_label[WIDTH * HEIGHT * 3], */disp_range ddr_disp[WIDTH * HEIGHT],
#else
    float cost_bout[LAYER0_SIZE * LAYER0_SIZE], float label_bout[LAYER0_SIZE * LAYER0_SIZE * 3], float disp_bout[LAYER0_SIZE * LAYER0_SIZE], 
#endif
    /*ap_uint<DISP_DDR_BIT> local_trans_cost[LAYER0_SIZE][LAYER0_SIZE], label_range local_trans_label[LAYER0_SIZE][LAYER0_SIZE][3], */disp_range local_trans_disp[LAYER0_SIZE][LAYER0_SIZE]) {
#pragma HLS inline off
//#pragma HLS array_partition variable=local_trans_cost type=complete dim=2
//#pragma HLS array_partition variable=local_trans_label type=complete dim=2
//#pragma HLS array_partition variable=local_trans_label type=complete dim=3
#pragma HLS array_partition variable=local_trans_disp type=complete dim=2
    // 
    int tran_ver_region_i = (iter - (ver_region_num + 2)) % ver_region_num;
    int tran_hor_region_i = (iter - (ver_region_num + 2)) / ver_region_num;
    int tran_unit_region_x, tran_unit_region_y, tran_unit_region_w, tran_unit_region_h;
    int tran_expn_region_x, tran_expn_region_y, tran_expn_region_w, tran_expn_region_h;
    genRegionInfo(WIDTH, HEIGHT, tran_hor_region_i, tran_ver_region_i, LAYER0_SIZE, tran_unit_region_x, tran_unit_region_y, tran_unit_region_w, tran_unit_region_h, 
                  tran_expn_region_x, tran_expn_region_y, tran_expn_region_w, tran_expn_region_h); 
    
        for (int y = 0; y < LAYER0_SIZE; y++) {
            for (int x = 0; x < LAYER0_SIZE; x++) {   
#pragma HLS pipeline II=1
if ((x >= tran_unit_region_w) || (y >= tran_unit_region_h))
                continue;
//                ddr_cost[(tran_unit_region_y + y) * WIDTH + tran_unit_region_x + x] = local_trans_cost[y][x];
//                ddr_label[((tran_unit_region_y + y) * WIDTH + tran_unit_region_x + x) * 3 + 0] = local_trans_label[y][x][0];
//                ddr_label[((tran_unit_region_y + y) * WIDTH + tran_unit_region_x + x) * 3 + 1] = local_trans_label[y][x][1];
//                ddr_label[((tran_unit_region_y + y) * WIDTH + tran_unit_region_x + x) * 3 + 2] = local_trans_label[y][x][2];
                ddr_disp[(tran_unit_region_y + y) * WIDTH + tran_unit_region_x + x] = local_trans_disp[y][x];
//                float ddd = local_trans_disp[y][x] >> 8;
//                if(ddd > 120) printf("iter = %d, tran_ver_region_i = %d, tran_hor_region_i = %d, tran_unit_region_w = %d, tran_unit_region_h = %d, x = %d, y = %d, disp = %f\n",iter, tran_ver_region_i, tran_hor_region_i, tran_unit_region_w, tran_unit_region_h, x, y, ddd);
            }
        }

}


void layer0TransWrapper(int iter, int ver_region_num, int hor_region_num, 
#if HLS_LAYER0_ONLY == 0
    hls::stream_of_blocks<layer0_cost_blk> &cost_blk_out, hls::stream_of_blocks<layer0_label_blk> &label_blk_out, hls::stream_of_blocks<layer0_disp_blk> &disp_blk_out, 
    hls::stream_of_blocks<layer0_cost_blk> &init_cost_blk_out, hls::stream_of_blocks<layer0_label_blk> &init_label_blk_out, hls::stream_of_blocks<layer0_disp_blk> &init_disp_blk_out,
#else
    float cost_blk_out[LAYER0_SIZE * LAYER0_SIZE], float label_blk_out[LAYER0_SIZE * LAYER0_SIZE * 3], float disp_blk_out[LAYER0_SIZE * LAYER0_SIZE], 
#endif
    ap_uint<DISP_DDR_BIT> local_trans_cost[LAYER0_SIZE][LAYER0_SIZE], label_range local_trans_label[LAYER0_SIZE][LAYER0_SIZE][3], disp_range local_trans_disp[LAYER0_SIZE][LAYER0_SIZE]) {
#pragma HLS inline off
    static int blk_num;
    if (iter >= (ver_region_num + 2)){
        layer0Trans(iter, ver_region_num, hor_region_num, blk_num, cost_blk_out, label_blk_out, disp_blk_out, init_cost_blk_out, init_label_blk_out, init_disp_blk_out, local_trans_cost, local_trans_label, local_trans_disp);
        blk_num++;
    }
    else {
        blk_num = 0;
    }
}

void layer0TransWrapper(int iter, int ver_region_num, int hor_region_num, 
#if HLS_LAYER0_ONLY == 0
    /*ap_uint<DISP_DDR_BIT> ddr_cost[WIDTH * HEIGHT], label_range ddr_label[WIDTH * HEIGHT * 3], */disp_range ddr_disp[WIDTH * HEIGHT],
#else
    float cost_blk_out[LAYER0_SIZE * LAYER0_SIZE], float label_blk_out[LAYER0_SIZE * LAYER0_SIZE * 3], float disp_blk_out[LAYER0_SIZE * LAYER0_SIZE], 
#endif
    /*ap_uint<DISP_DDR_BIT> local_trans_cost[LAYER0_SIZE][LAYER0_SIZE], label_range local_trans_label[LAYER0_SIZE][LAYER0_SIZE][3], */disp_range local_trans_disp[LAYER0_SIZE][LAYER0_SIZE]) {
#pragma HLS inline off
    static int blk_num;
    if (iter >= (ver_region_num + 2)){
        layer0Transnew(iter, ver_region_num, hor_region_num, blk_num, /*ddr_cost, ddr_label, */ddr_disp, /*local_trans_cost, local_trans_label, */local_trans_disp);
        blk_num++;
    }
    else {
        blk_num = 0;
    }
}

void layer1Trans(int iter, 
#if LAYER2_ENABLE == 0
    int tran_unit_region_x, int tran_unit_region_y, int tran_unit_region_w, int tran_unit_region_h,
    ap_uint<DISP_DDR_BIT> ddr_cost[WIDTH * HEIGHT], label_range ddr_label[WIDTH * HEIGHT * 3], disp_range ddr_disp[WIDTH * HEIGHT],
#elif HLS_LAYER1_ONLY == 0
    hls::stream_of_blocks<layer1_cost_blk> &cost_blk_out, hls::stream_of_blocks<layer1_label_blk> &label_blk_out, hls::stream_of_blocks<layer1_disp_blk> &disp_blk_out,
#else
    float cost_bout[LAYER1_SIZE * LAYER1_SIZE], float label_bout[LAYER1_SIZE * LAYER1_SIZE * 3], float disp_bout[LAYER1_SIZE * LAYER1_SIZE], 
#endif
    ap_uint<DISP_DDR_BIT> local_trans_cost[LAYER1_SIZE][LAYER1_SIZE], label_range local_trans_label[LAYER1_SIZE][LAYER1_SIZE][3], disp_range local_trans_disp[LAYER1_SIZE][LAYER1_SIZE]) {
#pragma HLS inline off
#pragma HLS array_partition variable=local_trans_cost type=complete dim=2
#pragma HLS array_partition variable=local_trans_label type=complete dim=2
#pragma HLS array_partition variable=local_trans_label type=complete dim=3
#pragma HLS array_partition variable=local_trans_disp type=complete dim=2
#if HLS_LAYER1_ONLY == 0 && LAYER2_ENABLE == 1
    hls::write_lock<layer1_cost_blk> cost_bout(cost_blk_out);
    hls::write_lock<layer1_label_blk> label_bout(label_blk_out);
    hls::write_lock<layer1_disp_blk> disp_bout(disp_blk_out);
#endif
    // 
    LOOP7:
    for (int y = 0; y < LAYER1_SIZE; y++) {
        LOOP8:
        for (int x = 0; x < LAYER1_SIZE; x++) {
#pragma HLS pipeline II=1
            if ((x >= tran_unit_region_w) || (y >= tran_unit_region_h))
                continue;
#if LAYER2_ENABLE == 1
            cost_bout[y * LAYER1_SIZE + x] = local_trans_cost[y][x];
            label_bout[(y * LAYER1_SIZE + x) * 3 + 0] = local_trans_label[y][x][0];
            label_bout[(y * LAYER1_SIZE + x) * 3 + 1] = local_trans_label[y][x][1];
            label_bout[(y * LAYER1_SIZE + x) * 3 + 2] = local_trans_label[y][x][2];
            disp_bout[y * LAYER1_SIZE + x] = local_trans_disp[y][x];
#else
            ddr_cost[(tran_unit_region_y + y) * WIDTH + tran_unit_region_x + x] = local_trans_cost[y][x];
            ddr_label[((tran_unit_region_y + y) * WIDTH + tran_unit_region_x + x) * 3 + 0] = local_trans_label[y][x][0];
            ddr_label[((tran_unit_region_y + y) * WIDTH + tran_unit_region_x + x) * 3 + 1] = local_trans_label[y][x][1];
            ddr_label[((tran_unit_region_y + y) * WIDTH + tran_unit_region_x + x) * 3 + 2] = local_trans_label[y][x][2];
            ddr_disp[(tran_unit_region_y + y) * WIDTH + tran_unit_region_x + x] = local_trans_disp[y][x];
#endif
        }
    }
}


void layer1TransWrapper(int l1_iter, 
#if LAYER2_ENABLE == 0
    int tran_unit_region_x, int tran_unit_region_y, int tran_unit_region_w, int tran_unit_region_h,
    ap_uint<DISP_DDR_BIT> ddr_cost[WIDTH * HEIGHT], label_range ddr_label[WIDTH * HEIGHT * 3], disp_range ddr_disp[WIDTH * HEIGHT],
#elif HLS_LAYER1_ONLY == 0
    hls::stream_of_blocks<layer1_cost_blk> &cost_blk_out, hls::stream_of_blocks<layer1_label_blk> &label_blk_out, hls::stream_of_blocks<layer1_disp_blk> &disp_blk_out,
#else
    float cost_blk_out[LAYER1_SIZE * LAYER1_SIZE], float label_blk_out[LAYER1_SIZE * LAYER1_SIZE * 3], float disp_blk_out[LAYER1_SIZE * LAYER1_SIZE], 
#endif
    ap_uint<DISP_DDR_BIT> local_trans_cost[LAYER1_SIZE][LAYER1_SIZE], label_range local_trans_label[LAYER1_SIZE][LAYER1_SIZE][3], disp_range local_trans_disp[LAYER1_SIZE][LAYER1_SIZE]) {
#pragma HLS inline off
    if (l1_iter >= (LAYER1_VER_REGION_NUM + 2)){
#if LAYER2_ENABLE == 1
        layer1Trans(l1_iter, cost_blk_out, label_blk_out, disp_blk_out, local_trans_cost, local_trans_label, local_trans_disp);
#else
        layer1Trans(l1_iter,  tran_unit_region_x, tran_unit_region_y, tran_unit_region_w, tran_unit_region_h, ddr_cost, ddr_label, ddr_disp, local_trans_cost, local_trans_label, local_trans_disp);
#endif
    }
}

/*
template<size_t CUR_SIZE>
void layerInloopCache(int iter, int ver_region_num, int hor_region_num, int unary_cost_y, 
    float local_unary_cost[4][CUR_SIZE][CUR_SIZE * 3][DISP_MAX], float ddr_unary_cost[WIDTH * HEIGHT * DISP_MAX],
    uchar local_im0_gray[4][CUR_SIZE][CUR_SIZE * 3], uchar ddr_im0_gray[WIDTH * HEIGHT]) {
#pragma HLS inline off
    int ver_region_i = (iter + 2) % ver_region_num;
    int hor_region_i = (iter + 2) / ver_region_num;
    // 
    int unit_region_x, unit_region_y, unit_region_w, unit_region_h;
    int expn_region_x, expn_region_y, expn_region_w, expn_region_h;
    genRegionInfo(WIDTH, HEIGHT, hor_region_i, ver_region_i, CUR_SIZE, unit_region_x, unit_region_y, unit_region_w, unit_region_h, 
        expn_region_x, expn_region_y, expn_region_w, expn_region_h);
    // 
    for (int y = 0; y < CUR_SIZE; y++) {
        for (int x = 0; x < CUR_SIZE * 3; x++) {
            if ((x >= expn_region_w) || (y >= unit_region_h))
                continue;
            for (int d = 0; d < DISP_MAX; d++) {
                local_unary_cost[unary_cost_y][y][x][d] = ddr_unary_cost[((unit_region_y + y) * WIDTH + (expn_region_x + x)) * DISP_MAX + d];
            }
        }
    }
    for (int y = 0; y < CUR_SIZE; y++) {
        for (int x = 0; x < CUR_SIZE * 3; x++) {
            if ((x >= expn_region_w) || (y >= unit_region_h))
                continue;
            local_im0_gray[unary_cost_y][y][x] = ddr_im0_gray[(unit_region_y + y) * WIDTH + (expn_region_x + x)];
        }
    }
}
*/

void layer0InloopCache(unary_cost_t local_buffer[2][DISP_DDR_MAX0], int iter, int ver_region_num, int hor_region_num, int unary_cost_y, 
    unary_cost_t local_unary_cost[4][LAYER0_SIZE][LAYER0_SIZE * 3][DISP_DDR_MAX], unary_cost_t0 ddr_unary_cost[WIDTH * HEIGHT * DISP_DDR_MAX0]) {
#pragma HLS inline
#pragma HLS array_partition variable=local_unary_cost type=complete dim=1
#pragma HLS array_partition variable=local_unary_cost type=complete dim=3
// DO_PRAGMA(HLS array_reshape variable=local_unary_cost type=cyclic factor=DISP_DDR_NUM dim=4)
// #pragma HLS interface mode=m_axi bundle=BUS_LAYER0 port=ddr_unary_cost offset=slave
    //
    assert(iter >= 0 && iter < (LAYER0_VER_REGION_NUM * LAYER0_HOR_REGION_NUM - 2));
    assert(unary_cost_y >= 0 && unary_cost_y < 4);
    int ver_region_i = (iter + 2) % LAYER0_VER_REGION_NUM;
    int hor_region_i = (iter + 2) / LAYER0_VER_REGION_NUM;
    // 
    int unit_region_x, unit_region_y, unit_region_w, unit_region_h;
    int expn_region_x, expn_region_y, expn_region_w, expn_region_h;
    genRegionInfo(WIDTH, HEIGHT, hor_region_i, ver_region_i, LAYER0_SIZE, unit_region_x, unit_region_y, unit_region_w, unit_region_h, 
        expn_region_x, expn_region_y, expn_region_w, expn_region_h);
    /*
    LOOP_COST0:
   for (int xyd = 0; xyd < DISP_DDR_MAX0; xyd++) {
#pragma HLS pipeline II=1
        int y = 0;
        int x = 0;
        int d = xyd;
//        if ((y >= unit_region_h) || (expn_region_x + x >= WIDTH))
//                continue;
        bool index = 0;
        int dd = d << 1;
        local_unary_cost[unary_cost_y][y][x][dd] = ddr_unary_cost[((unit_region_y + y) * WIDTH + (expn_region_x + x)) * DISP_DDR_MAX0 + d](127,0); 
        local_buffer[index][d] = ddr_unary_cost[((unit_region_y + y) * WIDTH + (expn_region_x + x)) * DISP_DDR_MAX0 + d](255,128);
    }
    */
   
    LOOP_COST1:
    for (int xyd = 0; xyd < DISP_DDR_MAX0 * (LAYER0_SIZE * LAYER0_SIZE * 3); xyd++) {
#pragma HLS pipeline II=1
        int y = (xyd) / (3 * LAYER0_SIZE * DISP_DDR_MAX0);
        int x = (xyd) % (3 * LAYER0_SIZE * DISP_DDR_MAX0) / DISP_DDR_MAX0;
        int d = (xyd) % (3 * LAYER0_SIZE * DISP_DDR_MAX0) % DISP_DDR_MAX0;
//        if ((y >= unit_region_h) || (expn_region_x + x >= WIDTH))
//               continue;
        bool index = (3 * LAYER0_SIZE * y + x) % 2;
        int dd = d << 1;
        local_unary_cost[unary_cost_y][y][x][dd] = ddr_unary_cost[((unit_region_y + y) * WIDTH + (expn_region_x + x)) * DISP_DDR_MAX0 + d](127,0); 
        local_buffer[index][d] = ddr_unary_cost[((unit_region_y + y) * WIDTH + (expn_region_x + x)) * DISP_DDR_MAX0 + d](255,128);
//        if ((y >= unit_region_h) || (expn_region_x + x >= WIDTH)) local_unary_cost[unary_cost_y][y][x][dd] = DISP_DDR_VAL_MAX;
        if ((dd >= DISP_DDR_MAX - 1) || (xyd < DISP_DDR_MAX0))
            continue;  
        int yy = (x == 0) ? (y - 1) : y;
        int xx = (x == 0) ? 26 : (x - 1);
        local_unary_cost[unary_cost_y][yy][xx][dd + 1] = local_buffer[!index][d];
        
    }
    LOOP_COST2:
    for (int xyd = 0; xyd <  DISP_DDR_MAX0; xyd++) {
#pragma HLS pipeline II=1
        int y = 9;
        int x = 0;
        int d = xyd;
        bool index = 1;
        int dd = d << 1;
//        if (y >= unit_region_h) 
//               continue;
//        if ((y >= unit_region_h) || (expn_region_x + x >= WIDTH) || (dd >= DISP_DDR_MAX - 1))
        if (dd >= DISP_DDR_MAX - 1)
            continue; 
        int yy = (x == 0) ? (y - 1) : y;
        int xx = (x == 0) ? 26 : (x - 1);
        local_unary_cost[unary_cost_y][yy][xx][dd + 1] = local_buffer[!index][d];
    }
    /*
    
    for (int xyd = 0; xyd < LAYER0_SIZE * LAYER0_SIZE * 3 * DISP_DDR_MAX0; xyd++) {
#pragma HLS pipeline II=1
        int y = xyd / (LAYER0_SIZE * 3 * DISP_DDR_MAX0);
        int x = xyd % (LAYER0_SIZE * 3 * DISP_DDR_MAX0) / DISP_DDR_MAX0;
        int d = xyd % (LAYER0_SIZE * 3 * DISP_DDR_MAX0) % DISP_DDR_MAX0; 
        int dd = d * 2;     
        local_unary_cost[unary_cost_y][y][x][dd] = ddr_unary_cost[((unit_region_y + y) * WIDTH + (expn_region_x + x)) * DISP_DDR_MAX0 + d](127,0);
        if (dd >= DISP_DDR_MAX - 1)
            continue; 
        local_unary_cost[unary_cost_y][y][x][dd + 1] = ddr_unary_cost[((unit_region_y + y) * WIDTH + (expn_region_x + x)) * DISP_DDR_MAX0 + d](255,128);
    }
    */
    
}

void layer0CacheWrapper(unary_cost_t local_buffer[2][DISP_DDR_MAX0], int iter, int unary_cost_y3, int buff_id, 
    unary_cost_t local_unary_cost[4][LAYER0_SIZE][LAYER0_SIZE * 3][DISP_DDR_MAX], 
    unary_cost_t0 ddr_unary_cost[WIDTH * HEIGHT * DISP_DDR_MAX0]) {
#pragma HLS inline off
// #pragma HLS array_partition variable=local_unary_cost type=complete dim=1
// #pragma HLS array_partition variable=local_unary_cost type=complete dim=2
// DO_PRAGMA(HLS array_reshape variable=local_unary_cost type=cyclic factor=DISP_DDR_NUM dim=4)   
// #pragma HLS interface mode=m_axi bundle=BUS_LAYER0 port=ddr_unary_cost offset=slave
    assert(buff_id == 0 || buff_id == 1);
    /* cache unary cost from DDR */
    if (iter < (LAYER0_VER_REGION_NUM * LAYER0_HOR_REGION_NUM - 2)) {
        layer0InloopCache(local_buffer, iter, LAYER0_VER_REGION_NUM, LAYER0_HOR_REGION_NUM, unary_cost_y3, local_unary_cost, ddr_unary_cost);
    }
    // if (iter < (LAYER0_VER_REGION_NUM * LAYER0_HOR_REGION_NUM - LAYER0_HOR_REGION_NUM - 3)) {
    //     layer0InloopCacheInit(iter, LAYER0_VER_REGION_NUM, LAYER0_HOR_REGION_NUM, buff_id, local_unary_cost_init, ddr_unary_cost);
    // }
}

void layer1InloopCache(int iter, int unary_cost_y, 
    unary_cost_t local_unary_cost[LAYER1_SIZE][4][LAYER1_SIZE * 3][DISP_DDR_MAX], unary_cost_t ddr_unary_cost[WIDTH * HEIGHT * DISP_DDR_MAX],
    uchar local_im0_gray[LAYER1_SIZE][4][LAYER1_SIZE * 3], uchar ddr_im0_gray[WIDTH * HEIGHT]) {
#pragma HLS inline off
#pragma HLS array_partition variable=local_unary_cost type=complete dim=2
#pragma HLS array_partition variable=local_unary_cost type=complete dim=3
//DO_PRAGMA(HLS array_reshape variable=local_unary_cost type=cyclic factor=DISP_DDR_NUM dim=4)
#pragma HLS array_partition variable=local_im0_gray type=complete dim=2
#pragma HLS array_partition variable=local_im0_gray type=complete dim=3
// #pragma HLS interface mode=m_axi bundle=BUS_LAYER1 port=ddr_unary_cost offset=slave
// #pragma HLS interface mode=m_axi bundle=BUS_LAYER1 port=ddr_im0_gray offset=slave
    /*  */
    assert(iter >= 0 && iter < (LAYER1_VER_REGION_NUM * LAYER1_HOR_REGION_NUM - 2));
    assert(unary_cost_y >= 0 && unary_cost_y < 4);
    //
    int ver_region_i = (iter + 2) % LAYER1_VER_REGION_NUM;
    int hor_region_i = (iter + 2) / LAYER1_VER_REGION_NUM;
    // 
    int unit_region_x, unit_region_y, unit_region_w, unit_region_h;
    int expn_region_x, expn_region_y, expn_region_w, expn_region_h;
    genRegionInfo(WIDTH, HEIGHT, hor_region_i, ver_region_i, LAYER1_SIZE, unit_region_x, unit_region_y, unit_region_w, unit_region_h, 
        expn_region_x, expn_region_y, expn_region_w, expn_region_h);
    // 
    // int cach_x = (iter + 2) % 4;
    // 
    LOOP_COST:
    for (int xyd = 0; xyd < LAYER1_SIZE * LAYER1_SIZE * 3 * DISP_DDR_MAX; xyd++) {
#pragma HLS pipeline II=1
        int y = xyd / (LAYER1_SIZE * 3 * DISP_DDR_MAX);
        int x = xyd % (LAYER1_SIZE * 3 * DISP_DDR_MAX) / DISP_DDR_MAX;
        int d = xyd % (LAYER1_SIZE * 3 * DISP_DDR_MAX) % DISP_DDR_MAX;
        local_unary_cost[y][unary_cost_y][x][d] = ddr_unary_cost[((unit_region_y + y) * WIDTH + (expn_region_x + x)) * DISP_DDR_MAX + d]; 
    }
    LOOP_IMAGE: 
    for (int xy = 0; xy < LAYER1_SIZE * LAYER1_SIZE * 3; xy++) {
#pragma HLS pipeline II=1
        int y = xy / (LAYER1_SIZE * 3);
        int x = xy % (LAYER1_SIZE * 3);     
        local_im0_gray[y][unary_cost_y][x] = ddr_im0_gray[(unit_region_y + y) * WIDTH + (expn_region_x + x)]; 
    }
}


void layer1CacheWrapper(int l1_iter, int unary_cost_y, 
    unary_cost_t local_unary_cost[LAYER1_SIZE][4][LAYER1_SIZE * 3][DISP_DDR_MAX], unary_cost_t ddr_unary_cost[WIDTH * HEIGHT * DISP_DDR_MAX],
    uchar local_im0_gray[LAYER1_SIZE][4][LAYER1_SIZE * 3], uchar ddr_im0_gray[WIDTH * HEIGHT]) {
#pragma HLS inline off
    // cache unary & binary cost from DDR
    if ((l1_iter >= 0) && (l1_iter < (LAYER1_VER_REGION_NUM * LAYER1_HOR_REGION_NUM - 2))) {
        layer1InloopCache(l1_iter, unary_cost_y, local_unary_cost, ddr_unary_cost, local_im0_gray, ddr_im0_gray);
    }
}

/*
template<size_t CUR_SIZE>
void costLabelUpdate(int unary_cost_y0, int unary_cost_y1, int unary_cost_y2, 
    int expn_region_x, int expn_region_y, int expn_region_w, int expn_region_h, int unit_region_y,
    float plane_a, float plane_b, float plane_c, 
    float local_unary_cost[4][CUR_SIZE][CUR_SIZE * 3][DISP_MAX], uchar local_im0_gray_cyclic[4][CUR_SIZE][CUR_SIZE * 3], 
    float local_cost[CUR_SIZE * 3][CUR_SIZE * 3], float local_label[CUR_SIZE * 3][CUR_SIZE * 3][3], float local_disp[CUR_SIZE * 3][CUR_SIZE * 3],
    bool is_binary_enable, bool is_debug = false) {
#pragma HLS array_partition variable=local_unary_cost type=complete dim=1
#pragma HLS inline
    //
    const int nb_dx_dy[8][2] = {{-1, 0}, {+1, 0}, {0, -1}, {0, 1}, {-1, -1}, {1, -1}, {-1, 1}, {1, 1}};
    const ap_uint<5> region_size = CUR_SIZE;
     gray image re-order 
    uchar local_im0_gray[CUR_SIZE * 3][CUR_SIZE * 3];
#pragma HLS array_partition variable=local_im0_gray type=complete dim=1
    for (int y = 0; y < CUR_SIZE * 3; y++) {
#pragma HLS pipeline II=1
        int y_remainder = y % region_size;
        int local_y_id = 0;
        if (expn_region_y == unit_region_y) {
            if (y < CUR_SIZE)
                local_y_id = unary_cost_y1;
            else if (y < CUR_SIZE * 2)
                local_y_id = unary_cost_y2;
        }
        else {
            if (y < CUR_SIZE)
                local_y_id = unary_cost_y0;
            else if (y < CUR_SIZE * 2)
                local_y_id = unary_cost_y1;
            else
                local_y_id = unary_cost_y2;
        }
        for (int x = 0; x < CUR_SIZE * 3; x++) {
#pragma HLS unroll
            local_im0_gray[y][x] = local_im0_gray_cyclic[local_y_id][y_remainder][x];
        }
    }
    // 
    LOOP1:
    for (int y = 0; y < CUR_SIZE * 3; y++) {
#pragma HLS pipeline II=1
        // int x_quotient = x / region_size;
        int y_remainder = y % region_size;
        int local_y_id = 0;
        if (expn_region_y == unit_region_y) {
            if (y < CUR_SIZE)
                local_y_id = unary_cost_y1;
            else if (y < CUR_SIZE * 2)
                local_y_id = unary_cost_y2;
        }
        else {
            if (y < CUR_SIZE)
                local_y_id = unary_cost_y0;
            else if (y < CUR_SIZE * 2)
                local_y_id = unary_cost_y1;
            else
                local_y_id = unary_cost_y2;
        }
        LOOP2:
        for (int x = 0; x < CUR_SIZE * 3; x++) {
#pragma HLS unroll
            if ((x >= expn_region_w) || (y >= expn_region_h))
                continue;
            //
            float d_prop = plane_a * (expn_region_x + x) + plane_b * (expn_region_y + y) + plane_c;
            float cost_unary_prop = 0.0;
            float cost_binary_prop = 0.0;
            float d_orig = local_disp[y][x];
            float cost_unary_orig = local_cost[y][x];
            float cost_binary_orig = 0.0;
            uchar p_gray = local_im0_gray[y][x];
            // unary cost
            if (d_prop < DISP_MIN) cost_unary_prop = local_unary_cost[local_y_id][y_remainder][x][0];
            else if (d_prop >= DISP_MAX - 1) cost_unary_prop = local_unary_cost[local_y_id][y_remainder][x][DISP_MAX - 1];
            else {
                int d_l = std::floor(d_prop);
                int d_h = d_l + 1;
                float f_h = d_prop - d_l;
                float f_l = 1.0 - f_h;
                cost_unary_prop = f_l * local_unary_cost[local_y_id][y_remainder][x][d_l] + f_h * local_unary_cost[local_y_id][y_remainder][x][d_h];
            }
            // binary cost
            if (is_binary_enable == true) {
                for(int nb = 0; nb < 8; nb++) {
                    int x_nb = x + nb_dx_dy[nb][0]; 
                    int y_nb = y + nb_dx_dy[nb][1];
                    if (x_nb < 0 || x_nb >= expn_region_w || y_nb < 0 || y_nb >= expn_region_h)
                        continue;
                    float d_nb = local_disp[y_nb][x_nb];
                    uchar p_gray_nb = local_im0_gray[y_nb][x_nb];
                    float weight = (std::abs(p_gray_nb - p_gray) < 10) ? 0.3 : 0.05;
                    cost_binary_prop += weight * std::abs(d_prop - d_nb);
                    cost_binary_orig += weight * std::abs(d_orig - d_nb);
                }
            }
            // update local buffer
            if ((cost_unary_prop + cost_binary_prop) < (cost_unary_orig + cost_binary_orig)) {
                local_cost[y][x] = cost_unary_prop;
                local_label[y][x][0] = plane_a; local_label[y][x][1] = plane_b; local_label[y][x][2] = plane_c;
                local_disp[y][x] = d_prop;
            }
        }
    }
}
*/

ap_uint<8> bit_select_108b(ap_uint<4> d_l, ap_uint<108> c_l_in) {
#pragma HLS inline off
    ap_uint<54>  c_l_2 = (d_l[3] == 0) ? c_l_in.range(53, 0) : c_l_in.range(107, 54);
    ap_uint<27>  c_l_3 = (d_l[2] == 0) ? c_l_2.range(26, 0) : c_l_2.range(53, 27);
    ap_uint<28>  c_l_3_tmp = c_l_3 << 1;
    ap_uint<14>  c_l_4 = (d_l[1] == 0) ? c_l_3_tmp.range(13, 0) : c_l_3_tmp.range(27, 14);
    ap_uint<7>   c_l_5 = (d_l[0] == 0) ? c_l_4.range(6, 0)  : c_l_4.range(13, 7);
    return c_l_5;
}

ap_uint<16> bit_select_256b(ap_uint<4> d_l, ap_uint<256> c_l_in) {
#pragma HLS inline off
    ap_uint<128> c_l_2 = (d_l[3] == 0) ? c_l_in.range(127, 0) : c_l_in.range(255, 128);
    ap_uint<64>  c_l_3 = (d_l[2] == 0) ? c_l_2.range(63, 0) : c_l_2.range(127, 64);
    ap_uint<32>  c_l_4 = (d_l[1] == 0) ? c_l_3.range(31, 0) : c_l_3.range(63, 32);
    ap_uint<16>  c_l_5 = (d_l[0] == 0) ? c_l_4.range(15, 0)  : c_l_4.range(31, 16);
    return c_l_5;
}

ap_uint<8> bit_select_128b(ap_uint<4> d_l, ap_uint<128> c_l_in) {
#pragma HLS inline off
    ap_uint<64>  c_l_2 = (d_l[3] == 0) ? c_l_in.range(63, 0) : c_l_in.range(127, 64);
    ap_uint<32>  c_l_3 = (d_l[2] == 0) ? c_l_2.range(31, 0) : c_l_2.range(63, 32);
    ap_uint<16>  c_l_4 = (d_l[1] == 0) ? c_l_3.range(15, 0) : c_l_3.range(31, 16);
    ap_uint<8>   c_l_5 = (d_l[0] == 0) ? c_l_4.range(7, 0)  : c_l_4.range(15, 8);
    return c_l_5;
}

ap_uint<7> bit_select_112b(ap_uint<4> d_l, ap_uint<112> c_l_in) {
#pragma HLS inline off
    ap_uint<56>  c_l_2 = (d_l[3] == 0) ? c_l_in.range(55, 0) : c_l_in.range(111, 56);
    ap_uint<28>  c_l_3 = (d_l[2] == 0) ? c_l_2.range(27, 0) : c_l_2.range(55, 28);
    ap_uint<14>  c_l_4 = (d_l[1] == 0) ? c_l_3.range(13, 0) : c_l_3.range(27, 14);
    ap_uint<7>   c_l_5 = (d_l[0] == 0) ? c_l_4.range(6, 0)  : c_l_4.range(13, 7);
    return c_l_5;
}

ap_uint<6> bit_select_96b(ap_uint<4> d_l, ap_uint<96> c_l_in) {
#pragma HLS inline off
    ap_uint<48>  c_l_2 = (d_l[3] == 0) ? c_l_in.range(47, 0) : c_l_in.range(95, 48);
    ap_uint<24>  c_l_3 = (d_l[2] == 0) ? c_l_2.range(23, 0) : c_l_2.range(47, 24);
    ap_uint<12>  c_l_4 = (d_l[1] == 0) ? c_l_3.range(11, 0) : c_l_3.range(23, 12);
    ap_uint<6>   c_l_5 = (d_l[0] == 0) ? c_l_4.range(5, 0)  : c_l_4.range(11, 6);
    return c_l_5;
}

void layer0CostLabelUpdate(ap_uint<3> unary_cost_y0, ap_uint<3> unary_cost_y1, ap_uint<3> unary_cost_y2, ap_uint<11> expn_region_x, ap_uint<11> expn_region_y, 
    ap_uint<5> expn_region_w, ap_uint<5> expn_region_h, ap_uint<11> unit_region_y,
    label_range plane_a[LAYER0_PROP_NUM], label_range plane_b[LAYER0_PROP_NUM], label_range plane_c[LAYER0_PROP_NUM], 
    unary_cost_t local_unary_cost[4][LAYER0_SIZE][LAYER0_SIZE * 3][DISP_DDR_MAX], 
    ap_uint<DISP_DDR_BIT> local_cost[LAYER0_SIZE * 3][LAYER0_SIZE * 3], label_range local_label[LAYER0_SIZE * 3][LAYER0_SIZE * 3][3], disp_range local_disp[LAYER0_SIZE * 3][LAYER0_SIZE * 3]) {
#pragma HLS array_partition variable=plane_a type=complete dim=1
#pragma HLS array_partition variable=plane_b type=complete dim=1
#pragma HLS array_partition variable=plane_c type=complete dim=1
#pragma HLS array_partition variable=local_unary_cost type=complete dim=1
#pragma HLS array_partition variable=local_unary_cost type=complete dim=3
//DO_PRAGMA(HLS array_reshape variable=local_unary_cost type=cyclic factor=DISP_DDR_NUM dim=4) 
// #pragma HLS INTERFACE mode=ap_memory port=local_cost storage_type=ram_2p
#pragma HLS array_partition variable=local_cost type=complete dim=2
#pragma HLS array_partition variable=local_label type=complete dim=2
#pragma HLS array_partition variable=local_label type=complete dim=3
#pragma HLS array_partition variable=local_disp type=complete dim=2
#pragma HLS inline off
    assert((unary_cost_y0 != unary_cost_y1) && (unary_cost_y0 != unary_cost_y2) && (unary_cost_y1 != unary_cost_y2));
    assert(unary_cost_y0 >= 0 && unary_cost_y0 < 4 && unary_cost_y1 >= 0 && unary_cost_y1 < 4 && unary_cost_y2 >= 0 && unary_cost_y2 < 4);
    // 
    bool is_top_region = (expn_region_y == unit_region_y);
             
    for (int py = 0; py < LAYER0_PROP_NUM * LAYER0_SIZE * 3; py++) {
    // for (int p = 0; p < 3; p++) {
    //     // LOOP_CALC_HOR:
    //     for (int x = 0; x < LAYER0_SIZE * 3; x++) {
#pragma HLS loop_flatten
#pragma HLS pipeline II=1
            // LOOP_CALC_VER:
            // int x_base = buf_x[x/LAYER0_SIZE];
            // int x_ofst = x%LAYER0_SIZE;
            ap_uint<4> p = py / (3 * LAYER0_SIZE); // (4, 0)
            int y = py % (3 * LAYER0_SIZE); // (5, 0)
            ap_uint<4> y_ofst = y % LAYER0_SIZE; // (4, 0)
            ap_uint<3> y_base; // (3, 0)
            if (is_top_region) {
                if (y < LAYER0_SIZE)
                    y_base = unary_cost_y1;
                else if (y < LAYER0_SIZE * 2)
                    y_base = unary_cost_y2;
            }
            else {
                if (y < LAYER0_SIZE)
                    y_base = unary_cost_y0;
                else if (y < LAYER0_SIZE * 2)
                    y_base = unary_cost_y1;
                else
                    y_base = unary_cost_y2;
            }
            ap_int<17> d_prop_base = plane_b[p] * (expn_region_y + y) + plane_c[p];
            for (int x = 0; x < LAYER0_SIZE * 3; x++) {
#pragma HLS unroll
                ap_int<17> d_prop = d_prop_base + plane_a[p] * (expn_region_x + x);
                ap_uint<DISP_DDR_BIT> cost_unary_prop = 0;
                ap_uint<DISP_DDR_BIT> cost_unary_orig = local_cost[y][x];
                /* unary cost */
                // if (d_prop < DISP_MIN) cost_unary_prop = local_unary_cost[y_base][y_ofst][x][0].range(6-1,0) << 1;
                // else if (d_prop.range(16,8) >= DISP_MAX - 1) cost_unary_prop = local_unary_cost[y_base][y_ofst][x][DISP_DDR_MAX-1].range(107,101);
                if (d_prop< DISP_MIN) cost_unary_prop = local_unary_cost[y_base][y_ofst][x][0].range(DISP_DDR_BIT-1,0);
                else if (d_prop.range(16,8) >= DISP_MAX - 1) cost_unary_prop = local_unary_cost[y_base][y_ofst][x][DISP_DDR_MAX-1].range(DISP_DDR_BIT*DISP_DDR_NUM-1,DISP_DDR_BIT*DISP_DDR_NUM-DISP_DDR_BIT);
                else {
                    // int d_l = d_prop.to_int();
                    ap_uint<8> d_l_int = d_prop.range(15,8);
                    if (d_prop[7]==1)
                        d_l_int = d_l_int + 1;
                    unary_cost_t c_l_int = local_unary_cost[y_base][y_ofst][x][d_l_int.range(7, DISP_DDR_NUM_LOG2)];
                    // ap_uint<8>  c_l_5_int =  bit_select_108b(d_l_int.range(DISP_DDR_NUM_LOG2-1, 0), c_l_512b_int);
                    // ap_uint<8>  c_l_5_int =  bit_select_128b(d_l_int.range(DISP_DDR_NUM_LOG2-1, 0), c_l_512b_int);
                    // ap_uint<8>  c_l_5_int =  bit_select_112b(d_l_int.range(DISP_DDR_NUM_LOG2-1, 0), c_l_512b_int);
#if DISP_DDR_BIT == 6
                    ap_uint<DISP_DDR_BIT>  c_l_5_int =  bit_select_96b(d_l_int.range(DISP_DDR_NUM_LOG2-1, 0), c_l_int);
#elif DISP_DDR_BIT == 8
                    ap_uint<DISP_DDR_BIT>  c_l_5_int =  bit_select_128b(d_l_int.range(DISP_DDR_NUM_LOG2-1, 0), c_l_int);
#elif DISP_DDR_BIT == 16
                    ap_uint<DISP_DDR_BIT>  c_l_5_int =  bit_select_256b(d_l_int.range(DISP_DDR_NUM_LOG2-1, 0), c_l_int);
#endif
                    cost_unary_prop = c_l_5_int;
                }             
                /* update local buffer */
                if (cost_unary_prop < cost_unary_orig && ((x < expn_region_w) && (y < expn_region_h))) {
                    local_cost[y][x] = cost_unary_prop;
                    local_label[y][x][0] = plane_a[p]; local_label[y][x][1] = plane_b[p]; local_label[y][x][2] = plane_c[p];
                    local_disp[y][x] = d_prop;
                }
            }
        // }
    }
}

void layer0CostLabelUpdate(int iter, ap_uint<3> unary_cost_y0, ap_uint<3> unary_cost_y1, ap_uint<3> unary_cost_y2, ap_uint<11> expn_region_x, ap_uint<11> expn_region_y, 
    ap_uint<5> expn_region_w, ap_uint<5> expn_region_h, ap_uint<11> unit_region_y,
    label_range plane_a[LAYER0_PROP_NUM], label_range plane_b[LAYER0_PROP_NUM], label_range plane_c[LAYER0_PROP_NUM], 
    label_range plane_a1[LAYER0_PROP_NUM], label_range plane_b1[LAYER0_PROP_NUM], label_range plane_c1[LAYER0_PROP_NUM], 
    unary_cost_t local_unary_cost[4][LAYER0_SIZE][LAYER0_SIZE * 3][DISP_DDR_MAX], 
    ap_uint<DISP_DDR_BIT> local_cost[LAYER0_SIZE * 3][LAYER0_SIZE * 3], label_range local_label[LAYER0_SIZE * 3][LAYER0_SIZE * 3][3], disp_range local_disp[LAYER0_SIZE * 3][LAYER0_SIZE * 3]) {
#pragma HLS array_partition variable=plane_a type=complete dim=1
#pragma HLS array_partition variable=plane_b type=complete dim=1
#pragma HLS array_partition variable=plane_c type=complete dim=1
#pragma HLS array_partition variable=local_unary_cost type=complete dim=1
#pragma HLS array_partition variable=local_unary_cost type=complete dim=3
//DO_PRAGMA(HLS array_reshape variable=local_unary_cost type=cyclic factor=DISP_DDR_NUM dim=4) 
// #pragma HLS INTERFACE mode=ap_memory port=local_cost storage_type=ram_2p
#pragma HLS array_partition variable=local_cost type=complete dim=2
#pragma HLS array_partition variable=local_label type=complete dim=2
#pragma HLS array_partition variable=local_label type=complete dim=3
#pragma HLS array_partition variable=local_disp type=complete dim=2
#pragma HLS inline off
    assert((unary_cost_y0 != unary_cost_y1) && (unary_cost_y0 != unary_cost_y2) && (unary_cost_y1 != unary_cost_y2));
    assert(unary_cost_y0 >= 0 && unary_cost_y0 < 4 && unary_cost_y1 >= 0 && unary_cost_y1 < 4 && unary_cost_y2 >= 0 && unary_cost_y2 < 4);
    // 
    bool is_top_region = (expn_region_y == unit_region_y);
             
    for (int py = 0; py < LAYER0_PROP_NUM * LAYER0_SIZE * 3 * 2; py++) {
    // for (int p = 0; p < 3; p++) {
    //     // LOOP_CALC_HOR:
    //     for (int x = 0; x < LAYER0_SIZE * 3; x++) {
#pragma HLS loop_flatten
#pragma HLS pipeline II=1
            int index = (py / (3 * LAYER0_SIZE)) / LAYER0_PROP_NUM;
            ap_uint<4> p = (py / (3 * LAYER0_SIZE)) % LAYER0_PROP_NUM; // (4, 0)
            plane_a[p] = (index == 0) ? plane_a[p] : plane_a1[p];
            plane_b[p] = (index == 0) ? plane_b[p] : plane_b1[p];
            plane_c[p] = (index == 0) ? plane_c[p] : plane_c1[p];
            int y = py % (3 * LAYER0_SIZE); // (5, 0)
            ap_uint<4> y_ofst = y % LAYER0_SIZE; // (4, 0)
            ap_uint<3> y_base; // (3, 0)
            if (is_top_region) {
                if (y < LAYER0_SIZE)
                    y_base = unary_cost_y1;
                else if (y < LAYER0_SIZE * 2)
                    y_base = unary_cost_y2;
            }
            else {
                if (y < LAYER0_SIZE)
                    y_base = unary_cost_y0;
                else if (y < LAYER0_SIZE * 2)
                    y_base = unary_cost_y1;
                else
                    y_base = unary_cost_y2;
            }
            ap_int<17> d_prop_base = plane_b[p] * (expn_region_y + y) + plane_c[p];
            for (int x = 0; x < LAYER0_SIZE * 3; x++) {
#pragma HLS unroll
                ap_int<17> d_prop = d_prop_base + plane_a[p] * (expn_region_x + x);
                ap_uint<DISP_DDR_BIT> cost_unary_prop = 0;
                ap_uint<DISP_DDR_BIT> cost_unary_orig = local_cost[y][x];
                /* unary cost */
                // if (d_prop < DISP_MIN) cost_unary_prop = local_unary_cost[y_base][y_ofst][x][0].range(6-1,0) << 1;
                // else if (d_prop.range(16,8) >= DISP_MAX - 1) cost_unary_prop = local_unary_cost[y_base][y_ofst][x][DISP_DDR_MAX-1].range(107,101);
                if (d_prop< DISP_MIN) cost_unary_prop = local_unary_cost[y_base][y_ofst][x][0].range(DISP_DDR_BIT-1,0);
                else if (d_prop.range(16,8) >= DISP_MAX - 1) cost_unary_prop = local_unary_cost[y_base][y_ofst][x][DISP_DDR_MAX-1].range(DISP_DDR_BIT*DISP_DDR_NUM-1,DISP_DDR_BIT*DISP_DDR_NUM-DISP_DDR_BIT);
                else {
                    // int d_l = d_prop.to_int();
                    ap_uint<8> d_l_int = d_prop.range(15,8);
                    if (d_prop[7]==1)
                        d_l_int = d_l_int + 1;
                    unary_cost_t c_l_int = local_unary_cost[y_base][y_ofst][x][d_l_int.range(7, DISP_DDR_NUM_LOG2)];
                    // ap_uint<8>  c_l_5_int =  bit_select_108b(d_l_int.range(DISP_DDR_NUM_LOG2-1, 0), c_l_512b_int);
                    // ap_uint<8>  c_l_5_int =  bit_select_128b(d_l_int.range(DISP_DDR_NUM_LOG2-1, 0), c_l_512b_int);
                    // ap_uint<8>  c_l_5_int =  bit_select_112b(d_l_int.range(DISP_DDR_NUM_LOG2-1, 0), c_l_512b_int);
#if DISP_DDR_BIT == 6
                    ap_uint<DISP_DDR_BIT>  c_l_5_int =  bit_select_96b(d_l_int.range(DISP_DDR_NUM_LOG2-1, 0), c_l_int);
#elif DISP_DDR_BIT == 8
                    ap_uint<DISP_DDR_BIT>  c_l_5_int =  bit_select_128b(d_l_int.range(DISP_DDR_NUM_LOG2-1, 0), c_l_int);
#elif DISP_DDR_BIT == 16
                    ap_uint<DISP_DDR_BIT>  c_l_5_int =  bit_select_256b(d_l_int.range(DISP_DDR_NUM_LOG2-1, 0), c_l_int);
#endif
                    cost_unary_prop = c_l_5_int;
                }             
                /* update local buffer */
                if (cost_unary_prop < cost_unary_orig && ((x < expn_region_w) && (y < expn_region_h))) {
//                    if((iter % LAYER0_VER_REGION_NUM == 1) || (iter % LAYER0_VER_REGION_NUM == 0)){

//                    }
//                    else{
                    local_cost[y][x] = cost_unary_prop;
                    local_label[y][x][0] = plane_a[p]; local_label[y][x][1] = plane_b[p]; local_label[y][x][2] = plane_c[p];
                    local_disp[y][x] = d_prop;
//                    }
                }
            }
        // }
    }
}

void layer1CostLabelUpdate(ap_uint<3> unary_cost_y0, ap_uint<3> unary_cost_y1, ap_uint<3> unary_cost_y2, 
    ap_uint<11> expn_region_x, ap_uint<11> expn_region_y, ap_uint<6> expn_region_w, ap_uint<6> expn_region_h, ap_uint<11> unit_region_y,
    label_range plane_a[LAYER1_PROP_NUM], label_range plane_b[LAYER1_PROP_NUM], label_range plane_c[LAYER1_PROP_NUM], 
    unary_cost_t local_unary_cost[LAYER1_SIZE][4][LAYER1_SIZE * 3][DISP_DDR_MAX], uchar local_im0_gray[LAYER1_SIZE * 3][LAYER1_SIZE * 3], 
    ap_uint<DISP_DDR_BIT> local_cost[LAYER1_SIZE * 3][LAYER1_SIZE * 3], label_range local_label[LAYER1_SIZE * 3][LAYER1_SIZE * 3][3], disp_range local_disp[LAYER1_SIZE * 3][LAYER1_SIZE * 3]) {
#pragma HLS inline off
#pragma HLS array_partition variable=local_unary_cost type=complete dim=2
#pragma HLS array_partition variable=local_unary_cost type=complete dim=3
//DO_PRAGMA(HLS array_reshape variable=local_unary_cost type=cyclic factor=DISP_DDR_NUM dim=4) 
#pragma HLS array_partition variable=local_im0_gray type=complete dim=2
#pragma HLS array_partition variable=local_cost type=complete dim=2
#pragma HLS array_partition variable=local_label type=complete dim=2
#pragma HLS array_partition variable=local_label type=complete dim=3
#pragma HLS array_partition variable=local_disp type=complete dim=2
#pragma HLS array_partition variable=plane_a type=complete dim=1
#pragma HLS array_partition variable=plane_b type=complete dim=1
#pragma HLS array_partition variable=plane_c type=complete dim=1
    assert((unary_cost_y0 != unary_cost_y1) && (unary_cost_y0 != unary_cost_y2) && (unary_cost_y1 != unary_cost_y2));
    assert(unary_cost_y0 >= 0 && unary_cost_y0 < 4 && unary_cost_y1 >= 0 && unary_cost_y1 < 4 && unary_cost_y2 >= 0 && unary_cost_y2 < 4);
    assert(expn_region_h <= (LAYER1_SIZE * 3));
    assert(expn_region_w <= (LAYER1_SIZE * 3));
    //
    bool is_top_region = (expn_region_y == unit_region_y);
    //
    ap_uint<8> gray_reg[3][LAYER1_SIZE * 3];
#pragma HLS array_partition variable=gray_reg type=complete dim=0
	disp_range disp_reg[3][LAYER1_SIZE * 3];
#pragma HLS array_partition variable=disp_reg type=complete dim=0	
    /* load to last colume of data registers*/
    LOOP_2ND_CACHE0:
    for (int i = 0; i < LAYER1_SIZE * 3; i++) {
#pragma HLS unroll
        gray_reg[2][i] = local_im0_gray[0][i];
        disp_reg[2][i] = local_disp[0][i];
    }
    // 
//     for (int p = 0; p < LAYER1_PROP_NUM; p++) {
//         for (int x = 0; x < LAYER1_SIZE * 3; x++) {
// #pragma HLS loop_flatten
// #pragma HLS pipeline II=1
    for (int py = 0; py < LAYER1_PROP_NUM * LAYER1_SIZE * 3; py++) {
#pragma HLS pipeline II=2
            int p = py / (3 * LAYER1_SIZE);
            int y = py % (3 * LAYER1_SIZE);
            ap_uint<5> y_ofst;
            if (y >= LAYER1_SIZE * 2)
                y_ofst = y - LAYER1_SIZE * 2;
            else if (y >= LAYER1_SIZE)
                y_ofst = y - LAYER1_SIZE;
            else
                y_ofst = y;
            ap_uint<3> y_base = 0;
            if (is_top_region) {
                if (y < LAYER1_SIZE)
                    y_base = unary_cost_y1;
                else if (y < LAYER1_SIZE * 2)
                    y_base = unary_cost_y2;
            }
            else {
                if (y < LAYER1_SIZE)
                    y_base = unary_cost_y0;
                else if (y < LAYER1_SIZE * 2)
                    y_base = unary_cost_y1;
                else
                    y_base = unary_cost_y2;
            }
            /* update data registers for binary cost calculation */
            /* shift first 2 columns */
            LOOP_2ND_CACHE1:
            for (int j = 0; j < 2; j++) {
#pragma HLS unroll
                for (int i = 0; i < LAYER1_SIZE * 3; i++) {
#pragma HLS unroll
                    gray_reg[j][i] = gray_reg[j + 1][i];
                    disp_reg[j][i] = disp_reg[j + 1][i];
                }
            }
            /* load to last column */
            if (y < (expn_region_h - 1)) {
                LOOP_2ND_CACHE2:
                for (int i = 0; i < LAYER1_SIZE * 3; i++) {
#pragma HLS unroll
                    gray_reg[2][i] = local_im0_gray[y + 1][i];
                    disp_reg[2][i] = local_disp[y + 1][i];
                }
            }
            ap_int<17> d_prop_y = plane_b[p] * (expn_region_y + y) + plane_c[p];
            LOOP_CALC_VER:
            for (int x = 0; x < LAYER1_SIZE * 3; x++) {
#pragma HLS unroll
// #pragma HLS dependence variable=local_cost type=inter false
// #pragma HLS dependence variable=local_label type=inter false
// #pragma HLS pipeline II=1
                //
                disp_range d_prop = d_prop_y + plane_a[p] * (expn_region_x + x);
                ap_uint<DISP_DDR_BIT> cost_unary_prop = 0;
                ap_uint<19> cost_binary_prop = 0;
                ap_uint<DISP_DDR_BIT> cost_unary_orig = local_cost[y][x];
                ap_uint<19> cost_binary_orig = 0;
                disp_range d_orig = disp_reg[1][x];
                ap_uint<8> p_gray = gray_reg[1][x];
                /* unary cost */
                // if (d_prop < DISP_MIN) cost_unary_prop = local_unary_cost[y_base][y_ofst][x][0].range(6-1,0) << 1;
                // else if (d_prop.range(16,8) >= DISP_MAX - 1) cost_unary_prop = local_unary_cost[y_base][y_ofst][x][DISP_DDR_MAX-1].range(107,101);
                if (d_prop< DISP_MIN) cost_unary_prop = local_unary_cost[y_ofst][y_base][x][0].range(DISP_DDR_BIT-1,0);
                else if (d_prop.range(16,8) >= DISP_MAX - 1) cost_unary_prop = local_unary_cost[y_ofst][y_base][x][DISP_DDR_MAX-1].range(DISP_DDR_BIT*DISP_DDR_NUM-1,DISP_DDR_BIT*DISP_DDR_NUM-DISP_DDR_BIT);
                else {
                    ap_uint<8> d_l_int = d_prop.range(15,8);
                    if (d_prop[7]==1)
                        d_l_int = d_l_int + 1;
                    unary_cost_t c_l_int = local_unary_cost[y_ofst][y_base][x][d_l_int.range(7, DISP_DDR_NUM_LOG2)];
                    // ap_uint<8>  c_l_5_int =  bit_select_108b(d_l_int.range(DISP_DDR_NUM_LOG2-1, 0), c_l_512b_int);
                    // ap_uint<8>  c_l_5_int =  bit_select_128b(d_l_int.range(DISP_DDR_NUM_LOG2-1, 0), c_l_512b_int);
                    // ap_uint<8>  c_l_5_int =  bit_select_112b(d_l_int.range(DISP_DDR_NUM_LOG2-1, 0), c_l_512b_int);
#if DISP_DDR_BIT == 6
                    ap_uint<DISP_DDR_BIT>  c_l_5_int =  bit_select_96b(d_l_int.range(DISP_DDR_NUM_LOG2-1, 0), c_l_int);
#elif DISP_DDR_BIT == 8
                    ap_uint<DISP_DDR_BIT>  c_l_5_int =  bit_select_128b(d_l_int.range(DISP_DDR_NUM_LOG2-1, 0), c_l_int);
#elif DISP_DDR_BIT == 16
                    ap_uint<DISP_DDR_BIT>  c_l_5_int =  bit_select_256b(d_l_int.range(DISP_DDR_NUM_LOG2-1, 0), c_l_int);
#endif
                    cost_unary_prop = c_l_5_int;
                }
                /* binary cost */
                ap_uint<15> cost_binary_prop_array[9];
#pragma HLS array_partition variable=cost_binary_prop_array type=complete dim=1
                ap_uint<15> cost_binary_orig_array[9];
#pragma HLS array_partition variable=cost_binary_orig_array type=complete dim=1
                LOOP_BINARY_CALC:
                for (int j = 0; j < 3; j++) {
#pragma HLS unroll
                    for (int i = 0; i < 3; i++) {
#pragma HLS unroll
                        if (((i == 1) && (j == 1)) || ((x + j - 1) < 0) || ((x + j - 1) >= expn_region_w) || ((y + i - 1) < 0) || ((y + i - 1) >= expn_region_h)){
                            cost_binary_prop_array[i * 3 + j] = 0;
                            cost_binary_orig_array[i * 3 + j] = 0;
                        }
                        else {
                            assert((x + j - 1) < LAYER1_SIZE * 3);
                            disp_range d_nb = disp_reg[i][x + j - 1];
                            ap_uint<8> p_gray_nb = gray_reg[i][x + j - 1];
                            bool flag = ((p_gray_nb - p_gray) < 16) && ((p_gray_nb - p_gray) > -16);
                            disp_range d_prop_diff = (d_prop > d_nb) ? (d_prop - d_nb) : (d_nb - d_prop);
                            disp_range d_orig_diff = (d_orig > d_nb) ? (d_orig - d_nb) : (d_nb - d_orig);
                            cost_binary_prop_array[i * 3 + j] = (flag == true) ? d_prop_diff.range(16, 0) : d_prop_diff.range(16, 2);
                            cost_binary_orig_array[i * 3 + j] = (flag == true) ? d_orig_diff.range(16, 0) : d_orig_diff.range(16, 2);
                        }
                    }
                }
                for (int i = 0; i < 3; i++) {
#pragma HLS unroll
                    for (int j = 0; j < 3; j++) {
#pragma HLS unroll
                        cost_binary_prop += cost_binary_prop_array[i * 3 + j];
                        cost_binary_orig += cost_binary_orig_array[i * 3 + j];
                    }
                }
                if ((cost_unary_prop + cost_binary_prop) < (cost_unary_orig + cost_binary_orig)) {
                // if ((cost_unary_prop) < (cost_unary_orig)) {
                    local_cost[y][x] = cost_unary_prop;
                    local_label[y][x][0] = plane_a[p]; local_label[y][x][1] = plane_b[p]; local_label[y][x][2] = plane_c[p];
                    local_disp[y][x] = d_prop;
                }
            }  
        }
    // }
}

/*
void layer0Calc(int iter, int ver_region_num, int hor_region_num, int unary_cost_y0, int unary_cost_y1, int unary_cost_y2,
    ap_uint<32>* Q, ap_uint<32>& I, ap_uint<32>& C,
    float local_unary_cost[4][LAYER0_SIZE][LAYER0_SIZE * 3][DISP_MAX], uchar local_im0_gray[4][LAYER0_SIZE][LAYER0_SIZE * 3], 
    float uram_cost[HEIGHT][LAYER0_SIZE * 3], float uram_label[HEIGHT][LAYER0_SIZE * 3][3], float uram_disp[HEIGHT][LAYER0_SIZE * 3]) {
    //
    float local_cost[LAYER0_SIZE * 3][LAYER0_SIZE * 3];
    float local_label[LAYER0_SIZE * 3][LAYER0_SIZE * 3][3];
    float local_disp[LAYER0_SIZE * 3][LAYER0_SIZE * 3];
    // 
    float plane_a, plane_b, plane_c;
    //
    int ver_region_i = iter % ver_region_num;
    int hor_region_i = iter / ver_region_num;
    int unit_region_x, unit_region_y, unit_region_w, unit_region_h;
    int expn_region_x, expn_region_y, expn_region_w, expn_region_h;
    genRegionInfo(WIDTH, HEIGHT, hor_region_i, ver_region_i, LAYER0_SIZE, unit_region_x, unit_region_y, unit_region_w, unit_region_h, 
        expn_region_x, expn_region_y, expn_region_w, expn_region_h);
    // address mapping table for uram
    int uram_x_addr_id = hor_region_i % 3;
    // uram -> local ram
    LOOP_LOAD:
    for (int y = 0; y < LAYER0_SIZE * 3; y++) {
#pragma HLS pipeline II=1
        if ((expn_region_y + y) >= HEIGHT)   
            continue;
        if (hor_region_i == 0) {
            L0_GLOBAL_LOCAL_VSCAN_ADDR1
        }
        else {
            if (uram_x_addr_id == 0) {
                L0_GLOBAL_LOCAL_VSCAN_ADDR0
            }
            else if (uram_x_addr_id == 2) {
                L0_GLOBAL_LOCAL_VSCAN_ADDR2
            }
            else {
                L0_GLOBAL_LOCAL_VSCAN_ADDR1
            }
        }
    }
    // process local ram
#if LAYER0_ENABLE
    LOOP_PROPOSAL:
    for (int inner_iter = 0; inner_iter < 8; inner_iter++) {
#pragma HLS pipeline off
        if (inner_iter == 0)
            layerExpnProposal<LAYER0_SIZE>(ver_region_i, Q, I, C, expn_region_x, expn_region_y, unit_region_x, unit_region_y, unit_region_w, unit_region_h, 
                local_label, plane_a, plane_b, plane_c);
        else
            layerRandProposal<LAYER0_SIZE>(ver_region_i, Q, I, C, 0, inner_iter -1, expn_region_x, expn_region_y, unit_region_x, unit_region_y, unit_region_w, unit_region_h, 
                local_label, plane_a, plane_b, plane_c);
        costLabelUpdate<LAYER0_SIZE>(unary_cost_y0, unary_cost_y1, unary_cost_y2, expn_region_x, expn_region_y, expn_region_w, expn_region_h, unit_region_y, 
            plane_a, plane_b, plane_c, local_unary_cost, local_im0_gray, local_cost, local_label, local_disp, false, false);
    }
#endif
    // local ram -> uram
    LOOP_STORE:
    for (int y = 0; y < LAYER0_SIZE * 3; y++) {
#pragma HLS pipeline II=1
        if ((expn_region_y + y) >= HEIGHT)
            continue;
        if (hor_region_i == 0) {
            L0_LOCAL_GLOBAL_VSCAN_ADDR1
        }
        else {
            if (uram_x_addr_id == 0) {
                L0_LOCAL_GLOBAL_VSCAN_ADDR0
            }
            else if (uram_x_addr_id == 2) {
                L0_LOCAL_GLOBAL_VSCAN_ADDR2
            }
            else {
                L0_LOCAL_GLOBAL_VSCAN_ADDR1
            }
        }
    }
}
*/

void layer0UramLoad(int iter, int ver_region_num, int hor_region_num, 
    ap_uint<DISP_DDR_BIT> uram_cost[HEIGHT][LAYER0_SIZE * 3], label_range uram_label[HEIGHT][LAYER0_SIZE * 3][3], disp_range uram_disp[HEIGHT][LAYER0_SIZE * 3],
    ap_uint<DISP_DDR_BIT> local_cost[LAYER0_SIZE * 3][LAYER0_SIZE * 3], label_range local_label[LAYER0_SIZE * 3][LAYER0_SIZE * 3][3], disp_range local_disp[LAYER0_SIZE * 3][LAYER0_SIZE * 3],
    /*ap_uint<DISP_DDR_BIT> local_trans_cost[LAYER0_SIZE][LAYER0_SIZE], label_range local_trans_label[LAYER0_SIZE][LAYER0_SIZE][3], */disp_range local_trans_disp[LAYER0_SIZE][LAYER0_SIZE]) {
#pragma HLS array_partition variable=uram_cost type=complete dim=2      
#pragma HLS array_partition variable=uram_label type=complete dim=2
#pragma HLS array_partition variable=uram_label type=complete dim=3
#pragma HLS array_partition variable=uram_disp type=complete dim=2
#pragma HLS array_partition variable=local_cost type=complete dim=2
#pragma HLS array_partition variable=local_label type=complete dim=2
#pragma HLS array_partition variable=local_label type=complete dim=3
#pragma HLS array_partition variable=local_disp type=complete dim=2
//#pragma HLS array_partition variable=local_trans_cost type=complete dim=2
//#pragma HLS array_partition variable=local_trans_label type=complete dim=2
//#pragma HLS array_partition variable=local_trans_label type=complete dim=3
#pragma HLS array_partition variable=local_trans_disp type=complete dim=2
    /*  */
    int ver_region_i = iter % ver_region_num;
    int hor_region_i = iter / ver_region_num;
    int unit_region_x, unit_region_y, unit_region_w, unit_region_h;
    int expn_region_x, expn_region_y, expn_region_w, expn_region_h;
    genRegionInfo(WIDTH, HEIGHT, hor_region_i, ver_region_i, LAYER0_SIZE, unit_region_x, unit_region_y, unit_region_w, unit_region_h, 
        expn_region_x, expn_region_y, expn_region_w, expn_region_h);
    /* address mapping table for uram */
    int uram_x_addr_id = hor_region_i % 3;
    // uram -> local ram
    LOOP_LOAD:
    for (int y = 0; y < LAYER0_SIZE * 3; y++) {
#pragma HLS pipeline II=1
        if ((expn_region_y + y) >= HEIGHT)   
            continue;
        if (hor_region_i == 0) {
            L0_GLOBAL_LOCAL_VSCAN_ADDR1
        }
        else {
            if (uram_x_addr_id == 0) {
                L0_GLOBAL_LOCAL_VSCAN_ADDR0
            }
            else if (uram_x_addr_id == 2) {
                L0_GLOBAL_LOCAL_VSCAN_ADDR2
            }
            else {
                L0_GLOBAL_LOCAL_VSCAN_ADDR1
            }
        }
    }

    int tran_ver_region_i = (iter - (ver_region_num + 2)) % ver_region_num;
    int tran_hor_region_i = (iter - (ver_region_num + 2)) / ver_region_num;
    int tran_unit_region_x, tran_unit_region_y, tran_unit_region_w, tran_unit_region_h;
    int tran_expn_region_x, tran_expn_region_y, tran_expn_region_w, tran_expn_region_h;
    genRegionInfo(WIDTH, HEIGHT, tran_hor_region_i, tran_ver_region_i, LAYER0_SIZE, tran_unit_region_x, tran_unit_region_y, tran_unit_region_w, tran_unit_region_h, 
        tran_expn_region_x, tran_expn_region_y, tran_expn_region_w, tran_expn_region_h);
    if (tran_ver_region_i < 0 || tran_hor_region_i < 0)
        return;
    for (int y = 0; y < LAYER0_SIZE; y++) {
#pragma HLS pipeline II=1
        // for (int y = 0; y < LAYER0_SIZE; y++) {
// #pragma HLS unroll
            // if ((x >= tran_unit_region_w) || (y >= tran_unit_region_h)) continue;
            if ((tran_unit_region_y + y) >= HEIGHT)   
                continue;
            if ((tran_hor_region_i % 3) == 0) {
                L0_GLOBAL_LOCAL_TRANS_VSCAN_ADDR0
            }
            else if ((tran_hor_region_i % 3) == 1) {
                L0_GLOBAL_LOCAL_TRANS_VSCAN_ADDR1
            }
            else {
                L0_GLOBAL_LOCAL_TRANS_VSCAN_ADDR2
            }
            // local_trans_cost[y][x] = uram_cost[(tran_ver_region_i % 3) * LAYER0_SIZE + y][tran_unit_region_x + x];
            // local_trans_label[y][x][0] = uram_label[(tran_ver_region_i % 3) * LAYER0_SIZE + y][tran_unit_region_x + x][0];
            // local_trans_label[y][x][1] = uram_label[(tran_ver_region_i % 3) * LAYER0_SIZE + y][tran_unit_region_x + x][1];
            // local_trans_label[y][x][2] = uram_label[(tran_ver_region_i % 3) * LAYER0_SIZE + y][tran_unit_region_x + x][2];
            // local_trans_disp[y][x] = uram_disp[(tran_ver_region_i % 3) * LAYER0_SIZE + y][tran_unit_region_x + x];
        // }
    }
}

struct ABC {
    ap_fixed<32, 17, AP_RND, AP_SAT> a, b, c;
};

// 计算3x3矩阵的行列式
fixed_min determinant3x3(ap_uint11 a11, ap_uint10 a12, ap_uint8 a13,
                        ap_uint11 a21, ap_uint10 a22, ap_uint8 a23,
                        ap_uint11 a31, ap_uint10 a32, ap_uint8 a33) {
    return a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31) + a13 * (a21 * a32 - a22 * a31);
}

// 计算a, b, c的函数
ABC calculateABC(ap_uint11 x0, ap_uint10 y0, ap_uint8 d0,
                 ap_uint11 x1, ap_uint10 y1, ap_uint8 d1,
                 ap_uint11 x2, ap_uint10 y2, ap_uint8 d2
                 ) {
//#pragma HLS inline off
    // 计算系数矩阵A的行列式D
    fixed_min D = determinant3x3(x0, y0, 1, x1, y1, 1, x2, y2, 1);

    
    if (D == 0) {
        D = 1;
    }

    // 计算Da, Db, Dc，并行处理
    fixed_min Da, Db, Dc;
    Da = determinant3x3(d0, y0, 1, d1, y1, 1, d2, y2, 1);
    Db = determinant3x3(x0, d0, 1, x1, d1, 1, x2, d2, 1);
    Dc = determinant3x3(x0, y0, d0, x1, y1, d1, x2, y2, d2);
    ap_fixed<32, 17, AP_RND, AP_SAT> a, b, c;          
    a = Da / D;
    b = Db / D;
    c = Dc / D;    
    ABC outcome;
    outcome.a=a;
    outcome.b=b;
    outcome.c=c;
    return outcome; 
}

void choose_further(ap_uint<8> best_byte[DISP_DDR_MAX], ap_uint<8> best_index[DISP_DDR_MAX], ap_uint<8>& index){
ap_uint<8> min_byte = 255;  // 假设字节的最大值为255
    ap_uint<8> min_index = 0;

    for(int j=0; j<DISP_DDR_MAX; j++){
        #pragma HLS unroll
        if(best_byte[j] < min_byte){
            min_byte = best_byte[j];
            min_index = best_index[j];
        }
    }
    index = min_index;
/*   
//#pragma HLS inline off
//ap_uint<8> cost_one[6];
ap_uint<8> cost_min0[6][2];
ap_uint<8> cost_min1[3][2];
ap_uint<8> byte;
//ap_uint<8> cost_min3[2];
//#pragma HLS array_partition variable=cost_one type=complete dim=0
#pragma HLS array_partition variable=cost_min0 type=complete dim=0
#pragma HLS array_partition variable=cost_min1 type=complete dim=0
//#pragma HLS array_partition variable=cost_min2 type=complete dim=0
//#pragma HLS array_partition variable=cost_min3 type=complete dim=0

for(int j=0; j<6; j++){
    #pragma HLS unroll
    cost_min0[j][0] = (best_byte[j]<best_byte[j+6])? best_byte[j]:best_byte[j+6];
    cost_min0[j][1] = (best_byte[j]<best_byte[j+6])? best_index[j]:best_index[j+6];
}

for(int j=0; j<3; j++){
    #pragma HLS unroll
    cost_min1[j][0] = (cost_min0[j][0]<cost_min0[j+3][0])? cost_min0[j][0]:cost_min0[j+3][0];
    cost_min1[j][1] = (cost_min0[j][0]<cost_min0[j+3][0])? cost_min0[j][1]:cost_min0[j+3][1];
}

    byte = (cost_min1[0][0]<cost_min1[1][0])? cost_min1[0][0]:cost_min1[1][0];
    index = (cost_min1[0][0]<cost_min1[1][0])? cost_min1[0][1]:cost_min1[1][1];
    index = (byte<cost_min1[2][0])? index:cost_min1[2][1];
    */ 
}

void choose_min_cost(int base, ap_uint<128> cost_all, ap_uint<8>& byte, ap_uint<8>& index){
//#pragma HLS inline off
ap_uint<8> cost_one[16];
ap_uint<8> cost_min0[8][2];
ap_uint<8> cost_min1[4][2];
ap_uint<8> cost_min2[2][2];
//ap_uint<8> cost_min3[2];
#pragma HLS array_partition variable=cost_one type=complete dim=0
#pragma HLS array_partition variable=cost_min0 type=complete dim=0
#pragma HLS array_partition variable=cost_min1 type=complete dim=0
#pragma HLS array_partition variable=cost_min2 type=complete dim=0
//#pragma HLS array_partition variable=cost_min3 type=complete dim=0

for(int i=0; i<16; i++){
    #pragma HLS unroll
    cost_one[i] = cost_all(8*i, 8*i + 7);
}
for(int j=0; j<8; j++){
    #pragma HLS unroll

    cost_min0[j][0] = (cost_one[j]<cost_one[j+8])? cost_one[j]:cost_one[j+8];
    cost_min0[j][1] = (cost_one[j]<cost_one[j+8])? j:j+8;
}
for(int j=0; j<4; j++){
    #pragma HLS unroll

    cost_min1[j][0] = (cost_min0[j][0]<cost_min0[j+4][0])? cost_min0[j][0]:cost_min0[j+4][0];
    cost_min1[j][1] = (cost_min0[j][0]<cost_min0[j+4][0])? cost_min0[j][1]:cost_min0[j+4][1];
}
for(int j=0; j<2; j++){
    #pragma HLS unroll

    cost_min2[j][0] = (cost_min1[j][0]<cost_min1[j+2][0])? cost_min1[j][0]:cost_min1[j+2][0];
    cost_min2[j][1] = (cost_min1[j][0]<cost_min1[j+2][0])? cost_min1[j][1]:cost_min1[j+2][1];
}
byte = (cost_min2[0][0]<cost_min2[1][0])? cost_min2[0][0]:cost_min2[1][0];
index = (cost_min2[0][0]<cost_min2[1][0])? base + cost_min2[0][1]: base + cost_min2[1][1];
}


void new_abc_hls(ap_uint<4> p_xx[LAYER0_PROP_NUM], ap_uint<4> p_yy[LAYER0_PROP_NUM], int unary_cost_y1, int unary_cost_y2,
ap_uint<12> expn_region_x, ap_uint<12> expn_region_y, ap_uint<12> unit_region_x, ap_uint<12> unit_region_y, ap_uint<7> unit_region_w, ap_uint<7> unit_region_h, 
unary_cost_t local_unary_cost[4][LAYER0_SIZE][LAYER0_SIZE * 3][DISP_DDR_MAX], label_range p_a[LAYER0_PROP_NUM], label_range p_b[LAYER0_PROP_NUM], label_range p_c[LAYER0_PROP_NUM]){
#pragma HLS inline off
//#pragma HLS array_partition variable=local_unary_cost type=complete dim=1
//#pragma HLS array_partition variable=local_unary_cost type=complete dim=2
#pragma HLS array_partition variable=p_a type=complete dim=1
//#pragma HLS array_partition variable=p_a type=complete dim=2
#pragma HLS array_partition variable=p_b type=complete dim=1
//#pragma HLS array_partition variable=p_b type=complete dim=2
#pragma HLS array_partition variable=p_c type=complete dim=1
//#pragma HLS array_partition variable=p_c type=complete dim=2
ap_uint<8> best_byte0[LAYER0_PROP_NUM][DISP_DDR_MAX]; ap_uint<8> best_index0[LAYER0_PROP_NUM][DISP_DDR_MAX];
ap_uint<8> best_byte1[LAYER0_PROP_NUM][DISP_DDR_MAX]; ap_uint<8> best_index1[LAYER0_PROP_NUM][DISP_DDR_MAX];
ap_uint<8> best_byte2[LAYER0_PROP_NUM][DISP_DDR_MAX]; ap_uint<8> best_index2[LAYER0_PROP_NUM][DISP_DDR_MAX];
#pragma HLS array_partition variable=best_byte0 type=complete dim=0
#pragma HLS array_partition variable=best_byte1 type=complete dim=0
#pragma HLS array_partition variable=best_byte2 type=complete dim=0
#pragma HLS array_partition variable=best_index0 type=complete dim=0
#pragma HLS array_partition variable=best_index1 type=complete dim=0
#pragma HLS array_partition variable=best_index2 type=complete dim=0
unary_cost_t bank0[DISP_DDR_MAX*DISP_DDR_NUM];
unary_cost_t bank1[DISP_DDR_MAX*DISP_DDR_NUM];
unary_cost_t bank2[DISP_DDR_MAX*DISP_DDR_NUM];
#pragma HLS array_partition variable=bank0 type=complete dim=0
#pragma HLS array_partition variable=bank1 type=complete dim=0
#pragma HLS array_partition variable=bank2 type=complete dim=0
ap_uint<8> d0, d1, d2;
ap_uint<4> p_x[LAYER0_PROP_NUM][3];
ap_uint<4> p_y[LAYER0_PROP_NUM][3]; 
#pragma HLS array_partition variable=p_x type=complete dim=0
#pragma HLS array_partition variable=p_y type=complete dim=0
ap_uint<4> x_base = (expn_region_x == unit_region_x)? 0 : 9;
ap_uint<4> y_base = unary_cost_y1;
ap_uint<4> para1 = 2;
ap_uint<4> para2 = 6;
for (int i = 0; i < LAYER0_PROP_NUM; i++) {
#pragma HLS unroll
 p_x[i][0] = p_xx[i];//随机选三个像素找最小cost
 p_y[i][0] = p_yy[i];
 ap_uint<4> para11 = p_x[i][0]-1;
 ap_uint<4> para22 = p_x[i][0]+1;
 p_x[i][1] = ((p_x[i][0]==0)? para1: para11);
 p_y[i][1] = p_x[i][0];
 p_x[i][2] = ((p_x[i][0]==8)? para2: para22);
 p_y[i][2] = p_x[i][1];
}
for (int k = 0; k < LAYER0_PROP_NUM * DISP_DDR_MAX; k++) {
//for (int i = 0; i < LAYER0_PROP_NUM; i++) {
 //for (int j = 0; j < DISP_DDR_MAX; j++) { 
 #pragma HLS pipeline II=1
    int i = k / DISP_DDR_MAX;
    int j = k % DISP_DDR_MAX;
    int base = j << 4;
    bank0[j] = local_unary_cost[y_base][p_y[i][0]][x_base + p_x[i][0]][j];
    choose_min_cost(base, bank0[j], best_byte0[i][j], best_index0[i][j]);
    bank1[j] = local_unary_cost[y_base][p_y[i][1]][x_base + p_x[i][1]][j];
    choose_min_cost(base, bank1[j], best_byte1[i][j], best_index1[i][j]);
    bank2[j] = local_unary_cost[y_base][p_y[i][2]][x_base + p_x[i][2]][j];
    choose_min_cost(base, bank2[j], best_byte2[i][j], best_index2[i][j]);
 //}
}

for (int i = 0; i < LAYER0_PROP_NUM; i++) {
#pragma HLS unroll
choose_further(best_byte0[i], best_index0[i], d0);
choose_further(best_byte1[i], best_index1[i], d1);
choose_further(best_byte2[i], best_index2[i], d2);

ABC label = calculateABC(unit_region_x+p_x[i][0], unit_region_y+p_y[i][0], d0, 
                         unit_region_x+p_x[i][1], unit_region_y+p_y[i][1], d1,
                         unit_region_x+p_x[i][2], unit_region_y+p_y[i][2], d2);
    p_a[i]=((label.a) << 8).to_ap_int();
    p_b[i]=((label.b) << 8).to_ap_int();
    p_c[i]=((label.c) << 8).to_ap_int();
  }
/*
for (int i = 0; i < LAYER0_PROP_NUM; i++) {
   p_a[index][i]=864;
    p_b[index][i]=723;
    p_c[index][i]=1024;
}
*/
}

void random_xy(int iter, int inner_iter, ap_uint<32> Q[LAYER0_PROP_NUM][4096], ap_uint<32> I[LAYER0_PROP_NUM], ap_uint<32> C[LAYER0_PROP_NUM], ap_uint<4> p_x[LAYER0_PROP_NUM], ap_uint<4> p_y[LAYER0_PROP_NUM], ap_uint<32> pp[LAYER0_PROP_NUM][3], ap_uint<4> p_xx[LAYER0_PROP_NUM], ap_uint<4> p_yy[LAYER0_PROP_NUM]){
#pragma HLS array_partition variable=p_x type=complete dim=0
#pragma HLS array_partition variable=p_y type=complete dim=0
#pragma HLS array_partition variable=pp type=complete dim=0
#pragma HLS array_partition variable=p_xx type=complete dim=0
#pragma HLS array_partition variable=p_yy type=complete dim=0 
#pragma HLS array_partition variable=Q type=complete dim=1
#pragma HLS array_partition variable=I type=complete dim=0
#pragma HLS array_partition variable=C type=complete dim=0   
    int ver_region_i = iter / LAYER0_HOR_REGION_NUM;
    int hor_region_i = iter % LAYER0_HOR_REGION_NUM;
    int unit_region_x, unit_region_y, unit_region_w, unit_region_h;
    int expn_region_x, expn_region_y, expn_region_w, expn_region_h;
    genRegionInfo(WIDTH, HEIGHT, hor_region_i, ver_region_i, LAYER0_SIZE, unit_region_x, unit_region_y, unit_region_w, unit_region_h, 
        expn_region_x, expn_region_y, expn_region_w, expn_region_h);
    for (int i = 0; i < LAYER0_PROP_NUM; i++) {
    #pragma HLS unroll
    if((unit_region_w==0)||(unit_region_h==0)){
        printf("iter = %d, inner_iter = %d, w = %d, h = %d\n", iter,inner_iter,unit_region_w, unit_region_h);
    }
    p_x[i] = randMwc(Q[i], I[i], C[i]) % unit_region_w;//随机选三个像素找最小cost
    p_y[i] = randMwc(Q[i], I[i], C[i]) % unit_region_h;
    p_xx[i] = randMwc(Q[i], I[i], C[i]) % unit_region_w;//随机选三个像素找最小cost
    p_yy[i] = randMwc(Q[i], I[i], C[i]) % unit_region_h;
    }
    for (int i = 0; i < LAYER0_PROP_NUM * 3; i++) {
    #pragma HLS unroll
    int j = i / 3;
    int k = i % 3;
    pp[j][k] = randMwc(Q[k], I[k], C[k]);//随机选三个像素找最小cost 
    }
    /*
    for (int i = 0; i < LAYER0_PROP_NUM; i++) {
    #pragma HLS unroll
    p_xx[i] = randMwc(Q[i], I[i], C[i]) % unit_region_w;//随机选三个像素找最小cost
    p_yy[i] = randMwc(Q[i], I[i], C[i]) % unit_region_h;
    }
    */
}

void layer0Calc(int iter, ap_uint<4> p_x[LAYER0_PROP_NUM], ap_uint<4> p_y[LAYER0_PROP_NUM], ap_uint<32> pp[LAYER0_PROP_NUM][3], ap_uint<4> p_xx[LAYER0_PROP_NUM], ap_uint<4> p_yy[LAYER0_PROP_NUM], ap_uint<32> Q[LAYER0_PROP_NUM][4096], ap_uint<32> I[LAYER0_PROP_NUM], ap_uint<32> C[LAYER0_PROP_NUM], 
    int unary_cost_y0, int unary_cost_y1, int unary_cost_y2,  unary_cost_t local_unary_cost[4][LAYER0_SIZE][LAYER0_SIZE * 3][DISP_DDR_MAX], 
    ap_uint<DISP_DDR_BIT> local_cost[LAYER0_SIZE * 3][LAYER0_SIZE * 3], label_range local_label[LAYER0_SIZE * 3][LAYER0_SIZE * 3][3], disp_range local_disp[LAYER0_SIZE * 3][LAYER0_SIZE * 3]) {
#pragma HLS inline off
// #pragma HLS array_partition variable=local_unary_cost type=complete dim=1
// #pragma HLS array_partition variable=local_unary_cost type=complete dim=2
// DO_PRAGMA(HLS array_reshape variable=local_unary_cost type=cyclic factor=DISP_DDR_NUM dim=4) 
// #pragma HLS INTERFACE mode=bram port=local_cost storage_impl=bram storage_type=ram_2p
#pragma HLS array_partition variable=local_cost type=complete dim=2
#pragma HLS array_partition variable=local_label type=complete dim=2
#pragma HLS array_partition variable=local_label type=complete dim=3
#pragma HLS array_partition variable=local_disp type=complete dim=2
    // float plane_a, plane_b, plane_c;
    label_range plane_a[LAYER0_PROP_NUM], plane_b[LAYER0_PROP_NUM], plane_c[LAYER0_PROP_NUM];
#pragma HLS array_partition variable=plane_a type=complete dim=1
#pragma HLS array_partition variable=plane_b type=complete dim=1
#pragma HLS array_partition variable=plane_c type=complete dim=1
    label_range plane_a1[LAYER0_PROP_NUM], plane_b1[LAYER0_PROP_NUM], plane_c1[LAYER0_PROP_NUM];
#pragma HLS array_partition variable=plane_a1 type=complete dim=1
#pragma HLS array_partition variable=plane_b1 type=complete dim=1
#pragma HLS array_partition variable=plane_c1 type=complete dim=1
    //
    assert((iter >= 0) && (iter < LAYER0_VER_REGION_NUM * LAYER0_HOR_REGION_NUM));
    int ver_region_i = iter % LAYER0_VER_REGION_NUM;
    int hor_region_i = iter / LAYER0_VER_REGION_NUM;
    int unit_region_x, unit_region_y, unit_region_w, unit_region_h;
    int expn_region_x, expn_region_y, expn_region_w, expn_region_h;
    genRegionInfo(WIDTH, HEIGHT, hor_region_i, ver_region_i, LAYER0_SIZE, unit_region_x, unit_region_y, unit_region_w, unit_region_h, 
        expn_region_x, expn_region_y, expn_region_w, expn_region_h);
    /* process local ram */
#if LAYER0_ENABLE
    LOOP_PROPOSAL:
#if LAYER0_PROP_NUM==3
    for (int inner_iter = 0; inner_iter < LAYER0_PROP_LOOP; inner_iter++) {
#pragma HLS pipeline off
        layer0ProposalParall(iter, ver_region_i, p_x, p_y, pp, inner_iter, expn_region_x, expn_region_y, unit_region_x, unit_region_y, unit_region_w, unit_region_h,
           local_label, plane_a, plane_b, plane_c);   
        new_abc_hls(p_xx, p_yy, unary_cost_y1, unary_cost_y2, expn_region_x, expn_region_y, unit_region_x, unit_region_y, unit_region_w, unit_region_h, local_unary_cost, plane_a1, plane_b1, plane_c1);
        layer0CostLabelUpdate(iter, unary_cost_y0, unary_cost_y1, unary_cost_y2, expn_region_x, expn_region_y, expn_region_w, expn_region_h, unit_region_x, 
            plane_a, plane_b, plane_c, plane_a1, plane_b1, plane_c1, local_unary_cost, local_cost, local_label, local_disp);  
        int iter0 = (inner_iter==0)? iter : ((iter==LAYER0_HOR_REGION_NUM*LAYER0_VER_REGION_NUM - 1) ? iter : (iter + 1));
        random_xy(iter0, inner_iter, Q, I, C, p_x, p_y, pp, p_xx, p_yy);
    }
#else
   for (int inner_iter = 0; inner_iter < LAYER0_PROP_LOOP; inner_iter++) {
#pragma HLS pipeline off
        layer0ProposalParall(ver_region_i, Q, I, C, inner_iter, expn_region_x, expn_region_y, unit_region_x, unit_region_y, unit_region_w, unit_region_h,
            local_label, plane_a, plane_b, plane_c);
        layer0CostLabelUpdate(unary_cost_y0, unary_cost_y1, unary_cost_y2, expn_region_x, expn_region_y, expn_region_w, expn_region_h, unit_region_x, 
            plane_a, plane_b, plane_c, local_unary_cost, local_cost, local_label, local_disp);
    }
#endif
#endif
}


void layer0CalcWrapper(int iter, ap_uint<4> p_x[LAYER0_PROP_NUM], ap_uint<4> p_y[LAYER0_PROP_NUM], ap_uint<32> pp[LAYER0_PROP_NUM][3], ap_uint<4> p_xx[LAYER0_PROP_NUM], ap_uint<4> p_yy[LAYER0_PROP_NUM], int ver_region_num, int hor_region_num, int unary_cost_y0, int unary_cost_y1, int unary_cost_y2,
    ap_uint<32> Q[LAYER0_PROP_NUM][4096], ap_uint<32> I[LAYER0_PROP_NUM], ap_uint<32> C[LAYER0_PROP_NUM],
    unary_cost_t local_unary_cost[4][LAYER0_SIZE][LAYER0_SIZE * 3][DISP_DDR_MAX],
    ap_uint<DISP_DDR_BIT> local_cost[LAYER0_SIZE * 3][LAYER0_SIZE * 3], label_range local_label[LAYER0_SIZE * 3][LAYER0_SIZE * 3][3], disp_range local_disp[LAYER0_SIZE * 3][LAYER0_SIZE * 3]) {
#pragma HLS inline off
    if (iter < ver_region_num * hor_region_num) {
        layer0Calc(iter, p_x, p_y, pp, p_xx, p_yy, Q, I, C, unary_cost_y0, unary_cost_y1, unary_cost_y2, local_unary_cost, local_cost, local_label, local_disp);
    }
}

void layer0UramStore(int iter, int ver_region_num, int hor_region_num, 
    ap_uint<DISP_DDR_BIT> uram_cost[HEIGHT][LAYER0_SIZE * 3], label_range uram_label[HEIGHT][LAYER0_SIZE * 3][3], disp_range uram_disp[HEIGHT][LAYER0_SIZE * 3],
    ap_uint<DISP_DDR_BIT> local_cost[LAYER0_SIZE * 3][LAYER0_SIZE * 3], label_range local_label[LAYER0_SIZE * 3][LAYER0_SIZE * 3][3], disp_range local_disp[LAYER0_SIZE * 3][LAYER0_SIZE * 3]
    /*,ap_uint<DISP_DDR_BIT> local_init_cost[LAYER0_SIZE][LAYER0_SIZE], label_range local_init_label[LAYER0_SIZE][LAYER0_SIZE][3], disp_range local_init_disp[LAYER0_SIZE][LAYER0_SIZE]*/) {
    /*  */
    int ver_region_i = iter % ver_region_num;
    int hor_region_i = iter / ver_region_num;
    int unit_region_x, unit_region_y, unit_region_w, unit_region_h;
    int expn_region_x, expn_region_y, expn_region_w, expn_region_h;
    genRegionInfo(WIDTH, HEIGHT, hor_region_i, ver_region_i, LAYER0_SIZE, unit_region_x, unit_region_y, unit_region_w, unit_region_h, 
        expn_region_x, expn_region_y, expn_region_w, expn_region_h);
    /* address mapping table for uram */
    int uram_x_addr_id = hor_region_i % 3;
    int init_ver_region_i = (iter + ver_region_num + 2) % ver_region_num;
    int init_hor_region_i = (iter + ver_region_num + 2) / ver_region_num;
    int calc_ver_region_i = iter % ver_region_num;
    int calc_hor_region_i = iter / ver_region_num;
    int init_unit_region_x, init_unit_region_y, init_unit_region_w, init_unit_region_h;
    int init_expn_region_x, init_expn_region_y, init_expn_region_w, init_expn_region_h;
    genRegionInfo(WIDTH, HEIGHT, init_hor_region_i, init_ver_region_i, LAYER0_SIZE, init_unit_region_x, init_unit_region_y, init_unit_region_w, init_unit_region_h, 
        init_expn_region_x, init_expn_region_y, init_expn_region_w, init_expn_region_h);
    /*  */
    LOOP_STORE:
    for (int y = 0; y < LAYER0_SIZE * 3; y++) {
#pragma HLS pipeline II=1
        if ((expn_region_y + y) >= HEIGHT)
            continue;
        if (hor_region_i == 0) {
            L0_LOCAL_GLOBAL_VSCAN_ADDR1
        }
        else {
            if (uram_x_addr_id == 0) {
                L0_LOCAL_GLOBAL_VSCAN_ADDR0
            }
            else if (uram_x_addr_id == 2) {
                L0_LOCAL_GLOBAL_VSCAN_ADDR2
            }
            else {
                L0_LOCAL_GLOBAL_VSCAN_ADDR1
            }
        }
    }
    /*  */
    for (int y = 0; y < LAYER0_SIZE; y++) {
#pragma HLS pipeline II=1
        for (int x = 0; x < LAYER0_SIZE; x++) {
#pragma HLS unroll
            if ((init_unit_region_y + y) >= HEIGHT)
                continue;
            uram_cost[(init_unit_region_y + y)][((init_hor_region_i % 3) * LAYER0_SIZE + x)] = DISP_DDR_VAL_MAX;
            uram_label[(init_unit_region_y + y)][((init_hor_region_i % 3) * LAYER0_SIZE + x)][0] = 0;
            uram_label[(init_unit_region_y + y)][((init_hor_region_i % 3) * LAYER0_SIZE + x)][1] = 0;
            uram_label[(init_unit_region_y + y)][((init_hor_region_i % 3) * LAYER0_SIZE + x)][2] = 0;
            uram_disp[(init_unit_region_y + y)][((init_hor_region_i % 3) * LAYER0_SIZE + x)] = 0;
        }
    }
}

/*
void localExpLayer0(float ddr_unary_cost[WIDTH * HEIGHT * DISP_MAX], uchar ddr_im0_gray[WIDTH * HEIGHT],
    // float ddr_cost[WIDTH * HEIGHT], float ddr_label[WIDTH * HEIGHT * 3], float ddr_disp[WIDTH * HEIGHT]) {
    hls::stream_of_blocks<layer0_cost_blk> &cost_blk_out, hls::stream_of_blocks<layer0_label_blk> &label_blk_out, hls::stream_of_blocks<layer0_disp_blk> &disp_blk_out) {
    //intermediate cache 
    static float uram_cost[HEIGHT][LAYER0_SIZE * 3]; 
    static float uram_label[HEIGHT][LAYER0_SIZE * 3][3]; 
    static float uram_disp[HEIGHT][LAYER0_SIZE * 3]; 
    // local cache 
    static uchar local_im0_gray[4][LAYER0_SIZE][LAYER0_SIZE * 3];
    static float local_unary_cost[4][LAYER0_SIZE][LAYER0_SIZE * 3][DISP_MAX];
    //init random generators 
    static bool random_generator_init = false;
    static ap_uint<32> Q[4096];  static ap_uint<32> I  = 4095;   static ap_uint<32> C  = 362436;
    static ap_uint<32> Q1[4096]; static ap_uint<32> I1 = 963512; static ap_uint<32> C1 = 781265;
    static ap_uint<32> Q2[4096]; static ap_uint<32> I2 = 124125; static ap_uint<32> C2 = 84812;
    static ap_uint<32> Q3[4096]; static ap_uint<32> I3 = 6321;   static ap_uint<32> C3 = 98411;
    static ap_uint<32> Q4[4096]; static ap_uint<32> I4 = 320541; static ap_uint<32> C4 = 3334;
    static ap_uint<32> Q5[4096]; static ap_uint<32> I5 = 94;     static ap_uint<32> C5 = 515954;
    static ap_uint<32> Q6[4096]; static ap_uint<32> I6 = 756411; static ap_uint<32> C6 = 913354;
    if (random_generator_init == false) {
        initRandGen(0, Q, I, C); initRandGen(6413, Q1, I1, C1); initRandGen(67, Q2, I2, C2);
        initRandGen(13359, Q3, I3, C3); initRandGen(7541, Q4, I4, C4); initRandGen(986548, Q5, I5, C5);
        initRandGen(156411, Q6, I6, C6);
        random_generator_init = true;
    }
    // pre initialization 1 col & 1 unit region 
    const int hor_region_num = LAYER0_HOR_REGION_NUM;
    const int ver_region_num = LAYER0_VER_REGION_NUM;
    layer0OutloopInit(ver_region_num, Q, I, C, Q1, I1, C1, Q2, I2, C2, Q3, I3, C3, Q4, I4, C4, Q5, I5, C5, uram_cost, uram_label, uram_disp, ddr_unary_cost);
    // pre cache 4 unit region 
    layer0OutloopCache(local_unary_cost, ddr_unary_cost);
    
    for (int iter = 0; iter < (ver_region_num * hor_region_num + (ver_region_num + 1)); iter++) {
        // local cache buffer index 
        int unary_cost_y0 = (iter - 1) % 4; // calc pre 
        int unary_cost_y1 = (iter + 0) % 4; // calc curr
        int unary_cost_y2 = (iter + 1) % 4; // calc post
        int unary_cost_y3 = (iter + 2) % 4; // cache
        // cache unary & binary cost from DDR 
        if (iter < (ver_region_num * hor_region_num - 2)) {
            layerInloopCache<LAYER0_SIZE>(iter, ver_region_num, hor_region_num, unary_cost_y3, local_unary_cost, ddr_unary_cost, local_im0_gray, ddr_im0_gray);
        }
        // init bottom-right unit region 
        if (iter < (ver_region_num * hor_region_num - ver_region_num - 1)) {
            layer0InloopInit(iter, ver_region_num, hor_region_num, unary_cost_y2, Q, I, C, Q1, I1, C1, Q2, I2, C2, Q3, I3, C3, Q4, I4, C4, Q5, I5, C5, local_unary_cost, uram_cost, uram_label, uram_disp);
        }
        // calculation 
        if (iter < ver_region_num * hor_region_num) {
            layer0Calc(iter, ver_region_num, hor_region_num, unary_cost_y0, unary_cost_y1, unary_cost_y2, Q6, I6, C6, local_unary_cost, local_im0_gray, uram_cost, uram_label, uram_disp);
        }
        // transmission 
        if (iter >= (ver_region_num + 1)){
            // layerTransDDR<LAYER0_SIZE * 3, 3, LAYER0_SIZE>(iter, ver_region_num, uram_cost, uram_label, uram_disp, ddr_cost, ddr_label, ddr_disp);
            layerTransLayer<LAYER0_SIZE * 3, 3, LAYER0_SIZE, layer0_cost_blk, layer0_label_blk, layer0_disp_blk>(iter, ver_region_num, cost_blk_out, label_blk_out, disp_blk_out, uram_cost, uram_label, uram_disp);
        }
    }
}
*/

void layer0LoadCalcTransWrapper(int iter, ap_uint<4> p_x[LAYER0_PROP_NUM], ap_uint<4> p_y[LAYER0_PROP_NUM], ap_uint<32> pp[LAYER0_PROP_NUM][3], ap_uint<4> p_xx[LAYER0_PROP_NUM],  ap_uint<4> p_yy[LAYER0_PROP_NUM], int ver_region_num, int hor_region_num, 
    ap_uint<DISP_DDR_BIT> uram_cost[HEIGHT][LAYER0_SIZE * 3], label_range uram_label[HEIGHT][LAYER0_SIZE * 3][3], disp_range uram_disp[HEIGHT][LAYER0_SIZE * 3],
    ap_uint<DISP_DDR_BIT> local_cost[LAYER0_SIZE * 3][LAYER0_SIZE * 3], label_range local_label[LAYER0_SIZE * 3][LAYER0_SIZE * 3][3], disp_range local_disp[LAYER0_SIZE * 3][LAYER0_SIZE * 3],
    /*ap_uint<DISP_DDR_BIT> local_trans_cost[LAYER0_SIZE][LAYER0_SIZE], label_range local_trans_label[LAYER0_SIZE][LAYER0_SIZE][3], */disp_range local_trans_disp[LAYER0_SIZE][LAYER0_SIZE],
    int unary_cost_y0, int unary_cost_y1, int unary_cost_y2, ap_uint<32> Q[LAYER0_PROP_NUM][4096], ap_uint<32> I[LAYER0_PROP_NUM], ap_uint<32> C[LAYER0_PROP_NUM], 
#if HLS_LAYER0_ONLY == 0
    /*ap_uint<DISP_DDR_BIT> ddr_cost[WIDTH * HEIGHT], label_range ddr_label[WIDTH * HEIGHT * 3], */disp_range ddr_disp[WIDTH * HEIGHT],
#else
    float cost_blk_out[LAYER0_SIZE * LAYER0_SIZE], float label_blk_out[LAYER0_SIZE * LAYER0_SIZE * 3], float disp_blk_out[LAYER0_SIZE * LAYER0_SIZE], 
#endif
    unary_cost_t local_unary_cost[4][LAYER0_SIZE][LAYER0_SIZE * 3][DISP_DDR_MAX]) {
#pragma HLS inline off
    /* load from uram */
    layer0UramLoad(iter, LAYER0_VER_REGION_NUM, LAYER0_HOR_REGION_NUM, uram_cost, uram_label, uram_disp, local_cost, local_label, local_disp, /*local_trans_cost, local_trans_label, */local_trans_disp);
    /* calculation */
    layer0CalcWrapper(iter, p_x, p_y, pp, p_xx, p_yy, LAYER0_VER_REGION_NUM, LAYER0_HOR_REGION_NUM, unary_cost_y0, unary_cost_y1, unary_cost_y2, Q, I, C, local_unary_cost, local_cost, local_label, local_disp);
    /* transmission */
    layer0TransWrapper(iter, LAYER0_VER_REGION_NUM, LAYER0_HOR_REGION_NUM, /*ddr_cost, ddr_label, */ddr_disp, /*local_trans_cost, local_trans_label, */local_trans_disp);
}

void layer0InitCalcTransWrapper(int iter, ap_uint<4> p_x[LAYER0_PROP_NUM], ap_uint<4> p_y[LAYER0_PROP_NUM], ap_uint<32> pp[LAYER0_PROP_NUM][3], ap_uint<4> p_xx[LAYER0_PROP_NUM],  ap_uint<4> p_yy[LAYER0_PROP_NUM], int unary_cost_y0, int unary_cost_y1, int unary_cost_y2, int buff_id, 
    ap_uint<32> Q[LAYER0_PROP_NUM][4096], ap_uint<32> I[LAYER0_PROP_NUM], ap_uint<32> C[LAYER0_PROP_NUM], 
    unary_cost_t local_unary_cost[4][LAYER0_SIZE][LAYER0_SIZE * 3][DISP_DDR_MAX], 
    ap_uint<DISP_DDR_BIT> uram_cost[HEIGHT][LAYER0_SIZE * 3], label_range uram_label[HEIGHT][LAYER0_SIZE * 3][3], disp_range uram_disp[HEIGHT][LAYER0_SIZE * 3], 
#if HLS_LAYER0_ONLY == 0
    /*ap_uint<DISP_DDR_BIT> ddr_cost[WIDTH * HEIGHT], label_range ddr_label[WIDTH * HEIGHT * 3], */disp_range ddr_disp[WIDTH * HEIGHT]) {
#else
    float cost_blk_out[LAYER0_SIZE * LAYER0_SIZE], float label_blk_out[LAYER0_SIZE * LAYER0_SIZE * 3], float disp_blk_out[LAYER0_SIZE * LAYER0_SIZE]) {
#endif
#pragma HLS inline off
#pragma HLS array_partition variable=uram_cost type=complete dim=2
#pragma HLS array_partition variable=uram_label type=complete dim=2
#pragma HLS array_partition variable=uram_label type=complete dim=3
#pragma HLS array_partition variable=uram_disp type=complete dim=2
// #pragma HLS array_partition variable=local_unary_cost type=complete dim=1
// #pragma HLS array_partition variable=local_unary_cost type=complete dim=2
// // DO_PRAGMA(HLS array_partition variable=local_unary_cost type=cyclic factor=DISP_DDR_NUM dim=4) 
// DO_PRAGMA(HLS array_reshape variable=local_unary_cost type=cyclic factor=DISP_DDR_NUM dim=4) 
    assert((iter >= 0) && (iter < LAYER0_VER_REGION_NUM * LAYER0_HOR_REGION_NUM + LAYER0_VER_REGION_NUM + 2));
    // int unary_cost_x0 = (iter - 1) % 4; // calc pre 
    // int unary_cost_x1 = (iter + 0) % 4; // calc curr
    // int unary_cost_x2 = (iter + 1) % 4; // calc post
    /* local buffers */
    static ap_uint<DISP_DDR_BIT> local_trans_cost[LAYER0_SIZE][LAYER0_SIZE]; 
    static label_range local_trans_label[LAYER0_SIZE][LAYER0_SIZE][3];
    static disp_range local_trans_disp[LAYER0_SIZE][LAYER0_SIZE];
#pragma HLS array_partition variable=local_trans_cost type=complete dim=2
#pragma HLS array_partition variable=local_trans_label type=complete dim=2
#pragma HLS array_partition variable=local_trans_label type=complete dim=3
#pragma HLS array_partition variable=local_trans_disp type=complete dim=2
    static ap_uint<DISP_DDR_BIT> local_cost[LAYER0_SIZE * 3][LAYER0_SIZE * 3];
// #pragma HLS bind_storage variable=local_cost type=RAM_T2P impl=bram
    static label_range local_label[LAYER0_SIZE * 3][LAYER0_SIZE * 3][3];
    static disp_range local_disp[LAYER0_SIZE * 3][LAYER0_SIZE * 3];
#pragma HLS array_partition variable=local_cost type=complete dim=2
#pragma HLS array_partition variable=local_label type=complete dim=2
#pragma HLS array_partition variable=local_label type=complete dim=3
#pragma HLS array_partition variable=local_disp type=complete dim=2
//     uchar local_im0_gray_reorder[LAYER0_SIZE * 3][LAYER0_SIZE * 3];
// #pragma HLS array_partition variable=local_im0_gray_reorder type=complete dim=1
    // /* load from uram */
    // layer0UramLoad(iter, LAYER0_VER_REGION_NUM, LAYER0_HOR_REGION_NUM, uram_cost, uram_label, uram_disp, local_cost, local_label, local_disp, local_trans_cost, local_trans_label, local_trans_disp);
    // /* calculation */
    // layer0CalcWrapper(iter, LAYER0_VER_REGION_NUM, LAYER0_HOR_REGION_NUM, unary_cost_x0, unary_cost_x1, unary_cost_x2, Q6, I6, C6, Q7, I7, C7, Q8, I8, C8, local_unary_cost, local_cost, local_label, local_disp);
    // /* transmission */
    // layer0TransWrapper(iter, LAYER0_VER_REGION_NUM, LAYER0_HOR_REGION_NUM, cost_blk_out, label_blk_out, disp_blk_out, local_trans_cost, local_trans_label, local_trans_disp);
    layer0LoadCalcTransWrapper(iter, p_x, p_y, pp, p_xx, p_yy, LAYER0_VER_REGION_NUM, LAYER0_HOR_REGION_NUM, uram_cost, uram_label, uram_disp, local_cost, local_label, local_disp, /*local_trans_cost, local_trans_label, */local_trans_disp,
        unary_cost_y0, unary_cost_y1, unary_cost_y2, Q, I, C, /*ddr_cost, ddr_label, */ddr_disp, local_unary_cost);
    /* init bottom-right unit region */
    //layer0InitWrapper(iter, LAYER0_VER_REGION_NUM, LAYER0_HOR_REGION_NUM, buff_id, /* Q, I, C, Q1, I1, C1, Q2, I2, C2, Q3, I3, C3, Q4, I4, C4, Q5, I5, C5, */ local_init_cost, local_init_label, local_init_disp);
    /* save to uram */
    layer0UramStore(iter, LAYER0_VER_REGION_NUM, LAYER0_HOR_REGION_NUM, uram_cost, uram_label, uram_disp, local_cost, local_label, local_disp/*, local_init_cost, local_init_label, local_init_disp*/);
}

#if HLS_LAYER0_ONLY == 0
void localExpLayer0(
#if PROPOSER_RANDOM_INIT == 1
    ap_uint<32> random_num[LAYER0_PROP_NUM][3], 
#endif // PROPOSER_RANDOM_INIT == 1
    unary_cost_t0 ddr_unary_cost[WIDTH * HEIGHT * DISP_DDR_MAX0], 
    //uchar ddr_im0_gray[WIDTH * HEIGHT],
    /*ap_uint<DISP_DDR_BIT> ddr_cost[WIDTH * HEIGHT], label_range ddr_label[WIDTH * HEIGHT * 3], */disp_range ddr_disp[WIDTH * HEIGHT]) {
#else
void localExpLayer0(ap_uint<32> random_num[LAYER0_PROP_NUM][3], unary_cost_t ddr_unary_cost[WIDTH * HEIGHT * DISP_DDR_MAX], uchar ddr_im0_gray[WIDTH * HEIGHT],
    hls::stream_of_blocks<layer0_cost_blk> &cost_blk_out, hls::stream_of_blocks<layer0_label_blk> &label_blk_out, hls::stream_of_blocks<layer0_disp_blk> &disp_blk_out) {}
void localExpLayer0Hls(ap_uint<32> random_num[LAYER0_PROP_NUM][3], unary_cost_t ddr_unary_cost[WIDTH * HEIGHT * DISP_DDR_MAX], uchar ddr_im0_gray[WIDTH * HEIGHT],
    float cost_blk_out[LAYER0_SIZE * LAYER0_SIZE], float label_blk_out[LAYER0_SIZE * LAYER0_SIZE * 3], float disp_blk_out[LAYER0_SIZE * LAYER0_SIZE]) {
#endif
// #pragma HLS interface mode=m_axi bundle=BUS_LAYER0 port=ddr_unary_cost offset=slave
    // 
    static ap_uint<DISP_DDR_BIT> uram_cost[HEIGHT][LAYER0_SIZE * 3]; // row partition
#pragma HLS array_partition variable=uram_cost type=complete dim=2
    static label_range uram_label[HEIGHT][LAYER0_SIZE * 3][3]; 
#pragma HLS array_partition variable=uram_label type=complete dim=2
#pragma HLS array_partition variable=uram_label type=complete dim=3
    static disp_range uram_disp[HEIGHT][LAYER0_SIZE * 3]; 
#pragma HLS array_partition variable=uram_disp type=complete dim=2
    // 
    static uchar local_im0_gray[4][LAYER0_SIZE][LAYER0_SIZE * 3];
#pragma HLS array_partition variable=local_im0_gray type=complete dim=1
#pragma HLS array_partition variable=local_im0_gray type=complete dim=3
    //static float local_unary_cost[LAYER0_SIZE * 3][4][LAYER0_SIZE][DISP_DDR_MAX * DISP_DDR_NUM];
    static unary_cost_t local_unary_cost[4][LAYER0_SIZE][LAYER0_SIZE * 3][DISP_DDR_MAX];
#pragma HLS array_partition variable=local_unary_cost type=complete dim=1
#pragma HLS array_partition variable=local_unary_cost type=complete dim=3
#pragma HLS resource variable=local_unary_cost core=RAM_S2P_LUTRAM
//#pragma HLS array_partition variable=local_unary_cost type=cyclic dim=4 factor=2
//#pragma HLS BIND_STORAGE variable=local_unary_cost type=ram_2p impl=lutram

//DO_PRAGMA(HLS array_reshape variable=local_unary_cost type=cyclic factor=DISP_DDR_NUM dim=4) 
    // 
    static bool random_generator_init = false;
    static ap_uint<32> Q[LAYER0_PROP_NUM][4096]; static ap_uint<32> I[LAYER0_PROP_NUM]; static ap_uint<32> C[LAYER0_PROP_NUM];
#pragma HLS array_partition variable=Q type=complete dim=1
#pragma HLS array_partition variable=I type=complete dim=0
#pragma HLS array_partition variable=C type=complete dim=0
#if LAYER0_PROP_NUM == 3
    ap_uint<4> p_x[LAYER0_PROP_NUM] = {0, 1, 2};
    ap_uint<4> p_y[LAYER0_PROP_NUM] = {4, 5, 6}; 
    ap_uint<32> pp[LAYER0_PROP_NUM][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    ap_uint<4> p_xx[LAYER0_PROP_NUM] = {0, 1, 2};
    ap_uint<4> p_yy[LAYER0_PROP_NUM] = {4, 5, 6};
#else 
    ap_uint<4> p_x[LAYER0_PROP_NUM];
    ap_uint<4> p_y[LAYER0_PROP_NUM]; 
    ap_uint<32> pp[LAYER0_PROP_NUM][3];
    ap_uint<4> p_xx[LAYER0_PROP_NUM];
    ap_uint<4> p_yy[LAYER0_PROP_NUM]; 
#endif      
#pragma HLS array_partition variable=p_x type=complete dim=0
#pragma HLS array_partition variable=p_y type=complete dim=0
#pragma HLS array_partition variable=pp type=complete dim=0
#pragma HLS array_partition variable=p_xx type=complete dim=0
#pragma HLS array_partition variable=p_yy type=complete dim=0
#if PROPOSER_RANDOM_INIT == 0
    const ap_uint<32> random_num[2][3] = {
        {2357136044, 2546248239, 3071714933},
        {3626093760, 2588848963, 3684848379}
    };
#endif // PROPOSER_RANDOM_INIT == 0
#pragma HLS array_partition variable=random_num type=complete dim=0
    if (random_generator_init == false) {
        LOOP_INIT:
        for (int i = 0; i < LAYER0_PROP_NUM; i++) {
#pragma HLS unroll
            I[i] = random_num[i][0];
            C[i] = random_num[i][1];
            initRandGen(random_num[i][2], Q[i], I[i], C[i]);
        }
        random_generator_init = true;
    }
    const int hor_region_num = LAYER0_HOR_REGION_NUM;
    const int ver_region_num = LAYER0_VER_REGION_NUM;
    /* pre initialization 1 row & 1 unit region */
    layer0OutloopInit(ver_region_num, uram_cost, uram_label, uram_disp);
    /* pre cache 4 unit region (2 columns) */
    bool buff_id = false;
    static unary_cost_t local_buffer[2][DISP_DDR_MAX0];
#pragma HLS array_partition variable=local_buffer type=complete dim=1
    layer0OutloopCache(local_buffer, local_unary_cost, ddr_unary_cost);
    // int ver_region_i = 0;
    LOOP0:
    for (int iter = 0; iter < (ver_region_num * hor_region_num + (ver_region_num + 2)); iter++) {
// #pragma HLS DATAFLOW
#pragma HLS dependence variable=local_unary_cost type=intra false
        //
        int unary_cost_y0 = (((iter - 1) % 4) == -1) ? 3 : ((iter - 1) % 4); // calc pre 
        int unary_cost_y1 = (iter + 0) % 4; // calc curr
        int unary_cost_y2 = (iter + 1) % 4; // calc post
        int unary_cost_y3 = (iter + 2) % 4; // cache
        assert((unary_cost_y0 != unary_cost_y1) && (unary_cost_y0 != unary_cost_y2) && (unary_cost_y0 != unary_cost_y3) && 
               (unary_cost_y1 != unary_cost_y2) && (unary_cost_y1 != unary_cost_y3) && (unary_cost_y2 != unary_cost_y3));
        // 
        layer0CacheWrapper(local_buffer, iter, unary_cost_y3, !buff_id, local_unary_cost, ddr_unary_cost);
        // 
        layer0InitCalcTransWrapper(iter, p_x, p_y, pp, p_xx, p_yy, unary_cost_y0, unary_cost_y1, unary_cost_y2, buff_id, Q, I, C, local_unary_cost, uram_cost, uram_label, uram_disp, /*ddr_cost, ddr_label, */ddr_disp);
        buff_id = !buff_id;
#if PRINT_STATUS == 1 && !defined(__SYNTHESIS__)
        printf("\rlayer0 process: %4d / %d", iter, (ver_region_num * hor_region_num + (hor_region_num + 2)) - 1);  
        fflush(stdout);
    }
    printf("\n");
#else
    }
#endif
}

/*
template<size_t CUR_SIZE>
void layerOutloopCache(float local_unary_cost[4][CUR_SIZE][CUR_SIZE * 3][DISP_MAX], float ddr_unary_cost[WIDTH * HEIGHT * DISP_MAX], 
    uchar local_im0_gray[4][CUR_SIZE][CUR_SIZE * 3], uchar ddr_im0_gray[WIDTH * HEIGHT]) {
#pragma HLS inline off
#pragma HLS array_partition variable=local_unary_cost type=complete dim=1
#pragma HLS interface mode=m_axi bundle=BUS_LAYER0 port=ddr_unary_cost offset=slave
    for (int y = 0; y < CUR_SIZE; y++) {
        for (int x = 0; x < CUR_SIZE; x++) {
            for (int d = 0; d < DISP_MAX; d++) {
#pragma HLS pipeline II=1
                local_unary_cost[0][y][x][d] = ddr_unary_cost[(y * WIDTH + x) * DISP_MAX + d];
                local_unary_cost[1][y][x][d] = ddr_unary_cost[((y + CUR_SIZE) * WIDTH + x) * DISP_MAX + d];
                local_unary_cost[0][y][x + CUR_SIZE][d] = ddr_unary_cost[(y * WIDTH + x + CUR_SIZE) * DISP_MAX + d];
                local_unary_cost[1][y][x + CUR_SIZE][d] = ddr_unary_cost[((y + CUR_SIZE) * WIDTH + x + CUR_SIZE) * DISP_MAX + d];
            }
        }
    }
    for (int y = 0; y < CUR_SIZE; y++) {
        for (int x = 0; x < CUR_SIZE; x++) {
            local_im0_gray[0][y][x] = ddr_im0_gray[y * WIDTH + x];
            local_im0_gray[1][y][x] = ddr_im0_gray[(y + CUR_SIZE) * WIDTH + x];
            local_im0_gray[0][y][x + CUR_SIZE] = ddr_im0_gray[y * WIDTH + x + CUR_SIZE];
            local_im0_gray[1][y][x + CUR_SIZE] = ddr_im0_gray[(y + CUR_SIZE) * WIDTH + x + CUR_SIZE];
        }
    }
}
*/

void layer1OutloopCache(unary_cost_t local_unary_cost[LAYER1_SIZE][4][LAYER1_SIZE * 3][DISP_DDR_MAX], unary_cost_t ddr_unary_cost[WIDTH * HEIGHT * DISP_DDR_MAX],
    uchar local_im0_gray[LAYER1_SIZE][4][LAYER1_SIZE * 3], uchar ddr_im0_gray[WIDTH * HEIGHT]) {
#pragma HLS inline off
#pragma HLS array_partition variable=local_unary_cost type=complete dim=2
#pragma HLS array_partition variable=local_unary_cost type=complete dim=3
//DO_PRAGMA(HLS array_reshape variable=local_unary_cost type=cyclic factor=DISP_DDR_NUM dim=4)
#pragma HLS array_partition variable=local_im0_gray type=complete dim=2
#pragma HLS array_partition variable=local_im0_gray type=complete dim=3
// #pragma HLS interface mode=m_axi bundle=BUS_LAYER1 port=ddr_unary_cost offset=slave
// #pragma HLS interface mode=m_axi bundle=BUS_LAYER1 port=ddr_im0_gray offset=slave
    LOOP_COST0:
    for (int xyd = 0; xyd < LAYER1_SIZE * LAYER1_SIZE * DISP_DDR_MAX; xyd++) {
#pragma HLS pipeline II=1
        int y = xyd / (LAYER1_SIZE * DISP_DDR_MAX);
        int x = xyd % (LAYER1_SIZE * DISP_DDR_MAX) / DISP_DDR_MAX;
        int d = xyd % (LAYER1_SIZE * DISP_DDR_MAX) % DISP_DDR_MAX;
        local_unary_cost[y][0][x][d] = ddr_unary_cost[(y * WIDTH + x) * DISP_DDR_MAX + d];
    }

    LOOP_COST1:
    for (int xyd = 0; xyd < LAYER1_SIZE * LAYER1_SIZE * DISP_DDR_MAX; xyd++) {
#pragma HLS pipeline II=1
        int y = xyd / (LAYER1_SIZE * DISP_DDR_MAX);
        int x = xyd % (LAYER1_SIZE * DISP_DDR_MAX) / DISP_DDR_MAX;
        int d = xyd % (LAYER1_SIZE * DISP_DDR_MAX) % DISP_DDR_MAX;
        local_unary_cost[y][1][x][d] = ddr_unary_cost[((y + LAYER1_SIZE) * WIDTH + x) * DISP_DDR_MAX + d];
    }

    LOOP_COST2:
    for (int xyd = 0; xyd < LAYER1_SIZE * LAYER1_SIZE * DISP_DDR_MAX; xyd++) {
#pragma HLS pipeline II=1
        int y = xyd / (LAYER1_SIZE * DISP_DDR_MAX);
        int x = xyd % (LAYER1_SIZE * DISP_DDR_MAX) / DISP_DDR_MAX;
        int d = xyd % (LAYER1_SIZE * DISP_DDR_MAX) % DISP_DDR_MAX;
        local_unary_cost[y][0][x + LAYER1_SIZE][d] = ddr_unary_cost[(y * WIDTH + x + LAYER1_SIZE) * DISP_DDR_MAX + d];
    }

    LOOP_COST3:
    for (int xyd = 0; xyd < LAYER1_SIZE * LAYER1_SIZE * DISP_DDR_MAX; xyd++) {
#pragma HLS pipeline II=1
        int y = xyd / (LAYER1_SIZE * DISP_DDR_MAX);
        int x = xyd % (LAYER1_SIZE * DISP_DDR_MAX) / DISP_DDR_MAX;
        int d = xyd % (LAYER1_SIZE * DISP_DDR_MAX) % DISP_DDR_MAX;
        local_unary_cost[y][1][x + LAYER1_SIZE][d] = ddr_unary_cost[((y + LAYER1_SIZE) * WIDTH + x + LAYER1_SIZE) * DISP_DDR_MAX + d];
    }

    LOOP_IMG:
    for (int y = 0; y < LAYER1_SIZE; y++) {
        for (int x = 0; x < LAYER1_SIZE; x++) {
#pragma HLS loop_flatten
#pragma HLS pipeline II=1
            local_im0_gray[y][0][x] = ddr_im0_gray[y * WIDTH + x];
            local_im0_gray[y][1][x] = ddr_im0_gray[(y + LAYER1_SIZE) * WIDTH + x];
            local_im0_gray[y][0][x + LAYER1_SIZE] = ddr_im0_gray[y * WIDTH + x + LAYER1_SIZE];
            local_im0_gray[y][1][x + LAYER1_SIZE] = ddr_im0_gray[(y + LAYER1_SIZE) * WIDTH + x + LAYER1_SIZE];
        }
    }
}

/*
template<int prev_layer_size, int post_layer_size, typename prev_cost_blk, typename prev_label_blk, typename prev_disp_blk>
void layerInit(int& prev_iter, int post_iter, int prev_hor_region_num, int prev_ver_region_num, int post_hor_region_num, int post_ver_region_num,
    hls::stream_of_blocks<prev_cost_blk> &cost_blk_in, hls::stream_of_blocks<prev_label_blk> &label_blk_in, hls::stream_of_blocks<prev_disp_blk> &disp_blk_in,
    float uram_cost[HEIGHT][post_layer_size * 4], float uram_label[HEIGHT][post_layer_size * 4][3], float uram_disp[HEIGHT][post_layer_size * 4]) {
#pragma HLS inline off
    int blk_num;
    // generate receiving block number
    if (post_iter == -1) {
        blk_num = prev_ver_region_num * (post_layer_size / prev_layer_size) * 2; // init 2 post_layer_size rows before loop starts
    }
    else {
        int post_init_ver_region_i = (post_iter + post_ver_region_num * 2) % post_ver_region_num;
        int post_init_hor_region_i = (post_iter + post_ver_region_num * 2) / post_ver_region_num;
        int post_init_unit_region_x, post_init_unit_region_y, post_init_unit_region_w, post_init_unit_region_h;
        int post_init_expn_region_x, post_init_expn_region_y, post_init_expn_region_w, post_init_expn_region_h;
        genRegionInfo(WIDTH, HEIGHT, post_init_hor_region_i, post_init_ver_region_i, post_layer_size, post_init_unit_region_x, post_init_unit_region_y, 
            post_init_unit_region_w, post_init_unit_region_h, post_init_expn_region_x, post_init_expn_region_y, post_init_expn_region_w, post_init_expn_region_h);
        int blk_num_h = ((post_init_unit_region_h % prev_layer_size) == 0) ? (post_init_unit_region_h / prev_layer_size) : (post_init_unit_region_h / prev_layer_size) + 1;
        int blk_num_w = ((post_init_unit_region_w % prev_layer_size) == 0) ? (post_init_unit_region_w / prev_layer_size) : (post_init_unit_region_w / prev_layer_size) + 1;
        blk_num = blk_num_h * blk_num_w;
    }
    // receive blocks
    for (int i = 0; i < blk_num; i++) {
#pragma HLS loop_tripcount max=9
        hls::read_lock<prev_cost_blk> cost_bin(cost_blk_in);
        hls::read_lock<prev_label_blk> label_bin(label_blk_in);
        hls::read_lock<prev_disp_blk> disp_bin(disp_blk_in);
        int prev_ver_region_i = (prev_iter + i) % prev_ver_region_num;
        int prev_hor_region_i = (prev_iter + i) / prev_ver_region_num;
        int prev_unit_region_x, prev_unit_region_y, prev_unit_region_w, prev_unit_region_h;
        int prev_expn_region_x, prev_expn_region_y, prev_expn_region_w, prev_expn_region_h;
        genRegionInfo(WIDTH, HEIGHT, prev_hor_region_i, prev_ver_region_i, prev_layer_size, prev_unit_region_x, prev_unit_region_y, 
            prev_unit_region_w, prev_unit_region_h, prev_expn_region_x, prev_expn_region_y, prev_expn_region_w, prev_expn_region_h);
        // prev_ver_region_i % (3 * 4)
        int uram_x = (prev_hor_region_i % ((post_layer_size / prev_layer_size) * 4)) * prev_layer_size;
        for (int y = 0; y < prev_layer_size; y++) {
            if ((prev_unit_region_y + y) >= HEIGHT)
                continue;
            for (int x = 0; x < prev_layer_size; x++) {
#pragma HLS pipeline II=1
                uram_cost[(prev_unit_region_y + y)][uram_x  + x]    = cost_bin[y * prev_layer_size + x];
                uram_label[(prev_unit_region_y + y)][uram_x + x][0] = label_bin[(y * prev_layer_size + x) * 3 + 0];
                uram_label[(prev_unit_region_y + y)][uram_x + x][1] = label_bin[(y * prev_layer_size + x) * 3 + 1];
                uram_label[(prev_unit_region_y + y)][uram_x + x][2] = label_bin[(y * prev_layer_size + x) * 3 + 2];
                uram_disp[(prev_unit_region_y + y)][uram_x  + x]    = disp_bin[y * prev_layer_size + x];
            }
        }
    }
    prev_iter += blk_num;
}
*/

int layer1InitOutloop(int l0_iter, hls::stream_of_blocks<layer0_cost_blk> &cost_blk_in, hls::stream_of_blocks<layer0_label_blk> &label_blk_in, hls::stream_of_blocks<layer0_disp_blk> &disp_blk_in,
    ap_uint<DISP_DDR_BIT> uram_cost[HEIGHT][4][LAYER1_SIZE], label_range uram_label[HEIGHT][4][LAYER1_SIZE][3], disp_range uram_disp[HEIGHT][4][LAYER1_SIZE]) {
    /*  */
    int blk_num = LAYER0_VER_REGION_NUM * (LAYER1_SIZE / LAYER0_SIZE) * 2; // init 2 LAYER1_SIZE rows before loop starts
    // receive blocks
    LOOP_CACHE_LAYER0_OUTLOOP:
    for (int i = 0; i < blk_num; i++) {
// #pragma HLS loop_tripcount max=4
#if HLS_LAYER1_ONLY == 0
#pragma HLS loop_flatten off
        hls::read_lock<layer0_cost_blk> cost_bin(cost_blk_in);
        hls::read_lock<layer0_label_blk> label_bin(label_blk_in);
        hls::read_lock<layer0_disp_blk> disp_bin(disp_blk_in);
#endif
        int l0_ver_region_i = (l0_iter + i) % LAYER0_VER_REGION_NUM;
        int l0_hor_region_i = (l0_iter + i) / LAYER0_VER_REGION_NUM;
        int l0_unit_region_x, l0_unit_region_y, l0_unit_region_w, l0_unit_region_h;
        int l0_expn_region_x, l0_expn_region_y, l0_expn_region_w, l0_expn_region_h;
        genRegionInfo(WIDTH, HEIGHT, l0_hor_region_i, l0_ver_region_i, LAYER0_SIZE, l0_unit_region_x, l0_unit_region_y, 
            l0_unit_region_w, l0_unit_region_h, l0_expn_region_x, l0_expn_region_y, l0_expn_region_w, l0_expn_region_h);
        // l0_ver_region_i % (3 * 4)
        int base_x = l0_unit_region_x % LAYER1_SIZE;
        int uram_x = l0_unit_region_x / LAYER1_SIZE;
        // fprintf(log_file, "l1_iter=%d, l0_iter=%d, i=%d, y=%d, x=%d\n", l1_iter, l0_iter, i, uram_y * 18 + base_y, l0_unit_region_x);
        for (int y = 0; y < LAYER0_SIZE; y++) {
            for (int x = 0; x < LAYER0_SIZE; x++) {
#pragma HLS pipeline II=1
                if ((l0_unit_region_y + y) >= HEIGHT)
                    continue;
                uram_cost[(l0_unit_region_y + y)][uram_x][base_x + x]    = cost_bin[y * LAYER0_SIZE + x];
                uram_label[(l0_unit_region_y + y)][uram_x][base_x + x][0] = label_bin[(y * LAYER0_SIZE + x) * 3 + 0];
                uram_label[(l0_unit_region_y + y)][uram_x][base_x + x][1] = label_bin[(y * LAYER0_SIZE + x) * 3 + 1];
                uram_label[(l0_unit_region_y + y)][uram_x][base_x + x][2] = label_bin[(y * LAYER0_SIZE + x) * 3 + 2];
                uram_disp[(l0_unit_region_y + y)][uram_x][base_x + x]    = disp_bin[y * LAYER0_SIZE + x];
            }
        }
    }
    return blk_num;
}

int layer1InitInloop(int l1_iter, int l0_iter, int uram_x3, hls::stream_of_blocks<layer0_cost_blk> &cost_blk_in, hls::stream_of_blocks<layer0_label_blk> &label_blk_in, hls::stream_of_blocks<layer0_disp_blk> &disp_blk_in,
    ap_uint<DISP_DDR_BIT> uram_cost[HEIGHT][4][LAYER1_SIZE], label_range uram_label[HEIGHT][4][LAYER1_SIZE][3], disp_range uram_disp[HEIGHT][4][LAYER1_SIZE]) {
    /*  */
    int l1_init_ver_region_i = (l1_iter + LAYER1_VER_REGION_NUM * 2) % LAYER1_VER_REGION_NUM;
    int l1_init_hor_region_i = (l1_iter + LAYER1_VER_REGION_NUM * 2) / LAYER1_VER_REGION_NUM;
    int l1_init_unit_region_x, l1_init_unit_region_y, l1_init_unit_region_w, l1_init_unit_region_h;
    int l1_init_expn_region_x, l1_init_expn_region_y, l1_init_expn_region_w, l1_init_expn_region_h;
    genRegionInfo(WIDTH, HEIGHT, l1_init_hor_region_i, l1_init_ver_region_i, LAYER1_SIZE, l1_init_unit_region_x, l1_init_unit_region_y, 
        l1_init_unit_region_w, l1_init_unit_region_h, l1_init_expn_region_x, l1_init_expn_region_y, l1_init_expn_region_w, l1_init_expn_region_h);
    int blk_num_h = ((l1_init_unit_region_h % LAYER0_SIZE) == 0) ? (l1_init_unit_region_h / LAYER0_SIZE) : (l1_init_unit_region_h / LAYER0_SIZE) + 1;
    int blk_num_w = ((l1_init_unit_region_w % LAYER0_SIZE) == 0) ? (l1_init_unit_region_w / LAYER0_SIZE) : (l1_init_unit_region_w / LAYER0_SIZE) + 1;
    int blk_num = blk_num_h * blk_num_w;
    /* receive blocks */
    if (l1_iter < (LAYER1_VER_REGION_NUM * LAYER1_HOR_REGION_NUM - 2 * LAYER1_VER_REGION_NUM)) {
        LOOP_CACHE_LAYER0_INLOOP:
        for (int i = 0; i < blk_num; i++) {
#pragma HLS loop_tripcount max=4
#pragma HLS loop_flatten off
#if HLS_LAYER1_ONLY == 0
            hls::read_lock<layer0_cost_blk> cost_bin(cost_blk_in);
            hls::read_lock<layer0_label_blk> label_bin(label_blk_in);
            hls::read_lock<layer0_disp_blk> disp_bin(disp_blk_in);
#endif
            int l0_ver_region_i = (l0_iter + i) % LAYER0_VER_REGION_NUM;
            int l0_hor_region_i = (l0_iter + i) / LAYER0_VER_REGION_NUM;
            int l0_unit_region_x, l0_unit_region_y, l0_unit_region_w, l0_unit_region_h;
            int l0_expn_region_x, l0_expn_region_y, l0_expn_region_w, l0_expn_region_h;
            genRegionInfo(WIDTH, HEIGHT, l0_hor_region_i, l0_ver_region_i, LAYER0_SIZE, l0_unit_region_x, l0_unit_region_y, 
                l0_unit_region_w, l0_unit_region_h, l0_expn_region_x, l0_expn_region_y, l0_expn_region_w, l0_expn_region_h);
            // l0_ver_region_i % (3 * 4)
            int base_x = l0_unit_region_x % LAYER1_SIZE;
            // fprintf(log_file, "l1_iter=%d, l0_iter=%d, i=%d, y=%d, x=%d\n", l1_iter, l0_iter, i, uram_y3 * 18 + base_y, l0_unit_region_x);
            // LOOP_CACHE_LAYER0:
            for (int y = 0; y < LAYER0_SIZE; y++) {
                for (int x = 0; x < LAYER0_SIZE; x++) {
#pragma HLS pipeline II=1
                    if ((l0_unit_region_y + y) >= HEIGHT)
                        continue;
                    uram_cost[(l0_unit_region_y + y)][uram_x3][base_x + x]    = cost_bin[y * LAYER0_SIZE + x];
                    uram_label[(l0_unit_region_y + y)][uram_x3][base_x + x][0] = label_bin[(y * LAYER0_SIZE + x) * 3 + 0];
                    uram_label[(l0_unit_region_y + y)][uram_x3][base_x + x][1] = label_bin[(y * LAYER0_SIZE + x) * 3 + 1];
                    uram_label[(l0_unit_region_y + y)][uram_x3][base_x + x][2] = label_bin[(y * LAYER0_SIZE + x) * 3 + 2];
                    uram_disp[(l0_unit_region_y + y)][uram_x3][base_x + x]    = disp_bin[y * LAYER0_SIZE + x];
                }
            }
        }
    }
    return blk_num;
}

/*
void layer1Calc(int iter, int ver_region_num, int hor_region_num, int unary_cost_x0, int unary_cost_x1, int unary_cost_x2,
    ap_uint<32>* Q, ap_uint<32>& I, ap_uint<32>& C,
    float local_unary_cost[4][LAYER1_SIZE][LAYER1_SIZE * 3][DISP_MAX], uchar local_im0_gray[4][LAYER1_SIZE][LAYER1_SIZE * 3], 
    float uram_cost[HEIGHT][LAYER1_SIZE * 4], float uram_label[HEIGHT][LAYER1_SIZE * 4][3], float uram_disp[HEIGHT][LAYER1_SIZE * 4]) {
    //
    float local_cost[LAYER1_SIZE * 3][LAYER1_SIZE * 3];
    float local_label[LAYER1_SIZE * 3][LAYER1_SIZE * 3][3];
    float local_disp[LAYER1_SIZE * 3][LAYER1_SIZE * 3];
    // 
    float plane_a, plane_b, plane_c;
    //
    int ver_region_i = iter % ver_region_num;
    int hor_region_i = iter / ver_region_num;
    int unit_region_x, unit_region_y, unit_region_w, unit_region_h;
    int expn_region_x, expn_region_y, expn_region_w, expn_region_h;
    genRegionInfo(WIDTH, HEIGHT, hor_region_i, ver_region_i, LAYER1_SIZE, unit_region_x, unit_region_y, unit_region_w, unit_region_h, 
        expn_region_x, expn_region_y, expn_region_w, expn_region_h);
    // address mapping table for uram
    int uram_x_addr_id = hor_region_i % 4;
    // uram -> local ram
    LOOP_LOAD:
    for (int y = 0; y < LAYER1_SIZE * 3; y++) {
#pragma HLS pipeline II=1
        if ((expn_region_y + y) >= HEIGHT)   
            continue;
        if (hor_region_i == 0) {
            L1_GLOBAL_LOCAL_VSCAN_ADDR1
        }
        else {
            if (uram_x_addr_id == 0) {
                L1_GLOBAL_LOCAL_VSCAN_ADDR0
            }
            else if (uram_x_addr_id == 1) {
                L1_GLOBAL_LOCAL_VSCAN_ADDR1
            }
            else if (uram_x_addr_id == 2) {
                L1_GLOBAL_LOCAL_VSCAN_ADDR2
            }
            else if (uram_x_addr_id == 3) {
                L1_GLOBAL_LOCAL_VSCAN_ADDR3
            }
        }
    }
    // process local ram
#if LAYER1_ENABLE
    LOOP_PROPOSAL:
    for (int inner_iter = 0; inner_iter < 8; inner_iter++) {
#pragma HLS pipeline off
        if (inner_iter == 0)
            layerExpnProposal<LAYER1_SIZE>(ver_region_i, Q, I, C, expn_region_x, expn_region_y, unit_region_x, unit_region_y, unit_region_w, unit_region_h, 
                local_label, plane_a, plane_b, plane_c);
        else
            layerRandProposal<LAYER1_SIZE>(ver_region_i, Q, I, C, 0, inner_iter -1, expn_region_x, expn_region_y, unit_region_x, unit_region_y, unit_region_w, unit_region_h, 
                local_label, plane_a, plane_b, plane_c);
        costLabelUpdate<LAYER1_SIZE>(unary_cost_x0, unary_cost_x1, unary_cost_x2, expn_region_x, expn_region_y, expn_region_w, expn_region_h, unit_region_y, 
            plane_a, plane_b, plane_c, local_unary_cost, local_im0_gray, local_cost, local_label, local_disp, true, false);
    }
#endif
    // local ram -> uram
    LOOP_STORE:
    for (int y = 0; y < LAYER1_SIZE * 3; y++) {
#pragma HLS pipeline II=1
        if ((expn_region_y + y) >= HEIGHT)
            continue;
        if (hor_region_i == 0) {
            L1_LOCAL_GLOBAL_VSCAN_ADDR1
        }
        else {
            if (uram_x_addr_id == 0) {
                L1_LOCAL_GLOBAL_VSCAN_ADDR0
            }
            else if (uram_x_addr_id == 1) {
                L1_LOCAL_GLOBAL_VSCAN_ADDR1
            }
            else if (uram_x_addr_id == 2) {
                L1_LOCAL_GLOBAL_VSCAN_ADDR2
            }
            else if (uram_x_addr_id == 3) {
                L1_LOCAL_GLOBAL_VSCAN_ADDR3
            }
        }
    }
}
*/

void layer1UramLoad(int iter, int uram_x0, int uram_x1, int uram_x2, int& tran_unit_region_x, int& tran_unit_region_y, int& tran_unit_region_w, int& tran_unit_region_h,
    ap_uint<DISP_DDR_BIT> uram_cost[HEIGHT][4][LAYER1_SIZE], label_range uram_label[HEIGHT][4][LAYER1_SIZE][3], disp_range uram_disp[HEIGHT][4][LAYER1_SIZE],
    ap_uint<DISP_DDR_BIT> local_cost[LAYER1_SIZE * 3][LAYER1_SIZE * 3], label_range local_label[LAYER1_SIZE * 3][LAYER1_SIZE * 3][3], disp_range local_disp[LAYER1_SIZE * 3][LAYER1_SIZE * 3],
    ap_uint<DISP_DDR_BIT> local_trans_cost[LAYER1_SIZE][LAYER1_SIZE], label_range local_trans_label[LAYER1_SIZE][LAYER1_SIZE][3], disp_range local_trans_disp[LAYER1_SIZE][LAYER1_SIZE],
    ap_uint<DISP_DDR_BIT> local_trans_cost_tmp[LAYER1_SIZE * 2][LAYER1_SIZE], label_range local_trans_label_tmp[LAYER1_SIZE * 2][LAYER1_SIZE][3], disp_range local_trans_disp_tmp[LAYER1_SIZE * 2][LAYER1_SIZE]) {
// #pragma HLS array_partition variable=uram_cost type=complete dim=1
// #pragma HLS array_partition variable=uram_label type=complete dim=1
// #pragma HLS array_partition variable=uram_label type=complete dim=3 
// #pragma HLS array_partition variable=uram_disp type=complete dim=1
#pragma HLS array_partition variable=local_cost type=complete dim=2
#pragma HLS array_partition variable=local_label type=complete dim=2
#pragma HLS array_partition variable=local_label type=complete dim=3
#pragma HLS array_partition variable=local_disp type=complete dim=2
#pragma HLS array_partition variable=local_trans_cost type=complete dim=2
#pragma HLS array_partition variable=local_trans_label type=complete dim=2
#pragma HLS array_partition variable=local_trans_label type=complete dim=3
#pragma HLS array_partition variable=local_trans_disp type=complete dim=2
    // FILE* log_file = fopen("/home/jyc/prj/0GIT/stereo_project/prj_local_expansion_hls/prj_linux/new.log", "a");
    // if (log_file == NULL) {
    //     printf("Error: failed to create log file.\n");
    //     return;
    // }
    // fprintf(log_file, "iter=%d, %d, %d, %d\n", iter, uram_y0, uram_y1, uram_y2);
    /*  */
    int ver_region_i = iter % LAYER1_VER_REGION_NUM;
    int hor_region_i = iter / LAYER1_VER_REGION_NUM;
    int unit_region_x, unit_region_y, unit_region_w, unit_region_h;
    int expn_region_x, expn_region_y, expn_region_w, expn_region_h;
    genRegionInfo(WIDTH, HEIGHT, hor_region_i, ver_region_i, LAYER1_SIZE, unit_region_x, unit_region_y, unit_region_w, unit_region_h, 
        expn_region_x, expn_region_y, expn_region_w, expn_region_h);
    //
     int uram_x_addr_id = hor_region_i % 4;
    // uram -> local ram
    LOOP_LOAD:
    for (int y = 0; y < LAYER1_SIZE * 3; y++) {
#pragma HLS pipeline II=1
        if ((expn_region_y + y) >= HEIGHT)   
            continue;
        L1_GLOBAL_LOCAL_URAM_VSCAN_Y
    }
    /*  */
    int tran_ver_region_i = (iter - (LAYER1_VER_REGION_NUM + 2)) % LAYER1_VER_REGION_NUM;
    int tran_hor_region_i = (iter - (LAYER1_VER_REGION_NUM + 2)) / LAYER1_VER_REGION_NUM;
    // int tran_unit_region_x, tran_unit_region_y, tran_unit_region_w, tran_unit_region_h;
    int tran_expn_region_x, tran_expn_region_y, tran_expn_region_w, tran_expn_region_h;
    genRegionInfo(WIDTH, HEIGHT, tran_hor_region_i, tran_ver_region_i, LAYER1_SIZE, tran_unit_region_x, tran_unit_region_y, tran_unit_region_w, tran_unit_region_h, 
        tran_expn_region_x, tran_expn_region_y, tran_expn_region_w, tran_expn_region_h);
    if (tran_ver_region_i < 0 || tran_hor_region_i < 0)
        return;
    // fprintf(log_file, "iter=%d, %d\n", iter, uram_y0);
    LOOP_TRANS_LOAD:
    if ((iter % LAYER1_VER_REGION_NUM) == 0) {
        for (int y = 0; y < LAYER1_SIZE; y++) {
#pragma HLS pipeline II=1
            LOCAL_TRANS_TMP_VSCAN_LOAD0
        }
    }
    else if (((iter - 1) % LAYER1_VER_REGION_NUM) == 0) {
        for (int y = 0; y < LAYER1_SIZE; y++) {
#pragma HLS pipeline II=1
            LOCAL_TRANS_TMP_VSCAN_LOAD1
        }
    }
    else {
        for (int y = 0; y < LAYER1_SIZE; y++) {
#pragma HLS pipeline II=1
            if ((tran_unit_region_y + y) >= HEIGHT)
                continue;
            L1_GLOBAL_LOCAL_TRANS_URAM_VSCAN_Y
        }
    }
    // fclose(log_file);
    // float dbg_local_trans_disp[LAYER0_SIZE][LAYER0_SIZE];
    // for (int y = 0; y < LAYER0_SIZE; y++) {
    //     for (int x = 0; x < LAYER0_SIZE; x++) {
    //         dbg_local_trans_disp[y][x] = local_trans_disp[y][x];
    //     }
    // }
    // printf("unit_region_x: %d ,unit_region_y: %d, unit_region_w: %d ,unit_region_h: %d", tran_unit_region_x, tran_unit_region_y, tran_unit_region_w, tran_unit_region_h);
    // printf("\n");
}

void layer1Calc(int iter, int ver_region_num, int hor_region_num, int unary_cost_y0, int unary_cost_y1, int unary_cost_y2,
    ap_uint<32> Q[LAYER1_PROP_NUM][4096], ap_uint<32> I[LAYER1_PROP_NUM], ap_uint<32> C[LAYER1_PROP_NUM],
    unary_cost_t local_unary_cost[LAYER1_SIZE][4][LAYER1_SIZE * 3][DISP_DDR_MAX], uchar local_im0_gray[LAYER1_SIZE][4][LAYER1_SIZE * 3], 
    ap_uint<DISP_DDR_BIT> local_cost[LAYER1_SIZE * 3][LAYER1_SIZE * 3], label_range local_label[LAYER1_SIZE * 3][LAYER1_SIZE * 3][3], disp_range local_disp[LAYER1_SIZE * 3][LAYER1_SIZE * 3]) {
#pragma HLS inline off
#pragma HLS array_partition variable=local_cost type=complete dim=2
#pragma HLS array_partition variable=local_label type=complete dim=2
#pragma HLS array_partition variable=local_label type=complete dim=3
#pragma HLS array_partition variable=local_disp type=complete dim=2
#pragma HLS array_partition variable=local_unary_cost type=complete dim=2
#pragma HLS array_partition variable=local_unary_cost type=complete dim=3
//DO_PRAGMA(HLS array_reshape variable=local_unary_cost type=cyclic factor=DISP_DDR_NUM dim=4) 
#pragma HLS array_partition variable=local_im0_gray type=complete dim=2
#pragma HLS array_partition variable=local_im0_gray type=complete dim=3
    //
    label_range plane_a[LAYER1_PROP_NUM], plane_b[LAYER1_PROP_NUM], plane_c[LAYER1_PROP_NUM];
#pragma HLS array_partition variable=plane_a type=complete dim=1
#pragma HLS array_partition variable=plane_b type=complete dim=1
#pragma HLS array_partition variable=plane_c type=complete dim=1
    //
    int ver_region_i = iter % LAYER1_VER_REGION_NUM;
    int hor_region_i = iter / LAYER1_VER_REGION_NUM;
    int unit_region_x, unit_region_y, unit_region_w, unit_region_h;
    int expn_region_x, expn_region_y, expn_region_w, expn_region_h;
    genRegionInfo(WIDTH, HEIGHT, hor_region_i, ver_region_i, LAYER1_SIZE, unit_region_x, unit_region_y, unit_region_w, unit_region_h, 
        expn_region_x, expn_region_y, expn_region_w, expn_region_h);
    //printf("unit_region_x1: %d ,unit_region_y1: %d\n", unit_region_x, unit_region_y);
#if LAYER1_ENABLE 
    /* gray image re-order */
    uchar local_im0_gray_reorder[LAYER1_SIZE * 3][LAYER1_SIZE * 3];
#pragma HLS array_partition variable=local_im0_gray_reorder type=complete dim=2
    LOOP_IMG_REORDER:
    for (int y = 0; y < LAYER1_SIZE * 3; y++) {
#pragma HLS pipeline II=1
        int y_remainder = y % LAYER1_SIZE;
        int local_y_id = 0;
        if (expn_region_y == unit_region_y) {
            if (y < LAYER1_SIZE)
                local_y_id = unary_cost_y1;
            else if (y < LAYER1_SIZE * 2)
                local_y_id = unary_cost_y2;
        }
        else {
            if (y < LAYER1_SIZE)
                local_y_id = unary_cost_y0;
            else if (y < LAYER1_SIZE * 2)
                local_y_id = unary_cost_y1;
            else
                local_y_id = unary_cost_y2;
        }
        for (int x = 0; x < LAYER1_SIZE * 3; x++) {
#pragma HLS unroll
            local_im0_gray_reorder[y][x] = local_im0_gray[y_remainder][local_y_id][x];
        }
    }
    /* process local ram */
    LOOP_PROPOSAL:
    for (int inner_iter = 0; inner_iter < LAYER1_PROP_LOOP; inner_iter++) {
#pragma HLS pipeline off
        layer1ProposalParall(ver_region_i, Q, I, C, inner_iter, expn_region_x, expn_region_y, unit_region_x, unit_region_y, unit_region_w, unit_region_h,
            local_label, plane_a, plane_b, plane_c);
        layer1CostLabelUpdate(unary_cost_y0, unary_cost_y1, unary_cost_y2, expn_region_x, expn_region_y, expn_region_w, expn_region_h, unit_region_y, plane_a, plane_b, plane_c, 
            local_unary_cost, local_im0_gray_reorder, local_cost, local_label, local_disp);
    }
#endif
}

void layer1CalcWrapper(int l1_iter, int unary_cost_y0, int unary_cost_y1, int unary_cost_y2, 
    ap_uint<32> Q[LAYER1_PROP_NUM][4096], ap_uint<32> I[LAYER1_PROP_NUM], ap_uint<32> C[LAYER1_PROP_NUM],
    unary_cost_t local_unary_cost[LAYER1_SIZE][4][LAYER1_SIZE * 3][DISP_DDR_MAX], uchar local_im0_gray[LAYER1_SIZE][4][LAYER1_SIZE * 3], 
    ap_uint<DISP_DDR_BIT> local_cost[LAYER1_SIZE * 3][LAYER1_SIZE * 3], label_range local_label[LAYER1_SIZE * 3][LAYER1_SIZE * 3][3], disp_range local_disp[LAYER1_SIZE * 3][LAYER1_SIZE * 3]) {
#pragma HLS inline off
    /*  */
    if ((l1_iter >= 0) && (l1_iter < LAYER1_VER_REGION_NUM * LAYER1_HOR_REGION_NUM)) {
        layer1Calc(l1_iter, LAYER1_VER_REGION_NUM, LAYER1_HOR_REGION_NUM, unary_cost_y0, unary_cost_y1, unary_cost_y2, Q, I, C, local_unary_cost, local_im0_gray, local_cost, local_label, local_disp);
    }
}

void layer1UramStore(int iter, int uram_x0, int uram_x1, int uram_x2, 
    ap_uint<DISP_DDR_BIT> uram_cost[HEIGHT][4][LAYER1_SIZE], label_range uram_label[HEIGHT][4][LAYER1_SIZE][3], disp_range uram_disp[HEIGHT][4][LAYER1_SIZE],
    ap_uint<DISP_DDR_BIT> local_cost[LAYER1_SIZE * 3][LAYER1_SIZE * 3], label_range local_label[LAYER1_SIZE * 3][LAYER1_SIZE * 3][3], disp_range local_disp[LAYER1_SIZE * 3][LAYER1_SIZE * 3],
    ap_uint<DISP_DDR_BIT> local_trans_cost_tmp[LAYER1_SIZE * 2][LAYER1_SIZE], label_range local_trans_label_tmp[LAYER1_SIZE * 2][LAYER1_SIZE][3], disp_range local_trans_disp_tmp[LAYER1_SIZE * 2][LAYER1_SIZE]) {
// #pragma HLS array_partition variable=uram_cost type=complete dim=1
// #pragma HLS array_partition variable=uram_label type=complete dim=1
// #pragma HLS array_partition variable=uram_label type=complete dim=3 
// #pragma HLS array_partition variable=uram_disp type=complete dim=1
#pragma HLS array_partition variable=local_cost type=complete dim=2
#pragma HLS array_partition variable=local_label type=complete dim=2
#pragma HLS array_partition variable=local_label type=complete dim=3
#pragma HLS array_partition variable=local_disp type=complete dim=2
    //
    int ver_region_i = iter % LAYER1_VER_REGION_NUM;
    int hor_region_i = iter / LAYER1_VER_REGION_NUM;
    int unit_region_x, unit_region_y, unit_region_w, unit_region_h;
    int expn_region_x, expn_region_y, expn_region_w, expn_region_h;
    genRegionInfo(WIDTH, HEIGHT, hor_region_i, ver_region_i, LAYER1_SIZE, unit_region_x, unit_region_y, unit_region_w, unit_region_h, 
        expn_region_x, expn_region_y, expn_region_w, expn_region_h);
    //
    int uram_x_addr_id = hor_region_i % 4;
    /* local ram -> uram */
    LOOP_STORE:
    for (int y = 0; y < LAYER1_SIZE * 3; y++) {
#pragma HLS pipeline II=1
        if ((expn_region_y + y) >= HEIGHT)
            continue;
        L1_LOCAL_GLOBAL_URAM_VSCAN_Y
    }
    if ((iter + 1) % LAYER1_VER_REGION_NUM == 0) {
        for (int y = 0; y < LAYER1_SIZE * 2; y++) {
#pragma HLS pipeline II=1
            LOCAL_TRANS_TMP_VSCAN_STORE
        }
    }
}

/*
void localExpLayer1(float ddr_unary_cost[WIDTH * HEIGHT * DISP_MAX], uchar ddr_im0_gray[WIDTH * HEIGHT], 
    hls::stream_of_blocks<layer0_cost_blk> &cost_blk_in, hls::stream_of_blocks<layer0_label_blk> &label_blk_in, hls::stream_of_blocks<layer0_disp_blk> &disp_blk_in,
    hls::stream_of_blocks<layer1_cost_blk> &cost_blk_out, hls::stream_of_blocks<layer1_label_blk> &label_blk_out, hls::stream_of_blocks<layer1_disp_blk> &disp_blk_out) {
    // float ddr_cost[WIDTH * HEIGHT], float ddr_label[WIDTH * HEIGHT * 3], float ddr_disp[WIDTH * HEIGHT]) {
    // intermediate cache 
    static float uram_cost[HEIGHT][LAYER1_SIZE * 4]; 
    static float uram_label[HEIGHT][LAYER1_SIZE * 4][3]; 
    static float uram_disp[HEIGHT][LAYER1_SIZE * 4]; 
    // local cache 
    static uchar local_im0_gray[4][LAYER1_SIZE][LAYER1_SIZE * 3];
    static float local_unary_cost[4][LAYER1_SIZE][LAYER1_SIZE * 3][DISP_MAX];
    // init random generators 
    static bool random_generator_init = false;
    static ap_uint<32> Q[4096];
    static ap_uint<32> I = 1731;
    static ap_uint<32> C = 793451;
    if (random_generator_init == false) {
        initRandGen(0, Q, I, C); 
        random_generator_init = true;
    }
    // region parameters 
    const int l0_hor_region_num = LAYER0_HOR_REGION_NUM;
    const int l0_ver_region_num = LAYER0_VER_REGION_NUM;
    const int l1_hor_region_num = LAYER1_HOR_REGION_NUM;
    const int l1_ver_region_num = LAYER1_VER_REGION_NUM;
    // pre cache 4 unit region 
    layerOutloopCache<LAYER1_SIZE>(local_unary_cost, ddr_unary_cost, local_im0_gray, ddr_im0_gray);
    // loop expansion 
    for (int l0_iter = 0, l1_iter = -1; l1_iter < (l1_ver_region_num * l1_hor_region_num + (l1_ver_region_num + 1)); l1_iter++) {
        // local cache buffer index 
        int unary_cost_y0 = (l1_iter - 1) % 4; // calc pre
        int unary_cost_y1 = (l1_iter + 0) % 4; // calc curr
        int unary_cost_y2 = (l1_iter + 1) % 4; // calc post
        int unary_cost_y3 = (l1_iter + 2) % 4; // cache
        // init using data blocks from layer0 
        if (l1_iter < (l1_ver_region_num * l1_hor_region_num - 2 * l1_ver_region_num)) {
            layerInit<LAYER0_SIZE, LAYER1_SIZE, layer0_cost_blk, layer0_label_blk, layer0_disp_blk>(l0_iter, l1_iter, l0_hor_region_num, l0_ver_region_num, l1_hor_region_num, l1_ver_region_num, cost_blk_in, label_blk_in, disp_blk_in, uram_cost, uram_label, uram_disp);
        }
        // cache unary cost & gray image from DDR 
        if ((l1_iter >= 0) && (l1_iter < (l1_ver_region_num * l1_hor_region_num - 2))) {
            layerInloopCache<LAYER1_SIZE>(l1_iter, l1_ver_region_num, l1_hor_region_num, unary_cost_y3, local_unary_cost, ddr_unary_cost, local_im0_gray, ddr_im0_gray);
        }
        // calculation 
        if ((l1_iter >= 0) && (l1_iter < l1_ver_region_num * l1_hor_region_num)) {
            layer1Calc(l1_iter, l1_ver_region_num, l1_hor_region_num, unary_cost_y0, unary_cost_y1, unary_cost_y2, Q, I, C, local_unary_cost, local_im0_gray, uram_cost, uram_label, uram_disp);
        }
        // transmit data blocks to layer2 
        if (l1_iter >= (l1_ver_region_num + 1)){
            // layerTransDDR<LAYER1_SIZE * 4, 4, LAYER1_SIZE>(l1_iter, l1_ver_region_num, uram_cost, uram_label, uram_disp, ddr_cost, ddr_label, ddr_disp);
            layerTransLayer<LAYER1_SIZE * 4, 4, LAYER1_SIZE, layer1_cost_blk, layer1_label_blk, layer1_disp_blk>(l1_iter, l1_ver_region_num, cost_blk_out, label_blk_out, disp_blk_out, uram_cost, uram_label, uram_disp);
        }
    }
}
*/

void layer1CalcTransWrapper(int& l0_iter, int l1_iter, int unary_cost_y0, int unary_cost_y1, int unary_cost_y2, int uram_x0, int uram_x1, int uram_x2, int uram_x3, 
    ap_uint<32> Q[LAYER1_PROP_NUM][4096], ap_uint<32> I[LAYER1_PROP_NUM], ap_uint<32> C[LAYER1_PROP_NUM],
    unary_cost_t local_unary_cost[LAYER1_SIZE][4][LAYER1_SIZE * 3][DISP_DDR_MAX], uchar local_im0_gray[LAYER1_SIZE][4][LAYER1_SIZE * 3], 
    ap_uint<DISP_DDR_BIT> uram_cost[HEIGHT][4][LAYER1_SIZE], label_range uram_label[HEIGHT][4][LAYER1_SIZE][3], disp_range uram_disp[HEIGHT][4][LAYER1_SIZE],
#if HLS_LAYER1_ONLY == 0
    hls::stream_of_blocks<layer0_cost_blk> &cost_blk_in, hls::stream_of_blocks<layer0_label_blk> &label_blk_in, hls::stream_of_blocks<layer0_disp_blk> &disp_blk_in,
    /* hls::stream_of_blocks<layer0_cost_blk> &init_cost_blk_in, hls::stream_of_blocks<layer0_label_blk> &init_label_blk_in, hls::stream_of_blocks<layer0_disp_blk> &init_disp_blk_in, */
#else
    float cost_blk_in[LAYER0_SIZE * LAYER0_SIZE], float label_blk_in[LAYER0_SIZE * LAYER0_SIZE * 3], float disp_blk_in[LAYER0_SIZE * LAYER0_SIZE],
#endif
#if LAYER2_ENABLE == 0
    ap_uint<DISP_DDR_BIT> ddr_cost[WIDTH * HEIGHT], label_range ddr_label[WIDTH * HEIGHT * 3], disp_range ddr_disp[WIDTH * HEIGHT]) {
#elif HLS_LAYER1_ONLY == 0
    hls::stream_of_blocks<layer1_cost_blk> &cost_blk_out, hls::stream_of_blocks<layer1_label_blk> &label_blk_out, hls::stream_of_blocks<layer1_disp_blk> &disp_blk_out) {
#else
    float cost_blk_out[LAYER1_SIZE * LAYER1_SIZE], float label_blk_out[LAYER1_SIZE * LAYER1_SIZE * 3], float disp_blk_out[LAYER1_SIZE * LAYER1_SIZE]) {
#endif
#pragma HLS inline off
    int tran_unit_region_x, tran_unit_region_y, tran_unit_region_w, tran_unit_region_h;
    /* local buffer */
    static ap_uint<DISP_DDR_BIT> local_cost[LAYER1_SIZE * 3][LAYER1_SIZE * 3];
// #pragma HLS bind_storage variable=local_cost type=RAM_T2P impl=bram
    static label_range local_label[LAYER1_SIZE * 3][LAYER1_SIZE * 3][3];
    static disp_range local_disp[LAYER1_SIZE * 3][LAYER1_SIZE * 3];
#pragma HLS array_partition variable=local_cost type=complete dim=2
#pragma HLS array_partition variable=local_label type=complete dim=2
#pragma HLS array_partition variable=local_label type=complete dim=3
#pragma HLS array_partition variable=local_disp type=complete dim=2
    static ap_uint<DISP_DDR_BIT> local_trans_cost[LAYER1_SIZE][LAYER1_SIZE]; 
    static label_range local_trans_label[LAYER1_SIZE][LAYER1_SIZE][3];
    static disp_range local_trans_disp[LAYER1_SIZE][LAYER1_SIZE];
#pragma HLS array_partition variable=local_trans_cost type=complete dim=2
#pragma HLS array_partition variable=local_trans_label type=complete dim=2
#pragma HLS array_partition variable=local_trans_label type=complete dim=3
#pragma HLS array_partition variable=local_trans_disp type=complete dim=2
    static ap_uint<DISP_DDR_BIT> local_trans_cost_tmp[LAYER1_SIZE * 2][LAYER1_SIZE]; 
    static label_range local_trans_label_tmp[LAYER1_SIZE * 2][LAYER1_SIZE][3];
    static disp_range local_trans_disp_tmp[LAYER1_SIZE * 2][LAYER1_SIZE];
#pragma HLS array_partition variable=local_trans_cost type=complete dim=2
#pragma HLS array_partition variable=local_trans_label type=complete dim=2
#pragma HLS array_partition variable=local_trans_label type=complete dim=3
#pragma HLS array_partition variable=local_trans_disp type=complete dim=2
    /* load to local buffer */
    layer1UramLoad(l1_iter, uram_x0, uram_x1, uram_x2, tran_unit_region_x, tran_unit_region_y, tran_unit_region_w, tran_unit_region_h, uram_cost, uram_label, uram_disp, 
        local_cost, local_label, local_disp, local_trans_cost, local_trans_label, local_trans_disp, local_trans_cost_tmp, local_trans_label_tmp, local_trans_disp_tmp);
    /* transmit data blocks to layer2 */
#if LAYER2_ENABLE == 1
    layer1TransWrapper(l1_iter, cost_blk_out, label_blk_out, disp_blk_out, local_trans_cost, local_trans_label, local_trans_disp);
#else
    layer1TransWrapper(l1_iter, tran_unit_region_x, tran_unit_region_y, tran_unit_region_w, tran_unit_region_h, ddr_cost, ddr_label, ddr_disp, local_trans_cost, local_trans_label, local_trans_disp);
#endif
    /* calculation */
    layer1CalcWrapper(l1_iter, unary_cost_y0, unary_cost_y1, unary_cost_y2, Q, I, C, local_unary_cost, local_im0_gray, local_cost, local_label, local_disp);
    /* cache layer0 output */
    l0_iter += layer1InitInloop(l1_iter, l0_iter, uram_x3, cost_blk_in, label_blk_in, disp_blk_in, uram_cost, uram_label, uram_disp);
    /* store to global buffer & trans_tmp buffer */
    layer1UramStore(l1_iter, uram_x0, uram_x1, uram_x2, uram_cost, uram_label, uram_disp, local_cost, local_label, local_disp, local_trans_cost_tmp, local_trans_label_tmp, local_trans_disp_tmp);
    // }
}

#if HLS_LAYER1_ONLY == 0
void localExpLayer1(
#if PROPOSER_RANDOM_INIT == 1
    ap_uint<32> random_num[LAYER1_PROP_NUM][3], 
#endif // PROPOSER_RANDOM_INIT == 1
    unary_cost_t ddr_unary_cost[WIDTH * HEIGHT * DISP_DDR_MAX], uchar ddr_im0_gray[WIDTH * HEIGHT], 
    hls::stream_of_blocks<layer0_cost_blk> &cost_blk_in, hls::stream_of_blocks<layer0_label_blk> &label_blk_in, hls::stream_of_blocks<layer0_disp_blk> &disp_blk_in,
    hls::stream_of_blocks<layer0_cost_blk> &init_cost_blk_in, hls::stream_of_blocks<layer0_label_blk> &init_label_blk_in, hls::stream_of_blocks<layer0_disp_blk> &init_disp_blk_in,
#if LAYER2_ENABLE == 1
    hls::stream_of_blocks<layer1_cost_blk> &cost_blk_out, hls::stream_of_blocks<layer1_label_blk> &label_blk_out, hls::stream_of_blocks<layer1_disp_blk> &disp_blk_out) {
#else
    ap_uint<DISP_DDR_BIT> ddr_cost[WIDTH * HEIGHT], label_range ddr_label[WIDTH * HEIGHT * 3], disp_range ddr_disp[WIDTH * HEIGHT]) {
#endif
#else
void localExpLayer1(ap_uint<32> random_num[LAYER1_PROP_NUM][3], unary_cost_t ddr_unary_cost[WIDTH * HEIGHT * DISP_DDR_MAX], uchar ddr_im0_gray[WIDTH * HEIGHT], 
    hls::stream_of_blocks<layer0_cost_blk> &cost_blk_in, hls::stream_of_blocks<layer0_label_blk> &label_blk_in, hls::stream_of_blocks<layer0_disp_blk> &disp_blk_in,
#if LAYER2_ENABLE == 1
    hls::stream_of_blocks<layer1_cost_blk> &cost_blk_out, hls::stream_of_blocks<layer1_label_blk> &label_blk_out, hls::stream_of_blocks<layer1_disp_blk> &disp_blk_out) {}
#else
    float ddr_cost[WIDTH * HEIGHT], float ddr_label[WIDTH * HEIGHT * 3], float ddr_disp[WIDTH * HEIGHT]) {}
#endif
void localExpLayer1Hls(ap_uint<32> random_num[LAYER1_PROP_NUM][3], unary_cost_t ddr_unary_cost[WIDTH * HEIGHT * DISP_DDR_MAX], uchar ddr_im0_gray[WIDTH * HEIGHT], 
    float cost_blk_in[LAYER0_SIZE * LAYER0_SIZE], float label_blk_in[LAYER0_SIZE * LAYER0_SIZE * 3], float disp_blk_in[LAYER0_SIZE * LAYER0_SIZE],
#if LAYER2_ENABLE == 1
    float cost_blk_out[LAYER1_SIZE * LAYER1_SIZE], float label_blk_out[LAYER1_SIZE * LAYER1_SIZE * 3], float disp_blk_out[LAYER1_SIZE * LAYER1_SIZE]) {
#else
    float ddr_cost[WIDTH * HEIGHT], float ddr_label[WIDTH * HEIGHT * 3], float ddr_disp[WIDTH * HEIGHT]) {
#endif
#endif
// #pragma HLS interface mode=m_axi bundle=BUS_LAYER1 port=ddr_unary_cost offset=slave
// #pragma HLS interface mode=m_axi bundle=BUS_LAYER1 port=ddr_im0_gray offset=slave
#if LAYER2_ENABLE == 0
// #pragma HLS interface mode=m_axi bundle=BUS_OUT port=ddr_cost offset=slave
// #pragma HLS interface mode=m_axi bundle=BUS_OUT port=ddr_label offset=slave
// #pragma HLS interface mode=m_axi bundle=BUS_OUT port=ddr_disp offset=slave
#endif
    // 
    static ap_uint<DISP_DDR_BIT> uram_cost[HEIGHT][4][LAYER1_SIZE]; // row partition
#pragma HLS array_partition variable=uram_cost type=complete dim=2
#pragma HLS array_partition variable=uram_cost type=complete dim=3
    static label_range uram_label[HEIGHT][4][LAYER1_SIZE][3]; 
#pragma HLS array_partition variable=uram_label type=complete dim=2
#pragma HLS array_partition variable=uram_label type=complete dim=3
#pragma HLS array_partition variable=uram_label type=complete dim=4
    static disp_range uram_disp[HEIGHT][4][LAYER1_SIZE]; 
#pragma HLS array_partition variable=uram_disp type=complete dim=2
#pragma HLS array_partition variable=uram_disp type=complete dim=3
// #pragma HLS bind_storage variable=uram_disp type=RAM_1P impl=uram
    static unary_cost_t local_unary_cost[LAYER1_SIZE][4][LAYER1_SIZE * 3][DISP_DDR_MAX];
#pragma HLS array_partition variable=local_unary_cost type=complete dim=2
#pragma HLS array_partition variable=local_unary_cost type=complete dim=3
//DO_PRAGMA(HLS array_reshape variable=local_unary_cost type=cyclic factor=DISP_DDR_NUM dim=4) 
    static uchar local_im0_gray[LAYER1_SIZE][4][LAYER1_SIZE * 3];
#pragma HLS array_partition variable=local_im0_gray type=complete dim=2
#pragma HLS array_partition variable=local_im0_gray type=complete dim=3
    // 
    static bool random_generator_init = false;
    static ap_uint<32> Q[LAYER1_PROP_NUM][4096]; static ap_uint<32> I[LAYER1_PROP_NUM]; static ap_uint<32> C[LAYER1_PROP_NUM];
#pragma HLS array_partition variable=Q type=complete dim=1
#pragma HLS array_partition variable=I type=complete dim=0
#pragma HLS array_partition variable=C type=complete dim=0
#if PROPOSER_RANDOM_INIT == 0
    const ap_uint<32> random_num[2][3] = {
        {2340255427, 3638918503, 1819583497},
        {2678185683, 2774094101, 1650906866}
    };
#endif // PROPOSER_RANDOM_INIT == 0
#pragma HLS array_partition variable=random_num type=complete dim=0
    if (random_generator_init == false) {
        for (int i = 0; i < LAYER1_PROP_NUM; i++) {
#pragma HLS unroll
            I[i] = random_num[i][0];
            C[i] = random_num[i][1];
            initRandGen(random_num[i][2], Q[i], I[i], C[i]);
        }
        random_generator_init = true;
    }
    //
    const int l0_hor_region_num = LAYER0_HOR_REGION_NUM;
    const int l0_ver_region_num = LAYER0_VER_REGION_NUM;
    const int l1_hor_region_num = LAYER1_HOR_REGION_NUM;
    const int l1_ver_region_num = LAYER1_VER_REGION_NUM;
    // pre cache 4 unit region (2 columns)
    layer1OutloopCache(local_unary_cost, ddr_unary_cost, local_im0_gray, ddr_im0_gray);
    // 
    // int l0_iter = 0;
    int l0_iter = layer1InitOutloop(0, init_cost_blk_in, init_label_blk_in, init_disp_blk_in, uram_cost, uram_label, uram_disp);
    // 
    for (int l1_iter = 0; l1_iter < (l1_ver_region_num * l1_hor_region_num + (l1_ver_region_num + 2)); l1_iter++) {
// #pragma HLS dependence variable=uram_cost type=intra false
// #pragma HLS dependence variable=uram_disp type=intra false
// #pragma HLS dependence variable=uram_label type=intra false
#pragma HLS dependence variable=local_im0_gray type=intra false
#pragma HLS dependence variable=local_unary_cost type=intra false
        // 
        int unary_cost_y0 = ((l1_iter - 1) % 4) < 0 ? 3 : ((l1_iter - 1) % 4); // calc pre
        int unary_cost_y1 = (l1_iter + 0) % 4; // calc curr
        int unary_cost_y2 = (l1_iter + 1) % 4; // calc post
        int unary_cost_y3 = (l1_iter + 2) % 4; // cache
        assert((unary_cost_y0 != unary_cost_y1) && (unary_cost_y0 != unary_cost_y2) && (unary_cost_y0 != unary_cost_y3) && 
               (unary_cost_y1 != unary_cost_y2) && (unary_cost_y1 != unary_cost_y3) && (unary_cost_y2 != unary_cost_y3));
        int uram_x0 = (l1_iter < l1_ver_region_num) ? 0 : (l1_iter / l1_ver_region_num - 1) % 4;
        int uram_x1 = (l1_iter < l1_ver_region_num) ? 1 : (l1_iter / l1_ver_region_num + 0) % 4;
        int uram_x2 = (l1_iter < l1_ver_region_num) ? 3 : (l1_iter / l1_ver_region_num + 1) % 4;
        int uram_x3 = (l1_iter < l1_ver_region_num) ? 2 : (l1_iter / l1_ver_region_num + 2) % 4;
        assert((uram_x0 != uram_x1) && (uram_x0 != uram_x2) && (uram_x0 != uram_x3) && 
               (uram_x1 != uram_x2) && (uram_x1 != uram_x3) && (uram_x2 != uram_x3));
        // 
        // layer1InitWrapper(l0_iter, l1_iter, uram_y3, cost_blk_in, label_blk_in, disp_blk_in, uram_cost, uram_label, uram_disp);
        // 
        layer1CalcTransWrapper(l0_iter, l1_iter, unary_cost_y0, unary_cost_y1, unary_cost_y2, uram_x0, uram_x1, uram_x2, uram_x3, Q, I, C, local_unary_cost, local_im0_gray, 
            uram_cost, uram_label, uram_disp, cost_blk_in, label_blk_in, disp_blk_in, ddr_cost, ddr_label, ddr_disp);
        layer1CacheWrapper(l1_iter, unary_cost_y3, local_unary_cost, ddr_unary_cost, local_im0_gray, ddr_im0_gray);
        // 
// #if LAYER2_ENABLE == 1
//         layer1CalcTransWrapper(l1_iter, unary_cost_y0, unary_cost_y1, unary_cost_y2, Q, I, C, local_unary_cost, local_im0_gray, uram_cost, uram_label, uram_disp, cost_blk_out, label_blk_out, disp_blk_out);
// #else
//         layer1CalcTransWrapper(l0_iter, l1_iter, unary_cost_y0, unary_cost_y1, unary_cost_y2, uram_x0, uram_x1, uram_x2, uram_x3, Q, I, C, local_unary_cost, local_im0_gray, 
//             uram_cost, uram_label, uram_disp, cost_blk_in, label_blk_in, disp_blk_in, /* init_cost_blk_in, init_label_blk_in, init_disp_blk_in, */ ddr_cost, ddr_label, ddr_disp);
// #endif
#if PRINT_STATUS == 1 && !defined(__SYNTHESIS__)
        printf("\rlayer1 process: %4d / %d", l1_iter, (l1_ver_region_num * l1_hor_region_num + (l1_ver_region_num + 2)) - 1);  
        fflush(stdout);
    }
    printf("\n");
#else
    }
#endif
}

#if 0
void layer2Calc(int iter, int ver_region_num, int hor_region_num, int unary_cost_x0, int unary_cost_x1, int unary_cost_x2,
    ap_uint<32>* Q, ap_uint<32>& I, ap_uint<32>& C,
    float local_unary_cost[4][LAYER2_SIZE][LAYER2_SIZE * 3][DISP_MAX], uchar local_im0_gray[4][LAYER2_SIZE][LAYER2_SIZE * 3], 
    float uram_cost[HEIGHT][LAYER2_SIZE * 4], float uram_label[HEIGHT][LAYER2_SIZE * 4][3], float uram_disp[HEIGHT][LAYER2_SIZE * 4]) {
    //
    float local_cost[LAYER2_SIZE * 3][LAYER2_SIZE * 3];
    float local_label[LAYER2_SIZE * 3][LAYER2_SIZE * 3][3];
    float local_disp[LAYER2_SIZE * 3][LAYER2_SIZE * 3];
    // 
    float plane_a, plane_b, plane_c;
    //
    int ver_region_i = iter % ver_region_num;
    int hor_region_i = iter / ver_region_num;
    int unit_region_x, unit_region_y, unit_region_w, unit_region_h;
    int expn_region_x, expn_region_y, expn_region_w, expn_region_h;
    genRegionInfo(WIDTH, HEIGHT, hor_region_i, ver_region_i, LAYER2_SIZE, unit_region_x, unit_region_y, unit_region_w, unit_region_h, 
        expn_region_x, expn_region_y, expn_region_w, expn_region_h);
    // address mapping table for uram
    int uram_x_addr_id = hor_region_i % 4;
    // uram -> local ram
    LOOP_LOAD:
    for (int y = 0; y < LAYER2_SIZE * 3; y++) {
#pragma HLS pipeline II=1
        if ((expn_region_y + y) >= HEIGHT)   
            continue;
        if (hor_region_i == 0) {
            L2_GLOBAL_LOCAL_VSCAN_ADDR1
        }
        else {
            if (uram_x_addr_id == 0) {
                L2_GLOBAL_LOCAL_VSCAN_ADDR0
            }
            else if (uram_x_addr_id == 1) {
                L2_GLOBAL_LOCAL_VSCAN_ADDR1
            }
            else if (uram_x_addr_id == 2) {
                L2_GLOBAL_LOCAL_VSCAN_ADDR2
            }
            else if (uram_x_addr_id == 3) {
                L2_GLOBAL_LOCAL_VSCAN_ADDR3
            }
        }
    }
    // process local ram
#if LAYER1_ENABLE
    LOOP_PROPOSAL:
    for (int inner_iter = 0; inner_iter < 8; inner_iter++) {
#pragma HLS pipeline off
        if (inner_iter == 0)
            layerExpnProposal<LAYER2_SIZE>(ver_region_i, Q, I, C, expn_region_x, expn_region_y, unit_region_x, unit_region_y, unit_region_w, unit_region_h, 
                local_label, plane_a, plane_b, plane_c);
        else
            layerRandProposal<LAYER2_SIZE>(ver_region_i, Q, I, C, 0, inner_iter -1, expn_region_x, expn_region_y, unit_region_x, unit_region_y, unit_region_w, unit_region_h, 
                local_label, plane_a, plane_b, plane_c);
        costLabelUpdate<LAYER2_SIZE>(unary_cost_x0, unary_cost_x1, unary_cost_x2, expn_region_x, expn_region_y, expn_region_w, expn_region_h, unit_region_y, 
            plane_a, plane_b, plane_c, local_unary_cost, local_im0_gray, local_cost, local_label, local_disp, true, false);
    }
#endif
    // local ram -> uram
    LOOP_STORE:
    for (int y = 0; y < LAYER2_SIZE * 3; y++) {
#pragma HLS pipeline II=1
        if ((expn_region_y + y) >= HEIGHT)
            continue;
        if (hor_region_i == 0) {
            L2_LOCAL_GLOBAL_VSCAN_ADDR1
        }
        else {
            if (uram_x_addr_id == 0) {
                L2_LOCAL_GLOBAL_VSCAN_ADDR0
            }
            else if (uram_x_addr_id == 1) {
                L2_LOCAL_GLOBAL_VSCAN_ADDR1
            }
            else if (uram_x_addr_id == 2) {
                L2_LOCAL_GLOBAL_VSCAN_ADDR2
            }
            else if (uram_x_addr_id == 3) {
                L2_LOCAL_GLOBAL_VSCAN_ADDR3
            }
        }
    }
}

void localExpLayer2(float ddr_unary_cost[WIDTH * HEIGHT * DISP_MAX], uchar ddr_im0_gray[WIDTH * HEIGHT], 
    hls::stream_of_blocks<layer1_cost_blk> &cost_blk_in, hls::stream_of_blocks<layer1_label_blk> &label_blk_in, hls::stream_of_blocks<layer1_disp_blk> &disp_blk_in,
    float ddr_cost[WIDTH * HEIGHT], float ddr_label[WIDTH * HEIGHT * 3], float ddr_disp[WIDTH * HEIGHT]) {
    /* intermediate cache */
    static float uram_cost[HEIGHT][LAYER2_SIZE * 4]; 
    static float uram_label[HEIGHT][LAYER2_SIZE * 4][3]; 
    static float uram_disp[HEIGHT][LAYER2_SIZE * 4]; 
    /* local cache */
    static uchar local_im0_gray[4][LAYER2_SIZE][LAYER2_SIZE * 3];
    static float local_unary_cost[4][LAYER2_SIZE][LAYER2_SIZE * 3][DISP_MAX];
    /* init random generators */
    static bool random_generator_init = false;
    static ap_uint<32> Q[4096];
    static ap_uint<32> I = 1731;
    static ap_uint<32> C = 793451;
    if (random_generator_init == false) {
        initRandGen(0, Q, I, C); 
        random_generator_init = true;
    }
    /* region parameters */
    const int l1_hor_region_num = LAYER1_HOR_REGION_NUM;
    const int l1_ver_region_num = LAYER1_VER_REGION_NUM;
    const int l2_hor_region_num = LAYER2_HOR_REGION_NUM;
    const int l2_ver_region_num = LAYER2_VER_REGION_NUM;
    /* pre cache 4 unit region (2 columns) */
    layerOutloopCache<LAYER2_SIZE>(local_unary_cost, ddr_unary_cost, local_im0_gray, ddr_im0_gray);
    /* loop expansion */
    for (int l1_iter = 0, l2_iter = -1; l2_iter < (l2_ver_region_num * l2_hor_region_num + (l2_ver_region_num + 1)); l2_iter++) {
        /* local cache buffer index */
        int unary_cost_y0 = (l2_iter - 1) % 4; // calc pre
        int unary_cost_y1 = (l2_iter + 0) % 4; // calc curr
        int unary_cost_y2 = (l2_iter + 1) % 4; // calc post
        int unary_cost_y3 = (l2_iter + 2) % 4; // cache
        /* init using data blocks from layer1 */
        if (l2_iter < (l2_ver_region_num * l2_hor_region_num - 2 * l2_ver_region_num)) {
            layerInit<LAYER1_SIZE, LAYER2_SIZE, layer1_cost_blk, layer1_label_blk, layer1_disp_blk>(l1_iter, l2_iter, l1_hor_region_num, l1_ver_region_num, l2_hor_region_num, l2_ver_region_num, cost_blk_in, label_blk_in, disp_blk_in, uram_cost, uram_label, uram_disp);
        }
        /* cache unary cost & gray image from DDR */
        if ((l2_iter >= 0) && (l2_iter < (l2_ver_region_num * l2_hor_region_num - 2))) {
            layerInloopCache<LAYER2_SIZE>(l2_iter, l2_ver_region_num, l2_hor_region_num, unary_cost_y3, local_unary_cost, ddr_unary_cost, local_im0_gray, ddr_im0_gray);
        }
        /* calculation */
        if ((l2_iter >= 0) && (l2_iter < l2_ver_region_num * l2_hor_region_num)) {
            layer2Calc(l2_iter, l2_ver_region_num, l2_hor_region_num, unary_cost_y0, unary_cost_y1, unary_cost_y2, Q, I, C, local_unary_cost, local_im0_gray, uram_cost, uram_label, uram_disp);
        }
        /* transmit data blocks to DDR */
        if (l2_iter >= (l2_ver_region_num + 1)){
            layerTransDDR<LAYER2_SIZE * 4, 4, LAYER2_SIZE>(l2_iter, l2_ver_region_num, uram_cost, uram_label, uram_disp, ddr_cost, ddr_label, ddr_disp);
        }
    }
    
}
#endif
#endif // COST_LOAD_MODE == 1 && SCAN_DIRECTION == 1

#if COST_LOAD_MODE == 1
void localExpLayers(
#if PROPOSER_RANDOM_INIT == 1 
   ap_uint<32> l0_random_num[LAYER0_PROP_NUM][3], ap_uint<32> l1_random_num[LAYER1_PROP_NUM][3], 
#endif // PROPOSER_RANDOM_INIT == 1
    unary_cost_t0 ddr_unary_cost_layer0[WIDTH * HEIGHT * DISP_DDR_MAX0], 
    uchar ddr_im0_gray_layer0[WIDTH * HEIGHT], 
   unary_cost_t0 ddr_unary_cost_layer1[WIDTH * HEIGHT * DISP_DDR_MAX0], uchar ddr_im0_gray_layer1[WIDTH * HEIGHT], 
#if LAYER2_ENABLE == 1
    unary_cost_t ddr_unary_cost_layer2[WIDTH * HEIGHT * DISP_DDR_MAX], uchar ddr_im0_gray_layer2[WIDTH * HEIGHT], 
#endif
    ap_uint<DISP_DDR_BIT> ddr_cost[WIDTH * HEIGHT], label_range ddr_label[WIDTH * HEIGHT * 3], 
    disp_range ddr_disp[WIDTH * HEIGHT]) {
        /*
#elif COST_LOAD_MODE == 1 && SCAN_DIRECTION == 1
void localExpLayers(float ddr_unary_cost_layer0[WIDTH * HEIGHT * DISP_MAX], uchar ddr_im0_gray_layer0[WIDTH * HEIGHT], 
    float ddr_unary_cost_layer1[WIDTH * HEIGHT * DISP_MAX], uchar ddr_im0_gray_layer1[WIDTH * HEIGHT], 
    float ddr_unary_cost_layer2[WIDTH * HEIGHT * DISP_MAX], uchar ddr_im0_gray_layer2[WIDTH * HEIGHT], 
    float ddr_cost[WIDTH * HEIGHT], float ddr_label[WIDTH * HEIGHT * 3], float ddr_disp[WIDTH * HEIGHT]) {
        */
#elif COST_CALC_MODE == 1
void localExpLayers(uchar ddr_im0_gray_layer0[WIDTH * HEIGHT], uchar ddr_im1_gray_layer0[WIDTH * HEIGHT], uint64_t ddr_im0_census_layer0[WIDTH * HEIGHT], uint64_t ddr_im1_census_layer0[WIDTH * HEIGHT], 
    uchar ddr_im0_gray_layer1[WIDTH * HEIGHT], uchar ddr_im1_gray_layer1[WIDTH * HEIGHT], uint64_t ddr_im0_census_layer1[WIDTH * HEIGHT], uint64_t ddr_im1_census_layer1[WIDTH * HEIGHT],
    uchar ddr_im0_gray_layer2[WIDTH * HEIGHT], uchar ddr_im1_gray_layer2[WIDTH * HEIGHT], uint64_t ddr_im0_census_layer2[WIDTH * HEIGHT], uint64_t ddr_im1_census_layer2[WIDTH * HEIGHT],
    float ddr_cost[WIDTH * HEIGHT], float ddr_label[WIDTH * HEIGHT * 3], float ddr_disp[WIDTH * HEIGHT]) {
#endif
#pragma HLS INTERFACE m_axi port=ddr_unary_cost_layer0 offset=slave bundle=ddr_bus max_read_burst_length=256 
#pragma HLS INTERFACE s_axilite port=ddr_unary_cost_layer0 bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
#if LAYER2_ENABLE == 1
#pragma HLS interface mode=m_axi bundle=BUS_LAYER2 port=ddr_unary_cost_layer2 offset=slave
#pragma HLS interface mode=m_axi bundle=BUS_LAYER2 port=ddr_im0_gray_layer2 offset=slave
#endif
/*
#pragma HLS interface mode=m_axi bundle=BUS_OUT port=ddr_cost 
#pragma HLS interface mode=m_axi bundle=BUS_OUT port=ddr_label 
#pragma HLS interface mode=m_axi bundle=BUS_OUT port=ddr_disp 

#pragma HLS dataflow
    hls::stream_of_blocks<layer0_cost_blk> cost_blk;
    hls::stream_of_blocks<layer0_cost_blk> init_cost_blk;
    hls::stream_of_blocks<layer0_label_blk> label_blk;
    hls::stream_of_blocks<layer0_label_blk> init_label_blk;
    hls::stream_of_blocks<layer0_disp_blk> disp_blk;
    hls::stream_of_blocks<layer0_disp_blk> init_disp_blk;
*/
ap_uint<32> l00_random_num[3][3];
l00_random_num[0][0] = 3039794975;
l00_random_num[0][1] = 1965046491;
l00_random_num[0][2] = 2706562054;
l00_random_num[1][0] = 3664584971;
l00_random_num[1][1] = 915655065;
l00_random_num[1][2] = 2659113310;
l00_random_num[2][0] = 2950877296;
l00_random_num[2][1] = 2504661326;
l00_random_num[2][2] = 1805160153;
#if COST_LOAD_MODE == 1 
        localExpLayer0(
#if PROPOSER_RANDOM_INIT == 1
        l0_random_num, 
#endif // #if PROPOSER_RANDOM_INIT == 1
        ddr_unary_cost_layer0, 
#if SCAN_DIRECTION == 0 
        /*ddr_im0_gray_layer0, */ddr_cost, ddr_label, 
#elif SCAN_DIRECTION == 1
        /*ddr_im0_gray_layer0, ddr_cost, ddr_label,*/
#endif        
        ddr_disp);
#elif COST_CALC_MODE == 1
    localExpLayer0(ddr_im0_gray_layer0, ddr_im1_gray_layer0, ddr_im0_census_layer0, ddr_im1_census_layer0, cost_blk0, label_blk0, disp_blk0);
    localExpLayer1(ddr_im0_gray_layer1, ddr_im1_gray_layer1, ddr_im0_census_layer1, ddr_im1_census_layer1, cost_blk0, label_blk0, disp_blk0, cost_blk1, label_blk1, disp_blk1);
    localExpLayer2(ddr_im0_gray_layer2, ddr_im1_gray_layer2, ddr_im0_census_layer2, ddr_im1_census_layer2, cost_blk1, label_blk1, disp_blk1, ddr_cost, ddr_label, ddr_disp);
#endif
}

