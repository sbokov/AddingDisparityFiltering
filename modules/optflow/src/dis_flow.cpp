/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "opencv2/core/hal/intrin.hpp"
#include "opencv2/imgproc.hpp"
#include "precomp.hpp"
using namespace std;
#define EPS 0.001F
#define INF 1E+10F

namespace cv
{
namespace optflow
{

class DISOpticalFlowImpl : public DISOpticalFlow
{
  public:
    DISOpticalFlowImpl();

    void calc(InputArray I0, InputArray I1, InputOutputArray flow);
    void collectGarbage();

  protected: // algorithm parameters
    int finest_scale, coarsest_scale;
    int patch_size;
    int patch_stride;
    int grad_descent_iter;
    int variational_refinement_iter;
    int border_size;

  public: // getters and setters
    int getFinestScale() const { return finest_scale; }
    void setFinestScale(int val) { finest_scale = val; }
    int getPatchSize() const { return patch_size; }
    void setPatchSize(int val) { patch_size = val; }
    int getPatchStride() const { return patch_stride; }
    void setPatchStride(int val) { patch_stride = val; }
    int getGradientDescentIterations() const { return grad_descent_iter; }
    void setGradientDescentIterations(int val) { grad_descent_iter = val; }
    int getVariationalRefinementIterations() const { return variational_refinement_iter; }
    void setVariationalRefinementIterations(int val) { variational_refinement_iter = val; }

  protected:                     // internal buffers
    vector<Mat_<uchar>> I0s;     // gaussian pyramid for the current frame
    vector<Mat_<uchar>> I1s;     // gaussian pyramid for the next frame
    vector<Mat_<uchar>> I1s_ext; // I1s with borders

    vector<Mat_<short>> I0xs; // gaussian pyramid for the x gradient of the current frame
    vector<Mat_<short>> I0ys; // gaussian pyramid for the y gradient of the current frame

    vector<Mat_<float>> Ux; // x component of the flow vectors
    vector<Mat_<float>> Uy; // y component of the flow vectors

    Mat_<Vec2f> U; // buffers for the merged flow

    Mat_<float> Sx; // x component of the sparse flow vectors (before densification)
    Mat_<float> Sy; // y component of the sparse flow vectors (before densification)

    // structure tensor components and auxiliary buffers:
    Mat_<float> I0xx_buf; // sum of squares of x gradient values
    Mat_<float> I0yy_buf; // sum of squares of y gradient values
    Mat_<float> I0xy_buf; // sum of x and y gradient products

    Mat_<float> I0xx_buf_aux; // for computing sums using the summed area table
    Mat_<float> I0yy_buf_aux;
    Mat_<float> I0xy_buf_aux;
    ////////////////////////////////////////////////////////////

    vector<Ptr<VariationalRefinement>> variational_refinement_processors;

  private: // private methods
    void prepareBuffers(Mat &I0, Mat &I1);
    void precomputeStructureTensor(Mat &dst_I0xx, Mat &dst_I0yy, Mat &dst_I0xy, Mat &I0x, Mat &I0y);

    struct PatchGradientDescent_ParBody : public ParallelLoopBody
    {
        DISOpticalFlowImpl *dis;
        Mat *Sx, *Sy, *Ux, *Uy, *I0, *I1, *I0x, *I0y;
        int nstripes, stripe_sz;
        int hs;

        PatchGradientDescent_ParBody(DISOpticalFlowImpl &_dis, int _nstripes, int _h, Mat &dst_Sx, Mat &dst_Sy,
                                     Mat &src_Ux, Mat &src_Uy, Mat &_I0, Mat &_I1, Mat &_I0x, Mat &_I0y);
        void operator()(const Range &range) const;
    };

    struct Densification_ParBody : public ParallelLoopBody
    {
        DISOpticalFlowImpl *dis;
        Mat *Ux, *Uy, *Sx, *Sy, *I0, *I1;
        int nstripes, stripe_sz;
        int h;

        Densification_ParBody(DISOpticalFlowImpl &_dis, int _nstripes, int _h, Mat &dst_Ux, Mat &dst_Uy, Mat &src_Sx,
                              Mat &src_Sy, Mat &_I0, Mat &_I1);
        void operator()(const Range &range) const;
    };
};

DISOpticalFlowImpl::DISOpticalFlowImpl()
{
    finest_scale = 2;
    patch_size = 8;
    patch_stride = 4;
    grad_descent_iter = 16;
    variational_refinement_iter = 5;
    border_size = 16;

    int max_possible_scales = 10;
    for (int i = 0; i < max_possible_scales; i++)
        variational_refinement_processors.push_back(createVariationalFlowRefinement());
}

void DISOpticalFlowImpl::prepareBuffers(Mat &I0, Mat &I1)
{
    I0s.resize(coarsest_scale + 1);
    I1s.resize(coarsest_scale + 1);
    I1s_ext.resize(coarsest_scale + 1);
    I0xs.resize(coarsest_scale + 1);
    I0ys.resize(coarsest_scale + 1);
    Ux.resize(coarsest_scale + 1);
    Uy.resize(coarsest_scale + 1);

    int fraction = 1;
    int cur_rows = 0, cur_cols = 0;

    for (int i = 0; i <= coarsest_scale; i++)
    {
        if (i == finest_scale)
        {
            cur_rows = I0.rows / fraction;
            cur_cols = I0.cols / fraction;
            I0s[i].create(cur_rows, cur_cols);
            resize(I0, I0s[i], I0s[i].size(), 0.0, 0.0, INTER_AREA);
            I1s[i].create(cur_rows, cur_cols);
            resize(I1, I1s[i], I1s[i].size(), 0.0, 0.0, INTER_AREA);

            Sx.create(cur_rows / patch_stride, cur_cols / patch_stride);
            Sy.create(cur_rows / patch_stride, cur_cols / patch_stride);
            I0xx_buf.create(cur_rows / patch_stride, cur_cols / patch_stride);
            I0yy_buf.create(cur_rows / patch_stride, cur_cols / patch_stride);
            I0xy_buf.create(cur_rows / patch_stride, cur_cols / patch_stride);
            I0xx_buf_aux.create(cur_rows, cur_cols / patch_stride);
            I0yy_buf_aux.create(cur_rows, cur_cols / patch_stride);
            I0xy_buf_aux.create(cur_rows, cur_cols / patch_stride);
            U.create(cur_rows, cur_cols);
        }
        else if (i > finest_scale)
        {
            cur_rows = I0s[i - 1].rows / 2;
            cur_cols = I0s[i - 1].cols / 2;
            I0s[i].create(cur_rows, cur_cols);
            resize(I0s[i - 1], I0s[i], I0s[i].size(), 0.0, 0.0, INTER_AREA);
            I1s[i].create(cur_rows, cur_cols);
            resize(I1s[i - 1], I1s[i], I1s[i].size(), 0.0, 0.0, INTER_AREA);
        }

        fraction *= 2;

        if (i >= finest_scale)
        {
            I1s_ext[i].create(cur_rows + 2 * border_size, cur_cols + 2 * border_size);
            copyMakeBorder(I1s[i], I1s_ext[i], border_size, border_size, border_size, border_size, BORDER_REPLICATE);
            I0xs[i].create(cur_rows, cur_cols);
            I0ys[i].create(cur_rows, cur_cols);
            spatialGradient(I0s[i], I0xs[i], I0ys[i]);
            Ux[i].create(cur_rows, cur_cols);
            Uy[i].create(cur_rows, cur_cols);
            variational_refinement_processors[i]->setAlpha(20.0f);
            variational_refinement_processors[i]->setDelta(5.0f);
            variational_refinement_processors[i]->setGamma(10.0f);
            variational_refinement_processors[i]->setSorIterations(5);
            variational_refinement_processors[i]->setFixedPointIterations(variational_refinement_iter);
        }
    }
}

void DISOpticalFlowImpl::precomputeStructureTensor(Mat &dst_I0xx, Mat &dst_I0yy, Mat &dst_I0xy, Mat &I0x, Mat &I0y)
{
    short *I0x_ptr = I0x.ptr<short>();
    short *I0y_ptr = I0y.ptr<short>();

    float *I0xx_ptr = dst_I0xx.ptr<float>();
    float *I0yy_ptr = dst_I0yy.ptr<float>();
    float *I0xy_ptr = dst_I0xy.ptr<float>();

    float *I0xx_aux_ptr = I0xx_buf_aux.ptr<float>();
    float *I0yy_aux_ptr = I0yy_buf_aux.ptr<float>();
    float *I0xy_aux_ptr = I0xy_buf_aux.ptr<float>();

    int w = I0x.cols;
    int h = I0x.rows;
    // width of the sparse OF fields:
    int ws = 1 + (w - patch_size) / patch_stride;

    // separable box filter for computing patch sums on a sparse
    // grid (determined by patch_stride)
    for (int i = 0; i < h; i++)
    {
        float sum_xx = 0.0f, sum_yy = 0.0f, sum_xy = 0.0f;
        short *x_row = I0x_ptr + i * w, *y_row = I0y_ptr + i * w;
        for (int j = 0; j < patch_size; j++)
        {
            sum_xx += x_row[j] * x_row[j];
            sum_yy += y_row[j] * y_row[j];
            sum_xy += x_row[j] * y_row[j];
        }
        I0xx_aux_ptr[i * ws] = sum_xx;
        I0yy_aux_ptr[i * ws] = sum_yy;
        I0xy_aux_ptr[i * ws] = sum_xy;
        int js = 1;
        for (int j = patch_size; j < w; j++)
        {
            sum_xx += (x_row[j] * x_row[j] - x_row[j - patch_size] * x_row[j - patch_size]);
            sum_yy += (y_row[j] * y_row[j] - y_row[j - patch_size] * y_row[j - patch_size]);
            sum_xy += (x_row[j] * y_row[j] - x_row[j - patch_size] * y_row[j - patch_size]);
            if ((j - patch_size + 1) % patch_stride == 0)
            {
                I0xx_aux_ptr[i * ws + js] = sum_xx;
                I0yy_aux_ptr[i * ws + js] = sum_yy;
                I0xy_aux_ptr[i * ws + js] = sum_xy;
                js++;
            }
        }
    }

    AutoBuffer<float> sum_xx_buf(ws), sum_yy_buf(ws), sum_xy_buf(ws);
    float *sum_xx = (float *)sum_xx_buf;
    float *sum_yy = (float *)sum_yy_buf;
    float *sum_xy = (float *)sum_xy_buf;
    for (int j = 0; j < ws; j++)
    {
        sum_xx[j] = 0.0f;
        sum_yy[j] = 0.0f;
        sum_xy[j] = 0.0f;
    }

    for (int i = 0; i < patch_size; i++)
        for (int j = 0; j < ws; j++)
        {
            sum_xx[j] += I0xx_aux_ptr[i * ws + j];
            sum_yy[j] += I0yy_aux_ptr[i * ws + j];
            sum_xy[j] += I0xy_aux_ptr[i * ws + j];
        }
    for (int j = 0; j < ws; j++)
    {
        I0xx_ptr[j] = sum_xx[j];
        I0yy_ptr[j] = sum_yy[j];
        I0xy_ptr[j] = sum_xy[j];
    }
    int is = 1;
    for (int i = patch_size; i < h; i++)
    {
        for (int j = 0; j < ws; j++)
        {
            sum_xx[j] += (I0xx_aux_ptr[i * ws + j] - I0xx_aux_ptr[(i - patch_size) * ws + j]);
            sum_yy[j] += (I0yy_aux_ptr[i * ws + j] - I0yy_aux_ptr[(i - patch_size) * ws + j]);
            sum_xy[j] += (I0xy_aux_ptr[i * ws + j] - I0xy_aux_ptr[(i - patch_size) * ws + j]);
        }
        if ((i - patch_size + 1) % patch_stride == 0)
        {
            for (int j = 0; j < ws; j++)
            {
                I0xx_ptr[is * ws + j] = sum_xx[j];
                I0yy_ptr[is * ws + j] = sum_yy[j];
                I0xy_ptr[is * ws + j] = sum_xy[j];
            }
            is++;
        }
    }
}

DISOpticalFlowImpl::PatchGradientDescent_ParBody::PatchGradientDescent_ParBody(DISOpticalFlowImpl &_dis, int _nstripes,
                                                                               int _hs, Mat &dst_Sx, Mat &dst_Sy,
                                                                               Mat &src_Ux, Mat &src_Uy, Mat &_I0,
                                                                               Mat &_I1, Mat &_I0x, Mat &_I0y)
    : dis(&_dis), nstripes(_nstripes), hs(_hs), Sx(&dst_Sx), Sy(&dst_Sy), Ux(&src_Ux), Uy(&src_Uy), I0(&_I0), I1(&_I1),
      I0x(&_I0x), I0y(&_I0y)
{
    stripe_sz = (int)ceil(hs / (double)nstripes);
}

// returns current SSD between patches
// I0_ptr, I1_ptr - already point to patches
// w00, w01, w10, w11 - bilinear interpolation weights
inline float processPatch(float &dst_dUx, float &dst_dUy, uchar *I0_ptr, uchar *I1_ptr, short *I0x_ptr,
                          short *I0y_ptr, int I0_stride, int I1_stride, float w00, float w01, float w10, float w11, int patch_sz)
{
    float SSD = 0.0f;
#ifdef CV_SIMD128
    if (patch_sz == 8)
    {
        // sum values:
        v_float32x4 Ux_vec = v_setall_f32(0);
        v_float32x4 Uy_vec = v_setall_f32(0);
        v_float32x4 SSD_vec = v_setall_f32(0);

        v_float32x4 w00v = v_setall_f32(w00);
        v_float32x4 w01v = v_setall_f32(w01);
        v_float32x4 w10v = v_setall_f32(w10);
        v_float32x4 w11v = v_setall_f32(w11);

        v_uint8x16 I0_row_16, I1_row_16, I1_row_shifted_16, I1_row_next_16, I1_row_next_shifted_16;
        v_uint16x8 I0_row_8, I1_row_8, I1_row_shifted_8, I1_row_next_8, I1_row_next_shifted_8, tmp;
        v_uint32x4 I0_row_4_left, I1_row_4_left, I1_row_shifted_4_left, I1_row_next_4_left, I1_row_next_shifted_4_left;
        v_uint32x4 I0_row_4_right, I1_row_4_right, I1_row_shifted_4_right, I1_row_next_4_right, I1_row_next_shifted_4_right;

        v_int16x8 I0x_row, I0y_row;
        v_int32x4 I0x_row_4_left, I0x_row_4_right, I0y_row_4_left, I0y_row_4_right;
        v_float32x4 I_diff;
        v_int32x4 Ux_mul_left, Ux_mul_right, Uy_mul_left, Uy_mul_right;
        v_int32x4 SSD_mul_left, SSD_mul_right;

        // preprocess first row of I1:
        I1_row_16 = v_load(I1_ptr);
        I1_row_shifted_16 = v_extract<1, v_uint8x16>(I1_row_16, I1_row_16);
        v_expand(I1_row_16, I1_row_8, tmp);
        v_expand(I1_row_shifted_16, I1_row_shifted_8, tmp);
        v_expand(I1_row_8, I1_row_4_left, I1_row_4_right);
        v_expand(I1_row_shifted_8, I1_row_shifted_4_left, I1_row_shifted_4_right);
        I1_ptr += I1_stride;

        for (int row = 0; row < 8; row++)
        {
            // load next row of I1:
            I1_row_next_16 = v_load(I1_ptr);
            // circular shift left by 1:
            I1_row_next_shifted_16 = v_extract<1, v_uint8x16>(I1_row_next_16, I1_row_next_16);
            // expand to 8 ushorts (we only need the first 8 values):
            v_expand(I1_row_next_16, I1_row_next_8, tmp);
            v_expand(I1_row_next_shifted_16, I1_row_next_shifted_8, tmp);
            v_expand(I1_row_next_8, I1_row_next_4_left, I1_row_next_4_right);
            v_expand(I1_row_next_shifted_8, I1_row_next_shifted_4_left, I1_row_next_shifted_4_right);

            // load current rows of I0, I0x, I0y:
            I0_row_16 = v_load(I0_ptr);
            v_expand(I0_row_16, I0_row_8, tmp);
            v_expand(I0_row_8, I0_row_4_left, I0_row_4_right);
            I0x_row = v_load(I0x_ptr);
            v_expand(I0x_row, I0x_row_4_left, I0x_row_4_right);
            I0y_row = v_load(I0y_ptr);
            v_expand(I0y_row, I0y_row_4_left, I0y_row_4_right);

            // difference of I0 row and bilinearly interpolated I1 row:
            I_diff = w00v * v_cvt_f32(v_reinterpret_as_s32(I1_row_4_left)) +
                w01v * v_cvt_f32(v_reinterpret_as_s32(I1_row_shifted_4_left)) +
                w10v * v_cvt_f32(v_reinterpret_as_s32(I1_row_next_4_left)) +
                w11v * v_cvt_f32(v_reinterpret_as_s32(I1_row_next_shifted_4_left)) -
                v_cvt_f32(v_reinterpret_as_s32(I0_row_4_left));
            Ux_vec += I_diff * v_cvt_f32(I0x_row_4_left);
            Uy_vec += I_diff * v_cvt_f32(I0y_row_4_left);
            SSD_vec += I_diff * I_diff;

            I_diff = w00v * v_cvt_f32(v_reinterpret_as_s32(I1_row_4_right)) +
                w01v * v_cvt_f32(v_reinterpret_as_s32(I1_row_shifted_4_right)) +
                w10v * v_cvt_f32(v_reinterpret_as_s32(I1_row_next_4_right)) +
                w11v * v_cvt_f32(v_reinterpret_as_s32(I1_row_next_shifted_4_right)) -
                v_cvt_f32(v_reinterpret_as_s32(I0_row_4_right));
            Ux_vec += I_diff * v_cvt_f32(I0x_row_4_right);
            Uy_vec += I_diff * v_cvt_f32(I0y_row_4_right);
            SSD_vec += I_diff * I_diff;

            I0_ptr += I0_stride;
            I0x_ptr += I0_stride;
            I0y_ptr += I0_stride;
            I1_ptr += I1_stride;

            I1_row_4_left = I1_row_next_4_left;
            I1_row_4_right = I1_row_next_4_right;
            I1_row_shifted_4_left = I1_row_next_shifted_4_left;
            I1_row_shifted_4_right = I1_row_next_shifted_4_right;
        }

        // final reduce operations:
        dst_dUx = v_reduce_sum(Ux_vec);
        dst_dUy = v_reduce_sum(Uy_vec);
        SSD = v_reduce_sum(SSD_vec);
    }
    else
    {
#endif
        dst_dUx = 0.0f;
        dst_dUy = 0.0f;
        float diff;
        for (int i = 0; i < patch_sz; i++)
            for (int j = 0; j < patch_sz; j++)
            {
                diff = w00 * I1_ptr[i * I1_stride + j] + w01 * I1_ptr[i * I1_stride + j + 1] +
                    w10 * I1_ptr[(i + 1) * I1_stride + j] + w11 * I1_ptr[(i + 1) * I1_stride + j + 1] -
                    I0_ptr[i * I0_stride + j];

                SSD += diff * diff;
                dst_dUx += diff * I0x_ptr[i * I0_stride + j];
                dst_dUy += diff * I0y_ptr[i * I0_stride + j];
            }
#ifdef CV_SIMD128
    }
#endif
    return SSD;
}

void DISOpticalFlowImpl::PatchGradientDescent_ParBody::operator()(const Range &range) const
{
    int start = min(range.start * stripe_sz, hs);
    int end = min(range.end * stripe_sz, hs);
    int w = I0->cols;
    int h = I0->rows;
    // width of the sparse OF field
    int ws = 1 + (w - dis->patch_size) / dis->patch_stride;
    int psz = dis->patch_size;
    int psz2 = psz / 2;
    int w_ext = w + 2 * dis->border_size;
    int bsz = dis->border_size;

    float *Ux_ptr = Ux->ptr<float>();
    float *Uy_ptr = Uy->ptr<float>();
    float *Sx_ptr = Sx->ptr<float>();
    float *Sy_ptr = Sy->ptr<float>();
    uchar *I0_ptr = I0->ptr<uchar>();
    uchar *I1_ptr = I1->ptr<uchar>();
    short *I0x_ptr = I0x->ptr<short>();
    short *I0y_ptr = I0y->ptr<short>();

    float *xx_ptr = dis->I0xx_buf.ptr<float>();
    float *yy_ptr = dis->I0yy_buf.ptr<float>();
    float *xy_ptr = dis->I0xy_buf.ptr<float>();

    int i = start * dis->patch_stride;
    float i_l = bsz - psz + 1.0f;
    float i_u = bsz + h - 1.0f;
    float j_l = bsz - psz + 1.0f;
    float j_u = bsz + w - 1.0f;
    for (int is = start; is < end; is++)
    {
        int j = 0;
        for (int js = 0; js < ws; js++)
        {
            // get initial approximation from the center of a patch:
            float cur_Ux = Ux_ptr[(i + psz2) * w + j + psz2];
            float cur_Uy = Uy_ptr[(i + psz2) * w + j + psz2];
            float detH = xx_ptr[is * ws + js] * yy_ptr[is * ws + js] - xy_ptr[is * ws + js] * xy_ptr[is * ws + js];
            if (abs(detH) < EPS)
                detH = EPS;
            float invH11 = yy_ptr[is * ws + js] / detH;
            float invH12 = -xy_ptr[is * ws + js] / detH;
            float invH22 = xx_ptr[is * ws + js] / detH;
            float prev_SSD = INF;
            for (int t = 0; t < dis->grad_descent_iter; t++)
            {
                float dUx, dUy;
                float i_I1 = min(max(i + cur_Uy + bsz, i_l), i_u);
                float j_I1 = min(max(j + cur_Ux + bsz, j_l), j_u);
                float w11 = (i_I1 - floor(i_I1)) * (j_I1 - floor(j_I1));
                float w10 = (i_I1 - floor(i_I1)) * (floor(j_I1) + 1 - j_I1);
                float w01 = (floor(i_I1) + 1 - i_I1) * (j_I1 - floor(j_I1));
                float w00 = (floor(i_I1) + 1 - i_I1) * (floor(j_I1) + 1 - j_I1);

                float SSD = processPatch(dUx, dUy, I0_ptr + i * w + j, I1_ptr + (int)i_I1 * w_ext + (int)j_I1,
                                         I0x_ptr + i * w + j, I0y_ptr + i * w + j, w, w_ext, w00, w01, w10, w11, psz);
                cur_Ux -= invH11 * dUx + invH12 * dUy;
                cur_Uy -= invH12 * dUx + invH22 * dUy;
                if (SSD > prev_SSD)
                    break;
                prev_SSD = SSD;
            }
            if (norm(Vec2f(cur_Ux - Ux_ptr[i * w + j], cur_Uy - Uy_ptr[i * w + j])) <= psz)
            {
                Sx_ptr[is * ws + js] = cur_Ux;
                Sy_ptr[is * ws + js] = cur_Uy;
            }
            else
            {
                Sx_ptr[is * ws + js] = Ux_ptr[(i + psz2) * w + j + psz2];
                Sy_ptr[is * ws + js] = Uy_ptr[(i + psz2) * w + j + psz2];
            }
            j += dis->patch_stride;
        }
        i += dis->patch_stride;
    }
}

DISOpticalFlowImpl::Densification_ParBody::Densification_ParBody(DISOpticalFlowImpl &_dis, int _nstripes, int _h,
                                                                 Mat &dst_Ux, Mat &dst_Uy, Mat &src_Sx, Mat &src_Sy,
                                                                 Mat &_I0, Mat &_I1)
    : dis(&_dis), nstripes(_nstripes), h(_h), Sx(&src_Sx), Sy(&src_Sy), Ux(&dst_Ux), Uy(&dst_Uy), I0(&_I0), I1(&_I1)
{
    stripe_sz = (int)ceil(h / (double)nstripes);
}

void DISOpticalFlowImpl::Densification_ParBody::operator()(const Range &range) const
{
    int start = min(range.start * stripe_sz, h);
    int end = min(range.end * stripe_sz, h);
    float *Ux_ptr = Ux->ptr<float>();
    float *Uy_ptr = Uy->ptr<float>();
    float *Sx_ptr = Sx->ptr<float>();
    float *Sy_ptr = Sy->ptr<float>();
    uchar *I0_ptr = I0->ptr<uchar>();
    uchar *I1_ptr = I1->ptr<uchar>();
    int w = I0->cols;
    // width of the sparse OF field:
    int ws = 1 + (w - dis->patch_size) / dis->patch_stride;
    int psz = dis->patch_size;
    int pstr = dis->patch_stride;

    int start_is, end_is;
    int start_js, end_js;

#define UPDATE_SPARSE_I_COORDINATES                                                                                    \
    if (i % pstr == 0 && i + psz <= h)                                                                                 \
        end_is++;                                                                                                      \
    if (i - psz >= 0 && (i - psz) % pstr == 0 && start_is < end_is)                                                    \
        start_is++;

#define UPDATE_SPARSE_J_COORDINATES                                                                                    \
    if (j % pstr == 0 && j + psz <= w)                                                                                 \
        end_js++;                                                                                                      \
    if (j - psz >= 0 && (j - psz) % pstr == 0 && start_js < end_js)                                                    \
        start_js++;

    start_is = 0;
    end_is = -1;
    for (int i = 0; i < start; i++)
    {
        UPDATE_SPARSE_I_COORDINATES;
    }
    for (int i = start; i < end; i++)
    {
        UPDATE_SPARSE_I_COORDINATES;
        start_js = 0;
        end_js = -1;
        for (int j = 0; j < w; j++)
        {
            UPDATE_SPARSE_J_COORDINATES;
            float coef, sum_coef = 0.0f;
            float sum_Ux = 0.0f;
            float sum_Uy = 0.0f;

            for (int is = start_is; is <= end_is; is++)
                for (int js = start_js; js <= end_js; js++)
                {
                    float diff;
                    float j_m = min(max(j + Sx_ptr[is * ws + js], 0.0f), w - 1.0f - EPS);
                    float i_m = min(max(i + Sy_ptr[is * ws + js], 0.0f), h - 1.0f - EPS);
                    int j_l = (int)j_m;
                    int j_u = j_l + 1;
                    int i_l = (int)i_m;
                    int i_u = i_l + 1;
                    diff = (j_m - j_l) * (i_m - i_l) * I1_ptr[i_u * w + j_u] +
                           (j_u - j_m) * (i_m - i_l) * I1_ptr[i_u * w + j_l] +
                           (j_m - j_l) * (i_u - i_m) * I1_ptr[i_l * w + j_u] +
                           (j_u - j_m) * (i_u - i_m) * I1_ptr[i_l * w + j_l] - I0_ptr[i * w + j];
                    coef = 1 / max(1.0f, abs(diff));
                    sum_Ux += coef * Sx_ptr[is * ws + js];
                    sum_Uy += coef * Sy_ptr[is * ws + js];
                    sum_coef += coef;
                }
            Ux_ptr[i * w + j] = sum_Ux / sum_coef;
            Uy_ptr[i * w + j] = sum_Uy / sum_coef;
        }
    }
}

void DISOpticalFlowImpl::calc(InputArray I0, InputArray I1, InputOutputArray flow)
{
    CV_Assert(!I0.empty() && I0.depth() == CV_8U && I0.channels() == 1);
    CV_Assert(!I1.empty() && I1.depth() == CV_8U && I1.channels() == 1);
    CV_Assert(I0.sameSize(I1));

    Mat I0Mat = I0.getMat();
    Mat I1Mat = I1.getMat();
    flow.create(I1Mat.size(), CV_32FC2);
    Mat &flowMat = flow.getMatRef();
    coarsest_scale = (int)(log((2 * I0Mat.cols) / (4.0 * patch_size)) / log(2.0) + 0.5) - 1;
    int num_stripes = getNumThreads();

    prepareBuffers(I0Mat, I1Mat);
    Ux[coarsest_scale].setTo(0.0f);
    Uy[coarsest_scale].setTo(0.0f);

    int hs;
    for (int i = coarsest_scale; i >= finest_scale; i--)
    {
        hs = 1 + (I0s[i].rows - patch_size) / patch_stride; // height of the sparse OF field
        precomputeStructureTensor(I0xx_buf, I0yy_buf, I0xy_buf, I0xs[i], I0ys[i]);
        parallel_for_(Range(0, num_stripes), PatchGradientDescent_ParBody(*this, num_stripes, hs, Sx, Sy, Ux[i], Uy[i],
                                                                          I0s[i], I1s_ext[i], I0xs[i], I0ys[i]));
        parallel_for_(Range(0, num_stripes),
                      Densification_ParBody(*this, num_stripes, I0s[i].rows, Ux[i], Uy[i], Sx, Sy, I0s[i], I1s[i]));
        if(variational_refinement_iter>0)
            variational_refinement_processors[i]->calcUV(I0s[i], I1s[i], Ux[i], Uy[i]);

        if (i > finest_scale)
        {
            resize(Ux[i], Ux[i - 1], Ux[i - 1].size());
            resize(Uy[i], Uy[i - 1], Uy[i - 1].size());
            Ux[i - 1] *= 2;
            Uy[i - 1] *= 2;
        }
    }
    Mat uxy[] = {Ux[finest_scale], Uy[finest_scale]};
    merge(uxy, 2, U);
    resize(U, flowMat, flowMat.size());
    flowMat *= pow(2, finest_scale);
}

void DISOpticalFlowImpl::collectGarbage()
{
    I0s.clear();
    I1s.clear();
    I0xs.clear();
    I0ys.clear();
    Ux.clear();
    Uy.clear();
    U.release();
    Sx.release();
    Sy.release();
    I0xx_buf.release();
    I0yy_buf.release();
    I0xy_buf.release();
    I0xx_buf_aux.release();
    I0yy_buf_aux.release();
    I0xy_buf_aux.release();

    for (int i = finest_scale; i <= coarsest_scale; i++)
        variational_refinement_processors[i]->collectGarbage();
    variational_refinement_processors.clear();
}

Ptr<DISOpticalFlow> createOptFlow_DIS(int preset) 
{ 
    Ptr<DISOpticalFlow> dis = makePtr<DISOpticalFlowImpl>();
    dis->setPatchSize(8);
    if (preset == DISOpticalFlow::PRESET_ULTRAFAST)
    {
        dis->setFinestScale(2);
        dis->setPatchStride(4);
        dis->setGradientDescentIterations(12);
        dis->setVariationalRefinementIterations(0);
    }
    else if (preset == DISOpticalFlow::PRESET_FAST)
    {
        dis->setFinestScale(2);
        dis->setPatchStride(4);
        dis->setGradientDescentIterations(16);
        dis->setVariationalRefinementIterations(5);
    }
    else if (preset == DISOpticalFlow::PRESET_MEDIUM)
    {
        dis->setFinestScale(1);
        dis->setPatchStride(3);
        dis->setGradientDescentIterations(20);
        dis->setVariationalRefinementIterations(5);
    }

    return dis;
}
}
}
