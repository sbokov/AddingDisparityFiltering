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

#include "precomp.hpp"
using namespace std;

namespace cv
{
namespace optflow
{

class VariationalRefinementImpl : public VariationalRefinement
{
  public:
    VariationalRefinementImpl();

    void calc(InputArray I0, InputArray I1, InputOutputArray flow);
    void calcUV(InputArray I0, InputArray I1, InputOutputArray flow_u, InputOutputArray flow_v);
    void collectGarbage();

  protected: // algorithm parameters
    int fixedPointIterations, sorIterations;
    float omega;
    float alpha, delta, gamma;
    float zeta, epsilon;
    int num_stripes;

  public: // getters and setters
    int getFixedPointIterations() const { return fixedPointIterations; }
    void setFixedPointIterations(int val) { fixedPointIterations = val; }
    int getSorIterations() const { return sorIterations; }
    void setSorIterations(int val) { sorIterations = val; }
    float getOmega() const { return omega; }
    void setOmega(float val) { omega = val; }
    float getAlpha() const { return alpha; }
    void setAlpha(float val) { alpha = val; }
    float getDelta() const { return delta; }
    void setDelta(float val) { delta = val; }
    float getGamma() const { return gamma; }
    void setGamma(float val) { gamma = val; }

  protected:                                         // internal buffers
    Mat_<float> Ix, Iy, Iz, Ixx, Ixy, Iyy, Ixz, Iyz; // image derivatives
    Mat_<float> A11, A12, A22, b1, b2;               // linear system coefficients
    Mat_<float> weights;                             // smoothness term weights in the current fixed point iteration

    Mat_<float> mapX, mapY; // auxiliary buffers for remapping

    Mat_<float> tempW_u, tempW_v; // flow version that is modified in each fixed point iteration
    Mat_<float> dW_u, dW_v;       // optical flow increment

  private: // private methods
    void warpImage(Mat &dst, const Mat &src, const Mat &flow_u, const Mat &flow_v);
    void prepareBuffers(const Mat &I0, const Mat &I1, const Mat &W_u, const Mat &W_v);
    void computeDataTerm(const Mat &dW_u, const Mat &dW_v, int start_i, int end_i);
    void computeSmoothnessTerm(const Mat &W_u, const Mat &W_v, const Mat &curW_u, const Mat &curW_v, int start_i,
                               int end_i);

    struct ComputeTerms_ParBody : public ParallelLoopBody
    {
        VariationalRefinementImpl *var;
        Mat *W_u, *W_v, *dW_u, *dW_v, *tempW_u, *tempW_v;
        int nstripes, stripe_sz;
        int h;

        ComputeTerms_ParBody(VariationalRefinementImpl &_var, int _nstripes, int _h, Mat &W_u, Mat &_W_v, Mat &_dW_u,
                             Mat &_dW_v, Mat &_tempW_u, Mat &_tempW_v);
        void operator()(const Range &range) const;
    };

    struct RedBlackSOR_ParBody : public ParallelLoopBody
    {
        VariationalRefinementImpl *var;
        bool is_red;  // red of black pass
        int chunk_sz; // size of contiguous chunks of memory of the same color (for better data locality)
        int nstripes, stripe_sz;
        int h;
        Mat *dW_u, *dW_v;

        RedBlackSOR_ParBody(VariationalRefinementImpl &_var, bool is_red_pass, int _chunk_sz, int _nstripes, int _h,
                            Mat &_dW_u, Mat &_dW_v);
        void operator()(const Range &range) const;
    };
};

VariationalRefinementImpl::VariationalRefinementImpl()
{
    fixedPointIterations = 5;
    sorIterations = 5;
    alpha = 20.0f;
    delta = 5.0f;
    gamma = 10.0f;
    omega = 1.6f;
    zeta = 0.1f;
    epsilon = 0.001f;
    num_stripes = getNumThreads();
}

void VariationalRefinementImpl::warpImage(Mat &dst, const Mat &src, const Mat &flow_u, const Mat &flow_v)
{
    const float *pFlowU, *pFlowV;
    float *pMapX, *pMapY;
    for (int i = 0; i < flow_u.rows; i++)
    {
        pFlowU = flow_u.ptr<float>(i);
        pFlowV = flow_v.ptr<float>(i);
        pMapX = mapX.ptr<float>(i);
        pMapY = mapY.ptr<float>(i);
        for (int j = 0; j < flow_u.cols; j++)
        {
            pMapX[j] = j + pFlowU[j];
            pMapY[j] = i + pFlowV[j];
        }
    }
    remap(src, dst, mapX, mapY, INTER_LINEAR, BORDER_REPLICATE);
}

void VariationalRefinementImpl::prepareBuffers(const Mat &I0, const Mat &I1, const Mat &W_u, const Mat &W_v)
{
    Size s = I0.size();
    A11.create(s);
    A12.create(s);
    A22.create(s);
    b1.create(s);
    b2.create(s);
    weights.create(s);
    tempW_u.create(s);
    tempW_v.create(s);
    dW_u.create(s);
    dW_v.create(s);

    mapX.create(s);
    mapY.create(s);
    Mat I1flt, warpedI;
    I1.convertTo(I1flt, CV_32F); // works slightly better with floating-point warps
    warpImage(warpedI, I1flt, W_u, W_v);

    Ix.create(s);
    Iy.create(s);
    Iz.create(s);
    Ixx.create(s);
    Ixy.create(s);
    Iyy.create(s);
    Ixz.create(s);
    Iyz.create(s);

    // computing derivatives on the average of the current and warped next frame:
    int kernel_size = 1;
    Mat averagedI;
    addWeighted(I0, 0.5, warpedI, 0.5, 0.0, averagedI, CV_32F);
    Sobel(averagedI, Ix, -1, 1, 0, kernel_size, 1, 0.00, BORDER_REPLICATE);
    Sobel(averagedI, Iy, -1, 0, 1, kernel_size, 1, 0.00, BORDER_REPLICATE);
    Sobel(Ix, Ixx, -1, 1, 0, kernel_size, 1, 0.00, BORDER_REPLICATE);
    Sobel(Ix, Ixy, -1, 0, 1, kernel_size, 1, 0.00, BORDER_REPLICATE);
    Sobel(Iy, Iyy, -1, 0, 1, kernel_size, 1, 0.00, BORDER_REPLICATE);

    // computing temporal derivatives (along the flow):
    subtract(warpedI, I0, Iz, noArray(), CV_32F);
    Sobel(Iz, Ixz, -1, 1, 0, kernel_size, 1, 0.00, BORDER_REPLICATE);
    Sobel(Iz, Iyz, -1, 0, 1, kernel_size, 1, 0.00, BORDER_REPLICATE);
}

void VariationalRefinementImpl::computeDataTerm(const Mat &dW_u, const Mat &dW_v, int start_i, int end_i)
{
    /*Using robust data term based on color and gradient constancy assumptions*/

    /*In this function we compute linear system coefficients
      A11,A12,A22,b1,b1 based on the data term */

    const float zeta_squared = zeta * zeta;
    const float epsilon_squared = epsilon * epsilon;
    const float gamma2 = gamma / 2;
    const float delta2 = delta / 2;

    const float *pIx, *pIy, *pIz;
    const float *pIxx, *pIxy, *pIyy, *pIxz, *pIyz;
    const float *pdU, *pdV;
    float *pa11, *pa12, *pa22, *pb1, *pb2;

    float derivNorm, derivNorm2;
    float mult, mult2;
    float Ik1z, Ik1zx, Ik1zy;
    float weight;
    for (int i = start_i; i < end_i; i++)
    {
        pIx = Ix.ptr<float>(i);
        pIy = Iy.ptr<float>(i);
        pIz = Iz.ptr<float>(i);
        pIxx = Ixx.ptr<float>(i);
        pIxy = Ixy.ptr<float>(i);
        pIyy = Iyy.ptr<float>(i);
        pIxz = Ixz.ptr<float>(i);
        pIyz = Iyz.ptr<float>(i);

        pa11 = A11.ptr<float>(i);
        pa12 = A12.ptr<float>(i);
        pa22 = A22.ptr<float>(i);
        pb1 = b1.ptr<float>(i);
        pb2 = b2.ptr<float>(i);

        pdU = dW_u.ptr<float>(i);
        pdV = dW_v.ptr<float>(i);
        for (int j = 0; j < dW_u.cols; j++)
        {
            // Step 1: color contancy
            // Normalization factor:
            derivNorm = (float)*pIx * (*pIx) + (*pIy) * (*pIy) + zeta_squared;
            // Color constancy penalty (computed by Taylor expansion):
            Ik1z = *pIz + (*pIx * *pdU) + (*pIy * *pdV);
            // Weight of the color constancy term in the current fixed-point iteration:
            weight = delta2 / sqrt(Ik1z * Ik1z / derivNorm + epsilon_squared);
            // Add respective color constancy components to the linear sustem coefficients:
            // mult = weight / derivNorm;
            *pa11 = weight * ((float)*pIx * *pIx / derivNorm);
            *pa12 = weight * ((float)*pIx * *pIy / derivNorm);
            *pa22 = weight * ((float)*pIy * *pIy / derivNorm);
            *pb1 = -weight * ((float)*pIz * *pIx / derivNorm);
            *pb2 = -weight * ((float)*pIz * *pIy / derivNorm);

            // Step 2: gradient contancy
            // Normalization factor for x gradient:
            derivNorm = (float)(*pIxx) * *pIxx + *pIxy * *pIxy + zeta_squared;
            // Normalization factor for y gradient:
            derivNorm2 = (float)(*pIyy) * *pIyy + *pIxy * *pIxy + zeta_squared;
            // Gradient constancy penalties (computed by Taylor expansion):
            Ik1zx = *pIxz + *pIxx * *pdU + *pIxy * *pdV;
            Ik1zy = *pIyz + *pIxy * *pdU + *pIyy * *pdV;

            // Weight of the gradient constancy term in the current fixed-point iteration:
            weight = gamma2 / sqrt(Ik1zx * Ik1zx / derivNorm + Ik1zy * Ik1zy / derivNorm2 + epsilon_squared);
            // Add respective gradient constancy components to the linear system coefficients:
            mult = weight / derivNorm;
            mult2 = weight / derivNorm2;
            *pa11 += weight * ((float)*pIxx * *pIxx / derivNorm + (float)*pIxy * *pIxy / derivNorm2);
            *pa12 += weight * ((float)*pIxx * *pIxy / derivNorm + (float)*pIxy * *pIyy / derivNorm2);
            *pa22 += weight * ((float)*pIxy * *pIxy / derivNorm + (float)*pIyy * *pIyy / derivNorm2);
            *pb1 += -weight * ((float)*pIxx * *pIxz / derivNorm + (float)*pIxy * *pIyz / derivNorm2);
            *pb2 += -weight * ((float)*pIxy * *pIxz / derivNorm + (float)*pIyy * *pIyz / derivNorm2);

            pIx++;
            pIy++;
            pIz++;
            pIxx++;
            pIxy++;
            pIyy++;
            pIxz++;
            pIyz++;
            pdU++;
            pdV++;
            pa11++;
            pa12++;
            pa22++;
            pb1++;
            pb2++;
        }
    }
}

void VariationalRefinementImpl::computeSmoothnessTerm(const Mat &W_u, const Mat &W_v, const Mat &curW_u,
                                                      const Mat &curW_v, int start_i, int end_i)
{
    /*Using robust penalty on flow gradient*/

    /*In this function we update b1, b2, A11, A22 coefficients of the linear system
      and compute smoothness term weights for the current fixed-point iteration */

    const float epsilon_squared = epsilon * epsilon;
    const float alpha2 = alpha / 2;
    float *pW, *pB1, *pB2, *pB1_next, *pB2_next, *pA11, *pA22, *pA11_next, *pA22_next;
    const float *cW_u, *cW_v, *cW_u_next, *cW_v_next;
    const float *pW_u, *pW_v, *pW_u_next, *pW_v_next;
    float ux, uy, vx, vy, iB1, iB2;

#define PROC_ALL(cW_u, cW_v, pW_u, pW_v, cW_u_next, cW_v_next, pW_u_next, pW_v_next)                                   \
    /*gradients for the flow on the current fixed-point iteration:*/                                                   \
    ux = *(cW_u + 1) - *cW_u;                                                                                          \
    vx = *(cW_v + 1) - *cW_v;                                                                                          \
    uy = *cW_u_next - *cW_u;                                                                                           \
    vy = *cW_v_next - *cW_v;                                                                                           \
    /* weight of the smoothness term in the current fixed-point iteration:*/                                           \
    *pW = alpha2 / sqrt(ux * ux + vx * vx + uy * uy + vy * vy + epsilon_squared);                                      \
    /* gradients for initial raw flow multiplied by weight:*/                                                          \
    ux = *pW * (*(pW_u + 1) - *pW_u);                                                                                  \
    vx = *pW * (*(pW_v + 1) - *pW_v);                                                                                  \
    uy = *pW * (*pW_u_next - *pW_u);                                                                                   \
    vy = *pW * (*pW_v_next - *pW_v);

#define PROC_VERT(cW_u, cW_v, pW_u, pW_v, cW_u_next, cW_v_next, pW_u_next, pW_v_next)                                  \
    /*process only vertical gradients*/                                                                                \
    uy = *cW_u_next - *cW_u;                                                                                           \
    vy = *cW_v_next - *cW_v;                                                                                           \
    *pW = alpha2 / sqrt(uy * uy + vy * vy + epsilon_squared);                                                          \
    uy = *pW * (*pW_u_next - *pW_u);                                                                                   \
    vy = *pW * (*pW_v_next - *pW_v);

#define PROC_HORIZ(cW_u, cW_v, pW_u, pW_v)                                                                             \
    /*process only horizontal gradients*/                                                                              \
    ux = *(cW_u + 1) - *cW_u;                                                                                          \
    vx = *(cW_v + 1) - *cW_v;                                                                                          \
    *pW = alpha2 / sqrt(ux * ux + vx * vx + epsilon_squared);                                                          \
    ux = *pW * (*(pW_u + 1) - *pW_u);                                                                                  \
    vx = *pW * (*(pW_v + 1) - *pW_v);

#define INC_CUR                                                                                                        \
    pW++;                                                                                                              \
    cW_u++;                                                                                                            \
    cW_v++;                                                                                                            \
    pW_u++;                                                                                                            \
    pW_v++;                                                                                                            \
    pB1++;                                                                                                             \
    pB2++;                                                                                                             \
    pA11++;                                                                                                            \
    pA22++;

#define INC_NEXT                                                                                                       \
    cW_u_next++;                                                                                                       \
    cW_v_next++;                                                                                                       \
    pW_u_next++;                                                                                                       \
    pW_v_next++;                                                                                                       \
    pB1_next++;                                                                                                        \
    pB2_next++;                                                                                                        \
    pA11_next++;                                                                                                       \
    pA22_next++;

#define UPDATE_HORIZ_CUR                                                                                               \
    *pB1 += ux;                                                                                                        \
    *pA11 += *pW;                                                                                                      \
    *pB2 += vx;                                                                                                        \
    *pA22 += *pW;

#define UPDATE_HORIZ_NEXT                                                                                              \
    *(pB1 + 1) -= ux;                                                                                                  \
    *(pA11 + 1) += *pW;                                                                                                \
    *(pB2 + 1) -= vx;                                                                                                  \
    *(pA22 + 1) += *pW;

#define UPDATE_VERT_CUR                                                                                                \
    *pB1 += uy;                                                                                                        \
    *pA11 += *pW;                                                                                                      \
    *pB2 += vy;                                                                                                        \
    *pA22 += *pW;

#define UPDATE_VERT_NEXT(pB1_next, pA11_next, pB2_next, pA22_next)                                                     \
    *pB1_next -= uy;                                                                                                   \
    *pA11_next += *pW;                                                                                                 \
    *pB2_next -= vy;                                                                                                   \
    *pA22_next += *pW;

    for (int i = start_i; i < end_i; i++)
    {
        pW = weights.ptr<float>(i);
        pB1 = b1.ptr<float>(i);
        pB2 = b2.ptr<float>(i);
        pA11 = A11.ptr<float>(i);
        pA22 = A22.ptr<float>(i);

        cW_u = curW_u.ptr<float>(i);
        cW_v = curW_v.ptr<float>(i);
        pW_u = W_u.ptr<float>(i);
        pW_v = W_v.ptr<float>(i);
        if (i == W_u.rows - 1)
        {
            // only horizontal gradients:
            for (int j = 0; j < W_u.cols - 1; j++)
            {
                PROC_HORIZ(cW_u, cW_v, pW_u, pW_v);
                UPDATE_HORIZ_CUR;
                UPDATE_HORIZ_NEXT;
                INC_CUR;
            }
        }
        else
        {
            cW_u_next = curW_u.ptr<float>(i + 1);
            cW_v_next = curW_v.ptr<float>(i + 1);
            pW_u_next = W_u.ptr<float>(i + 1);
            pW_v_next = W_v.ptr<float>(i + 1);
            pB1_next = b1.ptr<float>(i + 1);
            pB2_next = b2.ptr<float>(i + 1);
            pA11_next = A11.ptr<float>(i + 1);
            pA22_next = A22.ptr<float>(i + 1);

            if (i == start_i && i > 0)
            {
                // also apply operations from the previous row:
                const float *cW_u_prev = curW_u.ptr<float>(i - 1);
                const float *cW_v_prev = curW_v.ptr<float>(i - 1);
                const float *pW_u_prev = W_u.ptr<float>(i - 1);
                const float *pW_v_prev = W_v.ptr<float>(i - 1);
                for (int j = 0; j < W_u.cols - 1; j++)
                {
                    PROC_ALL(cW_u_prev, cW_v_prev, pW_u_prev, pW_v_prev, cW_u, cW_v, pW_u, pW_v);
                    UPDATE_VERT_NEXT(pB1, pA11, pB2, pA22);
                    PROC_ALL(cW_u, cW_v, pW_u, pW_v, cW_u_next, cW_v_next, pW_u_next, pW_v_next);
                    UPDATE_HORIZ_CUR;
                    UPDATE_HORIZ_NEXT;
                    UPDATE_VERT_CUR;
                    UPDATE_VERT_NEXT(pB1_next, pA11_next, pB2_next, pA22_next);
                    INC_CUR;
                    INC_NEXT;
                    cW_u_prev++;
                    cW_v_prev++;
                    pW_u_prev++;
                    pW_v_prev++;
                }
                PROC_VERT(cW_u_prev, cW_v_prev, pW_u_prev, pW_v_prev, cW_u, cW_v, pW_u, pW_v);
                UPDATE_VERT_NEXT(pB1, pA11, pB2, pA22);
                PROC_VERT(cW_u, cW_v, pW_u, pW_v, cW_u_next, cW_v_next, pW_u_next, pW_v_next);
                UPDATE_VERT_CUR;
                UPDATE_VERT_NEXT(pB1_next, pA11_next, pB2_next, pA22_next);
            }
            else if (i == end_i - 1)
            {
                // do not update the next row (it's in other processor):
                for (int j = 0; j < W_u.cols - 1; j++)
                {
                    PROC_ALL(cW_u, cW_v, pW_u, pW_v, cW_u_next, cW_v_next, pW_u_next, pW_v_next);
                    UPDATE_HORIZ_CUR;
                    UPDATE_HORIZ_NEXT;
                    UPDATE_VERT_CUR;
                    INC_CUR;
                    INC_NEXT;
                }
                PROC_VERT(cW_u, cW_v, pW_u, pW_v, cW_u_next, cW_v_next, pW_u_next, pW_v_next);
                UPDATE_VERT_CUR;
            }
            else
            {
                for (int j = 0; j < W_u.cols - 1; j++)
                {
                    PROC_ALL(cW_u, cW_v, pW_u, pW_v, cW_u_next, cW_v_next, pW_u_next, pW_v_next);
                    UPDATE_HORIZ_CUR;
                    UPDATE_HORIZ_NEXT;
                    UPDATE_VERT_CUR;
                    UPDATE_VERT_NEXT(pB1_next, pA11_next, pB2_next, pA22_next);
                    INC_CUR;
                    INC_NEXT;
                }
                PROC_VERT(cW_u, cW_v, pW_u, pW_v, cW_u_next, cW_v_next, pW_u_next, pW_v_next);
                UPDATE_VERT_CUR;
                UPDATE_VERT_NEXT(pB1_next, pA11_next, pB2_next, pA22_next);
            }
        }
    }
#undef PROC_ALL
#undef PROC_VERT
#undef PROC_HORIZ
#undef INC_CUR
#undef INC_NEXT
#undef UPDATE_HORIZ_CUR
#undef UPDATE_HORIZ_NEXT
#undef UPDATE_VERT_CUR
#undef UPDATE_VERT_NEXT
}

VariationalRefinementImpl::ComputeTerms_ParBody::ComputeTerms_ParBody(VariationalRefinementImpl &_var, int _nstripes,
                                                                      int _h, Mat &_W_u, Mat &_W_v, Mat &_dW_u,
                                                                      Mat &_dW_v, Mat &_tempW_u, Mat &_tempW_v)
    : var(&_var), nstripes(_nstripes), h(_h), W_u(&_W_u), W_v(&_W_v), dW_u(&_dW_u), dW_v(&_dW_v), tempW_u(&_tempW_u),
      tempW_v(&_tempW_v)
{
    stripe_sz = (int)ceil(h / (double)nstripes);
}

void VariationalRefinementImpl::ComputeTerms_ParBody::operator()(const Range &range) const
{
    int start = min(range.start * stripe_sz, h);
    int end = min(range.end * stripe_sz, h);
    var->computeDataTerm(*dW_u, *dW_v, start, end);
    var->computeSmoothnessTerm(*W_u, *W_v, *tempW_u, *tempW_v, start, end);
}

VariationalRefinementImpl::RedBlackSOR_ParBody::RedBlackSOR_ParBody(VariationalRefinementImpl &_var, bool is_red_pass,
                                                                    int _chunk_sz, int _nstripes, int _h, Mat &_dW_u,
                                                                    Mat &_dW_v)
    : var(&_var), nstripes(_nstripes), h(_h), chunk_sz(_chunk_sz), is_red(is_red_pass), dW_u(&_dW_u), dW_v(&_dW_v)
{
    stripe_sz = (int)ceil(h / (double)nstripes);
}

#define UPDATE_UV                                                                                                      \
    *pdu += omega * ((sigmaU + *pb1 - *pdv * *pa12) / *pa11 - *pdu);                                                   \
    *pdv += omega * ((sigmaV + *pb2 - *pdu * *pa12) / *pa22 - *pdv);                                                   \
    pa11++;                                                                                                            \
    pa12++;                                                                                                            \
    pa22++;                                                                                                            \
    pb1++;                                                                                                             \
    pb2++;                                                                                                             \
    pdu++;                                                                                                             \
    pdv++;                                                                                                             \
    pW++;

inline void processChunkLT(float *pa11, float *pa12, float *pa22, float *pb1, float *pb2, float *pW, float *pdu,
                           float *pdv, float omega, int s, int chunk_sz)
{
    float sigmaU, sigmaV;
    sigmaU = *pW * *(pdu + 1) + *pW * *(pdu + s);
    sigmaV = *pW * *(pdv + 1) + *pW * *(pdv + s);
    UPDATE_UV;
    for (int i = 1; i < chunk_sz; i++)
    {
        sigmaU = *(pW - 1) * *(pdu - 1) + *pW * *(pdu + 1) + *pW * *(pdu + s);
        sigmaV = *(pW - 1) * *(pdv - 1) + *pW * *(pdv + 1) + *pW * *(pdv + s);
        UPDATE_UV;
    }
}

inline void processChunkMT(float *pa11, float *pa12, float *pa22, float *pb1, float *pb2, float *pW, float *pdu,
                           float *pdv, float omega, int s, int chunk_sz)
{
    float sigmaU, sigmaV;
    for (int i = 0; i < chunk_sz; i++)
    {
        sigmaU = *(pW - 1) * *(pdu - 1) + *pW * *(pdu + 1) + *pW * *(pdu + s);
        sigmaV = *(pW - 1) * *(pdv - 1) + *pW * *(pdv + 1) + *pW * *(pdv + s);
        UPDATE_UV;
    }
}

inline void processChunkRT(float *pa11, float *pa12, float *pa22, float *pb1, float *pb2, float *pW, float *pdu,
                           float *pdv, float omega, int s, int chunk_sz)
{
    float sigmaU, sigmaV;
    for (int i = 0; i < chunk_sz - 1; i++)
    {
        sigmaU = *(pW - 1) * *(pdu - 1) + *pW * *(pdu + 1) + *pW * *(pdu + s);
        sigmaV = *(pW - 1) * *(pdv - 1) + *pW * *(pdv + 1) + *pW * *(pdv + s);
        UPDATE_UV;
    }
    sigmaU = *(pW - 1) * *(pdu - 1) + *pW * *(pdu + s);
    sigmaV = *(pW - 1) * *(pdv - 1) + *pW * *(pdv + s);
    UPDATE_UV;
}

inline void processChunkLM(float *pa11, float *pa12, float *pa22, float *pb1, float *pb2, float *pW, float *pdu,
                           float *pdv, float omega, int s, int chunk_sz)
{
    float sigmaU, sigmaV;
    sigmaU = *pW * *(pdu + 1) + *(pW - s) * *(pdu - s) + *pW * *(pdu + s);
    sigmaV = *pW * *(pdv + 1) + *(pW - s) * *(pdv - s) + *pW * *(pdv + s);
    UPDATE_UV;
    for (int i = 1; i < chunk_sz; i++)
    {
        sigmaU = *(pW - 1) * *(pdu - 1) + *pW * *(pdu + 1) + *(pW - s) * *(pdu - s) + *pW * *(pdu + s);
        sigmaV = *(pW - 1) * *(pdv - 1) + *pW * *(pdv + 1) + *(pW - s) * *(pdv - s) + *pW * *(pdv + s);
        UPDATE_UV;
    }
}

inline void processChunkMM(float *pa11, float *pa12, float *pa22, float *pb1, float *pb2, float *pW, float *pdu,
                           float *pdv, float omega, int s, int chunk_sz)
{
    float sigmaU, sigmaV;
    for (int i = 0; i < chunk_sz; i++)
    {
        sigmaU = *(pW - 1) * *(pdu - 1) + *pW * *(pdu + 1) + *(pW - s) * *(pdu - s) + *pW * *(pdu + s);
        sigmaV = *(pW - 1) * *(pdv - 1) + *pW * *(pdv + 1) + *(pW - s) * *(pdv - s) + *pW * *(pdv + s);
        UPDATE_UV;
    }
}

inline void processChunkRM(float *pa11, float *pa12, float *pa22, float *pb1, float *pb2, float *pW, float *pdu,
                           float *pdv, float omega, int s, int chunk_sz)
{
    float sigmaU, sigmaV;
    for (int i = 0; i < chunk_sz - 1; i++)
    {
        sigmaU = *(pW - 1) * *(pdu - 1) + *pW * *(pdu + 1) + *(pW - s) * *(pdu - s) + *pW * *(pdu + s);
        sigmaV = *(pW - 1) * *(pdv - 1) + *pW * *(pdv + 1) + *(pW - s) * *(pdv - s) + *pW * *(pdv + s);
        UPDATE_UV;
    }
    sigmaU = *(pW - 1) * *(pdu - 1) + *(pW - s) * *(pdu - s) + *pW * *(pdu + s);
    sigmaV = *(pW - 1) * *(pdv - 1) + *(pW - s) * *(pdv - s) + *pW * *(pdv + s);
    UPDATE_UV;
}

inline void processChunkLB(float *pa11, float *pa12, float *pa22, float *pb1, float *pb2, float *pW, float *pdu,
                           float *pdv, float omega, int s, int chunk_sz)
{
    float sigmaU, sigmaV;
    sigmaU = *pW * *(pdu + 1) + *(pW - s) * *(pdu - s);
    sigmaV = *pW * *(pdv + 1) + *(pW - s) * *(pdv - s);
    UPDATE_UV;
    for (int i = 1; i < chunk_sz; i++)
    {
        sigmaU = *(pW - 1) * *(pdu - 1) + *pW * *(pdu + 1) + *(pW - s) * *(pdu - s);
        sigmaV = *(pW - 1) * *(pdv - 1) + *pW * *(pdv + 1) + *(pW - s) * *(pdv - s);
        UPDATE_UV;
    }
}

inline void processChunkMB(float *pa11, float *pa12, float *pa22, float *pb1, float *pb2, float *pW, float *pdu,
                           float *pdv, float omega, int s, int chunk_sz)
{
    float sigmaU, sigmaV;
    for (int i = 0; i < chunk_sz; i++)
    {
        sigmaU = *(pW - 1) * *(pdu - 1) + *pW * *(pdu + 1) + *(pW - s) * *(pdu - s);
        sigmaV = *(pW - 1) * *(pdv - 1) + *pW * *(pdv + 1) + *(pW - s) * *(pdv - s);
        UPDATE_UV;
    }
}

inline void processChunkRB(float *pa11, float *pa12, float *pa22, float *pb1, float *pb2, float *pW, float *pdu,
                           float *pdv, float omega, int s, int chunk_sz)
{
    float sigmaU, sigmaV;
    for (int i = 0; i < chunk_sz - 1; i++)
    {
        sigmaU = *(pW - 1) * *(pdu - 1) + *pW * *(pdu + 1) + *(pW - s) * *(pdu - s);
        sigmaV = *(pW - 1) * *(pdv - 1) + *pW * *(pdv + 1) + *(pW - s) * *(pdv - s);
        UPDATE_UV;
    }
    sigmaU = *(pW - 1) * *(pdu - 1) + *(pW - s) * *(pdu - s);
    sigmaV = *(pW - 1) * *(pdv - 1) + *(pW - s) * *(pdv - s);
    UPDATE_UV;
}

#undef UPDATE_UV

void VariationalRefinementImpl::RedBlackSOR_ParBody::operator()(const Range &range) const
{
    CV_Assert(dW_u->cols > chunk_sz);
    int start = min(range.start * stripe_sz, h);
    int end = min(range.end * stripe_sz, h);

    float *pa11, *pa12, *pa22, *pb1, *pb2, *pW;
    float *pdu, *pdv;

    int cols = dW_u->cols;
    int rows = dW_u->rows;
    int s = dW_u->cols; // step between rows

#define PROC_ROW(lfunc, mfunc, rfunc)                                                                                  \
    {                                                                                                                  \
        if (j == 0)                                                                                                    \
        {                                                                                                              \
            lfunc(pa11 + j, pa12 + j, pa22 + j, pb1 + j, pb2 + j, pW + j, pdu + j, pdv + j, omega, s, chunk_sz);       \
            j += 2 * chunk_sz;                                                                                         \
        }                                                                                                              \
        for (; j < cols - 2 * chunk_sz; j += 2 * chunk_sz)                                                             \
            mfunc(pa11 + j, pa12 + j, pa22 + j, pb1 + j, pb2 + j, pW + j, pdu + j, pdv + j, omega, s, chunk_sz);       \
        if (j + chunk_sz >= cols)                                                                                      \
            rfunc(pa11 + j, pa12 + j, pa22 + j, pb1 + j, pb2 + j, pW + j, pdu + j, pdv + j, omega, s, cols - j);       \
        else                                                                                                           \
            mfunc(pa11 + j, pa12 + j, pa22 + j, pb1 + j, pb2 + j, pW + j, pdu + j, pdv + j, omega, s, chunk_sz);       \
    }

    float omega = var->omega;
    int j;
    for (int i = start; i < end; i++)
    {
        pa11 = var->A11.ptr<float>(i);
        pa12 = var->A12.ptr<float>(i);
        pa22 = var->A22.ptr<float>(i);
        pb1 = var->b1.ptr<float>(i);
        pb2 = var->b2.ptr<float>(i);
        pW = var->weights.ptr<float>(i);
        pdu = dW_u->ptr<float>(i);
        pdv = dW_v->ptr<float>(i);

        if (is_red)
            j = (i % 2) * chunk_sz;
        else
            j = ((i + 1) % 2) * chunk_sz;

        if (i == 0)
            PROC_ROW(processChunkLT, processChunkMT, processChunkRT)
        else if (i == rows - 1)
            PROC_ROW(processChunkLB, processChunkMB, processChunkRB)
        else
            PROC_ROW(processChunkLM, processChunkMM, processChunkRM)
    }
#undef PROC_ROW
}

void VariationalRefinementImpl::calc(InputArray I0, InputArray I1, InputOutputArray flow)
{
    CV_Assert(!I0.empty() && I0.depth() == CV_8U && I0.channels() == 1);
    CV_Assert(!I1.empty() && I1.depth() == CV_8U && I1.channels() == 1);
    CV_Assert(I0.sameSize(I1));
    CV_Assert(!flow.empty() && flow.depth() == CV_32F && flow.channels() == 2);
    CV_Assert(I0.sameSize(flow));

    Mat u, v;
    Mat &flowMat = flow.getMatRef();
    Mat uv[] = {u, v};
    split(flowMat, uv);
    calcUV(I0, I1, u, v);
    merge(uv, 2, flowMat);
}
void VariationalRefinementImpl::calcUV(InputArray I0, InputArray I1, InputOutputArray flow_u, InputOutputArray flow_v)
{
    CV_Assert(!I0.empty() && I0.depth() == CV_8U && I0.channels() == 1);
    CV_Assert(!I1.empty() && I1.depth() == CV_8U && I1.channels() == 1);
    CV_Assert(I0.sameSize(I1));
    CV_Assert(!flow_u.empty() && flow_u.depth() == CV_32F && flow_u.channels() == 1);
    CV_Assert(!flow_v.empty() && flow_v.depth() == CV_32F && flow_v.channels() == 1);
    CV_Assert(I0.sameSize(flow_u));
    CV_Assert(flow_u.sameSize(flow_v));

    Mat I0Mat = I0.getMat();
    Mat I1Mat = I1.getMat();
    Mat &W_u = flow_u.getMatRef();
    Mat &W_v = flow_v.getMatRef();
    prepareBuffers(I0Mat, I1Mat, W_u, W_v);

    W_u.copyTo(tempW_u);
    W_v.copyTo(tempW_v);
    dW_u.setTo(0.0f);
    dW_v.setTo(0.0f);
    int SOR_chunk_size = 10; // ideally, something that just fits into one cache line
    for (int i = 0; i < fixedPointIterations; i++)
    {
        parallel_for_(Range(0, num_stripes),
                      ComputeTerms_ParBody(*this, num_stripes, I0Mat.rows, W_u, W_v, dW_u, dW_v, tempW_u, tempW_v));
        for (int j = 0; j < sorIterations; j++)
        {
            // compute all red vertices in parallel:
            parallel_for_(Range(0, num_stripes),
                          RedBlackSOR_ParBody(*this, true, SOR_chunk_size, num_stripes, I0Mat.rows, dW_u, dW_v));
            // compute all black vertices in parallel:
            parallel_for_(Range(0, num_stripes),
                          RedBlackSOR_ParBody(*this, false, SOR_chunk_size, num_stripes, I0Mat.rows, dW_u, dW_v));
        }
        tempW_u = W_u + dW_u;
        tempW_v = W_v + dW_v;
    }
    tempW_u.copyTo(W_u);
    tempW_v.copyTo(W_v);
}
void VariationalRefinementImpl::collectGarbage()
{
    A11.release();
    A12.release();
    A22.release();
    b1.release();
    b2.release();
    weights.release();
    tempW_u.release();
    tempW_v.release();
    dW_u.release();
    dW_v.release();

    mapX.release();
    mapY.release();

    Ix.release();
    Iy.release();
    Iz.release();
    Ixx.release();
    Ixy.release();
    Iyy.release();
    Ixz.release();
    Iyz.release();
}

Ptr<VariationalRefinement> createVariationalFlowRefinement() { return makePtr<VariationalRefinementImpl>(); }
}
}
