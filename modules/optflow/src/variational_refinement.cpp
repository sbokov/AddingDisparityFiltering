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

  protected: // internal buffers
    struct RedBlackBuffer
    {
        // special data layout (separate storage of "red" and "black" elements),
        // more SIMD-friendly and easier to parallelize
        // uses padding to simplify border processing
        Mat_<float> red;   // (i+j)%2==0
        Mat_<float> black; // (i+j)%2==1

        // can be different if width%2==1:
        int red_even_len, red_odd_len;
        int black_even_len, black_odd_len;

        void create(Size s);
        void release();
    };
    Mat_<float> Ix, Iy, Iz, Ixx, Ixy, Iyy, Ixz, Iyz; // image derivatives
    RedBlackBuffer Ix_rb, Iy_rb, Iz_rb, Ixx_rb, Ixy_rb, Iyy_rb, Ixz_rb, Iyz_rb;
    RedBlackBuffer A11, A12, A22, b1, b2; // linear system coefficients
    RedBlackBuffer weights;               // smoothness term weights in the current fixed point iteration

    Mat_<float> mapX, mapY; // auxiliary buffers for remapping

    RedBlackBuffer tempW_u, tempW_v; // flow version that is modified in each fixed point iteration
    RedBlackBuffer dW_u, dW_v;       // optical flow increment
    RedBlackBuffer W_u_rb, W_v_rb;   // split version of the input flow

  private: // private methods
    void splitCheckerboard(RedBlackBuffer &dst, Mat &src);
    void mergeCheckerboard(Mat &dst, RedBlackBuffer &src);
    void updateRepeatedBorders(RedBlackBuffer &dst);
    void warpImage(Mat &dst, const Mat &src, const Mat &flow_u, const Mat &flow_v);
    void prepareBuffers(const Mat &I0, const Mat &I1, const Mat &W_u, const Mat &W_v);

    typedef void (VariationalRefinementImpl::*Op)(void* op1, void* op2, void* op3);

    struct ParallelOp_ParBody : public ParallelLoopBody
    {
        VariationalRefinementImpl* var;
        vector<Op> ops;
        vector<void*> op1s;
        vector<void*> op2s;
        vector<void*> op3s;

        ParallelOp_ParBody(VariationalRefinementImpl& _var, vector<Op> _ops, vector<void*>& _op1s, vector<void*>& _op2s, vector<void*>& _op3s);
        void operator () (const Range& range) const;
    };

    void gradHorizAndSplitOp(void* src, void* dst, void* dst_split)
    {
        Sobel(*(Mat*)src, *(Mat*)dst, -1, 1, 0, 1, 1, 0.00, BORDER_REPLICATE);
        splitCheckerboard(*(RedBlackBuffer*)dst_split, *(Mat*)dst);
    }

    void gradVertAndSplitOp(void* src, void* dst, void* dst_split)
    {
        Sobel(*(Mat*)src, *(Mat*)dst, -1, 0, 1, 1, 1, 0.00, BORDER_REPLICATE);
        splitCheckerboard(*(RedBlackBuffer*)dst_split, *(Mat*)dst);
    }

    void averageOp(void* src1, void* src2, void* dst)
    {
        addWeighted(*(Mat*)src1, 0.5, *(Mat*)src2, 0.5, 0.0, *(Mat*)dst, CV_32F);
    }

    void subtractOp(void* src1, void* src2, void* dst)
    {
        subtract(*(Mat*)src1, *(Mat*)src2, *(Mat*)dst, noArray(), CV_32F);
    }

    struct ComputeDataTerm_ParBody : public ParallelLoopBody
    {
        VariationalRefinementImpl *var;
        RedBlackBuffer *dW_u, *dW_v;
        int nstripes, stripe_sz;
        int h;
        bool red_pass;

        ComputeDataTerm_ParBody(VariationalRefinementImpl &_var, int _nstripes, int _h, RedBlackBuffer &_dW_u,
                                RedBlackBuffer &_dW_v, bool _red_pass);
        void operator()(const Range &range) const;
    };

    struct ComputeSmoothnessTermHorPass_ParBody : public ParallelLoopBody
    {
        VariationalRefinementImpl *var;
        RedBlackBuffer *W_u, *W_v, *curW_u, *curW_v;
        int nstripes, stripe_sz;
        int h;
        bool red_pass;

        ComputeSmoothnessTermHorPass_ParBody(VariationalRefinementImpl &_var, int _nstripes, int _h, RedBlackBuffer &W_u,
                                             RedBlackBuffer &_W_v, RedBlackBuffer &_tempW_u, RedBlackBuffer &_tempW_v,
                                             bool _red_pass);
        void operator()(const Range &range) const;
    };

    struct ComputeSmoothnessTermVertPass_ParBody : public ParallelLoopBody
    {
        VariationalRefinementImpl *var;
        RedBlackBuffer *W_u, *W_v;
        int nstripes, stripe_sz;
        int h;
        bool red_pass;

        ComputeSmoothnessTermVertPass_ParBody(VariationalRefinementImpl &_var, int _nstripes, int _h, RedBlackBuffer &W_u,
            RedBlackBuffer &_W_v,
            bool _red_pass);
        void operator()(const Range &range) const;
    };

    struct RedBlackSOR_ParBody : public ParallelLoopBody
    {
        VariationalRefinementImpl *var;
        RedBlackBuffer *dW_u, *dW_v;
        int nstripes, stripe_sz;
        int h;
        bool red_pass; // red of black pass

        RedBlackSOR_ParBody(VariationalRefinementImpl &_var, int _nstripes, int _h, RedBlackBuffer &_dW_u,
                            RedBlackBuffer &_dW_v, bool _red_pass);
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
}

/////////////////// RedBlackBuffer auxiliary functions ///////////////////

void VariationalRefinementImpl::splitCheckerboard(RedBlackBuffer &dst, Mat &src)
{
    // splits one buffer into two using a checkerboard pattern
    // assumes that enough memory is already allocated
    // adds repeated-border padding
    int buf_j, j;
    int buf_w = (int)ceil(src.cols / 2.0) + 2;
    for (int i = 0; i < src.rows; i++)
    {
        float *src_buf = src.ptr<float>(i);
        float *r_buf = dst.red.ptr<float>(i + 1);
        float *b_buf = dst.black.ptr<float>(i + 1);
        buf_j = 1;
        r_buf[0] = b_buf[0] = src_buf[0];
        if (i % 2 == 0)
        {
            for (j = 0; j < src.cols - 1; j += 2)
            {
                r_buf[buf_j] = src_buf[j];
                b_buf[buf_j] = src_buf[j + 1];
                buf_j++;
            }
            if (j < src.cols)
                r_buf[buf_j] = b_buf[buf_j] = src_buf[j];
            else
                j--;
        }
        else
        {
            for (j = 0; j < src.cols - 1; j += 2)
            {
                b_buf[buf_j] = src_buf[j];
                r_buf[buf_j] = src_buf[j + 1];
                buf_j++;
            }
            if (j < src.cols)
                r_buf[buf_j] = b_buf[buf_j] = src_buf[j];
            else
                j--;
        }
        r_buf[buf_w - 1] = b_buf[buf_w - 1] = src_buf[j];
    }
    {
        float *r_buf_border = dst.red.ptr<float>(dst.red.rows - 1);
        float *b_buf_border = dst.black.ptr<float>(dst.black.rows - 1);
        float *r_buf = dst.red.ptr<float>(dst.red.rows - 2);
        float *b_buf = dst.black.ptr<float>(dst.black.rows - 2);
        memcpy(r_buf_border, b_buf, buf_w * sizeof(float));
        memcpy(b_buf_border, r_buf, buf_w * sizeof(float));
    }
    {
        float *r_buf_border = dst.red.ptr<float>(0);
        float *b_buf_border = dst.black.ptr<float>(0);
        float *r_buf = dst.red.ptr<float>(1);
        float *b_buf = dst.black.ptr<float>(1);
        memcpy(r_buf_border, b_buf, buf_w * sizeof(float));
        memcpy(b_buf_border, r_buf, buf_w * sizeof(float));
    }
}

void VariationalRefinementImpl::mergeCheckerboard(Mat &dst, RedBlackBuffer &src)
{
    // merge two buffers into one using a checkerboard pattern
    // assumes that enough memory is already allocated
    int buf_j, j;
    for (int i = 0; i < dst.rows; i++)
    {
        float *src_r_buf = src.red.ptr<float>(i + 1);
        float *src_b_buf = src.black.ptr<float>(i + 1);
        float *dst_buf = dst.ptr<float>(i);
        buf_j = 1;

        if (i % 2 == 0)
        {
            for (j = 0; j < dst.cols - 1; j += 2)
            {
                dst_buf[j] = src_r_buf[buf_j];
                dst_buf[j + 1] = src_b_buf[buf_j];
                buf_j++;
            }
            if (j < dst.cols)
                dst_buf[j] = src_r_buf[buf_j];
        }
        else
        {
            for (j = 0; j < dst.cols - 1; j += 2)
            {
                dst_buf[j] = src_b_buf[buf_j];
                dst_buf[j + 1] = src_r_buf[buf_j];
                buf_j++;
            }
            if (j < dst.cols)
                dst_buf[j] = src_b_buf[buf_j];
        }
    }
}

void VariationalRefinementImpl::updateRepeatedBorders(RedBlackBuffer &dst)
{
    int buf_w = dst.red.cols;
    for (int i = 0; i < dst.red.rows - 2; i++)
    {
        float *r_buf = dst.red.ptr<float>(i + 1);
        float *b_buf = dst.black.ptr<float>(i + 1);

        if (i % 2 == 0)
        {
            b_buf[0] = r_buf[1];
            if (dst.red_even_len > dst.black_even_len)
                b_buf[dst.black_even_len + 1] = r_buf[dst.red_even_len];
            else
                r_buf[dst.red_even_len + 1] = b_buf[dst.black_even_len];
        }
        else
        {
            r_buf[0] = b_buf[1];
            if (dst.red_odd_len < dst.black_odd_len)
                r_buf[dst.red_odd_len + 1] = b_buf[dst.black_odd_len];
            else
                b_buf[dst.black_odd_len + 1] = r_buf[dst.red_odd_len];
        }
    }

    {
        float *r_buf_border = dst.red.ptr<float>(dst.red.rows - 1);
        float *b_buf_border = dst.black.ptr<float>(dst.black.rows - 1);
        float *r_buf = dst.red.ptr<float>(dst.red.rows - 2);
        float *b_buf = dst.black.ptr<float>(dst.black.rows - 2);
        memcpy(r_buf_border, b_buf, buf_w * sizeof(float));
        memcpy(b_buf_border, r_buf, buf_w * sizeof(float));
    }
    {
        float *r_buf_border = dst.red.ptr<float>(0);
        float *b_buf_border = dst.black.ptr<float>(0);
        float *r_buf = dst.red.ptr<float>(1);
        float *b_buf = dst.black.ptr<float>(1);
        memcpy(r_buf_border, b_buf, buf_w * sizeof(float));
        memcpy(b_buf_border, r_buf, buf_w * sizeof(float));
    }
}

void VariationalRefinementImpl::RedBlackBuffer::create(Size s)
{
    int w = (int)ceil(s.width / 2.0) + 2;
    red.create(s.height + 2, w);
    black.create(s.height + 2, w);

    if (s.width % 2 == 0)
        red_even_len = red_odd_len = black_even_len = black_odd_len = w - 2;
    else
    {
        red_even_len = black_odd_len = w - 2;
        red_odd_len = black_even_len = w - 3;
    }
}

void VariationalRefinementImpl::RedBlackBuffer::release()
{
    red.release();
    black.release();
}

//////////////////////////////////////////////////////////////////////////

VariationalRefinementImpl::ParallelOp_ParBody::ParallelOp_ParBody(VariationalRefinementImpl& _var, vector<Op> _ops, vector<void*>& _op1s, vector<void*>& _op2s, vector<void*>& _op3s):
    var(&_var), ops(_ops), op1s(_op1s), op2s(_op2s), op3s(_op3s)
{}

void VariationalRefinementImpl::ParallelOp_ParBody::operator() (const Range& range) const
{
    for (int i = range.start; i<range.end; i++)
        (var->*ops[i])(op1s[i], op2s[i], op3s[i]);
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
    double startTick, time;
    Size s = I0.size();
    A11.create(s);
    A12.create(s);
    A22.create(s);
    b1.create(s);
    b2.create(s);
    weights.create(s);
    weights.red.setTo(0.0f);
    weights.black.setTo(0.0f);
    tempW_u.create(s);
    tempW_v.create(s);
    dW_u.create(s);
    dW_v.create(s);
    W_u_rb.create(s);
    W_v_rb.create(s);

    Ix.create(s);
    Iy.create(s);
    Iz.create(s);
    Ixx.create(s);
    Ixy.create(s);
    Iyy.create(s);
    Ixz.create(s);
    Iyz.create(s);

    Ix_rb.create(s);
    Iy_rb.create(s);
    Iz_rb.create(s);
    Ixx_rb.create(s);
    Ixy_rb.create(s);
    Iyy_rb.create(s);
    Ixz_rb.create(s);
    Iyz_rb.create(s);

    mapX.create(s);
    mapY.create(s);
    
    Mat I1flt, warpedI;
    I1.convertTo(I1flt, CV_32F); // works slightly better with floating-point warps
    warpImage(warpedI, I1flt, W_u, W_v);

    // computing derivatives on the average of the current and warped next frame:
    Mat averagedI;

    {
        vector<void*> op1s; op1s.push_back((void*)&I0); op1s.push_back((void*)&warpedI);
        vector<void*> op2s; op2s.push_back((void*)&warpedI); op2s.push_back((void*)&I0);
        vector<void*> op3s; op3s.push_back((void*)&averagedI); op3s.push_back((void*)&Iz);
        vector<Op> ops; ops.push_back(&VariationalRefinementImpl::averageOp); ops.push_back(&VariationalRefinementImpl::subtractOp);
        parallel_for_(Range(0, 2), ParallelOp_ParBody(*this, ops, op1s, op2s, op3s));
    }

    splitCheckerboard(Iz_rb, Iz);

    {
        vector<void*> op1s; op1s.push_back((void*)&averagedI); op1s.push_back((void*)&averagedI);
        op1s.push_back((void*)&Iz); op1s.push_back((void*)&Iz);
        vector<void*> op2s; op2s.push_back((void*)&Ix); op2s.push_back((void*)&Iy);
        op2s.push_back((void*)&Ixz); op2s.push_back((void*)&Iyz);
        vector<void*> op3s; op3s.push_back((void*)&Ix_rb); op3s.push_back((void*)&Iy_rb);
        op3s.push_back((void*)&Ixz_rb); op3s.push_back((void*)&Iyz_rb);
        vector<Op> ops; ops.push_back(&VariationalRefinementImpl::gradHorizAndSplitOp); ops.push_back(&VariationalRefinementImpl::gradVertAndSplitOp);
        ops.push_back(&VariationalRefinementImpl::gradHorizAndSplitOp); ops.push_back(&VariationalRefinementImpl::gradVertAndSplitOp);
        parallel_for_(Range(0, 4), ParallelOp_ParBody(*this, ops, op1s, op2s, op3s));
    }

    {
        vector<void*> op1s; op1s.push_back((void*)&Ix); op1s.push_back((void*)&Ix);
        op1s.push_back((void*)&Iy);
        vector<void*> op2s; op2s.push_back((void*)&Ixx); op2s.push_back((void*)&Ixy);
        op2s.push_back((void*)&Iyy);
        vector<void*> op3s; op3s.push_back((void*)&Ixx_rb); op3s.push_back((void*)&Ixy_rb);
        op3s.push_back((void*)&Iyy_rb);
        vector<Op> ops; ops.push_back(&VariationalRefinementImpl::gradHorizAndSplitOp); ops.push_back(&VariationalRefinementImpl::gradVertAndSplitOp);
        ops.push_back(&VariationalRefinementImpl::gradVertAndSplitOp);
        parallel_for_(Range(0, 3), ParallelOp_ParBody(*this, ops, op1s, op2s, op3s));
    }
}

VariationalRefinementImpl::ComputeDataTerm_ParBody::ComputeDataTerm_ParBody(VariationalRefinementImpl &_var,
                                                                            int _nstripes, int _h,
                                                                            RedBlackBuffer &_dW_u,
                                                                            RedBlackBuffer &_dW_v, bool _red_pass)
    : var(&_var), nstripes(_nstripes), h(_h), dW_u(&_dW_u), dW_v(&_dW_v), red_pass(_red_pass)
{
    stripe_sz = (int)ceil(h / (double)nstripes);
}

void VariationalRefinementImpl::ComputeDataTerm_ParBody::operator()(const Range &range) const
{
    /*Using robust data term based on color and gradient constancy assumptions*/

    /*In this function we compute linear system coefficients
    A11,A12,A22,b1,b1 based on the data term */

    int start_i = min(range.start * stripe_sz, h);
    int end_i = min(range.end * stripe_sz, h);

    const float zeta_squared = var->zeta * var->zeta;
    const float epsilon_squared = var->epsilon * var->epsilon;
    const float gamma2 = var->gamma / 2;
    const float delta2 = var->delta / 2;

    const float *pIx, *pIy, *pIz;
    const float *pIxx, *pIxy, *pIyy, *pIxz, *pIyz;
    const float *pdU, *pdV;
    float *pa11, *pa12, *pa22, *pb1, *pb2;

    float derivNorm, derivNorm2;
    float Ik1z, Ik1zx, Ik1zy;
    float weight;
    int len;
    for (int i = start_i; i < end_i; i++)
    {
        if (red_pass)
        {
            pIx = var->Ix_rb.red.ptr<float>(i + 1) + 1;
            pIy = var->Iy_rb.red.ptr<float>(i + 1) + 1;
            pIz = var->Iz_rb.red.ptr<float>(i + 1) + 1;
            pIxx = var->Ixx_rb.red.ptr<float>(i + 1) + 1;
            pIxy = var->Ixy_rb.red.ptr<float>(i + 1) + 1;
            pIyy = var->Iyy_rb.red.ptr<float>(i + 1) + 1;
            pIxz = var->Ixz_rb.red.ptr<float>(i + 1) + 1;
            pIyz = var->Iyz_rb.red.ptr<float>(i + 1) + 1;
            pa11 = var->A11.red.ptr<float>(i + 1) + 1;
            pa12 = var->A12.red.ptr<float>(i + 1) + 1;
            pa22 = var->A22.red.ptr<float>(i + 1) + 1;
            pb1 = var->b1.red.ptr<float>(i + 1) + 1;
            pb2 = var->b2.red.ptr<float>(i + 1) + 1;
            pdU = dW_u->red.ptr<float>(i + 1) + 1;
            pdV = dW_v->red.ptr<float>(i + 1) + 1;
            if (i % 2 == 0)
                len = var->Ix_rb.red_even_len;
            else
                len = var->Ix_rb.red_odd_len;
        }
        else
        {
            pIx = var->Ix_rb.black.ptr<float>(i + 1) + 1;
            pIy = var->Iy_rb.black.ptr<float>(i + 1) + 1;
            pIz = var->Iz_rb.black.ptr<float>(i + 1) + 1;
            pIxx = var->Ixx_rb.black.ptr<float>(i + 1) + 1;
            pIxy = var->Ixy_rb.black.ptr<float>(i + 1) + 1;
            pIyy = var->Iyy_rb.black.ptr<float>(i + 1) + 1;
            pIxz = var->Ixz_rb.black.ptr<float>(i + 1) + 1;
            pIyz = var->Iyz_rb.black.ptr<float>(i + 1) + 1;
            pa11 = var->A11.black.ptr<float>(i + 1) + 1;
            pa12 = var->A12.black.ptr<float>(i + 1) + 1;
            pa22 = var->A22.black.ptr<float>(i + 1) + 1;
            pb1 = var->b1.black.ptr<float>(i + 1) + 1;
            pb2 = var->b2.black.ptr<float>(i + 1) + 1;
            pdU = dW_u->black.ptr<float>(i + 1) + 1;
            pdV = dW_v->black.ptr<float>(i + 1) + 1;
            if (i % 2 == 0)
                len = var->Ix_rb.black_even_len;
            else
                len = var->Ix_rb.black_odd_len;
        }
        int j = 0;
#ifdef CV_SIMD128
        v_float32x4 zeta_vec = v_setall_f32(zeta_squared);
        v_float32x4 eps_vec = v_setall_f32(epsilon_squared);
        v_float32x4 delta_vec = v_setall_f32(delta2);
        v_float32x4 gamma_vec = v_setall_f32(gamma2);
        v_float32x4 zero_vec = v_setall_f32(0.0f);
        v_float32x4 pIx_vec, pIy_vec, pIz_vec, pdU_vec, pdV_vec;
        v_float32x4 pIxx_vec, pIxy_vec, pIyy_vec, pIxz_vec, pIyz_vec;
        v_float32x4 derivNorm_vec, derivNorm2_vec, weight_vec;
        v_float32x4 Ik1z_vec, Ik1zx_vec, Ik1zy_vec;
        v_float32x4 pa11_vec, pa12_vec, pa22_vec, pb1_vec, pb2_vec;

        for (; j < len - 3; j+=4)
        {
            pIx_vec = v_load(pIx + j);
            pIy_vec = v_load(pIy + j);
            pIz_vec = v_load(pIz + j);
            pdU_vec = v_load(pdU + j);
            pdV_vec = v_load(pdV + j);

            derivNorm_vec = pIx_vec*pIx_vec + pIy_vec*pIy_vec + zeta_vec;
            Ik1z_vec = pIz_vec + pIx_vec*pdU_vec + pIy_vec*pdV_vec;
            weight_vec = delta_vec / v_sqrt(Ik1z_vec*Ik1z_vec / derivNorm_vec + eps_vec);

            pa11_vec = weight_vec * (pIx_vec*pIx_vec / derivNorm_vec) + zeta_vec;
            pa12_vec = weight_vec * (pIx_vec*pIy_vec / derivNorm_vec);
            pa22_vec = weight_vec * (pIy_vec*pIy_vec / derivNorm_vec) + zeta_vec;
            pb1_vec = zero_vec-weight_vec * (pIz_vec*pIx_vec / derivNorm_vec);
            pb2_vec = zero_vec -weight_vec * (pIz_vec*pIy_vec / derivNorm_vec);

            pIxx_vec = v_load(pIxx + j);
            pIxy_vec = v_load(pIxy + j);
            pIyy_vec = v_load(pIyy + j);
            pIxz_vec = v_load(pIxz + j);
            pIyz_vec = v_load(pIyz + j);

            derivNorm_vec = pIxx_vec * pIxx_vec + pIxy_vec * pIxy_vec + zeta_vec;
            derivNorm2_vec = pIyy_vec * pIyy_vec + pIxy_vec * pIxy_vec + zeta_vec;
            Ik1zx_vec = pIxz_vec + pIxx_vec * pdU_vec + pIxy_vec * pdV_vec;
            Ik1zy_vec = pIyz_vec + pIxy_vec * pdU_vec + pIyy_vec * pdV_vec;
            weight_vec = gamma_vec / v_sqrt(Ik1zx_vec * Ik1zx_vec / derivNorm_vec + Ik1zy_vec * Ik1zy_vec / derivNorm2_vec + eps_vec);

            pa11_vec += weight_vec * (pIxx_vec * pIxx_vec / derivNorm_vec + pIxy_vec * pIxy_vec / derivNorm2_vec);
            pa12_vec += weight_vec * (pIxx_vec * pIxy_vec / derivNorm_vec + pIxy_vec * pIyy_vec / derivNorm2_vec);
            pa22_vec += weight_vec * (pIxy_vec * pIxy_vec / derivNorm_vec + pIyy_vec * pIyy_vec / derivNorm2_vec);
            pb1_vec -= weight_vec * (pIxx_vec * pIxz_vec / derivNorm_vec + pIxy_vec * pIyz_vec / derivNorm2_vec);
            pb2_vec -= weight_vec * (pIxy_vec * pIxz_vec / derivNorm_vec + pIyy_vec * pIyz_vec / derivNorm2_vec);

            v_store(pa11 + j, pa11_vec);
            v_store(pa12 + j, pa12_vec);
            v_store(pa22 + j, pa22_vec);
            v_store(pb1 + j, pb1_vec);
            v_store(pb2 + j, pb2_vec);
        }
#endif
        for (; j < len; j++)
        {
            // Step 1: color contancy
            // Normalization factor:
            derivNorm = pIx[j] * pIx[j] + pIy[j] * pIy[j] + zeta_squared;
            // Color constancy penalty (computed by Taylor expansion):
            Ik1z = pIz[j] + pIx[j] * pdU[j] + pIy[j] * pdV[j];
            // Weight of the color constancy term in the current fixed-point iteration:
            weight = delta2 / sqrt(Ik1z * Ik1z / derivNorm + epsilon_squared);
            // Add respective color constancy components to the linear sustem coefficients:
            pa11[j] = weight * (pIx[j] * pIx[j] / derivNorm) + zeta_squared;
            pa12[j] = weight * (pIx[j] * pIy[j] / derivNorm);
            pa22[j] = weight * (pIy[j] * pIy[j] / derivNorm) + zeta_squared;
            pb1[j] = -weight * (pIz[j] * pIx[j] / derivNorm);
            pb2[j] = -weight * (pIz[j] * pIy[j] / derivNorm);

            // Step 2: gradient contancy
            // Normalization factor for x gradient:
            derivNorm = pIxx[j] * pIxx[j] + pIxy[j] * pIxy[j] + zeta_squared;
            // Normalization factor for y gradient:
            derivNorm2 = pIyy[j] * pIyy[j] + pIxy[j] * pIxy[j] + zeta_squared;
            // Gradient constancy penalties (computed by Taylor expansion):
            Ik1zx = pIxz[j] + pIxx[j] * pdU[j] + pIxy[j] * pdV[j];
            Ik1zy = pIyz[j] + pIxy[j] * pdU[j] + pIyy[j] * pdV[j];

            // Weight of the gradient constancy term in the current fixed-point iteration:
            weight = gamma2 / sqrt(Ik1zx * Ik1zx / derivNorm + Ik1zy * Ik1zy / derivNorm2 + epsilon_squared);
            // Add respective gradient constancy components to the linear system coefficients:
            pa11[j] += weight * (pIxx[j] * pIxx[j] / derivNorm + pIxy[j] * pIxy[j] / derivNorm2);
            pa12[j] += weight * (pIxx[j] * pIxy[j] / derivNorm + pIxy[j] * pIyy[j] / derivNorm2);
            pa22[j] += weight * (pIxy[j] * pIxy[j] / derivNorm + pIyy[j] * pIyy[j] / derivNorm2);
            pb1[j] += -weight * (pIxx[j] * pIxz[j] / derivNorm + pIxy[j] * pIyz[j] / derivNorm2);
            pb2[j] += -weight * (pIxy[j] * pIxz[j] / derivNorm + pIyy[j] * pIyz[j] / derivNorm2);
        }
    }
}

VariationalRefinementImpl::ComputeSmoothnessTermHorPass_ParBody::ComputeSmoothnessTermHorPass_ParBody(
  VariationalRefinementImpl &_var, int _nstripes, int _h, RedBlackBuffer &_W_u, RedBlackBuffer &_W_v,
  RedBlackBuffer &_tempW_u, RedBlackBuffer &_tempW_v, bool _red_pass)
    : var(&_var), nstripes(_nstripes), h(_h), W_u(&_W_u), W_v(&_W_v), curW_u(&_tempW_u), curW_v(&_tempW_v),
      red_pass(_red_pass)
{
    stripe_sz = (int)ceil(h / (double)nstripes);
}

void VariationalRefinementImpl::ComputeSmoothnessTermHorPass_ParBody::operator()(const Range &range) const
{
    /*Using robust penalty on flow gradient*/

    /*In this function we update b1, b2, A11, A22 coefficients of the linear system
    and compute smoothness term weights for the current fixed-point iteration */

    int start_i = min(range.start * stripe_sz, h);
    int end_i = min(range.end * stripe_sz, h);

    const float epsilon_squared = var->epsilon * var->epsilon;
    const float alpha2 = var->alpha / 2;
    float *pWeight;
    float *pA_u, *pA_u_next;
    float *pA_v, *pA_v_next;
    float *pB_u, *pB_u_next;
    float *pB_v, *pB_v_next;
    const float *cW_u, *cW_u_next, *cW_u_next_row;
    const float *cW_v, *cW_v_next, *cW_v_next_row;
    const float *pW_u, *pW_u_next;
    const float *pW_v, *pW_v_next;
    float ux, uy, vx, vy;
    int len;
    bool touches_right_border = true;

    for (int i = start_i; i < end_i; i++)
    {
        if (red_pass)
        {
            pWeight = var->weights.red.ptr<float>(i + 1) + 1;
            pA_u = var->A11.red.ptr<float>(i + 1) + 1;
            pB_u = var->b1.red.ptr<float>(i + 1) + 1;
            cW_u = curW_u->red.ptr<float>(i + 1) + 1;
            pW_u = W_u->red.ptr<float>(i + 1) + 1;
            pA_v = var->A22.red.ptr<float>(i + 1) + 1;
            pB_v = var->b2.red.ptr<float>(i + 1) + 1;
            cW_v = curW_v->red.ptr<float>(i + 1) + 1;
            pW_v = W_v->red.ptr<float>(i + 1) + 1;

            cW_u_next_row = curW_u->black.ptr<float>(i + 2) + 1;
            cW_v_next_row = curW_v->black.ptr<float>(i + 2) + 1;

            if (i % 2 == 0)
            {
                pA_u_next = var->A11.black.ptr<float>(i + 1) + 1;
                pB_u_next = var->b1.black.ptr<float>(i + 1) + 1;
                cW_u_next = curW_u->black.ptr<float>(i + 1) + 1;
                pW_u_next = W_u->black.ptr<float>(i + 1) + 1;
                pA_v_next = var->A22.black.ptr<float>(i + 1) + 1;
                pB_v_next = var->b2.black.ptr<float>(i + 1) + 1;
                cW_v_next = curW_v->black.ptr<float>(i + 1) + 1;
                pW_v_next = W_v->black.ptr<float>(i + 1) + 1;
                len = var->A11.red_even_len;
                if (var->A11.red_even_len > var->A11.red_odd_len)
                    touches_right_border = true;
                else
                    touches_right_border = false;
            }
            else
            {
                pA_u_next = var->A11.black.ptr<float>(i + 1) + 2;
                pB_u_next = var->b1.black.ptr<float>(i + 1) + 2;
                cW_u_next = curW_u->black.ptr<float>(i + 1) + 2;
                pW_u_next = W_u->black.ptr<float>(i + 1) + 2;
                pA_v_next = var->A22.black.ptr<float>(i + 1) + 2;
                pB_v_next = var->b2.black.ptr<float>(i + 1) + 2;
                cW_v_next = curW_v->black.ptr<float>(i + 1) + 2;
                pW_v_next = W_v->black.ptr<float>(i + 1) + 2;
                len = var->A11.red_odd_len;
                if (var->A11.red_even_len > var->A11.red_odd_len)
                    touches_right_border = false;
                else
                    touches_right_border = true;
            }
        }
        else
        {
            pWeight = var->weights.black.ptr<float>(i + 1) + 1;
            pA_u = var->A11.black.ptr<float>(i + 1) + 1;
            pB_u = var->b1.black.ptr<float>(i + 1) + 1;
            cW_u = curW_u->black.ptr<float>(i + 1) + 1;
            pW_u = W_u->black.ptr<float>(i + 1) + 1;
            pA_v = var->A22.black.ptr<float>(i + 1) + 1;
            pB_v = var->b2.black.ptr<float>(i + 1) + 1;
            cW_v = curW_v->black.ptr<float>(i + 1) + 1;
            pW_v = W_v->black.ptr<float>(i + 1) + 1;

            cW_u_next_row = curW_u->red.ptr<float>(i + 2) + 1;
            cW_v_next_row = curW_v->red.ptr<float>(i + 2) + 1;

            if (i % 2 == 0)
            {
                pA_u_next = var->A11.red.ptr<float>(i + 1) + 2;
                pB_u_next = var->b1.red.ptr<float>(i + 1) + 2;
                cW_u_next = curW_u->red.ptr<float>(i + 1) + 2;
                pW_u_next = W_u->red.ptr<float>(i + 1) + 2;
                pA_v_next = var->A22.red.ptr<float>(i + 1) + 2;
                pB_v_next = var->b2.red.ptr<float>(i + 1) + 2;
                cW_v_next = curW_v->red.ptr<float>(i + 1) + 2;
                pW_v_next = W_v->red.ptr<float>(i + 1) + 2;
                len = var->A11.black_even_len;
                if (var->A11.black_even_len < var->A11.black_odd_len)
                    touches_right_border = false;
                else
                    touches_right_border = true;
            }
            else
            {
                pA_u_next = var->A11.red.ptr<float>(i + 1) + 1;
                pB_u_next = var->b1.red.ptr<float>(i + 1) + 1;
                cW_u_next = curW_u->red.ptr<float>(i + 1) + 1;
                pW_u_next = W_u->red.ptr<float>(i + 1) + 1;
                pA_v_next = var->A22.red.ptr<float>(i + 1) + 1;
                pB_v_next = var->b2.red.ptr<float>(i + 1) + 1;
                cW_v_next = curW_v->red.ptr<float>(i + 1) + 1;
                pW_v_next = W_v->red.ptr<float>(i + 1) + 1;
                len = var->A11.black_odd_len;
                if (var->A11.black_even_len < var->A11.black_odd_len)
                    touches_right_border = true;
                else
                    touches_right_border = false;
            }
        }

#define COMPUTE                                                                                                        \
    /*gradients for the flow on the current fixed-point iteration:*/                                                   \
    ux = cW_u_next[j] - cW_u[j];                                                                                       \
    vx = cW_v_next[j] - cW_v[j];                                                                                       \
    uy = cW_u_next_row[j] - cW_u[j];                                                                                   \
    vy = cW_v_next_row[j] - cW_v[j];                                                                                   \
    /* weight of the smoothness term in the current fixed-point iteration:*/                                           \
    pWeight[j] = alpha2 / sqrt(ux * ux + vx * vx + uy * uy + vy * vy + epsilon_squared);                               \
    /* gradients for initial raw flow multiplied by weight:*/                                                          \
    ux = pWeight[j] * (pW_u_next[j] - pW_u[j]);                                                                        \
    vx = pWeight[j] * (pW_v_next[j] - pW_v[j]);

#define UPDATE_HOR                                                                                                     \
    pB_u[j] += ux;                                                                                                     \
    pA_u[j] += pWeight[j];                                                                                             \
    pB_v[j] += vx;                                                                                                     \
    pA_v[j] += pWeight[j];                                                                                             \
    pB_u_next[j] -= ux;                                                                                                \
    pA_u_next[j] += pWeight[j];                                                                                        \
    pB_v_next[j] -= vx;                                                                                                \
    pA_v_next[j] += pWeight[j];

        int j = 0;
#ifdef CV_SIMD128
        v_float32x4 alpha2_vec = v_setall_f32(alpha2);
        v_float32x4 eps_vec = v_setall_f32(epsilon_squared);
        v_float32x4 cW_u_vec, cW_v_vec;
        v_float32x4 pWeight_vec, ux_vec, vx_vec, uy_vec, vy_vec;

        for (; j < len - 4; j += 4)
        {
            cW_u_vec = v_load(cW_u + j);
            cW_v_vec = v_load(cW_v + j);

            ux_vec = v_load(cW_u_next + j) - cW_u_vec;
            vx_vec = v_load(cW_v_next + j) - cW_v_vec;
            uy_vec = v_load(cW_u_next_row + j) - cW_u_vec;
            vy_vec = v_load(cW_v_next_row + j) - cW_v_vec;
            pWeight_vec =
              alpha2_vec / v_sqrt(ux_vec * ux_vec + vx_vec * vx_vec + uy_vec * uy_vec + vy_vec * vy_vec + eps_vec);
            v_store(pWeight + j, pWeight_vec);

            ux_vec = pWeight_vec * (v_load(pW_u_next + j) - v_load(pW_u + j));
            vx_vec = pWeight_vec * (v_load(pW_v_next + j) - v_load(pW_v + j));

            v_store(pA_u + j, v_load(pA_u + j) + pWeight_vec);
            v_store(pA_v + j, v_load(pA_v + j) + pWeight_vec);
            v_store(pB_u + j, v_load(pB_u + j) + ux_vec);
            v_store(pB_v + j, v_load(pB_v + j) + vx_vec);

            v_store(pA_u_next + j, v_load(pA_u_next + j) + pWeight_vec);
            v_store(pA_v_next + j, v_load(pA_v_next + j) + pWeight_vec);
            v_store(pB_u_next + j, v_load(pB_u_next + j) - ux_vec);
            v_store(pB_v_next + j, v_load(pB_v_next + j) - vx_vec);
        }
#endif
        for (; j < len - 1; j++)
        {
            COMPUTE;
            UPDATE_HOR;
        }
        if (touches_right_border)
        {
            COMPUTE;
        }
        else
        {
            COMPUTE;
            UPDATE_HOR;
        }
    }
}

VariationalRefinementImpl::ComputeSmoothnessTermVertPass_ParBody::ComputeSmoothnessTermVertPass_ParBody(
    VariationalRefinementImpl &_var, int _nstripes, int _h, RedBlackBuffer &_W_u, RedBlackBuffer &_W_v, bool _red_pass)
    : var(&_var), nstripes(_nstripes), h(_h-1), W_u(&_W_u), W_v(&_W_v),
    red_pass(_red_pass)
{
    stripe_sz = (int)ceil(h / (double)nstripes);
}

void VariationalRefinementImpl::ComputeSmoothnessTermVertPass_ParBody::operator()(const Range &range) const
{
    int start_i = min(range.start * stripe_sz, h);
    int end_i = min(range.end * stripe_sz, h);

    const float epsilon_squared = var->epsilon * var->epsilon;
    const float alpha2 = var->alpha / 2;
    float *pWeight;
    float *pA_u, *pA_u_next_row;
    float *pA_v, *pA_v_next_row;
    float *pB_u, *pB_u_next_row;
    float *pB_v, *pB_v_next_row;
    const float *pW_u, *pW_u_next_row;
    const float *pW_v, *pW_v_next_row;
    float vy, uy;
    int len;

    for (int i = start_i; i < end_i; i++)
    {
        if (red_pass)
        {
            pWeight = var->weights.red.ptr<float>(i + 1) + 1;
            pA_u = var->A11.red.ptr<float>(i + 1) + 1;
            pB_u = var->b1.red.ptr<float>(i + 1) + 1;
            pW_u = W_u->red.ptr<float>(i + 1) + 1;
            pA_v = var->A22.red.ptr<float>(i + 1) + 1;
            pB_v = var->b2.red.ptr<float>(i + 1) + 1;
            pW_v = W_v->red.ptr<float>(i + 1) + 1;

            pA_u_next_row = var->A11.black.ptr<float>(i + 2) + 1;
            pB_u_next_row = var->b1.black.ptr<float>(i + 2) + 1;
            pW_u_next_row = W_u->black.ptr<float>(i + 2) + 1;
            pA_v_next_row = var->A22.black.ptr<float>(i + 2) + 1;
            pB_v_next_row = var->b2.black.ptr<float>(i + 2) + 1;
            pW_v_next_row = W_v->black.ptr<float>(i + 2) + 1;

            if (i % 2 == 0)
                len = var->A11.red_even_len;
            else
                len = var->A11.red_odd_len;
        }
        else
        {
            pWeight = var->weights.black.ptr<float>(i + 1) + 1;
            pA_u = var->A11.black.ptr<float>(i + 1) + 1;
            pB_u = var->b1.black.ptr<float>(i + 1) + 1;
            pW_u = W_u->black.ptr<float>(i + 1) + 1;
            pA_v = var->A22.black.ptr<float>(i + 1) + 1;
            pB_v = var->b2.black.ptr<float>(i + 1) + 1;
            pW_v = W_v->black.ptr<float>(i + 1) + 1;

            pA_u_next_row = var->A11.red.ptr<float>(i + 2) + 1;
            pB_u_next_row = var->b1.red.ptr<float>(i + 2) + 1;
            pW_u_next_row = W_u->red.ptr<float>(i + 2) + 1;
            pA_v_next_row = var->A22.red.ptr<float>(i + 2) + 1;
            pB_v_next_row = var->b2.red.ptr<float>(i + 2) + 1;
            pW_v_next_row = W_v->red.ptr<float>(i + 2) + 1;

            if (i % 2 == 0)
                len = var->A11.black_even_len;
            else
                len = var->A11.black_odd_len;
        }
        int j = 0;
#ifdef CV_SIMD128
        v_float32x4 pWeight_vec, uy_vec, vy_vec;
        for (; j < len - 3; j += 4)
        {
            pWeight_vec = v_load(pWeight + j);
            uy_vec = pWeight_vec * (v_load(pW_u_next_row + j) - v_load(pW_u + j));
            vy_vec = pWeight_vec * (v_load(pW_v_next_row + j) - v_load(pW_v + j));

            v_store(pA_u + j, v_load(pA_u + j) + pWeight_vec);
            v_store(pA_v + j, v_load(pA_v + j) + pWeight_vec);
            v_store(pB_u + j, v_load(pB_u + j) + uy_vec);
            v_store(pB_v + j, v_load(pB_v + j) + vy_vec);

            v_store(pA_u_next_row + j, v_load(pA_u_next_row + j) + pWeight_vec);
            v_store(pA_v_next_row + j, v_load(pA_v_next_row + j) + pWeight_vec);
            v_store(pB_u_next_row + j, v_load(pB_u_next_row + j) - uy_vec);
            v_store(pB_v_next_row + j, v_load(pB_v_next_row + j) - vy_vec);
        }
#endif
        for (; j < len; j++)
        {
            uy = pWeight[j] * (pW_u_next_row[j] - pW_u[j]);
            vy = pWeight[j] * (pW_v_next_row[j] - pW_v[j]);
            pB_u[j] += uy;
            pA_u[j] += pWeight[j];
            pB_v[j] += vy;
            pA_v[j] += pWeight[j];
            pB_u_next_row[j] -= uy;
            pA_u_next_row[j] += pWeight[j];
            pB_v_next_row[j] -= vy;
            pA_v_next_row[j] += pWeight[j];
        }
    }
}

VariationalRefinementImpl::RedBlackSOR_ParBody::RedBlackSOR_ParBody(VariationalRefinementImpl &_var, int _nstripes,
                                                                    int _h, RedBlackBuffer &_dW_u,
                                                                    RedBlackBuffer &_dW_v, bool _red_pass)
    : var(&_var), nstripes(_nstripes), h(_h), dW_u(&_dW_u), dW_v(&_dW_v), red_pass(_red_pass)
{
    stripe_sz = (int)ceil(h / (double)nstripes);
}

void VariationalRefinementImpl::RedBlackSOR_ParBody::operator()(const Range &range) const
{
    int start = min(range.start * stripe_sz, h);
    int end = min(range.end * stripe_sz, h);

    float *pa11, *pa12, *pa22, *pb1, *pb2, *pW, *pdu, *pdv;
    float *pW_next, *pdu_next, *pdv_next;
    float *pW_prev_row, *pdu_prev_row, *pdv_prev_row;
    float *pdu_next_row, *pdv_next_row;

    float sigmaU, sigmaV;
    float omega = var->omega;
    int j,len;
    for (int i = start; i < end; i++)
    {
        if (red_pass)
        {
            pW = var->weights.red.ptr<float>(i + 1) + 1;
            pa11 = var->A11.red.ptr<float>(i + 1) + 1;
            pa12 = var->A12.red.ptr<float>(i + 1) + 1;
            pa22 = var->A22.red.ptr<float>(i + 1) + 1;
            pb1 = var->b1.red.ptr<float>(i + 1) + 1;
            pb2 = var->b2.red.ptr<float>(i + 1) + 1;
            pdu = dW_u->red.ptr<float>(i + 1) + 1;
            pdv = dW_v->red.ptr<float>(i + 1) + 1;

            pdu_next_row = dW_u->black.ptr<float>(i + 2) + 1;
            pdv_next_row = dW_v->black.ptr<float>(i + 2) + 1;

            pW_prev_row = var->weights.black.ptr<float>(i) + 1;
            pdu_prev_row = dW_u->black.ptr<float>(i) + 1;
            pdv_prev_row = dW_v->black.ptr<float>(i) + 1;

            if (i % 2 == 0)
            {
                pW_next = var->weights.black.ptr<float>(i + 1) + 1;
                pdu_next = dW_u->black.ptr<float>(i + 1) + 1;
                pdv_next = dW_v->black.ptr<float>(i + 1) + 1;
                len = var->A11.red_even_len;
            }
            else
            {
                pW_next = var->weights.black.ptr<float>(i + 1) + 2;
                pdu_next = dW_u->black.ptr<float>(i + 1) + 2;
                pdv_next = dW_v->black.ptr<float>(i + 1) + 2;
                len = var->A11.red_odd_len;
            }
        }
        else
        {
            pW = var->weights.black.ptr<float>(i + 1) + 1;
            pa11 = var->A11.black.ptr<float>(i + 1) + 1;
            pa12 = var->A12.black.ptr<float>(i + 1) + 1;
            pa22 = var->A22.black.ptr<float>(i + 1) + 1;
            pb1 = var->b1.black.ptr<float>(i + 1) + 1;
            pb2 = var->b2.black.ptr<float>(i + 1) + 1;
            pdu = dW_u->black.ptr<float>(i + 1) + 1;
            pdv = dW_v->black.ptr<float>(i + 1) + 1;

            pdu_next_row = dW_u->red.ptr<float>(i + 2) + 1;
            pdv_next_row = dW_v->red.ptr<float>(i + 2) + 1;

            pW_prev_row = var->weights.red.ptr<float>(i) + 1;
            pdu_prev_row = dW_u->red.ptr<float>(i) + 1;
            pdv_prev_row = dW_v->red.ptr<float>(i) + 1;

            if (i % 2 == 0)
            {
                pW_next = var->weights.red.ptr<float>(i + 1) + 2;
                pdu_next = dW_u->red.ptr<float>(i + 1) + 2;
                pdv_next = dW_v->red.ptr<float>(i + 1) + 2;
                len = var->A11.black_even_len;
            }
            else
            {
                pW_next = var->weights.red.ptr<float>(i + 1) + 1;
                pdu_next = dW_u->red.ptr<float>(i + 1) + 1;
                pdv_next = dW_v->red.ptr<float>(i + 1) + 1;
                len = var->A11.black_odd_len;
            }
        }
        j = 0;
#ifdef CV_SIMD128
        v_float32x4 pW_prev_vec = v_setall_f32(pW_next[-1]);
        v_float32x4 pdu_prev_vec = v_setall_f32(pdu_next[-1]);
        v_float32x4 pdv_prev_vec = v_setall_f32(pdv_next[-1]);
        v_float32x4 omega_vec = v_setall_f32(omega);
        v_float32x4 pW_vec, pW_next_vec, pW_prev_row_vec;
        v_float32x4 pdu_next_vec, pdu_prev_row_vec, pdu_next_row_vec;
        v_float32x4 pdv_next_vec, pdv_prev_row_vec, pdv_next_row_vec;
        v_float32x4 pW_shifted_vec, pdu_shifted_vec, pdv_shifted_vec;
        v_float32x4 pa12_vec, sigmaU_vec, sigmaV_vec, pdu_vec, pdv_vec;
        for (; j < len - 3; j += 4)
        {
            pW_vec = v_load(pW + j);
            pW_next_vec = v_load(pW_next + j);
            pW_prev_row_vec = v_load(pW_prev_row + j);
            pdu_next_vec = v_load(pdu_next + j);
            pdu_prev_row_vec = v_load(pdu_prev_row + j);
            pdu_next_row_vec = v_load(pdu_next_row + j);
            pdv_next_vec = v_load(pdv_next + j);
            pdv_prev_row_vec = v_load(pdv_prev_row + j);
            pdv_next_row_vec = v_load(pdv_next_row + j);
            pa12_vec = v_load(pa12 + j);
            pW_shifted_vec = v_reinterpret_as_f32(
              v_extract<3, v_int32x4>(v_reinterpret_as_s32(pW_prev_vec), v_reinterpret_as_s32(pW_next_vec)));
            pdu_shifted_vec = v_reinterpret_as_f32(
              v_extract<3, v_int32x4>(v_reinterpret_as_s32(pdu_prev_vec), v_reinterpret_as_s32(pdu_next_vec)));
            pdv_shifted_vec = v_reinterpret_as_f32(
              v_extract<3, v_int32x4>(v_reinterpret_as_s32(pdv_prev_vec), v_reinterpret_as_s32(pdv_next_vec)));

            sigmaU_vec = pW_shifted_vec * pdu_shifted_vec + pW_vec * pdu_next_vec + pW_prev_row_vec * pdu_prev_row_vec +
                         pW_vec * pdu_next_row_vec;
            sigmaV_vec = pW_shifted_vec * pdv_shifted_vec + pW_vec * pdv_next_vec + pW_prev_row_vec * pdv_prev_row_vec +
                         pW_vec * pdv_next_row_vec;

            pdu_vec = v_load(pdu + j);
            pdv_vec = v_load(pdv + j);
            pdu_vec += omega_vec * ((sigmaU_vec + v_load(pb1 + j) - pdv_vec * pa12_vec) / v_load(pa11 + j) - pdu_vec);
            pdv_vec += omega_vec * ((sigmaV_vec + v_load(pb2 + j) - pdu_vec * pa12_vec) / v_load(pa22 + j) - pdv_vec);
            v_store(pdu + j, pdu_vec);
            v_store(pdv + j, pdv_vec);

            pW_prev_vec = pW_next_vec;
            pdu_prev_vec = pdu_next_vec;
            pdv_prev_vec = pdv_next_vec;
        }
#endif
        for (; j < len; j++)
        {
            sigmaU = pW_next[j - 1] * pdu_next[j - 1] + pW[j] * pdu_next[j] + pW_prev_row[j] * pdu_prev_row[j] +
                     pW[j] * pdu_next_row[j];
            sigmaV = pW_next[j - 1] * pdv_next[j - 1] + pW[j] * pdv_next[j] + pW_prev_row[j] * pdv_prev_row[j] +
                     pW[j] * pdv_next_row[j];
            pdu[j] += omega * ((sigmaU + pb1[j] - pdv[j] * pa12[j]) / pa11[j] - pdu[j]);
            pdv[j] += omega * ((sigmaV + pb2[j] - pdu[j] * pa12[j]) / pa22[j] - pdv[j]);
        }
    }
}

void VariationalRefinementImpl::calc(InputArray I0, InputArray I1, InputOutputArray flow)
{
    CV_Assert(!I0.empty() && I0.depth() == CV_8U && I0.channels() == 1);
    CV_Assert(!I1.empty() && I1.depth() == CV_8U && I1.channels() == 1);
    CV_Assert(I0.sameSize(I1));
    CV_Assert(!flow.empty() && flow.depth() == CV_32F && flow.channels() == 2);
    CV_Assert(I0.sameSize(flow));

    Mat uv[2];
    Mat &flowMat = flow.getMatRef();
    split(flowMat, uv);
    calcUV(I0, I1, uv[0], uv[1]);
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

    int num_stripes = getNumThreads();
    Mat I0Mat = I0.getMat();
    Mat I1Mat = I1.getMat();
    Mat &W_u = flow_u.getMatRef();
    Mat &W_v = flow_v.getMatRef();
    prepareBuffers(I0Mat, I1Mat, W_u, W_v);

    splitCheckerboard(W_u_rb, W_u);
    splitCheckerboard(W_v_rb, W_v);
    W_u_rb.red.copyTo(tempW_u.red);
    W_u_rb.black.copyTo(tempW_u.black);
    W_v_rb.red.copyTo(tempW_v.red);
    W_v_rb.black.copyTo(tempW_v.black);
    dW_u.red.setTo(0.0f);
    dW_u.black.setTo(0.0f);
    dW_v.red.setTo(0.0f);
    dW_v.black.setTo(0.0f);
    

    for (int i = 0; i < fixedPointIterations; i++)
    {
        parallel_for_(Range(0, num_stripes), ComputeDataTerm_ParBody(*this, num_stripes, I0Mat.rows, dW_u, dW_v, true));
        parallel_for_(Range(0, num_stripes),
                      ComputeDataTerm_ParBody(*this, num_stripes, I0Mat.rows, dW_u, dW_v, false));

        parallel_for_(Range(0, num_stripes), ComputeSmoothnessTermHorPass_ParBody(*this, num_stripes, I0Mat.rows, W_u_rb,
                                                                           W_v_rb, tempW_u, tempW_v, true));
        parallel_for_(Range(0, num_stripes), ComputeSmoothnessTermHorPass_ParBody(*this, num_stripes, I0Mat.rows, W_u_rb,
                                                                           W_v_rb, tempW_u, tempW_v, false));
        parallel_for_(Range(0, num_stripes), ComputeSmoothnessTermVertPass_ParBody(*this, num_stripes, I0Mat.rows, W_u_rb,
            W_v_rb, true));
        parallel_for_(Range(0, num_stripes), ComputeSmoothnessTermVertPass_ParBody(*this, num_stripes, I0Mat.rows, W_u_rb,
            W_v_rb, false));

        for (int j = 0; j < sorIterations; j++)
        {
            parallel_for_(Range(0, num_stripes), RedBlackSOR_ParBody(*this, num_stripes, I0Mat.rows, dW_u, dW_v, true));
            parallel_for_(Range(0, num_stripes),
                          RedBlackSOR_ParBody(*this, num_stripes, I0Mat.rows, dW_u, dW_v, false));
        }

        tempW_u.red = W_u_rb.red + dW_u.red;
        tempW_u.black = W_u_rb.black + dW_u.black;
        updateRepeatedBorders(tempW_u);
        tempW_v.red = W_v_rb.red + dW_v.red;
        tempW_v.black = W_v_rb.black + dW_v.black;
        updateRepeatedBorders(tempW_v);
    }
    mergeCheckerboard(W_u, tempW_u);
    mergeCheckerboard(W_v, tempW_v);
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
