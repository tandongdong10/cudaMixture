#include <tuple>
#include <cstddef>
#include <iostream>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include "cudaMixture.H"
#include <fstream>

std::fstream fout("out.txt");

template<typename T>
__global__ void printOnDevice(int length,T* p)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (ix >= length) return;
    if constexpr (std::is_same_v<std::decay_t<T>,double>)
        printf("%lf ",p[ix]);
    else if constexpr (std::is_same_v<std::decay_t<T>, int>)
        printf("%d ",p[ix]);
}

__global__ void caculate_specieSOA(
    int     ncell,                     // 网格数
    int     nspecie,                   // 物种数
    double *Y_in,                      // 质量分数 [nspecie][ncell]
    double *Y_,                        // 纯态质量 [nspecie]
    double *Y0_,                       // 纯态（混态）各个物种所占的质量 [nspecie][nspecie] 第二个是物种所占成分列表
    double *molWeight_,                // 纯态摩尔质量 [nspecie]
    double  RR,                        // 气体常数
    double *diameter_,                 // 纯态分子的有效直径 [nspecie]
    double *omega_,                    // 纯态黏度系数 [nspecie]
    double *dissociationPotential_,    // 纯态解离能 [nspecie]
    double *iHat_,                     // 纯态第一电离能 [nspecie]
    int    *vibTempAssociativity_,     // 纯态振动温度是否符合 [nspecie]
    bool   *nonEqm_,                   // 纯态是否是非平衡态 [nspecie]
    int     nVibrationalList_,         // 下面的长度
    double *vibrationalList_,          // 纯态振动能级列表[nVibrationalList_][ncell]
    double *Y_out,                     // 混态质量 [ncell]
    double *Y0_out,                    // 混态各个物种所占的质量 [nspecie][ncell]
    double *molWeight_out,             // 混态摩尔质量 [ncell]
    double *R_out,                     // 混态气体常数(mol) [ncell]
    double *diameter_out,              // 混态分子的有效直径 [ncell]
    double *omega_out,                 // 混态黏度系数 [ncell]
    double *dissociationPotential_out, // 混态解离能 [ncell]
    double *iHat_out,                  // 混态第一电离能 [ncell]
    int *vibTempAssociativity_out,  // 混态振动温度是否耦合标志 [ncell]
    // janafThermoSOA
    double *Tlow_,                     // [nspecie]
    double *Thigh_,                    // [npsecie]
    double *Tcommon_,                  // [npsecie]
    double *dHa_high_,                 // [npsecie]
    double *dHa_low_,                  // [npsecie]
    int nCoeffs_,                      // 默认是7
    double *highCpCoeffs_,             // [nCoeffs_][npsecie]
    double *lowCpCoeffs_,              // [nCoeffs_][npsecie]
    double *decoupledCvCoeffs_,        // [nCoeffs_][npsecie]
    int     nElectronicList_,          // 下面变长数组中的长度
    double *electronicList_,           // [nElectronicList_][npsecie]
    double *Tlow_out,                  // [ncell]
    double *Thigh_out,                 // [ncell]
    double *Tcommon_out,               // [ncell]
    double *highCpCoeffs_out,          // [nCoeffs   _][ncell]
    double *lowCpCoeffs_out,           // [nCoeffs_][ncell]
    double *decoupledCvCoeffs_out,     // [nCoeffs_][ncell]
    double *vibrationalList_out,       // [nVibrationalList_][ncell]
    double *electronicList_out,        // [nElectronicList_][ncell]
    // sutherlandTransportSOA
    double *As_,                       // [npsecie]
    double *Ts_,                       // [npsecie]
    double *eta_s_,                    // [npsecie]
    double *Ak_,                       // [npsecie]
    double *Bk_,                       // [npsecie]
    double *Ck_,	                   // [npsecie]
    double *As_out,                    // [ncell]
    double *Ts_out,                    // [ncell]
    double *eta_s_out,                 // [ncell]
    double *Ak_out,                    // [ncell]
    double *Bk_out,                    // [ncell]
    double *Ck_out	                   // [ncell]
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    printf("%d ",ix);
    if (ix >= ncell) return;
    printf("%d\n",ix);

    double last_Y                = 0.0;
    double Y                     = 0.0;
    double last_sum_molWeight_   = 0.0;
    double sum_molWeight_        = 0.0;
    double diameter              = 0.0;
    double omega                 = 0.0;
    double dissociationPotential = 0.0;
    double iHat                  = 0.0;
    int    vibTempAssociativity  = INT_MAX;
    bool   nonEqm                = true; 

    // janafThermoSOA
    double Tlow = Tlow_[0];
    double Thigh = Thigh_[0];

    for (int i = 0; i < nspecie; i++)
    {
        last_Y = Y; // 网格[ix]上一轮[specie]的质量
        Y += Y_in[i * ncell + ix] * Y_[i]; // 网格[ix]这一轮[specie]累计的质量
        last_sum_molWeight_ = sum_molWeight_;
        sum_molWeight_ += Y_in[i * ncell + ix] * Y_[i] / molWeight_[i];

        // 每个ix 中的j物种
        for (int j = 0; j < nspecie; j++)
        {
            Y0_out[j * ncell + ix] += Y0_[j * nspecie + i] * Y_in[i * ncell + ix];
        }
        nonEqm = nonEqm && nonEqm_[i];
        if(nonEqm)
        {
            diameter              = diameter * last_sum_molWeight_ / sum_molWeight_ + diameter_[i] * (1 - last_sum_molWeight_ / sum_molWeight_);
            omega                 = omega * last_sum_molWeight_ / sum_molWeight_ + omega_[i] * (1 - last_sum_molWeight_ / sum_molWeight_);
            dissociationPotential = dissociationPotential * last_sum_molWeight_ / sum_molWeight_ + dissociationPotential_[i] * (1 - last_sum_molWeight_ / sum_molWeight_);
            iHat                  = iHat * last_sum_molWeight_ / sum_molWeight_ + iHat_[i] * (1 - last_sum_molWeight_ / sum_molWeight_);
            vibTempAssociativity  = min(vibTempAssociativity, vibTempAssociativity_[i]);
        }


        // janafThermoSOA
        Tlow  = max(Tlow,Tlow_[i]);
        Thigh = min(Thigh,Thigh_[i]);

        for (int j = 0; j < nCoeffs_; j++)
        {
            // 可以优化成先乘最后汇总除以
            highCpCoeffs_out[j * ncell + ix] = highCpCoeffs_out[j * ncell + ix]*(last_Y/Y) + highCpCoeffs_[j * nspecie + i]*((Y-last_Y)/Y);
            lowCpCoeffs_out [j * ncell + ix]  = lowCpCoeffs_out[j * ncell + ix]*(last_Y/Y) + lowCpCoeffs_[j * nspecie + i]*((Y-last_Y)/Y);
        }

    
        if(nonEqm)
        {
        
            for (int j = 0; j < nCoeffs_; j++)
            {
                decoupledCvCoeffs_out [j * ncell + ix]  = decoupledCvCoeffs_out[j * ncell + ix]*(last_Y/Y) + decoupledCvCoeffs_[j * nspecie + i]*((Y-last_Y)/Y);
            }

            for(int j =0; j < nVibrationalList_; j++)
            {
                vibrationalList_out [j * ncell + ix]  = vibrationalList_out[j * ncell + ix]*(last_Y/Y) + vibrationalList_[j * nspecie + i]*((Y-last_Y)/Y);
                // 过滤逻辑没写
            }


            

            for(int j = 0; j<nElectronicList_;j++)
            {
                electronicList_out [j * ncell + ix]  = electronicList_out[j * ncell + ix]*(last_Y/Y) + electronicList_[j * nspecie + i]*((Y-last_Y)/Y);
            }
            // check 逻辑没写，里面会调用一些函数，需要具体的实现
        }
        
        // sutherlandTransportSOA
        As_out[ix]      = As_out[ix]*(last_Y/Y)+As_[i]*((Y-last_Y)/Y);
        Ts_out[ix]      = Ts_out[ix]*(last_Y/Y)+Ts_[i]*((Y-last_Y)/Y);
        eta_s_out[ix]   = eta_s_out[ix]*(last_Y/Y)+eta_s_[i]*((Y-last_Y)/Y);
        Ak_out[ix]      = Ak_out[ix]*(last_Y/Y)+Ak_[i]*((Y-last_Y)/Y);
        Bk_out[ix]      = Bk_out[ix]*(last_Y/Y)+Bk_[i]*((Y-last_Y)/Y);
        Ck_out[ix]      = Ck_out[ix]*(last_Y/Y)+Ck_[i]*((Y-last_Y)/Y);
    }

    // 写回
    Y_out[ix]         = Y;
    molWeight_out[ix] = Y / sum_molWeight_;
    R_out[ix]         = RR / (Y / sum_molWeight_);
    diameter_out[ix] = diameter;
    omega_out[ix]     = omega;
    dissociationPotential_out[ix] = dissociationPotential;
    iHat_out[ix] = iHat;
    vibTempAssociativity_out[ix] = vibTempAssociativity;

    // janafThermoSOA
    Tlow_out[ix] = Tlow;
    Thigh_out[ix] = Thigh;
    Tcommon_out[ix] = Tcommon_[0];
}



void caculateOnHost_(
    int     ncell,                     // 网格数
    int     nspecie,                   // 物种数
    double *Y_in,                      // 质量分数 [nspecie][ncell]
    double *Y_,                        // 纯态质量 [nspecie]
    double *Y0_,                       // 纯态（混态）各个物种所占的质量 [nspecie][nspecie] 第二个是物种所占成分列表
    double *molWeight_,                // 纯态摩尔质量 [nspecie]
    double  RR,                        // 气体常数
    double *diameter_,                 // 纯态分子的有效直径 [nspecie]
    double *omega_,                    // 纯态黏度系数 [nspecie]
    double *dissociationPotential_,    // 纯态解离能 [nspecie]
    double *iHat_,                     // 纯态第一电离能 [nspecie]
    int    *vibTempAssociativity_,     // 纯态振动温度是否符合 [nspecie]
    bool   *nonEqm_,                   // 纯态是否是非平衡态 [nspecie]
    int     nVibrationalList_,         // 下面的长度
    double *vibrationalList_,          // 纯态振动能级列表[nVibrationalList_][ncell]
    double *Y_out,                     // 混态质量 [ncell]
    double *Y0_out,                    // 混态各个物种所占的质量 [nspecie][ncell]
    double *molWeight_out,             // 混态摩尔质量 [ncell]
    double *R_out,                     // 混态气体常数(mol) [ncell]
    double *diameter_out,              // 混态分子的有效直径 [ncell]
    double *omega_out,                 // 混态黏度系数 [ncell]
    double *dissociationPotential_out, // 混态解离能 [ncell]
    double *iHat_out,                  // 混态第一电离能 [ncell]
    int     *vibTempAssociativity_out, // 混态振动温度是否耦合标志 [ncell]
    // janafThermoSOA
    double *Tlow_,                     // [nspecie]
    double *Thigh_,                    // [npsecie]
    double *Tcommon_,                  // [npsecie]
    double *dHa_high_,                 // [npsecie]
    double *dHa_low_,                  // [npsecie]
    int nCoeffs_,                      // 默认是7
    double *highCpCoeffs_,             // [nCoeffs_][npsecie]
    double *lowCpCoeffs_,              // [nCoeffs_][npsecie]
    double *decoupledCvCoeffs_,        // [nCoeffs_][npsecie]
    int     nElectronicList_,          // 下面变长数组中的长度
    double *electronicList_,           // [nElectronicList_][npsecie]
    double *Tlow_out,                  // [ncell]
    double *Thigh_out,                 // [ncell]
    double *highCpCoeffs_out,          // [nCoeffs   _][ncell]
    double *lowCpCoeffs_out,           // [nCoeffs_][ncell]
    double *decoupledCvCoeffs_out,     // [nCoeffs_][ncell]
    double *vibrationalList_out,       // [nVibrationalList_][ncell]
    double *electronicList_out,        // [nElectronicList_][ncell]
    // sutherlandTransportSOA
    double *As_,                       // [npsecie]
    double *Ts_,                       // [npsecie]
    double *eta_s_,                    // [npsecie]
    double *Ak_,                       // [npsecie]
    double *Bk_,                       // [npsecie]
    double *Ck_,	                   // [npsecie]
    double *As_out,                    // [ncell]
    double *Ts_out,                    // [ncell]
    double *eta_s_out,                 // [ncell]
    double *Ak_out,                    // [ncell]
    double *Bk_out,                    // [ncell]
    double *Ck_out	                   // [ncell]
)
{
    // Y_out
    for(int celli = 0; celli<ncell;celli++)
    {
        Y_out[celli] = 0;
        for(int speciei = 0; speciei < nspecie; speciei++)
        {
            Y_out[celli] += Y_in[speciei*ncell + celli] * Y_[speciei];
        }
        
    }

    // Y0_out
    for(int i =0;i<nspecie;i++) // 物种成分列表循环
    {
        for(int j=0;j<ncell;j++)
        {
            Y0_out[i*ncell+j] = 0;
            for(int k=0;k<nspecie;k++)
            {
                Y0_out[i*ncell+j] += Y0_[i*nspecie+k]*Y_in[k*ncell+j];
            }
        }
    }


    // molWeight_out

    for(int i=0;i<ncell;i++) // 遍历网格
    {
        double molSum = 0.0;
        for(int j=0;j<nspecie;j++) // 遍历物种
        {
            double moltemp = Y_in[j*ncell+i]*Y_[j] / molWeight_[j]; // 计算该网格中该物质的摩尔量
            molSum += moltemp;
        }
        molWeight_out[i] = Y_out[i] / molSum; // 平均摩尔质量
        R_out[i] = RR / molWeight_out[i]; // 计算摩尔气体常数
    }

    // 摩尔质量加权
    for(int i=0;i<ncell;i++) // 遍历网格
    {
        double molSum = 0.0;
        diameter_out[i] = 0.0;
        omega_out[i] = 0.0;
        dissociationPotential_out[i] = 0.0;
        iHat_out[i] = 0.0;
        vibTempAssociativity_out[i] = INT_MAX;

        for(int j=0;j<nspecie;j++) // 遍历物种
        {
            double moltemp = Y_in[j*ncell+i] * Y_[j] / molWeight_[j]; // 计算该网格中该物质的摩尔量
            molSum += moltemp;

            diameter_out[i] += diameter_[j]*moltemp;
            omega_out[i] += omega_[j]*moltemp;
            dissociationPotential_out[i] += dissociationPotential_[j]*moltemp;
            iHat_out[i] += iHat_[j] * moltemp;


            vibTempAssociativity_out[i] = min(vibTempAssociativity_out[i],vibTempAssociativity_[j]);
        }

        diameter_out[i] /= molSum;
        omega_out[i] /= molSum;
        dissociationPotential_out[i] /= molSum;
        iHat_out[i] /= molSum;
    }

    // janafThermoSOA
    for(int i=0;i<ncell;i++) // 遍历网格
    {
        Tlow_out[i] = Tlow_[0];
        Thigh_out[i] = Thigh_[0];

        for(int j=0;j<nspecie;j++) // 遍历物种
        {
            Tlow_out[i] = max(Tlow_out[i],Tlow_[j]);
            Thigh_out[i] = min(Thigh_out[i],Thigh_[j]);
        
        }
    }


    for(int i=0;i<ncell;i++) // 遍历网格
    {
        for(int k=0;k<nCoeffs_;k++) // 对变量列表进行遍历
        {
            highCpCoeffs_out[k*ncell + i] = 0;
            lowCpCoeffs_out[k*ncell + i] = 0;
        }
        
        for(int j=0;j<nspecie;j++) // 遍历物种
        {
            double Y1 = Y_[j] * Y_in[j * ncell + i];
            for(int k=0;k<nCoeffs_;k++) // 对变量列表进行遍历
            {
                highCpCoeffs_out[k*ncell + i] += highCpCoeffs_[k*nspecie + j]*Y1;
                lowCpCoeffs_out[k*ncell + i] += lowCpCoeffs_[k*nspecie + j]*Y1;
            }
            
        }

        for(int k=0;k<nCoeffs_;k++) // 对变量列表进行遍历
        {
            highCpCoeffs_out[k*ncell + i] /= Y_out[i];
            lowCpCoeffs_out[k*ncell + i] /= Y_out[i];
        }
    }

    for(int i=0;i<ncell;i++) // 遍历网格
    {
        for(int k=0;k<nCoeffs_;k++) // 对变量列表进行遍历
        {
            decoupledCvCoeffs_out[k*ncell + i] = 0;
            
        }
        
        for(int j=0;j<nspecie;j++) // 遍历物种
        {
            double Y1 = Y_[j] * Y_in[j * ncell + i];
            for(int k=0;k<nCoeffs_;k++) // 对变量列表进行遍历
            {
                decoupledCvCoeffs_out[k*ncell + i] += decoupledCvCoeffs_[k*nspecie + j]*Y1;
            }
            
        }

        for(int k=0;k<nCoeffs_;k++) // 对变量列表进行遍历
        {
            decoupledCvCoeffs_out[k*ncell + i] /= Y_out[i];
        }
    }


    for(int i=0;i<ncell;i++) // 遍历网格
    {
        for(int k=0;k<nVibrationalList_;k++) // 对变量列表进行遍历
        {
            vibrationalList_out[k*ncell + i] = 0;
        }
        
        for(int j=0;j<nspecie;j++) // 遍历物种
        {
            double Y1 = Y_[j] * Y_in[j * ncell + i];
            for(int k=0;k<nVibrationalList_;k++) // 对变量列表进行遍历
            {
                vibrationalList_out[k*ncell + i] += vibrationalList_[k*nspecie + j]*Y1;
            }
            
        }

        for(int k=0;k<nVibrationalList_;k++) // 对变量列表进行遍历
        {
            vibrationalList_out[k*ncell + i] /= Y_out[i];
        }
    }



    for(int i=0;i<ncell;i++) // 遍历网格
    {
        for(int k=0;k<nElectronicList_;k++) // 对变量列表进行遍历
        {
            electronicList_out[k*ncell + i] = 0;
            
        }
        
        for(int j=0;j<nspecie;j++) // 遍历物种
        {
            double Y1 = Y_[j] * Y_in[j * ncell + i];
            for(int k=0;k<nElectronicList_;k++) // 对变量列表进行遍历
            {
                electronicList_out[k*ncell + i] += electronicList_[k*nspecie + j]*Y1;
            }
            
        }

        for(int k=0;k<nElectronicList_;k++) // 对变量列表进行遍历
        {
            electronicList_out[k*ncell + i] /= Y_out[i];
        }
    }

    for(int i=0;i<ncell;i++) // 遍历网格
    {
        As_out[i] = 0.0;
        Ts_out[i]=0.0;
        eta_s_out[i]=0.0;
        Ak_out[i]=0.0;
        Bk_out[i]=0.0;
        Ck_out[i]=0.0;


        for(int j=0;j<nspecie;j++) // 遍历物种
        {
            double Y1 = Y_[j] * Y_in[j * ncell + i];
            As_out[i] += As_[j] * Y1;
            Ts_out[i] += Ts_[j] * Y1;
            eta_s_out[i] += eta_s_[j] * Y1;
            Ak_out[i] += Ak_[j] * Y1;
            Bk_out[i] += Bk_[j] * Y1;
            Ck_out[i] += Ck_[j] * Y1;
        }
        As_out[i] /= Y_out[i];
        Ts_out[i]/=Y_out[i];
        eta_s_out[i]/=Y_out[i];
        Ak_out[i]/=Y_out[i];
        Bk_out[i]/=Y_out[i];
        Ck_out[i]/=Y_out[i];
    }




}



static int nspecie = 3;
static int ncell   = 5;
static double RR = 287.0;
static int nVibrationalList_ = 2;
static int nCoeffs_ = 5;  // 默认是7
static int nElectronicList_ = 30;          // 下面变长数组中的长度




struct Y_in { static int getN(){ return nspecie*ncell; } };
struct Y_ { static int getN(){ return nspecie; }};
struct Y0_ { static int getN(){ return nspecie*nspecie; }};
struct molWeight_ { static int getN(){ return nspecie; } };
struct diameter_{ static int getN(){ return nspecie; } };
struct omega_{ static int getN(){ return nspecie; } };
struct dissociationPotential_{ static int getN(){ return nspecie; } };
struct iHat_{ static int getN(){ return nspecie; } };
struct vibTempAssociativity_{ static int getN(){ return nspecie; } };
struct nonEqm_{ static int getN(){ return nspecie; } };
struct vibrationalList_{ static int getN(){ return nspecie*nVibrationalList_; } };
struct Y_out{ static int getN(){ return ncell; } };
struct Y0_out{ static int getN(){ return nspecie*ncell; } };


struct molWeight_out{ static int getN(){ return ncell; } };
struct R_out{ static int getN(){ return ncell; } };
struct diameter_out{ static int getN(){ return ncell; } };
struct omega_out{ static int getN(){ return ncell; } };
struct dissociationPotential_out{ static int getN(){ return ncell; } };
struct iHat_out{ static int getN(){ return ncell; } };
struct vibTempAssociativity_out{ static int getN(){ return ncell; } };


struct Tlow_{ static int getN(){ return nspecie; } };
struct Thigh_{ static int getN(){ return nspecie; } };
struct Tcommon_{ static int getN(){ return nspecie; } };
struct dHa_high_{ static int getN(){ return nspecie; } };
struct dHa_low_{ static int getN(){ return nspecie; } };
                     
struct highCpCoeffs_{ static int getN(){ return nCoeffs_*nspecie; } };
struct lowCpCoeffs_{ static int getN(){ return nCoeffs_*nspecie; } };
struct decoupledCvCoeffs_{ static int getN(){ return nCoeffs_*nspecie; } };

struct electronicList_{ static int getN(){ return nElectronicList_*nspecie; } };



    
struct Tlow_out{ static int getN(){ return ncell; } };
struct Thigh_out{ static int getN(){ return ncell; } };
struct Tcommon_out{ static int getN(){ return ncell; } };
struct highCpCoeffs_out{ static int getN(){ return nCoeffs_*ncell; } };
struct lowCpCoeffs_out{ static int getN(){ return nCoeffs_*ncell; } };
struct decoupledCvCoeffs_out{ static int getN(){ return nCoeffs_*ncell; } };
struct vibrationalList_out{ static int getN(){ return nVibrationalList_*ncell; } };
struct electronicList_out{ static int getN(){ return nElectronicList_*ncell; } };


// sutherlandTransportSOA
struct As_{ static int getN(){ return nspecie; } };
struct Ts_{ static int getN(){ return nspecie; } };
struct eta_s_{ static int getN(){ return nspecie; } };
struct Ak_{ static int getN(){ return nspecie; } };
struct Bk_{ static int getN(){ return nspecie; } };
struct Ck_{ static int getN(){ return nspecie; } };
struct As_out{ static int getN(){ return ncell; } };
struct Ts_out{ static int getN(){ return ncell; } };
struct eta_s_out{ static int getN(){ return ncell; } };
struct Ak_out{ static int getN(){ return ncell; } };
struct Bk_out{ static int getN(){ return ncell; } };
struct Ck_out{ static int getN(){ return ncell; } };




/**
 * Tag_ 变量名，但是实际上一个类型
 * Type_ 类型名，表示这个变量管理的数据类型是double，还是int
 * N_ 管理的长度
 */
template<typename Tag_, typename Type_>
struct Var {
    using Tag  = Tag_;
    using Type = Type_;
};

/**
 * 上面组合成一个元数据列表，放在Vars中,Vars中每一项表述了物理变量，数据类型，以及长度
 */
template<typename... Vars>
struct VarList {};


// 在 Vars 中查找和Var相同类型的 下标
template<typename T>
struct always_false : std::false_type {};

// 前向声明，主模板
template<typename Var, typename... Vars>
struct index_of;

// 第一个和第二个都是Var的偏特化
template<typename Var, typename... Rest>
struct index_of<Var, Var, Rest...> : std::integral_constant<size_t, 0> {};

// 第一个和第二个不是相同的类型，继续递归
template<typename Var, typename First, typename... Rest>
struct index_of<Var, First, Rest...>
    : std::integral_constant<size_t, 1 + index_of<Var, Rest...>::value> {};

// 都不匹配，直接编译不通过
template<typename Var>
struct index_of<Var> {
    static_assert(always_false<Var>::value, "index_of error: Var not found in list");
};


// -------- HostData --------
// 这里接收一个泛型VL，主模板（VL最后偏特化为VarList类型）
template<typename VL>
struct HostData;

// 对泛型 VL 偏特化为 VarList<Vars...>
template<typename... Vars>
struct HostData<VarList<Vars...>> {

    std::tuple<typename Vars::Type*...> data;

    template<typename Var>
    void init(std::initializer_list<typename Var::Type> list)
    {
        constexpr size_t I = index_of<Var, Vars...>::value;
        using Type = typename Var::Type;
        using Tag = typename Var::Tag; 
        size_t N = Tag::getN();
        Type* ptr = std::get<I>(data);

        if (list.size() != N) {
            throw std::runtime_error("initializer_list size mismatch");
        }

        if(!ptr) {
            throw std::runtime_error("指针未初始化");
        }
        std::copy(list.begin(), list.end(), ptr);
        // for(int i=0;i<N;i++)
        // {
        //     std::cout<<ptr[i]<<std::endl;
        // }
    }

    template<typename Var>
    void init(typename Var::Type* p)
    {
        constexpr size_t I = index_of<Var, Vars...>::value;
        std::get<I>(data) = p;
    }

    template<size_t I = 0>
    void allocate_host() {
        if constexpr (I < sizeof...(Vars)) {
            using VarI = std::tuple_element_t<I, std::tuple<Vars...>>;
            using Type = typename VarI::Type;
            using Tag = typename VarI::Tag;
            size_t N = Tag::getN();

            std::get<I>(data) = new Type[N];
            std::cout<< typeid(Tag).name() <<" 开辟空间 类型: "<<typeid(Type).name()<<" 长度: "<<N<<std::endl; 

            allocate_host<I+1>();
        }
    }

    // 将管理的所以指针指向的数据重置为0
    template<size_t I = 0>
    void clear_to_zero() {
        if constexpr (I < sizeof...(Vars)) {
            using VarI = std::tuple_element_t<I, std::tuple<Vars...>>;
            using Type = typename VarI::Type;
            using Tag = typename VarI::Tag;
            size_t N = Tag::getN();

            memset(std::get<I>(data),0,sizeof(Type)*N);
            for(int i=0;i<N;i++)
            {
                std::cout<<std::get<I>(data)[i]<<" ";
            }
            std::cout<<std::endl;

            clear_to_zero<I+1>();
        }
    }

    // 打印管理的指针指向的数据
    template<size_t I = 0>
    void print() {
        if constexpr (I < sizeof...(Vars)) {
            using VarI = std::tuple_element_t<I, std::tuple<Vars...>>;
            using Type = typename VarI::Type;
            using Tag = typename VarI::Tag;
            size_t N = Tag::getN();
            fout<<typeid(Tag).name()<<"\n";
            for(int i=0;i<N;i++)
            {
                fout<<std::get<I>(data)[i]<<" ";
            }
            fout<<"\n";

            print<I+1>();
        }
    }

    template<size_t I = 0>
    void copy(HostData<VarList<Vars...>>& rhs)
    {
        if constexpr (I < sizeof...(Vars)) {
            using VarI = std::tuple_element_t<I, std::tuple<Vars...>>;
            using Type = typename VarI::Type;
            using Tag = typename VarI::Tag;
            size_t N = Tag::getN();

            Type* d1 = std::get<I>(data);
            Type* d2 = std::get<I>(rhs.data);
            for(int i=0;i<N;i++)
            {
                d1[i] = d2[i];
            }
            // std::cout<< typeid(Tag).name() <<" 拷贝成功 类型: "<<typeid(Type).name()<<" 长度: "<<N<<std::endl; 
            
            copy<I+1>(rhs);
        }

    }

    template<typename Var>
    auto get() {
        constexpr size_t I = index_of<Var,Vars...>::value;
        return std::get<I>(data);
    }

    void caculateOnHost()
    {
        
        caculateOnHost_
        (
            ncell,
            nspecie,
            get<Var<Y_in                        ,double >>(),
            get<Var<Y_                          ,double >>(),
            get<Var<Y0_                         ,double >>(),
            get<Var<molWeight_                  ,double >>(),
            RR                                              ,
            get<Var<diameter_                   ,double >>(),
            get<Var<omega_                      ,double >>(),
            get<Var<dissociationPotential_      ,double >>(),
            get<Var<iHat_                       ,double >>(),
            get<Var<vibTempAssociativity_       ,int    >>(),
            get<Var<nonEqm_                     ,bool   >>(),
            nVibrationalList_                              ,
            get<Var<vibrationalList_            ,double >>(),
            get<Var<Y_out                       ,double >>(),
            get<Var<Y0_out                      ,double >>(),
            get<Var<molWeight_out               ,double >>(),
            get<Var<R_out                       ,double >>(),
            get<Var<diameter_out                ,double >>(),
            get<Var<omega_out                   ,double >>(),
            get<Var<dissociationPotential_out   ,double >>(),
            get<Var<iHat_out                    ,double >>(),
            get<Var<vibTempAssociativity_out    ,int    >>(),
            get<Var< Tlow_                      ,double >>(),
            get<Var< Thigh_                     ,double >>(),
            get<Var< Tcommon_                   ,double >>(),
            get<Var< dHa_high_                  ,double >>(),
            get<Var< dHa_low_                   ,double >>(),
            nCoeffs_                                        ,
            get<Var< highCpCoeffs_              ,double >>(),
            get<Var< lowCpCoeffs_               ,double >>(),
            get<Var< decoupledCvCoeffs_         ,double >>(),
            nElectronicList_                                ,
            get<Var< electronicList_            ,double >>(),
            get<Var<Tlow_out                    ,double >>(),
            get<Var<Thigh_out                   ,double >>(),
            get<Var<highCpCoeffs_out            ,double >>(),
            get<Var<lowCpCoeffs_out             ,double >>(),
            get<Var<decoupledCvCoeffs_out       ,double >>(),
            get<Var<vibrationalList_out         ,double >>(),
            get<Var<electronicList_out          ,double >>(),
            get<Var<As_                         ,double >>(),
            get<Var<Ts_                         ,double >>(),
            get<Var<eta_s_                      ,double >>(),
            get<Var<Ak_                         ,double >>(),
            get<Var<Bk_                         ,double >>(),
            get<Var<Ck_                         ,double >>(),
            get<Var<As_out                      ,double >>(),
            get<Var<Ts_out                      ,double >>(),
            get<Var<eta_s_out                   ,double >>(),
            get<Var<Ak_out                      ,double >>(),
            get<Var<Bk_out                      ,double >>(),
            get<Var<Ck_out                      ,double >>()
        );
        
    }

    template<size_t I = 0>
    void check(HostData<VarList<Vars...>>& rhs)
    {
        if constexpr (I < sizeof...(Vars)) {
            using VarI = std::tuple_element_t<I, std::tuple<Vars...>>;
            using Type = typename VarI::Type;
            using Tag = typename VarI::Tag;
            size_t N = Tag::getN();

            Type* d1 = std::get<I>(data);
            Type* d2 = std::get<I>(rhs.data);
            
            for(int i=0;i<N;i++)
            {
                if(abs(d1[i]-d2[i]) > 1e-10)
                {
                    std::cout<<d1[i]<<" "<<d2[i]<<std::endl;
                    throw std::runtime_error(typeid(typename VarI::Tag).name()+  std::string(" index of ") + std::to_string(i)  + " mismatch");
                }
            }
            
            check<I+1>(rhs);
        }

    }

};

// -------- DeviceData --------
template<typename VL>
struct DeviceData;

template<typename... Vars>
struct DeviceData<VarList<Vars...>> {
    std::tuple<typename Vars::Type*...> data;

    template<size_t I = 0>
    void allocate_device(){
        if constexpr (I < sizeof...(Vars)) {
            using VarI = std::tuple_element_t<I,std::tuple<Vars...>>;
            using Type = typename VarI::Type;
            using Tag = typename VarI::Tag; 
            size_t N = Tag::getN();

            std::cout<< typeid(Tag).name() <<" 申请cuda空间 类型: "<<typeid(Type).name()<<" 长度: "<<N<<std::endl;
            cudaError_t err = cudaMalloc(&std::get<I>(data),sizeof(Type)*N);
            if(err != cudaSuccess)
            {
                throw std::runtime_error("memcpyToDevice 失败");
            }

            allocate_device<I+1>();
        }
    }

    // 将管理的所以指针指向的数据重置为0
    template<size_t I = 0>
    void clear_to_zero() {
        if constexpr (I < sizeof...(Vars)) {
            using VarI = std::tuple_element_t<I, std::tuple<Vars...>>;
            using Type = typename VarI::Type;
            using Tag = typename VarI::Tag;
            size_t N = Tag::getN();


            cudaError_t err = cudaMemset(std::get<I>(data),0,sizeof(Type)*N);
            if(err != cudaSuccess)
            {
                std::ostringstream os;
                os << "cudaMemset failed at I=" << I <<", 类型名=" <<typeid(Tag).name() << ", error=" << cudaGetErrorString(err);

                throw std::runtime_error(os.str());
            }

            clear_to_zero<I+1>();
        }
    }

    template<size_t I = 0>
    void memcpyToDevice(HostData<VarList<Vars...>>& hd)
    {
        if constexpr (I < sizeof...(Vars)) {
            using VarI = std::tuple_element_t<I,std::tuple<Vars...>>;
            using Type = typename VarI::Type;
            using Tag = typename VarI::Tag; 
            size_t N = Tag::getN();

            
            std::cout<< typeid(Tag).name() <<" host -> device 类型 "<<typeid(Type).name()<<" 长度: "<<N<<std::endl;

            cudaError_t err = cudaMemcpy(std::get<I>(data),std::get<I>(hd.data),sizeof(Type)*N,cudaMemcpyHostToDevice);
            if(err != cudaSuccess)
            {
                throw std::runtime_error("memcpyToDevice 失败");
            }


            memcpyToDevice<I+1>(hd);
        }
    }

    template<size_t I = 0,typename... VarsOther>
    void copyFromOtherDevice(DeviceData<VarList<VarsOther...>>& dd)
    {
        // 遍历VarsOther的类型
        if constexpr (I < sizeof...(VarsOther)) {
            // 获取VarsOther第I个的类型
            using VarsOtherI = std::tuple_element_t<I,std::tuple<VarsOther...>>;
            using Type = typename VarsOtherI::Type;
            using Tag = typename VarsOtherI::Tag; 
            size_t N = Tag::getN();

            this->template get<VarsOtherI>() = dd.template get<VarsOtherI>();
            copyFromOtherDevice<I+1>(dd);
        }
    }


    template<typename Tag>
    auto& get() {
        constexpr size_t I = index_of<Tag,Vars...>::value;
        return std::get<I>(data);
    }

    template<size_t I = 0>
    void caculate()
    {
        int  iLen = 256; // 512 无法启动
        dim3 block(iLen);
        dim3 grid((ncell + block.x - 1) / block.x);
        caculate_specieSOA<<<grid, block>>>
        (
            ncell,
            nspecie,
            get<Var<Y_in                        ,double >>(),
            get<Var<Y_                          ,double >>(),
            get<Var<Y0_                         ,double >>(),
            get<Var<molWeight_                  ,double >>(),
            RR                                              ,
            get<Var<diameter_                   ,double >>(),
            get<Var<omega_                      ,double >>(),
            get<Var<dissociationPotential_      ,double >>(),
            get<Var<iHat_                       ,double >>(),
            get<Var<vibTempAssociativity_       ,int    >>(),
            get<Var<nonEqm_                     ,bool   >>(),
            nVibrationalList_                              ,
            get<Var<vibrationalList_            ,double >>(),
            get<Var<Y_out                       ,double >>(),
            get<Var<Y0_out                      ,double >>(),
            get<Var<molWeight_out               ,double >>(),
            get<Var<R_out                       ,double >>(),
            get<Var<diameter_out                ,double >>(),
            get<Var<omega_out                   ,double >>(),
            get<Var<dissociationPotential_out   ,double >>(),
            get<Var<iHat_out                    ,double >>(),
            get<Var<vibTempAssociativity_out    ,int    >>(),
            get<Var<Tlow_                       ,double >>(),
            get<Var<Thigh_                      ,double >>(),
            get<Var<Tcommon_                    ,double >>(),
            get<Var<dHa_high_                   ,double >>(),
            get<Var<dHa_low_                    ,double >>(),
            nCoeffs_                                        ,
            get<Var<highCpCoeffs_               ,double >>(),
            get<Var<lowCpCoeffs_                ,double >>(),
            get<Var<decoupledCvCoeffs_          ,double >>(),
            nElectronicList_                                ,
            get<Var< electronicList_            ,double >>(),
            get<Var<Tlow_out                    ,double >>(),
            get<Var<Thigh_out                   ,double >>(),
            get<Var<Tcommon_out                 ,double >>(),
            get<Var<highCpCoeffs_out            ,double >>(),
            get<Var<lowCpCoeffs_out             ,double >>(),
            get<Var<decoupledCvCoeffs_out       ,double >>(),
            get<Var<vibrationalList_out         ,double >>(),
            get<Var<electronicList_out          ,double >>(),
            get<Var<As_                         ,double >>(),
            get<Var<Ts_                         ,double >>(),
            get<Var<eta_s_                      ,double >>(),
            get<Var<Ak_                         ,double >>(),
            get<Var<Bk_                         ,double >>(),
            get<Var<Ck_                         ,double >>(),
            get<Var<As_out                      ,double >>(),
            get<Var<Ts_out                      ,double >>(),
            get<Var<eta_s_out                   ,double >>(),
            get<Var<Ak_out                      ,double >>(),
            get<Var<Bk_out                      ,double >>(),
            get<Var<Ck_out                      ,double >>()
        );

    }

    template<size_t I = 0>
    void memcpyToHost(HostData<VarList<Vars...>>& hd)
    {
        if constexpr (I < sizeof...(Vars)) {
            using VarI = std::tuple_element_t<I,std::tuple<Vars...>>;
            using Type = typename VarI::Type;
            using Tag = typename VarI::Tag; 
            size_t N = Tag::getN();

            // std::cout<<"device -> host "<<typeid(Type).name()<<" "<<N<<std::endl;
            cudaError_t err = cudaMemcpy(std::get<I>(hd.data),std::get<I>(data),sizeof(Type)*N,cudaMemcpyDeviceToHost);
            if(err != cudaSuccess)
            {
                throw std::runtime_error("memcpyToHost 失败");
            }

            memcpyToHost<I+1>(hd);
        }

    }


    template<size_t I = 0,bool syncAtEnd = true>
    void print()
    {
        if constexpr (I < sizeof...(Vars)) {
            using VarI = std::tuple_element_t<I,std::tuple<Vars...>>;
            using Type = typename VarI::Type;
            using Tag = typename VarI::Tag; 
            size_t N = Tag::getN();
            
            dim3 block(32);
            dim3 grid((N + block.x - 1) / block.x);

        
            printOnDevice<<<grid,block>>>(N,std::get<I>(data));
            // cudaDeviceSynchronize(); // 刷新输出
            cudaDeviceSynchronize();
            printf("\n");
            print<I+1,false>();
            
    
            // 最后一层（顶层）才同步
            if constexpr (I == 0 && syncAtEnd) {
                // cudaDeviceSynchronize();
                // printf("\n");
            }
        }

    }


};


// Define variable list
using MyVars = VarList<
    Var<Y_in                        ,double >,
    Var<Y_                          ,double >,
    Var<Y0_                         ,double >,
    Var<molWeight_                  ,double >,
    Var<diameter_                   ,double >,
    Var<omega_                      ,double >,
    Var<dissociationPotential_      ,double >,
    Var<iHat_                       ,double >,
    Var<vibTempAssociativity_       ,int    >,
    Var<nonEqm_                     ,bool   >,
    Var<vibrationalList_            ,double >,
    // specie 输出
    Var<Y_out                       ,double >,
    Var<Y0_out                      ,double >,
    Var<molWeight_out               ,double >,
    Var<R_out                       ,double >,
    Var<diameter_out                ,double >,
    Var<omega_out                   ,double >,
    Var<dissociationPotential_out   ,double >,
    Var<iHat_out                    ,double >,
    Var<vibTempAssociativity_out    ,int    >,
    // janafThermoSOA
    Var<Tlow_                       ,double >,
    Var<Thigh_                      ,double >,
    Var<Tcommon_                    ,double >,
    Var<dHa_high_                   ,double >,
    Var<dHa_low_                    ,double >,
    Var<highCpCoeffs_               ,double >,
    Var<lowCpCoeffs_                ,double >,
    Var<decoupledCvCoeffs_          ,double >,
    Var<electronicList_             ,double >,
    Var<Tlow_out                    ,double >,
    Var<Thigh_out                   ,double >,
    Var<Tcommon_out                 ,double >,
    Var<highCpCoeffs_out            ,double >,
    Var<lowCpCoeffs_out             ,double >,
    Var<decoupledCvCoeffs_out       ,double >,
    Var<vibrationalList_out         ,double >,
    Var<electronicList_out          ,double >,
    // sutherlandTransportSOA
    Var<As_                         ,double >,
    Var<Ts_                         ,double >,
    Var<eta_s_                      ,double >,
    Var<Ak_                         ,double >,
    Var<Bk_                         ,double >,
    Var<Ck_                         ,double >,
    Var<As_out                      ,double >, 
    Var<Ts_out                      ,double >, 
    Var<eta_s_out                   ,double >, 
    Var<Ak_out                      ,double >, 
    Var<Bk_out                      ,double >, 
    Var<Ck_out                      ,double >
>;


void test(int ncell_,int nspecie_)
{
    ncell = ncell_;
    nspecie = nspecie_;
    HostData<MyVars> hostdata;
    
    hostdata.allocate_host();

    // ----- 输入数据 -----
    // Y_in 形状: [nspecie][ncell]
    // specie0: 0.1 0.2 0.3 0.4 0.5 H2
    // specie1: 0.6 0.7 0.8 0.9 1.0 O2
    // specie2: 1.1 1.2 1.3 1.4 1.5 N2
    hostdata.init<Var<Y_in, double>>(
        {
            0.1,0.2,0.3,0.4,0.5,
            0.6,0.7,0.8,0.9,1.0,
            1.1,1.2,1.3,1.4,1.5
        }
    );

    // 质量贡献系数 Y_
    hostdata.init<Var<Y_,   double>>({2.0,32.0,28.0 });

     // Y0_: [nspecie][nspecie] 简单 3×3
    hostdata.init<Var<Y0_,  double>>
    (
        {
            2.0 ,0.0 ,0.0 ,
            0.0 ,32.0,0.0 ,
            0.0 ,0.0 ,28.0,
        }
    );

    hostdata.init<Var<molWeight_, double>>({2.0, 32.0, 28.0});
    hostdata.init<Var<diameter_, double>>({4.07e-10,3.617e-10,4.17e-10});
    hostdata.init<Var<omega_, double>>({0.74,0.77,0.74});
    hostdata.init<Var<dissociationPotential_, double>>({4.52,5.12,9.76});
    hostdata.init<Var<iHat_, double>>({15.43,12.07,15.58});
    hostdata.init<Var<vibTempAssociativity_, int>>({1,2,3});
    hostdata.init<Var<nonEqm_, bool>>({1,1,1});



    hostdata.init<Var<vibrationalList_, double>>
    (
        {
            1   , 1   , 1   ,
            6215, 2256, 3371,
        }
    );

    // janafThermoSOA
    hostdata.init<Var< Tlow_              , double>>({1,2,3});
    hostdata.init<Var< Thigh_             , double>>({1,2,3});
    hostdata.init<Var< Tcommon_           , double>>({1,2,3});
    hostdata.init<Var< dHa_high_          , double>>({1,2,3});
    hostdata.init<Var< dHa_low_           , double>>({1,2,3});
    hostdata.init<Var< highCpCoeffs_      , double>>({1,2,3,4,5,6,7,8,9,10,11,12,13,14,15});
    hostdata.init<Var< lowCpCoeffs_       , double>>({1,2,3,4,5,6,7,8,9,10,11,12,13,14,15});
    hostdata.init<Var< decoupledCvCoeffs_ , double>>({1,2,3,4,5,6,7,8,9,10,11,12,13,14,15});
    hostdata.init<Var< electronicList_    , double>>
    (
        {
        1, 3         ,1                 ,
        0, 0         ,0                 ,
        0, 2         ,3                 ,
        0, 11391.56  ,72231.57000000001 ,
        0, 1         ,6                 ,
        0, 18984.74  ,85778.63          ,
        0, 1         ,6                 ,
        0, 47559.74  ,86050.27          ,
        0, 6         ,3                 ,
        0, 49912.42  ,95351.19          ,
        0, 3         ,1                 ,
        0, 50922.69  ,98056.36          ,
        0, 3         ,2                 ,
        0, 71898.63  ,99682.67999999999 ,
        0, 0         ,2                 ,
        0, 0         ,104897.6          ,
        0, 0         ,5                 ,
        0, 0         ,111649            ,
        0, 0         ,1                 ,
        0, 0         ,122583.6          ,
        0, 0         ,6                 ,
        0, 0         ,124885.7          ,
        0, 0         ,6                 ,
        0, 0         ,128247.6          ,
        0, 0         ,10                ,
        0, 0         ,133806.1          ,
        0, 0         ,6                 ,
        0, 0         ,140429.6          ,
        0, 0         ,6                 ,
        0, 0         ,150495.9          ,
        }    
    );





    hostdata.init<Var<As_          ,double>>({6.362e-07,1.69e-06   ,1.41e-06   });
    hostdata.init<Var<Ts_          ,double>>({72       ,127        ,111        });
    hostdata.init<Var<eta_s_       ,double>>({1.2      ,1.2        ,1.2        });
    hostdata.init<Var<Ak_          ,double>>({0.0203   ,0.0449     ,0.0268     });
    hostdata.init<Var<Bk_          ,double>>({0.429    ,-0.0826    ,0.318      });
    hostdata.init<Var<Ck_          ,double>>({-11.6    ,-9.2       ,-11.3      });



    DeviceData<MyVars> devicedata;

    devicedata.allocate_device();

    devicedata.memcpyToDevice(hostdata);

    devicedata.caculate();

    HostData<MyVars> devicedataRef;
    devicedataRef.allocate_host();
    devicedataRef.copy(hostdata);
    // 拷贝到主存Ref
    devicedata.memcpyToHost(devicedataRef);

    hostdata.caculateOnHost();

    hostdata.check(devicedataRef);
}



/** 一些固定不变的量 */
using constVars = VarList<
    Var<Y_                          ,double >,
    Var<Y0_                         ,double >,
    Var<molWeight_                  ,double >,
    Var<diameter_                   ,double >,
    Var<omega_                      ,double >,
    Var<dissociationPotential_      ,double >,
    Var<iHat_                       ,double >,
    Var<vibTempAssociativity_       ,int    >,
    Var<nonEqm_                     ,bool   >,
    Var<vibrationalList_            ,double >,
    // janafThermoSOA
    Var<Tlow_                       ,double >,
    Var<Thigh_                      ,double >,
    Var<Tcommon_                    ,double >,
    Var<dHa_high_                   ,double >,
    Var<dHa_low_                    ,double >,
    Var<highCpCoeffs_               ,double >,
    Var<lowCpCoeffs_                ,double >,
    Var<decoupledCvCoeffs_          ,double >,
    Var<electronicList_             ,double >,

    // sutherlandTransportSOA
    Var<As_                         ,double >,
    Var<Ts_                         ,double >,
    Var<eta_s_                      ,double >,
    Var<Ak_                         ,double >,
    Var<Bk_                         ,double >,
    Var<Ck_                         ,double >
>;


/** 输入变量 */
using inVars = VarList<
    Var<Y_in                        ,double >
>;

/** 输出变量 */
using outVars = VarList<
    
    // specie 输出
    Var<Y_out                       ,double >,
    Var<Y0_out                      ,double >,
    Var<molWeight_out               ,double >,
    Var<R_out                       ,double >,
    Var<diameter_out                ,double >,
    Var<omega_out                   ,double >,
    Var<dissociationPotential_out   ,double >,
    Var<iHat_out                    ,double >,
    Var<vibTempAssociativity_out    ,int    >,
    // janafThermoSOA
    Var<Tlow_out                    ,double >,
    Var<Thigh_out                   ,double >,
    Var<Tcommon_out                 ,double >,
    Var<highCpCoeffs_out            ,double >,
    Var<lowCpCoeffs_out             ,double >,
    Var<decoupledCvCoeffs_out       ,double >,
    Var<vibrationalList_out         ,double >,
    Var<electronicList_out          ,double >,
    // sutherlandTransportSOA
    Var<As_out                      ,double >, 
    Var<Ts_out                      ,double >, 
    Var<eta_s_out                   ,double >, 
    Var<Ak_out                      ,double >, 
    Var<Bk_out                      ,double >, 
    Var<Ck_out                      ,double >
>;


void test2(int _ncell,int _nspecie_)
{
    ncell = _ncell;
    nspecie = _nspecie_;
    HostData<inVars> hostData;
    
    hostData.allocate_host();
    hostData.init<Var<Y_in                        ,double >>({1,2,3,4,5,6,7,8,9,10,11,12,13,14,15});
    hostData.print();
    hostData.clear_to_zero();
}

void test_for_DeviceData_clear_to_zero()
{
    HostData<inVars> hostData;
    
    hostData.allocate_host();
    hostData.init<Var<Y_in                        ,double >>({1,2,3,4,5,6,7,8,9,10,11,12,13,14,15});
    

    DeviceData<inVars> deviceData;
    deviceData.allocate_device();
    deviceData.memcpyToDevice(hostData);
    deviceData.print();
    deviceData.clear_to_zero();
    deviceData.print();
}


void test_for_copyFromOtherDevice()
{
    HostData<inVars> hostData1;
    hostData1.allocate_host();
    hostData1.init<Var<Y_in                        ,double >>({1,2,3,4,5,6,7,8,9,10,11,12,13,14,15});
    DeviceData<inVars> deviceData1;
    deviceData1.allocate_device();
    deviceData1.memcpyToDevice(hostData1);


    using testVar = VarList<Var<Y_,double>>;
    HostData<testVar> hostData2;
    hostData2.allocate_host();
    hostData2.init<Var<Y_                        ,double >>({1,2,3});

    DeviceData<testVar> deviceData2;
    deviceData2.allocate_device();
    deviceData2.memcpyToDevice(hostData2);

    using mergeVar = VarList<
        Var<Y_in                        ,double >,
        Var<Y_                        ,double >
    >;
    DeviceData<mergeVar> deviceData3;
    // deviceData3.allocate_device();
    // deviceData2.memcpyToDevice(hostData2);

    deviceData3.copyFromOtherDevice(deviceData1);
    deviceData3.copyFromOtherDevice(deviceData2);

    deviceData3.print();
}


// 主存上
HostData<constVars> constHostData;
HostData<inVars>    inHostData;
HostData<outVars>   outHostData;

// 设备上
DeviceData<constVars>   constDeviceData;
DeviceData<inVars>      inDeviceData;
DeviceData<outVars>     outDeviceData;
DeviceData<MyVars>      allDeviceData;

// 初始化物种常量矩阵，并拷贝到设备上
void initConstData
(
    int     _nspecie                ,
    double  _RR                     ,
    int     _nVibrationalList_      ,
    int     _nCoeffs_               ,
    int     _nElectronicList_       ,
    double* _Y_                     ,
    double* _Y0_                    ,
    double* _molWeight_             ,
    double* _diameter_              ,
    double* _omega_                 ,
    double* _dissociationPotential_ ,
    double* _iHat_                  ,
    int   * _vibTempAssociativity_  ,
    bool  * _nonEqm_                ,
    double* _vibrationalList_       ,  
    double* _Tlow_                  ,
    double* _Thigh_                 ,
    double* _Tcommon_               ,
    double* _dHa_high_              ,
    double* _dHa_low_               ,
    double* _highCpCoeffs_          ,
    double* _lowCpCoeffs_           ,
    double* _decoupledCvCoeffs_     ,
    double* _electronicList_        ,          
    double* _As_                    ,
    double* _Ts_                    ,
    double* _eta_s_                 ,
    double* _Ak_                    ,
    double* _Bk_                    ,
    double* _Ck_                    
)
{
    nspecie = _nspecie;
    RR = _RR;
    nVibrationalList_ =  _nVibrationalList_ ;
    nCoeffs_ = _nCoeffs_;
    nElectronicList_ = _nElectronicList_;


    constHostData.init<Var<Y_                          ,double >>(_Y_                     );
    constHostData.init<Var<Y0_                         ,double >>(_Y0_                    );
    constHostData.init<Var<molWeight_                  ,double >>(_molWeight_             );
    constHostData.init<Var<diameter_                   ,double >>(_diameter_              );
    constHostData.init<Var<omega_                      ,double >>(_omega_                 );
    constHostData.init<Var<dissociationPotential_      ,double >>(_dissociationPotential_ );
    constHostData.init<Var<iHat_                       ,double >>(_iHat_                  );
    constHostData.init<Var<vibTempAssociativity_       ,int    >>(_vibTempAssociativity_  );
    constHostData.init<Var<nonEqm_                     ,bool   >>(_nonEqm_                );
    constHostData.init<Var<vibrationalList_            ,double >>(_vibrationalList_       );
    constHostData.init<Var<Tlow_                       ,double >>(_Tlow_                  );
    constHostData.init<Var<Thigh_                      ,double >>(_Thigh_                 );
    constHostData.init<Var<Tcommon_                    ,double >>(_Tcommon_               );
    constHostData.init<Var<dHa_high_                   ,double >>(_dHa_high_              );
    constHostData.init<Var<dHa_low_                    ,double >>(_dHa_low_               );
    constHostData.init<Var<highCpCoeffs_               ,double >>(_highCpCoeffs_          );
    constHostData.init<Var<lowCpCoeffs_                ,double >>(_lowCpCoeffs_           );
    constHostData.init<Var<decoupledCvCoeffs_          ,double >>(_decoupledCvCoeffs_     );
    constHostData.init<Var<electronicList_             ,double >>(_electronicList_        );
    constHostData.init<Var<As_                         ,double >>(_As_                    );
    constHostData.init<Var<Ts_                         ,double >>(_Ts_                    );
    constHostData.init<Var<eta_s_                      ,double >>(_eta_s_                 );
    constHostData.init<Var<Ak_                         ,double >>(_Ak_                    );
    constHostData.init<Var<Bk_                         ,double >>(_Bk_                    );
    constHostData.init<Var<Ck_                         ,double >>(_Ck_                    );

    fout<<"constHostData"<<std::endl;
    constHostData.print();

    constDeviceData.allocate_device();              // 分配显存
    constDeviceData.memcpyToDevice(constHostData); // 拷贝数据

    
}

// 初始化Y_in矩阵，并拷贝到设备上
void initInData
(
    int     _ncell                  ,
    int     _nspecie                ,
    double* _Y_in                   
)
{
    ncell = _ncell;

    inHostData.init<Var<Y_in                          ,double >>(_Y_in                     );
    fout<<"constHostData"<<std::endl;
    inHostData.print();

    inDeviceData.allocate_device();             // 分配显存
    inDeviceData.memcpyToDevice(inHostData);    // 拷贝数据
}

// 初始化
void initOuthostData()
{
    outHostData.allocate_host(); // 非拷贝回来的主存分配空间
    outDeviceData.allocate_device(); // 给cuda计算分配输数据空间
}


// 计算
void caculate()
{
    // 设备输出位置清零
    outDeviceData.clear_to_zero();
    // 聚合
    allDeviceData.copyFromOtherDevice(constDeviceData);
    allDeviceData.copyFromOtherDevice(inDeviceData);
    allDeviceData.copyFromOtherDevice(outDeviceData);
    // 计算
    allDeviceData.caculate();
    outDeviceData.memcpyToHost(outHostData);    // 搬运数据
    fout<<"caculate"<<std::endl;
    outHostData.print();
    
    
}


void printOuthostData()
{
    outDeviceData.memcpyToHost(outHostData);
    outHostData.print();
}

// 把数据取出来
void getOutHostData
(
    double* &_Y_out,                     // 混态质量 [ncell]
    double* &_Y0_out,                    // 混态各个物种所占的质量 [nspecie][ncell]
    double* &_molWeight_out,             // 混态摩尔质量 [ncell]
    double* &_R_out,                     // 混态气体常数(mol) [ncell]
    double* &_diameter_out,              // 混态分子的有效直径 [ncell]
    double* &_omega_out,                 // 混态黏度系数 [ncell]
    double* &_dissociationPotential_out, // 混态解离能 [ncell]
    double* &_iHat_out,                  // 混态第一电离能 [ncell]
    int   * &_vibTempAssociativity_out,  // 混态振动温度是否耦合标志 [ncell]
    double* &_Tlow_out,                  // [ncell]
    double* &_Thigh_out,                 // [ncell]
    double* &_Tcommon_out,               // [ncell]
    double* &_highCpCoeffs_out,          // [nCoeffs   _][ncell]
    double* &_lowCpCoeffs_out,           // [nCoeffs_][ncell]
    double* &_decoupledCvCoeffs_out,     // [nCoeffs_][ncell]
    double* &_vibrationalList_out,       // [nVibrationalList_][ncell]
    double* &_electronicList_out,        // [nElectronicList_][ncell]
    double* &_As_out,                    // [ncell]
    double* &_Ts_out,                    // [ncell]
    double* &_eta_s_out,                 // [ncell]
    double* &_Ak_out,                    // [ncell]
    double* &_Bk_out,                    // [ncell]
    double* &_Ck_out	                    // [ncell]
)
{

    _Y_out                      = outHostData.get<Var<Y_out                       ,double >>();
    _Y0_out                     = outHostData.get<Var<Y0_out                      ,double >>();
    _molWeight_out              = outHostData.get<Var<molWeight_out               ,double >>();
    _R_out                      = outHostData.get<Var<R_out                       ,double >>();
    _diameter_out               = outHostData.get<Var<diameter_out                ,double >>();
    _omega_out                  = outHostData.get<Var<omega_out                   ,double >>();
    _dissociationPotential_out  = outHostData.get<Var<dissociationPotential_out   ,double >>();
    _iHat_out                   = outHostData.get<Var<iHat_out                    ,double >>();
    _vibTempAssociativity_out   = outHostData.get<Var<vibTempAssociativity_out    ,int    >>();
    _Tlow_out                   = outHostData.get<Var<Tlow_out                    ,double >>();
    _Thigh_out                  = outHostData.get<Var<Thigh_out                   ,double >>();
    _Tcommon_out                = outHostData.get<Var<Tcommon_out                 ,double >>();
    _highCpCoeffs_out           = outHostData.get<Var<highCpCoeffs_out            ,double >>();
    _lowCpCoeffs_out            = outHostData.get<Var<lowCpCoeffs_out             ,double >>();
    _decoupledCvCoeffs_out      = outHostData.get<Var<decoupledCvCoeffs_out       ,double >>();
    _vibrationalList_out        = outHostData.get<Var<vibrationalList_out         ,double >>();
    _electronicList_out         = outHostData.get<Var<electronicList_out          ,double >>();
    _As_out                     = outHostData.get<Var<As_out                      ,double >>();
    _Ts_out                     = outHostData.get<Var<Ts_out                      ,double >>();
    _eta_s_out                  = outHostData.get<Var<eta_s_out                   ,double >>();
    _Ak_out                     = outHostData.get<Var<Ak_out                      ,double >>();
    _Bk_out                     = outHostData.get<Var<Bk_out                      ,double >>();
    _Ck_out                     = outHostData.get<Var<Ck_out                      ,double >>();
    
}