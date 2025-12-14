#ifndef TEST3_H
#define TEST3_H

extern void test(int ncell_,int nspecie_);
extern void test2(int ncell,int nspecie_);
extern void test_for_DeviceData_clear_to_zero();
extern void test_for_copyFromOtherDevice();
extern void initConstData
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
);

void initInData
(
    int     _ncell                  ,
    int     _nspecie                ,
    double* _Y_in                   
);

extern void initOuthostData();
extern void caculate();
extern void printOuthostData();
extern void getOutHostData
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
    double* &_TCommon_out,               // [ncell]
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
);
#endif