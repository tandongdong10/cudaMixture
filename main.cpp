#include <iostream>
#include "cudaMixture.H"


void test_for_caculate()
{
    static int nspecie = 3;
    static int ncell   = 5;
    static double RR = 287.0;
    static int nVibrationalList_ = 2;
    static int nCoeffs_ = 5;  // 默认是7
    static int nElectronicList_ = 30;          // 下面变长数组中的长度
    


    double Y_in[] =
        {
            0.1,0.2,0.3,0.4,0.5,
            0.6,0.7,0.8,0.9,1.0,
            1.1,1.2,1.3,1.4,1.5
        };

    // 质量贡献系数 Y_
    double Y_[] = {2.0,32.0,28.0 };

     // Y0_: [nspecie][nspecie] 简单 3×3
    double Y0_[] =
        {
            2.0 ,0.0 ,0.0 ,
            0.0 ,32.0,0.0 ,
            0.0 ,0.0 ,28.0,
        };

    double molWeight_[] ={2.0, 32.0, 28.0};
    double diameter_[] ={4.07e-10,3.617e-10,4.17e-10};
    double omega_[] ={0.74,0.77,0.74};
    double dissociationPotential_[] ={4.52,5.12,9.76};
    double iHat_[] ={15.43,12.07,15.58};
    int    vibTempAssociativity_[] ={1,2,3};
    bool   nonEqm_[] ={1,1,1};



    double vibrationalList_[] = 
    
        {
            1   , 1   , 1   ,
            6215, 2256, 3371,
        }
    ;

    // janafThermoSOA
    double Tlow_             [] = {1,2,3};
    double Thigh_            [] = {1,2,3};
    double Tcommon_          [] = {1,2,3};
    double dHa_high_         [] = {1,2,3};
    double dHa_low_          [] = {1,2,3};
    double highCpCoeffs_     [] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    double lowCpCoeffs_      [] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    double decoupledCvCoeffs_[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    double electronicList_   [] = 
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
        };





    double As_   [] = {6.362e-07,1.69e-06   ,1.41e-06   };
    double Ts_   [] = {72       ,127        ,111        };
    double eta_s_[] = {1.2      ,1.2        ,1.2        };
    double Ak_   [] = {0.0203   ,0.0449     ,0.0268     };
    double Bk_   [] = {0.429    ,-0.0826    ,0.318      };
    double Ck_   [] = {-11.6    ,-9.2       ,-11.3      };


    initConstData
    (
        nspecie                ,
        RR                     ,
        nVibrationalList_      ,
        nCoeffs_               ,
        nElectronicList_       ,
        Y_                     ,
        Y0_                    ,
        molWeight_             ,
        diameter_              ,
        omega_                 ,
        dissociationPotential_ ,
        iHat_                  ,
        vibTempAssociativity_  ,
        nonEqm_                ,
        vibrationalList_       ,  
        Tlow_                  ,
        Thigh_                 ,
        Tcommon_               ,
        dHa_high_              ,
        dHa_low_               ,
        highCpCoeffs_          ,
        lowCpCoeffs_           ,
        decoupledCvCoeffs_     ,
        electronicList_        ,          
        As_                    ,
        Ts_                    ,
        eta_s_                 ,
        Ak_                    ,
        Bk_                    ,
        Ck_                    
    );



    initInData
    (

        ncell                  ,
        nspecie                ,
        Y_in                   
    );



    initOuthostData();
    caculate();

    printOuthostData();

}

void test_for_getOutHostData()
{
    test_for_caculate();
    double * Y_out                      ;
    double * Y0_out                     ;
    double * molWeight_out              ;
    double * R_out                      ;
    double * diameter_out               ;
    double * omega_out                  ;
    double * dissociationPotential_out  ;
    double * iHat_out                   ;
    int    * vibTempAssociativity_out   ;
    double * Tlow_out                   ;
    double * Thigh_out                  ;
    double * TCommon_out                ;
    double * highCpCoeffs_out           ;
    double * lowCpCoeffs_out            ;
    double * decoupledCvCoeffs_out      ;
    double * vibrationalList_out        ;
    double * electronicList_out         ;
    double * As_out                     ;
    double * Ts_out                     ;
    double * eta_s_out                  ;
    double * Ak_out                     ;
    double * Bk_out                     ;
    double * Ck_out                     ;

    getOutHostData
    (
        Y_out                      ,
        Y0_out                     ,
        molWeight_out              ,
        R_out                      ,
        diameter_out               ,
        omega_out                  ,
        dissociationPotential_out  ,
        iHat_out                   ,
        vibTempAssociativity_out   ,
        Tlow_out                   ,
        Thigh_out                  ,
        TCommon_out                ,
        highCpCoeffs_out           ,
        lowCpCoeffs_out            ,
        decoupledCvCoeffs_out      ,
        vibrationalList_out        ,
        electronicList_out         ,
        As_out                     ,
        Ts_out                     ,
        eta_s_out                  ,
        Ak_out                     ,
        Bk_out                     ,
        Ck_out                     
    );

    for(int i=0;i<5;i++)
    {
        printf("%lf ",Y_out[i]);
    }
    printf("\n");

}

int main()
{
    int a = 5;
    test_for_getOutHostData();
}
// export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH