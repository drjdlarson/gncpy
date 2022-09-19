// Model version                  : 4.118
// Simulink Coder version         : 9.7 (R2022a) 13-Nov-2021
// C/C++ source code generated on : Sun Sep 18 18:11:46 2022
//

#include "autocode.h"
#include "rtwtypes.h"
#include <cmath>
#include <cstring>
#include <cfloat>
#include <stddef.h>
#include "zero_crossing_types.h"
#define NumBitsPerChar                 8U
#ifndef UCHAR_MAX
#include <limits.h>
#endif

#if ( UCHAR_MAX != (0xFFU) ) || ( SCHAR_MAX != (0x7F) )
#error Code was generated for compiler with different sized uchar/char. \
Consider adjusting Test hardware word size settings on the \
Hardware Implementation pane to match your compiler word sizes as \
defined in limits.h of the compiler. Alternatively, you can \
select the Test hardware is the same as production hardware option and \
select the Enable portable word sizes option on the Code Generation > \
Verification pane for ERT based targets, which will disable the \
preprocessor word size checks.
#endif

#if ( USHRT_MAX != (0xFFFFU) ) || ( SHRT_MAX != (0x7FFF) )
#error Code was generated for compiler with different sized ushort/short. \
Consider adjusting Test hardware word size settings on the \
Hardware Implementation pane to match your compiler word sizes as \
defined in limits.h of the compiler. Alternatively, you can \
select the Test hardware is the same as production hardware option and \
select the Enable portable word sizes option on the Code Generation > \
Verification pane for ERT based targets, which will disable the \
preprocessor word size checks.
#endif

#if ( UINT_MAX != (0xFFFFFFFFU) ) || ( INT_MAX != (0x7FFFFFFF) )
#error Code was generated for compiler with different sized uint/int. \
Consider adjusting Test hardware word size settings on the \
Hardware Implementation pane to match your compiler word sizes as \
defined in limits.h of the compiler. Alternatively, you can \
select the Test hardware is the same as production hardware option and \
select the Enable portable word sizes option on the Code Generation > \
Verification pane for ERT based targets, which will disable the \
preprocessor word size checks.
#endif

// Skipping ulong/long check: insufficient preprocessor integer range.

// Invariant block signals (default storage)
const bfs::Autocode::ConstBlockIO rtConstB = {
    {
        0.7F,
        -0.2F,
        0.0F,
        -0.1F,
        0.7F,
        0.2F,
        0.0F,
        0.1F,
        0.7F,
        0.1F,
        0.1F,
        -0.1F,
        0.7F,
        -0.1F,
        -0.1F,
        0.1F,
        0.7F,
        -0.1F,
        0.1F,
        0.1F,
        0.7F,
        0.1F,
        -0.1F,
        -0.1F,
        0.0F,
        0.0F,
        0.0F,
        0.0F,
        0.0F,
        0.0F,
        0.0F,
        0.0F
    }
    ,                                  // '<S4>/Transpose'
    0.333333343F
    ,                                  // '<S23>/ramp_time_intergrator signal'
    -0.333333343F
    ,                                  // '<S23>/Gain1'
    -2.0F
    // '<S366>/Gain'
};

// Constant parameters (default storage)
const bfs::Autocode::ConstParam rtConstP = {
    // Expression: [172, 172, 172, 172, 172, 172]

    //  Referenced by: '<S13>/Constant'
    { 172.0, 172.0, 172.0, 172.0, 172.0, 172.0 },

    // Computed Parameter: Constant_Value_i

    //  Referenced by: '<S22>/Constant'
    { 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F }
};

// Pooled Parameter (Mixed Expressions)
//  Referenced by:
//    '<S23>/motor_arm_ramp_integrator'
//    '<S707>/Constant'
//    '<S646>/Constant'
//    '<S692>/Unit Delay'
//    '<S685>/Unit Delay'
#define rtCP_pooled5                   (0.0)

// Pooled Parameter (Expression: 1)
//  Referenced by:
//    '<S4>/Constant'
//    '<S23>/motor_arm_ramp_integrator'
//    '<S474>/Constant'
#define rtCP_pooled6                   (1.0)

// Pooled Parameter (Mixed Expressions)
//  Referenced by:
//    '<S475>/Constant'
//    '<S693>/Constant'
#define rtCP_pooled7                   (10.0)

// Expression: const
//  Referenced by: '<S686>/Constant'
#define rtCP_Constant_Value            (15.0)

// Pooled Parameter (Mixed Expressions)
//  Referenced by:
//    '<S23>/motor_arm_ramp_integrator'
//    '<S692>/Constant'
//    '<S685>/Constant'
#define rtCP_pooled8                   (0.01)

// Expression: Aircraft.Control.wp_radius
//  Referenced by: '<S696>/Constant'
#define rtCP_Constant_Value_f          (1.5)

// Expression: Aircraft.Control.motor_spin_min
//  Referenced by: '<S4>/Constant1'
#define rtCP_Constant1_Value           (0.15)

// Expression: -100
//  Referenced by: '<S23>/Constant'
#define rtCP_Constant_Value_e          (-100.0)

// Pooled Parameter (Mixed Expressions)
//  Referenced by:
//    '<S363>/Constant'
//    '<S191>/Constant'
//    '<S690>/Constant'
//    '<S697>/Constant'
#define rtCP_pooled9                   (10.0F)

// Pooled Parameter (Mixed Expressions)
//  Referenced by:
//    '<S2>/remaining_prcnt'
//    '<S2>/remaining_time_s'
//    '<S24>/stab_pitch_rate_saturation'
//    '<S25>/stab_roll_rate_saturation'
//    '<S190>/Constant'
//    '<S194>/Saturation'
//    '<S357>/Constant'
//    '<S366>/Constant1'
//    '<S366>/P_vz1'
//    '<S652>/Constant3'
//    '<S653>/Constant3'
//    '<S656>/Constant2'
//    '<S657>/Constant2'
//    '<S659>/Constant2'
//    '<S660>/Constant2'
//    '<S661>/Constant2'
//    '<S662>/Constant2'
//    '<S663>/Constant2'
//    '<S664>/Constant2'
//    '<S536>/Constant1'
//    '<S536>/P_alt'
//    '<S590>/P_heading'
#define rtCP_pooled10                  (1.0F)

// Pooled Parameter (Mixed Expressions)
//  Referenced by:
//    '<S190>/Constant1'
//    '<S691>/Constant'
#define rtCP_pooled11                  (0.3F)

// Pooled Parameter (Mixed Expressions)
//  Referenced by:
//    '<S357>/Normalize at Zero'
//    '<S364>/Constant3'
//    '<S676>/Constant'
//    '<S136>/Constant'
//    '<S196>/Constant'
//    '<S197>/Constant'
#define rtCP_pooled12                  (0.5F)

// Pooled Parameter (Mixed Expressions)
//  Referenced by:
//    '<S196>/Constant2'
//    '<S197>/Constant2'
//    '<S703>/Constant'
#define rtCP_pooled13                  (0.1F)

// Computed Parameter: Constant2_Value
//  Referenced by: '<S194>/Constant2'
#define rtCP_Constant2_Value           (0.6724F)

// Pooled Parameter (Mixed Expressions)
//  Referenced by:
//    '<S2>/Discrete-Time Integrator'
//    '<S194>/D_vz'
#define rtCP_pooled14                  (0.005F)

// Pooled Parameter (Mixed Expressions)
//  Referenced by:
//    '<S194>/I_vz'
//    '<S137>/Constant'
#define rtCP_pooled15                  (0.05F)

// Computed Parameter: P_vz_Value
//  Referenced by: '<S194>/P_vz'
#define rtCP_P_vz_Value                (0.09F)

// Pooled Parameter (Mixed Expressions)
//  Referenced by:
//    '<S23>/Gain1'
//    '<S24>/stab_pitch_rate_saturation'
//    '<S25>/stab_roll_rate_saturation'
//    '<S194>/Gain'
//    '<S357>/Gain'
//    '<S366>/Gain'
//    '<S659>/Constant1'
//    '<S661>/Constant1'
//    '<S664>/Constant1'
//    '<S675>/Gain'
//    '<S195>/Gain'
//    '<S196>/Gain'
//    '<S419>/Gain'
//    '<S536>/Gain'
//    '<S684>/Constant'
//    '<S535>/Gain'
#define rtCP_pooled16                  (-1.0F)

// Pooled Parameter (Mixed Expressions)
//  Referenced by:
//    '<S196>/Constant1'
//    '<S197>/Constant1'
//    '<S590>/D_heading'
//    '<S590>/I_heading'
//    '<S64>/Integrator'
//    '<S118>/Integrator'
//    '<S173>/Integrator'
//    '<S340>/Integrator'
//    '<S234>/Integrator'
//    '<S287>/Integrator'
//    '<S572>/Integrator'
//    '<S627>/Integrator'
#define rtCP_pooled17                  (0.01F)

// Pooled Parameter (Mixed Expressions)
//  Referenced by:
//    '<S2>/Discrete-Time Integrator'
//    '<S8>/Constant3'
//    '<S194>/Saturation'
//    '<S364>/Constant'
//    '<S364>/Saturation'
//    '<S474>/cur_target_heading_rad'
//    '<S474>/max_v_z_mps'
//    '<S474>/max_v_hor_mps'
//    '<S474>/cur_target_pos_m'
//    '<S652>/Constant'
//    '<S653>/Constant'
//    '<S656>/Constant1'
//    '<S657>/Constant1'
//    '<S658>/Constant1'
//    '<S660>/Constant1'
//    '<S662>/Constant1'
//    '<S663>/Constant1'
//    '<S680>/Unit Delay'
//    '<S135>/UD'
//    '<S358>/Constant'
//    '<S359>/Constant'
//    '<S647>/pref_target_pos'
//    '<S705>/time_arm_valid_s'
//    '<S705>/Discrete-Time Integrator'
//    '<S479>/Constant'
//    '<S479>/Saturation'
//    '<S55>/Constant1'
//    '<S55>/Constant5'
//    '<S64>/Integrator'
//    '<S109>/Constant1'
//    '<S109>/Constant5'
//    '<S118>/Integrator'
//    '<S164>/Constant1'
//    '<S164>/Constant5'
//    '<S173>/Integrator'
//    '<S329>/Constant1'
//    '<S329>/Constant5'
//    '<S333>/UD'
//    '<S340>/Integrator'
//    '<S591>/UD'
//    '<S223>/Constant1'
//    '<S223>/Constant5'
//    '<S227>/UD'
//    '<S234>/Integrator'
//    '<S276>/Constant1'
//    '<S276>/Constant5'
//    '<S280>/UD'
//    '<S287>/Integrator'
//    '<S562>/Constant1'
//    '<S562>/Constant2'
//    '<S572>/Integrator'
//    '<S618>/Constant1'
//    '<S618>/Constant2'
//    '<S627>/Integrator'
#define rtCP_pooled18                  (0.0F)

// Pooled Parameter (Mixed Expressions)
//  Referenced by:
//    '<Root>/Gain'
//    '<S135>/TSamp'
//    '<S591>/TSamp'
//    '<S335>/Tsamp'
//    '<S229>/Tsamp'
//    '<S282>/Tsamp'
#define rtCP_pooled19                  (100.0F)

// Pooled Parameter (Mixed Expressions)
//  Referenced by:
//    '<S9>/Gain2'
//    '<S196>/Saturation'
//    '<S197>/Saturation'
#define rtCP_pooled20                  (0.175F)

// Pooled Parameter (Expression: -Aircraft.Control.pitch_angle_lim)
//  Referenced by:
//    '<S9>/Gain1'
//    '<S196>/Saturation'
//    '<S197>/Saturation'
#define rtCP_pooled21                  (-0.175F)

// Pooled Parameter (Mixed Expressions)
//  Referenced by:
//    '<S357>/Constant1'
//    '<S357>/Double'
//    '<S366>/Constant'
//    '<S474>/Constant2'
//    '<S658>/Constant2'
//    '<S536>/Constant'
#define rtCP_pooled22                  (2.0F)

// Pooled Parameter (Expression: Aircraft.Control.v_hor_max)
//  Referenced by:
//    '<S6>/Gain1'
//    '<S6>/Gain2'
//    '<S7>/Gain1'
//    '<S7>/Gain2'
//    '<S364>/Constant1'
//    '<S474>/Constant3'
//    '<S479>/Constant1'
#define rtCP_pooled23                  (5.0F)

// Pooled Parameter (Expression: Aircraft.Control.yaw_rate_max)
//  Referenced by:
//    '<S6>/Gain3'
//    '<S7>/Gain3'
//    '<S9>/Gain3'
//    '<S590>/Saturation'
#define rtCP_pooled24                  (1.74533F)

// Computed Parameter: Constant_Value_j
//  Referenced by: '<S8>/Constant'
#define rtCP_Constant_Value_j          (-35.0F)

// Pooled Parameter (Expression: [0,0,0])
//  Referenced by:
//    '<S8>/Constant1'
//    '<S696>/Constant1'
#define rtCP_pooled25_EL_0             (0.0F)
#define rtCP_pooled25_EL_1             (0.0F)
#define rtCP_pooled25_EL_2             (0.0F)

// Computed Parameter: Constant2_Value_p
//  Referenced by: '<S8>/Constant2'
#define rtCP_Constant2_Value_p_EL_0    (0.0F)
#define rtCP_Constant2_Value_p_EL_1    (0.0F)

// Computed Parameter: Constant_Value_b
//  Referenced by: '<S420>/Constant'
#define rtCP_Constant_Value_b          (1.5F)

// Pooled Parameter (Expression: 20)
//  Referenced by:
//    '<S364>/Saturation'
//    '<S479>/Saturation'
#define rtCP_pooled26                  (20.0F)

// Pooled Parameter (Mixed Expressions)
//  Referenced by:
//    '<S23>/Constant1'
//    '<S702>/Constant'
//    '<S479>/Constant2'
//    '<S479>/Constant3'
#define rtCP_pooled27                  (3.0F)

// Computed Parameter: P_alt1_Value
//  Referenced by: '<S536>/P_alt1'
#define rtCP_P_alt1_Value              (0.2F)

// Computed Parameter: Constant_Value_g
//  Referenced by: '<S644>/Constant'
#define rtCP_Constant_Value_g          (3.14159274F)

// Computed Parameter: Constant_Value_k
//  Referenced by: '<S593>/Constant'
#define rtCP_Constant_Value_k          (6.28318548F)

// Computed Parameter: Saturation_LowerSat
//  Referenced by: '<S590>/Saturation'
#define rtCP_Saturation_LowerSat       (-1.74533F)

// Pooled Parameter (Mixed Expressions)
//  Referenced by:
//    '<S27>/Constant'
//    '<S28>/Constant'
//    '<S81>/Constant'
//    '<S82>/Constant'
#define rtCP_pooled28                  (0.04F)

// Pooled Parameter (Mixed Expressions)
//  Referenced by:
//    '<S29>/Constant'
//    '<S83>/Constant'
//    '<S138>/Constant'
#define rtCP_pooled29                  (0.02F)

// Computed Parameter: Constant1_Value_l
//  Referenced by: '<S652>/Constant1'
#define rtCP_Constant1_Value_l         (900.0F)

// Computed Parameter: Constant2_Value_n
//  Referenced by: '<S652>/Constant2'
#define rtCP_Constant2_Value_n         (2100.0F)

// Computed Parameter: Constant1_Value_g
//  Referenced by: '<S653>/Constant1'
#define rtCP_Constant1_Value_g         (1100.0F)

// Computed Parameter: Constant2_Value_a
//  Referenced by: '<S653>/Constant2'
#define rtCP_Constant2_Value_a         (1900.0F)

// Pooled Parameter (Expression: 172)
//  Referenced by:
//    '<S656>/Constant'
//    '<S657>/Constant'
//    '<S658>/Constant'
//    '<S659>/Constant'
//    '<S660>/Constant'
//    '<S661>/Constant'
//    '<S662>/Constant'
//    '<S663>/Constant'
//    '<S664>/Constant'
#define rtCP_pooled30                  (172.0F)

// Pooled Parameter (Expression: 1811)
//  Referenced by:
//    '<S656>/Constant3'
//    '<S657>/Constant3'
//    '<S658>/Constant3'
//    '<S659>/Constant3'
//    '<S660>/Constant3'
//    '<S661>/Constant3'
//    '<S662>/Constant3'
//    '<S663>/Constant3'
//    '<S664>/Constant3'
#define rtCP_pooled31                  (1811.0F)

// Computed Parameter: Constant_Value_m
//  Referenced by: '<S701>/Constant'
#define rtCP_Constant_Value_m          (0.9F)

// Computed Parameter: Gain_Gain
//  Referenced by: '<S2>/Gain'
#define rtCP_Gain_Gain                 (18.95F)

// Computed Parameter: Gain1_Gain
//  Referenced by: '<S2>/Gain1'
#define rtCP_Gain1_Gain                (125650.0F)

// Computed Parameter: DelayInput1_InitialCondition
//  Referenced by: '<S645>/Delay Input1'
#define rtCP_DelayInput1_InitialConditi (-1)

// Pooled Parameter (Mixed Expressions)
//  Referenced by:
//    '<S11>/wp_reached'
//    '<S11>/Unit Delay'
//    '<S21>/Constant'
//    '<S471>/Constant'
//    '<S471>/Constant1'
//    '<S680>/Constant'
//    '<S692>/trigger_disarm'
//    '<S706>/arm_state'
//    '<S685>/battery_low_trigger'
#define rtCP_pooled32                  (false)

// Pooled Parameter (Expression: )
//  Referenced by:
//    '<S19>/Constant'
//    '<S704>/Constant'
//    '<S55>/Constant'
//    '<S55>/Constant3'
//    '<S109>/Constant'
//    '<S109>/Constant3'
//    '<S164>/Constant'
//    '<S164>/Constant3'
//    '<S329>/Constant'
//    '<S329>/Constant3'
//    '<S223>/Constant'
//    '<S223>/Constant3'
//    '<S276>/Constant'
//    '<S276>/Constant3'
//    '<S562>/Constant4'
//    '<S562>/Constant6'
//    '<S618>/Constant4'
//    '<S618>/Constant6'
#define rtCP_pooled33                  (1)

// Pooled Parameter (Expression: )
//  Referenced by:
//    '<S55>/Constant2'
//    '<S55>/Constant4'
//    '<S109>/Constant2'
//    '<S109>/Constant4'
//    '<S164>/Constant2'
//    '<S164>/Constant4'
//    '<S329>/Constant2'
//    '<S329>/Constant4'
//    '<S223>/Constant2'
//    '<S223>/Constant4'
//    '<S276>/Constant2'
//    '<S276>/Constant4'
//    '<S562>/Constant5'
//    '<S562>/Constant7'
//    '<S618>/Constant5'
//    '<S618>/Constant7'
#define rtCP_pooled34                  (-1)

// Pooled Parameter (Expression: )
//  Referenced by:
//    '<S16>/Constant'
//    '<S18>/Constant'
//    '<S696>/rtl_submode'
//    '<S697>/wp_submode'
#define rtCP_pooled35                  (0)

// Pooled Parameter (Expression: )
//  Referenced by:
//    '<S15>/Constant'
//    '<S678>/Constant'
//    '<S688>/Constant'
#define rtCP_pooled36                  (4)

// Pooled Parameter (Mixed Expressions)
//  Referenced by:
//    '<S17>/Constant'
//    '<S21>/rtl_mode'
//    '<S694>/Constant'
//    '<S687>/Constant'
#define rtCP_pooled37                  (3)

// Pooled Parameter (Expression: )
//  Referenced by:
//    '<S20>/Constant'
//    '<S695>/Constant'
#define rtCP_pooled38                  (2)

// Pooled Parameter (Mixed Expressions)
//  Referenced by:
//    '<S21>/land_mode'
//    '<S683>/Constant'
//    '<S689>/Constant'
#define rtCP_pooled39                  (5)

// Computed Parameter: DiscreteTimeIntegrator_gainval
//  Referenced by: '<S705>/Discrete-Time Integrator'
#define rtCP_DiscreteTimeIntegrator_gai (41)

extern real32_T rt_remf_snf(real32_T u0, real32_T u1);
extern real32_T rt_atan2f_snf(real32_T u0, real32_T u1);

//===========*
//  Constants *
// ===========
#define RT_PI                          3.14159265358979323846
#define RT_PIF                         3.1415927F
#define RT_LN_10                       2.30258509299404568402
#define RT_LN_10F                      2.3025851F
#define RT_LOG10E                      0.43429448190325182765
#define RT_LOG10EF                     0.43429449F
#define RT_E                           2.7182818284590452354
#define RT_EF                          2.7182817F

//
//  UNUSED_PARAMETER(x)
//    Used to specify that a function parameter (argument) is required but not
//    accessed by the function body.
#ifndef UNUSED_PARAMETER
#if defined(__LCC__)
#define UNUSED_PARAMETER(x)                                      // do nothing
#else

//
//  This is the semi-ANSI standard way of indicating that an
//  unused function parameter is required.
#define UNUSED_PARAMETER(x)            (void) (x)
#endif
#endif

extern "C" {
    real_T rtInf;
    real_T rtMinusInf;
    real_T rtNaN;
    real32_T rtInfF;
    real32_T rtMinusInfF;
    real32_T rtNaNF;
}
    extern "C"
{
    //
    // Initialize rtInf needed by the generated code.
    // Inf is initialized as non-signaling. Assumes IEEE.
    //
    static real_T rtGetInf(void)
    {
        size_t bitsPerReal = sizeof(real_T) * (NumBitsPerChar);
        real_T inf = 0.0;
        if (bitsPerReal == 32U) {
            inf = rtGetInfF();
        } else {
            union {
                LittleEndianIEEEDouble bitVal;
                real_T fltVal;
            } tmpVal;

            tmpVal.bitVal.words.wordH = 0x7FF00000U;
            tmpVal.bitVal.words.wordL = 0x00000000U;
            inf = tmpVal.fltVal;
        }

        return inf;
    }

    //
    // Initialize rtInfF needed by the generated code.
    // Inf is initialized as non-signaling. Assumes IEEE.
    //
    static real32_T rtGetInfF(void)
    {
        IEEESingle infF;
        infF.wordL.wordLuint = 0x7F800000U;
        return infF.wordL.wordLreal;
    }

    //
    // Initialize rtMinusInf needed by the generated code.
    // Inf is initialized as non-signaling. Assumes IEEE.
    //
    static real_T rtGetMinusInf(void)
    {
        size_t bitsPerReal = sizeof(real_T) * (NumBitsPerChar);
        real_T minf = 0.0;
        if (bitsPerReal == 32U) {
            minf = rtGetMinusInfF();
        } else {
            union {
                LittleEndianIEEEDouble bitVal;
                real_T fltVal;
            } tmpVal;

            tmpVal.bitVal.words.wordH = 0xFFF00000U;
            tmpVal.bitVal.words.wordL = 0x00000000U;
            minf = tmpVal.fltVal;
        }

        return minf;
    }

    //
    // Initialize rtMinusInfF needed by the generated code.
    // Inf is initialized as non-signaling. Assumes IEEE.
    //
    static real32_T rtGetMinusInfF(void)
    {
        IEEESingle minfF;
        minfF.wordL.wordLuint = 0xFF800000U;
        return minfF.wordL.wordLreal;
    }
}

extern "C" {
    //
    // Initialize rtNaN needed by the generated code.
    // NaN is initialized as non-signaling. Assumes IEEE.
    //
    static real_T rtGetNaN(void)
    {
        size_t bitsPerReal = sizeof(real_T) * (NumBitsPerChar);
        real_T nan = 0.0;
        if (bitsPerReal == 32U) {
            nan = rtGetNaNF();
        } else {
            union {
                LittleEndianIEEEDouble bitVal;
                real_T fltVal;
            } tmpVal;

            tmpVal.bitVal.words.wordH = 0xFFF80000U;
            tmpVal.bitVal.words.wordL = 0x00000000U;
            nan = tmpVal.fltVal;
        }

        return nan;
    }

    //
    // Initialize rtNaNF needed by the generated code.
    // NaN is initialized as non-signaling. Assumes IEEE.
    //
    static real32_T rtGetNaNF(void)
    {
        IEEESingle nanF = { { 0.0F } };

        nanF.wordL.wordLuint = 0xFFC00000U;
        return nanF.wordL.wordLreal;
    }
}
    extern "C"
{
    //
    // Initialize the rtInf, rtMinusInf, and rtNaN needed by the
    // generated code. NaN is initialized as non-signaling. Assumes IEEE.
    //
    static void rt_InitInfAndNaN(size_t realSize)
    {
        (void) (realSize);
        rtNaN = rtGetNaN();
        rtNaNF = rtGetNaNF();
        rtInf = rtGetInf();
        rtInfF = rtGetInfF();
        rtMinusInf = rtGetMinusInf();
        rtMinusInfF = rtGetMinusInfF();
    }

    // Test if value is infinite
    static boolean_T rtIsInf(real_T value)
    {
        return (boolean_T)((value==rtInf || value==rtMinusInf) ? 1U : 0U);
    }

    // Test if single-precision value is infinite
    static boolean_T rtIsInfF(real32_T value)
    {
        return (boolean_T)(((value)==rtInfF || (value)==rtMinusInfF) ? 1U : 0U);
    }

    // Test if value is not a number
    static boolean_T rtIsNaN(real_T value)
    {
        boolean_T result = (boolean_T) 0;
        size_t bitsPerReal = sizeof(real_T) * (NumBitsPerChar);
        if (bitsPerReal == 32U) {
            result = rtIsNaNF((real32_T)value);
        } else {
            union {
                LittleEndianIEEEDouble bitVal;
                real_T fltVal;
            } tmpVal;

            tmpVal.fltVal = value;
            result = (boolean_T)((tmpVal.bitVal.words.wordH & 0x7FF00000) ==
                                 0x7FF00000 &&
                                 ( (tmpVal.bitVal.words.wordH & 0x000FFFFF) != 0
                                  ||
                                  (tmpVal.bitVal.words.wordL != 0) ));
        }

        return result;
    }

    // Test if single-precision value is not a number
    static boolean_T rtIsNaNF(real32_T value)
    {
        IEEESingle tmp;
        tmp.wordL.wordLreal = value;
        return (boolean_T)( (tmp.wordL.wordLuint & 0x7F800000) == 0x7F800000 &&
                           (tmp.wordL.wordLuint & 0x007FFFFF) != 0 );
    }
}

namespace bfs
{
    //
    // Output and update for atomic system:
    //    '<S656>/remap'
    //    '<S657>/remap'
    //    '<S658>/remap'
    //    '<S660>/remap'
    //    '<S662>/remap'
    //    '<S663>/remap'
    //
    void Autocode::remap(real32_T rtu_raw_in, real32_T rtu_in_min, real32_T
                         rtu_in_max, real32_T rtu_out_min, real32_T rtu_out_max,
                         real32_T *rty_norm_out)
    {
        // MATLAB Function 'command selection/e_stop_norm/remap': '<S665>:1'
        // '<S665>:1:2' norm_out = (raw_in - in_min) * (out_max - out_min)/(in_max-in_min) + out_min;
        *rty_norm_out = (((rtu_raw_in - rtu_in_min) * (rtu_out_max - rtu_out_min))
                         / (rtu_in_max - rtu_in_min)) + rtu_out_min;
    }
}

real32_T rt_remf_snf(real32_T u0, real32_T u1)
{
    real32_T y;
    if (static_cast<boolean_T>(static_cast<int32_T>(((static_cast<boolean_T>(
             static_cast<int32_T>((rtIsNaNF(u0) ? (static_cast<int32_T>(1)) : (
                static_cast<int32_T>(0))) | (rtIsNaNF(u1) ? (static_cast<int32_T>
                (1)) : (static_cast<int32_T>(0)))))) ? (static_cast<int32_T>(1))
           : (static_cast<int32_T>(0))) | (rtIsInfF(u0) ? (static_cast<int32_T>
            (1)) : (static_cast<int32_T>(0)))))) {
        y = (rtNaNF);
    } else if (rtIsInfF(u1)) {
        y = u0;
    } else {
        if (u1 < 0.0F) {
            y = std::ceil(u1);
        } else {
            y = std::floor(u1);
        }

        if (static_cast<boolean_T>(static_cast<int32_T>(((u1 != 0.0F) ? (
                static_cast<int32_T>(1)) : (static_cast<int32_T>(0))) & ((u1 !=
                y) ? (static_cast<int32_T>(1)) : (static_cast<int32_T>(0)))))) {
            real32_T q;
            q = std::abs(u0 / u1);
            if (static_cast<boolean_T>(static_cast<int32_T>(((std::abs(q - std::
                     floor(q + 0.5F)) > (FLT_EPSILON * q)) ?
                   (static_cast<int32_T>(1)) : (static_cast<int32_T>(0))) ^ 1)))
            {
                y = 0.0F * u0;
            } else {
                y = std::fmod(u0, u1);
            }
        } else {
            y = std::fmod(u0, u1);
        }
    }

    return y;
}

namespace bfs
{
    // Function for MATLAB Function: '<S647>/determine_prev_tar_pos'
    void Autocode::cosd(real32_T *x)
    {
        if (static_cast<boolean_T>(static_cast<int32_T>((rtIsInfF(*x) ? (
                static_cast<int32_T>(1)) : (static_cast<int32_T>(0))) |
              (rtIsNaNF(*x) ? (static_cast<int32_T>(1)) : (static_cast<int32_T>
                (0)))))) {
            *x = (rtNaNF);
        } else {
            real32_T absx;
            real32_T b_x;
            b_x = rt_remf_snf(*x, 360.0F);
            absx = std::abs(b_x);
            if (absx > 180.0F) {
                if (b_x > 0.0F) {
                    b_x -= 360.0F;
                } else {
                    b_x += 360.0F;
                }

                absx = std::abs(b_x);
            }

            if (absx <= 45.0F) {
                b_x *= 0.0174532924F;
                *x = std::cos(b_x);
            } else {
                int8_T n;
                if (absx <= 135.0F) {
                    if (b_x > 0.0F) {
                        b_x = 0.0174532924F * (b_x - 90.0F);
                        n = 1;
                    } else {
                        b_x = 0.0174532924F * (b_x + 90.0F);
                        n = -1;
                    }
                } else if (b_x > 0.0F) {
                    b_x = 0.0174532924F * (b_x - 180.0F);
                    n = 2;
                } else {
                    b_x = 0.0174532924F * (b_x + 180.0F);
                    n = -2;
                }

                if (n == 1) {
                    *x = -std::sin(b_x);
                } else if (n == -1) {
                    *x = std::sin(b_x);
                } else {
                    *x = -std::cos(b_x);
                }
            }
        }
    }

    // Function for MATLAB Function: '<S647>/determine_prev_tar_pos'
    void Autocode::sind(real32_T *x)
    {
        if (static_cast<boolean_T>(static_cast<int32_T>((rtIsInfF(*x) ? (
                static_cast<int32_T>(1)) : (static_cast<int32_T>(0))) |
              (rtIsNaNF(*x) ? (static_cast<int32_T>(1)) : (static_cast<int32_T>
                (0)))))) {
            *x = (rtNaNF);
        } else {
            real32_T absx;
            real32_T b_x;
            b_x = rt_remf_snf(*x, 360.0F);
            absx = std::abs(b_x);
            if (absx > 180.0F) {
                if (b_x > 0.0F) {
                    b_x -= 360.0F;
                } else {
                    b_x += 360.0F;
                }

                absx = std::abs(b_x);
            }

            if (absx <= 45.0F) {
                b_x *= 0.0174532924F;
                *x = std::sin(b_x);
            } else {
                int8_T n;
                if (absx <= 135.0F) {
                    if (b_x > 0.0F) {
                        b_x = 0.0174532924F * (b_x - 90.0F);
                        n = 1;
                    } else {
                        b_x = 0.0174532924F * (b_x + 90.0F);
                        n = -1;
                    }
                } else if (b_x > 0.0F) {
                    b_x = 0.0174532924F * (b_x - 180.0F);
                    n = 2;
                } else {
                    b_x = 0.0174532924F * (b_x + 180.0F);
                    n = -2;
                }

                if (n == 1) {
                    *x = std::cos(b_x);
                } else if (n == -1) {
                    *x = -std::cos(b_x);
                } else {
                    *x = -std::sin(b_x);
                }
            }
        }
    }

    //
    // Function for MATLAB Function: '<S647>/determine_prev_tar_pos'
    // function ecef_pos = lla_to_ECEF(lla)
    //  lat, lon, alt; lat/lon in degrees, alt in m
    //
    void Autocode::lla_to_ECEF(const real32_T lla[3], real32_T ecef_pos[3])
    {
        real32_T b;
        real32_T c;
        real32_T c_lat;
        real32_T d;
        real32_T re;

        // 'lla_to_ECEF:4' FLATTENING = 1 / 298.257223563;
        // 'lla_to_ECEF:5' ECCENTRICITY = sqrt(FLATTENING * (2 - FLATTENING));
        // 'lla_to_ECEF:7' re = calc_ew_rad(lla(1));
        // 'lla_to_ECEF:22' FLATTENING = 1 / 298.257223563;
        // 'lla_to_ECEF:23' ECCENTRICITY = sqrt(FLATTENING * (2 - FLATTENING));
        // 'lla_to_ECEF:24' EQ_RAD = 6378137;
        // 'lla_to_ECEF:26' ew_rad = EQ_RAD / sqrt(1 - ECCENTRICITY^2 * sind(lat)^2);
        re = lla[0];
        sind(&re);
        re = 6.378137E+6F / std::sqrt(1.0F - (0.00669438F * (re * re)));

        // 'lla_to_ECEF:8' c_lat = cosd(lla(1));
        c_lat = lla[0];
        cosd(&c_lat);

        // 'lla_to_ECEF:9' s_lat = sind(lla(1));
        // 'lla_to_ECEF:10' c_lon = cosd(lla(2));
        // 'lla_to_ECEF:11' s_lon = sind(lla(2));
        // 'lla_to_ECEF:13' x = (re + lla(3)) * c_lat * c_lon;
        // 'lla_to_ECEF:14' y = (re + lla(3)) * c_lat * s_lon;
        // 'lla_to_ECEF:15' z = ((1 - ECCENTRICITY^2) * re + lla(3)) * s_lat;
        // 'lla_to_ECEF:17' ecef_pos = [x; y; z];
        b = lla[1];
        cosd(&b);
        c = lla[1];
        sind(&c);
        d = lla[0];
        sind(&d);
        ecef_pos[0] = ((re + lla[2]) * c_lat) * b;
        ecef_pos[1] = ((re + lla[2]) * c_lat) * c;
        ecef_pos[2] = ((0.993305624F * re) + lla[2]) * d;
    }
}

real32_T rt_atan2f_snf(real32_T u0, real32_T u1)
{
    real32_T y;
    if (static_cast<boolean_T>(static_cast<int32_T>((rtIsNaNF(u0) ? (
            static_cast<int32_T>(1)) : (static_cast<int32_T>(0))) | (rtIsNaNF(u1)
           ? (static_cast<int32_T>(1)) : (static_cast<int32_T>(0)))))) {
        y = (rtNaNF);
    } else if (static_cast<boolean_T>(static_cast<int32_T>((rtIsInfF(u0) ? (
                   static_cast<int32_T>(1)) : (static_cast<int32_T>(0))) &
                 (rtIsInfF(u1) ? (static_cast<int32_T>(1)) :
                  (static_cast<int32_T>(0)))))) {
        int32_T tmp;
        int32_T tmp_0;
        if (u1 > 0.0F) {
            tmp = 1;
        } else {
            tmp = -1;
        }

        if (u0 > 0.0F) {
            tmp_0 = 1;
        } else {
            tmp_0 = -1;
        }

        y = std::atan2(static_cast<real32_T>(tmp_0), static_cast<real32_T>(tmp));
    } else if (u1 == 0.0F) {
        if (u0 > 0.0F) {
            y = RT_PIF / 2.0F;
        } else if (u0 < 0.0F) {
            y = -(RT_PIF / 2.0F);
        } else {
            y = 0.0F;
        }
    } else {
        y = std::atan2(u0, u1);
    }

    return y;
}

namespace bfs
{
    // Model step function
    void Autocode::Run(const SysData &sys, const SensorData &sensor, const
                       NavData &nav, const TelemData &telem, VmsData *ctrl)
    {
        const MissionItem *expl_temp;
        real_T rtb_Product[8];
        real_T rtb_UnitDelay;
        int32_T i;
        int32_T rtb_Selector_y;
        real32_T rtb_aux[24];
        real32_T in_avg_0[9];
        real32_T rtb_Command[8];
        real32_T rtb_Switch2[8];
        real32_T i_0[3];
        real32_T rtb_Switch_e[3];
        real32_T rtb_target_pos[3];
        real32_T tmp[3];
        const real32_T *tmp_0;
        real32_T c_lon;
        real32_T in_avg;
        real32_T in_deadband_low;
        real32_T in_deadband_range;
        real32_T out_avg;
        real32_T rtb_DataTypeConversion2;
        real32_T rtb_DataTypeConversion3;
        real32_T rtb_Gain_h;
        real32_T rtb_Integrator_e;
        real32_T rtb_Sum_h;
        real32_T rtb_TmpSignalConversionAtProd_0;
        real32_T rtb_TmpSignalConversionAtProd_1;
        real32_T rtb_Transpose_idx_1;
        real32_T rtb_battery_failsafe_flag;
        real32_T rtb_pitch_angle_cmd_rad;
        real32_T rtb_radio_failsafe_flag;
        real32_T rtb_roll_angle_cmd_rad;
        real32_T rtb_stab_pitch_rate_saturation;
        real32_T rtb_stab_roll_rate_saturation;
        real32_T rtb_throttle_cc;
        real32_T rtb_yaw_rate_cmd_radps_a;
        real32_T rtb_yaw_rate_cmd_radps_f;
        real32_T s_lon;
        int16_T rtb_cnt_g[8];
        int8_T rtb_Switch1_a;
        int8_T rtb_Switch2_b;
        int8_T rtb_Switch2_c;
        boolean_T rtb_Compare;
        boolean_T rtb_Compare_d;
        boolean_T rtb_Compare_e;
        boolean_T rtb_Compare_hs;
        boolean_T rtb_Compare_i;
        boolean_T rtb_Equal1;
        boolean_T rtb_OR1;
        boolean_T rtb_OR1_f;
        boolean_T rtb_enable;
        boolean_T zcEvent;
        UNUSED_PARAMETER(sys);

        // DataTypeConversion: '<S14>/Data Type Conversion2' incorporates:
        //   Inport: '<Root>/Sensor Data'
        rtb_DataTypeConversion2 = static_cast<real32_T>(sensor.inceptor.ch[3]);

        // MATLAB Function: '<S664>/remap_with_deadband' incorporates:
        //   Constant: '<S664>/Constant'
        //   Constant: '<S664>/Constant1'
        //   Constant: '<S664>/Constant2'
        //   Constant: '<S664>/Constant3'
        //   Inport: '<Root>/Telemetry Data'

        // MATLAB Function 'command selection/yaw_norm_deadband/remap_with_deadband': '<S673>:1'
        // '<S673>:1:2' in_avg = (in_min + in_max) / 2;
        in_avg = (rtCP_pooled30 + rtCP_pooled31) / 2.0F;

        // '<S673>:1:3' out_avg = (out_min + out_max) / 2;
        out_avg = (rtCP_pooled16 + rtCP_pooled10) / 2.0F;

        // '<S673>:1:5' in_deadband_range = (in_max - in_min) * deadband / 2;
        in_deadband_range = ((rtCP_pooled31 - rtCP_pooled30) * telem.param[22]) /
            2.0F;

        // '<S673>:1:6' in_deadband_low = in_avg - in_deadband_range;
        in_deadband_low = in_avg - in_deadband_range;

        // '<S673>:1:7' in_deadband_hi = in_avg + in_deadband_range;
        in_avg += in_deadband_range;

        // '<S673>:1:9' if raw_in < in_deadband_low
        if (rtb_DataTypeConversion2 < in_deadband_low) {
            // '<S673>:1:10' norm_out = (raw_in - in_deadband_low) .* (out_max - out_avg)./(in_deadband_low - in_min) + out_avg;
            out_avg += ((rtb_DataTypeConversion2 - in_deadband_low) *
                        (rtCP_pooled10 - out_avg)) / (in_deadband_low -
                rtCP_pooled30);
        } else if (static_cast<boolean_T>(static_cast<int32_T>
                    (((rtb_DataTypeConversion2 > in_deadband_low) ? (
                       static_cast<int32_T>(1)) : (static_cast<int32_T>(0))) &
                     ((rtb_DataTypeConversion2 < in_avg) ? (static_cast<int32_T>
                       (1)) : (static_cast<int32_T>(0)))))) {
            // '<S673>:1:11' elseif raw_in > in_deadband_low && raw_in < in_deadband_hi
            // '<S673>:1:12' norm_out = out_avg;
        } else {
            // '<S673>:1:13' else
            // '<S673>:1:14' norm_out = (raw_in - in_deadband_hi) .* (out_max - out_avg)./(in_max - in_deadband_hi) + out_avg;
            out_avg += ((rtb_DataTypeConversion2 - in_avg) * (rtCP_pooled10 -
                         out_avg)) / (rtCP_pooled31 - in_avg);
        }

        // End of MATLAB Function: '<S664>/remap_with_deadband'

        // DataTypeConversion: '<S14>/Data Type Conversion7' incorporates:
        //   Inport: '<Root>/Sensor Data'
        in_deadband_low = static_cast<real32_T>(sensor.inceptor.ch[4]);

        // MATLAB Function: '<S658>/remap' incorporates:
        //   Constant: '<S658>/Constant'
        //   Constant: '<S658>/Constant1'
        //   Constant: '<S658>/Constant2'
        //   Constant: '<S658>/Constant3'
        remap(in_deadband_low, rtCP_pooled30, rtCP_pooled31, rtCP_pooled18,
              rtCP_pooled22, &rtb_DataTypeConversion2);

        // DataTypeConversion: '<S680>/Cast'
        rtb_Switch2_b = static_cast<int8_T>(std::floor(out_avg));

        // RelationalOperator: '<S704>/Compare' incorporates:
        //   Constant: '<S704>/Constant'
        rtb_Compare = (rtb_Switch2_b >= rtCP_pooled33);

        // DataTypeConversion: '<S14>/Data Type Conversion1' incorporates:
        //   Inport: '<Root>/Sensor Data'
        in_deadband_low = static_cast<real32_T>(sensor.inceptor.ch[6]);

        // MATLAB Function: '<S656>/remap' incorporates:
        //   Constant: '<S656>/Constant'
        //   Constant: '<S656>/Constant1'
        //   Constant: '<S656>/Constant2'
        //   Constant: '<S656>/Constant3'
        remap(in_deadband_low, rtCP_pooled30, rtCP_pooled31, rtCP_pooled18,
              rtCP_pooled10, &in_avg);

        // RelationalOperator: '<S676>/Compare' incorporates:
        //   Constant: '<S676>/Constant'
        rtb_Compare_e = (in_avg > rtCP_pooled12);

        // UnitDelay: '<S680>/Unit Delay'
        in_deadband_low = rtDWork.DiscreteTimeIntegrator;

        // Switch: '<S680>/Switch' incorporates:
        //   Constant: '<S680>/Constant'
        if (rtb_Compare_e) {
            rtb_Compare_d = rtCP_pooled32;
        } else {
            // RelationalOperator: '<S702>/Compare' incorporates:
            //   Constant: '<S702>/Constant'
            rtb_Compare_d = (in_deadband_low >= rtCP_pooled27);
        }

        // End of Switch: '<S680>/Switch'

        // Outputs for Triggered SubSystem: '<S680>/manual_arming' incorporates:
        //   TriggerPort: '<S706>/Trigger'
        zcEvent = (rtb_Compare_d != (static_cast<uint32_T>
                    (rtPrevZCSigState.manual_arming_Trig_ZCE) == POS_ZCSIG)) & (
            static_cast<uint32_T>(rtPrevZCSigState.manual_arming_Trig_ZCE) !=
            UNINITIALIZED_ZCSIG);
        if (zcEvent) {
            // SignalConversion generated from: '<S706>/yaw_arm'
            rtDWork.yaw_arm = rtb_Compare;
        }

        rtPrevZCSigState.manual_arming_Trig_ZCE = rtb_Compare_d ?
            (static_cast<ZCSigState>(1)) : (static_cast<ZCSigState>(0));

        // End of Outputs for SubSystem: '<S680>/manual_arming'

        // Logic: '<S21>/nav_init AND motor_enable' incorporates:
        //   Inport: '<Root>/Navigation Filter Data'
        rtb_Compare = rtDWork.yaw_arm & nav.nav_initialized;

        // RelationalOperator: '<S684>/Compare' incorporates:
        //   Constant: '<S684>/Constant'
        //   Inport: '<Root>/Sensor Data'
        rtb_Compare_d = (sensor.power_module.voltage_v <= rtCP_pooled16);

        // Outputs for Enabled SubSystem: '<S681>/disarm motor' incorporates:
        //   EnablePort: '<S685>/Enable'
        if (rtb_Compare_d) {
            if (static_cast<boolean_T>(static_cast<int32_T>
                 ((rtDWork.disarmmotor_MODE_k ? (
                    static_cast<int32_T>(1)) : (static_cast<int32_T>(0))) ^ 1)))
            {
                // InitializeConditions for UnitDelay: '<S685>/Unit Delay'
                rtDWork.UnitDelay_DSTATE_m = rtCP_pooled5;
                rtDWork.disarmmotor_MODE_k = true;
            }

            // UnitDelay: '<S685>/Unit Delay'
            rtb_UnitDelay = rtDWork.UnitDelay_DSTATE_m;

            // Sum: '<S685>/Sum' incorporates:
            //   Constant: '<S685>/Constant'
            rtb_UnitDelay += rtCP_pooled8;

            // RelationalOperator: '<S686>/Compare' incorporates:
            //   Constant: '<S686>/Constant'
            rtDWork.Compare_d = (rtb_UnitDelay > rtCP_Constant_Value);

            // Update for UnitDelay: '<S685>/Unit Delay'
            rtDWork.UnitDelay_DSTATE_m = rtb_UnitDelay;
        } else {
            rtDWork.disarmmotor_MODE_k = false;
        }

        // End of Outputs for SubSystem: '<S681>/disarm motor'

        // DataTypeConversion: '<S21>/mode_type_conversion'
        in_deadband_low = std::abs(rtb_DataTypeConversion2);
        if (in_deadband_low < 8.388608E+6F) {
            if (in_deadband_low >= 0.5F) {
                in_deadband_low = std::floor(rtb_DataTypeConversion2 + 0.5F);
            } else {
                in_deadband_low = rtb_DataTypeConversion2 * 0.0F;
            }
        } else {
            in_deadband_low = rtb_DataTypeConversion2;
        }

        rtb_Switch2_b = static_cast<int8_T>(in_deadband_low);

        // End of DataTypeConversion: '<S21>/mode_type_conversion'

        // RelationalOperator: '<S683>/Compare' incorporates:
        //   Constant: '<S683>/Constant'
        rtb_Compare_d = (rtb_Switch2_b != rtCP_pooled39);

        // Logic: '<S681>/AND1'
        rtb_Compare_d = rtb_Compare_d & rtb_Compare;

        // Logic: '<S681>/AND'
        rtb_Compare_d = rtDWork.Compare_d & rtb_Compare_d;

        // RelationalOperator: '<S687>/Compare' incorporates:
        //   Constant: '<S687>/Constant'
        rtb_OR1 = (rtb_Switch2_b != rtCP_pooled37);

        // RelationalOperator: '<S688>/Compare' incorporates:
        //   Constant: '<S688>/Constant'
        rtb_OR1_f = (rtb_Switch2_b != rtCP_pooled36);

        // RelationalOperator: '<S689>/Compare' incorporates:
        //   Constant: '<S689>/Constant'
        rtb_Equal1 = (rtb_Switch2_b != rtCP_pooled39);

        // Logic: '<S682>/NOT'
        rtb_Compare_hs = rtb_Compare ^ 1;

        // Logic: '<S682>/NOR'
        rtb_OR1 = static_cast<boolean_T>(static_cast<int32_T>
            (((static_cast<boolean_T>(static_cast<int32_T>
            (((static_cast<boolean_T>(static_cast<int32_T>((rtb_OR1 ? (
            static_cast<int32_T>(1)) : (static_cast<int32_T>(0))) ^ 1))) ? (
            static_cast<int32_T>(1)) : (static_cast<int32_T>(0))) & ((
            static_cast<boolean_T>(static_cast<int32_T>((rtb_OR1_f ? (
            static_cast<int32_T>(1)) : (static_cast<int32_T>(0))) ^ 1))) ? (
            static_cast<int32_T>(1)) : (static_cast<int32_T>(0)))))) ? (
            static_cast<int32_T>(1)) : (static_cast<int32_T>(0))) & ((
            static_cast<boolean_T>(static_cast<int32_T>((rtb_Equal1 ? (
            static_cast<int32_T>(1)) : (static_cast<int32_T>(0))) ^ 1))) ? (
            static_cast<int32_T>(1)) : (static_cast<int32_T>(0))))) &
            static_cast<boolean_T>(static_cast<int32_T>((rtb_Compare_hs ? (
            static_cast<int32_T>(1)) : (static_cast<int32_T>(0))) ^ 1));

        // Logic: '<S682>/AND' incorporates:
        //   Inport: '<Root>/Sensor Data'
        zcEvent = sensor.inceptor.failsafe & rtb_OR1;

        // Switch: '<S21>/Switch1' incorporates:
        //   Constant: '<S21>/land_mode'
        //   Switch: '<S21>/Switch2'
        if (rtb_Compare_d) {
            rtb_Switch2_c = rtCP_pooled39;
        } else if (zcEvent) {
            // Switch: '<S21>/Switch2' incorporates:
            //   Constant: '<S21>/rtl_mode'
            rtb_Switch2_c = rtCP_pooled37;
        } else {
            // Switch: '<S21>/Switch2'
            rtb_Switch2_c = rtb_Switch2_b;
        }

        // End of Switch: '<S21>/Switch1'

        // Logic: '<S21>/OR'
        rtb_Compare_hs = rtb_Compare_d | zcEvent;

        // RelationalOperator: '<S695>/Compare' incorporates:
        //   Constant: '<S695>/Constant'
        rtb_Compare_i = (rtb_Switch2_c == rtCP_pooled38);

        // Outputs for Enabled SubSystem: '<S679>/waypoint submodes' incorporates:
        //   EnablePort: '<S697>/Enable'
        if (rtb_Compare_i) {
            uint16_T rtb_Selector_cmd;

            // Outputs for Enabled SubSystem: '<Root>/WAYPOINT CONTROLLER' incorporates:
            //   EnablePort: '<S11>/Enable'

            // Outputs for Enabled SubSystem: '<S11>/determine target' incorporates:
            //   EnablePort: '<S474>/Enable'

            // Outputs for Enabled SubSystem: '<S474>/calc_prev_target_pos' incorporates:
            //   EnablePort: '<S647>/Enable'

            // Selector: '<S697>/Selector' incorporates:
            //   Inport: '<Root>/Telemetry Data'
            //   Selector: '<S474>/Selector'
            //   Selector: '<S475>/Selector'
            //   Selector: '<S647>/Selector1'
            expl_temp = &telem.flight_plan[telem.current_waypoint];

            // End of Outputs for SubSystem: '<S474>/calc_prev_target_pos'
            // End of Outputs for SubSystem: '<S11>/determine target'
            // End of Outputs for SubSystem: '<Root>/WAYPOINT CONTROLLER'
            rtb_Compare_i = expl_temp->autocontinue;
            rtb_Selector_cmd = expl_temp->cmd;
            i = expl_temp->x;
            rtb_Selector_y = expl_temp->y;
            in_deadband_range = expl_temp->z;

            // MATLAB Function: '<S697>/determine_target_pos' incorporates:
            //   Inport: '<Root>/Navigation Filter Data'
            //   Selector: '<S697>/Selector'

            //  target_pos = flight_plan_to_ned(flight_plan, cur_wp, home_lat_rad, home_lon_rad, home_alt);
            // MATLAB Function 'determine arm and mode selection/determine submode/waypoint submodes/determine_target_pos': '<S699>:1'
            // '<S699>:1:6' target_pos = mission_item_to_ned(fp_x, fp_y, fp_z, home_lat_rad, home_lon_rad, home_alt);
            //  ind assumes flight_plan is 0 based index but matlab is 1 based
            // 'mission_item_to_ned:3' r2d = 180 / pi;
            // 'mission_item_to_ned:5' lat = single(x) * 0.0000001;
            // 'mission_item_to_ned:6' lon = single(y) * 0.0000001;
            // 'mission_item_to_ned:7' alt = single(z + home_alt);
            // 'mission_item_to_ned:9' ned_pos = single(lla_to_ned([lat, lon, alt], ...
            // 'mission_item_to_ned:10'                             [home_lat_rad * r2d, home_lon_rad * r2d, home_alt]));
            rtb_Switch_e[0] = static_cast<real32_T>(static_cast<real_T>
                (nav.home_lat_rad * 57.295779513082323));
            rtb_Switch_e[1] = static_cast<real32_T>(static_cast<real_T>
                (nav.home_lon_rad * 57.295779513082323));
            rtb_Switch_e[2] = nav.home_alt_wgs84_m;

            //  lat, lon, alt; lat/lon in degrees, alt in m
            // 'lla_to_ned:4' c_lat = cosd(ref_lla(1));
            in_deadband_low = rtb_Switch_e[0];
            cosd(&in_deadband_low);

            // 'lla_to_ned:5' s_lat = sind(ref_lla(1));
            in_avg = rtb_Switch_e[0];
            sind(&in_avg);

            // 'lla_to_ned:6' c_lon = cosd(ref_lla(2));
            c_lon = rtb_Switch_e[1];
            cosd(&c_lon);

            // 'lla_to_ned:7' s_lon = sind(ref_lla(2));
            s_lon = rtb_Switch_e[1];
            sind(&s_lon);

            // 'lla_to_ned:8' R = [-s_lat * c_lon, -s_lon, -c_lat * c_lon; ...
            // 'lla_to_ned:9'      -s_lat * s_lon, c_lon, -c_lat * s_lon; ...
            // 'lla_to_ned:10'      c_lat, 0, -s_lat];
            // 'lla_to_ned:12' ref_E = lla_to_ECEF(ref_lla);
            // 'lla_to_ned:13' pos_E = lla_to_ECEF(pos_lla);
            // 'lla_to_ned:14' ned_pos = single(R' * (pos_E - ref_E));
            i_0[0] = static_cast<real32_T>(i) * 1.0E-7F;
            i_0[1] = static_cast<real32_T>(rtb_Selector_y) * 1.0E-7F;
            i_0[2] = in_deadband_range + nav.home_alt_wgs84_m;
            lla_to_ECEF(i_0, tmp);
            lla_to_ECEF(rtb_Switch_e, i_0);
            in_avg_0[0] = -in_avg * c_lon;
            in_avg_0[1] = -s_lon;
            in_avg_0[2] = -in_deadband_low * c_lon;
            in_avg_0[3] = -in_avg * s_lon;
            in_avg_0[4] = c_lon;
            in_avg_0[5] = -in_deadband_low * s_lon;
            in_avg_0[6] = in_deadband_low;
            in_avg_0[7] = 0.0F;
            in_avg_0[8] = -in_avg;
            in_deadband_low = tmp[0] - i_0[0];
            in_avg = tmp[1] - i_0[1];
            c_lon = tmp[2] - i_0[2];
            for (i = 0; i < 3; i++) {
                rtb_target_pos[i] = 0.0F;
                rtb_throttle_cc = rtb_target_pos[i];
                rtb_throttle_cc += in_avg_0[i] * in_deadband_low;
                rtb_target_pos[i] = rtb_throttle_cc;
                rtb_throttle_cc = rtb_target_pos[i];
                rtb_throttle_cc += in_avg_0[i + 3] * in_avg;
                rtb_target_pos[i] = rtb_throttle_cc;
                rtb_throttle_cc = rtb_target_pos[i];
                rtb_throttle_cc += in_avg_0[i + 6] * c_lon;
                rtb_target_pos[i] = rtb_throttle_cc;
            }

            // End of MATLAB Function: '<S697>/determine_target_pos'

            // Selector: '<S697>/Selector1' incorporates:
            //   Constant: '<S697>/Constant'
            //   Inport: '<Root>/Telemetry Data'
            in_deadband_range = telem.param[static_cast<int32_T>(rtCP_pooled9)];

            // MATLAB Function: '<S697>/determine_wp_submode' incorporates:
            //   Inport: '<Root>/Navigation Filter Data'
            //   Selector: '<S697>/Selector'

            //  Sub mode select RTL if mission command 20 is reached in the flight plan
            //  Sub mode select Pos_Hold if auto continued is false
            // MATLAB Function 'determine arm and mode selection/determine submode/waypoint submodes/determine_wp_submode': '<S700>:1'
            // '<S700>:1:5' sub_mode = cur_mode;
            rtDWork.sub_mode = rtb_Switch2_c;

            // '<S700>:1:7' if cmd == 20
            if (rtb_Selector_cmd == 20UL) {
                // '<S700>:1:8' sub_mode = int8(3);
                rtDWork.sub_mode = 3;

                //  RTL
            } else if (rtb_Selector_cmd == 16UL) {
                // '<S700>:1:10' elseif cmd == 16
                // '<S700>:1:11' diff = target_pos - ned_pos;
                // '<S700>:1:12' reached_wp = norm(diff) <= Aircraft_Control_wp_radius;
                in_deadband_low = 1.29246971E-26F;
                rtb_throttle_cc = rtb_target_pos[0];
                rtb_throttle_cc -= nav.ned_pos_m[0];
                c_lon = std::abs(rtb_throttle_cc);
                if (c_lon > 1.29246971E-26F) {
                    in_avg = 1.0F;
                    in_deadband_low = c_lon;
                } else {
                    s_lon = c_lon / 1.29246971E-26F;
                    in_avg = s_lon * s_lon;
                }

                rtb_throttle_cc = rtb_target_pos[1];
                rtb_throttle_cc -= nav.ned_pos_m[1];
                c_lon = std::abs(rtb_throttle_cc);
                if (c_lon > in_deadband_low) {
                    s_lon = in_deadband_low / c_lon;
                    in_avg = ((in_avg * s_lon) * s_lon) + 1.0F;
                    in_deadband_low = c_lon;
                } else {
                    s_lon = c_lon / in_deadband_low;
                    in_avg += s_lon * s_lon;
                }

                rtb_throttle_cc = rtb_target_pos[2];
                rtb_throttle_cc -= nav.ned_pos_m[2];
                c_lon = std::abs(rtb_throttle_cc);
                if (c_lon > in_deadband_low) {
                    s_lon = in_deadband_low / c_lon;
                    in_avg = ((in_avg * s_lon) * s_lon) + 1.0F;
                    in_deadband_low = c_lon;
                } else {
                    s_lon = c_lon / in_deadband_low;
                    in_avg += s_lon * s_lon;
                }

                in_avg = in_deadband_low * std::sqrt(in_avg);

                // '<S700>:1:13' if cur_mode ~= 3 && reached_wp && ~autocontinue
                if (static_cast<boolean_T>(static_cast<int32_T>
                     (((static_cast<boolean_T>(
                         static_cast<int32_T>(((rtb_Switch2_c != 3) ? (
                            static_cast<int32_T>(1)) : (static_cast<int32_T>(0)))
                          & ((in_avg <= in_deadband_range) ?
                             (static_cast<int32_T>(1)) : (static_cast<int32_T>(0))))))
                       ? (static_cast<int32_T>(1)) : (static_cast<int32_T>(0)))
                      & ((static_cast<boolean_T>(
                         static_cast<int32_T>((rtb_Compare_i ?
                           (static_cast<int32_T>(1)) : (static_cast<int32_T>(0)))
                          ^ 1))) ? (static_cast<int32_T>(1)) :
                         (static_cast<int32_T>(0)))))) {
                    // '<S700>:1:14' sub_mode = int8(1);
                    rtDWork.sub_mode = 1;

                    //  pos hold
                }
            } else {
                // no actions
            }

            // End of MATLAB Function: '<S697>/determine_wp_submode'
        }

        // End of Outputs for SubSystem: '<S679>/waypoint submodes'

        // RelationalOperator: '<S694>/Compare' incorporates:
        //   Constant: '<S694>/Constant'
        rtb_Compare_i = (rtb_Switch2_c == rtCP_pooled37);

        // Outputs for Enabled SubSystem: '<S679>/rtl submodes' incorporates:
        //   EnablePort: '<S696>/Enable'
        if (rtb_Compare_i) {
            // MATLAB Function: '<S696>/determine_fast_rtl_mode' incorporates:
            //   Constant: '<S696>/Constant'
            //   Constant: '<S696>/Constant1'
            //   Inport: '<Root>/Navigation Filter Data'
            rtDWork.sub_mode_d = rtb_Switch2_c;

            // MATLAB Function 'determine arm and mode selection/determine submode/rtl submodes/determine_fast_rtl_mode': '<S698>:1'
            // '<S698>:1:3' sub_mode = cur_mode;
            // '<S698>:1:4' diff = target_pos - ned_pos;
            // '<S698>:1:5' if norm(diff) <= Aircraft_Control_wp_radius
            in_deadband_low = 1.29246971E-26F;
            rtb_Integrator_e = rtCP_pooled25_EL_0 - nav.ned_pos_m[0];
            c_lon = std::abs(rtb_Integrator_e);
            if (c_lon > 1.29246971E-26F) {
                in_avg = 1.0F;
                in_deadband_low = c_lon;
            } else {
                s_lon = c_lon / 1.29246971E-26F;
                in_avg = s_lon * s_lon;
            }

            rtb_Integrator_e = rtCP_pooled25_EL_1 - nav.ned_pos_m[1];
            c_lon = std::abs(rtb_Integrator_e);
            if (c_lon > in_deadband_low) {
                s_lon = in_deadband_low / c_lon;
                in_avg = ((in_avg * s_lon) * s_lon) + 1.0F;
                in_deadband_low = c_lon;
            } else {
                s_lon = c_lon / in_deadband_low;
                in_avg += s_lon * s_lon;
            }

            rtb_Integrator_e = rtCP_pooled25_EL_2 - nav.ned_pos_m[2];
            c_lon = std::abs(rtb_Integrator_e);
            if (c_lon > in_deadband_low) {
                s_lon = in_deadband_low / c_lon;
                in_avg = ((in_avg * s_lon) * s_lon) + 1.0F;
                in_deadband_low = c_lon;
            } else {
                s_lon = c_lon / in_deadband_low;
                in_avg += s_lon * s_lon;
            }

            in_avg = in_deadband_low * std::sqrt(in_avg);
            if (static_cast<real_T>(in_avg) <= rtCP_Constant_Value_f) {
                // '<S698>:1:6' sub_mode = int8(4);
                rtDWork.sub_mode_d = 4;

                //  land
            }

            // End of MATLAB Function: '<S696>/determine_fast_rtl_mode'
        }

        // End of Outputs for SubSystem: '<S679>/rtl submodes'

        // Switch: '<S21>/Switch3'
        if (rtb_Compare_hs) {
        } else {
            // MultiPortSwitch: '<S679>/Multiport Switch'
            switch (rtb_Switch2_c) {
              case 2:
                rtb_Switch2_c = rtDWork.sub_mode;
                break;

              case 3:
                rtb_Switch2_c = rtDWork.sub_mode_d;
                break;

              default:
                // no actions
                break;
            }

            // End of MultiPortSwitch: '<S679>/Multiport Switch'
        }

        // End of Switch: '<S21>/Switch3'

        // RelationalOperator: '<S678>/Compare' incorporates:
        //   Constant: '<S678>/Constant'
        rtb_Compare_hs = (rtb_Switch2_c == rtCP_pooled36);

        // Logic: '<S21>/motor_armed AND mode_4'
        rtb_Compare_i = rtb_Compare & rtb_Compare_hs;

        // Outputs for Enabled SubSystem: '<S21>/auto_disarm' incorporates:
        //   EnablePort: '<S675>/Enable'
        if (rtb_Compare_i) {
            rtDWork.auto_disarm_MODE = true;

            // Gain: '<S675>/Gain' incorporates:
            //   Inport: '<Root>/Navigation Filter Data'
            in_avg = rtCP_pooled16 * nav.ned_pos_m[2];

            // RelationalOperator: '<S690>/Compare' incorporates:
            //   Constant: '<S690>/Constant'
            rtb_Compare_hs = (in_avg <= rtCP_pooled9);

            // Abs: '<S675>/Abs' incorporates:
            //   Inport: '<Root>/Navigation Filter Data'
            in_avg = std::abs(nav.ned_vel_mps[2]);

            // RelationalOperator: '<S691>/Compare' incorporates:
            //   Constant: '<S691>/Constant'
            rtb_Equal1 = (in_avg <= rtCP_pooled11);

            // Logic: '<S675>/AND'
            rtb_Compare_i = rtb_Compare_hs & rtb_Equal1;

            // Outputs for Enabled SubSystem: '<S675>/disarm motor' incorporates:
            //   EnablePort: '<S692>/Enable'
            if (rtb_Compare_i) {
                if (static_cast<boolean_T>(static_cast<int32_T>
                     ((rtDWork.disarmmotor_MODE ? (
                        static_cast<int32_T>(1)) : (static_cast<int32_T>(0))) ^
                      1))) {
                    // InitializeConditions for UnitDelay: '<S692>/Unit Delay'
                    rtDWork.UnitDelay_DSTATE = rtCP_pooled5;
                    rtDWork.disarmmotor_MODE = true;
                }

                // UnitDelay: '<S692>/Unit Delay'
                rtb_UnitDelay = rtDWork.UnitDelay_DSTATE;

                // Sum: '<S692>/Sum' incorporates:
                //   Constant: '<S692>/Constant'
                rtb_UnitDelay += rtCP_pooled8;

                // RelationalOperator: '<S693>/Compare' incorporates:
                //   Constant: '<S693>/Constant'
                rtDWork.Compare = (rtb_UnitDelay > rtCP_pooled7);

                // Update for UnitDelay: '<S692>/Unit Delay'
                rtDWork.UnitDelay_DSTATE = rtb_UnitDelay;
            } else {
                rtDWork.disarmmotor_MODE = false;
            }

            // End of Outputs for SubSystem: '<S675>/disarm motor'
        } else if (rtDWork.auto_disarm_MODE) {
            // Disable for Enabled SubSystem: '<S675>/disarm motor'
            rtDWork.disarmmotor_MODE = false;

            // End of Disable for SubSystem: '<S675>/disarm motor'
            rtDWork.auto_disarm_MODE = false;
        } else {
            // no actions
        }

        // End of Outputs for SubSystem: '<S21>/auto_disarm'

        // Switch: '<S21>/Switch' incorporates:
        //   Constant: '<S21>/Constant'
        if (rtDWork.Compare) {
            rtb_Compare = rtCP_pooled32;
        }

        // End of Switch: '<S21>/Switch'

        // RelationalOperator: '<S20>/Compare' incorporates:
        //   Constant: '<S20>/Constant'
        rtb_Compare_hs = (rtb_Switch2_c == rtCP_pooled38);

        // Logic: '<Root>/motor_armed AND mode_2'
        rtb_Compare_i = rtb_Compare & rtb_Compare_hs;

        // Outputs for Enabled SubSystem: '<Root>/WAYPOINT CONTROLLER' incorporates:
        //   EnablePort: '<S11>/Enable'
        if (rtb_Compare_i) {
            int16_T rtb_Uk1;

            // UnitDelay: '<S645>/Delay Input1'
            //
            //  Block description for '<S645>/Delay Input1':
            //
            //   Store in Global RAM
            rtb_Uk1 = rtDWork.DelayInput1_DSTATE;

            // RelationalOperator: '<S645>/FixPt Relational Operator' incorporates:
            //   Inport: '<Root>/Telemetry Data'
            rtb_Compare_hs = (telem.current_waypoint != rtb_Uk1);

            // UnitDelay: '<S11>/Unit Delay'
            rtb_Equal1 = rtDWork.reached;

            // Logic: '<S11>/OR' incorporates:
            //   Inport: '<Root>/Telemetry Data'
            rtb_enable = static_cast<boolean_T>(static_cast<int32_T>
                ((telem.waypoints_updated ? (static_cast<int32_T>(1)) : (
                static_cast<int32_T>(0))) | (rtb_Compare_hs ?
                (static_cast<int32_T>(1)) : (static_cast<int32_T>(0))))) |
                rtb_Equal1;

            // Outputs for Enabled SubSystem: '<S11>/determine target' incorporates:
            //   EnablePort: '<S474>/Enable'
            if (rtb_enable) {
                // Outputs for Enabled SubSystem: '<S474>/calc_prev_target_pos' incorporates:
                //   EnablePort: '<S647>/Enable'

                // Outputs for Enabled SubSystem: '<S679>/waypoint submodes' incorporates:
                //   EnablePort: '<S697>/Enable'

                // Selector: '<S474>/Selector' incorporates:
                //   Inport: '<Root>/Telemetry Data'
                //   Selector: '<S475>/Selector'
                //   Selector: '<S647>/Selector1'
                //   Selector: '<S697>/Selector'
                expl_temp = &telem.flight_plan[telem.current_waypoint];

                // End of Outputs for SubSystem: '<S679>/waypoint submodes'
                // End of Outputs for SubSystem: '<S474>/calc_prev_target_pos'
                i = expl_temp->x;
                rtb_Selector_y = expl_temp->y;
                in_deadband_range = expl_temp->z;

                // MATLAB Function: '<S474>/determine_current_tar_pos' incorporates:
                //   Inport: '<Root>/Navigation Filter Data'
                //   Selector: '<S474>/Selector'

                // MATLAB Function 'WAYPOINT CONTROLLER/determine target/determine_current_tar_pos': '<S648>:1'
                // '<S648>:1:4' cur_target_pos = mission_item_to_ned(fp_x, fp_y, fp_z, home_lat_rad, home_lon_rad, home_alt);
                //  ind assumes flight_plan is 0 based index but matlab is 1 based
                // 'mission_item_to_ned:3' r2d = 180 / pi;
                // 'mission_item_to_ned:5' lat = single(x) * 0.0000001;
                // 'mission_item_to_ned:6' lon = single(y) * 0.0000001;
                // 'mission_item_to_ned:7' alt = single(z + home_alt);
                // 'mission_item_to_ned:9' ned_pos = single(lla_to_ned([lat, lon, alt], ...
                // 'mission_item_to_ned:10'                             [home_lat_rad * r2d, home_lon_rad * r2d, home_alt]));
                rtb_Switch_e[0] = static_cast<real32_T>(static_cast<real_T>
                    (nav.home_lat_rad * 57.295779513082323));
                rtb_Switch_e[1] = static_cast<real32_T>(static_cast<real_T>
                    (nav.home_lon_rad * 57.295779513082323));
                rtb_Switch_e[2] = nav.home_alt_wgs84_m;

                //  lat, lon, alt; lat/lon in degrees, alt in m
                // 'lla_to_ned:4' c_lat = cosd(ref_lla(1));
                in_deadband_low = rtb_Switch_e[0];
                cosd(&in_deadband_low);

                // 'lla_to_ned:5' s_lat = sind(ref_lla(1));
                in_avg = rtb_Switch_e[0];
                sind(&in_avg);

                // 'lla_to_ned:6' c_lon = cosd(ref_lla(2));
                c_lon = rtb_Switch_e[1];
                cosd(&c_lon);

                // 'lla_to_ned:7' s_lon = sind(ref_lla(2));
                s_lon = rtb_Switch_e[1];
                sind(&s_lon);

                // 'lla_to_ned:8' R = [-s_lat * c_lon, -s_lon, -c_lat * c_lon; ...
                // 'lla_to_ned:9'      -s_lat * s_lon, c_lon, -c_lat * s_lon; ...
                // 'lla_to_ned:10'      c_lat, 0, -s_lat];
                // 'lla_to_ned:12' ref_E = lla_to_ECEF(ref_lla);
                // 'lla_to_ned:13' pos_E = lla_to_ECEF(pos_lla);
                // 'lla_to_ned:14' ned_pos = single(R' * (pos_E - ref_E));
                i_0[0] = static_cast<real32_T>(i) * 1.0E-7F;
                i_0[1] = static_cast<real32_T>(rtb_Selector_y) * 1.0E-7F;
                i_0[2] = in_deadband_range + nav.home_alt_wgs84_m;
                lla_to_ECEF(i_0, tmp);
                lla_to_ECEF(rtb_Switch_e, i_0);
                in_avg_0[0] = -in_avg * c_lon;
                in_avg_0[1] = -s_lon;
                in_avg_0[2] = -in_deadband_low * c_lon;
                in_avg_0[3] = -in_avg * s_lon;
                in_avg_0[4] = c_lon;
                in_avg_0[5] = -in_deadband_low * s_lon;
                in_avg_0[6] = in_deadband_low;
                in_avg_0[7] = 0.0F;
                in_avg_0[8] = -in_avg;
                in_deadband_low = tmp[0] - i_0[0];
                in_avg = tmp[1] - i_0[1];
                c_lon = tmp[2] - i_0[2];
                for (i = 0; i < 3; i++) {
                    rtDWork.cur_target_pos[i] = 0.0F;
                    rtDWork.cur_target_pos[i] += in_avg_0[i] * in_deadband_low;
                    rtDWork.cur_target_pos[i] += in_avg_0[i + 3] * in_avg;
                    rtDWork.cur_target_pos[i] += in_avg_0[i + 6] * c_lon;
                }

                // End of MATLAB Function: '<S474>/determine_current_tar_pos'

                // Sum: '<S474>/Sum' incorporates:
                //   Constant: '<S474>/Constant'
                //   Inport: '<Root>/Telemetry Data'
                rtb_UnitDelay = static_cast<real_T>(telem.current_waypoint) -
                    rtCP_pooled6;

                // RelationalOperator: '<S646>/Compare' incorporates:
                //   Constant: '<S646>/Constant'
                rtb_Compare_hs = (rtb_UnitDelay >= rtCP_pooled5);

                // Outputs for Enabled SubSystem: '<S474>/calc_prev_target_pos' incorporates:
                //   EnablePort: '<S647>/Enable'
                if (rtb_Compare_hs) {
                    // Outputs for Enabled SubSystem: '<S679>/waypoint submodes' incorporates:
                    //   EnablePort: '<S697>/Enable'

                    // Selector: '<S647>/Selector1' incorporates:
                    //   Inport: '<Root>/Telemetry Data'
                    //   Selector: '<S474>/Selector'
                    //   Selector: '<S475>/Selector'
                    //   Selector: '<S697>/Selector'
                    expl_temp = &telem.flight_plan[static_cast<int32_T>
                        (rtb_UnitDelay)];

                    // End of Outputs for SubSystem: '<S679>/waypoint submodes'
                    i = expl_temp->x;
                    rtb_Selector_y = expl_temp->y;
                    in_deadband_range = expl_temp->z;

                    // MATLAB Function: '<S647>/determine_prev_tar_pos' incorporates:
                    //   Inport: '<Root>/Navigation Filter Data'
                    //   Selector: '<S647>/Selector1'

                    // MATLAB Function 'WAYPOINT CONTROLLER/determine target/calc_prev_target_pos/determine_prev_tar_pos': '<S650>:1'
                    // '<S650>:1:4' prev_target_pos = mission_item_to_ned(fp_x, fp_y, fp_z, home_lat_rad, home_lon_rad, home_alt);
                    //  ind assumes flight_plan is 0 based index but matlab is 1 based
                    // 'mission_item_to_ned:3' r2d = 180 / pi;
                    // 'mission_item_to_ned:5' lat = single(x) * 0.0000001;
                    // 'mission_item_to_ned:6' lon = single(y) * 0.0000001;
                    // 'mission_item_to_ned:7' alt = single(z + home_alt);
                    // 'mission_item_to_ned:9' ned_pos = single(lla_to_ned([lat, lon, alt], ...
                    // 'mission_item_to_ned:10'                             [home_lat_rad * r2d, home_lon_rad * r2d, home_alt]));
                    rtb_Switch_e[0] = static_cast<real32_T>(static_cast<real_T>
                        (nav.home_lat_rad * 57.295779513082323));
                    rtb_Switch_e[1] = static_cast<real32_T>(static_cast<real_T>
                        (nav.home_lon_rad * 57.295779513082323));
                    rtb_Switch_e[2] = nav.home_alt_wgs84_m;

                    //  lat, lon, alt; lat/lon in degrees, alt in m
                    // 'lla_to_ned:4' c_lat = cosd(ref_lla(1));
                    in_deadband_low = rtb_Switch_e[0];
                    cosd(&in_deadband_low);

                    // 'lla_to_ned:5' s_lat = sind(ref_lla(1));
                    in_avg = rtb_Switch_e[0];
                    sind(&in_avg);

                    // 'lla_to_ned:6' c_lon = cosd(ref_lla(2));
                    c_lon = rtb_Switch_e[1];
                    cosd(&c_lon);

                    // 'lla_to_ned:7' s_lon = sind(ref_lla(2));
                    s_lon = rtb_Switch_e[1];
                    sind(&s_lon);

                    // 'lla_to_ned:8' R = [-s_lat * c_lon, -s_lon, -c_lat * c_lon; ...
                    // 'lla_to_ned:9'      -s_lat * s_lon, c_lon, -c_lat * s_lon; ...
                    // 'lla_to_ned:10'      c_lat, 0, -s_lat];
                    // 'lla_to_ned:12' ref_E = lla_to_ECEF(ref_lla);
                    // 'lla_to_ned:13' pos_E = lla_to_ECEF(pos_lla);
                    // 'lla_to_ned:14' ned_pos = single(R' * (pos_E - ref_E));
                    i_0[0] = static_cast<real32_T>(i) * 1.0E-7F;
                    i_0[1] = static_cast<real32_T>(rtb_Selector_y) * 1.0E-7F;
                    i_0[2] = in_deadband_range + nav.home_alt_wgs84_m;
                    lla_to_ECEF(i_0, tmp);
                    lla_to_ECEF(rtb_Switch_e, i_0);
                    in_avg_0[0] = -in_avg * c_lon;
                    in_avg_0[1] = -s_lon;
                    in_avg_0[2] = -in_deadband_low * c_lon;
                    in_avg_0[3] = -in_avg * s_lon;
                    in_avg_0[4] = c_lon;
                    in_avg_0[5] = -in_deadband_low * s_lon;
                    in_avg_0[6] = in_deadband_low;
                    in_avg_0[7] = 0.0F;
                    in_avg_0[8] = -in_avg;
                    in_deadband_low = tmp[0] - i_0[0];
                    in_avg = tmp[1] - i_0[1];
                    c_lon = tmp[2] - i_0[2];
                    for (i = 0; i < 3; i++) {
                        rtDWork.prev_target_pos[i] = 0.0F;
                        rtDWork.prev_target_pos[i] += in_avg_0[i] *
                            in_deadband_low;
                        rtDWork.prev_target_pos[i] += in_avg_0[i + 3] * in_avg;
                        rtDWork.prev_target_pos[i] += in_avg_0[i + 6] * c_lon;
                    }

                    // End of MATLAB Function: '<S647>/determine_prev_tar_pos'
                }

                // End of Outputs for SubSystem: '<S474>/calc_prev_target_pos'

                // Switch: '<S474>/Switch' incorporates:
                //   Inport: '<Root>/Navigation Filter Data'

                // MATLAB Function 'WAYPOINT CONTROLLER/determine target/determine_target': '<S649>:1'
                // '<S649>:1:5' diff = cur_target_pos_m - prev_target_pos_m;
                if (rtb_Compare_hs) {
                    rtb_Integrator_e = rtDWork.prev_target_pos[0];
                } else {
                    rtb_Integrator_e = nav.ned_pos_m[0];
                }

                // MATLAB Function: '<S474>/determine_target'
                rtb_Integrator_e = rtDWork.cur_target_pos[0] - rtb_Integrator_e;

                // Switch: '<S474>/Switch' incorporates:
                //   Inport: '<Root>/Navigation Filter Data'
                rtb_Switch_e[0] = rtb_Integrator_e;
                if (rtb_Compare_hs) {
                    rtb_Integrator_e = rtDWork.prev_target_pos[1];
                } else {
                    rtb_Integrator_e = nav.ned_pos_m[1];
                }

                // MATLAB Function: '<S474>/determine_target'
                rtb_Integrator_e = rtDWork.cur_target_pos[1] - rtb_Integrator_e;

                // Switch: '<S474>/Switch' incorporates:
                //   Inport: '<Root>/Navigation Filter Data'
                rtb_Switch_e[1] = rtb_Integrator_e;
                if (rtb_Compare_hs) {
                    rtb_Integrator_e = rtDWork.prev_target_pos[2];
                } else {
                    rtb_Integrator_e = nav.ned_pos_m[2];
                }

                // MATLAB Function: '<S474>/determine_target'
                rtb_Integrator_e = rtDWork.cur_target_pos[2] - rtb_Integrator_e;

                // Switch: '<S474>/Switch'
                rtb_Switch_e[2] = rtb_Integrator_e;

                // MATLAB Function: '<S474>/determine_target' incorporates:
                //   Constant: '<S474>/Constant2'
                //   Constant: '<S474>/Constant3'

                // '<S649>:1:7' xy_mag = norm(diff(1:2));
                in_deadband_low = 1.29246971E-26F;
                c_lon = std::abs(rtb_Switch_e[0]);
                if (c_lon > 1.29246971E-26F) {
                    in_avg = 1.0F;
                    in_deadband_low = c_lon;
                } else {
                    s_lon = c_lon / 1.29246971E-26F;
                    in_avg = s_lon * s_lon;
                }

                c_lon = std::abs(rtb_Switch_e[1]);
                if (c_lon > in_deadband_low) {
                    s_lon = in_deadband_low / c_lon;
                    in_avg = ((in_avg * s_lon) * s_lon) + 1.0F;
                    in_deadband_low = c_lon;
                } else {
                    s_lon = c_lon / in_deadband_low;
                    in_avg += s_lon * s_lon;
                }

                in_avg = in_deadband_low * std::sqrt(in_avg);

                // '<S649>:1:9' cur_target_heading_rad = atan2(diff(2), diff(1));
                rtDWork.cur_target_heading_rad = rt_atan2f_snf(rtb_Switch_e[1],
                    rtb_Switch_e[0]);

                //  Determine max velocity when navigating to initial WP. This is to ensure
                //  that UAV arrive at WP vertically and laterally at the same time
                // '<S649>:1:13' time_vert = abs(diff(3) / Aircraft_Control_v_z_up_max) + 3;
                // '<S649>:1:14' max_v_hor_mps = min ([xy_mag / time_vert, Aircraft_Control_v_hor_max]);
                rtb_TmpSignalConversionAtProd_1 = in_avg / (std::abs
                    (rtb_Switch_e[2] / rtCP_pooled22) + 3.0F);
                rtb_TmpSignalConversionAtProd_0 = rtCP_pooled23;
                if (rtb_TmpSignalConversionAtProd_1 >
                    rtb_TmpSignalConversionAtProd_0) {
                    rtDWork.max_v_hor_mps = rtb_TmpSignalConversionAtProd_0;
                } else if (rtIsNaNF(rtb_TmpSignalConversionAtProd_1)) {
                    if (static_cast<boolean_T>(static_cast<int32_T>((rtIsNaNF
                           (rtb_TmpSignalConversionAtProd_0) ?
                           (static_cast<int32_T>(1)) : (static_cast<int32_T>(0)))
                          ^ 1))) {
                        rtDWork.max_v_hor_mps = rtb_TmpSignalConversionAtProd_0;
                    } else {
                        rtDWork.max_v_hor_mps = rtb_TmpSignalConversionAtProd_1;
                    }
                } else {
                    rtDWork.max_v_hor_mps = rtb_TmpSignalConversionAtProd_1;
                }

                // '<S649>:1:16' time_horz = (xy_mag / Aircraft_Control_v_hor_max) + 3 ;
                // '<S649>:1:18' max_v_z_mps = min ([abs(diff(3))/time_horz, Aircraft_Control_v_z_up_max]);
                rtb_TmpSignalConversionAtProd_1 = std::abs(rtb_Switch_e[2]) /
                    ((in_avg / rtCP_pooled23) + 3.0F);
                rtb_TmpSignalConversionAtProd_0 = rtCP_pooled22;
                if (rtb_TmpSignalConversionAtProd_1 >
                    rtb_TmpSignalConversionAtProd_0) {
                    rtDWork.max_v_z_mps = rtb_TmpSignalConversionAtProd_0;
                } else if (rtIsNaNF(rtb_TmpSignalConversionAtProd_1)) {
                    if (static_cast<boolean_T>(static_cast<int32_T>((rtIsNaNF
                           (rtb_TmpSignalConversionAtProd_0) ?
                           (static_cast<int32_T>(1)) : (static_cast<int32_T>(0)))
                          ^ 1))) {
                        rtDWork.max_v_z_mps = rtb_TmpSignalConversionAtProd_0;
                    } else {
                        rtDWork.max_v_z_mps = rtb_TmpSignalConversionAtProd_1;
                    }
                } else {
                    rtDWork.max_v_z_mps = rtb_TmpSignalConversionAtProd_1;
                }
            }

            // End of Outputs for SubSystem: '<S11>/determine target'

            // Outputs for Enabled SubSystem: '<S11>/WP_NAV' incorporates:
            //   EnablePort: '<S472>/Enable'

            // Trigonometry: '<S535>/Cos' incorporates:
            //   Inport: '<Root>/Navigation Filter Data'
            in_avg = std::cos(nav.heading_rad);

            // Trigonometry: '<S535>/Sin' incorporates:
            //   Inport: '<Root>/Navigation Filter Data'
            in_deadband_range = std::sin(nav.heading_rad);

            // Gain: '<S535>/Gain'
            rtb_battery_failsafe_flag = rtCP_pooled16 * in_deadband_range;

            // Reshape: '<S535>/Reshape' incorporates:
            //   Reshape: '<S195>/Reshape'
            rtb_throttle_cc = in_avg;
            rtb_stab_pitch_rate_saturation = in_deadband_range;
            rtb_stab_roll_rate_saturation = rtb_battery_failsafe_flag;
            rtb_Sum_h = in_avg;

            // Sum: '<S480>/Subtract' incorporates:
            //   Inport: '<Root>/Navigation Filter Data'
            rtb_Switch_e[0] = rtDWork.cur_target_pos[0] - nav.ned_pos_m[0];
            rtb_Switch_e[1] = rtDWork.cur_target_pos[1] - nav.ned_pos_m[1];

            // MinMax: '<S479>/Min'
            if (static_cast<boolean_T>(static_cast<int32_T>
                 (((rtDWork.max_v_hor_mps <= rtCP_pooled23)
                   ? (static_cast<int32_T>(1)) : (static_cast<int32_T>(0))) |
                  (rtIsNaNF(rtCP_pooled23) ? (static_cast<
                    int32_T>(1)) : (static_cast<int32_T>(0)))))) {
                in_deadband_low = rtDWork.max_v_hor_mps;
            } else {
                in_deadband_low = rtCP_pooled23;
            }

            if (static_cast<boolean_T>(static_cast<int32_T>(((in_deadband_low <=
                    rtCP_pooled27) ? (static_cast<int32_T>(1)) :
                   (static_cast<int32_T>(0))) | (rtIsNaNF(rtCP_pooled27) ? (
                    static_cast<int32_T>(1)) : (static_cast<int32_T>(0)))))) {
                in_avg = in_deadband_low;
            } else {
                in_avg = rtCP_pooled27;
            }

            // End of MinMax: '<S479>/Min'

            // Math: '<S480>/Transpose'
            rtb_Integrator_e = rtb_Switch_e[0];

            // Product: '<S480>/MatrixMultiply'
            in_deadband_range = rtb_Integrator_e * rtb_Integrator_e;

            // Math: '<S480>/Transpose'
            rtb_Integrator_e = rtb_Switch_e[1];

            // Product: '<S480>/MatrixMultiply'
            in_deadband_range += rtb_Integrator_e * rtb_Integrator_e;

            // Sqrt: '<S480>/Sqrt'
            in_deadband_range = std::sqrt(in_deadband_range);

            // Saturate: '<S479>/Saturation'
            rtb_radio_failsafe_flag = rtCP_pooled18;
            rtb_battery_failsafe_flag = rtCP_pooled26;
            if (in_deadband_range > rtb_battery_failsafe_flag) {
                in_deadband_range = rtb_battery_failsafe_flag;
            } else if (in_deadband_range < rtb_radio_failsafe_flag) {
                in_deadband_range = rtb_radio_failsafe_flag;
            } else {
                // no actions
            }

            // End of Saturate: '<S479>/Saturation'

            // Product: '<S521>/PProd Out' incorporates:
            //   Constant: '<S479>/Constant3'
            in_deadband_low = in_deadband_range * rtCP_pooled27;

            // RelationalOperator: '<S524>/LowerRelop1'
            rtb_Compare_hs = (in_deadband_low > in_avg);

            // Switch: '<S524>/Switch2'
            if (rtb_Compare_hs) {
            } else {
                // RelationalOperator: '<S524>/UpperRelop' incorporates:
                //   Constant: '<S479>/Constant'
                rtb_Compare_hs = (in_deadband_low < rtCP_pooled18);

                // Switch: '<S524>/Switch' incorporates:
                //   Constant: '<S479>/Constant'
                if (rtb_Compare_hs) {
                    in_avg = rtCP_pooled18;
                } else {
                    in_avg = in_deadband_low;
                }

                // End of Switch: '<S524>/Switch'
            }

            // End of Switch: '<S524>/Switch2'

            // Trigonometry: '<S480>/Atan2'
            in_deadband_range = rt_atan2f_snf(rtb_Switch_e[1], rtb_Switch_e[0]);

            // Trigonometry: '<S482>/Cos'
            rtb_battery_failsafe_flag = std::cos(in_deadband_range);

            // Product: '<S482>/Product'
            rtb_battery_failsafe_flag *= in_avg;

            // Trigonometry: '<S482>/Sin'
            in_deadband_range = std::sin(in_deadband_range);

            // Product: '<S482>/Product1'
            in_avg *= in_deadband_range;

            // SignalConversion generated from: '<S484>/Product'
            rtb_Transpose_idx_1 = in_avg;

            // Product: '<S484>/Product'
            rtb_TmpSignalConversionAtProd_1 = rtb_throttle_cc *
                rtb_battery_failsafe_flag;
            rtb_TmpSignalConversionAtProd_1 += rtb_stab_pitch_rate_saturation *
                rtb_Transpose_idx_1;

            // Product: '<S484>/Product'
            rtDWork.vb_xy[0] = rtb_TmpSignalConversionAtProd_1;

            // Product: '<S484>/Product'
            rtb_TmpSignalConversionAtProd_1 = rtb_stab_roll_rate_saturation *
                rtb_battery_failsafe_flag;
            rtb_TmpSignalConversionAtProd_1 += rtb_Sum_h * rtb_Transpose_idx_1;

            // End of Outputs for SubSystem: '<S11>/WP_NAV'
            rtb_TmpSignalConversionAtProd_0 = rtb_TmpSignalConversionAtProd_1;

            // Outputs for Enabled SubSystem: '<S11>/WP_NAV' incorporates:
            //   EnablePort: '<S472>/Enable'

            // Product: '<S484>/Product'
            rtDWork.vb_xy[1] = rtb_TmpSignalConversionAtProd_0;

            // Sum: '<S536>/Sum3' incorporates:
            //   Inport: '<Root>/Navigation Filter Data'
            in_avg = rtDWork.cur_target_pos[2] - nav.ned_pos_m[2];

            // Product: '<S569>/IProd Out' incorporates:
            //   Constant: '<S536>/P_alt1'
            in_deadband_range = in_avg * rtCP_P_alt1_Value;

            // DiscreteIntegrator: '<S572>/Integrator'
            rtb_battery_failsafe_flag = rtDWork.Integrator_DSTATE_e;

            // Sum: '<S582>/Sum'
            in_avg += rtb_battery_failsafe_flag;

            // Product: '<S570>/PProd Out' incorporates:
            //   Constant: '<S536>/P_alt'
            c_lon = in_avg * rtCP_pooled10;

            // RelationalOperator: '<S565>/u_GTE_up' incorporates:
            //   Constant: '<S536>/Constant1'
            rtb_Compare_hs = (c_lon >= rtCP_pooled10);

            // MinMax: '<S536>/Min'
            if (static_cast<boolean_T>(static_cast<int32_T>
                 (((rtDWork.max_v_z_mps <= rtCP_pooled22) ?
                   (static_cast<int32_T>(1)) : (static_cast<int32_T>(0))) |
                  (rtIsNaNF(rtCP_pooled22) ? (static_cast<
                    int32_T>(1)) : (static_cast<int32_T>(0)))))) {
                in_avg = rtDWork.max_v_z_mps;
            } else {
                in_avg = rtCP_pooled22;
            }

            // End of MinMax: '<S536>/Min'

            // Gain: '<S536>/Gain'
            s_lon = rtCP_pooled16 * in_avg;

            // Switch: '<S565>/Switch' incorporates:
            //   Constant: '<S536>/Constant1'
            if (rtb_Compare_hs) {
                in_avg = rtCP_pooled10;
            } else {
                // RelationalOperator: '<S565>/u_GT_lo'
                rtb_Compare_hs = (c_lon > s_lon);

                // Switch: '<S565>/Switch1'
                if (rtb_Compare_hs) {
                    in_avg = c_lon;
                } else {
                    in_avg = s_lon;
                }

                // End of Switch: '<S565>/Switch1'
            }

            // End of Switch: '<S565>/Switch'

            // Sum: '<S565>/Diff'
            in_avg = c_lon - in_avg;

            // RelationalOperator: '<S562>/Relational Operator' incorporates:
            //   Constant: '<S562>/Constant2'
            rtb_Compare_hs = (rtCP_pooled18 != in_avg);

            // RelationalOperator: '<S562>/fix for DT propagation issue' incorporates:
            //   Constant: '<S562>/Constant2'
            rtb_Equal1 = (in_avg > rtCP_pooled18);

            // Switch: '<S562>/Switch3' incorporates:
            //   Constant: '<S562>/Constant6'
            //   Constant: '<S562>/Constant7'
            if (rtb_Equal1) {
                rtb_Switch1_a = rtCP_pooled33;
            } else {
                rtb_Switch1_a = rtCP_pooled34;
            }

            // End of Switch: '<S562>/Switch3'

            // RelationalOperator: '<S562>/fix for DT propagation issue1' incorporates:
            //   Constant: '<S562>/Constant2'
            rtb_Equal1 = (in_deadband_range > rtCP_pooled18);

            // Switch: '<S562>/Switch2' incorporates:
            //   Constant: '<S562>/Constant4'
            //   Constant: '<S562>/Constant5'
            if (rtb_Equal1) {
                rtb_Switch2_b = rtCP_pooled33;
            } else {
                rtb_Switch2_b = rtCP_pooled34;
            }

            // End of Switch: '<S562>/Switch2'

            // RelationalOperator: '<S562>/Equal1'
            rtb_Equal1 = (rtb_Switch1_a == rtb_Switch2_b);

            // RelationalOperator: '<S562>/Relational Operator1' incorporates:
            //   Constant: '<S536>/P_alt'
            //   Constant: '<S562>/Constant2'
            rtb_OR1_f = (rtCP_pooled10 > rtCP_pooled18);

            // Logic: '<S562>/AND1'
            rtb_OR1 = rtb_Equal1 & rtb_OR1_f;

            // Logic: '<S562>/NOT1'
            rtb_Equal1 = rtb_Equal1 ^ 1;

            // Logic: '<S562>/NOT2'
            rtb_OR1_f = rtb_OR1_f ^ 1;

            // Logic: '<S562>/AND2'
            rtb_Equal1 = rtb_Equal1 & rtb_OR1_f;

            // Logic: '<S562>/OR1'
            rtb_OR1 = rtb_OR1 | rtb_Equal1;

            // Logic: '<S562>/AND3'
            rtb_Compare_hs = rtb_Compare_hs & rtb_OR1;

            // Switch: '<S562>/Switch' incorporates:
            //   Constant: '<S562>/Constant1'
            if (rtb_Compare_hs) {
                in_deadband_low = rtCP_pooled18;
            } else {
                in_deadband_low = in_deadband_range;
            }

            // End of Switch: '<S562>/Switch'

            // RelationalOperator: '<S580>/LowerRelop1' incorporates:
            //   Constant: '<S536>/Constant1'
            rtb_Compare_hs = (c_lon > rtCP_pooled10);

            // Switch: '<S580>/Switch2'
            if (rtb_Compare_hs) {
                // Switch: '<S580>/Switch2' incorporates:
                //   Constant: '<S536>/Constant1'
                rtDWork.Switch2 = rtCP_pooled10;
            } else {
                // RelationalOperator: '<S580>/UpperRelop'
                rtb_Compare_hs = (c_lon < s_lon);

                // Switch: '<S580>/Switch'
                if (rtb_Compare_hs) {
                    in_avg = s_lon;
                } else {
                    in_avg = c_lon;
                }

                // End of Switch: '<S580>/Switch'

                // Switch: '<S580>/Switch2'
                rtDWork.Switch2 = in_avg;
            }

            // End of Switch: '<S580>/Switch2'

            // SampleTimeMath: '<S591>/TSamp' incorporates:
            //   Inport: '<Root>/Navigation Filter Data'
            //
            //  About '<S591>/TSamp':
            //   y = u * K where K = 1 / ( w * Ts )
            c_lon = nav.heading_rad * rtCP_pooled19;

            // UnitDelay: '<S591>/UD'
            //
            //  Block description for '<S591>/UD':
            //
            //   Store in Global RAM
            in_avg = rtDWork.UD_DSTATE_h;

            // Sum: '<S591>/Diff'
            //
            //  Block description for '<S591>/Diff':
            //
            //   Add in CPU
            in_avg = c_lon - in_avg;

            // Product: '<S590>/Product' incorporates:
            //   Constant: '<S590>/D_heading'
            in_avg *= rtCP_pooled17;

            // Sum: '<S593>/Subtract' incorporates:
            //   Inport: '<Root>/Navigation Filter Data'
            s_lon = rtDWork.cur_target_heading_rad - nav.heading_rad;

            // Abs: '<S593>/Abs'
            in_deadband_range = std::abs(s_lon);

            // RelationalOperator: '<S644>/Compare' incorporates:
            //   Constant: '<S644>/Constant'
            rtb_Compare_hs = (in_deadband_range > rtCP_Constant_Value_g);

            // Switch: '<S593>/Switch'
            if (rtb_Compare_hs) {
                // Signum: '<S593>/Sign'
                if (rtIsNaNF(s_lon)) {
                    in_deadband_range = s_lon;
                } else if (s_lon < 0.0F) {
                    in_deadband_range = -1.0F;
                } else {
                    in_deadband_range = static_cast<real32_T>((s_lon > 0.0F) ? (
                        static_cast<int32_T>(1)) : (static_cast<int32_T>(0)));
                }

                // End of Signum: '<S593>/Sign'

                // Product: '<S593>/Product' incorporates:
                //   Constant: '<S593>/Constant'
                in_deadband_range *= rtCP_Constant_Value_k;

                // Sum: '<S593>/Subtract1'
                in_deadband_range = s_lon - in_deadband_range;
            } else {
                in_deadband_range = s_lon;
            }

            // End of Switch: '<S593>/Switch'

            // DiscreteIntegrator: '<S627>/Integrator'
            rtb_radio_failsafe_flag = rtDWork.Integrator_DSTATE_bm;

            // Sum: '<S636>/Sum'
            rtb_Integrator_e = in_deadband_range + rtb_radio_failsafe_flag;

            // Product: '<S625>/PProd Out' incorporates:
            //   Constant: '<S590>/P_heading'
            rtb_Integrator_e *= rtCP_pooled10;

            // Sum: '<S590>/Sum'
            in_avg = rtb_Integrator_e - in_avg;

            // Saturate: '<S590>/Saturation'
            rtb_radio_failsafe_flag = rtCP_Saturation_LowerSat;
            rtb_battery_failsafe_flag = rtCP_pooled24;
            if (in_avg > rtb_battery_failsafe_flag) {
                // Saturate: '<S590>/Saturation'
                rtDWork.Saturation = rtb_battery_failsafe_flag;
            } else if (in_avg < rtb_radio_failsafe_flag) {
                // Saturate: '<S590>/Saturation'
                rtDWork.Saturation = rtb_radio_failsafe_flag;
            } else {
                // Saturate: '<S590>/Saturation'
                rtDWork.Saturation = in_avg;
            }

            // End of Saturate: '<S590>/Saturation'

            // DeadZone: '<S620>/DeadZone'
            if (rtb_Integrator_e > rtInfF) {
                rtb_Integrator_e -= rtInfF;
            } else if (rtb_Integrator_e >= rtMinusInfF) {
                rtb_Integrator_e = 0.0F;
            } else {
                rtb_Integrator_e -= rtMinusInfF;
            }

            // End of DeadZone: '<S620>/DeadZone'

            // RelationalOperator: '<S618>/fix for DT propagation issue' incorporates:
            //   Constant: '<S618>/Constant2'
            rtb_Compare_hs = (rtb_Integrator_e > rtCP_pooled18);

            // Switch: '<S618>/Switch3' incorporates:
            //   Constant: '<S618>/Constant6'
            //   Constant: '<S618>/Constant7'
            if (rtb_Compare_hs) {
                rtb_Switch1_a = rtCP_pooled33;
            } else {
                rtb_Switch1_a = rtCP_pooled34;
            }

            // End of Switch: '<S618>/Switch3'

            // Product: '<S624>/IProd Out' incorporates:
            //   Constant: '<S590>/I_heading'
            in_deadband_range *= rtCP_pooled17;

            // RelationalOperator: '<S618>/fix for DT propagation issue1' incorporates:
            //   Constant: '<S618>/Constant2'
            rtb_Compare_hs = (in_deadband_range > rtCP_pooled18);

            // Switch: '<S618>/Switch2' incorporates:
            //   Constant: '<S618>/Constant4'
            //   Constant: '<S618>/Constant5'
            if (rtb_Compare_hs) {
                rtb_Switch2_b = rtCP_pooled33;
            } else {
                rtb_Switch2_b = rtCP_pooled34;
            }

            // End of Switch: '<S618>/Switch2'

            // RelationalOperator: '<S618>/Equal1'
            rtb_Compare_hs = (rtb_Switch1_a == rtb_Switch2_b);

            // RelationalOperator: '<S618>/Relational Operator1' incorporates:
            //   Constant: '<S590>/P_heading'
            //   Constant: '<S618>/Constant2'
            rtb_Equal1 = (rtCP_pooled10 > rtCP_pooled18);

            // Logic: '<S618>/AND1'
            rtb_OR1_f = rtb_Compare_hs & rtb_Equal1;

            // Logic: '<S618>/NOT1'
            rtb_Compare_hs = rtb_Compare_hs ^ 1;

            // Logic: '<S618>/NOT2'
            rtb_Equal1 = rtb_Equal1 ^ 1;

            // Logic: '<S618>/AND2'
            rtb_Compare_hs = rtb_Compare_hs & rtb_Equal1;

            // Logic: '<S618>/OR1'
            rtb_OR1_f = rtb_OR1_f | rtb_Compare_hs;

            // RelationalOperator: '<S618>/Relational Operator' incorporates:
            //   Constant: '<S618>/Constant2'
            rtb_Compare_hs = (rtCP_pooled18 != rtb_Integrator_e);

            // Logic: '<S618>/AND3'
            rtb_Compare_hs = rtb_Compare_hs & rtb_OR1_f;

            // Switch: '<S618>/Switch' incorporates:
            //   Constant: '<S618>/Constant1'
            if (rtb_Compare_hs) {
                in_avg = rtCP_pooled18;
            } else {
                in_avg = in_deadband_range;
            }

            // End of Switch: '<S618>/Switch'

            // Update for DiscreteIntegrator: '<S572>/Integrator'
            rtDWork.Integrator_DSTATE_e += rtCP_pooled17 * in_deadband_low;

            // Update for UnitDelay: '<S591>/UD'
            //
            //  Block description for '<S591>/UD':
            //
            //   Store in Global RAM
            rtDWork.UD_DSTATE_h = c_lon;

            // Update for DiscreteIntegrator: '<S627>/Integrator'
            rtDWork.Integrator_DSTATE_bm += rtCP_pooled17 * in_avg;

            // End of Outputs for SubSystem: '<S11>/WP_NAV'

            // Outputs for Enabled SubSystem: '<S11>/determine target' incorporates:
            //   EnablePort: '<S474>/Enable'

            // Outputs for Enabled SubSystem: '<S474>/calc_prev_target_pos' incorporates:
            //   EnablePort: '<S647>/Enable'

            // Outputs for Enabled SubSystem: '<S679>/waypoint submodes' incorporates:
            //   EnablePort: '<S697>/Enable'

            // Selector: '<S475>/Selector' incorporates:
            //   Inport: '<Root>/Telemetry Data'
            //   Selector: '<S474>/Selector'
            //   Selector: '<S647>/Selector1'
            //   Selector: '<S697>/Selector'
            expl_temp = &telem.flight_plan[telem.current_waypoint];

            // End of Outputs for SubSystem: '<S679>/waypoint submodes'
            // End of Outputs for SubSystem: '<S474>/calc_prev_target_pos'
            // End of Outputs for SubSystem: '<S11>/determine target'
            rtb_Compare_hs = expl_temp->autocontinue;

            // Selector: '<S475>/Selector1' incorporates:
            //   Constant: '<S475>/Constant'
            //   Inport: '<Root>/Telemetry Data'
            in_deadband_range = telem.param[static_cast<int32_T>(rtCP_pooled7)];

            // MATLAB Function: '<S475>/check_wp_reached'
            // MATLAB Function 'WAYPOINT CONTROLLER/wp_completion_check/check_wp_reached': '<S651>:1'
            // '<S651>:1:3' diff = target_pos - ned_pos;
            // '<S651>:1:4' reached = norm(diff) <= Aircraft_Control_wp_radius && autocontinue;
            in_deadband_low = 1.29246971E-26F;

            // SignalConversion generated from: '<S11>/dbg'
            rtDWork.cur_target_pos_m[0] = rtDWork.cur_target_pos[0];

            // MATLAB Function: '<S475>/check_wp_reached' incorporates:
            //   Inport: '<Root>/Navigation Filter Data'
            rtb_Integrator_e = rtDWork.cur_target_pos[0] - nav.ned_pos_m[0];
            c_lon = std::abs(rtb_Integrator_e);
            if (c_lon > 1.29246971E-26F) {
                in_avg = 1.0F;
                in_deadband_low = c_lon;
            } else {
                s_lon = c_lon / 1.29246971E-26F;
                in_avg = s_lon * s_lon;
            }

            // SignalConversion generated from: '<S11>/dbg'
            rtDWork.cur_target_pos_m[1] = rtDWork.cur_target_pos[1];

            // MATLAB Function: '<S475>/check_wp_reached' incorporates:
            //   Inport: '<Root>/Navigation Filter Data'
            rtb_Integrator_e = rtDWork.cur_target_pos[1] - nav.ned_pos_m[1];
            c_lon = std::abs(rtb_Integrator_e);
            if (c_lon > in_deadband_low) {
                s_lon = in_deadband_low / c_lon;
                in_avg = ((in_avg * s_lon) * s_lon) + 1.0F;
                in_deadband_low = c_lon;
            } else {
                s_lon = c_lon / in_deadband_low;
                in_avg += s_lon * s_lon;
            }

            // SignalConversion generated from: '<S11>/dbg'
            rtDWork.cur_target_pos_m[2] = rtDWork.cur_target_pos[2];

            // MATLAB Function: '<S475>/check_wp_reached' incorporates:
            //   Inport: '<Root>/Navigation Filter Data'
            //   Selector: '<S475>/Selector'
            rtb_Integrator_e = rtDWork.cur_target_pos[2] - nav.ned_pos_m[2];
            c_lon = std::abs(rtb_Integrator_e);
            if (c_lon > in_deadband_low) {
                s_lon = in_deadband_low / c_lon;
                in_avg = ((in_avg * s_lon) * s_lon) + 1.0F;
                in_deadband_low = c_lon;
            } else {
                s_lon = c_lon / in_deadband_low;
                in_avg += s_lon * s_lon;
            }

            in_avg = in_deadband_low * std::sqrt(in_avg);
            rtDWork.reached = (in_avg <= in_deadband_range) & rtb_Compare_hs;

            // SignalConversion generated from: '<S11>/dbg'
            rtDWork.enable = rtb_enable;

            // SignalConversion generated from: '<S11>/dbg' incorporates:
            //   Inport: '<Root>/Telemetry Data'
            rtDWork.current_waypoint = telem.current_waypoint;

            // Update for UnitDelay: '<S645>/Delay Input1' incorporates:
            //   Inport: '<Root>/Telemetry Data'
            //
            //  Block description for '<S645>/Delay Input1':
            //
            //   Store in Global RAM
            rtDWork.DelayInput1_DSTATE = telem.current_waypoint;
        }

        // End of Outputs for SubSystem: '<Root>/WAYPOINT CONTROLLER'

        // DataTypeConversion: '<S14>/Data Type Conversion4' incorporates:
        //   Inport: '<Root>/Sensor Data'
        s_lon = static_cast<real32_T>(sensor.inceptor.ch[2]);

        // MATLAB Function: '<S659>/remap_with_deadband' incorporates:
        //   Constant: '<S659>/Constant'
        //   Constant: '<S659>/Constant1'
        //   Constant: '<S659>/Constant2'
        //   Constant: '<S659>/Constant3'
        //   Inport: '<Root>/Telemetry Data'

        // MATLAB Function 'command selection/pitch_norm_deadband/remap_with_deadband': '<S668>:1'
        // '<S668>:1:2' in_avg = (in_min + in_max) / 2;
        in_avg = (rtCP_pooled30 + rtCP_pooled31) / 2.0F;

        // '<S668>:1:3' out_avg = (out_min + out_max) / 2;
        c_lon = (rtCP_pooled16 + rtCP_pooled10) / 2.0F;

        // '<S668>:1:5' in_deadband_range = (in_max - in_min) * deadband / 2;
        in_deadband_range = ((rtCP_pooled31 - rtCP_pooled30) * telem.param[22]) /
            2.0F;

        // '<S668>:1:6' in_deadband_low = in_avg - in_deadband_range;
        in_deadband_low = in_avg - in_deadband_range;

        // '<S668>:1:7' in_deadband_hi = in_avg + in_deadband_range;
        in_avg += in_deadband_range;

        // '<S668>:1:9' if raw_in < in_deadband_low
        if (s_lon < in_deadband_low) {
            // '<S668>:1:10' norm_out = (raw_in - in_deadband_low) .* (out_max - out_avg)./(in_deadband_low - in_min) + out_avg;
            c_lon += ((s_lon - in_deadband_low) * (rtCP_pooled10 - c_lon)) /
                (in_deadband_low - rtCP_pooled30);
        } else if (static_cast<boolean_T>(static_cast<int32_T>(((s_lon >
                       in_deadband_low) ? (static_cast<int32_T>(1)) : (
                       static_cast<int32_T>(0))) & ((s_lon < in_avg) ? (
                       static_cast<int32_T>(1)) : (static_cast<int32_T>(0))))))
        {
            // '<S668>:1:11' elseif raw_in > in_deadband_low && raw_in < in_deadband_hi
            // '<S668>:1:12' norm_out = out_avg;
        } else {
            // '<S668>:1:13' else
            // '<S668>:1:14' norm_out = (raw_in - in_deadband_hi) .* (out_max - out_avg)./(in_max - in_deadband_hi) + out_avg;
            c_lon += ((s_lon - in_avg) * (rtCP_pooled10 - c_lon)) /
                (rtCP_pooled31 - in_avg);
        }

        // End of MATLAB Function: '<S659>/remap_with_deadband'

        // DataTypeConversion: '<S14>/Data Type Conversion3' incorporates:
        //   Inport: '<Root>/Sensor Data'
        rtb_DataTypeConversion3 = static_cast<real32_T>(sensor.inceptor.ch[1]);

        // MATLAB Function: '<S661>/remap_with_deadband' incorporates:
        //   Constant: '<S661>/Constant'
        //   Constant: '<S661>/Constant1'
        //   Constant: '<S661>/Constant2'
        //   Constant: '<S661>/Constant3'
        //   Inport: '<Root>/Telemetry Data'

        // MATLAB Function 'command selection/roll_norm_deadband/remap_with_deadband': '<S670>:1'
        // '<S670>:1:2' in_avg = (in_min + in_max) / 2;
        in_avg = (rtCP_pooled30 + rtCP_pooled31) / 2.0F;

        // '<S670>:1:3' out_avg = (out_min + out_max) / 2;
        s_lon = (rtCP_pooled16 + rtCP_pooled10) / 2.0F;

        // '<S670>:1:5' in_deadband_range = (in_max - in_min) * deadband / 2;
        in_deadband_range = ((rtCP_pooled31 - rtCP_pooled30) * telem.param[22]) /
            2.0F;

        // '<S670>:1:6' in_deadband_low = in_avg - in_deadband_range;
        in_deadband_low = in_avg - in_deadband_range;

        // '<S670>:1:7' in_deadband_hi = in_avg + in_deadband_range;
        in_avg += in_deadband_range;

        // '<S670>:1:9' if raw_in < in_deadband_low
        if (rtb_DataTypeConversion3 < in_deadband_low) {
            // '<S670>:1:10' norm_out = (raw_in - in_deadband_low) .* (out_max - out_avg)./(in_deadband_low - in_min) + out_avg;
            s_lon += ((rtb_DataTypeConversion3 - in_deadband_low) *
                      (rtCP_pooled10 - s_lon)) / (in_deadband_low -
                rtCP_pooled30);
        } else if (static_cast<boolean_T>(static_cast<int32_T>
                    (((rtb_DataTypeConversion3 > in_deadband_low) ? (
                       static_cast<int32_T>(1)) : (static_cast<int32_T>(0))) &
                     ((rtb_DataTypeConversion3 < in_avg) ? (static_cast<int32_T>
                       (1)) : (static_cast<int32_T>(0)))))) {
            // '<S670>:1:11' elseif raw_in > in_deadband_low && raw_in < in_deadband_hi
            // '<S670>:1:12' norm_out = out_avg;
        } else {
            // '<S670>:1:13' else
            // '<S670>:1:14' norm_out = (raw_in - in_deadband_hi) .* (out_max - out_avg)./(in_max - in_deadband_hi) + out_avg;
            s_lon += ((rtb_DataTypeConversion3 - in_avg) * (rtCP_pooled10 -
                       s_lon)) / (rtCP_pooled31 - in_avg);
        }

        // End of MATLAB Function: '<S661>/remap_with_deadband'

        // DataTypeConversion: '<S14>/Data Type Conversion5' incorporates:
        //   Inport: '<Root>/Sensor Data'
        in_avg = static_cast<real32_T>(sensor.inceptor.ch[0]);

        // MATLAB Function: '<S663>/remap' incorporates:
        //   Constant: '<S663>/Constant'
        //   Constant: '<S663>/Constant1'
        //   Constant: '<S663>/Constant2'
        //   Constant: '<S663>/Constant3'
        remap(in_avg, rtCP_pooled30, rtCP_pooled31, rtCP_pooled18, rtCP_pooled10,
              &in_deadband_low);

        // RelationalOperator: '<S19>/Compare' incorporates:
        //   Constant: '<S19>/Constant'
        rtb_Compare_hs = (rtb_Switch2_c == rtCP_pooled33);

        // Logic: '<Root>/motor_armed AND mode_5'
        rtb_enable = rtb_Compare & rtb_Compare_hs;

        // Outputs for Enabled SubSystem: '<Root>/Pos_Hold_input_conversion' incorporates:
        //   EnablePort: '<S6>/Enable'
        if (rtb_enable) {
            // Gain: '<S6>/Gain1'
            rtDWork.vb_x_cmd_mps_d = rtCP_pooled23 * c_lon;

            // Sum: '<S357>/Sum' incorporates:
            //   Constant: '<S357>/Normalize at Zero'
            in_avg = in_deadband_low - rtCP_pooled12;

            // Product: '<S357>/v_z_cmd (-1 to 1)' incorporates:
            //   Constant: '<S357>/Double'
            in_avg *= rtCP_pooled22;

            // RelationalOperator: '<S358>/Compare' incorporates:
            //   Constant: '<S358>/Constant'
            rtb_Compare_hs = (in_avg < rtCP_pooled18);

            // Product: '<S357>/Product' incorporates:
            //   Constant: '<S357>/Constant'
            in_deadband_range = (rtCP_pooled10 * static_cast<real32_T>
                                 (rtb_Compare_hs ? 1.0F : 0.0F)) * in_avg;

            // RelationalOperator: '<S359>/Compare' incorporates:
            //   Constant: '<S359>/Constant'
            rtb_Compare_hs = (in_avg >= rtCP_pooled18);

            // Product: '<S357>/Product1' incorporates:
            //   Constant: '<S357>/Constant1'
            rtb_battery_failsafe_flag = (static_cast<real32_T>(rtb_Compare_hs ?
                1.0F : 0.0F) * in_avg) * rtCP_pooled22;

            // Sum: '<S357>/Sum1'
            in_deadband_range += rtb_battery_failsafe_flag;

            // Gain: '<S357>/Gain'
            rtDWork.Gain = rtCP_pooled16 * in_deadband_range;

            // Gain: '<S6>/Gain2'
            rtDWork.vb_y_cmd_mps_f = rtCP_pooled23 * s_lon;

            // Gain: '<S6>/Gain3'
            rtDWork.yaw_rate_cmd_radps_p = rtCP_pooled24 * out_avg;
        }

        // End of Outputs for SubSystem: '<Root>/Pos_Hold_input_conversion'

        // RelationalOperator: '<S17>/Compare' incorporates:
        //   Constant: '<S17>/Constant'
        rtb_Compare_hs = (rtb_Switch2_c == rtCP_pooled37);

        // Logic: '<Root>/motor_armed AND mode_3'
        rtb_Compare_hs = rtb_Compare & rtb_Compare_hs;

        // Outputs for Enabled SubSystem: '<Root>/RTL CONTROLLER' incorporates:
        //   EnablePort: '<S8>/Enable'
        if (rtb_Compare_hs) {
            // SignalConversion generated from: '<S8>/Bus Selector2' incorporates:
            //   Inport: '<Root>/Navigation Filter Data'
            rtb_Integrator_e = nav.ned_pos_m[0];

            // Sum: '<S365>/Subtract' incorporates:
            //   Constant: '<S8>/Constant1'
            rtb_Integrator_e = rtCP_pooled25_EL_0 - rtb_Integrator_e;

            // SignalConversion generated from: '<S8>/Bus Selector2' incorporates:
            //   Inport: '<Root>/Navigation Filter Data'
            rtb_Switch_e[0] = rtb_Integrator_e;
            rtb_Integrator_e = nav.ned_pos_m[1];

            // Sum: '<S365>/Subtract' incorporates:
            //   Constant: '<S8>/Constant1'
            rtb_Integrator_e = rtCP_pooled25_EL_1 - rtb_Integrator_e;

            // SignalConversion generated from: '<S8>/Bus Selector2'
            rtb_Switch_e[1] = rtb_Integrator_e;

            // Math: '<S365>/Transpose'
            rtb_Integrator_e = rtb_Switch_e[0];

            // Product: '<S365>/MatrixMultiply'
            in_avg = rtb_Integrator_e * rtb_Integrator_e;

            // Math: '<S365>/Transpose'
            rtb_Integrator_e = rtb_Switch_e[1];

            // Product: '<S365>/MatrixMultiply'
            in_avg += rtb_Integrator_e * rtb_Integrator_e;

            // Sqrt: '<S365>/Sqrt'
            in_avg = std::sqrt(in_avg);

            // Saturate: '<S364>/Saturation'
            in_deadband_range = rtCP_pooled18;
            rtb_battery_failsafe_flag = rtCP_pooled26;
            if (in_avg > rtb_battery_failsafe_flag) {
                in_deadband_range = rtb_battery_failsafe_flag;
            } else if (in_avg < in_deadband_range) {
            } else {
                in_deadband_range = in_avg;
            }

            // End of Saturate: '<S364>/Saturation'

            // Product: '<S406>/PProd Out' incorporates:
            //   Constant: '<S364>/Constant3'
            rtb_DataTypeConversion3 = in_deadband_range * rtCP_pooled12;

            // RelationalOperator: '<S363>/Compare' incorporates:
            //   Constant: '<S363>/Constant'
            rtb_Compare_hs = (in_avg <= rtCP_pooled9);

            // Switch: '<S8>/Switch1' incorporates:
            //   Inport: '<Root>/Navigation Filter Data'
            if (rtb_Compare_hs) {
                in_avg = nav.ned_pos_m[2];
            } else {
                // MinMax: '<S8>/Min' incorporates:
                //   Inport: '<Root>/Navigation Filter Data'
                rtb_radio_failsafe_flag = nav.ned_pos_m[2];
                if (static_cast<boolean_T>(static_cast<int32_T>
                     (((rtCP_Constant_Value_j <=
                        rtb_radio_failsafe_flag) ? (static_cast<int32_T>(1)) : (
                        static_cast<int32_T>(0))) | (rtIsNaNF
                       (rtb_radio_failsafe_flag) ? (static_cast<int32_T>(1)) :
                       (static_cast<int32_T>(0)))))) {
                    rtb_radio_failsafe_flag = rtCP_Constant_Value_j;
                }

                rtb_target_pos[2] = rtb_radio_failsafe_flag;

                // End of MinMax: '<S8>/Min'
                in_avg = rtb_target_pos[2];
            }

            // End of Switch: '<S8>/Switch1'

            // Sum: '<S366>/Sum3' incorporates:
            //   Inport: '<Root>/Navigation Filter Data'
            in_avg -= nav.ned_pos_m[2];

            // Abs: '<S366>/Abs'
            in_deadband_range = std::abs(in_avg);

            // RelationalOperator: '<S420>/Compare' incorporates:
            //   Constant: '<S420>/Constant'
            rtb_Compare_hs = (in_deadband_range <= rtCP_Constant_Value_b);

            // Switch: '<S8>/Switch' incorporates:
            //   Product: '<S369>/Product'
            if (rtb_Compare_hs) {
                // Trigonometry: '<S365>/Atan2'
                in_deadband_range = rt_atan2f_snf(rtb_Switch_e[1], rtb_Switch_e
                    [0]);

                // Trigonometry: '<S368>/Sin'
                rtb_Gain_h = std::sin(in_deadband_range);

                // RelationalOperator: '<S409>/LowerRelop1' incorporates:
                //   Constant: '<S364>/Constant1'
                rtb_Compare_hs = (rtb_DataTypeConversion3 > rtCP_pooled23);

                // Switch: '<S409>/Switch2' incorporates:
                //   Constant: '<S364>/Constant1'
                if (rtb_Compare_hs) {
                    rtb_DataTypeConversion3 = rtCP_pooled23;
                } else {
                    // RelationalOperator: '<S409>/UpperRelop' incorporates:
                    //   Constant: '<S364>/Constant'
                    rtb_Compare_hs = (rtb_DataTypeConversion3 < rtCP_pooled18);

                    // Switch: '<S409>/Switch' incorporates:
                    //   Constant: '<S364>/Constant'
                    if (rtb_Compare_hs) {
                        rtb_DataTypeConversion3 = rtCP_pooled18;
                    }

                    // End of Switch: '<S409>/Switch'
                }

                // End of Switch: '<S409>/Switch2'

                // Product: '<S368>/Product1'
                rtb_Gain_h *= rtb_DataTypeConversion3;

                // Trigonometry: '<S368>/Cos'
                in_deadband_range = std::cos(in_deadband_range);

                // Product: '<S368>/Product'
                rtb_DataTypeConversion3 *= in_deadband_range;

                // SignalConversion generated from: '<S369>/Product'
                rtb_TmpSignalConversionAtProd_1 = rtb_DataTypeConversion3;
                rtb_TmpSignalConversionAtProd_0 = rtb_Gain_h;

                // Trigonometry: '<S419>/Sin' incorporates:
                //   Inport: '<Root>/Navigation Filter Data'
                rtb_DataTypeConversion3 = std::sin(nav.heading_rad);

                // Gain: '<S419>/Gain'
                rtb_Gain_h = rtCP_pooled16 * rtb_DataTypeConversion3;

                // Trigonometry: '<S419>/Cos' incorporates:
                //   Inport: '<Root>/Navigation Filter Data'
                in_deadband_range = std::cos(nav.heading_rad);

                // Reshape: '<S419>/Reshape'
                rtb_throttle_cc = in_deadband_range;
                rtb_stab_pitch_rate_saturation = rtb_DataTypeConversion3;
                rtb_stab_roll_rate_saturation = rtb_Gain_h;
                rtb_Sum_h = in_deadband_range;

                // Product: '<S369>/Product'
                rtb_battery_failsafe_flag = rtb_throttle_cc *
                    rtb_TmpSignalConversionAtProd_1;
                rtb_battery_failsafe_flag += rtb_stab_pitch_rate_saturation *
                    rtb_TmpSignalConversionAtProd_0;

                // Switch: '<S8>/Switch'
                rtDWork.Switch[0] = rtb_battery_failsafe_flag;

                // Product: '<S369>/Product'
                rtb_battery_failsafe_flag = rtb_stab_roll_rate_saturation *
                    rtb_TmpSignalConversionAtProd_1;
                rtb_battery_failsafe_flag += rtb_Sum_h *
                    rtb_TmpSignalConversionAtProd_0;
                rtb_Transpose_idx_1 = rtb_battery_failsafe_flag;

                // Switch: '<S8>/Switch' incorporates:
                //   Product: '<S369>/Product'
                rtDWork.Switch[1] = rtb_Transpose_idx_1;
            } else {
                // Switch: '<S8>/Switch' incorporates:
                //   Constant: '<S8>/Constant2'
                rtDWork.Switch[0] = rtCP_Constant2_Value_p_EL_0;
                rtDWork.Switch[1] = rtCP_Constant2_Value_p_EL_1;
            }

            // End of Switch: '<S8>/Switch'

            // Product: '<S458>/PProd Out' incorporates:
            //   Constant: '<S366>/P_vz1'
            in_avg *= rtCP_pooled10;

            // RelationalOperator: '<S461>/LowerRelop1' incorporates:
            //   Constant: '<S366>/Constant1'
            rtb_Compare_hs = (in_avg > rtCP_pooled10);

            // Switch: '<S461>/Switch2'
            if (rtb_Compare_hs) {
                // Switch: '<S461>/Switch2' incorporates:
                //   Constant: '<S366>/Constant1'
                rtDWork.Switch2_h = rtCP_pooled10;
            } else {
                // RelationalOperator: '<S461>/UpperRelop'
                rtb_Compare_hs = (in_avg < rtConstB.Gain);

                // Switch: '<S461>/Switch'
                if (rtb_Compare_hs) {
                    in_avg = rtConstB.Gain;
                }

                // End of Switch: '<S461>/Switch'

                // Switch: '<S461>/Switch2'
                rtDWork.Switch2_h = in_avg;
            }

            // End of Switch: '<S461>/Switch2'

            // SignalConversion generated from: '<S8>/Constant3' incorporates:
            //   Constant: '<S8>/Constant3'
            rtDWork.yaw_rate_cmd_radps_c = rtCP_pooled18;
        }

        // End of Outputs for SubSystem: '<Root>/RTL CONTROLLER'

        // RelationalOperator: '<S15>/Compare' incorporates:
        //   Constant: '<S15>/Constant'
        rtb_Compare_hs = (rtb_Switch2_c == rtCP_pooled36);

        // Logic: '<Root>/motor_armed AND mode_4'
        rtb_Compare_hs = rtb_Compare & rtb_Compare_hs;

        // Outputs for Enabled SubSystem: '<Root>/Pos_Hold_input_conversion2' incorporates:
        //   EnablePort: '<S7>/Enable'
        if (rtb_Compare_hs) {
            // Gain: '<S7>/Gain1'
            rtDWork.vb_x_cmd_mps = rtCP_pooled23 * c_lon;

            // Gain: '<S7>/Gain2'
            rtDWork.vb_y_cmd_mps = rtCP_pooled23 * s_lon;

            // Gain: '<S7>/Gain3'
            rtDWork.yaw_rate_cmd_radps_a = rtCP_pooled24 * out_avg;

            // Outputs for Enabled SubSystem: '<Root>/LAND CONTROLLER' incorporates:
            //   EnablePort: '<S3>/Enable'

            // SignalConversion generated from: '<S3>/land_cmd'
            rtDWork.vb_x_cmd_mps_o = rtDWork.vb_x_cmd_mps;

            // RelationalOperator: '<S191>/Compare' incorporates:
            //   Constant: '<S191>/Constant'
            //   Inport: '<Root>/Navigation Filter Data'
            rtb_Compare_hs = (nav.ned_pos_m[2] <= rtCP_pooled9);

            // Switch: '<S190>/Switch'
            if (rtb_Compare_hs) {
                // Switch: '<S190>/Switch' incorporates:
                //   Constant: '<S190>/Constant1'
                rtDWork.Switch_h = rtCP_pooled11;
            } else {
                // Switch: '<S190>/Switch' incorporates:
                //   Constant: '<S190>/Constant'
                rtDWork.Switch_h = rtCP_pooled10;
            }

            // End of Switch: '<S190>/Switch'

            // SignalConversion generated from: '<S3>/land_cmd'
            rtDWork.vb_y_cmd_mps_l = rtDWork.vb_y_cmd_mps;

            // SignalConversion generated from: '<S3>/land_cmd'
            rtDWork.yaw_rate_cmd_radps_c53 = rtDWork.yaw_rate_cmd_radps_a;

            // End of Outputs for SubSystem: '<Root>/LAND CONTROLLER'
        }

        // End of Outputs for SubSystem: '<Root>/Pos_Hold_input_conversion2'

        // Logic: '<Root>/NOT1'
        rtb_Compare_hs = rtb_enable ^ 1;

        // Switch generated from: '<Root>/Switch1'
        if (rtb_Compare_hs) {
            // MultiPortSwitch generated from: '<Root>/Multiport Switch'
            switch (rtb_Switch2_c) {
              case 2:
                rtb_DataTypeConversion3 = rtDWork.Switch2;
                break;

              case 3:
                rtb_DataTypeConversion3 = rtDWork.Switch2_h;
                break;

              default:
                rtb_DataTypeConversion3 = rtDWork.Switch_h;
                break;
            }

            // MultiPortSwitch generated from: '<Root>/Multiport Switch'
            switch (rtb_Switch2_c) {
              case 2:
                rtb_Gain_h = rtDWork.vb_xy[0];
                break;

              case 3:
                rtb_Gain_h = rtDWork.Switch[0];
                break;

              default:
                rtb_Gain_h = rtDWork.vb_x_cmd_mps_o;
                break;
            }

            // MultiPortSwitch generated from: '<Root>/Multiport Switch'
            switch (rtb_Switch2_c) {
              case 2:
                rtb_Integrator_e = rtDWork.vb_xy[1];
                break;

              case 3:
                rtb_Integrator_e = rtDWork.Switch[1];
                break;

              default:
                rtb_Integrator_e = rtDWork.vb_y_cmd_mps_l;
                break;
            }

            // MultiPortSwitch generated from: '<Root>/Multiport Switch'
            switch (rtb_Switch2_c) {
              case 2:
                rtb_yaw_rate_cmd_radps_f = rtDWork.Saturation;
                break;

              case 3:
                rtb_yaw_rate_cmd_radps_f = rtDWork.yaw_rate_cmd_radps_c;
                break;

              default:
                rtb_yaw_rate_cmd_radps_f = rtDWork.yaw_rate_cmd_radps_c53;
                break;
            }
        } else {
            rtb_DataTypeConversion3 = rtDWork.Gain;
            rtb_Gain_h = rtDWork.vb_x_cmd_mps_d;
            rtb_Integrator_e = rtDWork.vb_y_cmd_mps_f;
            rtb_yaw_rate_cmd_radps_f = rtDWork.yaw_rate_cmd_radps_p;
        }

        // End of Switch generated from: '<Root>/Switch1'

        // RelationalOperator: '<S16>/Compare' incorporates:
        //   Constant: '<S16>/Constant'
        rtb_Compare_hs = (rtb_Switch2_c > rtCP_pooled35);

        // Logic: '<Root>/motor_armed AND mode_1'
        rtb_Compare_hs = rtb_Compare & rtb_Compare_hs;

        // Outputs for Enabled SubSystem: '<Root>/POS_HOLD CONTROLLER' incorporates:
        //   EnablePort: '<S5>/Enable'
        if (rtb_Compare_hs) {
            // Trigonometry: '<S195>/Cos' incorporates:
            //   Inport: '<Root>/Navigation Filter Data'
            in_avg = std::cos(nav.heading_rad);

            // Trigonometry: '<S195>/Sin' incorporates:
            //   Inport: '<Root>/Navigation Filter Data'
            in_deadband_range = std::sin(nav.heading_rad);

            // Gain: '<S195>/Gain'
            rtb_battery_failsafe_flag = rtCP_pooled16 * in_deadband_range;

            // Reshape: '<S195>/Reshape'
            rtb_throttle_cc = in_avg;
            rtb_stab_pitch_rate_saturation = in_deadband_range;
            rtb_stab_roll_rate_saturation = rtb_battery_failsafe_flag;
            rtb_Sum_h = in_avg;

            // Product: '<S193>/Product' incorporates:
            //   Inport: '<Root>/Navigation Filter Data'
            rtb_TmpSignalConversionAtProd_1 = nav.ned_vel_mps[0];
            rtb_TmpSignalConversionAtProd_0 = nav.ned_vel_mps[1];
            rtb_Transpose_idx_1 = rtb_throttle_cc *
                rtb_TmpSignalConversionAtProd_1;
            rtb_Transpose_idx_1 += rtb_stab_pitch_rate_saturation *
                rtb_TmpSignalConversionAtProd_0;
            rtb_battery_failsafe_flag = rtb_Transpose_idx_1;
            rtb_Transpose_idx_1 = rtb_stab_roll_rate_saturation *
                rtb_TmpSignalConversionAtProd_1;
            rtb_Transpose_idx_1 += rtb_Sum_h * rtb_TmpSignalConversionAtProd_0;

            // Sum: '<S196>/Sum'
            in_avg = rtb_Gain_h - rtb_battery_failsafe_flag;

            // Product: '<S239>/PProd Out' incorporates:
            //   Constant: '<S196>/Constant'
            in_deadband_range = in_avg * rtCP_pooled12;

            // DiscreteIntegrator: '<S234>/Integrator'
            rtb_battery_failsafe_flag = rtDWork.Integrator_DSTATE_c;

            // Product: '<S226>/DProd Out' incorporates:
            //   Constant: '<S196>/Constant2'
            rtb_radio_failsafe_flag = in_avg * rtCP_pooled13;

            // SampleTimeMath: '<S229>/Tsamp'
            //
            //  About '<S229>/Tsamp':
            //   y = u * K where K = 1 / ( w * Ts )
            rtb_Gain_h = rtb_radio_failsafe_flag * rtCP_pooled19;

            // Delay: '<S227>/UD'
            rtb_radio_failsafe_flag = rtDWork.UD_DSTATE_k;

            // Sum: '<S227>/Diff'
            rtb_radio_failsafe_flag = rtb_Gain_h - rtb_radio_failsafe_flag;

            // Sum: '<S243>/Sum'
            in_deadband_range = in_deadband_range + rtb_battery_failsafe_flag +
                rtb_radio_failsafe_flag;

            // Saturate: '<S196>/Saturation'
            rtb_radio_failsafe_flag = rtCP_pooled21;
            rtb_battery_failsafe_flag = rtCP_pooled20;
            if (in_deadband_range > rtb_battery_failsafe_flag) {
                rtb_radio_failsafe_flag = rtb_battery_failsafe_flag;
            } else if (in_deadband_range < rtb_radio_failsafe_flag) {
            } else {
                rtb_radio_failsafe_flag = in_deadband_range;
            }

            // End of Saturate: '<S196>/Saturation'

            // Gain: '<S196>/Gain'
            rtDWork.Gain_a = rtCP_pooled16 * rtb_radio_failsafe_flag;

            // DeadZone: '<S225>/DeadZone'
            if (in_deadband_range > rtInfF) {
                in_deadband_range -= rtInfF;
            } else if (in_deadband_range >= rtMinusInfF) {
                in_deadband_range = 0.0F;
            } else {
                in_deadband_range -= rtMinusInfF;
            }

            // End of DeadZone: '<S225>/DeadZone'

            // RelationalOperator: '<S223>/Relational Operator' incorporates:
            //   Constant: '<S223>/Constant5'
            rtb_Compare_hs = (rtCP_pooled18 != in_deadband_range);

            // RelationalOperator: '<S223>/fix for DT propagation issue' incorporates:
            //   Constant: '<S223>/Constant5'
            rtb_Equal1 = (in_deadband_range > rtCP_pooled18);

            // Switch: '<S223>/Switch1' incorporates:
            //   Constant: '<S223>/Constant'
            //   Constant: '<S223>/Constant2'
            if (rtb_Equal1) {
                rtb_Switch1_a = rtCP_pooled33;
            } else {
                rtb_Switch1_a = rtCP_pooled34;
            }

            // End of Switch: '<S223>/Switch1'

            // Product: '<S231>/IProd Out' incorporates:
            //   Constant: '<S196>/Constant1'
            in_avg *= rtCP_pooled17;

            // RelationalOperator: '<S223>/fix for DT propagation issue1' incorporates:
            //   Constant: '<S223>/Constant5'
            rtb_Equal1 = (in_avg > rtCP_pooled18);

            // Switch: '<S223>/Switch2' incorporates:
            //   Constant: '<S223>/Constant3'
            //   Constant: '<S223>/Constant4'
            if (rtb_Equal1) {
                rtb_Switch2_b = rtCP_pooled33;
            } else {
                rtb_Switch2_b = rtCP_pooled34;
            }

            // End of Switch: '<S223>/Switch2'

            // RelationalOperator: '<S223>/Equal1'
            rtb_Equal1 = (rtb_Switch1_a == rtb_Switch2_b);

            // Logic: '<S223>/AND3'
            rtb_Compare_hs = rtb_Compare_hs & rtb_Equal1;

            // Switch: '<S223>/Switch' incorporates:
            //   Constant: '<S223>/Constant1'
            if (rtb_Compare_hs) {
                rtb_TmpSignalConversionAtProd_1 = rtCP_pooled18;
            } else {
                rtb_TmpSignalConversionAtProd_1 = in_avg;
            }

            // End of Switch: '<S223>/Switch'

            // Sum: '<S197>/Sum'
            in_avg = rtb_Integrator_e - rtb_Transpose_idx_1;

            // Product: '<S292>/PProd Out' incorporates:
            //   Constant: '<S197>/Constant'
            in_deadband_range = in_avg * rtCP_pooled12;

            // DiscreteIntegrator: '<S287>/Integrator'
            rtb_radio_failsafe_flag = rtDWork.Integrator_DSTATE_n;

            // Product: '<S279>/DProd Out' incorporates:
            //   Constant: '<S197>/Constant2'
            rtb_Integrator_e = in_avg * rtCP_pooled13;

            // SampleTimeMath: '<S282>/Tsamp'
            //
            //  About '<S282>/Tsamp':
            //   y = u * K where K = 1 / ( w * Ts )
            rtb_Transpose_idx_1 = rtb_Integrator_e * rtCP_pooled19;

            // Delay: '<S280>/UD'
            rtb_Integrator_e = rtDWork.UD_DSTATE_a;

            // Sum: '<S280>/Diff'
            rtb_Integrator_e = rtb_Transpose_idx_1 - rtb_Integrator_e;

            // Sum: '<S296>/Sum'
            in_deadband_range = in_deadband_range + rtb_radio_failsafe_flag +
                rtb_Integrator_e;

            // Saturate: '<S197>/Saturation'
            rtb_radio_failsafe_flag = rtCP_pooled21;
            if (in_deadband_range > rtb_battery_failsafe_flag) {
                // Saturate: '<S197>/Saturation'
                rtDWork.Saturation_n = rtb_battery_failsafe_flag;
            } else if (in_deadband_range < rtb_radio_failsafe_flag) {
                // Saturate: '<S197>/Saturation'
                rtDWork.Saturation_n = rtb_radio_failsafe_flag;
            } else {
                // Saturate: '<S197>/Saturation'
                rtDWork.Saturation_n = in_deadband_range;
            }

            // End of Saturate: '<S197>/Saturation'

            // DeadZone: '<S278>/DeadZone'
            if (in_deadband_range > rtInfF) {
                in_deadband_range -= rtInfF;
            } else if (in_deadband_range >= rtMinusInfF) {
                in_deadband_range = 0.0F;
            } else {
                in_deadband_range -= rtMinusInfF;
            }

            // End of DeadZone: '<S278>/DeadZone'

            // RelationalOperator: '<S276>/Relational Operator' incorporates:
            //   Constant: '<S276>/Constant5'
            rtb_Compare_hs = (rtCP_pooled18 != in_deadband_range);

            // RelationalOperator: '<S276>/fix for DT propagation issue' incorporates:
            //   Constant: '<S276>/Constant5'
            rtb_Equal1 = (in_deadband_range > rtCP_pooled18);

            // Switch: '<S276>/Switch1' incorporates:
            //   Constant: '<S276>/Constant'
            //   Constant: '<S276>/Constant2'
            if (rtb_Equal1) {
                rtb_Switch1_a = rtCP_pooled33;
            } else {
                rtb_Switch1_a = rtCP_pooled34;
            }

            // End of Switch: '<S276>/Switch1'

            // Product: '<S284>/IProd Out' incorporates:
            //   Constant: '<S197>/Constant1'
            in_avg *= rtCP_pooled17;

            // RelationalOperator: '<S276>/fix for DT propagation issue1' incorporates:
            //   Constant: '<S276>/Constant5'
            rtb_Equal1 = (in_avg > rtCP_pooled18);

            // Switch: '<S276>/Switch2' incorporates:
            //   Constant: '<S276>/Constant3'
            //   Constant: '<S276>/Constant4'
            if (rtb_Equal1) {
                rtb_Switch2_b = rtCP_pooled33;
            } else {
                rtb_Switch2_b = rtCP_pooled34;
            }

            // End of Switch: '<S276>/Switch2'

            // RelationalOperator: '<S276>/Equal1'
            rtb_Equal1 = (rtb_Switch1_a == rtb_Switch2_b);

            // Logic: '<S276>/AND3'
            rtb_Compare_hs = rtb_Compare_hs & rtb_Equal1;

            // Switch: '<S276>/Switch' incorporates:
            //   Constant: '<S276>/Constant1'
            if (rtb_Compare_hs) {
                rtb_TmpSignalConversionAtProd_0 = rtCP_pooled18;
            } else {
                rtb_TmpSignalConversionAtProd_0 = in_avg;
            }

            // End of Switch: '<S276>/Switch'

            // SignalConversion generated from: '<S5>/Command out'
            rtDWork.yaw_rate_cmd_radps_c5 = rtb_yaw_rate_cmd_radps_f;

            // Sum: '<S194>/Sum' incorporates:
            //   Inport: '<Root>/Navigation Filter Data'
            in_avg = rtb_DataTypeConversion3 - nav.ned_vel_mps[2];

            // Product: '<S345>/PProd Out' incorporates:
            //   Constant: '<S194>/P_vz'
            in_deadband_range = in_avg * rtCP_P_vz_Value;

            // DiscreteIntegrator: '<S340>/Integrator'
            rtb_Integrator_e = rtDWork.Integrator_DSTATE_cr;

            // Product: '<S332>/DProd Out' incorporates:
            //   Constant: '<S194>/D_vz'
            rtb_DataTypeConversion3 = in_avg * rtCP_pooled14;

            // SampleTimeMath: '<S335>/Tsamp'
            //
            //  About '<S335>/Tsamp':
            //   y = u * K where K = 1 / ( w * Ts )
            rtb_yaw_rate_cmd_radps_f = rtb_DataTypeConversion3 * rtCP_pooled19;

            // Delay: '<S333>/UD'
            rtb_DataTypeConversion3 = rtDWork.UD_DSTATE_c;

            // Sum: '<S333>/Diff'
            rtb_DataTypeConversion3 = rtb_yaw_rate_cmd_radps_f -
                rtb_DataTypeConversion3;

            // Sum: '<S349>/Sum'
            in_deadband_range = in_deadband_range + rtb_Integrator_e +
                rtb_DataTypeConversion3;

            // Gain: '<S194>/Gain'
            rtb_DataTypeConversion3 = rtCP_pooled16 * in_deadband_range;

            // Sum: '<S194>/Sum1' incorporates:
            //   Constant: '<S194>/Constant2'
            rtb_DataTypeConversion3 += rtCP_Constant2_Value;

            // Saturate: '<S194>/Saturation'
            rtb_radio_failsafe_flag = rtCP_pooled18;
            rtb_battery_failsafe_flag = rtCP_pooled10;
            if (rtb_DataTypeConversion3 > rtb_battery_failsafe_flag) {
                // Saturate: '<S194>/Saturation'
                rtDWork.Saturation_d = rtb_battery_failsafe_flag;
            } else if (rtb_DataTypeConversion3 < rtb_radio_failsafe_flag) {
                // Saturate: '<S194>/Saturation'
                rtDWork.Saturation_d = rtb_radio_failsafe_flag;
            } else {
                // Saturate: '<S194>/Saturation'
                rtDWork.Saturation_d = rtb_DataTypeConversion3;
            }

            // End of Saturate: '<S194>/Saturation'

            // DeadZone: '<S331>/DeadZone'
            if (in_deadband_range > rtInfF) {
                in_deadband_range -= rtInfF;
            } else if (in_deadband_range >= rtMinusInfF) {
                in_deadband_range = 0.0F;
            } else {
                in_deadband_range -= rtMinusInfF;
            }

            // End of DeadZone: '<S331>/DeadZone'

            // RelationalOperator: '<S329>/Relational Operator' incorporates:
            //   Constant: '<S329>/Constant5'
            rtb_Compare_hs = (rtCP_pooled18 != in_deadband_range);

            // RelationalOperator: '<S329>/fix for DT propagation issue' incorporates:
            //   Constant: '<S329>/Constant5'
            rtb_Equal1 = (in_deadband_range > rtCP_pooled18);

            // Switch: '<S329>/Switch1' incorporates:
            //   Constant: '<S329>/Constant'
            //   Constant: '<S329>/Constant2'
            if (rtb_Equal1) {
                rtb_Switch1_a = rtCP_pooled33;
            } else {
                rtb_Switch1_a = rtCP_pooled34;
            }

            // End of Switch: '<S329>/Switch1'

            // Product: '<S337>/IProd Out' incorporates:
            //   Constant: '<S194>/I_vz'
            in_avg *= rtCP_pooled15;

            // RelationalOperator: '<S329>/fix for DT propagation issue1' incorporates:
            //   Constant: '<S329>/Constant5'
            rtb_Equal1 = (in_avg > rtCP_pooled18);

            // Switch: '<S329>/Switch2' incorporates:
            //   Constant: '<S329>/Constant3'
            //   Constant: '<S329>/Constant4'
            if (rtb_Equal1) {
                rtb_Switch2_b = rtCP_pooled33;
            } else {
                rtb_Switch2_b = rtCP_pooled34;
            }

            // End of Switch: '<S329>/Switch2'

            // RelationalOperator: '<S329>/Equal1'
            rtb_Equal1 = (rtb_Switch1_a == rtb_Switch2_b);

            // Logic: '<S329>/AND3'
            rtb_Compare_hs = rtb_Compare_hs & rtb_Equal1;

            // Switch: '<S329>/Switch' incorporates:
            //   Constant: '<S329>/Constant1'
            if (rtb_Compare_hs) {
                in_avg = rtCP_pooled18;
            }

            // End of Switch: '<S329>/Switch'

            // Update for DiscreteIntegrator: '<S234>/Integrator'
            rtDWork.Integrator_DSTATE_c += rtCP_pooled17 *
                rtb_TmpSignalConversionAtProd_1;

            // Update for Delay: '<S227>/UD'
            rtDWork.UD_DSTATE_k = rtb_Gain_h;

            // Update for DiscreteIntegrator: '<S287>/Integrator'
            rtDWork.Integrator_DSTATE_n += rtCP_pooled17 *
                rtb_TmpSignalConversionAtProd_0;

            // Update for Delay: '<S280>/UD'
            rtDWork.UD_DSTATE_a = rtb_Transpose_idx_1;

            // Update for DiscreteIntegrator: '<S340>/Integrator'
            rtDWork.Integrator_DSTATE_cr += rtCP_pooled17 * in_avg;

            // Update for Delay: '<S333>/UD'
            rtDWork.UD_DSTATE_c = rtb_yaw_rate_cmd_radps_f;
        }

        // End of Outputs for SubSystem: '<Root>/POS_HOLD CONTROLLER'

        // RelationalOperator: '<S18>/Compare' incorporates:
        //   Constant: '<S18>/Constant'
        rtb_Compare_hs = (rtb_Switch2_c <= rtCP_pooled35);

        // Logic: '<Root>/motor_armed AND mode_0'
        rtb_Compare_hs = rtb_Compare & rtb_Compare_hs;

        // Outputs for Enabled SubSystem: '<Root>/Stab_input_conversion' incorporates:
        //   EnablePort: '<S9>/Enable'
        if (rtb_Compare_hs) {
            // SignalConversion generated from: '<S9>/angle_ctrl_input'
            rtDWork.throttle_cc = in_deadband_low;

            // Gain: '<S9>/Gain1'
            rtDWork.pitch_angle_cmd_rad = rtCP_pooled21 * c_lon;

            // Gain: '<S9>/Gain2'
            rtDWork.roll_angle_cmd_rad = rtCP_pooled20 * s_lon;

            // Gain: '<S9>/Gain3'
            rtDWork.yaw_rate_cmd_radps = rtCP_pooled24 * out_avg;
        }

        // End of Outputs for SubSystem: '<Root>/Stab_input_conversion'

        // Logic: '<Root>/NOT'
        rtb_Compare_hs = rtb_Compare_hs ^ 1;

        // Switch generated from: '<Root>/Switch'
        if (rtb_Compare_hs) {
            rtb_throttle_cc = rtDWork.Saturation_d;
            rtb_roll_angle_cmd_rad = rtDWork.Saturation_n;
        } else {
            rtb_throttle_cc = rtDWork.throttle_cc;
            rtb_roll_angle_cmd_rad = rtDWork.roll_angle_cmd_rad;
        }

        // Sum: '<S25>/stab_roll_angle_error_calc' incorporates:
        //   Inport: '<Root>/Navigation Filter Data'
        rtb_Gain_h = rtb_roll_angle_cmd_rad - nav.roll_rad;

        // Switch: '<S81>/Switch' incorporates:
        //   Constant: '<S81>/Constant'
        rtb_TmpSignalConversionAtProd_1 = rtCP_pooled28;

        // Product: '<S123>/PProd Out'
        rtb_TmpSignalConversionAtProd_1 *= rtb_Gain_h;

        // DiscreteIntegrator: '<S118>/Integrator'
        in_avg = rtDWork.Integrator_DSTATE;

        // Sum: '<S127>/Sum'
        rtb_TmpSignalConversionAtProd_1 += in_avg;

        // Switch: '<S83>/Switch' incorporates:
        //   Constant: '<S83>/Constant'
        rtb_pitch_angle_cmd_rad = rtCP_pooled29;

        // Product: '<S25>/Product' incorporates:
        //   Inport: '<Root>/Navigation Filter Data'
        rtb_pitch_angle_cmd_rad *= nav.gyro_radps[0];

        // Sum: '<S25>/Sum'
        rtb_pitch_angle_cmd_rad = rtb_TmpSignalConversionAtProd_1 -
            rtb_pitch_angle_cmd_rad;

        // Saturate: '<S25>/stab_roll_rate_saturation'
        rtb_stab_roll_rate_saturation = rtCP_pooled16;
        rtb_battery_failsafe_flag = rtCP_pooled10;
        if (rtb_pitch_angle_cmd_rad > rtb_battery_failsafe_flag) {
            rtb_stab_roll_rate_saturation = rtb_battery_failsafe_flag;
        } else if (rtb_pitch_angle_cmd_rad < rtb_stab_roll_rate_saturation) {
        } else {
            rtb_stab_roll_rate_saturation = rtb_pitch_angle_cmd_rad;
        }

        // End of Saturate: '<S25>/stab_roll_rate_saturation'

        // Switch generated from: '<Root>/Switch'
        if (rtb_Compare_hs) {
            rtb_pitch_angle_cmd_rad = rtDWork.Gain_a;
        } else {
            rtb_pitch_angle_cmd_rad = rtDWork.pitch_angle_cmd_rad;
        }

        // Sum: '<S24>/stab_pitch_angle_error_calc' incorporates:
        //   Inport: '<Root>/Navigation Filter Data'
        rtb_yaw_rate_cmd_radps_f = rtb_pitch_angle_cmd_rad - nav.pitch_rad;

        // Switch: '<S27>/Switch' incorporates:
        //   Constant: '<S27>/Constant'
        rtb_Transpose_idx_1 = rtCP_pooled28;

        // Product: '<S69>/PProd Out'
        rtb_Transpose_idx_1 *= rtb_yaw_rate_cmd_radps_f;

        // DiscreteIntegrator: '<S64>/Integrator'
        in_avg = rtDWork.Integrator_DSTATE_l;

        // Sum: '<S73>/Sum'
        rtb_Transpose_idx_1 += in_avg;

        // Switch: '<S29>/Switch' incorporates:
        //   Constant: '<S29>/Constant'
        rtb_yaw_rate_cmd_radps_a = rtCP_pooled29;

        // Product: '<S24>/Product' incorporates:
        //   Inport: '<Root>/Navigation Filter Data'
        rtb_yaw_rate_cmd_radps_a *= nav.gyro_radps[1];

        // Sum: '<S24>/Sum'
        rtb_yaw_rate_cmd_radps_a = rtb_Transpose_idx_1 -
            rtb_yaw_rate_cmd_radps_a;

        // Saturate: '<S24>/stab_pitch_rate_saturation'
        rtb_stab_pitch_rate_saturation = rtCP_pooled16;
        if (rtb_yaw_rate_cmd_radps_a > rtb_battery_failsafe_flag) {
            rtb_stab_pitch_rate_saturation = rtb_battery_failsafe_flag;
        } else if (rtb_yaw_rate_cmd_radps_a < rtb_stab_pitch_rate_saturation) {
        } else {
            rtb_stab_pitch_rate_saturation = rtb_yaw_rate_cmd_radps_a;
        }

        // End of Saturate: '<S24>/stab_pitch_rate_saturation'

        // Switch generated from: '<Root>/Switch'
        if (rtb_Compare_hs) {
            rtb_yaw_rate_cmd_radps_a = rtDWork.yaw_rate_cmd_radps_c5;
        } else {
            rtb_yaw_rate_cmd_radps_a = rtDWork.yaw_rate_cmd_radps;
        }

        // Sum: '<S26>/stab_yaw_rate_error_calc' incorporates:
        //   Inport: '<Root>/Navigation Filter Data'
        rtb_TmpSignalConversionAtProd_0 = rtb_yaw_rate_cmd_radps_a -
            nav.gyro_radps[2];

        // Switch: '<S136>/Switch' incorporates:
        //   Constant: '<S136>/Constant'
        rtb_DataTypeConversion3 = rtCP_pooled12;

        // Product: '<S178>/PProd Out'
        rtb_DataTypeConversion3 *= rtb_TmpSignalConversionAtProd_0;

        // DiscreteIntegrator: '<S173>/Integrator'
        rtb_Integrator_e = rtDWork.Integrator_DSTATE_b;

        // Sum: '<S182>/Sum'
        rtb_DataTypeConversion3 += rtb_Integrator_e;

        // SampleTimeMath: '<S135>/TSamp' incorporates:
        //   Inport: '<Root>/Navigation Filter Data'
        //
        //  About '<S135>/TSamp':
        //   y = u * K where K = 1 / ( w * Ts )
        rtb_Integrator_e = nav.gyro_radps[2] * rtCP_pooled19;

        // UnitDelay: '<S135>/UD'
        //
        //  Block description for '<S135>/UD':
        //
        //   Store in Global RAM
        rtb_radio_failsafe_flag = rtDWork.UD_DSTATE;

        // Sum: '<S135>/Diff'
        //
        //  Block description for '<S135>/Diff':
        //
        //   Add in CPU
        rtb_radio_failsafe_flag = rtb_Integrator_e - rtb_radio_failsafe_flag;

        // Switch: '<S138>/Switch' incorporates:
        //   Constant: '<S138>/Constant'
        rtb_battery_failsafe_flag = rtCP_pooled29;

        // Product: '<S26>/Product'
        rtb_radio_failsafe_flag *= rtb_battery_failsafe_flag;

        // Sum: '<S26>/Sum'
        rtb_Sum_h = rtb_DataTypeConversion3 - rtb_radio_failsafe_flag;

        // DataTypeConversion: '<Root>/Data Type Conversion'
        rtb_battery_failsafe_flag = rtb_Compare_d ? 1.0F : 0.0F;

        // DataTypeConversion: '<Root>/Data Type Conversion1'
        rtb_radio_failsafe_flag = zcEvent ? 1.0F : 0.0F;

        // DataTypeConversion: '<Root>/Data Type Conversion5'
        in_deadband_range = static_cast<real32_T>(rtDWork.current_waypoint);

        // DataTypeConversion: '<Root>/Data Type Conversion6'
        in_avg = rtDWork.enable ? 1.0F : 0.0F;

        // SignalConversion generated from: '<S10>/Bus Creator' incorporates:
        //   Inport: '<Root>/Navigation Filter Data'
        //   Inport: '<Root>/Sensor Data'
        rtb_aux[0] = rtb_throttle_cc;
        rtb_aux[1] = rtb_stab_roll_rate_saturation;
        rtb_aux[2] = rtb_roll_angle_cmd_rad;
        rtb_aux[3] = nav.roll_rad;
        rtb_aux[4] = rtb_stab_pitch_rate_saturation;
        rtb_aux[5] = rtb_pitch_angle_cmd_rad;
        rtb_aux[6] = nav.pitch_rad;
        rtb_aux[7] = rtb_Sum_h;
        rtb_aux[8] = rtb_yaw_rate_cmd_radps_a;
        rtb_aux[9] = nav.gyro_radps[2];
        rtb_aux[10] = rtb_battery_failsafe_flag;
        rtb_aux[11] = rtb_radio_failsafe_flag;
        rtb_aux[12] = rtDWork.cur_target_pos_m[0];
        rtb_aux[13] = rtDWork.cur_target_pos_m[1];
        rtb_aux[14] = rtDWork.cur_target_pos_m[2];
        rtb_aux[15] = in_deadband_low;
        rtb_aux[16] = c_lon;
        rtb_aux[17] = s_lon;
        rtb_aux[18] = out_avg;
        rtb_aux[19] = rtb_DataTypeConversion2;
        rtb_aux[20] = sensor.power_module.voltage_v;
        rtb_aux[21] = sensor.power_module.current_v;
        rtb_aux[22] = in_deadband_range;
        rtb_aux[23] = in_avg;

        // DiscreteIntegrator: '<S23>/motor_arm_ramp_integrator'
        rtb_UnitDelay = rtDWork.motor_arm_ramp_integrator_DSTAT;

        // DataTypeConversion: '<S14>/Data Type Conversion8' incorporates:
        //   Inport: '<Root>/Sensor Data'
        in_avg = static_cast<real32_T>(sensor.inceptor.ch[5]);

        // MATLAB Function: '<S660>/remap' incorporates:
        //   Constant: '<S660>/Constant'
        //   Constant: '<S660>/Constant1'
        //   Constant: '<S660>/Constant2'
        //   Constant: '<S660>/Constant3'
        remap(in_avg, rtCP_pooled30, rtCP_pooled31, rtCP_pooled18, rtCP_pooled10,
              &rtb_DataTypeConversion2);

        // DataTypeConversion: '<S14>/Data Type Conversion6' incorporates:
        //   Inport: '<Root>/Sensor Data'
        in_avg = static_cast<real32_T>(sensor.inceptor.ch[7]);

        // MATLAB Function: '<S657>/remap' incorporates:
        //   Constant: '<S657>/Constant'
        //   Constant: '<S657>/Constant1'
        //   Constant: '<S657>/Constant2'
        //   Constant: '<S657>/Constant3'
        remap(in_avg, rtCP_pooled30, rtCP_pooled31, rtCP_pooled18, rtCP_pooled10,
              &c_lon);

        // Switch: '<S22>/emergency_switch' incorporates:
        //   Constant: '<S22>/Constant'
        if (rtb_Compare_e) {
            for (i = 0; i < 8; i++) {
                rtb_Command[i] = rtConstP.Constant_Value_i[i];
            }
        } else {
            real_T rtb_engine;
            real_T rtb_relay;

            // Product: '<S4>/Multiply' incorporates:
            //   Math: '<S4>/Transpose'
            tmp_0 = &rtConstB.Transpose[0];
            for (i = 0; i < 8; i++) {
                in_avg = rtb_throttle_cc * tmp_0[i * 4];
                in_avg += rtb_stab_roll_rate_saturation * tmp_0[(i * 4) + 1];
                in_avg += rtb_stab_pitch_rate_saturation * tmp_0[(i * 4) + 2];
                in_avg += rtb_Sum_h * tmp_0[(i * 4) + 3];

                // RelationalOperator: '<S192>/UpperRelop' incorporates:
                //   Constant: '<S4>/Constant1'
                rtb_Compare_d = (static_cast<real_T>(in_avg) <
                                 rtCP_Constant1_Value);

                // Switch: '<S192>/Switch' incorporates:
                //   Constant: '<S4>/Constant1'
                if (rtb_Compare_d) {
                    in_deadband_range = static_cast<real32_T>
                        (rtCP_Constant1_Value);
                } else {
                    in_deadband_range = in_avg;
                }

                // End of Switch: '<S192>/Switch'

                // RelationalOperator: '<S192>/LowerRelop1' incorporates:
                //   Constant: '<S4>/Constant'
                rtb_Compare_d = (static_cast<real_T>(in_avg) > rtCP_pooled6);

                // Switch: '<S192>/Switch2' incorporates:
                //   Constant: '<S4>/Constant'
                if (rtb_Compare_d) {
                    in_deadband_range = static_cast<real32_T>(rtCP_pooled6);
                }

                // End of Switch: '<S192>/Switch2'

                // Product: '<Root>/Product'
                rtb_Product[i] = static_cast<real_T>(in_deadband_range) *
                    rtb_UnitDelay;
                rtb_Command[i] = in_avg;
            }

            // End of Product: '<S4>/Multiply'

            // Sum: '<S12>/Sum'
            rtb_relay = rtb_Product[6] + static_cast<real_T>
                (rtb_DataTypeConversion2);

            // Sum: '<S12>/Sum1'
            rtb_engine = rtb_Product[7] + static_cast<real_T>(c_lon);
            for (i = 0; i < 6; i++) {
                rtb_Command[i] = static_cast<real32_T>(rtb_Product[i]);
            }

            rtb_Command[6] = static_cast<real32_T>(rtb_relay);
            rtb_Command[7] = static_cast<real32_T>(rtb_engine);
        }

        // End of Switch: '<S22>/emergency_switch'

        // RelationalOperator: '<S707>/Compare' incorporates:
        //   Constant: '<S707>/Constant'
        rtb_Compare_hs = (rtb_UnitDelay <= rtCP_pooled5);

        // Logic: '<S13>/NOT'
        rtb_Compare_hs = rtb_Compare_hs ^ 1;

        // MATLAB Function: '<S653>/remap' incorporates:
        //   Constant: '<S653>/Constant'
        //   Constant: '<S653>/Constant1'
        //   Constant: '<S653>/Constant2'
        //   Constant: '<S653>/Constant3'

        // MATLAB Function 'cmd to raw pwm/motor_PWM_denormalize/remap': '<S655>:1'
        // '<S655>:1:2' raw_out = (norm_in - in_min) * (out_max - out_min)/(in_max-in_min) + out_min;
        rtb_DataTypeConversion2 = rtCP_Constant2_Value_a -
            rtCP_Constant1_Value_g;
        in_avg = rtCP_pooled10 - rtCP_pooled18;

        // MATLAB Function 'cmd to raw pwm/engine_PWM_denormalize/remap': '<S654>:1'
        // '<S654>:1:2' raw_out = (norm_in - in_min) * (out_max - out_min)/(in_max-in_min) + out_min;
        for (i = 0; i < 6; i++) {
            c_lon = (((rtb_Command[i] - rtCP_pooled18) * rtb_DataTypeConversion2)
                     / in_avg) + rtCP_Constant1_Value_g;

            // Switch: '<S13>/Switch' incorporates:
            //   Constant: '<S13>/Constant'
            if (rtb_Compare_hs) {
            } else {
                c_lon = static_cast<real32_T>(rtConstP.Constant_Value_fw[i]);
            }

            // End of Switch: '<S13>/Switch'

            // DataTypeConversion: '<S13>/Data Type Conversion'
            rtb_cnt_g[i] = static_cast<int16_T>(std::floor(c_lon));
        }

        // End of MATLAB Function: '<S653>/remap'

        // MATLAB Function: '<S652>/remap' incorporates:
        //   Constant: '<S652>/Constant'
        //   Constant: '<S652>/Constant1'
        //   Constant: '<S652>/Constant2'
        //   Constant: '<S652>/Constant3'
        rtb_DataTypeConversion2 = (((rtb_Command[7] - rtCP_pooled18) *
            (rtCP_Constant2_Value_n - rtCP_Constant1_Value_l)) / (rtCP_pooled10
            - rtCP_pooled18)) + rtCP_Constant1_Value_l;

        // DataTypeConversion: '<S13>/Data Type Conversion'
        rtb_cnt_g[6] = static_cast<int16_T>(std::floor(rtb_Command[6]));
        rtb_cnt_g[7] = static_cast<int16_T>(std::floor(rtb_DataTypeConversion2));

        // SignalConversion generated from: '<S10>/Bus Creator'
        for (i = 0; i < 8; i++) {
            rtb_Switch2[i] = 0.0F;
        }

        // End of SignalConversion generated from: '<S10>/Bus Creator'

        // Logic: '<Root>/AND'
        rtb_Compare_hs = rtb_Compare_i & rtDWork.reached;

        // Gain: '<Root>/Gain'
        in_avg = rtCP_pooled19 * rtb_throttle_cc;

        // BusCreator: '<S471>/Bus Creator' incorporates:
        //   Constant: '<S471>/Constant'
        //   Constant: '<S471>/Constant1'
        rtb_Compare_d = rtCP_pooled32;
        zcEvent = rtCP_pooled32;

        // Gain: '<S2>/Gain' incorporates:
        //   Inport: '<Root>/Sensor Data'
        in_deadband_range = rtCP_Gain_Gain * sensor.power_module.voltage_v;

        // Gain: '<S2>/Gain1' incorporates:
        //   Inport: '<Root>/Sensor Data'
        rtb_DataTypeConversion2 = rtCP_Gain1_Gain *
            sensor.power_module.current_v;

        // DiscreteIntegrator: '<S2>/Discrete-Time Integrator'
        c_lon = rtDWork.DiscreteTimeIntegrator_DSTATE + (rtCP_pooled14 *
            rtb_DataTypeConversion2);

        // BusCreator: '<S2>/Bus Creator3' incorporates:
        //   Constant: '<S2>/remaining_prcnt'
        //   Constant: '<S2>/remaining_time_s'
        s_lon = rtCP_pooled10;
        rtb_throttle_cc = rtCP_pooled10;

        // Outport: '<Root>/VMS Data' incorporates:
        //   BusCreator: '<S10>/Bus Creator'
        ctrl->motors_enabled = rtb_Compare;
        ctrl->waypoint_reached = rtb_Compare_hs;
        ctrl->mode = rtb_Switch2_c;
        ctrl->throttle_cmd_prcnt = in_avg;
        (void)std::memcpy(&ctrl->aux[0], &rtb_aux[0], 24U * sizeof(real32_T));
        ctrl->sbus.ch17 = rtb_Compare_d;
        ctrl->sbus.ch18 = zcEvent;
        (void)std::memset(&ctrl->sbus.cmd[0], 0, sizeof(real32_T) << 4UL);
        for (i = 0; i < 16; i++) {
            ctrl->sbus.cnt[i] = 0;
        }

        for (i = 0; i < 8; i++) {
            ctrl->pwm.cnt[i] = rtb_cnt_g[i];
            ctrl->pwm.cmd[i] = rtb_Command[i];
            ctrl->analog.val[i] = rtb_Switch2[i];
        }

        ctrl->battery.voltage_v = in_deadband_range;
        ctrl->battery.current_ma = rtb_DataTypeConversion2;
        ctrl->battery.consumed_mah = c_lon;
        ctrl->battery.remaining_prcnt = s_lon;
        ctrl->battery.remaining_time_s = rtb_throttle_cc;

        // End of Outport: '<Root>/VMS Data'

        // DeadZone: '<S166>/DeadZone'
        if (rtb_DataTypeConversion3 > rtInfF) {
            rtb_DataTypeConversion3 -= rtInfF;
        } else if (rtb_DataTypeConversion3 >= rtMinusInfF) {
            rtb_DataTypeConversion3 = 0.0F;
        } else {
            rtb_DataTypeConversion3 -= rtMinusInfF;
        }

        // End of DeadZone: '<S166>/DeadZone'

        // RelationalOperator: '<S164>/Relational Operator' incorporates:
        //   Constant: '<S164>/Constant5'
        rtb_Compare_hs = (rtCP_pooled18 != rtb_DataTypeConversion3);

        // RelationalOperator: '<S164>/fix for DT propagation issue' incorporates:
        //   Constant: '<S164>/Constant5'
        rtb_Equal1 = (rtb_DataTypeConversion3 > rtCP_pooled18);

        // Switch: '<S164>/Switch1' incorporates:
        //   Constant: '<S164>/Constant'
        //   Constant: '<S164>/Constant2'
        if (rtb_Equal1) {
            rtb_Switch2_b = rtCP_pooled33;
        } else {
            rtb_Switch2_b = rtCP_pooled34;
        }

        // End of Switch: '<S164>/Switch1'

        // Switch: '<S137>/Switch' incorporates:
        //   Constant: '<S137>/Constant'
        in_avg = rtCP_pooled15;

        // Product: '<S170>/IProd Out'
        rtb_TmpSignalConversionAtProd_0 *= in_avg;

        // RelationalOperator: '<S164>/fix for DT propagation issue1' incorporates:
        //   Constant: '<S164>/Constant5'
        rtb_Equal1 = (rtb_TmpSignalConversionAtProd_0 > rtCP_pooled18);

        // Switch: '<S164>/Switch2' incorporates:
        //   Constant: '<S164>/Constant3'
        //   Constant: '<S164>/Constant4'
        if (rtb_Equal1) {
            rtb_Switch1_a = rtCP_pooled33;
        } else {
            rtb_Switch1_a = rtCP_pooled34;
        }

        // End of Switch: '<S164>/Switch2'

        // RelationalOperator: '<S164>/Equal1'
        rtb_Equal1 = (rtb_Switch2_b == rtb_Switch1_a);

        // Logic: '<S164>/AND3'
        rtb_Compare_hs = rtb_Compare_hs & rtb_Equal1;

        // Switch: '<S164>/Switch' incorporates:
        //   Constant: '<S164>/Constant1'
        if (rtb_Compare_hs) {
            s_lon = rtCP_pooled18;
        } else {
            s_lon = rtb_TmpSignalConversionAtProd_0;
        }

        // End of Switch: '<S164>/Switch'

        // DeadZone: '<S57>/DeadZone'
        if (rtb_Transpose_idx_1 > rtInfF) {
            rtb_Transpose_idx_1 -= rtInfF;
        } else if (rtb_Transpose_idx_1 >= rtMinusInfF) {
            rtb_Transpose_idx_1 = 0.0F;
        } else {
            rtb_Transpose_idx_1 -= rtMinusInfF;
        }

        // End of DeadZone: '<S57>/DeadZone'

        // RelationalOperator: '<S55>/Relational Operator' incorporates:
        //   Constant: '<S55>/Constant5'
        rtb_Compare_hs = (rtCP_pooled18 != rtb_Transpose_idx_1);

        // RelationalOperator: '<S55>/fix for DT propagation issue' incorporates:
        //   Constant: '<S55>/Constant5'
        rtb_Equal1 = (rtb_Transpose_idx_1 > rtCP_pooled18);

        // Switch: '<S55>/Switch1' incorporates:
        //   Constant: '<S55>/Constant'
        //   Constant: '<S55>/Constant2'
        if (rtb_Equal1) {
            rtb_Switch1_a = rtCP_pooled33;
        } else {
            rtb_Switch1_a = rtCP_pooled34;
        }

        // End of Switch: '<S55>/Switch1'

        // Switch: '<S28>/Switch' incorporates:
        //   Constant: '<S28>/Constant'
        in_avg = rtCP_pooled28;

        // Product: '<S61>/IProd Out'
        rtb_yaw_rate_cmd_radps_f *= in_avg;

        // RelationalOperator: '<S55>/fix for DT propagation issue1' incorporates:
        //   Constant: '<S55>/Constant5'
        rtb_Equal1 = (rtb_yaw_rate_cmd_radps_f > rtCP_pooled18);

        // Switch: '<S55>/Switch2' incorporates:
        //   Constant: '<S55>/Constant3'
        //   Constant: '<S55>/Constant4'
        if (rtb_Equal1) {
            rtb_Switch2_b = rtCP_pooled33;
        } else {
            rtb_Switch2_b = rtCP_pooled34;
        }

        // End of Switch: '<S55>/Switch2'

        // RelationalOperator: '<S55>/Equal1'
        rtb_Equal1 = (rtb_Switch1_a == rtb_Switch2_b);

        // Logic: '<S55>/AND3'
        rtb_Compare_hs = rtb_Compare_hs & rtb_Equal1;

        // Switch: '<S55>/Switch' incorporates:
        //   Constant: '<S55>/Constant1'
        if (rtb_Compare_hs) {
            in_deadband_range = rtCP_pooled18;
        } else {
            in_deadband_range = rtb_yaw_rate_cmd_radps_f;
        }

        // End of Switch: '<S55>/Switch'

        // DeadZone: '<S111>/DeadZone'
        if (rtb_TmpSignalConversionAtProd_1 > rtInfF) {
            rtb_TmpSignalConversionAtProd_1 -= rtInfF;
        } else if (rtb_TmpSignalConversionAtProd_1 >= rtMinusInfF) {
            rtb_TmpSignalConversionAtProd_1 = 0.0F;
        } else {
            rtb_TmpSignalConversionAtProd_1 -= rtMinusInfF;
        }

        // End of DeadZone: '<S111>/DeadZone'

        // RelationalOperator: '<S109>/Relational Operator' incorporates:
        //   Constant: '<S109>/Constant5'
        rtb_Compare_hs = (rtCP_pooled18 != rtb_TmpSignalConversionAtProd_1);

        // RelationalOperator: '<S109>/fix for DT propagation issue' incorporates:
        //   Constant: '<S109>/Constant5'
        rtb_Equal1 = (rtb_TmpSignalConversionAtProd_1 > rtCP_pooled18);

        // Switch: '<S109>/Switch1' incorporates:
        //   Constant: '<S109>/Constant'
        //   Constant: '<S109>/Constant2'
        if (rtb_Equal1) {
            rtb_Switch1_a = rtCP_pooled33;
        } else {
            rtb_Switch1_a = rtCP_pooled34;
        }

        // End of Switch: '<S109>/Switch1'

        // Product: '<S115>/IProd Out'
        rtb_Gain_h *= in_avg;

        // RelationalOperator: '<S109>/fix for DT propagation issue1' incorporates:
        //   Constant: '<S109>/Constant5'
        rtb_Equal1 = (rtb_Gain_h > rtCP_pooled18);

        // Switch: '<S109>/Switch2' incorporates:
        //   Constant: '<S109>/Constant3'
        //   Constant: '<S109>/Constant4'
        if (rtb_Equal1) {
            rtb_Switch2_b = rtCP_pooled33;
        } else {
            rtb_Switch2_b = rtCP_pooled34;
        }

        // End of Switch: '<S109>/Switch2'

        // RelationalOperator: '<S109>/Equal1'
        rtb_Equal1 = (rtb_Switch1_a == rtb_Switch2_b);

        // Logic: '<S109>/AND3'
        rtb_Compare_hs = rtb_Compare_hs & rtb_Equal1;

        // Switch: '<S109>/Switch' incorporates:
        //   Constant: '<S109>/Constant1'
        if (rtb_Compare_hs) {
            rtb_DataTypeConversion3 = rtCP_pooled18;
        } else {
            rtb_DataTypeConversion3 = rtb_Gain_h;
        }

        // End of Switch: '<S109>/Switch'

        // RelationalOperator: '<S703>/Compare' incorporates:
        //   Constant: '<S703>/Constant'
        rtb_Compare_hs = (in_deadband_low <= rtCP_pooled13);

        // Abs: '<S680>/Abs'
        in_avg = std::abs(out_avg);

        // RelationalOperator: '<S701>/Compare' incorporates:
        //   Constant: '<S701>/Constant'
        rtb_Compare_d = (in_avg >= rtCP_Constant_Value_m);

        // Logic: '<S680>/AND'
        zcEvent = rtb_Compare_hs & rtb_Compare_d;

        // Outputs for Enabled SubSystem: '<S680>/Enabled Subsystem' incorporates:
        //   EnablePort: '<S705>/Enable'
        if (zcEvent) {
            if (static_cast<boolean_T>(static_cast<int32_T>
                 ((rtDWork.EnabledSubsystem_MODE ? (
                    static_cast<int32_T>(1)) : (static_cast<int32_T>(0))) ^ 1)))
            {
                // InitializeConditions for DiscreteIntegrator: '<S705>/Discrete-Time Integrator'
                rtDWork.DiscreteTimeIntegrator_DSTATE_k = rtCP_pooled18;
                rtDWork.EnabledSubsystem_MODE = true;
            }

            // DataTypeConversion: '<S705>/Cast'
            rtb_Switch2_b = static_cast<int8_T>(rtb_Compare_d ? 1 : 0);

            // DiscreteIntegrator: '<S705>/Discrete-Time Integrator'
            rtDWork.DiscreteTimeIntegrator =
                rtDWork.DiscreteTimeIntegrator_DSTATE_k + ((static_cast<real32_T>
                (rtCP_DiscreteTimeIntegrator_gai) * 0.000122070312F) *
                static_cast<real32_T>(rtb_Switch2_b));

            // Update for DiscreteIntegrator: '<S705>/Discrete-Time Integrator'
            rtDWork.DiscreteTimeIntegrator_DSTATE_k =
                rtDWork.DiscreteTimeIntegrator + ((static_cast<real32_T>
                (rtCP_DiscreteTimeIntegrator_gai) * 0.000122070312F) *
                static_cast<real32_T>(rtb_Switch2_b));
        } else {
            rtDWork.EnabledSubsystem_MODE = false;
        }

        // End of Outputs for SubSystem: '<S680>/Enabled Subsystem'

        // Switch: '<S23>/motor_arm_ramp'
        if (rtb_Compare) {
            rtb_UnitDelay = static_cast<real_T>
                (rtConstB.ramp_time_intergratorsignal);
        } else {
            // Logic: '<S23>/NOT'
            rtb_Compare_e = rtb_Compare_e ^ 1;

            // Switch: '<S23>/fast_dis_arm_time' incorporates:
            //   Constant: '<S23>/Constant'
            if (rtb_Compare_e) {
                rtb_UnitDelay = static_cast<real_T>(rtConstB.Gain1);
            } else {
                rtb_UnitDelay = rtCP_Constant_Value_e;
            }

            // End of Switch: '<S23>/fast_dis_arm_time'
        }

        // End of Switch: '<S23>/motor_arm_ramp'

        // DataTypeConversion: '<S14>/Data Type Conversion9' incorporates:
        //   Inport: '<Root>/Sensor Data'
        out_avg = static_cast<real32_T>(sensor.inceptor.ch[8]);

        // MATLAB Function: '<S662>/remap' incorporates:
        //   Constant: '<S662>/Constant'
        //   Constant: '<S662>/Constant1'
        //   Constant: '<S662>/Constant2'
        //   Constant: '<S662>/Constant3'
        remap(out_avg, rtCP_pooled30, rtCP_pooled31, rtCP_pooled18,
              rtCP_pooled10, &in_deadband_low);

        // Update for DiscreteIntegrator: '<S118>/Integrator'
        rtDWork.Integrator_DSTATE += rtCP_pooled17 * rtb_DataTypeConversion3;

        // Update for DiscreteIntegrator: '<S64>/Integrator'
        rtDWork.Integrator_DSTATE_l += rtCP_pooled17 * in_deadband_range;

        // Update for DiscreteIntegrator: '<S173>/Integrator'
        rtDWork.Integrator_DSTATE_b += rtCP_pooled17 * s_lon;

        // Update for UnitDelay: '<S135>/UD'
        //
        //  Block description for '<S135>/UD':
        //
        //   Store in Global RAM
        rtDWork.UD_DSTATE = rtb_Integrator_e;

        // Update for DiscreteIntegrator: '<S23>/motor_arm_ramp_integrator'
        rtDWork.motor_arm_ramp_integrator_DSTAT += rtCP_pooled8 * rtb_UnitDelay;
        if (rtDWork.motor_arm_ramp_integrator_DSTAT >= rtCP_pooled6) {
            rtDWork.motor_arm_ramp_integrator_DSTAT = rtCP_pooled6;
        } else if (rtDWork.motor_arm_ramp_integrator_DSTAT <= rtCP_pooled5) {
            rtDWork.motor_arm_ramp_integrator_DSTAT = rtCP_pooled5;
        } else {
            // no actions
        }

        // End of Update for DiscreteIntegrator: '<S23>/motor_arm_ramp_integrator'

        // Update for DiscreteIntegrator: '<S2>/Discrete-Time Integrator'
        rtDWork.DiscreteTimeIntegrator_DSTATE = c_lon + (rtCP_pooled14 *
            rtb_DataTypeConversion2);
    }

    // Model initialize function
    void Autocode::initialize()
    {
        // Registration code

        // initialize non-finites
        rt_InitInfAndNaN(sizeof(real_T));
        rtPrevZCSigState.manual_arming_Trig_ZCE = UNINITIALIZED_ZCSIG;

        // SystemInitialize for Enabled SubSystem: '<S681>/disarm motor'
        // InitializeConditions for UnitDelay: '<S685>/Unit Delay'
        rtDWork.UnitDelay_DSTATE_m = rtCP_pooled5;

        // End of SystemInitialize for SubSystem: '<S681>/disarm motor'

        // SystemInitialize for Enabled SubSystem: '<S21>/auto_disarm'
        // SystemInitialize for Enabled SubSystem: '<S675>/disarm motor'
        // InitializeConditions for UnitDelay: '<S692>/Unit Delay'
        rtDWork.UnitDelay_DSTATE = rtCP_pooled5;

        // End of SystemInitialize for SubSystem: '<S675>/disarm motor'
        // End of SystemInitialize for SubSystem: '<S21>/auto_disarm'

        // SystemInitialize for Enabled SubSystem: '<Root>/WAYPOINT CONTROLLER'
        // InitializeConditions for UnitDelay: '<S645>/Delay Input1'
        //
        //  Block description for '<S645>/Delay Input1':
        //
        //   Store in Global RAM
        rtDWork.DelayInput1_DSTATE = rtCP_DelayInput1_InitialConditi;

        // End of SystemInitialize for SubSystem: '<Root>/WAYPOINT CONTROLLER'

        // SystemInitialize for Enabled SubSystem: '<S680>/Enabled Subsystem'
        // InitializeConditions for DiscreteIntegrator: '<S705>/Discrete-Time Integrator'
        rtDWork.DiscreteTimeIntegrator_DSTATE_k = rtCP_pooled18;

        // End of SystemInitialize for SubSystem: '<S680>/Enabled Subsystem'
    }

    // Constructor
    Autocode::Autocode():
        rtDWork(),
        rtPrevZCSigState()
    {
        // Currently there is no constructor body generated.
    }

    // Destructor
    Autocode::~Autocode()
    {
        // Currently there is no destructor body generated.
    }
}

//
// File trailer for generated code.
//
// [EOF]
//
