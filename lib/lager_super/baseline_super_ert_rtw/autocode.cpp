//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// File: autocode.cpp
//
// Code generated for Simulink model 'baseline_super'.
//
// Model version                  : 2.492
// Simulink Coder version         : 9.5 (R2021a) 14-Nov-2020
// C/C++ source code generated on : Tue Apr 26 10:56:42 2022
//
// Target selection: ert.tlc
// Embedded hardware selection: AMD->x86-64 (Linux 64)
// Code generation objective: Execution efficiency
// Validation result: Not run
//

#include <array>
#include "autocode.h"
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
const bfs::Autocode::ConstB rtConstB = {
  { {
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
  }                                    // '<S6>/Transpose'
};

extern real32_T rt_atan2f_snf(real32_T u0, real32_T u1);
extern real_T rt_modd_snf(real_T u0, real_T u1);
extern real_T rt_atan2d_snf(real_T u0, real_T u1);

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
                           ( (tmpVal.bitVal.words.wordH & 0x000FFFFF) != 0 ||
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

real32_T rt_atan2f_snf(real32_T u0, real32_T u1)
{
  int32_T u0_0;
  int32_T u1_0;
  real32_T y;
  if (rtIsNaNF(u0) || rtIsNaNF(u1)) {
    y = (rtNaNF);
  } else if (rtIsInfF(u0) && rtIsInfF(u1)) {
    if (u0 > 0.0F) {
      u0_0 = 1;
    } else {
      u0_0 = -1;
    }

    if (u1 > 0.0F) {
      u1_0 = 1;
    } else {
      u1_0 = -1;
    }

    y = std::atan2(static_cast<real32_T>(u0_0), static_cast<real32_T>(u1_0));
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

real_T rt_modd_snf(real_T u0, real_T u1)
{
  real_T q;
  real_T y;
  boolean_T yEq;
  y = u0;
  if (u1 == 0.0) {
    if (u0 == 0.0) {
      y = u1;
    }
  } else if (rtIsNaN(u0) || rtIsNaN(u1) || rtIsInf(u0)) {
    y = (rtNaN);
  } else if (u0 == 0.0) {
    y = 0.0 / u1;
  } else if (rtIsInf(u1)) {
    if ((u1 < 0.0) != (u0 < 0.0)) {
      y = u1;
    }
  } else {
    y = std::fmod(u0, u1);
    yEq = (y == 0.0);
    if ((!yEq) && (u1 > std::floor(u1))) {
      q = std::abs(u0 / u1);
      yEq = !(std::abs(q - std::floor(q + 0.5)) > DBL_EPSILON * q);
    }

    if (yEq) {
      y = u1 * 0.0;
    } else if ((u0 < 0.0) != (u1 < 0.0)) {
      y += u1;
    }
  }

  return y;
}

real_T rt_atan2d_snf(real_T u0, real_T u1)
{
  real_T y;
  int32_T u0_0;
  int32_T u1_0;
  if (rtIsNaN(u0) || rtIsNaN(u1)) {
    y = (rtNaN);
  } else if (rtIsInf(u0) && rtIsInf(u1)) {
    if (u0 > 0.0) {
      u0_0 = 1;
    } else {
      u0_0 = -1;
    }

    if (u1 > 0.0) {
      u1_0 = 1;
    } else {
      u1_0 = -1;
    }

    y = std::atan2(static_cast<real_T>(u0_0), static_cast<real_T>(u1_0));
  } else if (u1 == 0.0) {
    if (u0 > 0.0) {
      y = RT_PI / 2.0;
    } else if (u0 < 0.0) {
      y = -(RT_PI / 2.0);
    } else {
      y = 0.0;
    }
  } else {
    y = std::atan2(u0, u1);
  }

  return y;
}

namespace bfs
{
  // Model step function
  void Autocode::Run(SysData sys, SensorData sensor, NavData nav, TelemData
                     telem, VmsData *ctrl)
  {
    std::array<real32_T, 8> rtb_val;
    real_T rtb_Abs1_n;
    real_T rtb_Gain_dj_idx_0;
    real_T rtb_Gain_dj_idx_1;
    real_T rtb_Subtract_idx_1;
    real_T rtb_Sum_cd;
    real_T rtb_Switch_ejf;
    real_T rtb_Switch_f;
    real_T rtb_Switch_fq;
    real_T rtb_UnitConversion_idx_0;
    real_T rtb_UnitConversion_idx_1;
    int32_T i;
    int32_T rtb_Reshape_l_tmp;
    real32_T rtb_Cos_k;
    real32_T rtb_Integrator_d;
    real32_T rtb_PProdOut_d;
    real32_T rtb_PProdOut_h;
    real32_T rtb_Product1_k;
    real32_T rtb_Reshape_l_0;
    real32_T rtb_Subtract_p_idx_0;
    real32_T rtb_Switch_dv;
    real32_T rtb_Tsamp_c;
    real32_T rtb_Tsamp_m;
    real32_T rtb_ZeroGain_p;
    real32_T rtb_pitch_angle_cmd_rad;
    real32_T rtb_roll;
    real32_T rtb_stab_pitch_rate_saturation;
    real32_T rtb_throttle;
    real32_T rtb_yaw;
    int8_T rtb_DataStoreRead_m;
    boolean_T rtb_Compare_ey;
    boolean_T rtb_Compare_id;
    boolean_T rtb_DataStoreRead1_c;
    boolean_T rtb_DataStoreRead1_nl;
    boolean_T rtb_motor_armedANDmode_0;
    boolean_T rtb_motor_armedANDmode_2;
    UNUSED_PARAMETER(sys);

    // DataStoreRead: '<Root>/Data Store Read1'
    rtb_DataStoreRead1_c = rtDW.motor_state;

    // Outputs for Enabled SubSystem: '<S13>/WP_NAV' incorporates:
    //   EnablePort: '<S488>/Enable'

    // Sum: '<S730>/Sum' incorporates:
    //   DataStoreRead: '<S491>/Data Store Read'
    //   Inport: '<Root>/Navigation Filter Data'
    //   Sum: '<S496>/Subtract'
    rtb_Product1_k = rtDW.cur_target_pos_m[0] - nav.ned_pos_m[0];

    // End of Outputs for SubSystem: '<S13>/WP_NAV'

    // DotProduct: '<S730>/Dot Product'
    rtb_PProdOut_h = rtb_Product1_k * rtb_Product1_k;

    // Sum: '<S730>/Sum' incorporates:
    //   DataStoreRead: '<S491>/Data Store Read'
    //   Inport: '<Root>/Navigation Filter Data'
    //   Sum: '<S496>/Subtract'
    //   Sum: '<S550>/Sum3'
    rtb_Subtract_p_idx_0 = rtb_Product1_k;

    // Outputs for Enabled SubSystem: '<S13>/WP_NAV' incorporates:
    //   EnablePort: '<S488>/Enable'
    rtb_Product1_k = rtDW.cur_target_pos_m[1] - nav.ned_pos_m[1];
    rtb_throttle = rtDW.cur_target_pos_m[2] - nav.ned_pos_m[2];

    // DotProduct: '<S730>/Dot Product' incorporates:
    //   Product: '<S496>/MatrixMultiply'
    rtb_Integrator_d = rtb_Product1_k * rtb_Product1_k;

    // End of Outputs for SubSystem: '<S13>/WP_NAV'

    // RelationalOperator: '<S731>/Compare' incorporates:
    //   Constant: '<S731>/Constant'
    //   DotProduct: '<S730>/Dot Product'
    //   Sqrt: '<S730>/sqrt'
    //   Sum: '<S730>/Sum'
    rtb_Compare_id = (std::sqrt(rtb_throttle * rtb_throttle + (rtb_Integrator_d
      + rtb_PProdOut_h)) <= 1.5F);

    // DataStoreRead: '<S491>/Data Store Read1'
    rtb_DataStoreRead1_nl = rtDW.autocontinue;

    // DataStoreRead: '<Root>/Data Store Read'
    rtb_DataStoreRead_m = rtDW.cur_mode;

    // Logic: '<Root>/motor_armed AND mode_2' incorporates:
    //   Constant: '<S19>/Constant'
    //   DataStoreRead: '<Root>/Data Store Read'
    //   DataStoreRead: '<Root>/Data Store Read1'
    //   RelationalOperator: '<S19>/Compare'
    rtb_motor_armedANDmode_2 = (rtDW.motor_state && (rtDW.cur_mode == 2));

    // Outputs for Enabled SubSystem: '<S13>/WP_NAV' incorporates:
    //   EnablePort: '<S488>/Enable'
    if (rtb_motor_armedANDmode_2) {
      // MinMax: '<S495>/Min'
      if (rtDW.max_v_hor_mps < 5.0F) {
        rtb_PProdOut_d = rtDW.max_v_hor_mps;
      } else {
        rtb_PProdOut_d = 5.0F;
      }

      if (!(rtb_PProdOut_d < 3.0F)) {
        rtb_PProdOut_d = 3.0F;
      }

      // End of MinMax: '<S495>/Min'

      // Sqrt: '<S496>/Sqrt' incorporates:
      //   Product: '<S496>/MatrixMultiply'
      //   Sum: '<S496>/Subtract'
      rtb_PProdOut_h = std::sqrt(rtb_Subtract_p_idx_0 * rtb_Subtract_p_idx_0 +
        rtb_Integrator_d);

      // Saturate: '<S495>/Saturation'
      if (rtb_PProdOut_h > 20.0F) {
        rtb_PProdOut_h = 20.0F;
      } else if (rtb_PProdOut_h < 0.0F) {
        rtb_PProdOut_h = 0.0F;
      }

      // End of Saturate: '<S495>/Saturation'

      // Product: '<S536>/PProd Out' incorporates:
      //   Constant: '<S495>/Constant3'
      rtb_PProdOut_h *= 3.0F;

      // Switch: '<S539>/Switch2' incorporates:
      //   RelationalOperator: '<S539>/LowerRelop1'
      //   Switch: '<S539>/Switch'
      if (!(rtb_PProdOut_h > rtb_PProdOut_d)) {
        rtb_PProdOut_d = rtb_PProdOut_h;
      }

      // End of Switch: '<S539>/Switch2'

      // Trigonometry: '<S496>/Atan2' incorporates:
      //   Sum: '<S496>/Subtract'
      rtb_Integrator_d = rt_atan2f_snf(rtb_Product1_k, rtb_Subtract_p_idx_0);

      // Product: '<S498>/Product' incorporates:
      //   Trigonometry: '<S498>/Cos'
      rtb_PProdOut_h = rtb_PProdOut_d * std::cos(rtb_Integrator_d);

      // Product: '<S498>/Product1' incorporates:
      //   Trigonometry: '<S498>/Sin'
      rtb_Product1_k = rtb_PProdOut_d * std::sin(rtb_Integrator_d);

      // Trigonometry: '<S549>/Cos' incorporates:
      //   Inport: '<Root>/Navigation Filter Data'
      rtb_PProdOut_d = std::cos(nav.heading_rad);

      // Trigonometry: '<S549>/Sin' incorporates:
      //   Inport: '<Root>/Navigation Filter Data'
      rtb_Integrator_d = std::sin(nav.heading_rad);

      // Product: '<S499>/Product' incorporates:
      //   Gain: '<S549>/Gain'
      //   Reshape: '<S549>/Reshape'
      //   SignalConversion generated from: '<S499>/Product'
      rtDW.vb_xy[0] = 0.0F;
      rtDW.vb_xy[0] += rtb_PProdOut_d * rtb_PProdOut_h;
      rtDW.vb_xy[0] += rtb_Integrator_d * rtb_Product1_k;
      rtDW.vb_xy[1] = 0.0F;
      rtDW.vb_xy[1] += -rtb_Integrator_d * rtb_PProdOut_h;
      rtDW.vb_xy[1] += rtb_PProdOut_d * rtb_Product1_k;

      // Switch: '<S591>/Switch2' incorporates:
      //   Constant: '<S550>/Constant1'
      //   MinMax: '<S550>/Min'
      //   RelationalOperator: '<S591>/LowerRelop1'
      if (rtb_throttle > 1.0F) {
        // Switch: '<S591>/Switch2'
        rtDW.Switch2 = 1.0F;
      } else {
        if (rtDW.max_v_z_mps < 2.0F) {
          // MinMax: '<S550>/Min'
          rtb_PProdOut_d = rtDW.max_v_z_mps;
        } else {
          // MinMax: '<S550>/Min'
          rtb_PProdOut_d = 2.0F;
        }

        // Switch: '<S591>/Switch' incorporates:
        //   Gain: '<S550>/Gain'
        //   RelationalOperator: '<S591>/UpperRelop'
        if (rtb_throttle < -rtb_PProdOut_d) {
          // Switch: '<S591>/Switch2'
          rtDW.Switch2 = -rtb_PProdOut_d;
        } else {
          // Switch: '<S591>/Switch2'
          rtDW.Switch2 = rtb_throttle;
        }

        // End of Switch: '<S591>/Switch'
      }

      // End of Switch: '<S591>/Switch2'

      // Sum: '<S603>/Subtract' incorporates:
      //   DataStoreRead: '<S494>/Data Store Read1'
      //   Inport: '<Root>/Navigation Filter Data'
      rtb_PProdOut_d = rtDW.cur_target_heading_rad - nav.heading_rad;

      // Switch: '<S603>/Switch' incorporates:
      //   Abs: '<S603>/Abs'
      //   Constant: '<S603>/Constant'
      //   Constant: '<S654>/Constant'
      //   Product: '<S603>/Product'
      //   RelationalOperator: '<S654>/Compare'
      //   Sum: '<S603>/Subtract1'
      if (std::abs(rtb_PProdOut_d) > 3.14159274F) {
        // Signum: '<S603>/Sign'
        if (rtb_PProdOut_d < 0.0F) {
          rtb_Switch_dv = -1.0F;
        } else if (rtb_PProdOut_d > 0.0F) {
          rtb_Switch_dv = 1.0F;
        } else if (rtb_PProdOut_d == 0.0F) {
          rtb_Switch_dv = 0.0F;
        } else {
          rtb_Switch_dv = (rtNaNF);
        }

        // End of Signum: '<S603>/Sign'
        rtb_PProdOut_d -= rtb_Switch_dv * 6.28318548F;
      }

      // End of Switch: '<S603>/Switch'

      // Sum: '<S646>/Sum' incorporates:
      //   DiscreteIntegrator: '<S637>/Integrator'
      //   Product: '<S642>/PProd Out'
      rtb_Integrator_d = rtb_PProdOut_d + rtDW.Integrator_DSTATE_p;

      // Product: '<S634>/IProd Out' incorporates:
      //   Constant: '<S601>/I_heading'
      rtb_PProdOut_d *= 0.01F;

      // Saturate: '<S601>/Saturation'
      if (rtb_Integrator_d > 0.524F) {
        // Saturate: '<S601>/Saturation'
        rtDW.Saturation = 0.524F;
      } else if (rtb_Integrator_d < -0.524F) {
        // Saturate: '<S601>/Saturation'
        rtDW.Saturation = -0.524F;
      } else {
        // Saturate: '<S601>/Saturation'
        rtDW.Saturation = rtb_Integrator_d;
      }

      // End of Saturate: '<S601>/Saturation'

      // DeadZone: '<S630>/DeadZone'
      if (rtb_Integrator_d >= (rtMinusInfF)) {
        rtb_ZeroGain_p = 0.0F;
      } else {
        rtb_ZeroGain_p = (rtNaNF);
      }

      // End of DeadZone: '<S630>/DeadZone'

      // Signum: '<S628>/SignPreIntegrator'
      if (rtb_PProdOut_d < 0.0F) {
        rtb_Switch_dv = -1.0F;
      } else if (rtb_PProdOut_d > 0.0F) {
        rtb_Switch_dv = 1.0F;
      } else if (rtb_PProdOut_d == 0.0F) {
        rtb_Switch_dv = 0.0F;
      } else {
        rtb_Switch_dv = (rtNaNF);
      }

      // End of Signum: '<S628>/SignPreIntegrator'

      // Switch: '<S628>/Switch' incorporates:
      //   Constant: '<S628>/Constant1'
      //   DataTypeConversion: '<S628>/DataTypeConv2'
      //   Gain: '<S628>/ZeroGain'
      //   Logic: '<S628>/AND3'
      //   RelationalOperator: '<S628>/Equal1'
      //   RelationalOperator: '<S628>/NotEqual'
      if ((0.0F * rtb_Integrator_d != rtb_ZeroGain_p) && (0 ==
           static_cast<int8_T>(rtb_Switch_dv))) {
        rtb_PProdOut_d = 0.0F;
      }

      // End of Switch: '<S628>/Switch'

      // Update for DiscreteIntegrator: '<S637>/Integrator'
      rtDW.Integrator_DSTATE_p += 0.01F * rtb_PProdOut_d;
    }

    // End of Outputs for SubSystem: '<S13>/WP_NAV'

    // Outputs for Enabled SubSystem: '<Root>/RTL CONTROLLER' incorporates:
    //   EnablePort: '<S10>/Enable'

    // Logic: '<Root>/motor_armed AND mode_3' incorporates:
    //   Constant: '<S16>/Constant'
    //   DataStoreRead: '<Root>/Data Store Read'
    //   DataStoreRead: '<Root>/Data Store Read1'
    //   RelationalOperator: '<S16>/Compare'
    if (rtDW.motor_state && (rtDW.cur_mode == 3)) {
      // Sqrt: '<S379>/Sqrt' incorporates:
      //   Inport: '<Root>/Navigation Filter Data'
      //   Product: '<S379>/MatrixMultiply'
      //   SignalConversion generated from: '<S10>/Bus Selector2'
      //   Sum: '<S379>/Subtract'
      rtb_PProdOut_d = std::sqrt((0.0F - nav.ned_pos_m[0]) * (0.0F -
        nav.ned_pos_m[0]) + (0.0F - nav.ned_pos_m[1]) * (0.0F - nav.ned_pos_m[1]));

      // RelationalOperator: '<S375>/Compare' incorporates:
      //   Constant: '<S375>/Constant'
      rtb_Compare_ey = (rtb_PProdOut_d <= 1.5F);

      // Outputs for Triggered SubSystem: '<S10>/Trigger Land' incorporates:
      //   TriggerPort: '<S378>/Trigger'
      if (rtb_Compare_ey && (rtPrevZCX.TriggerLand_Trig_ZCE != 1)) {
        // DataStoreWrite: '<S378>/Data Store Write' incorporates:
        //   Constant: '<S378>/Land_mode'
        rtDW.cur_mode = 4;
      }

      rtPrevZCX.TriggerLand_Trig_ZCE = rtb_Compare_ey;

      // End of Outputs for SubSystem: '<S10>/Trigger Land'

      // Saturate: '<S377>/Saturation'
      if (rtb_PProdOut_d > 20.0F) {
        rtb_Switch_dv = 20.0F;
      } else if (rtb_PProdOut_d < 0.0F) {
        rtb_Switch_dv = 0.0F;
      } else {
        rtb_Switch_dv = rtb_PProdOut_d;
      }

      // End of Saturate: '<S377>/Saturation'

      // Product: '<S420>/PProd Out' incorporates:
      //   Constant: '<S377>/Constant3'
      rtb_PProdOut_h = rtb_Switch_dv * 0.5F;

      // Switch: '<S10>/Switch1' incorporates:
      //   Constant: '<S376>/Constant'
      //   Inport: '<Root>/Navigation Filter Data'
      //   MinMax: '<S10>/Min'
      //   RelationalOperator: '<S376>/Compare'
      if (rtb_PProdOut_d <= 10.0F) {
        rtb_PProdOut_d = nav.ned_pos_m[2];
      } else if ((-100.0F < nav.ned_pos_m[2]) || rtIsNaNF(nav.ned_pos_m[2])) {
        // MinMax: '<S10>/Min'
        rtb_PProdOut_d = -100.0F;
      } else {
        // MinMax: '<S10>/Min' incorporates:
        //   Inport: '<Root>/Navigation Filter Data'
        rtb_PProdOut_d = nav.ned_pos_m[2];
      }

      // End of Switch: '<S10>/Switch1'

      // Sum: '<S380>/Sum3' incorporates:
      //   Inport: '<Root>/Navigation Filter Data'
      rtb_PProdOut_d -= nav.ned_pos_m[2];

      // Switch: '<S10>/Switch' incorporates:
      //   Abs: '<S380>/Abs'
      //   Constant: '<S434>/Constant'
      //   RelationalOperator: '<S434>/Compare'
      if (std::abs(rtb_PProdOut_d) <= 1.5F) {
        // Trigonometry: '<S379>/Atan2' incorporates:
        //   Inport: '<Root>/Navigation Filter Data'
        //   SignalConversion generated from: '<S10>/Bus Selector2'
        //   Sum: '<S379>/Subtract'
        rtb_Product1_k = rt_atan2f_snf(0.0F - nav.ned_pos_m[1], 0.0F -
          nav.ned_pos_m[0]);

        // Switch: '<S423>/Switch2' incorporates:
        //   Constant: '<S377>/Constant1'
        //   RelationalOperator: '<S423>/LowerRelop1'
        if (rtb_PProdOut_h > 5.0F) {
          rtb_PProdOut_h = 5.0F;
        }

        // End of Switch: '<S423>/Switch2'

        // Product: '<S382>/Product1' incorporates:
        //   Trigonometry: '<S382>/Sin'
        rtb_Integrator_d = rtb_PProdOut_h * std::sin(rtb_Product1_k);

        // Product: '<S382>/Product' incorporates:
        //   Trigonometry: '<S382>/Cos'
        rtb_Product1_k = rtb_PProdOut_h * std::cos(rtb_Product1_k);

        // Trigonometry: '<S433>/Sin' incorporates:
        //   Inport: '<Root>/Navigation Filter Data'
        rtb_PProdOut_h = std::sin(nav.heading_rad);

        // Trigonometry: '<S433>/Cos' incorporates:
        //   Inport: '<Root>/Navigation Filter Data'
        rtb_Cos_k = std::cos(nav.heading_rad);

        // Switch: '<S10>/Switch' incorporates:
        //   Gain: '<S433>/Gain'
        //   Product: '<S383>/Product'
        //   Reshape: '<S433>/Reshape'
        //   SignalConversion generated from: '<S383>/Product'
        rtDW.Switch[0] = 0.0F;
        rtDW.Switch[0] += rtb_Cos_k * rtb_Product1_k;
        rtDW.Switch[0] += rtb_PProdOut_h * rtb_Integrator_d;
        rtDW.Switch[1] = 0.0F;
        rtDW.Switch[1] += -rtb_PProdOut_h * rtb_Product1_k;
        rtDW.Switch[1] += rtb_Cos_k * rtb_Integrator_d;
      } else {
        // Switch: '<S10>/Switch'
        rtDW.Switch[0] = 0.0F;
        rtDW.Switch[1] = 0.0F;
      }

      // End of Switch: '<S10>/Switch'

      // SignalConversion generated from: '<S10>/Constant3' incorporates:
      //   Constant: '<S10>/Constant3'
      rtDW.yaw_rate_cmd_radps_c = 0.0F;

      // Switch: '<S475>/Switch2' incorporates:
      //   Constant: '<S380>/Constant1'
      //   RelationalOperator: '<S475>/LowerRelop1'
      //   RelationalOperator: '<S475>/UpperRelop'
      //   Switch: '<S475>/Switch'
      if (rtb_PProdOut_d > 1.0F) {
        // Switch: '<S475>/Switch2'
        rtDW.Switch2_g = 1.0F;
      } else if (rtb_PProdOut_d < -2.0F) {
        // Switch: '<S475>/Switch' incorporates:
        //   Switch: '<S475>/Switch2'
        rtDW.Switch2_g = -2.0F;
      } else {
        // Switch: '<S475>/Switch2' incorporates:
        //   Switch: '<S475>/Switch'
        rtDW.Switch2_g = rtb_PProdOut_d;
      }

      // End of Switch: '<S475>/Switch2'
    }

    // End of Logic: '<Root>/motor_armed AND mode_3'
    // End of Outputs for SubSystem: '<Root>/RTL CONTROLLER'

    // Polyval: '<Root>/pitch_norm' incorporates:
    //   DataTypeConversion: '<Root>/Data Type Conversion4'
    //   Inport: '<Root>/Sensor Data'
    rtb_Cos_k = 0.00122026F * static_cast<real32_T>(sensor.inceptor.ch[2]) +
      -1.20988405F;

    // Polyval: '<Root>/roll_norm' incorporates:
    //   DataTypeConversion: '<Root>/Data Type Conversion3'
    //   Inport: '<Root>/Sensor Data'
    rtb_roll = 0.00122026F * static_cast<real32_T>(sensor.inceptor.ch[1]) +
      -1.20988405F;

    // Polyval: '<Root>/yaw_norm' incorporates:
    //   DataTypeConversion: '<Root>/Data Type Conversion2'
    //   Inport: '<Root>/Sensor Data'
    rtb_yaw = 0.00122026F * static_cast<real32_T>(sensor.inceptor.ch[3]) +
      -1.20988405F;

    // Outputs for Enabled SubSystem: '<Root>/LAND CONTROLLER' incorporates:
    //   EnablePort: '<S5>/Enable'

    // Outputs for Enabled SubSystem: '<Root>/Pos_Hold_input_conversion2' incorporates:
    //   EnablePort: '<S9>/Enable'

    // Logic: '<Root>/motor_armed AND mode_4' incorporates:
    //   Abs: '<S198>/Abs'
    //   Constant: '<S14>/Constant'
    //   Constant: '<S200>/Constant'
    //   Constant: '<S201>/Constant'
    //   DataStoreRead: '<Root>/Data Store Read1'
    //   Gain: '<S198>/Gain'
    //   Inport: '<Root>/Navigation Filter Data'
    //   Logic: '<S198>/AND'
    //   RelationalOperator: '<S14>/Compare'
    //   RelationalOperator: '<S200>/Compare'
    //   RelationalOperator: '<S201>/Compare'
    if (rtDW.motor_state && (rtb_DataStoreRead_m == 4)) {
      rtDW.LANDCONTROLLER_MODE = true;

      // Outputs for Enabled SubSystem: '<S198>/disarm motor' incorporates:
      //   EnablePort: '<S202>/Enable'
      if ((-nav.ned_pos_m[2] <= 10.0F) && (std::abs(nav.ned_vel_mps[2]) <= 0.3F))
      {
        if (!rtDW.disarmmotor_MODE) {
          // InitializeConditions for UnitDelay: '<S202>/Unit Delay'
          rtDW.UnitDelay_DSTATE = 0.0;
          rtDW.disarmmotor_MODE = true;
        }

        // Outputs for Enabled SubSystem: '<S202>/Trigger RTL' incorporates:
        //   EnablePort: '<S204>/Enable'

        // DataStoreWrite: '<S204>/Data Store Write' incorporates:
        //   Constant: '<S202>/Constant'
        //   Constant: '<S203>/Constant'
        //   RelationalOperator: '<S203>/Compare'
        //   Sum: '<S202>/Sum'
        //   UnitDelay: '<S202>/Unit Delay'
        rtDW.motor_state = ((!(rtDW.UnitDelay_DSTATE + 0.01 > 10.0)) &&
                            rtDW.motor_state);

        // End of Outputs for SubSystem: '<S202>/Trigger RTL'

        // Update for UnitDelay: '<S202>/Unit Delay' incorporates:
        //   Constant: '<S202>/Constant'
        //   Sum: '<S202>/Sum'
        rtDW.UnitDelay_DSTATE += 0.01;
      } else {
        rtDW.disarmmotor_MODE = false;
      }

      // End of Outputs for SubSystem: '<S198>/disarm motor'

      // SignalConversion generated from: '<S5>/land_cmd' incorporates:
      //   Abs: '<S198>/Abs'
      //   Constant: '<S200>/Constant'
      //   Constant: '<S201>/Constant'
      //   Gain: '<S198>/Gain'
      //   Gain: '<S9>/Gain1'
      //   Inport: '<Root>/Navigation Filter Data'
      //   Logic: '<S198>/AND'
      //   RelationalOperator: '<S200>/Compare'
      //   RelationalOperator: '<S201>/Compare'
      rtDW.vb_x_cmd_mps_o = 5.0F * rtb_Cos_k;

      // SignalConversion generated from: '<S5>/land_cmd' incorporates:
      //   Gain: '<S9>/Gain2'
      rtDW.vb_y_cmd_mps_l = 5.0F * rtb_roll;

      // SignalConversion generated from: '<S5>/land_cmd' incorporates:
      //   Gain: '<S9>/Gain3'
      rtDW.yaw_rate_cmd_radps_c53 = 0.524F * rtb_yaw;

      // Switch: '<S197>/Switch' incorporates:
      //   Constant: '<S199>/Constant'
      //   Inport: '<Root>/Navigation Filter Data'
      //   RelationalOperator: '<S199>/Compare'
      if (nav.ned_pos_m[2] <= 10.0F) {
        // Switch: '<S197>/Switch' incorporates:
        //   Constant: '<S197>/Constant1'
        rtDW.Switch_m = 0.3F;
      } else {
        // Switch: '<S197>/Switch' incorporates:
        //   Constant: '<S197>/Constant'
        rtDW.Switch_m = 1.0F;
      }

      // End of Switch: '<S197>/Switch'
    } else if (rtDW.LANDCONTROLLER_MODE) {
      // Disable for Enabled SubSystem: '<S198>/disarm motor'
      rtDW.disarmmotor_MODE = false;

      // End of Disable for SubSystem: '<S198>/disarm motor'
      rtDW.LANDCONTROLLER_MODE = false;
    }

    // End of Logic: '<Root>/motor_armed AND mode_4'
    // End of Outputs for SubSystem: '<Root>/Pos_Hold_input_conversion2'
    // End of Outputs for SubSystem: '<Root>/LAND CONTROLLER'

    // Polyval: '<Root>/throttle_norm' incorporates:
    //   DataTypeConversion: '<Root>/Data Type Conversion5'
    //   Inport: '<Root>/Sensor Data'
    rtb_throttle = 0.00061013F * static_cast<real32_T>(sensor.inceptor.ch[0]) +
      -0.104942039F;

    // Outputs for Enabled SubSystem: '<Root>/Pos_Hold_input_conversion' incorporates:
    //   EnablePort: '<S8>/Enable'

    // Logic: '<Root>/motor_armed AND mode_5' incorporates:
    //   Constant: '<S18>/Constant'
    //   RelationalOperator: '<S18>/Compare'
    if (rtb_DataStoreRead1_c && (rtb_DataStoreRead_m == 1)) {
      // Gain: '<S8>/Gain1'
      rtDW.vb_x_cmd_mps_g = 5.0F * rtb_Cos_k;

      // Gain: '<S8>/Gain2'
      rtDW.vb_y_cmd_mps_d = 5.0F * rtb_roll;

      // Gain: '<S8>/Gain3'
      rtDW.yaw_rate_cmd_radps_f = 0.524F * rtb_yaw;

      // Product: '<S369>/v_z_cmd (-1 to 1)' incorporates:
      //   Constant: '<S369>/Double'
      //   Constant: '<S369>/Normalize at Zero'
      //   Sum: '<S369>/Sum'
      rtb_PProdOut_d = (rtb_throttle - 0.5F) * 2.0F;

      // Gain: '<S369>/Gain' incorporates:
      //   Constant: '<S369>/Constant1'
      //   Constant: '<S370>/Constant'
      //   Constant: '<S371>/Constant'
      //   Product: '<S369>/Product'
      //   Product: '<S369>/Product1'
      //   RelationalOperator: '<S370>/Compare'
      //   RelationalOperator: '<S371>/Compare'
      //   Sum: '<S369>/Sum1'
      rtDW.Gain = -(static_cast<real32_T>(rtb_PProdOut_d >= 0.0F) *
                    rtb_PProdOut_d * 2.0F + static_cast<real32_T>(rtb_PProdOut_d
        < 0.0F) * rtb_PProdOut_d);

      // Switch generated from: '<Root>/Switch1'
      rtb_Subtract_p_idx_0 = rtDW.Gain;

      // Switch generated from: '<Root>/Switch1'
      rtb_Product1_k = rtDW.vb_x_cmd_mps_g;

      // Switch generated from: '<Root>/Switch1'
      rtb_PProdOut_h = rtDW.vb_y_cmd_mps_d;

      // Switch generated from: '<Root>/Switch1'
      rtb_PProdOut_d = rtDW.yaw_rate_cmd_radps_f;
    } else {
      // MultiPortSwitch generated from: '<Root>/Multiport Switch' incorporates:
      //   Switch generated from: '<Root>/Switch1'
      switch (rtb_DataStoreRead_m) {
       case 2:
        rtb_Subtract_p_idx_0 = rtDW.Switch2;
        rtb_Product1_k = rtDW.vb_xy[0];
        rtb_PProdOut_h = rtDW.vb_xy[1];
        rtb_PProdOut_d = rtDW.Saturation;
        break;

       case 3:
        rtb_Subtract_p_idx_0 = rtDW.Switch2_g;
        rtb_Product1_k = rtDW.Switch[0];
        rtb_PProdOut_h = rtDW.Switch[1];
        rtb_PProdOut_d = rtDW.yaw_rate_cmd_radps_c;
        break;

       default:
        rtb_Subtract_p_idx_0 = rtDW.Switch_m;
        rtb_Product1_k = rtDW.vb_x_cmd_mps_o;
        rtb_PProdOut_h = rtDW.vb_y_cmd_mps_l;
        rtb_PProdOut_d = rtDW.yaw_rate_cmd_radps_c53;
        break;
      }

      // End of MultiPortSwitch generated from: '<Root>/Multiport Switch'
    }

    // End of Logic: '<Root>/motor_armed AND mode_5'
    // End of Outputs for SubSystem: '<Root>/Pos_Hold_input_conversion'

    // Outputs for Enabled SubSystem: '<Root>/POS_HOLD CONTROLLER' incorporates:
    //   EnablePort: '<S7>/Enable'

    // Logic: '<Root>/motor_armed AND mode_1' incorporates:
    //   Constant: '<S15>/Constant'
    //   RelationalOperator: '<S15>/Compare'
    if (rtb_DataStoreRead1_c && (rtb_DataStoreRead_m > 0)) {
      // SignalConversion generated from: '<S7>/Command out'
      rtDW.yaw_rate_cmd_radps_c5 = rtb_PProdOut_d;

      // Trigonometry: '<S207>/Cos' incorporates:
      //   Inport: '<Root>/Navigation Filter Data'
      rtb_PProdOut_d = std::cos(nav.heading_rad);

      // Trigonometry: '<S207>/Sin' incorporates:
      //   Inport: '<Root>/Navigation Filter Data'
      rtb_Integrator_d = std::sin(nav.heading_rad);

      // Product: '<S205>/Product' incorporates:
      //   Gain: '<S207>/Gain'
      //   Inport: '<Root>/Navigation Filter Data'
      //   Reshape: '<S207>/Reshape'
      rtb_Tsamp_c = -rtb_Integrator_d * nav.ned_vel_mps[0] + rtb_PProdOut_d *
        nav.ned_vel_mps[1];

      // Sum: '<S208>/Sum' incorporates:
      //   Inport: '<Root>/Navigation Filter Data'
      //   Product: '<S205>/Product'
      //   Reshape: '<S207>/Reshape'
      rtb_PProdOut_d = rtb_Product1_k - (rtb_PProdOut_d * nav.ned_vel_mps[0] +
        rtb_Integrator_d * nav.ned_vel_mps[1]);

      // SampleTimeMath: '<S241>/Tsamp' incorporates:
      //   Constant: '<S208>/Constant2'
      //   Product: '<S238>/DProd Out'
      //
      //  About '<S241>/Tsamp':
      //   y = u * K where K = 1 / ( w * Ts )
      rtb_Tsamp_m = rtb_PProdOut_d * 0.1F * 100.0F;

      // Sum: '<S255>/Sum' incorporates:
      //   Constant: '<S208>/Constant'
      //   Delay: '<S239>/UD'
      //   DiscreteIntegrator: '<S246>/Integrator'
      //   Product: '<S251>/PProd Out'
      //   Sum: '<S239>/Diff'
      rtb_Integrator_d = (rtb_PProdOut_d * 0.5F + rtDW.Integrator_DSTATE_ps) +
        (rtb_Tsamp_m - rtDW.UD_DSTATE_n);

      // Saturate: '<S208>/Saturation'
      if (rtb_Integrator_d > 0.523F) {
        // Gain: '<S208>/Gain'
        rtDW.Gain_i = -0.523F;
      } else if (rtb_Integrator_d < -0.523F) {
        // Gain: '<S208>/Gain'
        rtDW.Gain_i = 0.523F;
      } else {
        // Gain: '<S208>/Gain'
        rtDW.Gain_i = -rtb_Integrator_d;
      }

      // End of Saturate: '<S208>/Saturation'

      // Gain: '<S235>/ZeroGain'
      rtb_Product1_k = 0.0F * rtb_Integrator_d;

      // DeadZone: '<S237>/DeadZone'
      if (rtb_Integrator_d >= (rtMinusInfF)) {
        rtb_Integrator_d = 0.0F;
      }

      // End of DeadZone: '<S237>/DeadZone'

      // Product: '<S243>/IProd Out' incorporates:
      //   Constant: '<S208>/Constant1'
      rtb_PProdOut_d *= 0.01F;

      // Signum: '<S235>/SignPreIntegrator'
      if (rtb_PProdOut_d < 0.0F) {
        rtb_Switch_dv = -1.0F;
      } else if (rtb_PProdOut_d > 0.0F) {
        rtb_Switch_dv = 1.0F;
      } else if (rtb_PProdOut_d == 0.0F) {
        rtb_Switch_dv = 0.0F;
      } else {
        rtb_Switch_dv = (rtNaNF);
      }

      // End of Signum: '<S235>/SignPreIntegrator'

      // Switch: '<S235>/Switch' incorporates:
      //   Constant: '<S235>/Constant1'
      //   DataTypeConversion: '<S235>/DataTypeConv2'
      //   Logic: '<S235>/AND3'
      //   RelationalOperator: '<S235>/Equal1'
      //   RelationalOperator: '<S235>/NotEqual'
      if ((rtb_Product1_k != rtb_Integrator_d) && (0 == static_cast<int8_T>
           (rtb_Switch_dv))) {
        rtb_Integrator_d = 0.0F;
      } else {
        rtb_Integrator_d = rtb_PProdOut_d;
      }

      // End of Switch: '<S235>/Switch'

      // Sum: '<S209>/Sum'
      rtb_PProdOut_d = rtb_PProdOut_h - rtb_Tsamp_c;

      // SampleTimeMath: '<S294>/Tsamp' incorporates:
      //   Constant: '<S209>/Constant2'
      //   Product: '<S291>/DProd Out'
      //
      //  About '<S294>/Tsamp':
      //   y = u * K where K = 1 / ( w * Ts )
      rtb_Tsamp_c = rtb_PProdOut_d * 0.1F * 100.0F;

      // Sum: '<S308>/Sum' incorporates:
      //   Constant: '<S209>/Constant'
      //   Delay: '<S292>/UD'
      //   DiscreteIntegrator: '<S299>/Integrator'
      //   Product: '<S304>/PProd Out'
      //   Sum: '<S292>/Diff'
      rtb_Product1_k = (rtb_PProdOut_d * 0.5F + rtDW.Integrator_DSTATE_m) +
        (rtb_Tsamp_c - rtDW.UD_DSTATE_m);

      // Product: '<S296>/IProd Out' incorporates:
      //   Constant: '<S209>/Constant1'
      rtb_PProdOut_d *= 0.01F;

      // DeadZone: '<S290>/DeadZone'
      if (rtb_Product1_k >= (rtMinusInfF)) {
        rtb_PProdOut_h = 0.0F;
      } else {
        rtb_PProdOut_h = (rtNaNF);
      }

      // End of DeadZone: '<S290>/DeadZone'

      // Signum: '<S288>/SignPreIntegrator'
      if (rtb_PProdOut_d < 0.0F) {
        rtb_Switch_dv = -1.0F;
      } else if (rtb_PProdOut_d > 0.0F) {
        rtb_Switch_dv = 1.0F;
      } else if (rtb_PProdOut_d == 0.0F) {
        rtb_Switch_dv = 0.0F;
      } else {
        rtb_Switch_dv = (rtNaNF);
      }

      // End of Signum: '<S288>/SignPreIntegrator'

      // Switch: '<S288>/Switch' incorporates:
      //   Constant: '<S288>/Constant1'
      //   DataTypeConversion: '<S288>/DataTypeConv2'
      //   Gain: '<S288>/ZeroGain'
      //   Logic: '<S288>/AND3'
      //   RelationalOperator: '<S288>/Equal1'
      //   RelationalOperator: '<S288>/NotEqual'
      if ((0.0F * rtb_Product1_k != rtb_PProdOut_h) && (0 == static_cast<int8_T>
           (rtb_Switch_dv))) {
        rtb_Switch_dv = 0.0F;
      } else {
        rtb_Switch_dv = rtb_PProdOut_d;
      }

      // End of Switch: '<S288>/Switch'

      // Saturate: '<S209>/Saturation'
      if (rtb_Product1_k > 0.523F) {
        // Saturate: '<S209>/Saturation'
        rtDW.Saturation_o = 0.523F;
      } else if (rtb_Product1_k < -0.523F) {
        // Saturate: '<S209>/Saturation'
        rtDW.Saturation_o = -0.523F;
      } else {
        // Saturate: '<S209>/Saturation'
        rtDW.Saturation_o = rtb_Product1_k;
      }

      // End of Saturate: '<S209>/Saturation'

      // Sum: '<S206>/Sum' incorporates:
      //   Inport: '<Root>/Navigation Filter Data'
      rtb_PProdOut_d = rtb_Subtract_p_idx_0 - nav.ned_vel_mps[2];

      // SampleTimeMath: '<S347>/Tsamp' incorporates:
      //   Constant: '<S206>/D_vz'
      //   Product: '<S344>/DProd Out'
      //
      //  About '<S347>/Tsamp':
      //   y = u * K where K = 1 / ( w * Ts )
      rtb_Product1_k = rtb_PProdOut_d * 0.005F * 100.0F;

      // Sum: '<S361>/Sum' incorporates:
      //   Constant: '<S206>/P_vz'
      //   Delay: '<S345>/UD'
      //   DiscreteIntegrator: '<S352>/Integrator'
      //   Product: '<S357>/PProd Out'
      //   Sum: '<S345>/Diff'
      rtb_PProdOut_h = (rtb_PProdOut_d * 0.09F + rtDW.Integrator_DSTATE_h) +
        (rtb_Product1_k - rtDW.UD_DSTATE_ms);

      // Gain: '<S206>/Gain'
      rtb_Subtract_p_idx_0 = -rtb_PProdOut_h;

      // Gain: '<S341>/ZeroGain'
      rtb_ZeroGain_p = 0.0F * rtb_PProdOut_h;

      // DeadZone: '<S343>/DeadZone'
      if (rtb_PProdOut_h >= (rtMinusInfF)) {
        rtb_PProdOut_h = 0.0F;
      }

      // End of DeadZone: '<S343>/DeadZone'

      // Product: '<S349>/IProd Out' incorporates:
      //   Constant: '<S206>/I_vz'
      rtb_PProdOut_d *= 0.05F;

      // Saturate: '<S206>/Saturation' incorporates:
      //   Constant: '<S206>/Constant2'
      //   Sum: '<S206>/Sum1'
      if (rtb_Subtract_p_idx_0 + 0.6724F > 1.0F) {
        // Saturate: '<S206>/Saturation'
        rtDW.Saturation_a = 1.0F;
      } else if (rtb_Subtract_p_idx_0 + 0.6724F < 0.0F) {
        // Saturate: '<S206>/Saturation'
        rtDW.Saturation_a = 0.0F;
      } else {
        // Saturate: '<S206>/Saturation'
        rtDW.Saturation_a = rtb_Subtract_p_idx_0 + 0.6724F;
      }

      // End of Saturate: '<S206>/Saturation'

      // Update for DiscreteIntegrator: '<S246>/Integrator'
      rtDW.Integrator_DSTATE_ps += 0.01F * rtb_Integrator_d;

      // Update for Delay: '<S239>/UD'
      rtDW.UD_DSTATE_n = rtb_Tsamp_m;

      // Update for DiscreteIntegrator: '<S299>/Integrator'
      rtDW.Integrator_DSTATE_m += 0.01F * rtb_Switch_dv;

      // Update for Delay: '<S292>/UD'
      rtDW.UD_DSTATE_m = rtb_Tsamp_c;

      // Signum: '<S341>/SignPreIntegrator'
      if (rtb_PProdOut_d < 0.0F) {
        rtb_Switch_dv = -1.0F;
      } else if (rtb_PProdOut_d > 0.0F) {
        rtb_Switch_dv = 1.0F;
      } else if (rtb_PProdOut_d == 0.0F) {
        rtb_Switch_dv = 0.0F;
      } else {
        rtb_Switch_dv = (rtNaNF);
      }

      // End of Signum: '<S341>/SignPreIntegrator'

      // Switch: '<S341>/Switch' incorporates:
      //   Constant: '<S341>/Constant1'
      //   DataTypeConversion: '<S341>/DataTypeConv2'
      //   Logic: '<S341>/AND3'
      //   RelationalOperator: '<S341>/Equal1'
      //   RelationalOperator: '<S341>/NotEqual'
      if ((rtb_ZeroGain_p != rtb_PProdOut_h) && (0 == static_cast<int8_T>
           (rtb_Switch_dv))) {
        rtb_PProdOut_d = 0.0F;
      }

      // End of Switch: '<S341>/Switch'

      // Update for DiscreteIntegrator: '<S352>/Integrator'
      rtDW.Integrator_DSTATE_h += 0.01F * rtb_PProdOut_d;

      // Update for Delay: '<S345>/UD'
      rtDW.UD_DSTATE_ms = rtb_Product1_k;
    }

    // End of Logic: '<Root>/motor_armed AND mode_1'
    // End of Outputs for SubSystem: '<Root>/POS_HOLD CONTROLLER'

    // Logic: '<Root>/motor_armed AND mode_0' incorporates:
    //   Constant: '<S17>/Constant'
    //   RelationalOperator: '<S17>/Compare'
    rtb_motor_armedANDmode_0 = (rtb_DataStoreRead1_c && (rtb_DataStoreRead_m <=
      0));

    // Logic: '<Root>/NOT'
    rtb_Compare_ey = !rtb_motor_armedANDmode_0;

    // Outputs for Enabled SubSystem: '<Root>/Stab_input_conversion' incorporates:
    //   EnablePort: '<S11>/Enable'
    if (rtb_motor_armedANDmode_0) {
      // Gain: '<S11>/Gain'
      rtDW.throttle_cc = rtb_throttle;

      // Gain: '<S11>/Gain1'
      rtDW.pitch_angle_cmd_rad = -0.523F * rtb_Cos_k;

      // Gain: '<S11>/Gain2'
      rtDW.roll_angle_cmd_rad = 0.52F * rtb_roll;

      // Gain: '<S11>/Gain3'
      rtDW.yaw_rate_cmd_radps = 0.524F * rtb_yaw;
    }

    // End of Outputs for SubSystem: '<Root>/Stab_input_conversion'

    // Switch generated from: '<Root>/Switch'
    if (rtb_Compare_ey) {
      rtb_PProdOut_d = rtDW.Saturation_a;
      rtb_Switch_dv = rtDW.Saturation_o;
    } else {
      rtb_PProdOut_d = rtDW.throttle_cc;
      rtb_Switch_dv = rtDW.roll_angle_cmd_rad;
    }

    // Sum: '<S25>/stab_roll_angle_error_calc' incorporates:
    //   Inport: '<Root>/Navigation Filter Data'
    rtb_Cos_k = rtb_Switch_dv - nav.roll_rad;

    // SampleTimeMath: '<S111>/Tsamp' incorporates:
    //   Constant: '<S25>/Constant2'
    //   Product: '<S108>/DProd Out'
    //
    //  About '<S111>/Tsamp':
    //   y = u * K where K = 1 / ( w * Ts )
    rtb_roll = rtb_Cos_k * 0.02F * 100.0F;

    // Sum: '<S125>/Sum' incorporates:
    //   Constant: '<S25>/Constant'
    //   Delay: '<S109>/UD'
    //   DiscreteIntegrator: '<S116>/Integrator'
    //   Product: '<S121>/PProd Out'
    //   Sum: '<S109>/Diff'
    rtb_yaw = (rtb_Cos_k * 0.04F + rtDW.Integrator_DSTATE) + (rtb_roll -
      rtDW.UD_DSTATE);

    // Saturate: '<S25>/stab_roll_rate_saturation'
    if (rtb_yaw > 1.0F) {
      rtb_ZeroGain_p = 1.0F;
    } else if (rtb_yaw < -1.0F) {
      rtb_ZeroGain_p = -1.0F;
    } else {
      rtb_ZeroGain_p = rtb_yaw;
    }

    // End of Saturate: '<S25>/stab_roll_rate_saturation'

    // Switch generated from: '<Root>/Switch'
    if (rtb_Compare_ey) {
      rtb_pitch_angle_cmd_rad = rtDW.Gain_i;
    } else {
      rtb_pitch_angle_cmd_rad = rtDW.pitch_angle_cmd_rad;
    }

    // Sum: '<S24>/stab_pitch_angle_error_calc' incorporates:
    //   Inport: '<Root>/Navigation Filter Data'
    rtb_throttle = rtb_pitch_angle_cmd_rad - nav.pitch_rad;

    // SampleTimeMath: '<S58>/Tsamp' incorporates:
    //   Constant: '<S24>/Constant2'
    //   Product: '<S55>/DProd Out'
    //
    //  About '<S58>/Tsamp':
    //   y = u * K where K = 1 / ( w * Ts )
    rtb_Subtract_p_idx_0 = rtb_throttle * 0.02F * 100.0F;

    // Sum: '<S72>/Sum' incorporates:
    //   Constant: '<S24>/Constant'
    //   Delay: '<S56>/UD'
    //   DiscreteIntegrator: '<S63>/Integrator'
    //   Product: '<S68>/PProd Out'
    //   Sum: '<S56>/Diff'
    rtb_PProdOut_h = (rtb_throttle * 0.04F + rtDW.Integrator_DSTATE_l) +
      (rtb_Subtract_p_idx_0 - rtDW.UD_DSTATE_b);

    // Saturate: '<S24>/stab_pitch_rate_saturation'
    if (rtb_PProdOut_h > 1.0F) {
      rtb_stab_pitch_rate_saturation = 1.0F;
    } else if (rtb_PProdOut_h < -1.0F) {
      rtb_stab_pitch_rate_saturation = -1.0F;
    } else {
      rtb_stab_pitch_rate_saturation = rtb_PProdOut_h;
    }

    // End of Saturate: '<S24>/stab_pitch_rate_saturation'

    // Switch generated from: '<Root>/Switch'
    if (rtb_Compare_ey) {
      rtb_Product1_k = rtDW.yaw_rate_cmd_radps_c5;
    } else {
      rtb_Product1_k = rtDW.yaw_rate_cmd_radps;
    }

    // Sum: '<S26>/stab_yaw_rate_error_calc' incorporates:
    //   Inport: '<Root>/Navigation Filter Data'
    rtb_Integrator_d = rtb_Product1_k - nav.gyro_radps[2];

    // SampleTimeMath: '<S164>/Tsamp' incorporates:
    //   Constant: '<S26>/Constant2'
    //   Product: '<S161>/DProd Out'
    //
    //  About '<S164>/Tsamp':
    //   y = u * K where K = 1 / ( w * Ts )
    rtb_Tsamp_m = rtb_Integrator_d * 0.02F * 100.0F;

    // Sum: '<S178>/Sum' incorporates:
    //   Constant: '<S26>/Constant'
    //   Delay: '<S162>/UD'
    //   DiscreteIntegrator: '<S169>/Integrator'
    //   Product: '<S174>/PProd Out'
    //   Sum: '<S162>/Diff'
    rtb_Tsamp_c = (rtb_Integrator_d * 0.5F + rtDW.Integrator_DSTATE_f) +
      (rtb_Tsamp_m - rtDW.UD_DSTATE_l);

    // Switch: '<Root>/switch_motor_out'
    if (rtb_DataStoreRead1_c) {
      // Product: '<S6>/Multiply' incorporates:
      //   Math: '<S6>/Transpose'
      //   Reshape: '<S6>/Reshape'
      for (i = 0; i < 8; i++) {
        rtb_Reshape_l_tmp = i << 2;
        rtb_Reshape_l_0 = rtConstB.Transpose[rtb_Reshape_l_tmp + 3] *
          rtb_Tsamp_c + (rtConstB.Transpose[rtb_Reshape_l_tmp + 2] *
                         rtb_stab_pitch_rate_saturation +
                         (rtConstB.Transpose[rtb_Reshape_l_tmp + 1] *
                          rtb_ZeroGain_p + rtConstB.Transpose[rtb_Reshape_l_tmp]
                          * rtb_PProdOut_d));

        // Saturate: '<S6>/Saturation' incorporates:
        //   Math: '<S6>/Transpose'
        //   Reshape: '<S6>/Reshape'
        if (rtb_Reshape_l_0 <= 0.15F) {
          rtb_val[i] = 0.15F;
        } else {
          rtb_val[i] = rtb_Reshape_l_0;
        }

        // End of Saturate: '<S6>/Saturation'
      }

      // End of Product: '<S6>/Multiply'
    } else {
      for (i = 0; i < 8; i++) {
        rtb_val[i] = 0.0F;
      }
    }

    // End of Switch: '<Root>/switch_motor_out'

    // Outport: '<Root>/VmsData' incorporates:
    //   BusCreator: '<S12>/Bus Creator'
    //   BusCreator: '<S2>/Bus Creator3'
    //   BusCreator: '<S485>/Bus Creator'
    //   Constant: '<S2>/consumed_mah'
    //   Constant: '<S2>/current_ma'
    //   Constant: '<S2>/remaining_prcnt'
    //   Constant: '<S2>/remaining_time_s'
    //   Constant: '<S2>/voltage_v'
    //   Constant: '<S485>/Constant'
    //   Constant: '<S485>/Constant1'
    //   Constant: '<S486>/Constant'
    //   DataStoreRead: '<S491>/Data Store Read1'
    //   DataTypeConversion: '<Root>/Cast To Single'
    //   DataTypeConversion: '<S486>/Data Type Conversion'
    //   Gain: '<Root>/Gain'
    //   Gain: '<S486>/Gain'
    //   Inport: '<Root>/Navigation Filter Data'
    //   Logic: '<S491>/AND'
    //   SignalConversion generated from: '<S12>/Bus Creator'
    //   SignalConversion generated from: '<S485>/Bus Creator'
    //   Sum: '<S486>/Sum'
    //
    ctrl->motors_enabled = rtb_DataStoreRead1_c;
    ctrl->waypoint_reached = (rtb_Compare_id && rtDW.autocontinue);
    ctrl->mode = rtb_DataStoreRead_m;
    ctrl->throttle_cmd_prcnt = 100.0F * rtb_PProdOut_d;
    ctrl->aux[0] = rtb_PProdOut_d;
    ctrl->aux[1] = rtb_ZeroGain_p;
    ctrl->aux[2] = rtb_Switch_dv;
    ctrl->aux[3] = nav.roll_rad;
    ctrl->aux[4] = rtb_stab_pitch_rate_saturation;
    ctrl->aux[5] = rtb_pitch_angle_cmd_rad;
    ctrl->aux[6] = nav.pitch_rad;
    ctrl->aux[7] = rtb_Tsamp_c;
    ctrl->aux[8] = rtb_Product1_k;
    ctrl->aux[9] = nav.gyro_radps[2];
    ctrl->aux[10] = 0.0F;
    ctrl->aux[11] = 0.0F;
    ctrl->aux[12] = 0.0F;
    ctrl->aux[13] = 0.0F;
    ctrl->aux[14] = 0.0F;
    ctrl->aux[15] = 0.0F;
    ctrl->aux[16] = 0.0F;
    ctrl->aux[17] = 0.0F;
    ctrl->aux[18] = 0.0F;
    ctrl->aux[19] = 0.0F;
    ctrl->aux[20] = 0.0F;
    ctrl->aux[21] = 0.0F;
    ctrl->aux[22] = 0.0F;
    ctrl->aux[23] = 0.0F;
    ctrl->sbus.ch17 = false;
    ctrl->sbus.ch18 = false;
    std::memset(&ctrl->sbus.cmd[0], 0, sizeof(real32_T) << 4U);
    for (i = 0; i < 16; i++) {
      ctrl->sbus.cnt[i] = 0;
    }

    for (i = 0; i < 8; i++) {
      rtb_Product1_k = rtb_val[i];
      ctrl->pwm.cnt[i] = static_cast<int16_T>(std::floor(1000.0F *
        rtb_Product1_k + 1000.0));
      ctrl->pwm.cmd[i] = rtb_Product1_k;
      ctrl->analog.val[i] = 0.0F;
    }

    ctrl->battery.voltage_v = 1.0F;
    ctrl->battery.current_ma = 1.0F;
    ctrl->battery.consumed_mah = 1.0F;
    ctrl->battery.remaining_prcnt = 1.0F;
    ctrl->battery.remaining_time_s = 1.0F;

    // End of Outport: '<Root>/VmsData'

    // Outputs for Enabled SubSystem: '<S186>/disarm motor' incorporates:
    //   EnablePort: '<S191>/Enable'

    // RelationalOperator: '<S189>/Compare' incorporates:
    //   Constant: '<S189>/Constant'
    //   Inport: '<Root>/Sensor Data'
    if (sensor.power_module.voltage_v <= 46.8F) {
      if (!rtDW.disarmmotor_MODE_m) {
        // InitializeConditions for UnitDelay: '<S191>/Unit Delay'
        rtDW.UnitDelay_DSTATE_n = 0.0;
        rtDW.disarmmotor_MODE_m = true;
      }

      // RelationalOperator: '<S192>/Compare' incorporates:
      //   Constant: '<S191>/Constant'
      //   Constant: '<S192>/Constant'
      //   Sum: '<S191>/Sum'
      //   UnitDelay: '<S191>/Unit Delay'
      rtDW.Compare = (rtDW.UnitDelay_DSTATE_n + 0.01 > 15.0);

      // Update for UnitDelay: '<S191>/Unit Delay' incorporates:
      //   Constant: '<S191>/Constant'
      //   Sum: '<S191>/Sum'
      rtDW.UnitDelay_DSTATE_n += 0.01;
    } else {
      rtDW.disarmmotor_MODE_m = false;
    }

    // End of RelationalOperator: '<S189>/Compare'
    // End of Outputs for SubSystem: '<S186>/disarm motor'

    // Outputs for Enabled SubSystem: '<S186>/Trigger Land' incorporates:
    //   EnablePort: '<S190>/Enable'

    // Logic: '<S186>/AND' incorporates:
    //   Constant: '<S188>/Constant'
    //   DataStoreRead: '<S186>/Data Store Read'
    //   DataStoreRead: '<S186>/Data Store Read1'
    //   Logic: '<S186>/NOR'
    //   Logic: '<S186>/NOT'
    //   RelationalOperator: '<S188>/Compare'
    if (rtDW.Compare && ((rtDW.cur_mode == 5) && rtDW.motor_state)) {
      // DataStoreWrite: '<S190>/Data Store Write' incorporates:
      //   Constant: '<S190>/land_mode'
      rtDW.cur_mode = 5;
    }

    // End of Logic: '<S186>/AND'
    // End of Outputs for SubSystem: '<S186>/Trigger Land'

    // Outputs for Enabled SubSystem: '<S187>/Trigger RTL' incorporates:
    //   EnablePort: '<S196>/Enable'

    // Logic: '<S187>/AND' incorporates:
    //   Constant: '<S193>/Constant'
    //   Constant: '<S194>/Constant'
    //   Constant: '<S195>/Constant'
    //   DataStoreRead: '<S187>/Data Store Read'
    //   DataStoreRead: '<S187>/Data Store Read1'
    //   Inport: '<Root>/Sensor Data'
    //   Logic: '<S187>/NOR'
    //   Logic: '<S187>/NOT'
    //   RelationalOperator: '<S193>/Compare'
    //   RelationalOperator: '<S194>/Compare'
    //   RelationalOperator: '<S195>/Compare'
    if (sensor.inceptor.failsafe && ((rtDW.cur_mode == 3) && (rtDW.cur_mode == 4)
         && (rtDW.cur_mode == 5) && rtDW.motor_state)) {
      // DataStoreWrite: '<S196>/Data Store Write' incorporates:
      //   Constant: '<S196>/rtl_mode'
      rtDW.cur_mode = 3;
    }

    // End of Logic: '<S187>/AND'
    // End of Outputs for SubSystem: '<S187>/Trigger RTL'

    // Logic: '<Root>/nav_init AND motor_enable' incorporates:
    //   Constant: '<S3>/Constant'
    //   DataTypeConversion: '<Root>/Data Type Conversion'
    //   Inport: '<Root>/Navigation Filter Data'
    //   Inport: '<Root>/Sensor Data'
    //   Polyval: '<Root>/throttle_en_norm'
    //   RelationalOperator: '<S3>/Compare'
    rtb_DataStoreRead1_c = ((0.00122026F * static_cast<real32_T>
      (sensor.inceptor.ch[6]) + -1.20988405F > 0.0F) && nav.nav_initialized);

    // Outputs for Enabled SubSystem: '<Root>/manual_arming' incorporates:
    //   EnablePort: '<S22>/Enable'

    // RelationalOperator: '<S20>/FixPt Relational Operator' incorporates:
    //   UnitDelay: '<S20>/Delay Input1'
    //
    //  Block description for '<S20>/Delay Input1':
    //
    //   Store in Global RAM
    if (rtb_DataStoreRead1_c != rtDW.DelayInput1_DSTATE_h) {
      // DataStoreWrite: '<S22>/Data Store Write'
      rtDW.motor_state = rtb_DataStoreRead1_c;
    }

    // End of RelationalOperator: '<S20>/FixPt Relational Operator'
    // End of Outputs for SubSystem: '<Root>/manual_arming'

    // Polyval: '<Root>/mode_norm' incorporates:
    //   DataTypeConversion: '<Root>/Data Type Conversion1'
    //   Inport: '<Root>/Sensor Data'
    rtb_PProdOut_d = 0.00122026F * static_cast<real32_T>(sensor.inceptor.ch[4])
      + -0.209884077F;

    // DataTypeConversion: '<Root>/Data Type Conversion6'
    if (std::abs(rtb_PProdOut_d) >= 0.5F) {
      rtb_PProdOut_d = std::floor(rtb_PProdOut_d + 0.5F);
    } else {
      rtb_PProdOut_d *= 0.0F;
    }

    // Outputs for Enabled SubSystem: '<Root>/manual_mode_selection' incorporates:
    //   EnablePort: '<S23>/Enable'

    // RelationalOperator: '<S21>/FixPt Relational Operator' incorporates:
    //   DataTypeConversion: '<Root>/Data Type Conversion6'
    //   UnitDelay: '<S21>/Delay Input1'
    //
    //  Block description for '<S21>/Delay Input1':
    //
    //   Store in Global RAM
    if (static_cast<int8_T>(rtb_PProdOut_d) != rtDW.DelayInput1_DSTATE_b) {
      // DataStoreWrite: '<S23>/Data Store Write'
      rtDW.cur_mode = static_cast<int8_T>(rtb_PProdOut_d);
    }

    // End of RelationalOperator: '<S21>/FixPt Relational Operator'
    // End of Outputs for SubSystem: '<Root>/manual_mode_selection'

    // Gain: '<S52>/ZeroGain'
    rtb_Product1_k = 0.0F * rtb_PProdOut_h;

    // DeadZone: '<S54>/DeadZone'
    if (rtb_PProdOut_h >= (rtMinusInfF)) {
      rtb_PProdOut_h = 0.0F;
    }

    // End of DeadZone: '<S54>/DeadZone'

    // Product: '<S60>/IProd Out' incorporates:
    //   Constant: '<S24>/Constant1'
    rtb_throttle *= 0.04F;

    // Gain: '<S105>/ZeroGain'
    rtb_Switch_dv = 0.0F * rtb_yaw;

    // DeadZone: '<S107>/DeadZone'
    if (rtb_yaw >= (rtMinusInfF)) {
      rtb_yaw = 0.0F;
    }

    // End of DeadZone: '<S107>/DeadZone'

    // Product: '<S113>/IProd Out' incorporates:
    //   Constant: '<S25>/Constant1'
    rtb_Cos_k *= 0.04F;

    // Product: '<S166>/IProd Out' incorporates:
    //   Constant: '<S26>/Constant1'
    rtb_Integrator_d *= 0.05F;

    // DeadZone: '<S160>/DeadZone'
    if (rtb_Tsamp_c >= (rtMinusInfF)) {
      rtb_pitch_angle_cmd_rad = 0.0F;
    } else {
      rtb_pitch_angle_cmd_rad = (rtNaNF);
    }

    // End of DeadZone: '<S160>/DeadZone'

    // Signum: '<S158>/SignPreIntegrator'
    if (rtb_Integrator_d < 0.0F) {
      rtb_ZeroGain_p = -1.0F;
    } else if (rtb_Integrator_d > 0.0F) {
      rtb_ZeroGain_p = 1.0F;
    } else if (rtb_Integrator_d == 0.0F) {
      rtb_ZeroGain_p = 0.0F;
    } else {
      rtb_ZeroGain_p = (rtNaNF);
    }

    // End of Signum: '<S158>/SignPreIntegrator'

    // Switch: '<S158>/Switch' incorporates:
    //   Constant: '<S158>/Constant1'
    //   DataTypeConversion: '<S158>/DataTypeConv2'
    //   Gain: '<S158>/ZeroGain'
    //   Logic: '<S158>/AND3'
    //   RelationalOperator: '<S158>/Equal1'
    //   RelationalOperator: '<S158>/NotEqual'
    if ((0.0F * rtb_Tsamp_c != rtb_pitch_angle_cmd_rad) && (0 ==
         static_cast<int8_T>(rtb_ZeroGain_p))) {
      rtb_Tsamp_c = 0.0F;
    } else {
      rtb_Tsamp_c = rtb_Integrator_d;
    }

    // End of Switch: '<S158>/Switch'

    // Outputs for Triggered SubSystem: '<S13>/first_wp' incorporates:
    //   TriggerPort: '<S489>/Trigger'
    if (rtb_motor_armedANDmode_2 && (rtPrevZCX.first_wp_Trig_ZCE != 1)) {
      // Gain: '<S656>/Gain' incorporates:
      //   Inport: '<Root>/Navigation Filter Data'
      rtb_Gain_dj_idx_0 = 57.295779513082323 * nav.home_lat_rad;

      // Switch: '<S670>/Switch' incorporates:
      //   Abs: '<S670>/Abs'
      //   Bias: '<S670>/Bias'
      //   Bias: '<S670>/Bias1'
      //   Constant: '<S670>/Constant2'
      //   Constant: '<S671>/Constant'
      //   Math: '<S670>/Math Function1'
      //   RelationalOperator: '<S671>/Compare'
      if (std::abs(rtb_Gain_dj_idx_0) > 180.0) {
        rtb_Abs1_n = rt_modd_snf(rtb_Gain_dj_idx_0 + 180.0, 360.0) + -180.0;
      } else {
        rtb_Abs1_n = rtb_Gain_dj_idx_0;
      }

      // End of Switch: '<S670>/Switch'

      // Abs: '<S667>/Abs1'
      rtb_Sum_cd = std::abs(rtb_Abs1_n);

      // Switch: '<S667>/Switch' incorporates:
      //   Bias: '<S667>/Bias'
      //   Bias: '<S667>/Bias1'
      //   Constant: '<S658>/Constant'
      //   Constant: '<S658>/Constant1'
      //   Constant: '<S669>/Constant'
      //   Gain: '<S667>/Gain'
      //   Product: '<S667>/Divide1'
      //   RelationalOperator: '<S669>/Compare'
      //   Switch: '<S658>/Switch1'
      if (rtb_Sum_cd > 90.0) {
        // Signum: '<S667>/Sign1'
        if (rtb_Abs1_n < 0.0) {
          rtb_Abs1_n = -1.0;
        } else if (rtb_Abs1_n > 0.0) {
          rtb_Abs1_n = 1.0;
        } else if (rtb_Abs1_n == 0.0) {
          rtb_Abs1_n = 0.0;
        } else {
          rtb_Abs1_n = (rtNaN);
        }

        // End of Signum: '<S667>/Sign1'
        rtb_Abs1_n *= -(rtb_Sum_cd + -90.0) + 90.0;
        i = 180;
      } else {
        i = 0;
      }

      // End of Switch: '<S667>/Switch'

      // Sum: '<S658>/Sum' incorporates:
      //   Gain: '<S656>/Gain'
      //   Inport: '<Root>/Navigation Filter Data'
      rtb_Switch_ejf = 57.295779513082323 * nav.home_lon_rad +
        static_cast<real_T>(i);

      // Switch: '<S668>/Switch' incorporates:
      //   Abs: '<S668>/Abs'
      //   Bias: '<S668>/Bias'
      //   Bias: '<S668>/Bias1'
      //   Constant: '<S668>/Constant2'
      //   Constant: '<S672>/Constant'
      //   Math: '<S668>/Math Function1'
      //   RelationalOperator: '<S672>/Compare'
      if (std::abs(rtb_Switch_ejf) > 180.0) {
        rtb_Switch_ejf = rt_modd_snf(rtb_Switch_ejf + 180.0, 360.0) + -180.0;
      }

      // End of Switch: '<S668>/Switch'

      // Sum: '<S655>/Sum1' incorporates:
      //   DataTypeConversion: '<S489>/Data Type Conversion1'
      //   Gain: '<S489>/Gain3'
      //   Inport: '<Root>/Telemetry Data'
      //   Selector: '<S489>/Selector'
      rtb_Gain_dj_idx_0 = 1.0E-7 * static_cast<real_T>(telem.flight_plan[0].x) -
        rtb_Abs1_n;
      rtb_Gain_dj_idx_1 = 1.0E-7 * static_cast<real_T>(telem.flight_plan[0].y) -
        rtb_Switch_ejf;

      // Switch: '<S664>/Switch' incorporates:
      //   Abs: '<S664>/Abs'
      //   Bias: '<S664>/Bias'
      //   Bias: '<S664>/Bias1'
      //   Constant: '<S664>/Constant2'
      //   Constant: '<S665>/Constant'
      //   Math: '<S664>/Math Function1'
      //   RelationalOperator: '<S665>/Compare'
      if (std::abs(rtb_Gain_dj_idx_0) > 180.0) {
        rtb_Switch_ejf = rt_modd_snf(rtb_Gain_dj_idx_0 + 180.0, 360.0) + -180.0;
      } else {
        rtb_Switch_ejf = rtb_Gain_dj_idx_0;
      }

      // End of Switch: '<S664>/Switch'

      // Abs: '<S661>/Abs1'
      rtb_Sum_cd = std::abs(rtb_Switch_ejf);

      // Switch: '<S661>/Switch' incorporates:
      //   Bias: '<S661>/Bias'
      //   Bias: '<S661>/Bias1'
      //   Constant: '<S657>/Constant'
      //   Constant: '<S657>/Constant1'
      //   Constant: '<S663>/Constant'
      //   Gain: '<S661>/Gain'
      //   Product: '<S661>/Divide1'
      //   RelationalOperator: '<S663>/Compare'
      //   Switch: '<S657>/Switch1'
      if (rtb_Sum_cd > 90.0) {
        // Signum: '<S661>/Sign1'
        if (rtb_Switch_ejf < 0.0) {
          rtb_Switch_ejf = -1.0;
        } else if (rtb_Switch_ejf > 0.0) {
          rtb_Switch_ejf = 1.0;
        } else if (rtb_Switch_ejf == 0.0) {
          rtb_Switch_ejf = 0.0;
        } else {
          rtb_Switch_ejf = (rtNaN);
        }

        // End of Signum: '<S661>/Sign1'
        rtb_Switch_ejf *= -(rtb_Sum_cd + -90.0) + 90.0;
        i = 180;
      } else {
        i = 0;
      }

      // End of Switch: '<S661>/Switch'

      // Sum: '<S657>/Sum'
      rtb_Switch_fq = static_cast<real_T>(i) + rtb_Gain_dj_idx_1;

      // Switch: '<S662>/Switch' incorporates:
      //   Abs: '<S662>/Abs'
      //   Bias: '<S662>/Bias'
      //   Bias: '<S662>/Bias1'
      //   Constant: '<S662>/Constant2'
      //   Constant: '<S666>/Constant'
      //   Math: '<S662>/Math Function1'
      //   RelationalOperator: '<S666>/Compare'
      if (std::abs(rtb_Switch_fq) > 180.0) {
        rtb_Switch_fq = rt_modd_snf(rtb_Switch_fq + 180.0, 360.0) + -180.0;
      }

      // End of Switch: '<S662>/Switch'

      // UnitConversion: '<S660>/Unit Conversion'
      // Unit Conversion - from: deg to: rad
      // Expression: output = (0.0174533*input) + (0)
      rtb_UnitConversion_idx_0 = 0.017453292519943295 * rtb_Switch_ejf;
      rtb_UnitConversion_idx_1 = 0.017453292519943295 * rtb_Switch_fq;

      // UnitConversion: '<S675>/Unit Conversion'
      // Unit Conversion - from: deg to: rad
      // Expression: output = (0.0174533*input) + (0)
      rtb_Abs1_n *= 0.017453292519943295;

      // Trigonometry: '<S676>/Trigonometric Function1'
      rtb_Switch_ejf = std::sin(rtb_Abs1_n);

      // Sum: '<S676>/Sum1' incorporates:
      //   Constant: '<S676>/Constant'
      //   Product: '<S676>/Product1'
      rtb_Switch_ejf = 1.0 - 0.0066943799901413295 * rtb_Switch_ejf *
        rtb_Switch_ejf;

      // Product: '<S674>/Product1' incorporates:
      //   Constant: '<S674>/Constant1'
      //   Sqrt: '<S674>/sqrt'
      rtb_Switch_fq = 6.378137E+6 / std::sqrt(rtb_Switch_ejf);

      // Product: '<S659>/dNorth' incorporates:
      //   Constant: '<S674>/Constant2'
      //   Product: '<S674>/Product3'
      //   Trigonometry: '<S674>/Trigonometric Function1'
      rtb_Switch_ejf = rtb_UnitConversion_idx_0 / rt_atan2d_snf(1.0,
        rtb_Switch_fq * 0.99330562000985867 / rtb_Switch_ejf);

      // Product: '<S659>/dEast' incorporates:
      //   Constant: '<S674>/Constant3'
      //   Product: '<S674>/Product4'
      //   Trigonometry: '<S674>/Trigonometric Function'
      //   Trigonometry: '<S674>/Trigonometric Function2'
      rtb_Abs1_n = 1.0 / rt_atan2d_snf(1.0, rtb_Switch_fq * std::cos(rtb_Abs1_n))
        * rtb_UnitConversion_idx_1;

      // Sum: '<S659>/Sum2' incorporates:
      //   Product: '<S659>/x*cos'
      //   Product: '<S659>/y*sin'
      rtb_Switch_fq = rtb_Abs1_n * 0.0 + rtb_Switch_ejf;

      // Sum: '<S659>/Sum3' incorporates:
      //   Product: '<S659>/x*sin'
      //   Product: '<S659>/y*cos'
      rtb_Abs1_n -= rtb_Switch_ejf * 0.0;

      // Sum: '<S655>/Sum' incorporates:
      //   DataTypeConversion: '<S489>/Data Type Conversion4'
      //   Inport: '<Root>/Navigation Filter Data'
      //   Inport: '<Root>/Telemetry Data'
      //   Selector: '<S489>/Selector'
      //   Sum: '<S489>/Sum'
      rtb_Sum_cd = (static_cast<real_T>(telem.flight_plan[0].z) -
                    nav.home_alt_wgs84_m) + nav.home_alt_wgs84_m;

      // Sum: '<S489>/Subtract' incorporates:
      //   Inport: '<Root>/Navigation Filter Data'
      rtb_Gain_dj_idx_0 = rtb_Switch_fq - nav.ned_pos_m[0];
      rtb_Subtract_idx_1 = rtb_Abs1_n - nav.ned_pos_m[1];

      // DataTypeConversion: '<S489>/Cast To Single1' incorporates:
      //   DataStoreWrite: '<S489>/Data Store Write1'
      //   Trigonometry: '<S489>/Atan2'
      rtDW.cur_target_heading_rad = static_cast<real32_T>(rt_atan2d_snf
        (rtb_Subtract_idx_1, rtb_Gain_dj_idx_0));

      // DataTypeConversion: '<S489>/Cast To Single' incorporates:
      //   DataStoreWrite: '<S489>/Data Store Write2'
      //   UnaryMinus: '<S655>/Ze2height'
      rtDW.cur_target_pos_m[0] = static_cast<real32_T>(rtb_Switch_fq);
      rtDW.cur_target_pos_m[1] = static_cast<real32_T>(rtb_Abs1_n);
      rtDW.cur_target_pos_m[2] = static_cast<real32_T>(-rtb_Sum_cd);

      // DataTypeConversion: '<S489>/Cast To Single2' incorporates:
      //   Inport: '<Root>/Navigation Filter Data'
      //   Sum: '<S489>/Subtract'
      //   UnaryMinus: '<S655>/Ze2height'
      rtb_ZeroGain_p = static_cast<real32_T>(-rtb_Sum_cd - nav.ned_pos_m[2]);

      // Sqrt: '<S489>/Sqrt' incorporates:
      //   DataTypeConversion: '<S489>/Cast To Single2'
      //   Product: '<S489>/MatrixMultiply'
      rtb_Integrator_d = std::sqrt(static_cast<real32_T>(rtb_Gain_dj_idx_0) *
        static_cast<real32_T>(rtb_Gain_dj_idx_0) + static_cast<real32_T>
        (rtb_Subtract_idx_1) * static_cast<real32_T>(rtb_Subtract_idx_1));

      // Abs: '<S489>/Abs' incorporates:
      //   Constant: '<S489>/Constant2'
      //   DataStoreWrite: '<S489>/Data Store Write4'
      //   Gain: '<S489>/Gain'
      //   Product: '<S489>/Divide1'
      //   Product: '<S489>/Divide2'
      rtDW.max_v_z_mps = std::abs(-rtb_ZeroGain_p / (rtb_Integrator_d / 5.0F));

      // Abs: '<S489>/Abs1' incorporates:
      //   Constant: '<S489>/Constant1'
      //   DataStoreWrite: '<S489>/Data Store Write5'
      //   Gain: '<S489>/Gain'
      //   Product: '<S489>/Divide'
      //   Product: '<S489>/Divide3'
      rtDW.max_v_hor_mps = std::abs(rtb_Integrator_d / (-rtb_ZeroGain_p / 2.0F));
    }

    rtPrevZCX.first_wp_Trig_ZCE = rtb_motor_armedANDmode_2;

    // End of Outputs for SubSystem: '<S13>/first_wp'

    // RelationalOperator: '<S487>/FixPt Relational Operator' incorporates:
    //   Inport: '<Root>/Telemetry Data'
    //   UnitDelay: '<S487>/Delay Input1'
    //
    //  Block description for '<S487>/Delay Input1':
    //
    //   Store in Global RAM
    rtb_motor_armedANDmode_2 = (telem.current_waypoint !=
      rtDW.DelayInput1_DSTATE);

    // Outputs for Triggered SubSystem: '<S13>/other_wp' incorporates:
    //   TriggerPort: '<S490>/Trigger'
    if (rtb_motor_armedANDmode_2 && (rtPrevZCX.other_wp_Trig_ZCE != 1)) {
      // DataStoreWrite: '<S490>/Data Store Write' incorporates:
      //   Inport: '<Root>/Telemetry Data'
      //   Selector: '<S490>/Selector'
      rtDW.autocontinue = telem.flight_plan[0].autocontinue;

      // Gain: '<S681>/Gain' incorporates:
      //   Inport: '<Root>/Navigation Filter Data'
      rtb_Abs1_n = 57.295779513082323 * nav.home_lat_rad;
      rtb_Switch_ejf = 57.295779513082323 * nav.home_lon_rad;

      // Outputs for Enabled SubSystem: '<S490>/Subsystem1' incorporates:
      //   EnablePort: '<S682>/Enable'

      // RelationalOperator: '<S679>/Compare' incorporates:
      //   Constant: '<S679>/Constant'
      //   Inport: '<Root>/Telemetry Data'
      //   Selector: '<S490>/Selector'
      if (telem.flight_plan[0].cmd == 16) {
        // Abs: '<S699>/Abs' incorporates:
        //   Abs: '<S721>/Abs'
        rtb_Switch_fq = std::abs(rtb_Abs1_n);

        // Switch: '<S699>/Switch' incorporates:
        //   Abs: '<S699>/Abs'
        //   Bias: '<S699>/Bias'
        //   Bias: '<S699>/Bias1'
        //   Constant: '<S699>/Constant2'
        //   Constant: '<S700>/Constant'
        //   Math: '<S699>/Math Function1'
        //   RelationalOperator: '<S700>/Compare'
        if (rtb_Switch_fq > 180.0) {
          rtb_Sum_cd = rt_modd_snf(rtb_Abs1_n + 180.0, 360.0) + -180.0;
        } else {
          rtb_Sum_cd = rtb_Abs1_n;
        }

        // End of Switch: '<S699>/Switch'

        // Abs: '<S696>/Abs1'
        rtb_Subtract_idx_1 = std::abs(rtb_Sum_cd);

        // Switch: '<S696>/Switch' incorporates:
        //   Bias: '<S696>/Bias'
        //   Bias: '<S696>/Bias1'
        //   Constant: '<S687>/Constant'
        //   Constant: '<S687>/Constant1'
        //   Constant: '<S698>/Constant'
        //   Gain: '<S696>/Gain'
        //   Product: '<S696>/Divide1'
        //   RelationalOperator: '<S698>/Compare'
        //   Switch: '<S687>/Switch1'
        if (rtb_Subtract_idx_1 > 90.0) {
          // Signum: '<S696>/Sign1'
          if (rtb_Sum_cd < 0.0) {
            rtb_Sum_cd = -1.0;
          } else if (rtb_Sum_cd > 0.0) {
            rtb_Sum_cd = 1.0;
          } else if (rtb_Sum_cd == 0.0) {
            rtb_Sum_cd = 0.0;
          } else {
            rtb_Sum_cd = (rtNaN);
          }

          // End of Signum: '<S696>/Sign1'
          rtb_Sum_cd *= -(rtb_Subtract_idx_1 + -90.0) + 90.0;
          i = 180;
        } else {
          i = 0;
        }

        // End of Switch: '<S696>/Switch'

        // Sum: '<S687>/Sum'
        rtb_Switch_f = static_cast<real_T>(i) + rtb_Switch_ejf;

        // Switch: '<S697>/Switch' incorporates:
        //   Abs: '<S697>/Abs'
        //   Bias: '<S697>/Bias'
        //   Bias: '<S697>/Bias1'
        //   Constant: '<S697>/Constant2'
        //   Constant: '<S701>/Constant'
        //   Math: '<S697>/Math Function1'
        //   RelationalOperator: '<S701>/Compare'
        if (std::abs(rtb_Switch_f) > 180.0) {
          rtb_Switch_f = rt_modd_snf(rtb_Switch_f + 180.0, 360.0) + -180.0;
        }

        // End of Switch: '<S697>/Switch'

        // DataTypeConversion: '<S682>/Data Type Conversion1' incorporates:
        //   DataTypeConversion: '<S682>/Data Type Conversion2'
        //   Gain: '<S682>/Gain3'
        rtb_Gain_dj_idx_0 = 1.0E-7 * static_cast<real_T>(telem.flight_plan[0].x);
        rtb_Gain_dj_idx_1 = 1.0E-7 * static_cast<real_T>(telem.flight_plan[0].y);

        // Sum: '<S684>/Sum1' incorporates:
        //   DataTypeConversion: '<S682>/Data Type Conversion1'
        rtb_UnitConversion_idx_0 = rtb_Gain_dj_idx_0 - rtb_Sum_cd;
        rtb_UnitConversion_idx_1 = rtb_Gain_dj_idx_1 - rtb_Switch_f;

        // Switch: '<S693>/Switch' incorporates:
        //   Abs: '<S693>/Abs'
        //   Bias: '<S693>/Bias'
        //   Bias: '<S693>/Bias1'
        //   Constant: '<S693>/Constant2'
        //   Constant: '<S694>/Constant'
        //   Math: '<S693>/Math Function1'
        //   RelationalOperator: '<S694>/Compare'
        if (std::abs(rtb_UnitConversion_idx_0) > 180.0) {
          rtb_Switch_f = rt_modd_snf(rtb_UnitConversion_idx_0 + 180.0, 360.0) +
            -180.0;
        } else {
          rtb_Switch_f = rtb_UnitConversion_idx_0;
        }

        // End of Switch: '<S693>/Switch'

        // Abs: '<S690>/Abs1'
        rtb_Subtract_idx_1 = std::abs(rtb_Switch_f);

        // Switch: '<S690>/Switch' incorporates:
        //   Bias: '<S690>/Bias'
        //   Bias: '<S690>/Bias1'
        //   Constant: '<S686>/Constant'
        //   Constant: '<S686>/Constant1'
        //   Constant: '<S692>/Constant'
        //   Gain: '<S690>/Gain'
        //   Product: '<S690>/Divide1'
        //   RelationalOperator: '<S692>/Compare'
        //   Switch: '<S686>/Switch1'
        if (rtb_Subtract_idx_1 > 90.0) {
          // Signum: '<S690>/Sign1'
          if (rtb_Switch_f < 0.0) {
            rtb_Switch_f = -1.0;
          } else if (rtb_Switch_f > 0.0) {
            rtb_Switch_f = 1.0;
          } else if (rtb_Switch_f == 0.0) {
            rtb_Switch_f = 0.0;
          } else {
            rtb_Switch_f = (rtNaN);
          }

          // End of Signum: '<S690>/Sign1'
          rtb_Switch_f *= -(rtb_Subtract_idx_1 + -90.0) + 90.0;
          i = 180;
        } else {
          i = 0;
        }

        // End of Switch: '<S690>/Switch'

        // Sum: '<S686>/Sum'
        rtb_Subtract_idx_1 = static_cast<real_T>(i) + rtb_UnitConversion_idx_1;

        // Switch: '<S691>/Switch' incorporates:
        //   Abs: '<S691>/Abs'
        //   Bias: '<S691>/Bias'
        //   Bias: '<S691>/Bias1'
        //   Constant: '<S691>/Constant2'
        //   Constant: '<S695>/Constant'
        //   Math: '<S691>/Math Function1'
        //   RelationalOperator: '<S695>/Compare'
        if (std::abs(rtb_Subtract_idx_1) > 180.0) {
          rtb_Subtract_idx_1 = rt_modd_snf(rtb_Subtract_idx_1 + 180.0, 360.0) +
            -180.0;
        }

        // End of Switch: '<S691>/Switch'

        // UnitConversion: '<S689>/Unit Conversion'
        // Unit Conversion - from: deg to: rad
        // Expression: output = (0.0174533*input) + (0)
        rtb_UnitConversion_idx_0 = 0.017453292519943295 * rtb_Switch_f;
        rtb_UnitConversion_idx_1 = 0.017453292519943295 * rtb_Subtract_idx_1;

        // UnitConversion: '<S704>/Unit Conversion'
        // Unit Conversion - from: deg to: rad
        // Expression: output = (0.0174533*input) + (0)
        rtb_Sum_cd *= 0.017453292519943295;

        // Trigonometry: '<S705>/Trigonometric Function1'
        rtb_Subtract_idx_1 = std::sin(rtb_Sum_cd);

        // Sum: '<S705>/Sum1' incorporates:
        //   Constant: '<S705>/Constant'
        //   Product: '<S705>/Product1'
        rtb_Subtract_idx_1 = 1.0 - 0.0066943799901413295 * rtb_Subtract_idx_1 *
          rtb_Subtract_idx_1;

        // Product: '<S703>/Product1' incorporates:
        //   Constant: '<S703>/Constant1'
        //   Sqrt: '<S703>/sqrt'
        rtb_Switch_f = 6.378137E+6 / std::sqrt(rtb_Subtract_idx_1);

        // Product: '<S688>/dNorth' incorporates:
        //   Constant: '<S703>/Constant2'
        //   Product: '<S703>/Product3'
        //   Trigonometry: '<S703>/Trigonometric Function1'
        rtb_Subtract_idx_1 = rtb_UnitConversion_idx_0 / rt_atan2d_snf(1.0,
          rtb_Switch_f * 0.99330562000985867 / rtb_Subtract_idx_1);

        // Product: '<S688>/dEast' incorporates:
        //   Constant: '<S703>/Constant3'
        //   Product: '<S703>/Product4'
        //   Trigonometry: '<S703>/Trigonometric Function'
        //   Trigonometry: '<S703>/Trigonometric Function2'
        rtb_Switch_f = 1.0 / rt_atan2d_snf(1.0, rtb_Switch_f * std::cos
          (rtb_Sum_cd)) * rtb_UnitConversion_idx_1;

        // Sum: '<S688>/Sum2' incorporates:
        //   Product: '<S688>/x*cos'
        //   Product: '<S688>/y*sin'
        rtb_Sum_cd = rtb_Switch_f * 0.0 + rtb_Subtract_idx_1;

        // Sum: '<S688>/Sum3' incorporates:
        //   Product: '<S688>/x*sin'
        //   Product: '<S688>/y*cos'
        rtb_Subtract_idx_1 = rtb_Switch_f - rtb_Subtract_idx_1 * 0.0;

        // Sum: '<S684>/Sum' incorporates:
        //   DataTypeConversion: '<S490>/Cast To Double1'
        //   DataTypeConversion: '<S682>/Cast To Double'
        //   Inport: '<Root>/Navigation Filter Data'
        //   Sum: '<S682>/Sum'
        //   Sum: '<S685>/Sum'
        rtb_Switch_f = (static_cast<real_T>(telem.flight_plan[0].z) -
                        nav.home_alt_wgs84_m) + nav.home_alt_wgs84_m;

        // DataTypeConversion: '<S682>/Cast To Single' incorporates:
        //   DataStoreWrite: '<S682>/Data Store Write'
        //   Sum: '<S684>/Sum'
        //   UnaryMinus: '<S684>/Ze2height'
        rtDW.cur_target_pos_m[0] = static_cast<real32_T>(rtb_Sum_cd);
        rtDW.cur_target_pos_m[1] = static_cast<real32_T>(rtb_Subtract_idx_1);
        rtDW.cur_target_pos_m[2] = static_cast<real32_T>(-rtb_Switch_f);

        // Switch: '<S721>/Switch' incorporates:
        //   Bias: '<S721>/Bias'
        //   Bias: '<S721>/Bias1'
        //   Constant: '<S721>/Constant2'
        //   Constant: '<S722>/Constant'
        //   Math: '<S721>/Math Function1'
        //   RelationalOperator: '<S722>/Compare'
        if (rtb_Switch_fq > 180.0) {
          rtb_Switch_fq = rt_modd_snf(rtb_Abs1_n + 180.0, 360.0) + -180.0;
        } else {
          rtb_Switch_fq = rtb_Abs1_n;
        }

        // End of Switch: '<S721>/Switch'

        // Abs: '<S718>/Abs1'
        rtb_Abs1_n = std::abs(rtb_Switch_fq);

        // Switch: '<S718>/Switch' incorporates:
        //   Bias: '<S718>/Bias'
        //   Bias: '<S718>/Bias1'
        //   Constant: '<S709>/Constant'
        //   Constant: '<S709>/Constant1'
        //   Constant: '<S720>/Constant'
        //   Gain: '<S718>/Gain'
        //   Product: '<S718>/Divide1'
        //   RelationalOperator: '<S720>/Compare'
        //   Switch: '<S709>/Switch1'
        if (rtb_Abs1_n > 90.0) {
          // Signum: '<S718>/Sign1'
          if (rtb_Switch_fq < 0.0) {
            rtb_Switch_fq = -1.0;
          } else if (rtb_Switch_fq > 0.0) {
            rtb_Switch_fq = 1.0;
          } else if (rtb_Switch_fq == 0.0) {
            rtb_Switch_fq = 0.0;
          } else {
            rtb_Switch_fq = (rtNaN);
          }

          // End of Signum: '<S718>/Sign1'
          rtb_Switch_fq *= -(rtb_Abs1_n + -90.0) + 90.0;
          i = 180;
        } else {
          i = 0;
        }

        // End of Switch: '<S718>/Switch'

        // Sum: '<S709>/Sum'
        rtb_Switch_ejf += static_cast<real_T>(i);

        // Switch: '<S719>/Switch' incorporates:
        //   Abs: '<S719>/Abs'
        //   Bias: '<S719>/Bias'
        //   Bias: '<S719>/Bias1'
        //   Constant: '<S719>/Constant2'
        //   Constant: '<S723>/Constant'
        //   Math: '<S719>/Math Function1'
        //   RelationalOperator: '<S723>/Compare'
        if (std::abs(rtb_Switch_ejf) > 180.0) {
          rtb_Switch_ejf = rt_modd_snf(rtb_Switch_ejf + 180.0, 360.0) + -180.0;
        }

        // End of Switch: '<S719>/Switch'

        // Sum: '<S685>/Sum1' incorporates:
        //   DataTypeConversion: '<S682>/Data Type Conversion2'
        rtb_UnitConversion_idx_0 = rtb_Gain_dj_idx_0 - rtb_Switch_fq;
        rtb_UnitConversion_idx_1 = rtb_Gain_dj_idx_1 - rtb_Switch_ejf;

        // Switch: '<S715>/Switch' incorporates:
        //   Abs: '<S715>/Abs'
        //   Bias: '<S715>/Bias'
        //   Bias: '<S715>/Bias1'
        //   Constant: '<S715>/Constant2'
        //   Constant: '<S716>/Constant'
        //   Math: '<S715>/Math Function1'
        //   RelationalOperator: '<S716>/Compare'
        if (std::abs(rtb_UnitConversion_idx_0) > 180.0) {
          rtb_Switch_ejf = rt_modd_snf(rtb_UnitConversion_idx_0 + 180.0, 360.0)
            + -180.0;
        } else {
          rtb_Switch_ejf = rtb_UnitConversion_idx_0;
        }

        // End of Switch: '<S715>/Switch'

        // Abs: '<S712>/Abs1'
        rtb_Abs1_n = std::abs(rtb_Switch_ejf);

        // Switch: '<S712>/Switch' incorporates:
        //   Bias: '<S712>/Bias'
        //   Bias: '<S712>/Bias1'
        //   Constant: '<S708>/Constant'
        //   Constant: '<S708>/Constant1'
        //   Constant: '<S714>/Constant'
        //   Gain: '<S712>/Gain'
        //   Product: '<S712>/Divide1'
        //   RelationalOperator: '<S714>/Compare'
        //   Switch: '<S708>/Switch1'
        if (rtb_Abs1_n > 90.0) {
          // Signum: '<S712>/Sign1'
          if (rtb_Switch_ejf < 0.0) {
            rtb_Switch_ejf = -1.0;
          } else if (rtb_Switch_ejf > 0.0) {
            rtb_Switch_ejf = 1.0;
          } else if (rtb_Switch_ejf == 0.0) {
            rtb_Switch_ejf = 0.0;
          } else {
            rtb_Switch_ejf = (rtNaN);
          }

          // End of Signum: '<S712>/Sign1'
          rtb_Switch_ejf *= -(rtb_Abs1_n + -90.0) + 90.0;
          i = 180;
        } else {
          i = 0;
        }

        // End of Switch: '<S712>/Switch'

        // Sum: '<S708>/Sum'
        rtb_Abs1_n = static_cast<real_T>(i) + rtb_UnitConversion_idx_1;

        // Switch: '<S713>/Switch' incorporates:
        //   Abs: '<S713>/Abs'
        //   Bias: '<S713>/Bias'
        //   Bias: '<S713>/Bias1'
        //   Constant: '<S713>/Constant2'
        //   Constant: '<S717>/Constant'
        //   Math: '<S713>/Math Function1'
        //   RelationalOperator: '<S717>/Compare'
        if (std::abs(rtb_Abs1_n) > 180.0) {
          rtb_Abs1_n = rt_modd_snf(rtb_Abs1_n + 180.0, 360.0) + -180.0;
        }

        // End of Switch: '<S713>/Switch'

        // UnitConversion: '<S711>/Unit Conversion'
        // Unit Conversion - from: deg to: rad
        // Expression: output = (0.0174533*input) + (0)
        rtb_UnitConversion_idx_0 = 0.017453292519943295 * rtb_Switch_ejf;
        rtb_UnitConversion_idx_1 = 0.017453292519943295 * rtb_Abs1_n;

        // UnitConversion: '<S726>/Unit Conversion'
        // Unit Conversion - from: deg to: rad
        // Expression: output = (0.0174533*input) + (0)
        rtb_Switch_fq *= 0.017453292519943295;

        // Trigonometry: '<S727>/Trigonometric Function1'
        rtb_Abs1_n = std::sin(rtb_Switch_fq);

        // Sum: '<S727>/Sum1' incorporates:
        //   Constant: '<S727>/Constant'
        //   Product: '<S727>/Product1'
        rtb_Abs1_n = 1.0 - 0.0066943799901413295 * rtb_Abs1_n * rtb_Abs1_n;

        // Product: '<S725>/Product1' incorporates:
        //   Constant: '<S725>/Constant1'
        //   Sqrt: '<S725>/sqrt'
        rtb_Switch_ejf = 6.378137E+6 / std::sqrt(rtb_Abs1_n);

        // Product: '<S710>/dNorth' incorporates:
        //   Constant: '<S725>/Constant2'
        //   Product: '<S725>/Product3'
        //   Trigonometry: '<S725>/Trigonometric Function1'
        rtb_Abs1_n = rtb_UnitConversion_idx_0 / rt_atan2d_snf(1.0,
          rtb_Switch_ejf * 0.99330562000985867 / rtb_Abs1_n);

        // Product: '<S710>/dEast' incorporates:
        //   Constant: '<S725>/Constant3'
        //   Product: '<S725>/Product4'
        //   Trigonometry: '<S725>/Trigonometric Function'
        //   Trigonometry: '<S725>/Trigonometric Function2'
        rtb_Switch_ejf = 1.0 / rt_atan2d_snf(1.0, rtb_Switch_ejf * std::cos
          (rtb_Switch_fq)) * rtb_UnitConversion_idx_1;

        // Sum: '<S682>/Subtract' incorporates:
        //   Product: '<S710>/x*cos'
        //   Product: '<S710>/x*sin'
        //   Product: '<S710>/y*cos'
        //   Product: '<S710>/y*sin'
        //   Sum: '<S710>/Sum2'
        //   Sum: '<S710>/Sum3'
        rtb_Gain_dj_idx_0 = rtb_Sum_cd - (rtb_Switch_ejf * 0.0 + rtb_Abs1_n);
        rtb_Subtract_idx_1 -= rtb_Switch_ejf - rtb_Abs1_n * 0.0;

        // DataTypeConversion: '<S682>/Cast To Single1' incorporates:
        //   DataStoreWrite: '<S682>/Data Store Write1'
        //   Trigonometry: '<S682>/Atan2'
        rtDW.cur_target_heading_rad = static_cast<real32_T>(rt_atan2d_snf
          (rtb_Subtract_idx_1, rtb_Gain_dj_idx_0));

        // DataTypeConversion: '<S682>/Cast To Single2' incorporates:
        //   Sum: '<S682>/Subtract'
        //   Sum: '<S684>/Sum'
        //   UnaryMinus: '<S684>/Ze2height'
        //   UnaryMinus: '<S685>/Ze2height'
        rtb_ZeroGain_p = static_cast<real32_T>(-rtb_Switch_f - (-rtb_Switch_f));

        // Sqrt: '<S682>/Sqrt' incorporates:
        //   DataTypeConversion: '<S682>/Cast To Single2'
        //   Product: '<S682>/MatrixMultiply'
        rtb_Integrator_d = std::sqrt(static_cast<real32_T>(rtb_Gain_dj_idx_0) *
          static_cast<real32_T>(rtb_Gain_dj_idx_0) + static_cast<real32_T>
          (rtb_Subtract_idx_1) * static_cast<real32_T>(rtb_Subtract_idx_1));

        // Abs: '<S682>/Abs1' incorporates:
        //   Constant: '<S682>/Constant3'
        //   DataStoreWrite: '<S682>/Data Store Write4'
        //   Gain: '<S682>/Gain'
        //   Product: '<S682>/Divide1'
        //   Product: '<S682>/Divide2'
        rtDW.max_v_z_mps = std::abs(-rtb_ZeroGain_p / (rtb_Integrator_d / 5.0F));

        // Abs: '<S682>/Abs' incorporates:
        //   Constant: '<S682>/Constant2'
        //   DataStoreWrite: '<S682>/Data Store Write5'
        //   Gain: '<S682>/Gain'
        //   Product: '<S682>/Divide'
        //   Product: '<S682>/Divide3'
        rtDW.max_v_hor_mps = std::abs(rtb_Integrator_d / (-rtb_ZeroGain_p / 2.0F));
      }

      // End of RelationalOperator: '<S679>/Compare'
      // End of Outputs for SubSystem: '<S490>/Subsystem1'

      // Outputs for Enabled SubSystem: '<S490>/Trigger RTL' incorporates:
      //   EnablePort: '<S683>/Enable'

      // RelationalOperator: '<S680>/Compare' incorporates:
      //   Constant: '<S680>/Constant'
      //   Inport: '<Root>/Telemetry Data'
      //   Selector: '<S490>/Selector'
      if (telem.flight_plan[0].cmd == 20) {
        // DataStoreWrite: '<S683>/Data Store Write' incorporates:
        //   Constant: '<S683>/RTL_mode'
        rtDW.cur_mode = 3;
      }

      // End of RelationalOperator: '<S680>/Compare'
      // End of Outputs for SubSystem: '<S490>/Trigger RTL'
    }

    rtPrevZCX.other_wp_Trig_ZCE = rtb_motor_armedANDmode_2;

    // End of Outputs for SubSystem: '<S13>/other_wp'

    // Logic: '<S491>/AND1' incorporates:
    //   Constant: '<S732>/Constant'
    //   DataStoreRead: '<S491>/Data Store Read2'
    //   Logic: '<S491>/NOT'
    //   Logic: '<S491>/NOT1'
    //   RelationalOperator: '<S732>/Compare'
    rtb_Compare_id = (rtb_Compare_id && (!rtb_DataStoreRead1_nl) &&
                      (rtDW.cur_mode != 3));

    // Outputs for Triggered SubSystem: '<S491>/Trigger Pos_hold' incorporates:
    //   TriggerPort: '<S733>/Trigger'
    if (rtb_Compare_id && (rtPrevZCX.TriggerPos_hold_Trig_ZCE != 1)) {
      // DataStoreWrite: '<S733>/Data Store Write' incorporates:
      //   Constant: '<S733>/pos_hold_mode'
      rtDW.cur_mode = 1;
    }

    rtPrevZCX.TriggerPos_hold_Trig_ZCE = rtb_Compare_id;

    // End of Outputs for SubSystem: '<S491>/Trigger Pos_hold'

    // Signum: '<S105>/SignPreIntegrator'
    if (rtb_Cos_k < 0.0F) {
      rtb_Integrator_d = -1.0F;
    } else if (rtb_Cos_k > 0.0F) {
      rtb_Integrator_d = 1.0F;
    } else if (rtb_Cos_k == 0.0F) {
      rtb_Integrator_d = 0.0F;
    } else {
      rtb_Integrator_d = (rtNaNF);
    }

    // End of Signum: '<S105>/SignPreIntegrator'

    // Switch: '<S105>/Switch' incorporates:
    //   Constant: '<S105>/Constant1'
    //   DataTypeConversion: '<S105>/DataTypeConv2'
    //   Logic: '<S105>/AND3'
    //   RelationalOperator: '<S105>/Equal1'
    //   RelationalOperator: '<S105>/NotEqual'
    if ((rtb_Switch_dv != rtb_yaw) && (0 == static_cast<int8_T>(rtb_Integrator_d)))
    {
      rtb_Cos_k = 0.0F;
    }

    // End of Switch: '<S105>/Switch'

    // Update for DiscreteIntegrator: '<S116>/Integrator'
    rtDW.Integrator_DSTATE += 0.01F * rtb_Cos_k;

    // Update for Delay: '<S109>/UD'
    rtDW.UD_DSTATE = rtb_roll;

    // Signum: '<S52>/SignPreIntegrator'
    if (rtb_throttle < 0.0F) {
      rtb_Integrator_d = -1.0F;
    } else if (rtb_throttle > 0.0F) {
      rtb_Integrator_d = 1.0F;
    } else if (rtb_throttle == 0.0F) {
      rtb_Integrator_d = 0.0F;
    } else {
      rtb_Integrator_d = (rtNaNF);
    }

    // End of Signum: '<S52>/SignPreIntegrator'

    // Switch: '<S52>/Switch' incorporates:
    //   Constant: '<S52>/Constant1'
    //   DataTypeConversion: '<S52>/DataTypeConv2'
    //   Logic: '<S52>/AND3'
    //   RelationalOperator: '<S52>/Equal1'
    //   RelationalOperator: '<S52>/NotEqual'
    if ((rtb_Product1_k != rtb_PProdOut_h) && (0 == static_cast<int8_T>
         (rtb_Integrator_d))) {
      rtb_throttle = 0.0F;
    }

    // End of Switch: '<S52>/Switch'

    // Update for DiscreteIntegrator: '<S63>/Integrator'
    rtDW.Integrator_DSTATE_l += 0.01F * rtb_throttle;

    // Update for Delay: '<S56>/UD'
    rtDW.UD_DSTATE_b = rtb_Subtract_p_idx_0;

    // Update for DiscreteIntegrator: '<S169>/Integrator'
    rtDW.Integrator_DSTATE_f += 0.01F * rtb_Tsamp_c;

    // Update for Delay: '<S162>/UD'
    rtDW.UD_DSTATE_l = rtb_Tsamp_m;

    // Update for UnitDelay: '<S20>/Delay Input1'
    //
    //  Block description for '<S20>/Delay Input1':
    //
    //   Store in Global RAM
    rtDW.DelayInput1_DSTATE_h = rtb_DataStoreRead1_c;

    // Update for UnitDelay: '<S21>/Delay Input1' incorporates:
    //   DataTypeConversion: '<Root>/Data Type Conversion6'
    //
    //  Block description for '<S21>/Delay Input1':
    //
    //   Store in Global RAM
    rtDW.DelayInput1_DSTATE_b = static_cast<int8_T>(rtb_PProdOut_d);

    // Update for UnitDelay: '<S487>/Delay Input1' incorporates:
    //   Inport: '<Root>/Telemetry Data'
    //
    //  Block description for '<S487>/Delay Input1':
    //
    //   Store in Global RAM
    rtDW.DelayInput1_DSTATE = telem.current_waypoint;
  }

  // Model initialize function
  void Autocode::initialize()
  {
    // Registration code

    // initialize non-finites
    rt_InitInfAndNaN(sizeof(real_T));

    // Start for DataStoreMemory: '<S13>/Data Store Memory2'
    rtDW.cur_target_pos_m[0] = 10.0F;
    rtDW.cur_target_pos_m[1] = 10.0F;
    rtDW.cur_target_pos_m[2] = 10.0F;

    // Start for DataStoreMemory: '<S13>/Data Store Memory5'
    rtDW.autocontinue = true;
    rtPrevZCX.TriggerLand_Trig_ZCE = POS_ZCSIG;
    rtPrevZCX.first_wp_Trig_ZCE = POS_ZCSIG;
    rtPrevZCX.other_wp_Trig_ZCE = POS_ZCSIG;
    rtPrevZCX.TriggerPos_hold_Trig_ZCE = POS_ZCSIG;
  }

  // Constructor
  Autocode::Autocode() :
    rtDW(),
    rtPrevZCX()
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
