//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// File: autocode.cpp
//
// Code generated for Simulink model 'baseline_super_part2'.
//
// Model version                  : 2.15
// Simulink Coder version         : 9.5 (R2021a) 14-Nov-2020
// C/C++ source code generated on : Fri May  6 09:02:09 2022
//
// Target selection: ert.tlc
// Embedded hardware selection: Intel->x86-64 (Linux 64)
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
  }                                    // '<S4>/Transpose'
};

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

real32_T rt_remf_snf(real32_T u0, real32_T u1)
{
  real32_T u1_0;
  real32_T y;
  if (rtIsNaNF(u0) || rtIsNaNF(u1) || rtIsInfF(u0)) {
    y = (rtNaNF);
  } else if (rtIsInfF(u1)) {
    y = u0;
  } else {
    if (u1 < 0.0F) {
      u1_0 = std::ceil(u1);
    } else {
      u1_0 = std::floor(u1);
    }

    if ((u1 != 0.0F) && (u1 != u1_0)) {
      u1_0 = std::abs(u0 / u1);
      if (!(std::abs(u1_0 - std::floor(u1_0 + 0.5F)) > FLT_EPSILON * u1_0)) {
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
  // Function for MATLAB Function: '<S636>/determine_prev_tar_pos'
  void Autocode::cosd(real32_T *x)
  {
    real32_T absx;
    real32_T b_x;
    int8_T n;
    if (rtIsInfF(*x) || rtIsNaNF(*x)) {
      *x = (rtNaNF);
    } else {
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
        n = 0;
      } else if (absx <= 135.0F) {
        if (b_x > 0.0F) {
          b_x = (b_x - 90.0F) * 0.0174532924F;
          n = 1;
        } else {
          b_x = (b_x + 90.0F) * 0.0174532924F;
          n = -1;
        }
      } else if (b_x > 0.0F) {
        b_x = (b_x - 180.0F) * 0.0174532924F;
        n = 2;
      } else {
        b_x = (b_x + 180.0F) * 0.0174532924F;
        n = -2;
      }

      switch (n) {
       case 0:
        *x = std::cos(b_x);
        break;

       case 1:
        *x = -std::sin(b_x);
        break;

       case -1:
        *x = std::sin(b_x);
        break;

       default:
        *x = -std::cos(b_x);
        break;
      }
    }
  }

  // Function for MATLAB Function: '<S636>/determine_prev_tar_pos'
  void Autocode::sind(real32_T *x)
  {
    real32_T absx;
    real32_T c_x;
    int8_T n;
    if (rtIsInfF(*x) || rtIsNaNF(*x)) {
      *x = (rtNaNF);
    } else {
      c_x = rt_remf_snf(*x, 360.0F);
      absx = std::abs(c_x);
      if (absx > 180.0F) {
        if (c_x > 0.0F) {
          c_x -= 360.0F;
        } else {
          c_x += 360.0F;
        }

        absx = std::abs(c_x);
      }

      if (absx <= 45.0F) {
        c_x *= 0.0174532924F;
        n = 0;
      } else if (absx <= 135.0F) {
        if (c_x > 0.0F) {
          c_x = (c_x - 90.0F) * 0.0174532924F;
          n = 1;
        } else {
          c_x = (c_x + 90.0F) * 0.0174532924F;
          n = -1;
        }
      } else if (c_x > 0.0F) {
        c_x = (c_x - 180.0F) * 0.0174532924F;
        n = 2;
      } else {
        c_x = (c_x + 180.0F) * 0.0174532924F;
        n = -2;
      }

      switch (n) {
       case 0:
        *x = std::sin(c_x);
        break;

       case 1:
        *x = std::cos(c_x);
        break;

       case -1:
        *x = -std::cos(c_x);
        break;

       default:
        *x = -std::sin(c_x);
        break;
      }
    }
  }

  // Function for MATLAB Function: '<S636>/determine_prev_tar_pos'
  void Autocode::lla_to_ECEF(const real32_T lla[3], real32_T ecef_pos[3])
  {
    real32_T b;
    real32_T c;
    real32_T c_lat;
    real32_T d;
    real32_T re;
    re = lla[0];
    sind(&re);
    re = 6.378137E+6F / std::sqrt(1.0F - re * re * 0.00669438F);
    c_lat = lla[0];
    cosd(&c_lat);
    b = lla[1];
    cosd(&b);
    c = lla[1];
    sind(&c);
    d = lla[0];
    sind(&d);
    c_lat *= re + lla[2];
    ecef_pos[0] = c_lat * b;
    ecef_pos[1] = c_lat * c;
    ecef_pos[2] = (0.993305624F * re + lla[2]) * d;
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

namespace bfs
{
  // Model step function
  void Autocode::Run(const SysData &sys, const SensorData &sensor, const NavData
                     &nav, const TelemData &telem, VmsData *ctrl)
  {
    std::array<real32_T, 3> diff;
    std::array<real32_T, 8> rtb_switch_motor_out;
    std::array<real32_T, 9> s_lat_0;
    std::array<real32_T, 3> tmp;
    std::array<real32_T, 3> tmp_0;
    int32_T idx;
    int32_T rtb_Reshape_h_tmp;
    real32_T absxk;
    real32_T c_lat;
    real32_T c_lon;
    real32_T rtb_DataTypeConversion6;
    real32_T rtb_PProdOut_l;
    real32_T rtb_Product1_i;
    real32_T rtb_Reshape_h_idx_0;
    real32_T rtb_Reshape_h_idx_1;
    real32_T rtb_Subtract_f_idx_1;
    real32_T rtb_pitch_angle_cmd_rad;
    real32_T rtb_roll_angle_cmd_rad;
    real32_T rtb_stab_pitch_rate_saturation;
    real32_T rtb_stab_roll_rate_saturation;
    real32_T rtb_throttle_cc;
    real32_T rtb_yaw_rate_cmd_radps_g;
    real32_T s_lat;
    real32_T s_lon;
    real32_T t;
    real32_T y;
    int8_T rtb_DataTypeConversion6_k;
    int8_T rtb_sub_mode;
    boolean_T rtb_AND;
    boolean_T rtb_AND_n;
    boolean_T rtb_Compare_lx;
    boolean_T rtb_Compare_n;
    boolean_T rtb_Switch_lt;
    UNUSED_PARAMETER(sys);

    // Logic: '<S19>/nav_init AND motor_enable' incorporates:
    //   Constant: '<S667>/Constant'
    //   DataTypeConversion: '<S646>/Data Type Conversion'
    //   Inport: '<Root>/Navigation Filter Data'
    //   Inport: '<Root>/Sensor Data'
    //   Polyval: '<S646>/throttle_en_norm'
    //   RelationalOperator: '<S667>/Compare'
    rtb_Switch_lt = ((0.00122026F * static_cast<real32_T>(sensor.inceptor.ch[6])
                      + -1.20988405F > 0.0F) && nav.nav_initialized);

    // Outputs for Enabled SubSystem: '<S647>/disarm motor' incorporates:
    //   EnablePort: '<S651>/Enable'

    // RelationalOperator: '<S650>/Compare' incorporates:
    //   Constant: '<S650>/Constant'
    //   Inport: '<Root>/Sensor Data'
    if (sensor.power_module.voltage_v <= 46.8F) {
      if (!rtDW.disarmmotor_MODE_k) {
        // InitializeConditions for UnitDelay: '<S651>/Unit Delay'
        rtDW.UnitDelay_DSTATE_m = 0.0;
        rtDW.disarmmotor_MODE_k = true;
      }

      // RelationalOperator: '<S652>/Compare' incorporates:
      //   Constant: '<S651>/Constant'
      //   Constant: '<S652>/Constant'
      //   Sum: '<S651>/Sum'
      //   UnitDelay: '<S651>/Unit Delay'
      rtDW.Compare_d = (rtDW.UnitDelay_DSTATE_m + 0.01 > 15.0);

      // Update for UnitDelay: '<S651>/Unit Delay' incorporates:
      //   Constant: '<S651>/Constant'
      //   Sum: '<S651>/Sum'
      rtDW.UnitDelay_DSTATE_m += 0.01;
    } else {
      rtDW.disarmmotor_MODE_k = false;
    }

    // End of RelationalOperator: '<S650>/Compare'
    // End of Outputs for SubSystem: '<S647>/disarm motor'

    // Polyval: '<S645>/mode_norm' incorporates:
    //   DataTypeConversion: '<S645>/Data Type Conversion1'
    //   Inport: '<Root>/Sensor Data'
    rtb_roll_angle_cmd_rad = 0.00122026F * static_cast<real32_T>
      (sensor.inceptor.ch[4]) + -0.209884077F;

    // DataTypeConversion: '<S645>/Data Type Conversion6'
    if (std::abs(rtb_roll_angle_cmd_rad) >= 0.5F) {
      c_lat = std::floor(rtb_roll_angle_cmd_rad + 0.5F);
      rtb_DataTypeConversion6_k = static_cast<int8_T>(c_lat);
    } else {
      c_lat = rtb_roll_angle_cmd_rad * 0.0F;
      rtb_DataTypeConversion6_k = static_cast<int8_T>(rtb_roll_angle_cmd_rad *
        0.0F);
    }

    // Logic: '<S647>/AND' incorporates:
    //   Constant: '<S649>/Constant'
    //   DataTypeConversion: '<S645>/Data Type Conversion6'
    //   Logic: '<S647>/AND1'
    //   RelationalOperator: '<S649>/Compare'
    rtb_AND_n = (rtDW.Compare_d && ((static_cast<int8_T>(c_lat) != 5) &&
      rtb_Switch_lt));

    // Logic: '<S648>/AND' incorporates:
    //   Constant: '<S653>/Constant'
    //   Constant: '<S654>/Constant'
    //   Constant: '<S655>/Constant'
    //   DataTypeConversion: '<S645>/Data Type Conversion6'
    //   Inport: '<Root>/Sensor Data'
    //   Logic: '<S648>/NOR'
    //   Logic: '<S648>/NOT'
    //   RelationalOperator: '<S653>/Compare'
    //   RelationalOperator: '<S654>/Compare'
    //   RelationalOperator: '<S655>/Compare'
    rtb_AND = (sensor.inceptor.failsafe && ((static_cast<int8_T>(c_lat) == 3) &&
                (static_cast<int8_T>(c_lat) == 4) && (static_cast<int8_T>(c_lat)
      == 5) && rtb_Switch_lt));

    // Switch: '<S19>/Switch1' incorporates:
    //   Constant: '<S19>/land_mode'
    //   Switch: '<S19>/Switch2'
    if (rtb_AND_n) {
      rtb_DataTypeConversion6_k = 5;
    } else if (rtb_AND) {
      // Switch: '<S19>/Switch2' incorporates:
      //   Constant: '<S19>/rtl_mode'
      rtb_DataTypeConversion6_k = 3;
    }

    // End of Switch: '<S19>/Switch1'

    // Outputs for Enabled SubSystem: '<S644>/waypoint submodes' incorporates:
    //   EnablePort: '<S663>/Enable'

    // RelationalOperator: '<S661>/Compare' incorporates:
    //   Constant: '<S661>/Constant'
    if (rtb_DataTypeConversion6_k == 2) {
      // MATLAB Function: '<S663>/determine_target_pos' incorporates:
      //   Inport: '<Root>/Navigation Filter Data'
      diff[0] = static_cast<real32_T>(nav.home_lat_rad * 57.295779513082323);
      diff[1] = static_cast<real32_T>(nav.home_lon_rad * 57.295779513082323);
      diff[2] = nav.home_alt_wgs84_m;
      c_lat = diff[0];
      cosd(&c_lat);
      s_lat = diff[0];
      sind(&s_lat);
      c_lon = diff[1];
      cosd(&c_lon);
      s_lon = diff[1];
      sind(&s_lon);

      // MATLAB Function: '<S663>/determine_wp_submode' incorporates:
      //   Constant: '<S663>/Constant'
      //   Inport: '<Root>/Navigation Filter Data'
      //   Inport: '<Root>/Telemetry Data'
      //   MATLAB Function: '<S663>/determine_target_pos'
      //   Selector: '<S663>/Selector'
      rtDW.sub_mode = 2;
      switch (telem.flight_plan[telem.current_waypoint].cmd) {
       case 20:
        rtDW.sub_mode = 3;
        break;

       case 16:
        // MATLAB Function: '<S663>/determine_target_pos' incorporates:
        //   Inport: '<Root>/Navigation Filter Data'
        tmp[0] = static_cast<real32_T>(telem.flight_plan[telem.current_waypoint]
          .x) * 1.0E-7F;
        tmp[1] = static_cast<real32_T>(telem.flight_plan[telem.current_waypoint]
          .y) * 1.0E-7F;
        tmp[2] = telem.flight_plan[telem.current_waypoint].z +
          nav.home_alt_wgs84_m;
        lla_to_ECEF(&tmp[0], &tmp_0[0]);
        lla_to_ECEF(&diff[0], &tmp[0]);
        s_lat_0[0] = -s_lat * c_lon;
        s_lat_0[1] = -s_lon;
        s_lat_0[2] = -c_lat * c_lon;
        s_lat_0[3] = -s_lat * s_lon;
        s_lat_0[4] = c_lon;
        s_lat_0[5] = -c_lat * s_lon;
        s_lat_0[6] = c_lat;
        s_lat_0[7] = 0.0F;
        s_lat_0[8] = -s_lat;
        rtb_DataTypeConversion6 = tmp_0[0] - tmp[0];
        s_lat = tmp_0[1] - tmp[1];
        rtb_Reshape_h_idx_1 = tmp_0[2] - tmp[2];
        y = 0.0F;
        c_lat = 1.29246971E-26F;
        for (idx = 0; idx < 3; idx++) {
          absxk = std::abs(((s_lat_0[idx + 3] * s_lat + s_lat_0[idx] *
                             rtb_DataTypeConversion6) + s_lat_0[idx + 6] *
                            rtb_Reshape_h_idx_1) - nav.ned_pos_m[idx]);
          if (absxk > c_lat) {
            t = c_lat / absxk;
            y = y * t * t + 1.0F;
            c_lat = absxk;
          } else {
            t = absxk / c_lat;
            y += t * t;
          }
        }

        y = c_lat * std::sqrt(y);
        if ((y <= 1.5F) && (!telem.flight_plan[telem.current_waypoint].
                            autocontinue)) {
          rtDW.sub_mode = 1;
        }
        break;
      }

      // End of MATLAB Function: '<S663>/determine_wp_submode'
    }

    // End of RelationalOperator: '<S661>/Compare'
    // End of Outputs for SubSystem: '<S644>/waypoint submodes'

    // Outputs for Enabled SubSystem: '<S644>/rtl submodes' incorporates:
    //   EnablePort: '<S662>/Enable'

    // RelationalOperator: '<S660>/Compare' incorporates:
    //   Constant: '<S660>/Constant'
    if (rtb_DataTypeConversion6_k == 3) {
      // MATLAB Function: '<S662>/determine_rtl_submode' incorporates:
      //   Constant: '<S662>/Constant'
      //   Inport: '<Root>/Navigation Filter Data'
      rtDW.sub_mode_m = 3;
      c_lat = 1.29246971E-26F;
      absxk = std::abs(0.0F - nav.ned_pos_m[0]);
      if (absxk > 1.29246971E-26F) {
        y = 1.0F;
        c_lat = absxk;
      } else {
        t = absxk / 1.29246971E-26F;
        y = t * t;
      }

      absxk = std::abs(0.0F - nav.ned_pos_m[1]);
      if (absxk > c_lat) {
        t = c_lat / absxk;
        y = y * t * t + 1.0F;
        c_lat = absxk;
      } else {
        t = absxk / c_lat;
        y += t * t;
      }

      absxk = std::abs(0.0F - nav.ned_pos_m[2]);
      if (absxk > c_lat) {
        t = c_lat / absxk;
        y = y * t * t + 1.0F;
        c_lat = absxk;
      } else {
        t = absxk / c_lat;
        y += t * t;
      }

      y = c_lat * std::sqrt(y);
      if (y <= 1.5F) {
        rtDW.sub_mode_m = 4;
      }

      // End of MATLAB Function: '<S662>/determine_rtl_submode'
    }

    // End of RelationalOperator: '<S660>/Compare'
    // End of Outputs for SubSystem: '<S644>/rtl submodes'

    // Switch: '<S19>/Switch3' incorporates:
    //   Logic: '<S19>/OR'
    if ((!rtb_AND_n) && (!rtb_AND)) {
      // MultiPortSwitch: '<S644>/Multiport Switch'
      switch (rtb_DataTypeConversion6_k) {
       case 2:
        rtb_DataTypeConversion6_k = rtDW.sub_mode;
        break;

       case 3:
        rtb_DataTypeConversion6_k = rtDW.sub_mode_m;
        break;
      }

      // End of MultiPortSwitch: '<S644>/Multiport Switch'
    }

    // End of Switch: '<S19>/Switch3'

    // Outputs for Enabled SubSystem: '<S19>/auto_disarm' incorporates:
    //   EnablePort: '<S642>/Enable'

    // Logic: '<S19>/motor_armed AND mode_4' incorporates:
    //   Abs: '<S642>/Abs'
    //   Constant: '<S643>/Constant'
    //   Constant: '<S656>/Constant'
    //   Constant: '<S657>/Constant'
    //   Gain: '<S642>/Gain'
    //   Inport: '<Root>/Navigation Filter Data'
    //   Logic: '<S642>/AND'
    //   RelationalOperator: '<S643>/Compare'
    //   RelationalOperator: '<S656>/Compare'
    //   RelationalOperator: '<S657>/Compare'
    if (rtb_Switch_lt && (rtb_DataTypeConversion6_k == 4)) {
      rtDW.auto_disarm_MODE = true;

      // Outputs for Enabled SubSystem: '<S642>/disarm motor' incorporates:
      //   EnablePort: '<S658>/Enable'
      if ((-nav.ned_pos_m[2] <= 10.0F) && (std::abs(nav.ned_vel_mps[2]) <= 0.3F))
      {
        if (!rtDW.disarmmotor_MODE) {
          // InitializeConditions for UnitDelay: '<S658>/Unit Delay'
          rtDW.UnitDelay_DSTATE = 0.0;
          rtDW.disarmmotor_MODE = true;
        }

        // RelationalOperator: '<S659>/Compare' incorporates:
        //   Constant: '<S658>/Constant'
        //   Constant: '<S659>/Constant'
        //   Sum: '<S658>/Sum'
        //   UnitDelay: '<S658>/Unit Delay'
        rtDW.Compare = (rtDW.UnitDelay_DSTATE + 0.01 > 10.0);

        // Update for UnitDelay: '<S658>/Unit Delay' incorporates:
        //   Constant: '<S658>/Constant'
        //   Sum: '<S658>/Sum'
        rtDW.UnitDelay_DSTATE += 0.01;
      } else {
        rtDW.disarmmotor_MODE = false;
      }

      // End of Outputs for SubSystem: '<S642>/disarm motor'
    } else if (rtDW.auto_disarm_MODE) {
      // Disable for Enabled SubSystem: '<S642>/disarm motor'
      rtDW.disarmmotor_MODE = false;

      // End of Disable for SubSystem: '<S642>/disarm motor'
      rtDW.auto_disarm_MODE = false;
    }

    // End of Logic: '<S19>/motor_armed AND mode_4'
    // End of Outputs for SubSystem: '<S19>/auto_disarm'

    // Switch: '<S19>/Switch'
    rtb_Switch_lt = ((!rtDW.Compare) && rtb_Switch_lt);

    // Outputs for Enabled SubSystem: '<Root>/WAYPOINT CONTROLLER' incorporates:
    //   EnablePort: '<S11>/Enable'

    // Logic: '<Root>/motor_armed AND mode_2' incorporates:
    //   Constant: '<S18>/Constant'
    //   RelationalOperator: '<S18>/Compare'
    //   RelationalOperator: '<S633>/FixPt Relational Operator'
    //   UnitDelay: '<S633>/Delay Input1'
    //
    //  Block description for '<S633>/Delay Input1':
    //
    //   Store in Global RAM
    if (rtb_Switch_lt && (rtb_DataTypeConversion6_k == 2)) {
      // RelationalOperator: '<S634>/Compare' incorporates:
      //   Constant: '<S634>/Constant'
      //   Inport: '<Root>/Telemetry Data'
      //   RelationalOperator: '<S632>/FixPt Relational Operator'
      //   UnitDelay: '<S632>/Delay Input1'
      //
      //  Block description for '<S632>/Delay Input1':
      //
      //   Store in Global RAM
      rtb_Compare_lx = (telem.current_waypoint != rtDW.DelayInput1_DSTATE);

      // Outputs for Enabled SubSystem: '<S11>/determine target' incorporates:
      //   EnablePort: '<S467>/Enable'
      if (static_cast<int32_T>(rtb_Compare_lx) > static_cast<int32_T>
          (rtDW.DelayInput1_DSTATE_n)) {
        // RelationalOperator: '<S635>/Compare' incorporates:
        //   Constant: '<S467>/Constant'
        //   Constant: '<S635>/Constant'
        //   Inport: '<Root>/Telemetry Data'
        //   Sum: '<S467>/Sum'
        rtb_Compare_n = (static_cast<real_T>(telem.current_waypoint) - 1.0 >=
                         0.0);

        // Outputs for Enabled SubSystem: '<S467>/calc_prev_target_pos' incorporates:
        //   EnablePort: '<S636>/Enable'
        if (rtb_Compare_n) {
          // MATLAB Function: '<S636>/determine_prev_tar_pos' incorporates:
          //   Constant: '<S467>/Constant'
          //   Inport: '<Root>/Navigation Filter Data'
          //   Inport: '<Root>/Telemetry Data'
          //   Selector: '<S636>/Selector1'
          //   Sum: '<S467>/Sum'
          diff[0] = static_cast<real32_T>(nav.home_lat_rad * 57.295779513082323);
          diff[1] = static_cast<real32_T>(nav.home_lon_rad * 57.295779513082323);
          diff[2] = nav.home_alt_wgs84_m;
          c_lat = diff[0];
          cosd(&c_lat);
          s_lat = diff[0];
          sind(&s_lat);
          c_lon = diff[1];
          cosd(&c_lon);
          s_lon = diff[1];
          sind(&s_lon);
          idx = static_cast<int32_T>(static_cast<real_T>(telem.current_waypoint)
            - 1.0);
          tmp[0] = static_cast<real32_T>(telem.flight_plan[idx].x) * 1.0E-7F;
          tmp[1] = static_cast<real32_T>(telem.flight_plan[idx].y) * 1.0E-7F;
          tmp[2] = telem.flight_plan[idx].z + nav.home_alt_wgs84_m;
          lla_to_ECEF(&tmp[0], &tmp_0[0]);
          lla_to_ECEF(&diff[0], &tmp[0]);
          s_lat_0[0] = -s_lat * c_lon;
          s_lat_0[1] = -s_lon;
          s_lat_0[2] = -c_lat * c_lon;
          s_lat_0[3] = -s_lat * s_lon;
          s_lat_0[4] = c_lon;
          s_lat_0[5] = -c_lat * s_lon;
          s_lat_0[6] = c_lat;
          s_lat_0[7] = 0.0F;
          s_lat_0[8] = -s_lat;
          rtb_DataTypeConversion6 = tmp_0[0] - tmp[0];
          s_lat = tmp_0[1] - tmp[1];
          rtb_Reshape_h_idx_1 = tmp_0[2] - tmp[2];
          for (idx = 0; idx < 3; idx++) {
            rtDW.pref_target_pos[idx] = 0.0F;
            rtDW.pref_target_pos[idx] += s_lat_0[idx] * rtb_DataTypeConversion6;
            rtDW.pref_target_pos[idx] += s_lat_0[idx + 3] * s_lat;
            rtDW.pref_target_pos[idx] += s_lat_0[idx + 6] * rtb_Reshape_h_idx_1;
          }

          // End of MATLAB Function: '<S636>/determine_prev_tar_pos'
        }

        // End of Outputs for SubSystem: '<S467>/calc_prev_target_pos'

        // MATLAB Function: '<S467>/determine_current_tar_pos' incorporates:
        //   Inport: '<Root>/Navigation Filter Data'
        //   Inport: '<Root>/Telemetry Data'
        //   Selector: '<S467>/Selector'
        diff[0] = static_cast<real32_T>(nav.home_lat_rad * 57.295779513082323);
        diff[1] = static_cast<real32_T>(nav.home_lon_rad * 57.295779513082323);
        diff[2] = nav.home_alt_wgs84_m;
        c_lat = diff[0];
        cosd(&c_lat);
        s_lat = diff[0];
        sind(&s_lat);
        c_lon = diff[1];
        cosd(&c_lon);
        s_lon = diff[1];
        sind(&s_lon);
        tmp[0] = static_cast<real32_T>(telem.flight_plan[telem.current_waypoint]
          .x) * 1.0E-7F;
        tmp[1] = static_cast<real32_T>(telem.flight_plan[telem.current_waypoint]
          .y) * 1.0E-7F;
        tmp[2] = telem.flight_plan[telem.current_waypoint].z +
          nav.home_alt_wgs84_m;
        lla_to_ECEF(&tmp[0], &tmp_0[0]);
        lla_to_ECEF(&diff[0], &tmp[0]);
        s_lat_0[0] = -s_lat * c_lon;
        s_lat_0[1] = -s_lon;
        s_lat_0[2] = -c_lat * c_lon;
        s_lat_0[3] = -s_lat * s_lon;
        s_lat_0[4] = c_lon;
        s_lat_0[5] = -c_lat * s_lon;
        s_lat_0[6] = c_lat;
        s_lat_0[7] = 0.0F;
        s_lat_0[8] = -s_lat;
        rtb_DataTypeConversion6 = tmp_0[0] - tmp[0];
        s_lat = tmp_0[1] - tmp[1];
        rtb_Reshape_h_idx_1 = tmp_0[2] - tmp[2];
        for (idx = 0; idx < 3; idx++) {
          rtDW.cur_target_pos_m_c[idx] = 0.0F;
          rtDW.cur_target_pos_m_c[idx] += s_lat_0[idx] * rtb_DataTypeConversion6;
          rtDW.cur_target_pos_m_c[idx] += s_lat_0[idx + 3] * s_lat;
          rtDW.cur_target_pos_m_c[idx] += s_lat_0[idx + 6] * rtb_Reshape_h_idx_1;

          // Switch: '<S467>/Switch'
          if (rtb_Compare_n) {
            c_lat = rtDW.pref_target_pos[idx];
          } else {
            c_lat = nav.ned_pos_m[idx];
          }

          // End of Switch: '<S467>/Switch'

          // MATLAB Function: '<S467>/determine_target'
          diff[idx] = rtDW.cur_target_pos_m_c[idx] - c_lat;
        }

        // End of MATLAB Function: '<S467>/determine_current_tar_pos'

        // MATLAB Function: '<S467>/determine_target' incorporates:
        //   Constant: '<S467>/Constant2'
        //   Constant: '<S467>/Constant3'
        c_lat = 1.29246971E-26F;
        absxk = std::abs(diff[0]);
        if (absxk > 1.29246971E-26F) {
          s_lat = 1.0F;
          c_lat = absxk;
        } else {
          t = absxk / 1.29246971E-26F;
          s_lat = t * t;
        }

        absxk = std::abs(diff[1]);
        if (absxk > c_lat) {
          t = c_lat / absxk;
          s_lat = s_lat * t * t + 1.0F;
          c_lat = absxk;
        } else {
          t = absxk / c_lat;
          s_lat += t * t;
        }

        s_lat = c_lat * std::sqrt(s_lat);
        rtDW.cur_target_heading_rad = rt_atan2f_snf(diff[1], diff[0]);
        rtDW.max_v_z_mps = std::abs(-diff[2] * 5.0F / s_lat);
        rtDW.max_v_hor_mps = std::abs(s_lat * 2.0F / -diff[2]);
      }

      // End of Outputs for SubSystem: '<S11>/determine target'

      // Outputs for Enabled SubSystem: '<S11>/WP_NAV' incorporates:
      //   EnablePort: '<S465>/Enable'

      // Trigonometry: '<S526>/Sin' incorporates:
      //   Inport: '<Root>/Navigation Filter Data'
      //   RelationalOperator: '<S633>/FixPt Relational Operator'
      //   UnitDelay: '<S633>/Delay Input1'
      //
      //  Block description for '<S633>/Delay Input1':
      //
      //   Store in Global RAM
      rtb_DataTypeConversion6 = std::sin(nav.heading_rad);

      // Reshape: '<S526>/Reshape' incorporates:
      //   Gain: '<S526>/Gain'
      //   Inport: '<Root>/Navigation Filter Data'
      //   Reshape: '<S187>/Reshape'
      //   Trigonometry: '<S526>/Cos'
      rtb_Reshape_h_idx_0 = std::cos(nav.heading_rad);
      rtb_Reshape_h_idx_1 = -rtb_DataTypeConversion6;
      absxk = rtb_DataTypeConversion6;

      // Sum: '<S473>/Subtract' incorporates:
      //   Inport: '<Root>/Navigation Filter Data'
      //   SignalConversion generated from: '<S469>/Bus Selector2'
      rtb_DataTypeConversion6 = rtDW.cur_target_pos_m_c[0] - nav.ned_pos_m[0];
      rtb_Subtract_f_idx_1 = rtDW.cur_target_pos_m_c[1] - nav.ned_pos_m[1];

      // MinMax: '<S472>/Min'
      if (rtDW.max_v_hor_mps < 5.0F) {
        c_lat = rtDW.max_v_hor_mps;
      } else {
        c_lat = 5.0F;
      }

      if (!(c_lat < 3.0F)) {
        c_lat = 3.0F;
      }

      // End of MinMax: '<S472>/Min'

      // Sqrt: '<S473>/Sqrt' incorporates:
      //   Math: '<S473>/Transpose'
      //   Product: '<S473>/MatrixMultiply'
      s_lat = std::sqrt(rtb_DataTypeConversion6 * rtb_DataTypeConversion6 +
                        rtb_Subtract_f_idx_1 * rtb_Subtract_f_idx_1);

      // Saturate: '<S472>/Saturation'
      if (s_lat > 20.0F) {
        s_lat = 20.0F;
      } else if (s_lat < 0.0F) {
        s_lat = 0.0F;
      }

      // End of Saturate: '<S472>/Saturation'

      // Product: '<S513>/PProd Out' incorporates:
      //   Constant: '<S472>/Constant3'
      s_lat *= 3.0F;

      // Switch: '<S516>/Switch2' incorporates:
      //   RelationalOperator: '<S516>/LowerRelop1'
      //   Switch: '<S516>/Switch'
      if (!(s_lat > c_lat)) {
        c_lat = s_lat;
      }

      // End of Switch: '<S516>/Switch2'

      // Trigonometry: '<S473>/Atan2'
      rtb_DataTypeConversion6 = rt_atan2f_snf(rtb_Subtract_f_idx_1,
        rtb_DataTypeConversion6);

      // SignalConversion generated from: '<S476>/Product' incorporates:
      //   Product: '<S475>/Product'
      //   Product: '<S475>/Product1'
      //   Trigonometry: '<S475>/Cos'
      //   Trigonometry: '<S475>/Sin'
      s_lat = c_lat * std::cos(rtb_DataTypeConversion6);
      c_lat *= std::sin(rtb_DataTypeConversion6);

      // Product: '<S476>/Product' incorporates:
      //   Reshape: '<S187>/Reshape'
      rtDW.vb_xy[0] = 0.0F;
      rtDW.vb_xy[0] += rtb_Reshape_h_idx_0 * s_lat;
      rtDW.vb_xy[0] += absxk * c_lat;
      rtDW.vb_xy[1] = 0.0F;
      rtDW.vb_xy[1] += rtb_Reshape_h_idx_1 * s_lat;
      rtDW.vb_xy[1] += rtb_Reshape_h_idx_0 * c_lat;

      // Product: '<S565>/PProd Out' incorporates:
      //   Inport: '<Root>/Navigation Filter Data'
      //   Sum: '<S527>/Sum3'
      c_lat = rtDW.cur_target_pos_m_c[2] - nav.ned_pos_m[2];

      // Switch: '<S568>/Switch2' incorporates:
      //   Constant: '<S527>/Constant1'
      //   MinMax: '<S527>/Min'
      //   RelationalOperator: '<S568>/LowerRelop1'
      if (c_lat > 1.0F) {
        // Switch: '<S568>/Switch2'
        rtDW.Switch2 = 1.0F;
      } else {
        if (rtDW.max_v_z_mps < 2.0F) {
          // MinMax: '<S527>/Min'
          y = rtDW.max_v_z_mps;
        } else {
          // MinMax: '<S527>/Min'
          y = 2.0F;
        }

        // Switch: '<S568>/Switch' incorporates:
        //   Gain: '<S527>/Gain'
        //   RelationalOperator: '<S568>/UpperRelop'
        if (c_lat < -y) {
          // Switch: '<S568>/Switch2'
          rtDW.Switch2 = -y;
        } else {
          // Switch: '<S568>/Switch2'
          rtDW.Switch2 = c_lat;
        }

        // End of Switch: '<S568>/Switch'
      }

      // End of Switch: '<S568>/Switch2'

      // Sum: '<S580>/Subtract' incorporates:
      //   Inport: '<Root>/Navigation Filter Data'
      c_lat = rtDW.cur_target_heading_rad - nav.heading_rad;

      // Switch: '<S580>/Switch' incorporates:
      //   Abs: '<S580>/Abs'
      //   Constant: '<S580>/Constant'
      //   Constant: '<S631>/Constant'
      //   Product: '<S580>/Product'
      //   RelationalOperator: '<S631>/Compare'
      //   Sum: '<S580>/Subtract1'
      if (std::abs(c_lat) > 3.14159274F) {
        // Signum: '<S580>/Sign'
        if (c_lat < 0.0F) {
          y = -1.0F;
        } else if (c_lat > 0.0F) {
          y = 1.0F;
        } else if (c_lat == 0.0F) {
          y = 0.0F;
        } else {
          y = (rtNaNF);
        }

        // End of Signum: '<S580>/Sign'
        c_lat -= y * 6.28318548F;
      }

      // End of Switch: '<S580>/Switch'

      // Product: '<S611>/IProd Out' incorporates:
      //   Constant: '<S578>/I_heading'
      rtb_DataTypeConversion6 = c_lat * 0.01F;

      // Sum: '<S623>/Sum' incorporates:
      //   DiscreteIntegrator: '<S614>/Integrator'
      //   Product: '<S619>/PProd Out'
      rtb_Reshape_h_idx_1 = c_lat + rtDW.Integrator_DSTATE_bm;

      // Saturate: '<S578>/Saturation'
      if (rtb_Reshape_h_idx_1 > 0.524F) {
        // Saturate: '<S578>/Saturation'
        rtDW.Saturation = 0.524F;
      } else if (rtb_Reshape_h_idx_1 < -0.524F) {
        // Saturate: '<S578>/Saturation'
        rtDW.Saturation = -0.524F;
      } else {
        // Saturate: '<S578>/Saturation'
        rtDW.Saturation = rtb_Reshape_h_idx_1;
      }

      // End of Saturate: '<S578>/Saturation'

      // DeadZone: '<S607>/DeadZone'
      if (rtb_Reshape_h_idx_1 >= (rtMinusInfF)) {
        s_lat = 0.0F;
      } else {
        s_lat = (rtNaNF);
      }

      // End of DeadZone: '<S607>/DeadZone'

      // Signum: '<S605>/SignPreIntegrator'
      if (rtb_DataTypeConversion6 < 0.0F) {
        c_lat = -1.0F;
      } else if (rtb_DataTypeConversion6 > 0.0F) {
        c_lat = 1.0F;
      } else if (rtb_DataTypeConversion6 == 0.0F) {
        c_lat = 0.0F;
      } else {
        c_lat = (rtNaNF);
      }

      // End of Signum: '<S605>/SignPreIntegrator'

      // Switch: '<S605>/Switch' incorporates:
      //   Constant: '<S605>/Constant1'
      //   DataTypeConversion: '<S605>/DataTypeConv2'
      //   Gain: '<S605>/ZeroGain'
      //   Logic: '<S605>/AND3'
      //   RelationalOperator: '<S605>/Equal1'
      //   RelationalOperator: '<S605>/NotEqual'
      if ((0.0F * rtb_Reshape_h_idx_1 != s_lat) && (0 == static_cast<int8_T>
           (c_lat))) {
        rtb_DataTypeConversion6 = 0.0F;
      }

      // End of Switch: '<S605>/Switch'

      // Update for DiscreteIntegrator: '<S614>/Integrator'
      rtDW.Integrator_DSTATE_bm += 0.01F * rtb_DataTypeConversion6;

      // End of Outputs for SubSystem: '<S11>/WP_NAV'

      // MATLAB Function: '<S468>/check_wp_reached'
      c_lat = 1.29246971E-26F;

      // SignalConversion generated from: '<S11>/dbg'
      rtDW.cur_target_pos_m[0] = rtDW.cur_target_pos_m_c[0];

      // MATLAB Function: '<S468>/check_wp_reached' incorporates:
      //   Inport: '<Root>/Navigation Filter Data'
      absxk = std::abs(rtDW.cur_target_pos_m_c[0] - nav.ned_pos_m[0]);
      if (absxk > 1.29246971E-26F) {
        y = 1.0F;
        c_lat = absxk;
      } else {
        t = absxk / 1.29246971E-26F;
        y = t * t;
      }

      // SignalConversion generated from: '<S11>/dbg'
      rtDW.cur_target_pos_m[1] = rtDW.cur_target_pos_m_c[1];

      // MATLAB Function: '<S468>/check_wp_reached' incorporates:
      //   Inport: '<Root>/Navigation Filter Data'
      absxk = std::abs(rtDW.cur_target_pos_m_c[1] - nav.ned_pos_m[1]);
      if (absxk > c_lat) {
        t = c_lat / absxk;
        y = y * t * t + 1.0F;
        c_lat = absxk;
      } else {
        t = absxk / c_lat;
        y += t * t;
      }

      // SignalConversion generated from: '<S11>/dbg'
      rtDW.cur_target_pos_m[2] = rtDW.cur_target_pos_m_c[2];

      // MATLAB Function: '<S468>/check_wp_reached' incorporates:
      //   Constant: '<S468>/Constant'
      //   Inport: '<Root>/Navigation Filter Data'
      //   Inport: '<Root>/Telemetry Data'
      //   Selector: '<S468>/Selector'
      absxk = std::abs(rtDW.cur_target_pos_m_c[2] - nav.ned_pos_m[2]);
      if (absxk > c_lat) {
        t = c_lat / absxk;
        y = y * t * t + 1.0F;
        c_lat = absxk;
      } else {
        t = absxk / c_lat;
        y += t * t;
      }

      rtDW.reached = ((c_lat * std::sqrt(y) <= 1.5F) &&
                      telem.flight_plan[telem.current_waypoint].autocontinue);

      // SignalConversion generated from: '<S11>/dbg' incorporates:
      //   Inport: '<Root>/Telemetry Data'
      rtDW.current_waypoint = telem.current_waypoint;

      // Update for UnitDelay: '<S632>/Delay Input1' incorporates:
      //   Inport: '<Root>/Telemetry Data'
      //
      //  Block description for '<S632>/Delay Input1':
      //
      //   Store in Global RAM
      rtDW.DelayInput1_DSTATE = telem.current_waypoint;

      // Update for UnitDelay: '<S633>/Delay Input1'
      //
      //  Block description for '<S633>/Delay Input1':
      //
      //   Store in Global RAM
      rtDW.DelayInput1_DSTATE_n = rtb_Compare_lx;
    }

    // End of Logic: '<Root>/motor_armed AND mode_2'
    // End of Outputs for SubSystem: '<Root>/WAYPOINT CONTROLLER'

    // Polyval: '<S12>/pitch_norm' incorporates:
    //   DataTypeConversion: '<S12>/Data Type Conversion4'
    //   Inport: '<Root>/Sensor Data'
    s_lat = 0.00122026F * static_cast<real32_T>(sensor.inceptor.ch[2]) +
      -1.20988405F;

    // Polyval: '<S12>/roll_norm' incorporates:
    //   DataTypeConversion: '<S12>/Data Type Conversion3'
    //   Inport: '<Root>/Sensor Data'
    c_lon = 0.00122026F * static_cast<real32_T>(sensor.inceptor.ch[1]) +
      -1.20988405F;

    // Polyval: '<S12>/yaw_norm' incorporates:
    //   DataTypeConversion: '<S12>/Data Type Conversion2'
    //   Inport: '<Root>/Sensor Data'
    s_lon = 0.00122026F * static_cast<real32_T>(sensor.inceptor.ch[3]) +
      -1.20988405F;

    // Logic: '<Root>/motor_armed AND mode_5' incorporates:
    //   Constant: '<S17>/Constant'
    //   RelationalOperator: '<S17>/Compare'
    rtb_Compare_lx = (rtb_Switch_lt && (rtb_DataTypeConversion6_k == 1));

    // Polyval: '<S12>/throttle_norm' incorporates:
    //   DataTypeConversion: '<S12>/Data Type Conversion5'
    //   Inport: '<Root>/Sensor Data'
    rtb_Subtract_f_idx_1 = 0.00061013F * static_cast<real32_T>
      (sensor.inceptor.ch[0]) + -0.104942039F;

    // Outputs for Enabled SubSystem: '<Root>/Pos_Hold_input_conversion' incorporates:
    //   EnablePort: '<S6>/Enable'
    if (rtb_Compare_lx) {
      // Gain: '<S6>/Gain1'
      rtDW.vb_x_cmd_mps_d = 5.0F * s_lat;

      // Product: '<S349>/v_z_cmd (-1 to 1)' incorporates:
      //   Constant: '<S349>/Double'
      //   Constant: '<S349>/Normalize at Zero'
      //   Sum: '<S349>/Sum'
      c_lat = (rtb_Subtract_f_idx_1 - 0.5F) * 2.0F;

      // Gain: '<S349>/Gain' incorporates:
      //   Constant: '<S349>/Constant1'
      //   Constant: '<S350>/Constant'
      //   Constant: '<S351>/Constant'
      //   Product: '<S349>/Product'
      //   Product: '<S349>/Product1'
      //   RelationalOperator: '<S350>/Compare'
      //   RelationalOperator: '<S351>/Compare'
      //   Sum: '<S349>/Sum1'
      rtDW.Gain = -(static_cast<real32_T>(c_lat >= 0.0F) * c_lat * 2.0F +
                    static_cast<real32_T>(c_lat < 0.0F) * c_lat);

      // Gain: '<S6>/Gain2'
      rtDW.vb_y_cmd_mps_f = 5.0F * c_lon;

      // Gain: '<S6>/Gain3'
      rtDW.yaw_rate_cmd_radps_p = 0.524F * s_lon;
    }

    // End of Outputs for SubSystem: '<Root>/Pos_Hold_input_conversion'

    // Outputs for Enabled SubSystem: '<Root>/RTL CONTROLLER' incorporates:
    //   EnablePort: '<S8>/Enable'

    // Logic: '<Root>/motor_armed AND mode_3' incorporates:
    //   Constant: '<S15>/Constant'
    //   RelationalOperator: '<S15>/Compare'
    if (rtb_Switch_lt && (rtb_DataTypeConversion6_k == 3)) {
      // Sqrt: '<S357>/Sqrt' incorporates:
      //   Inport: '<Root>/Navigation Filter Data'
      //   Product: '<S357>/MatrixMultiply'
      //   SignalConversion generated from: '<S8>/Bus Selector2'
      //   Sum: '<S357>/Subtract'
      c_lat = std::sqrt((0.0F - nav.ned_pos_m[0]) * (0.0F - nav.ned_pos_m[0]) +
                        (0.0F - nav.ned_pos_m[1]) * (0.0F - nav.ned_pos_m[1]));

      // Saturate: '<S356>/Saturation'
      if (c_lat > 20.0F) {
        y = 20.0F;
      } else if (c_lat < 0.0F) {
        y = 0.0F;
      } else {
        y = c_lat;
      }

      // End of Saturate: '<S356>/Saturation'

      // Product: '<S398>/PProd Out' incorporates:
      //   Constant: '<S356>/Constant3'
      rtb_PProdOut_l = y * 0.5F;

      // Switch: '<S8>/Switch1' incorporates:
      //   Constant: '<S355>/Constant'
      //   Inport: '<Root>/Navigation Filter Data'
      //   MinMax: '<S8>/Min'
      //   RelationalOperator: '<S355>/Compare'
      if (c_lat <= 10.0F) {
        c_lat = nav.ned_pos_m[2];
      } else if ((-100.0F < nav.ned_pos_m[2]) || rtIsNaNF(nav.ned_pos_m[2])) {
        // MinMax: '<S8>/Min'
        c_lat = -100.0F;
      } else {
        // MinMax: '<S8>/Min' incorporates:
        //   Inport: '<Root>/Navigation Filter Data'
        c_lat = nav.ned_pos_m[2];
      }

      // End of Switch: '<S8>/Switch1'

      // Sum: '<S358>/Sum3' incorporates:
      //   Inport: '<Root>/Navigation Filter Data'
      c_lat -= nav.ned_pos_m[2];

      // Switch: '<S8>/Switch' incorporates:
      //   Abs: '<S358>/Abs'
      //   Constant: '<S412>/Constant'
      //   RelationalOperator: '<S412>/Compare'
      if (std::abs(c_lat) <= 1.5F) {
        // Trigonometry: '<S357>/Atan2' incorporates:
        //   Inport: '<Root>/Navigation Filter Data'
        //   SignalConversion generated from: '<S8>/Bus Selector2'
        //   Sum: '<S357>/Subtract'
        rtb_DataTypeConversion6 = rt_atan2f_snf(0.0F - nav.ned_pos_m[1], 0.0F -
          nav.ned_pos_m[0]);

        // Switch: '<S401>/Switch2' incorporates:
        //   Constant: '<S356>/Constant1'
        //   RelationalOperator: '<S401>/LowerRelop1'
        if (rtb_PProdOut_l > 5.0F) {
          rtb_PProdOut_l = 5.0F;
        }

        // End of Switch: '<S401>/Switch2'

        // Product: '<S360>/Product1' incorporates:
        //   Trigonometry: '<S360>/Sin'
        rtb_Product1_i = rtb_PProdOut_l * std::sin(rtb_DataTypeConversion6);

        // Product: '<S360>/Product' incorporates:
        //   Trigonometry: '<S360>/Cos'
        rtb_DataTypeConversion6 = rtb_PProdOut_l * std::cos
          (rtb_DataTypeConversion6);

        // Trigonometry: '<S411>/Sin' incorporates:
        //   Inport: '<Root>/Navigation Filter Data'
        rtb_PProdOut_l = std::sin(nav.heading_rad);

        // Trigonometry: '<S411>/Cos' incorporates:
        //   Inport: '<Root>/Navigation Filter Data'
        rtb_Reshape_h_idx_1 = std::cos(nav.heading_rad);

        // Switch: '<S8>/Switch' incorporates:
        //   Gain: '<S411>/Gain'
        //   Product: '<S361>/Product'
        //   Reshape: '<S411>/Reshape'
        //   SignalConversion generated from: '<S361>/Product'
        rtDW.Switch[0] = 0.0F;
        rtDW.Switch[0] += rtb_Reshape_h_idx_1 * rtb_DataTypeConversion6;
        rtDW.Switch[0] += rtb_PProdOut_l * rtb_Product1_i;
        rtDW.Switch[1] = 0.0F;
        rtDW.Switch[1] += -rtb_PProdOut_l * rtb_DataTypeConversion6;
        rtDW.Switch[1] += rtb_Reshape_h_idx_1 * rtb_Product1_i;
      } else {
        // Switch: '<S8>/Switch'
        rtDW.Switch[0] = 0.0F;
        rtDW.Switch[1] = 0.0F;
      }

      // End of Switch: '<S8>/Switch'

      // Switch: '<S453>/Switch2' incorporates:
      //   Constant: '<S358>/Constant1'
      //   RelationalOperator: '<S453>/LowerRelop1'
      //   RelationalOperator: '<S453>/UpperRelop'
      //   Switch: '<S453>/Switch'
      if (c_lat > 1.0F) {
        // Switch: '<S453>/Switch2'
        rtDW.Switch2_h = 1.0F;
      } else if (c_lat < -2.0F) {
        // Switch: '<S453>/Switch' incorporates:
        //   Switch: '<S453>/Switch2'
        rtDW.Switch2_h = -2.0F;
      } else {
        // Switch: '<S453>/Switch2' incorporates:
        //   Switch: '<S453>/Switch'
        rtDW.Switch2_h = c_lat;
      }

      // End of Switch: '<S453>/Switch2'

      // SignalConversion generated from: '<S8>/Constant3' incorporates:
      //   Constant: '<S8>/Constant3'
      rtDW.yaw_rate_cmd_radps_c = 0.0F;
    }

    // End of Logic: '<Root>/motor_armed AND mode_3'
    // End of Outputs for SubSystem: '<Root>/RTL CONTROLLER'

    // Outputs for Enabled SubSystem: '<Root>/LAND CONTROLLER' incorporates:
    //   EnablePort: '<S3>/Enable'

    // Outputs for Enabled SubSystem: '<Root>/Pos_Hold_input_conversion2' incorporates:
    //   EnablePort: '<S7>/Enable'

    // Logic: '<Root>/motor_armed AND mode_4' incorporates:
    //   Constant: '<S13>/Constant'
    //   RelationalOperator: '<S13>/Compare'
    if (rtb_Switch_lt && (rtb_DataTypeConversion6_k == 4)) {
      // SignalConversion generated from: '<S3>/land_cmd' incorporates:
      //   Gain: '<S7>/Gain1'
      rtDW.vb_x_cmd_mps_o = 5.0F * s_lat;

      // Switch: '<S183>/Switch' incorporates:
      //   Constant: '<S184>/Constant'
      //   Inport: '<Root>/Navigation Filter Data'
      //   RelationalOperator: '<S184>/Compare'
      if (nav.ned_pos_m[2] <= 10.0F) {
        // Switch: '<S183>/Switch' incorporates:
        //   Constant: '<S183>/Constant1'
        rtDW.Switch_h = 0.3F;
      } else {
        // Switch: '<S183>/Switch' incorporates:
        //   Constant: '<S183>/Constant'
        rtDW.Switch_h = 1.0F;
      }

      // End of Switch: '<S183>/Switch'

      // SignalConversion generated from: '<S3>/land_cmd' incorporates:
      //   Gain: '<S7>/Gain2'
      rtDW.vb_y_cmd_mps_l = 5.0F * c_lon;

      // SignalConversion generated from: '<S3>/land_cmd' incorporates:
      //   Gain: '<S7>/Gain3'
      rtDW.yaw_rate_cmd_radps_c53 = 0.524F * s_lon;
    }

    // End of Logic: '<Root>/motor_armed AND mode_4'
    // End of Outputs for SubSystem: '<Root>/Pos_Hold_input_conversion2'
    // End of Outputs for SubSystem: '<Root>/LAND CONTROLLER'

    // Switch generated from: '<Root>/Switch1' incorporates:
    //   Logic: '<Root>/NOT1'
    if (!rtb_Compare_lx) {
      // MultiPortSwitch generated from: '<Root>/Multiport Switch'
      switch (rtb_DataTypeConversion6_k) {
       case 2:
        rtb_PProdOut_l = rtDW.Switch2;
        rtb_Reshape_h_idx_1 = rtDW.vb_xy[0];
        rtb_Reshape_h_idx_0 = rtDW.vb_xy[1];
        rtb_Product1_i = rtDW.Saturation;
        break;

       case 3:
        rtb_PProdOut_l = rtDW.Switch2_h;
        rtb_Reshape_h_idx_1 = rtDW.Switch[0];
        rtb_Reshape_h_idx_0 = rtDW.Switch[1];
        rtb_Product1_i = rtDW.yaw_rate_cmd_radps_c;
        break;

       default:
        rtb_PProdOut_l = rtDW.Switch_h;
        rtb_Reshape_h_idx_1 = rtDW.vb_x_cmd_mps_o;
        rtb_Reshape_h_idx_0 = rtDW.vb_y_cmd_mps_l;
        rtb_Product1_i = rtDW.yaw_rate_cmd_radps_c53;
        break;
      }

      // End of MultiPortSwitch generated from: '<Root>/Multiport Switch'
    } else {
      rtb_PProdOut_l = rtDW.Gain;
      rtb_Reshape_h_idx_1 = rtDW.vb_x_cmd_mps_d;
      rtb_Reshape_h_idx_0 = rtDW.vb_y_cmd_mps_f;
      rtb_Product1_i = rtDW.yaw_rate_cmd_radps_p;
    }

    // End of Switch generated from: '<Root>/Switch1'

    // Outputs for Enabled SubSystem: '<Root>/POS_HOLD CONTROLLER' incorporates:
    //   EnablePort: '<S5>/Enable'

    // Logic: '<Root>/motor_armed AND mode_1' incorporates:
    //   Constant: '<S14>/Constant'
    //   RelationalOperator: '<S14>/Compare'
    if (rtb_Switch_lt && (rtb_DataTypeConversion6_k > 0)) {
      // Trigonometry: '<S187>/Cos' incorporates:
      //   Inport: '<Root>/Navigation Filter Data'
      c_lat = std::cos(nav.heading_rad);

      // Trigonometry: '<S187>/Sin' incorporates:
      //   Inport: '<Root>/Navigation Filter Data'
      rtb_DataTypeConversion6 = std::sin(nav.heading_rad);

      // Product: '<S185>/Product' incorporates:
      //   Gain: '<S187>/Gain'
      //   Inport: '<Root>/Navigation Filter Data'
      //   Reshape: '<S187>/Reshape'
      t = -rtb_DataTypeConversion6 * nav.ned_vel_mps[0] + c_lat *
        nav.ned_vel_mps[1];

      // Sum: '<S188>/Sum' incorporates:
      //   Inport: '<Root>/Navigation Filter Data'
      //   Product: '<S185>/Product'
      //   Reshape: '<S187>/Reshape'
      c_lat = rtb_Reshape_h_idx_1 - (c_lat * nav.ned_vel_mps[0] +
        rtb_DataTypeConversion6 * nav.ned_vel_mps[1]);

      // SampleTimeMath: '<S221>/Tsamp' incorporates:
      //   Constant: '<S188>/Constant2'
      //   Product: '<S218>/DProd Out'
      //
      //  About '<S221>/Tsamp':
      //   y = u * K where K = 1 / ( w * Ts )
      rtb_DataTypeConversion6 = c_lat * 0.1F * 100.0F;

      // Sum: '<S235>/Sum' incorporates:
      //   Constant: '<S188>/Constant'
      //   Delay: '<S219>/UD'
      //   DiscreteIntegrator: '<S226>/Integrator'
      //   Product: '<S231>/PProd Out'
      //   Sum: '<S219>/Diff'
      rtb_Reshape_h_idx_1 = (c_lat * 0.5F + rtDW.Integrator_DSTATE_c) +
        (rtb_DataTypeConversion6 - rtDW.UD_DSTATE_k);

      // Saturate: '<S188>/Saturation'
      if (rtb_Reshape_h_idx_1 > 0.523F) {
        // Gain: '<S188>/Gain'
        rtDW.Gain_a = -0.523F;
      } else if (rtb_Reshape_h_idx_1 < -0.523F) {
        // Gain: '<S188>/Gain'
        rtDW.Gain_a = 0.523F;
      } else {
        // Gain: '<S188>/Gain'
        rtDW.Gain_a = -rtb_Reshape_h_idx_1;
      }

      // End of Saturate: '<S188>/Saturation'

      // Gain: '<S215>/ZeroGain'
      absxk = 0.0F * rtb_Reshape_h_idx_1;

      // DeadZone: '<S217>/DeadZone'
      if (rtb_Reshape_h_idx_1 >= (rtMinusInfF)) {
        rtb_Reshape_h_idx_1 = 0.0F;
      }

      // End of DeadZone: '<S217>/DeadZone'

      // Product: '<S223>/IProd Out' incorporates:
      //   Constant: '<S188>/Constant1'
      c_lat *= 0.01F;

      // Signum: '<S215>/SignPreIntegrator'
      if (c_lat < 0.0F) {
        y = -1.0F;
      } else if (c_lat > 0.0F) {
        y = 1.0F;
      } else if (c_lat == 0.0F) {
        y = 0.0F;
      } else {
        y = (rtNaNF);
      }

      // End of Signum: '<S215>/SignPreIntegrator'

      // Switch: '<S215>/Switch' incorporates:
      //   Constant: '<S215>/Constant1'
      //   DataTypeConversion: '<S215>/DataTypeConv2'
      //   Logic: '<S215>/AND3'
      //   RelationalOperator: '<S215>/Equal1'
      //   RelationalOperator: '<S215>/NotEqual'
      if ((absxk != rtb_Reshape_h_idx_1) && (0 == static_cast<int8_T>(y))) {
        rtb_Reshape_h_idx_1 = 0.0F;
      } else {
        rtb_Reshape_h_idx_1 = c_lat;
      }

      // End of Switch: '<S215>/Switch'

      // Sum: '<S189>/Sum'
      c_lat = rtb_Reshape_h_idx_0 - t;

      // SampleTimeMath: '<S274>/Tsamp' incorporates:
      //   Constant: '<S189>/Constant2'
      //   Product: '<S271>/DProd Out'
      //
      //  About '<S274>/Tsamp':
      //   y = u * K where K = 1 / ( w * Ts )
      absxk = c_lat * 0.1F * 100.0F;

      // Sum: '<S288>/Sum' incorporates:
      //   Constant: '<S189>/Constant'
      //   Delay: '<S272>/UD'
      //   DiscreteIntegrator: '<S279>/Integrator'
      //   Product: '<S284>/PProd Out'
      //   Sum: '<S272>/Diff'
      rtb_Reshape_h_idx_0 = (c_lat * 0.5F + rtDW.Integrator_DSTATE_n) + (absxk -
        rtDW.UD_DSTATE_a);

      // Product: '<S276>/IProd Out' incorporates:
      //   Constant: '<S189>/Constant1'
      c_lat *= 0.01F;

      // DeadZone: '<S270>/DeadZone'
      if (rtb_Reshape_h_idx_0 >= (rtMinusInfF)) {
        t = 0.0F;
      } else {
        t = (rtNaNF);
      }

      // End of DeadZone: '<S270>/DeadZone'

      // Signum: '<S268>/SignPreIntegrator'
      if (c_lat < 0.0F) {
        y = -1.0F;
      } else if (c_lat > 0.0F) {
        y = 1.0F;
      } else if (c_lat == 0.0F) {
        y = 0.0F;
      } else {
        y = (rtNaNF);
      }

      // End of Signum: '<S268>/SignPreIntegrator'

      // Switch: '<S268>/Switch' incorporates:
      //   Constant: '<S268>/Constant1'
      //   DataTypeConversion: '<S268>/DataTypeConv2'
      //   Gain: '<S268>/ZeroGain'
      //   Logic: '<S268>/AND3'
      //   RelationalOperator: '<S268>/Equal1'
      //   RelationalOperator: '<S268>/NotEqual'
      if ((0.0F * rtb_Reshape_h_idx_0 != t) && (0 == static_cast<int8_T>(y))) {
        rtb_roll_angle_cmd_rad = 0.0F;
      } else {
        rtb_roll_angle_cmd_rad = c_lat;
      }

      // End of Switch: '<S268>/Switch'

      // Saturate: '<S189>/Saturation'
      if (rtb_Reshape_h_idx_0 > 0.523F) {
        // Saturate: '<S189>/Saturation'
        rtDW.Saturation_n = 0.523F;
      } else if (rtb_Reshape_h_idx_0 < -0.523F) {
        // Saturate: '<S189>/Saturation'
        rtDW.Saturation_n = -0.523F;
      } else {
        // Saturate: '<S189>/Saturation'
        rtDW.Saturation_n = rtb_Reshape_h_idx_0;
      }

      // End of Saturate: '<S189>/Saturation'

      // SignalConversion generated from: '<S5>/Command out'
      rtDW.yaw_rate_cmd_radps_c5 = rtb_Product1_i;

      // Sum: '<S186>/Sum' incorporates:
      //   Inport: '<Root>/Navigation Filter Data'
      c_lat = rtb_PProdOut_l - nav.ned_vel_mps[2];

      // SampleTimeMath: '<S327>/Tsamp' incorporates:
      //   Constant: '<S186>/D_vz'
      //   Product: '<S324>/DProd Out'
      //
      //  About '<S327>/Tsamp':
      //   y = u * K where K = 1 / ( w * Ts )
      rtb_PProdOut_l = c_lat * 0.005F * 100.0F;

      // Sum: '<S341>/Sum' incorporates:
      //   Constant: '<S186>/P_vz'
      //   Delay: '<S325>/UD'
      //   DiscreteIntegrator: '<S332>/Integrator'
      //   Product: '<S337>/PProd Out'
      //   Sum: '<S325>/Diff'
      t = (c_lat * 0.09F + rtDW.Integrator_DSTATE_a) + (rtb_PProdOut_l -
        rtDW.UD_DSTATE_h);

      // Saturate: '<S186>/Saturation' incorporates:
      //   Constant: '<S186>/Constant2'
      //   Gain: '<S186>/Gain'
      //   Sum: '<S186>/Sum1'
      if (-t + 0.6724F > 1.0F) {
        // Saturate: '<S186>/Saturation'
        rtDW.Saturation_k = 1.0F;
      } else if (-t + 0.6724F < 0.0F) {
        // Saturate: '<S186>/Saturation'
        rtDW.Saturation_k = 0.0F;
      } else {
        // Saturate: '<S186>/Saturation'
        rtDW.Saturation_k = -t + 0.6724F;
      }

      // End of Saturate: '<S186>/Saturation'

      // Gain: '<S321>/ZeroGain'
      rtb_Product1_i = 0.0F * t;

      // DeadZone: '<S323>/DeadZone'
      if (t >= (rtMinusInfF)) {
        t = 0.0F;
      }

      // End of DeadZone: '<S323>/DeadZone'

      // Product: '<S329>/IProd Out' incorporates:
      //   Constant: '<S186>/I_vz'
      c_lat *= 0.05F;

      // Update for DiscreteIntegrator: '<S226>/Integrator'
      rtDW.Integrator_DSTATE_c += 0.01F * rtb_Reshape_h_idx_1;

      // Update for Delay: '<S219>/UD'
      rtDW.UD_DSTATE_k = rtb_DataTypeConversion6;

      // Update for DiscreteIntegrator: '<S279>/Integrator'
      rtDW.Integrator_DSTATE_n += 0.01F * rtb_roll_angle_cmd_rad;

      // Update for Delay: '<S272>/UD'
      rtDW.UD_DSTATE_a = absxk;

      // Signum: '<S321>/SignPreIntegrator'
      if (c_lat < 0.0F) {
        y = -1.0F;
      } else if (c_lat > 0.0F) {
        y = 1.0F;
      } else if (c_lat == 0.0F) {
        y = 0.0F;
      } else {
        y = (rtNaNF);
      }

      // End of Signum: '<S321>/SignPreIntegrator'

      // Switch: '<S321>/Switch' incorporates:
      //   Constant: '<S321>/Constant1'
      //   DataTypeConversion: '<S321>/DataTypeConv2'
      //   Logic: '<S321>/AND3'
      //   RelationalOperator: '<S321>/Equal1'
      //   RelationalOperator: '<S321>/NotEqual'
      if ((rtb_Product1_i != t) && (0 == static_cast<int8_T>(y))) {
        c_lat = 0.0F;
      }

      // End of Switch: '<S321>/Switch'

      // Update for DiscreteIntegrator: '<S332>/Integrator'
      rtDW.Integrator_DSTATE_a += 0.01F * c_lat;

      // Update for Delay: '<S325>/UD'
      rtDW.UD_DSTATE_h = rtb_PProdOut_l;
    }

    // End of Logic: '<Root>/motor_armed AND mode_1'
    // End of Outputs for SubSystem: '<Root>/POS_HOLD CONTROLLER'

    // Logic: '<Root>/motor_armed AND mode_0' incorporates:
    //   Constant: '<S16>/Constant'
    //   RelationalOperator: '<S16>/Compare'
    rtb_Compare_lx = (rtb_Switch_lt && (rtb_DataTypeConversion6_k <= 0));

    // Outputs for Enabled SubSystem: '<Root>/Stab_input_conversion' incorporates:
    //   EnablePort: '<S9>/Enable'
    if (rtb_Compare_lx) {
      // Gain: '<S9>/Gain'
      rtDW.throttle_cc = rtb_Subtract_f_idx_1;

      // Gain: '<S9>/Gain1'
      rtDW.pitch_angle_cmd_rad = -0.523F * s_lat;

      // Gain: '<S9>/Gain2'
      rtDW.roll_angle_cmd_rad = 0.52F * c_lon;

      // Gain: '<S9>/Gain3'
      rtDW.yaw_rate_cmd_radps = 0.524F * s_lon;
    }

    // End of Outputs for SubSystem: '<Root>/Stab_input_conversion'

    // Logic: '<Root>/NOT'
    rtb_Compare_lx = !rtb_Compare_lx;

    // Switch generated from: '<Root>/Switch'
    if (rtb_Compare_lx) {
      rtb_throttle_cc = rtDW.Saturation_k;
      rtb_roll_angle_cmd_rad = rtDW.Saturation_n;
    } else {
      rtb_throttle_cc = rtDW.throttle_cc;
      rtb_roll_angle_cmd_rad = rtDW.roll_angle_cmd_rad;
    }

    // Sum: '<S22>/stab_roll_angle_error_calc' incorporates:
    //   Inport: '<Root>/Navigation Filter Data'
    s_lat = rtb_roll_angle_cmd_rad - nav.roll_rad;

    // SampleTimeMath: '<S108>/Tsamp' incorporates:
    //   Constant: '<S22>/Constant2'
    //   Product: '<S105>/DProd Out'
    //
    //  About '<S108>/Tsamp':
    //   y = u * K where K = 1 / ( w * Ts )
    c_lon = s_lat * 0.02F * 100.0F;

    // Sum: '<S122>/Sum' incorporates:
    //   Constant: '<S22>/Constant'
    //   Delay: '<S106>/UD'
    //   DiscreteIntegrator: '<S113>/Integrator'
    //   Product: '<S118>/PProd Out'
    //   Sum: '<S106>/Diff'
    s_lon = (s_lat * 0.04F + rtDW.Integrator_DSTATE) + (c_lon - rtDW.UD_DSTATE);

    // Saturate: '<S22>/stab_roll_rate_saturation'
    if (s_lon > 1.0F) {
      rtb_stab_roll_rate_saturation = 1.0F;
    } else if (s_lon < -1.0F) {
      rtb_stab_roll_rate_saturation = -1.0F;
    } else {
      rtb_stab_roll_rate_saturation = s_lon;
    }

    // End of Saturate: '<S22>/stab_roll_rate_saturation'

    // Switch generated from: '<Root>/Switch'
    if (rtb_Compare_lx) {
      rtb_pitch_angle_cmd_rad = rtDW.Gain_a;
    } else {
      rtb_pitch_angle_cmd_rad = rtDW.pitch_angle_cmd_rad;
    }

    // Sum: '<S21>/stab_pitch_angle_error_calc' incorporates:
    //   Inport: '<Root>/Navigation Filter Data'
    rtb_Subtract_f_idx_1 = rtb_pitch_angle_cmd_rad - nav.pitch_rad;

    // SampleTimeMath: '<S55>/Tsamp' incorporates:
    //   Constant: '<S21>/Constant2'
    //   Product: '<S52>/DProd Out'
    //
    //  About '<S55>/Tsamp':
    //   y = u * K where K = 1 / ( w * Ts )
    rtb_PProdOut_l = rtb_Subtract_f_idx_1 * 0.02F * 100.0F;

    // Sum: '<S69>/Sum' incorporates:
    //   Constant: '<S21>/Constant'
    //   Delay: '<S53>/UD'
    //   DiscreteIntegrator: '<S60>/Integrator'
    //   Product: '<S65>/PProd Out'
    //   Sum: '<S53>/Diff'
    rtb_Product1_i = (rtb_Subtract_f_idx_1 * 0.04F + rtDW.Integrator_DSTATE_l) +
      (rtb_PProdOut_l - rtDW.UD_DSTATE_f);

    // Saturate: '<S21>/stab_pitch_rate_saturation'
    if (rtb_Product1_i > 1.0F) {
      rtb_stab_pitch_rate_saturation = 1.0F;
    } else if (rtb_Product1_i < -1.0F) {
      rtb_stab_pitch_rate_saturation = -1.0F;
    } else {
      rtb_stab_pitch_rate_saturation = rtb_Product1_i;
    }

    // End of Saturate: '<S21>/stab_pitch_rate_saturation'

    // Switch generated from: '<Root>/Switch'
    if (rtb_Compare_lx) {
      rtb_yaw_rate_cmd_radps_g = rtDW.yaw_rate_cmd_radps_c5;
    } else {
      rtb_yaw_rate_cmd_radps_g = rtDW.yaw_rate_cmd_radps;
    }

    // Sum: '<S23>/stab_yaw_rate_error_calc' incorporates:
    //   Inport: '<Root>/Navigation Filter Data'
    rtb_DataTypeConversion6 = rtb_yaw_rate_cmd_radps_g - nav.gyro_radps[2];

    // SampleTimeMath: '<S161>/Tsamp' incorporates:
    //   Constant: '<S23>/Constant2'
    //   Product: '<S158>/DProd Out'
    //
    //  About '<S161>/Tsamp':
    //   y = u * K where K = 1 / ( w * Ts )
    rtb_Reshape_h_idx_1 = rtb_DataTypeConversion6 * 0.02F * 100.0F;

    // Sum: '<S175>/Sum' incorporates:
    //   Constant: '<S23>/Constant'
    //   Delay: '<S159>/UD'
    //   DiscreteIntegrator: '<S166>/Integrator'
    //   Product: '<S171>/PProd Out'
    //   Sum: '<S159>/Diff'
    rtb_Reshape_h_idx_0 = (rtb_DataTypeConversion6 * 0.5F +
      rtDW.Integrator_DSTATE_b) + (rtb_Reshape_h_idx_1 - rtDW.UD_DSTATE_m);

    // MATLAB Function: '<Root>/determine_wp_submode' incorporates:
    //   Constant: '<Root>/Constant1'
    //   Constant: '<Root>/Constant2'
    //   Inport: '<Root>/Navigation Filter Data'
    //   Inport: '<Root>/Telemetry Data'
    //   Selector: '<Root>/Selector'
    rtb_sub_mode = rtb_DataTypeConversion6_k;
    switch (telem.flight_plan[0].cmd) {
     case 20:
      rtb_sub_mode = 3;
      break;

     case 16:
      c_lat = 1.29246971E-26F;
      absxk = std::abs(1.5F - nav.ned_pos_m[0]);
      if (absxk > 1.29246971E-26F) {
        y = 1.0F;
        c_lat = absxk;
      } else {
        t = absxk / 1.29246971E-26F;
        y = t * t;
      }

      absxk = std::abs(1.5F - nav.ned_pos_m[1]);
      if (absxk > c_lat) {
        t = c_lat / absxk;
        y = y * t * t + 1.0F;
        c_lat = absxk;
      } else {
        t = absxk / c_lat;
        y += t * t;
      }

      absxk = std::abs(1.5F - nav.ned_pos_m[2]);
      if (absxk > c_lat) {
        t = c_lat / absxk;
        y = y * t * t + 1.0F;
        c_lat = absxk;
      } else {
        t = absxk / c_lat;
        y += t * t;
      }

      y = c_lat * std::sqrt(y);
      if ((rtb_DataTypeConversion6_k != 3) && (y <= 1.5F) &&
          (!telem.flight_plan[0].autocontinue)) {
        rtb_sub_mode = 1;
      }
      break;
    }

    // End of MATLAB Function: '<Root>/determine_wp_submode'

    // Switch: '<Root>/switch_motor_out'
    if (rtb_Switch_lt) {
      // Product: '<S4>/Multiply' incorporates:
      //   Math: '<S4>/Transpose'
      //   Reshape: '<S4>/Reshape'
      for (idx = 0; idx < 8; idx++) {
        rtb_Reshape_h_tmp = idx << 2;
        c_lat = rtConstB.Transpose[rtb_Reshape_h_tmp + 3] * rtb_Reshape_h_idx_0
          + (rtConstB.Transpose[rtb_Reshape_h_tmp + 2] *
             rtb_stab_pitch_rate_saturation +
             (rtConstB.Transpose[rtb_Reshape_h_tmp + 1] *
              rtb_stab_roll_rate_saturation +
              rtConstB.Transpose[rtb_Reshape_h_tmp] * rtb_throttle_cc));

        // Saturate: '<S4>/Saturation' incorporates:
        //   Math: '<S4>/Transpose'
        //   Reshape: '<S4>/Reshape'
        if (c_lat <= 0.15F) {
          rtb_switch_motor_out[idx] = 0.15F;
        } else {
          rtb_switch_motor_out[idx] = c_lat;
        }

        // End of Saturate: '<S4>/Saturation'
      }

      // End of Product: '<S4>/Multiply'
    } else {
      for (idx = 0; idx < 8; idx++) {
        rtb_switch_motor_out[idx] = 0.0F;
      }
    }

    // End of Switch: '<Root>/switch_motor_out'

    // Outport: '<Root>/VMS Data' incorporates:
    //   BusCreator: '<S10>/Bus Creator'
    //   BusCreator: '<S2>/Bus Creator3'
    //   BusCreator: '<S463>/Bus Creator'
    //   Constant: '<S2>/consumed_mah'
    //   Constant: '<S2>/current_ma'
    //   Constant: '<S2>/remaining_prcnt'
    //   Constant: '<S2>/remaining_time_s'
    //   Constant: '<S2>/voltage_v'
    //   Constant: '<S463>/Constant'
    //   Constant: '<S463>/Constant1'
    //   Constant: '<S464>/Constant'
    //   DataTypeConversion: '<Root>/Cast To Single'
    //   DataTypeConversion: '<Root>/Data Type Conversion'
    //   DataTypeConversion: '<Root>/Data Type Conversion1'
    //   DataTypeConversion: '<Root>/Data Type Conversion5'
    //   DataTypeConversion: '<Root>/Data Type Conversion6'
    //   DataTypeConversion: '<S464>/Data Type Conversion'
    //   Gain: '<Root>/Gain'
    //   Gain: '<S464>/Gain'
    //   Inport: '<Root>/Navigation Filter Data'
    //   SignalConversion generated from: '<S10>/Bus Creator'
    //   SignalConversion generated from: '<S463>/Bus Creator'
    //   Sum: '<S464>/Sum'
    //
    ctrl->motors_enabled = rtb_Switch_lt;
    ctrl->waypoint_reached = rtDW.reached;
    ctrl->mode = rtb_DataTypeConversion6_k;
    ctrl->throttle_cmd_prcnt = 100.0F * rtb_throttle_cc;
    ctrl->aux[0] = rtb_throttle_cc;
    ctrl->aux[1] = rtb_stab_roll_rate_saturation;
    ctrl->aux[2] = rtb_roll_angle_cmd_rad;
    ctrl->aux[3] = nav.roll_rad;
    ctrl->aux[4] = rtb_stab_pitch_rate_saturation;
    ctrl->aux[5] = rtb_pitch_angle_cmd_rad;
    ctrl->aux[6] = nav.pitch_rad;
    ctrl->aux[7] = rtb_Reshape_h_idx_0;
    ctrl->aux[8] = rtb_yaw_rate_cmd_radps_g;
    ctrl->aux[9] = nav.gyro_radps[2];
    ctrl->aux[10] = rtb_AND_n;
    ctrl->aux[11] = rtb_AND;
    ctrl->aux[12] = 0.0F;
    ctrl->aux[13] = 0.0F;
    ctrl->aux[14] = 0.0F;
    ctrl->aux[15] = 0.0F;
    ctrl->aux[16] = 0.0F;
    ctrl->aux[17] = rtb_sub_mode;
    ctrl->aux[18] = 0.0F;
    ctrl->aux[19] = 0.0F;
    ctrl->aux[20] = rtDW.current_waypoint;
    ctrl->aux[21] = rtDW.cur_target_pos_m[0];
    ctrl->aux[22] = rtDW.cur_target_pos_m[1];
    ctrl->aux[23] = rtDW.cur_target_pos_m[2];
    ctrl->sbus.ch17 = false;
    ctrl->sbus.ch18 = false;
    std::memset(&ctrl->sbus.cmd[0], 0, sizeof(real32_T) << 4U);
    for (idx = 0; idx < 16; idx++) {
      ctrl->sbus.cnt[idx] = 0;
    }

    for (idx = 0; idx < 8; idx++) {
      c_lat = rtb_switch_motor_out[idx];
      ctrl->pwm.cnt[idx] = static_cast<int16_T>(std::floor(1000.0F * c_lat +
        1000.0));
      ctrl->pwm.cmd[idx] = c_lat;
      ctrl->analog.val[idx] = 0.0F;
    }

    ctrl->battery.voltage_v = 1.0F;
    ctrl->battery.current_ma = 1.0F;
    ctrl->battery.consumed_mah = 1.0F;
    ctrl->battery.remaining_prcnt = 1.0F;
    ctrl->battery.remaining_time_s = 1.0F;

    // End of Outport: '<Root>/VMS Data'

    // Product: '<S163>/IProd Out' incorporates:
    //   Constant: '<S23>/Constant1'
    rtb_DataTypeConversion6 *= 0.05F;

    // Gain: '<S49>/ZeroGain'
    absxk = 0.0F * rtb_Product1_i;

    // DeadZone: '<S51>/DeadZone'
    if (rtb_Product1_i >= (rtMinusInfF)) {
      rtb_Product1_i = 0.0F;
    }

    // End of DeadZone: '<S51>/DeadZone'

    // Product: '<S57>/IProd Out' incorporates:
    //   Constant: '<S21>/Constant1'
    rtb_Subtract_f_idx_1 *= 0.04F;

    // Gain: '<S102>/ZeroGain'
    t = 0.0F * s_lon;

    // DeadZone: '<S104>/DeadZone'
    if (s_lon >= (rtMinusInfF)) {
      s_lon = 0.0F;
    }

    // End of DeadZone: '<S104>/DeadZone'

    // Product: '<S110>/IProd Out' incorporates:
    //   Constant: '<S22>/Constant1'
    s_lat *= 0.04F;

    // Signum: '<S102>/SignPreIntegrator'
    if (s_lat < 0.0F) {
      c_lat = -1.0F;
    } else if (s_lat > 0.0F) {
      c_lat = 1.0F;
    } else if (s_lat == 0.0F) {
      c_lat = 0.0F;
    } else {
      c_lat = (rtNaNF);
    }

    // End of Signum: '<S102>/SignPreIntegrator'

    // Switch: '<S102>/Switch' incorporates:
    //   Constant: '<S102>/Constant1'
    //   DataTypeConversion: '<S102>/DataTypeConv2'
    //   Logic: '<S102>/AND3'
    //   RelationalOperator: '<S102>/Equal1'
    //   RelationalOperator: '<S102>/NotEqual'
    if ((t != s_lon) && (0 == static_cast<int8_T>(c_lat))) {
      s_lat = 0.0F;
    }

    // End of Switch: '<S102>/Switch'

    // Update for DiscreteIntegrator: '<S113>/Integrator'
    rtDW.Integrator_DSTATE += 0.01F * s_lat;

    // Update for Delay: '<S106>/UD'
    rtDW.UD_DSTATE = c_lon;

    // Signum: '<S49>/SignPreIntegrator'
    if (rtb_Subtract_f_idx_1 < 0.0F) {
      c_lat = -1.0F;
    } else if (rtb_Subtract_f_idx_1 > 0.0F) {
      c_lat = 1.0F;
    } else if (rtb_Subtract_f_idx_1 == 0.0F) {
      c_lat = 0.0F;
    } else {
      c_lat = (rtNaNF);
    }

    // End of Signum: '<S49>/SignPreIntegrator'

    // Switch: '<S49>/Switch' incorporates:
    //   Constant: '<S49>/Constant1'
    //   DataTypeConversion: '<S49>/DataTypeConv2'
    //   Logic: '<S49>/AND3'
    //   RelationalOperator: '<S49>/Equal1'
    //   RelationalOperator: '<S49>/NotEqual'
    if ((absxk != rtb_Product1_i) && (0 == static_cast<int8_T>(c_lat))) {
      rtb_Subtract_f_idx_1 = 0.0F;
    }

    // End of Switch: '<S49>/Switch'

    // Update for DiscreteIntegrator: '<S60>/Integrator'
    rtDW.Integrator_DSTATE_l += 0.01F * rtb_Subtract_f_idx_1;

    // Update for Delay: '<S53>/UD'
    rtDW.UD_DSTATE_f = rtb_PProdOut_l;

    // DeadZone: '<S157>/DeadZone'
    if (rtb_Reshape_h_idx_0 >= (rtMinusInfF)) {
      t = 0.0F;
    } else {
      t = (rtNaNF);
    }

    // End of DeadZone: '<S157>/DeadZone'

    // Signum: '<S155>/SignPreIntegrator'
    if (rtb_DataTypeConversion6 < 0.0F) {
      c_lat = -1.0F;
    } else if (rtb_DataTypeConversion6 > 0.0F) {
      c_lat = 1.0F;
    } else if (rtb_DataTypeConversion6 == 0.0F) {
      c_lat = 0.0F;
    } else {
      c_lat = (rtNaNF);
    }

    // End of Signum: '<S155>/SignPreIntegrator'

    // Switch: '<S155>/Switch' incorporates:
    //   Constant: '<S155>/Constant1'
    //   DataTypeConversion: '<S155>/DataTypeConv2'
    //   Gain: '<S155>/ZeroGain'
    //   Logic: '<S155>/AND3'
    //   RelationalOperator: '<S155>/Equal1'
    //   RelationalOperator: '<S155>/NotEqual'
    if ((0.0F * rtb_Reshape_h_idx_0 != t) && (0 == static_cast<int8_T>(c_lat)))
    {
      rtb_DataTypeConversion6 = 0.0F;
    }

    // End of Switch: '<S155>/Switch'

    // Update for DiscreteIntegrator: '<S166>/Integrator'
    rtDW.Integrator_DSTATE_b += 0.01F * rtb_DataTypeConversion6;

    // Update for Delay: '<S159>/UD'
    rtDW.UD_DSTATE_m = rtb_Reshape_h_idx_1;
  }

  // Model initialize function
  void Autocode::initialize()
  {
    // Registration code

    // initialize non-finites
    rt_InitInfAndNaN(sizeof(real_T));

    // SystemInitialize for Enabled SubSystem: '<Root>/WAYPOINT CONTROLLER'
    // InitializeConditions for UnitDelay: '<S632>/Delay Input1'
    //
    //  Block description for '<S632>/Delay Input1':
    //
    //   Store in Global RAM
    rtDW.DelayInput1_DSTATE = -1;

    // End of SystemInitialize for SubSystem: '<Root>/WAYPOINT CONTROLLER'
  }

  // Constructor
  Autocode::Autocode() :
    rtDW()
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
