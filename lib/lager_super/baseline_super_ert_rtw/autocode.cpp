//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// File: autocode.cpp
//
// Code generated for Simulink model 'baseline_super'.
//
// Model version                  : 2.20
// Simulink Coder version         : 9.5 (R2021a) 14-Nov-2020
// C/C++ source code generated on : Thu May 12 15:45:48 2022
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
  // Function for MATLAB Function: '<S635>/determine_prev_tar_pos'
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

  // Function for MATLAB Function: '<S635>/determine_prev_tar_pos'
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

  // Function for MATLAB Function: '<S635>/determine_prev_tar_pos'
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
    real32_T c_lon;
    real32_T rtb_Cos_f;
    real32_T rtb_DataTypeConversion_h;
    real32_T rtb_Gain_h;
    real32_T rtb_Reshape_h_0;
    real32_T rtb_Reshape_h_idx_0;
    real32_T rtb_Subtract_f_idx_1;
    real32_T rtb_Switch_i;
    real32_T rtb_Tsamp_a;
    real32_T rtb_pitch_angle_cmd_rad;
    real32_T rtb_roll;
    real32_T rtb_roll_angle_cmd_rad;
    real32_T rtb_stab_pitch_rate_saturation;
    real32_T rtb_yaw;
    real32_T s_lat;
    real32_T s_lon;
    int8_T rtb_DataTypeConversion6;
    boolean_T rtb_AND;
    boolean_T rtb_AND_n;
    boolean_T rtb_Compare_lx;
    boolean_T rtb_Compare_n;
    boolean_T rtb_Switch_lt;
    UNUSED_PARAMETER(sys);

    // Logic: '<S19>/nav_init AND motor_enable' incorporates:
    //   Constant: '<S666>/Constant'
    //   DataTypeConversion: '<S645>/Data Type Conversion'
    //   Inport: '<Root>/Navigation Filter Data'
    //   Inport: '<Root>/Sensor Data'
    //   Polyval: '<S645>/throttle_en_norm'
    //   RelationalOperator: '<S666>/Compare'
    rtb_Switch_lt = ((0.00122026F * static_cast<real32_T>(sensor.inceptor.ch[6])
                      + -1.20988405F > 0.0F) && nav.nav_initialized);

    // Outputs for Enabled SubSystem: '<S646>/disarm motor' incorporates:
    //   EnablePort: '<S650>/Enable'

    // RelationalOperator: '<S649>/Compare' incorporates:
    //   Constant: '<S649>/Constant'
    //   Inport: '<Root>/Sensor Data'
    if (sensor.power_module.voltage_v <= 46.8F) {
      if (!rtDW.disarmmotor_MODE_k) {
        // InitializeConditions for UnitDelay: '<S650>/Unit Delay'
        rtDW.UnitDelay_DSTATE_m = 0.0;
        rtDW.disarmmotor_MODE_k = true;
      }

      // RelationalOperator: '<S651>/Compare' incorporates:
      //   Constant: '<S650>/Constant'
      //   Constant: '<S651>/Constant'
      //   Sum: '<S650>/Sum'
      //   UnitDelay: '<S650>/Unit Delay'
      rtDW.Compare_d = (rtDW.UnitDelay_DSTATE_m + 0.01 > 15.0);

      // Update for UnitDelay: '<S650>/Unit Delay' incorporates:
      //   Constant: '<S650>/Constant'
      //   Sum: '<S650>/Sum'
      rtDW.UnitDelay_DSTATE_m += 0.01;
    } else {
      rtDW.disarmmotor_MODE_k = false;
    }

    // End of RelationalOperator: '<S649>/Compare'
    // End of Outputs for SubSystem: '<S646>/disarm motor'

    // Polyval: '<S644>/mode_norm' incorporates:
    //   DataTypeConversion: '<S644>/Data Type Conversion1'
    //   Inport: '<Root>/Sensor Data'
    rtb_roll_angle_cmd_rad = 0.00122026F * static_cast<real32_T>
      (sensor.inceptor.ch[4]) + -0.209884077F;

    // DataTypeConversion: '<S644>/Data Type Conversion6'
    if (std::abs(rtb_roll_angle_cmd_rad) >= 0.5F) {
      s_lat = std::floor(rtb_roll_angle_cmd_rad + 0.5F);
      rtb_DataTypeConversion6 = static_cast<int8_T>(s_lat);
    } else {
      s_lat = rtb_roll_angle_cmd_rad * 0.0F;
      rtb_DataTypeConversion6 = static_cast<int8_T>(rtb_roll_angle_cmd_rad *
        0.0F);
    }

    // Logic: '<S646>/AND' incorporates:
    //   Constant: '<S648>/Constant'
    //   DataTypeConversion: '<S644>/Data Type Conversion6'
    //   Logic: '<S646>/AND1'
    //   RelationalOperator: '<S648>/Compare'
    rtb_AND_n = (rtDW.Compare_d && ((static_cast<int8_T>(s_lat) != 5) &&
      rtb_Switch_lt));

    // Logic: '<S647>/AND' incorporates:
    //   Constant: '<S652>/Constant'
    //   Constant: '<S653>/Constant'
    //   Constant: '<S654>/Constant'
    //   DataTypeConversion: '<S644>/Data Type Conversion6'
    //   Inport: '<Root>/Sensor Data'
    //   Logic: '<S647>/NOR'
    //   Logic: '<S647>/NOT'
    //   RelationalOperator: '<S652>/Compare'
    //   RelationalOperator: '<S653>/Compare'
    //   RelationalOperator: '<S654>/Compare'
    rtb_AND = (sensor.inceptor.failsafe && ((static_cast<int8_T>(s_lat) == 3) &&
                (static_cast<int8_T>(s_lat) == 4) && (static_cast<int8_T>(s_lat)
      == 5) && rtb_Switch_lt));

    // Switch: '<S19>/Switch1' incorporates:
    //   Constant: '<S19>/land_mode'
    //   Switch: '<S19>/Switch2'
    if (rtb_AND_n) {
      rtb_DataTypeConversion6 = 5;
    } else if (rtb_AND) {
      // Switch: '<S19>/Switch2' incorporates:
      //   Constant: '<S19>/rtl_mode'
      rtb_DataTypeConversion6 = 3;
    }

    // End of Switch: '<S19>/Switch1'

    // Outputs for Enabled SubSystem: '<S643>/waypoint submodes' incorporates:
    //   EnablePort: '<S662>/Enable'

    // RelationalOperator: '<S660>/Compare' incorporates:
    //   Constant: '<S660>/Constant'
    if (rtb_DataTypeConversion6 == 2) {
      // MATLAB Function: '<S662>/determine_target_pos' incorporates:
      //   Inport: '<Root>/Navigation Filter Data'
      diff[0] = static_cast<real32_T>(nav.home_lat_rad * 57.295779513082323);
      diff[1] = static_cast<real32_T>(nav.home_lon_rad * 57.295779513082323);
      diff[2] = nav.home_alt_wgs84_m;
      rtb_roll_angle_cmd_rad = diff[0];
      cosd(&rtb_roll_angle_cmd_rad);
      s_lat = diff[0];
      sind(&s_lat);
      c_lon = diff[1];
      cosd(&c_lon);
      s_lon = diff[1];
      sind(&s_lon);

      // MATLAB Function: '<S662>/determine_wp_submode' incorporates:
      //   Constant: '<S662>/Constant'
      //   Inport: '<Root>/Navigation Filter Data'
      //   Inport: '<Root>/Telemetry Data'
      //   MATLAB Function: '<S662>/determine_target_pos'
      //   Selector: '<S662>/Selector'
      rtDW.sub_mode = 2;
      switch (telem.flight_plan[telem.current_waypoint].cmd) {
       case 20:
        rtDW.sub_mode = 3;
        break;

       case 16:
        // MATLAB Function: '<S662>/determine_target_pos' incorporates:
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
        s_lat_0[2] = -rtb_roll_angle_cmd_rad * c_lon;
        s_lat_0[3] = -s_lat * s_lon;
        s_lat_0[4] = c_lon;
        s_lat_0[5] = -rtb_roll_angle_cmd_rad * s_lon;
        s_lat_0[6] = rtb_roll_angle_cmd_rad;
        s_lat_0[7] = 0.0F;
        s_lat_0[8] = -s_lat;
        rtb_DataTypeConversion_h = tmp_0[0] - tmp[0];
        rtb_Reshape_h_idx_0 = tmp_0[1] - tmp[1];
        rtb_Subtract_f_idx_1 = tmp_0[2] - tmp[2];
        s_lon = 0.0F;
        rtb_roll_angle_cmd_rad = 1.29246971E-26F;
        for (idx = 0; idx < 3; idx++) {
          s_lat = std::abs(((s_lat_0[idx + 3] * rtb_Reshape_h_idx_0 +
                             s_lat_0[idx] * rtb_DataTypeConversion_h) +
                            s_lat_0[idx + 6] * rtb_Subtract_f_idx_1) -
                           nav.ned_pos_m[idx]);
          if (s_lat > rtb_roll_angle_cmd_rad) {
            c_lon = rtb_roll_angle_cmd_rad / s_lat;
            s_lon = s_lon * c_lon * c_lon + 1.0F;
            rtb_roll_angle_cmd_rad = s_lat;
          } else {
            c_lon = s_lat / rtb_roll_angle_cmd_rad;
            s_lon += c_lon * c_lon;
          }
        }

        s_lon = rtb_roll_angle_cmd_rad * std::sqrt(s_lon);
        if ((s_lon <= 1.5F) && (!telem.flight_plan[telem.current_waypoint].
             autocontinue)) {
          rtDW.sub_mode = 1;
        }
        break;
      }

      // End of MATLAB Function: '<S662>/determine_wp_submode'
    }

    // End of RelationalOperator: '<S660>/Compare'
    // End of Outputs for SubSystem: '<S643>/waypoint submodes'

    // Outputs for Enabled SubSystem: '<S643>/rtl submodes' incorporates:
    //   EnablePort: '<S661>/Enable'

    // RelationalOperator: '<S659>/Compare' incorporates:
    //   Constant: '<S659>/Constant'
    if (rtb_DataTypeConversion6 == 3) {
      // MATLAB Function: '<S661>/determine_rtl_submode' incorporates:
      //   Constant: '<S661>/Constant'
      //   Inport: '<Root>/Navigation Filter Data'
      rtDW.sub_mode_m = 3;
      rtb_roll_angle_cmd_rad = 1.29246971E-26F;
      s_lat = std::abs(0.0F - nav.ned_pos_m[0]);
      if (s_lat > 1.29246971E-26F) {
        s_lon = 1.0F;
        rtb_roll_angle_cmd_rad = s_lat;
      } else {
        c_lon = s_lat / 1.29246971E-26F;
        s_lon = c_lon * c_lon;
      }

      s_lat = std::abs(0.0F - nav.ned_pos_m[1]);
      if (s_lat > rtb_roll_angle_cmd_rad) {
        c_lon = rtb_roll_angle_cmd_rad / s_lat;
        s_lon = s_lon * c_lon * c_lon + 1.0F;
        rtb_roll_angle_cmd_rad = s_lat;
      } else {
        c_lon = s_lat / rtb_roll_angle_cmd_rad;
        s_lon += c_lon * c_lon;
      }

      s_lat = std::abs(0.0F - nav.ned_pos_m[2]);
      if (s_lat > rtb_roll_angle_cmd_rad) {
        c_lon = rtb_roll_angle_cmd_rad / s_lat;
        s_lon = s_lon * c_lon * c_lon + 1.0F;
        rtb_roll_angle_cmd_rad = s_lat;
      } else {
        c_lon = s_lat / rtb_roll_angle_cmd_rad;
        s_lon += c_lon * c_lon;
      }

      s_lon = rtb_roll_angle_cmd_rad * std::sqrt(s_lon);
      if (s_lon <= 1.5F) {
        rtDW.sub_mode_m = 4;
      }

      // End of MATLAB Function: '<S661>/determine_rtl_submode'
    }

    // End of RelationalOperator: '<S659>/Compare'
    // End of Outputs for SubSystem: '<S643>/rtl submodes'

    // Switch: '<S19>/Switch3' incorporates:
    //   Logic: '<S19>/OR'
    if ((!rtb_AND_n) && (!rtb_AND)) {
      // MultiPortSwitch: '<S643>/Multiport Switch'
      switch (rtb_DataTypeConversion6) {
       case 2:
        rtb_DataTypeConversion6 = rtDW.sub_mode;
        break;

       case 3:
        rtb_DataTypeConversion6 = rtDW.sub_mode_m;
        break;
      }

      // End of MultiPortSwitch: '<S643>/Multiport Switch'
    }

    // End of Switch: '<S19>/Switch3'

    // Outputs for Enabled SubSystem: '<S19>/auto_disarm' incorporates:
    //   EnablePort: '<S641>/Enable'

    // Logic: '<S19>/motor_armed AND mode_4' incorporates:
    //   Abs: '<S641>/Abs'
    //   Constant: '<S642>/Constant'
    //   Constant: '<S655>/Constant'
    //   Constant: '<S656>/Constant'
    //   Gain: '<S641>/Gain'
    //   Inport: '<Root>/Navigation Filter Data'
    //   Logic: '<S641>/AND'
    //   RelationalOperator: '<S642>/Compare'
    //   RelationalOperator: '<S655>/Compare'
    //   RelationalOperator: '<S656>/Compare'
    if (rtb_Switch_lt && (rtb_DataTypeConversion6 == 4)) {
      rtDW.auto_disarm_MODE = true;

      // Outputs for Enabled SubSystem: '<S641>/disarm motor' incorporates:
      //   EnablePort: '<S657>/Enable'
      if ((-nav.ned_pos_m[2] <= 10.0F) && (std::abs(nav.ned_vel_mps[2]) <= 0.3F))
      {
        if (!rtDW.disarmmotor_MODE) {
          // InitializeConditions for UnitDelay: '<S657>/Unit Delay'
          rtDW.UnitDelay_DSTATE = 0.0;
          rtDW.disarmmotor_MODE = true;
        }

        // RelationalOperator: '<S658>/Compare' incorporates:
        //   Constant: '<S657>/Constant'
        //   Constant: '<S658>/Constant'
        //   Sum: '<S657>/Sum'
        //   UnitDelay: '<S657>/Unit Delay'
        rtDW.Compare = (rtDW.UnitDelay_DSTATE + 0.01 > 10.0);

        // Update for UnitDelay: '<S657>/Unit Delay' incorporates:
        //   Constant: '<S657>/Constant'
        //   Sum: '<S657>/Sum'
        rtDW.UnitDelay_DSTATE += 0.01;
      } else {
        rtDW.disarmmotor_MODE = false;
      }

      // End of Outputs for SubSystem: '<S641>/disarm motor'
    } else if (rtDW.auto_disarm_MODE) {
      // Disable for Enabled SubSystem: '<S641>/disarm motor'
      rtDW.disarmmotor_MODE = false;

      // End of Disable for SubSystem: '<S641>/disarm motor'
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
    //   Inport: '<Root>/Telemetry Data'
    //   Logic: '<S11>/OR'
    //   RelationalOperator: '<S18>/Compare'
    //   RelationalOperator: '<S632>/FixPt Relational Operator'
    //   UnitDelay: '<S632>/Delay Input1'
    //
    //  Block description for '<S632>/Delay Input1':
    //
    //   Store in Global RAM
    if (rtb_Switch_lt && (rtb_DataTypeConversion6 == 2)) {
      // RelationalOperator: '<S633>/Compare' incorporates:
      //   Constant: '<S633>/Constant'
      //   Inport: '<Root>/Telemetry Data'
      //   RelationalOperator: '<S631>/FixPt Relational Operator'
      //   UnitDelay: '<S631>/Delay Input1'
      //
      //  Block description for '<S631>/Delay Input1':
      //
      //   Store in Global RAM
      rtb_Compare_lx = (telem.current_waypoint != rtDW.DelayInput1_DSTATE);

      // Outputs for Enabled SubSystem: '<S11>/determine target' incorporates:
      //   EnablePort: '<S466>/Enable'
      if (telem.waypoints_updated || (static_cast<int32_T>(rtb_Compare_lx) >
           static_cast<int32_T>(rtDW.DelayInput1_DSTATE_n))) {
        // RelationalOperator: '<S634>/Compare' incorporates:
        //   Constant: '<S466>/Constant'
        //   Constant: '<S634>/Constant'
        //   Sum: '<S466>/Sum'
        rtb_Compare_n = (static_cast<real_T>(telem.current_waypoint) - 1.0 >=
                         0.0);

        // Outputs for Enabled SubSystem: '<S466>/calc_prev_target_pos' incorporates:
        //   EnablePort: '<S635>/Enable'
        if (rtb_Compare_n) {
          // MATLAB Function: '<S635>/determine_prev_tar_pos' incorporates:
          //   Constant: '<S466>/Constant'
          //   Inport: '<Root>/Navigation Filter Data'
          //   Selector: '<S635>/Selector1'
          //   Sum: '<S466>/Sum'
          diff[0] = static_cast<real32_T>(nav.home_lat_rad * 57.295779513082323);
          diff[1] = static_cast<real32_T>(nav.home_lon_rad * 57.295779513082323);
          diff[2] = nav.home_alt_wgs84_m;
          rtb_roll_angle_cmd_rad = diff[0];
          cosd(&rtb_roll_angle_cmd_rad);
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
          s_lat_0[2] = -rtb_roll_angle_cmd_rad * c_lon;
          s_lat_0[3] = -s_lat * s_lon;
          s_lat_0[4] = c_lon;
          s_lat_0[5] = -rtb_roll_angle_cmd_rad * s_lon;
          s_lat_0[6] = rtb_roll_angle_cmd_rad;
          s_lat_0[7] = 0.0F;
          s_lat_0[8] = -s_lat;
          rtb_DataTypeConversion_h = tmp_0[0] - tmp[0];
          rtb_Reshape_h_idx_0 = tmp_0[1] - tmp[1];
          rtb_Subtract_f_idx_1 = tmp_0[2] - tmp[2];
          for (idx = 0; idx < 3; idx++) {
            rtDW.pref_target_pos[idx] = 0.0F;
            rtDW.pref_target_pos[idx] += s_lat_0[idx] * rtb_DataTypeConversion_h;
            rtDW.pref_target_pos[idx] += s_lat_0[idx + 3] * rtb_Reshape_h_idx_0;
            rtDW.pref_target_pos[idx] += s_lat_0[idx + 6] * rtb_Subtract_f_idx_1;
          }

          // End of MATLAB Function: '<S635>/determine_prev_tar_pos'
        }

        // End of Outputs for SubSystem: '<S466>/calc_prev_target_pos'

        // MATLAB Function: '<S466>/determine_current_tar_pos' incorporates:
        //   Inport: '<Root>/Navigation Filter Data'
        //   Selector: '<S466>/Selector'
        diff[0] = static_cast<real32_T>(nav.home_lat_rad * 57.295779513082323);
        diff[1] = static_cast<real32_T>(nav.home_lon_rad * 57.295779513082323);
        diff[2] = nav.home_alt_wgs84_m;
        rtb_roll_angle_cmd_rad = diff[0];
        cosd(&rtb_roll_angle_cmd_rad);
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
        s_lat_0[2] = -rtb_roll_angle_cmd_rad * c_lon;
        s_lat_0[3] = -s_lat * s_lon;
        s_lat_0[4] = c_lon;
        s_lat_0[5] = -rtb_roll_angle_cmd_rad * s_lon;
        s_lat_0[6] = rtb_roll_angle_cmd_rad;
        s_lat_0[7] = 0.0F;
        s_lat_0[8] = -s_lat;
        rtb_DataTypeConversion_h = tmp_0[0] - tmp[0];
        rtb_Reshape_h_idx_0 = tmp_0[1] - tmp[1];
        rtb_Subtract_f_idx_1 = tmp_0[2] - tmp[2];
        for (idx = 0; idx < 3; idx++) {
          rtDW.cur_target_pos_m_c[idx] = 0.0F;
          rtDW.cur_target_pos_m_c[idx] += s_lat_0[idx] *
            rtb_DataTypeConversion_h;
          rtDW.cur_target_pos_m_c[idx] += s_lat_0[idx + 3] * rtb_Reshape_h_idx_0;
          rtDW.cur_target_pos_m_c[idx] += s_lat_0[idx + 6] *
            rtb_Subtract_f_idx_1;

          // Switch: '<S466>/Switch'
          if (rtb_Compare_n) {
            rtb_roll_angle_cmd_rad = rtDW.pref_target_pos[idx];
          } else {
            rtb_roll_angle_cmd_rad = nav.ned_pos_m[idx];
          }

          // End of Switch: '<S466>/Switch'

          // MATLAB Function: '<S466>/determine_target'
          diff[idx] = rtDW.cur_target_pos_m_c[idx] - rtb_roll_angle_cmd_rad;
        }

        // End of MATLAB Function: '<S466>/determine_current_tar_pos'

        // MATLAB Function: '<S466>/determine_target' incorporates:
        //   Constant: '<S466>/Constant2'
        //   Constant: '<S466>/Constant3'
        rtb_roll_angle_cmd_rad = 1.29246971E-26F;
        s_lat = std::abs(diff[0]);
        if (s_lat > 1.29246971E-26F) {
          s_lon = 1.0F;
          rtb_roll_angle_cmd_rad = s_lat;
        } else {
          c_lon = s_lat / 1.29246971E-26F;
          s_lon = c_lon * c_lon;
        }

        s_lat = std::abs(diff[1]);
        if (s_lat > rtb_roll_angle_cmd_rad) {
          c_lon = rtb_roll_angle_cmd_rad / s_lat;
          s_lon = s_lon * c_lon * c_lon + 1.0F;
          rtb_roll_angle_cmd_rad = s_lat;
        } else {
          c_lon = s_lat / rtb_roll_angle_cmd_rad;
          s_lon += c_lon * c_lon;
        }

        s_lon = rtb_roll_angle_cmd_rad * std::sqrt(s_lon);
        rtDW.cur_target_heading_rad = rt_atan2f_snf(diff[1], diff[0]);
        rtDW.max_v_z_mps = std::abs(-diff[2] * 5.0F / s_lon);
        rtDW.max_v_hor_mps = std::abs(s_lon * 2.0F / -diff[2]);
      }

      // End of Outputs for SubSystem: '<S11>/determine target'

      // Outputs for Enabled SubSystem: '<S11>/WP_NAV' incorporates:
      //   EnablePort: '<S464>/Enable'

      // Trigonometry: '<S525>/Sin' incorporates:
      //   Inport: '<Root>/Navigation Filter Data'
      //   Inport: '<Root>/Telemetry Data'
      //   Logic: '<S11>/OR'
      //   RelationalOperator: '<S632>/FixPt Relational Operator'
      //   UnitDelay: '<S632>/Delay Input1'
      //
      //  Block description for '<S632>/Delay Input1':
      //
      //   Store in Global RAM
      rtb_DataTypeConversion_h = std::sin(nav.heading_rad);

      // Reshape: '<S525>/Reshape' incorporates:
      //   Gain: '<S525>/Gain'
      //   Inport: '<Root>/Navigation Filter Data'
      //   Reshape: '<S186>/Reshape'
      //   Trigonometry: '<S525>/Cos'
      rtb_Reshape_h_idx_0 = std::cos(nav.heading_rad);
      c_lon = -rtb_DataTypeConversion_h;
      s_lon = rtb_DataTypeConversion_h;

      // Sum: '<S472>/Subtract' incorporates:
      //   Inport: '<Root>/Navigation Filter Data'
      //   SignalConversion generated from: '<S468>/Bus Selector2'
      rtb_DataTypeConversion_h = rtDW.cur_target_pos_m_c[0] - nav.ned_pos_m[0];
      rtb_Subtract_f_idx_1 = rtDW.cur_target_pos_m_c[1] - nav.ned_pos_m[1];

      // MinMax: '<S471>/Min'
      if (rtDW.max_v_hor_mps < 5.0F) {
        rtb_roll_angle_cmd_rad = rtDW.max_v_hor_mps;
      } else {
        rtb_roll_angle_cmd_rad = 5.0F;
      }

      if (!(rtb_roll_angle_cmd_rad < 3.0F)) {
        rtb_roll_angle_cmd_rad = 3.0F;
      }

      // End of MinMax: '<S471>/Min'

      // Sqrt: '<S472>/Sqrt' incorporates:
      //   Math: '<S472>/Transpose'
      //   Product: '<S472>/MatrixMultiply'
      s_lat = std::sqrt(rtb_DataTypeConversion_h * rtb_DataTypeConversion_h +
                        rtb_Subtract_f_idx_1 * rtb_Subtract_f_idx_1);

      // Saturate: '<S471>/Saturation'
      if (s_lat > 20.0F) {
        s_lat = 20.0F;
      } else if (s_lat < 0.0F) {
        s_lat = 0.0F;
      }

      // End of Saturate: '<S471>/Saturation'

      // Product: '<S512>/PProd Out' incorporates:
      //   Constant: '<S471>/Constant3'
      s_lat *= 3.0F;

      // Switch: '<S515>/Switch2' incorporates:
      //   RelationalOperator: '<S515>/LowerRelop1'
      //   Switch: '<S515>/Switch'
      if (!(s_lat > rtb_roll_angle_cmd_rad)) {
        rtb_roll_angle_cmd_rad = s_lat;
      }

      // End of Switch: '<S515>/Switch2'

      // Trigonometry: '<S472>/Atan2'
      rtb_DataTypeConversion_h = rt_atan2f_snf(rtb_Subtract_f_idx_1,
        rtb_DataTypeConversion_h);

      // SignalConversion generated from: '<S475>/Product' incorporates:
      //   Product: '<S474>/Product'
      //   Product: '<S474>/Product1'
      //   Trigonometry: '<S474>/Cos'
      //   Trigonometry: '<S474>/Sin'
      s_lat = rtb_roll_angle_cmd_rad * std::cos(rtb_DataTypeConversion_h);
      rtb_roll_angle_cmd_rad *= std::sin(rtb_DataTypeConversion_h);

      // Product: '<S475>/Product' incorporates:
      //   Reshape: '<S186>/Reshape'
      rtDW.vb_xy[0] = 0.0F;
      rtDW.vb_xy[0] += rtb_Reshape_h_idx_0 * s_lat;
      rtDW.vb_xy[0] += s_lon * rtb_roll_angle_cmd_rad;
      rtDW.vb_xy[1] = 0.0F;
      rtDW.vb_xy[1] += c_lon * s_lat;
      rtDW.vb_xy[1] += rtb_Reshape_h_idx_0 * rtb_roll_angle_cmd_rad;

      // Product: '<S564>/PProd Out' incorporates:
      //   Inport: '<Root>/Navigation Filter Data'
      //   Sum: '<S526>/Sum3'
      rtb_roll_angle_cmd_rad = rtDW.cur_target_pos_m_c[2] - nav.ned_pos_m[2];

      // Switch: '<S567>/Switch2' incorporates:
      //   Constant: '<S526>/Constant1'
      //   MinMax: '<S526>/Min'
      //   RelationalOperator: '<S567>/LowerRelop1'
      if (rtb_roll_angle_cmd_rad > 1.0F) {
        // Switch: '<S567>/Switch2'
        rtDW.Switch2 = 1.0F;
      } else {
        if (rtDW.max_v_z_mps < 2.0F) {
          // MinMax: '<S526>/Min'
          s_lat = rtDW.max_v_z_mps;
        } else {
          // MinMax: '<S526>/Min'
          s_lat = 2.0F;
        }

        // Switch: '<S567>/Switch' incorporates:
        //   Gain: '<S526>/Gain'
        //   RelationalOperator: '<S567>/UpperRelop'
        if (rtb_roll_angle_cmd_rad < -s_lat) {
          // Switch: '<S567>/Switch2'
          rtDW.Switch2 = -s_lat;
        } else {
          // Switch: '<S567>/Switch2'
          rtDW.Switch2 = rtb_roll_angle_cmd_rad;
        }

        // End of Switch: '<S567>/Switch'
      }

      // End of Switch: '<S567>/Switch2'

      // Sum: '<S579>/Subtract' incorporates:
      //   Inport: '<Root>/Navigation Filter Data'
      rtb_roll_angle_cmd_rad = rtDW.cur_target_heading_rad - nav.heading_rad;

      // Switch: '<S579>/Switch' incorporates:
      //   Abs: '<S579>/Abs'
      //   Constant: '<S579>/Constant'
      //   Constant: '<S630>/Constant'
      //   Product: '<S579>/Product'
      //   RelationalOperator: '<S630>/Compare'
      //   Sum: '<S579>/Subtract1'
      if (std::abs(rtb_roll_angle_cmd_rad) > 3.14159274F) {
        // Signum: '<S579>/Sign'
        if (rtb_roll_angle_cmd_rad < 0.0F) {
          rtb_Switch_i = -1.0F;
        } else if (rtb_roll_angle_cmd_rad > 0.0F) {
          rtb_Switch_i = 1.0F;
        } else if (rtb_roll_angle_cmd_rad == 0.0F) {
          rtb_Switch_i = 0.0F;
        } else {
          rtb_Switch_i = (rtNaNF);
        }

        // End of Signum: '<S579>/Sign'
        rtb_roll_angle_cmd_rad -= rtb_Switch_i * 6.28318548F;
      }

      // End of Switch: '<S579>/Switch'

      // Product: '<S610>/IProd Out' incorporates:
      //   Constant: '<S577>/I_heading'
      rtb_DataTypeConversion_h = rtb_roll_angle_cmd_rad * 0.01F;

      // Sum: '<S622>/Sum' incorporates:
      //   DiscreteIntegrator: '<S613>/Integrator'
      //   Product: '<S618>/PProd Out'
      c_lon = rtb_roll_angle_cmd_rad + rtDW.Integrator_DSTATE_bm;

      // Saturate: '<S577>/Saturation'
      if (c_lon > 0.524F) {
        // Saturate: '<S577>/Saturation'
        rtDW.Saturation = 0.524F;
      } else if (c_lon < -0.524F) {
        // Saturate: '<S577>/Saturation'
        rtDW.Saturation = -0.524F;
      } else {
        // Saturate: '<S577>/Saturation'
        rtDW.Saturation = c_lon;
      }

      // End of Saturate: '<S577>/Saturation'

      // DeadZone: '<S606>/DeadZone'
      if (c_lon >= (rtMinusInfF)) {
        rtb_roll_angle_cmd_rad = 0.0F;
      } else {
        rtb_roll_angle_cmd_rad = (rtNaNF);
      }

      // End of DeadZone: '<S606>/DeadZone'

      // Signum: '<S604>/SignPreIntegrator'
      if (rtb_DataTypeConversion_h < 0.0F) {
        s_lat = -1.0F;
      } else if (rtb_DataTypeConversion_h > 0.0F) {
        s_lat = 1.0F;
      } else if (rtb_DataTypeConversion_h == 0.0F) {
        s_lat = 0.0F;
      } else {
        s_lat = (rtNaNF);
      }

      // End of Signum: '<S604>/SignPreIntegrator'

      // Switch: '<S604>/Switch' incorporates:
      //   Constant: '<S604>/Constant1'
      //   DataTypeConversion: '<S604>/DataTypeConv2'
      //   Gain: '<S604>/ZeroGain'
      //   Logic: '<S604>/AND3'
      //   RelationalOperator: '<S604>/Equal1'
      //   RelationalOperator: '<S604>/NotEqual'
      if ((0.0F * c_lon != rtb_roll_angle_cmd_rad) && (0 == static_cast<int8_T>
           (s_lat))) {
        rtb_DataTypeConversion_h = 0.0F;
      }

      // End of Switch: '<S604>/Switch'

      // Update for DiscreteIntegrator: '<S613>/Integrator'
      rtDW.Integrator_DSTATE_bm += 0.01F * rtb_DataTypeConversion_h;

      // End of Outputs for SubSystem: '<S11>/WP_NAV'

      // MATLAB Function: '<S467>/check_wp_reached'
      rtb_roll_angle_cmd_rad = 1.29246971E-26F;

      // SignalConversion generated from: '<S11>/dbg'
      rtDW.cur_target_pos_m[0] = rtDW.cur_target_pos_m_c[0];

      // MATLAB Function: '<S467>/check_wp_reached' incorporates:
      //   Inport: '<Root>/Navigation Filter Data'
      s_lat = std::abs(rtDW.cur_target_pos_m_c[0] - nav.ned_pos_m[0]);
      if (s_lat > 1.29246971E-26F) {
        s_lon = 1.0F;
        rtb_roll_angle_cmd_rad = s_lat;
      } else {
        c_lon = s_lat / 1.29246971E-26F;
        s_lon = c_lon * c_lon;
      }

      // SignalConversion generated from: '<S11>/dbg'
      rtDW.cur_target_pos_m[1] = rtDW.cur_target_pos_m_c[1];

      // MATLAB Function: '<S467>/check_wp_reached' incorporates:
      //   Inport: '<Root>/Navigation Filter Data'
      s_lat = std::abs(rtDW.cur_target_pos_m_c[1] - nav.ned_pos_m[1]);
      if (s_lat > rtb_roll_angle_cmd_rad) {
        c_lon = rtb_roll_angle_cmd_rad / s_lat;
        s_lon = s_lon * c_lon * c_lon + 1.0F;
        rtb_roll_angle_cmd_rad = s_lat;
      } else {
        c_lon = s_lat / rtb_roll_angle_cmd_rad;
        s_lon += c_lon * c_lon;
      }

      // SignalConversion generated from: '<S11>/dbg'
      rtDW.cur_target_pos_m[2] = rtDW.cur_target_pos_m_c[2];

      // MATLAB Function: '<S467>/check_wp_reached' incorporates:
      //   Constant: '<S467>/Constant'
      //   Inport: '<Root>/Navigation Filter Data'
      //   Inport: '<Root>/Telemetry Data'
      //   Selector: '<S467>/Selector'
      s_lat = std::abs(rtDW.cur_target_pos_m_c[2] - nav.ned_pos_m[2]);
      if (s_lat > rtb_roll_angle_cmd_rad) {
        c_lon = rtb_roll_angle_cmd_rad / s_lat;
        s_lon = s_lon * c_lon * c_lon + 1.0F;
        rtb_roll_angle_cmd_rad = s_lat;
      } else {
        c_lon = s_lat / rtb_roll_angle_cmd_rad;
        s_lon += c_lon * c_lon;
      }

      rtDW.reached = ((rtb_roll_angle_cmd_rad * std::sqrt(s_lon) <= 1.5F) &&
                      telem.flight_plan[telem.current_waypoint].autocontinue);

      // Update for UnitDelay: '<S631>/Delay Input1' incorporates:
      //   Inport: '<Root>/Telemetry Data'
      //
      //  Block description for '<S631>/Delay Input1':
      //
      //   Store in Global RAM
      rtDW.DelayInput1_DSTATE = telem.current_waypoint;

      // Update for UnitDelay: '<S632>/Delay Input1'
      //
      //  Block description for '<S632>/Delay Input1':
      //
      //   Store in Global RAM
      rtDW.DelayInput1_DSTATE_n = rtb_Compare_lx;
    }

    // End of Logic: '<Root>/motor_armed AND mode_2'
    // End of Outputs for SubSystem: '<Root>/WAYPOINT CONTROLLER'

    // Polyval: '<S12>/pitch_norm' incorporates:
    //   DataTypeConversion: '<S12>/Data Type Conversion4'
    //   Inport: '<Root>/Sensor Data'
    s_lon = 0.00122026F * static_cast<real32_T>(sensor.inceptor.ch[2]) +
      -1.20988405F;

    // Polyval: '<S12>/roll_norm' incorporates:
    //   DataTypeConversion: '<S12>/Data Type Conversion3'
    //   Inport: '<Root>/Sensor Data'
    rtb_roll = 0.00122026F * static_cast<real32_T>(sensor.inceptor.ch[1]) +
      -1.20988405F;

    // Polyval: '<S12>/yaw_norm' incorporates:
    //   DataTypeConversion: '<S12>/Data Type Conversion2'
    //   Inport: '<Root>/Sensor Data'
    rtb_yaw = 0.00122026F * static_cast<real32_T>(sensor.inceptor.ch[3]) +
      -1.20988405F;

    // Logic: '<Root>/motor_armed AND mode_5' incorporates:
    //   Constant: '<S17>/Constant'
    //   RelationalOperator: '<S17>/Compare'
    rtb_Compare_lx = (rtb_Switch_lt && (rtb_DataTypeConversion6 == 1));

    // Polyval: '<S12>/throttle_norm' incorporates:
    //   DataTypeConversion: '<S12>/Data Type Conversion5'
    //   Inport: '<Root>/Sensor Data'
    rtb_Subtract_f_idx_1 = 0.00061013F * static_cast<real32_T>
      (sensor.inceptor.ch[0]) + -0.104942039F;

    // Outputs for Enabled SubSystem: '<Root>/Pos_Hold_input_conversion' incorporates:
    //   EnablePort: '<S6>/Enable'
    if (rtb_Compare_lx) {
      // Gain: '<S6>/Gain1'
      rtDW.vb_x_cmd_mps_d = 5.0F * s_lon;

      // Product: '<S348>/v_z_cmd (-1 to 1)' incorporates:
      //   Constant: '<S348>/Double'
      //   Constant: '<S348>/Normalize at Zero'
      //   Sum: '<S348>/Sum'
      rtb_roll_angle_cmd_rad = (rtb_Subtract_f_idx_1 - 0.5F) * 2.0F;

      // Gain: '<S348>/Gain' incorporates:
      //   Constant: '<S348>/Constant1'
      //   Constant: '<S349>/Constant'
      //   Constant: '<S350>/Constant'
      //   Product: '<S348>/Product'
      //   Product: '<S348>/Product1'
      //   RelationalOperator: '<S349>/Compare'
      //   RelationalOperator: '<S350>/Compare'
      //   Sum: '<S348>/Sum1'
      rtDW.Gain = -(static_cast<real32_T>(rtb_roll_angle_cmd_rad >= 0.0F) *
                    rtb_roll_angle_cmd_rad * 2.0F + static_cast<real32_T>
                    (rtb_roll_angle_cmd_rad < 0.0F) * rtb_roll_angle_cmd_rad);

      // Gain: '<S6>/Gain2'
      rtDW.vb_y_cmd_mps_f = 5.0F * rtb_roll;

      // Gain: '<S6>/Gain3'
      rtDW.yaw_rate_cmd_radps_p = 0.524F * rtb_yaw;
    }

    // End of Outputs for SubSystem: '<Root>/Pos_Hold_input_conversion'

    // Outputs for Enabled SubSystem: '<Root>/RTL CONTROLLER' incorporates:
    //   EnablePort: '<S8>/Enable'

    // Logic: '<Root>/motor_armed AND mode_3' incorporates:
    //   Constant: '<S15>/Constant'
    //   RelationalOperator: '<S15>/Compare'
    if (rtb_Switch_lt && (rtb_DataTypeConversion6 == 3)) {
      // Sqrt: '<S356>/Sqrt' incorporates:
      //   Inport: '<Root>/Navigation Filter Data'
      //   Product: '<S356>/MatrixMultiply'
      //   SignalConversion generated from: '<S8>/Bus Selector2'
      //   Sum: '<S356>/Subtract'
      rtb_roll_angle_cmd_rad = std::sqrt((0.0F - nav.ned_pos_m[0]) * (0.0F -
        nav.ned_pos_m[0]) + (0.0F - nav.ned_pos_m[1]) * (0.0F - nav.ned_pos_m[1]));

      // Saturate: '<S355>/Saturation'
      if (rtb_roll_angle_cmd_rad > 20.0F) {
        rtb_Switch_i = 20.0F;
      } else if (rtb_roll_angle_cmd_rad < 0.0F) {
        rtb_Switch_i = 0.0F;
      } else {
        rtb_Switch_i = rtb_roll_angle_cmd_rad;
      }

      // End of Saturate: '<S355>/Saturation'

      // Product: '<S397>/PProd Out' incorporates:
      //   Constant: '<S355>/Constant3'
      s_lat = rtb_Switch_i * 0.5F;

      // Switch: '<S8>/Switch1' incorporates:
      //   Constant: '<S354>/Constant'
      //   Inport: '<Root>/Navigation Filter Data'
      //   MinMax: '<S8>/Min'
      //   RelationalOperator: '<S354>/Compare'
      if (rtb_roll_angle_cmd_rad <= 10.0F) {
        rtb_roll_angle_cmd_rad = nav.ned_pos_m[2];
      } else if ((-100.0F < nav.ned_pos_m[2]) || rtIsNaNF(nav.ned_pos_m[2])) {
        // MinMax: '<S8>/Min'
        rtb_roll_angle_cmd_rad = -100.0F;
      } else {
        // MinMax: '<S8>/Min' incorporates:
        //   Inport: '<Root>/Navigation Filter Data'
        rtb_roll_angle_cmd_rad = nav.ned_pos_m[2];
      }

      // End of Switch: '<S8>/Switch1'

      // Sum: '<S357>/Sum3' incorporates:
      //   Inport: '<Root>/Navigation Filter Data'
      rtb_roll_angle_cmd_rad -= nav.ned_pos_m[2];

      // Switch: '<S8>/Switch' incorporates:
      //   Abs: '<S357>/Abs'
      //   Constant: '<S411>/Constant'
      //   RelationalOperator: '<S411>/Compare'
      if (std::abs(rtb_roll_angle_cmd_rad) <= 1.5F) {
        // Trigonometry: '<S356>/Atan2' incorporates:
        //   Inport: '<Root>/Navigation Filter Data'
        //   SignalConversion generated from: '<S8>/Bus Selector2'
        //   Sum: '<S356>/Subtract'
        rtb_Gain_h = rt_atan2f_snf(0.0F - nav.ned_pos_m[1], 0.0F -
          nav.ned_pos_m[0]);

        // Switch: '<S400>/Switch2' incorporates:
        //   Constant: '<S355>/Constant1'
        //   RelationalOperator: '<S400>/LowerRelop1'
        if (s_lat > 5.0F) {
          s_lat = 5.0F;
        }

        // End of Switch: '<S400>/Switch2'

        // Product: '<S359>/Product1' incorporates:
        //   Trigonometry: '<S359>/Sin'
        c_lon = s_lat * std::sin(rtb_Gain_h);

        // Product: '<S359>/Product' incorporates:
        //   Trigonometry: '<S359>/Cos'
        rtb_Gain_h = s_lat * std::cos(rtb_Gain_h);

        // Trigonometry: '<S410>/Sin' incorporates:
        //   Inport: '<Root>/Navigation Filter Data'
        s_lat = std::sin(nav.heading_rad);

        // Trigonometry: '<S410>/Cos' incorporates:
        //   Inport: '<Root>/Navigation Filter Data'
        rtb_Cos_f = std::cos(nav.heading_rad);

        // Switch: '<S8>/Switch' incorporates:
        //   Gain: '<S410>/Gain'
        //   Product: '<S360>/Product'
        //   Reshape: '<S410>/Reshape'
        //   SignalConversion generated from: '<S360>/Product'
        rtDW.Switch[0] = 0.0F;
        rtDW.Switch[0] += rtb_Cos_f * rtb_Gain_h;
        rtDW.Switch[0] += s_lat * c_lon;
        rtDW.Switch[1] = 0.0F;
        rtDW.Switch[1] += -s_lat * rtb_Gain_h;
        rtDW.Switch[1] += rtb_Cos_f * c_lon;
      } else {
        // Switch: '<S8>/Switch'
        rtDW.Switch[0] = 0.0F;
        rtDW.Switch[1] = 0.0F;
      }

      // End of Switch: '<S8>/Switch'

      // Switch: '<S452>/Switch2' incorporates:
      //   Constant: '<S357>/Constant1'
      //   RelationalOperator: '<S452>/LowerRelop1'
      //   RelationalOperator: '<S452>/UpperRelop'
      //   Switch: '<S452>/Switch'
      if (rtb_roll_angle_cmd_rad > 1.0F) {
        // Switch: '<S452>/Switch2'
        rtDW.Switch2_h = 1.0F;
      } else if (rtb_roll_angle_cmd_rad < -2.0F) {
        // Switch: '<S452>/Switch' incorporates:
        //   Switch: '<S452>/Switch2'
        rtDW.Switch2_h = -2.0F;
      } else {
        // Switch: '<S452>/Switch2' incorporates:
        //   Switch: '<S452>/Switch'
        rtDW.Switch2_h = rtb_roll_angle_cmd_rad;
      }

      // End of Switch: '<S452>/Switch2'

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
    if (rtb_Switch_lt && (rtb_DataTypeConversion6 == 4)) {
      // SignalConversion generated from: '<S3>/land_cmd' incorporates:
      //   Gain: '<S7>/Gain1'
      rtDW.vb_x_cmd_mps_o = 5.0F * s_lon;

      // Switch: '<S182>/Switch' incorporates:
      //   Constant: '<S183>/Constant'
      //   Inport: '<Root>/Navigation Filter Data'
      //   RelationalOperator: '<S183>/Compare'
      if (nav.ned_pos_m[2] <= 10.0F) {
        // Switch: '<S182>/Switch' incorporates:
        //   Constant: '<S182>/Constant1'
        rtDW.Switch_h = 0.3F;
      } else {
        // Switch: '<S182>/Switch' incorporates:
        //   Constant: '<S182>/Constant'
        rtDW.Switch_h = 1.0F;
      }

      // End of Switch: '<S182>/Switch'

      // SignalConversion generated from: '<S3>/land_cmd' incorporates:
      //   Gain: '<S7>/Gain2'
      rtDW.vb_y_cmd_mps_l = 5.0F * rtb_roll;

      // SignalConversion generated from: '<S3>/land_cmd' incorporates:
      //   Gain: '<S7>/Gain3'
      rtDW.yaw_rate_cmd_radps_c53 = 0.524F * rtb_yaw;
    }

    // End of Logic: '<Root>/motor_armed AND mode_4'
    // End of Outputs for SubSystem: '<Root>/Pos_Hold_input_conversion2'
    // End of Outputs for SubSystem: '<Root>/LAND CONTROLLER'

    // Switch generated from: '<Root>/Switch1' incorporates:
    //   Logic: '<Root>/NOT1'
    if (!rtb_Compare_lx) {
      // MultiPortSwitch generated from: '<Root>/Multiport Switch'
      switch (rtb_DataTypeConversion6) {
       case 2:
        rtb_Gain_h = rtDW.Switch2;
        c_lon = rtDW.vb_xy[0];
        s_lat = rtDW.vb_xy[1];
        rtb_Cos_f = rtDW.Saturation;
        break;

       case 3:
        rtb_Gain_h = rtDW.Switch2_h;
        c_lon = rtDW.Switch[0];
        s_lat = rtDW.Switch[1];
        rtb_Cos_f = rtDW.yaw_rate_cmd_radps_c;
        break;

       default:
        rtb_Gain_h = rtDW.Switch_h;
        c_lon = rtDW.vb_x_cmd_mps_o;
        s_lat = rtDW.vb_y_cmd_mps_l;
        rtb_Cos_f = rtDW.yaw_rate_cmd_radps_c53;
        break;
      }

      // End of MultiPortSwitch generated from: '<Root>/Multiport Switch'
    } else {
      rtb_Gain_h = rtDW.Gain;
      c_lon = rtDW.vb_x_cmd_mps_d;
      s_lat = rtDW.vb_y_cmd_mps_f;
      rtb_Cos_f = rtDW.yaw_rate_cmd_radps_p;
    }

    // End of Switch generated from: '<Root>/Switch1'

    // Outputs for Enabled SubSystem: '<Root>/POS_HOLD CONTROLLER' incorporates:
    //   EnablePort: '<S5>/Enable'

    // Logic: '<Root>/motor_armed AND mode_1' incorporates:
    //   Constant: '<S14>/Constant'
    //   RelationalOperator: '<S14>/Compare'
    if (rtb_Switch_lt && (rtb_DataTypeConversion6 > 0)) {
      // Trigonometry: '<S186>/Cos' incorporates:
      //   Inport: '<Root>/Navigation Filter Data'
      rtb_roll_angle_cmd_rad = std::cos(nav.heading_rad);

      // Trigonometry: '<S186>/Sin' incorporates:
      //   Inport: '<Root>/Navigation Filter Data'
      rtb_DataTypeConversion_h = std::sin(nav.heading_rad);

      // Product: '<S184>/Product' incorporates:
      //   Gain: '<S186>/Gain'
      //   Inport: '<Root>/Navigation Filter Data'
      //   Reshape: '<S186>/Reshape'
      rtb_Tsamp_a = -rtb_DataTypeConversion_h * nav.ned_vel_mps[0] +
        rtb_roll_angle_cmd_rad * nav.ned_vel_mps[1];

      // Sum: '<S187>/Sum' incorporates:
      //   Inport: '<Root>/Navigation Filter Data'
      //   Product: '<S184>/Product'
      //   Reshape: '<S186>/Reshape'
      rtb_roll_angle_cmd_rad = c_lon - (rtb_roll_angle_cmd_rad *
        nav.ned_vel_mps[0] + rtb_DataTypeConversion_h * nav.ned_vel_mps[1]);

      // SampleTimeMath: '<S220>/Tsamp' incorporates:
      //   Constant: '<S187>/Constant2'
      //   Product: '<S217>/DProd Out'
      //
      //  About '<S220>/Tsamp':
      //   y = u * K where K = 1 / ( w * Ts )
      rtb_DataTypeConversion_h = rtb_roll_angle_cmd_rad * 0.1F * 100.0F;

      // Sum: '<S234>/Sum' incorporates:
      //   Constant: '<S187>/Constant'
      //   Delay: '<S218>/UD'
      //   DiscreteIntegrator: '<S225>/Integrator'
      //   Product: '<S230>/PProd Out'
      //   Sum: '<S218>/Diff'
      c_lon = (rtb_roll_angle_cmd_rad * 0.5F + rtDW.Integrator_DSTATE_c) +
        (rtb_DataTypeConversion_h - rtDW.UD_DSTATE_k);

      // Saturate: '<S187>/Saturation'
      if (c_lon > 0.523F) {
        // Gain: '<S187>/Gain'
        rtDW.Gain_a = -0.523F;
      } else if (c_lon < -0.523F) {
        // Gain: '<S187>/Gain'
        rtDW.Gain_a = 0.523F;
      } else {
        // Gain: '<S187>/Gain'
        rtDW.Gain_a = -c_lon;
      }

      // End of Saturate: '<S187>/Saturation'

      // Gain: '<S214>/ZeroGain'
      rtb_Reshape_h_idx_0 = 0.0F * c_lon;

      // DeadZone: '<S216>/DeadZone'
      if (c_lon >= (rtMinusInfF)) {
        c_lon = 0.0F;
      }

      // End of DeadZone: '<S216>/DeadZone'

      // Product: '<S222>/IProd Out' incorporates:
      //   Constant: '<S187>/Constant1'
      rtb_roll_angle_cmd_rad *= 0.01F;

      // Signum: '<S214>/SignPreIntegrator'
      if (rtb_roll_angle_cmd_rad < 0.0F) {
        rtb_Switch_i = -1.0F;
      } else if (rtb_roll_angle_cmd_rad > 0.0F) {
        rtb_Switch_i = 1.0F;
      } else if (rtb_roll_angle_cmd_rad == 0.0F) {
        rtb_Switch_i = 0.0F;
      } else {
        rtb_Switch_i = (rtNaNF);
      }

      // End of Signum: '<S214>/SignPreIntegrator'

      // Switch: '<S214>/Switch' incorporates:
      //   Constant: '<S214>/Constant1'
      //   DataTypeConversion: '<S214>/DataTypeConv2'
      //   Logic: '<S214>/AND3'
      //   RelationalOperator: '<S214>/Equal1'
      //   RelationalOperator: '<S214>/NotEqual'
      if ((rtb_Reshape_h_idx_0 != c_lon) && (0 == static_cast<int8_T>
           (rtb_Switch_i))) {
        rtb_Reshape_h_idx_0 = 0.0F;
      } else {
        rtb_Reshape_h_idx_0 = rtb_roll_angle_cmd_rad;
      }

      // End of Switch: '<S214>/Switch'

      // Sum: '<S188>/Sum'
      rtb_roll_angle_cmd_rad = s_lat - rtb_Tsamp_a;

      // SampleTimeMath: '<S273>/Tsamp' incorporates:
      //   Constant: '<S188>/Constant2'
      //   Product: '<S270>/DProd Out'
      //
      //  About '<S273>/Tsamp':
      //   y = u * K where K = 1 / ( w * Ts )
      rtb_Tsamp_a = rtb_roll_angle_cmd_rad * 0.1F * 100.0F;

      // Sum: '<S287>/Sum' incorporates:
      //   Constant: '<S188>/Constant'
      //   Delay: '<S271>/UD'
      //   DiscreteIntegrator: '<S278>/Integrator'
      //   Product: '<S283>/PProd Out'
      //   Sum: '<S271>/Diff'
      s_lat = (rtb_roll_angle_cmd_rad * 0.5F + rtDW.Integrator_DSTATE_n) +
        (rtb_Tsamp_a - rtDW.UD_DSTATE_a);

      // Product: '<S275>/IProd Out' incorporates:
      //   Constant: '<S188>/Constant1'
      rtb_roll_angle_cmd_rad *= 0.01F;

      // DeadZone: '<S269>/DeadZone'
      if (s_lat >= (rtMinusInfF)) {
        c_lon = 0.0F;
      } else {
        c_lon = (rtNaNF);
      }

      // End of DeadZone: '<S269>/DeadZone'

      // Signum: '<S267>/SignPreIntegrator'
      if (rtb_roll_angle_cmd_rad < 0.0F) {
        rtb_Switch_i = -1.0F;
      } else if (rtb_roll_angle_cmd_rad > 0.0F) {
        rtb_Switch_i = 1.0F;
      } else if (rtb_roll_angle_cmd_rad == 0.0F) {
        rtb_Switch_i = 0.0F;
      } else {
        rtb_Switch_i = (rtNaNF);
      }

      // End of Signum: '<S267>/SignPreIntegrator'

      // Switch: '<S267>/Switch' incorporates:
      //   Constant: '<S267>/Constant1'
      //   DataTypeConversion: '<S267>/DataTypeConv2'
      //   Gain: '<S267>/ZeroGain'
      //   Logic: '<S267>/AND3'
      //   RelationalOperator: '<S267>/Equal1'
      //   RelationalOperator: '<S267>/NotEqual'
      if ((0.0F * s_lat != c_lon) && (0 == static_cast<int8_T>(rtb_Switch_i))) {
        rtb_Switch_i = 0.0F;
      } else {
        rtb_Switch_i = rtb_roll_angle_cmd_rad;
      }

      // End of Switch: '<S267>/Switch'

      // Saturate: '<S188>/Saturation'
      if (s_lat > 0.523F) {
        // Saturate: '<S188>/Saturation'
        rtDW.Saturation_n = 0.523F;
      } else if (s_lat < -0.523F) {
        // Saturate: '<S188>/Saturation'
        rtDW.Saturation_n = -0.523F;
      } else {
        // Saturate: '<S188>/Saturation'
        rtDW.Saturation_n = s_lat;
      }

      // End of Saturate: '<S188>/Saturation'

      // SignalConversion generated from: '<S5>/Command out'
      rtDW.yaw_rate_cmd_radps_c5 = rtb_Cos_f;

      // Sum: '<S185>/Sum' incorporates:
      //   Inport: '<Root>/Navigation Filter Data'
      rtb_roll_angle_cmd_rad = rtb_Gain_h - nav.ned_vel_mps[2];

      // SampleTimeMath: '<S326>/Tsamp' incorporates:
      //   Constant: '<S185>/D_vz'
      //   Product: '<S323>/DProd Out'
      //
      //  About '<S326>/Tsamp':
      //   y = u * K where K = 1 / ( w * Ts )
      s_lat = rtb_roll_angle_cmd_rad * 0.005F * 100.0F;

      // Sum: '<S340>/Sum' incorporates:
      //   Constant: '<S185>/P_vz'
      //   Delay: '<S324>/UD'
      //   DiscreteIntegrator: '<S331>/Integrator'
      //   Product: '<S336>/PProd Out'
      //   Sum: '<S324>/Diff'
      c_lon = (rtb_roll_angle_cmd_rad * 0.09F + rtDW.Integrator_DSTATE_a) +
        (s_lat - rtDW.UD_DSTATE_h);

      // Saturate: '<S185>/Saturation' incorporates:
      //   Constant: '<S185>/Constant2'
      //   Gain: '<S185>/Gain'
      //   Sum: '<S185>/Sum1'
      if (-c_lon + 0.6724F > 1.0F) {
        // Saturate: '<S185>/Saturation'
        rtDW.Saturation_k = 1.0F;
      } else if (-c_lon + 0.6724F < 0.0F) {
        // Saturate: '<S185>/Saturation'
        rtDW.Saturation_k = 0.0F;
      } else {
        // Saturate: '<S185>/Saturation'
        rtDW.Saturation_k = -c_lon + 0.6724F;
      }

      // End of Saturate: '<S185>/Saturation'

      // Gain: '<S320>/ZeroGain'
      rtb_Gain_h = 0.0F * c_lon;

      // DeadZone: '<S322>/DeadZone'
      if (c_lon >= (rtMinusInfF)) {
        c_lon = 0.0F;
      }

      // End of DeadZone: '<S322>/DeadZone'

      // Product: '<S328>/IProd Out' incorporates:
      //   Constant: '<S185>/I_vz'
      rtb_roll_angle_cmd_rad *= 0.05F;

      // Update for DiscreteIntegrator: '<S225>/Integrator'
      rtDW.Integrator_DSTATE_c += 0.01F * rtb_Reshape_h_idx_0;

      // Update for Delay: '<S218>/UD'
      rtDW.UD_DSTATE_k = rtb_DataTypeConversion_h;

      // Update for DiscreteIntegrator: '<S278>/Integrator'
      rtDW.Integrator_DSTATE_n += 0.01F * rtb_Switch_i;

      // Update for Delay: '<S271>/UD'
      rtDW.UD_DSTATE_a = rtb_Tsamp_a;

      // Signum: '<S320>/SignPreIntegrator'
      if (rtb_roll_angle_cmd_rad < 0.0F) {
        rtb_Switch_i = -1.0F;
      } else if (rtb_roll_angle_cmd_rad > 0.0F) {
        rtb_Switch_i = 1.0F;
      } else if (rtb_roll_angle_cmd_rad == 0.0F) {
        rtb_Switch_i = 0.0F;
      } else {
        rtb_Switch_i = (rtNaNF);
      }

      // End of Signum: '<S320>/SignPreIntegrator'

      // Switch: '<S320>/Switch' incorporates:
      //   Constant: '<S320>/Constant1'
      //   DataTypeConversion: '<S320>/DataTypeConv2'
      //   Logic: '<S320>/AND3'
      //   RelationalOperator: '<S320>/Equal1'
      //   RelationalOperator: '<S320>/NotEqual'
      if ((rtb_Gain_h != c_lon) && (0 == static_cast<int8_T>(rtb_Switch_i))) {
        rtb_roll_angle_cmd_rad = 0.0F;
      }

      // End of Switch: '<S320>/Switch'

      // Update for DiscreteIntegrator: '<S331>/Integrator'
      rtDW.Integrator_DSTATE_a += 0.01F * rtb_roll_angle_cmd_rad;

      // Update for Delay: '<S324>/UD'
      rtDW.UD_DSTATE_h = s_lat;
    }

    // End of Logic: '<Root>/motor_armed AND mode_1'
    // End of Outputs for SubSystem: '<Root>/POS_HOLD CONTROLLER'

    // Logic: '<Root>/motor_armed AND mode_0' incorporates:
    //   Constant: '<S16>/Constant'
    //   RelationalOperator: '<S16>/Compare'
    rtb_Compare_lx = (rtb_Switch_lt && (rtb_DataTypeConversion6 <= 0));

    // Outputs for Enabled SubSystem: '<Root>/Stab_input_conversion' incorporates:
    //   EnablePort: '<S9>/Enable'
    if (rtb_Compare_lx) {
      // Gain: '<S9>/Gain'
      rtDW.throttle_cc = rtb_Subtract_f_idx_1;

      // Gain: '<S9>/Gain1'
      rtDW.pitch_angle_cmd_rad = -0.523F * s_lon;

      // Gain: '<S9>/Gain2'
      rtDW.roll_angle_cmd_rad = 0.52F * rtb_roll;

      // Gain: '<S9>/Gain3'
      rtDW.yaw_rate_cmd_radps = 0.524F * rtb_yaw;
    }

    // End of Outputs for SubSystem: '<Root>/Stab_input_conversion'

    // Logic: '<Root>/NOT'
    rtb_Compare_lx = !rtb_Compare_lx;

    // Switch generated from: '<Root>/Switch'
    if (rtb_Compare_lx) {
      rtb_Tsamp_a = rtDW.Saturation_k;
      rtb_roll_angle_cmd_rad = rtDW.Saturation_n;
    } else {
      rtb_Tsamp_a = rtDW.throttle_cc;
      rtb_roll_angle_cmd_rad = rtDW.roll_angle_cmd_rad;
    }

    // Sum: '<S21>/stab_roll_angle_error_calc' incorporates:
    //   Inport: '<Root>/Navigation Filter Data'
    s_lon = rtb_roll_angle_cmd_rad - nav.roll_rad;

    // SampleTimeMath: '<S107>/Tsamp' incorporates:
    //   Constant: '<S21>/Constant2'
    //   Product: '<S104>/DProd Out'
    //
    //  About '<S107>/Tsamp':
    //   y = u * K where K = 1 / ( w * Ts )
    rtb_roll = s_lon * 0.02F * 100.0F;

    // Sum: '<S121>/Sum' incorporates:
    //   Constant: '<S21>/Constant'
    //   Delay: '<S105>/UD'
    //   DiscreteIntegrator: '<S112>/Integrator'
    //   Product: '<S117>/PProd Out'
    //   Sum: '<S105>/Diff'
    rtb_yaw = (s_lon * 0.04F + rtDW.Integrator_DSTATE) + (rtb_roll -
      rtDW.UD_DSTATE);

    // Saturate: '<S21>/stab_roll_rate_saturation'
    if (rtb_yaw > 1.0F) {
      rtb_Switch_i = 1.0F;
    } else if (rtb_yaw < -1.0F) {
      rtb_Switch_i = -1.0F;
    } else {
      rtb_Switch_i = rtb_yaw;
    }

    // End of Saturate: '<S21>/stab_roll_rate_saturation'

    // Switch generated from: '<Root>/Switch'
    if (rtb_Compare_lx) {
      rtb_pitch_angle_cmd_rad = rtDW.Gain_a;
    } else {
      rtb_pitch_angle_cmd_rad = rtDW.pitch_angle_cmd_rad;
    }

    // Sum: '<S20>/stab_pitch_angle_error_calc' incorporates:
    //   Inport: '<Root>/Navigation Filter Data'
    rtb_Subtract_f_idx_1 = rtb_pitch_angle_cmd_rad - nav.pitch_rad;

    // SampleTimeMath: '<S54>/Tsamp' incorporates:
    //   Constant: '<S20>/Constant2'
    //   Product: '<S51>/DProd Out'
    //
    //  About '<S54>/Tsamp':
    //   y = u * K where K = 1 / ( w * Ts )
    rtb_Gain_h = rtb_Subtract_f_idx_1 * 0.02F * 100.0F;

    // Sum: '<S68>/Sum' incorporates:
    //   Constant: '<S20>/Constant'
    //   Delay: '<S52>/UD'
    //   DiscreteIntegrator: '<S59>/Integrator'
    //   Product: '<S64>/PProd Out'
    //   Sum: '<S52>/Diff'
    rtb_Cos_f = (rtb_Subtract_f_idx_1 * 0.04F + rtDW.Integrator_DSTATE_l) +
      (rtb_Gain_h - rtDW.UD_DSTATE_f);

    // Saturate: '<S20>/stab_pitch_rate_saturation'
    if (rtb_Cos_f > 1.0F) {
      rtb_stab_pitch_rate_saturation = 1.0F;
    } else if (rtb_Cos_f < -1.0F) {
      rtb_stab_pitch_rate_saturation = -1.0F;
    } else {
      rtb_stab_pitch_rate_saturation = rtb_Cos_f;
    }

    // End of Saturate: '<S20>/stab_pitch_rate_saturation'

    // Switch generated from: '<Root>/Switch'
    if (rtb_Compare_lx) {
      c_lon = rtDW.yaw_rate_cmd_radps_c5;
    } else {
      c_lon = rtDW.yaw_rate_cmd_radps;
    }

    // Sum: '<S22>/stab_yaw_rate_error_calc' incorporates:
    //   Inport: '<Root>/Navigation Filter Data'
    s_lat = c_lon - nav.gyro_radps[2];

    // SampleTimeMath: '<S160>/Tsamp' incorporates:
    //   Constant: '<S22>/Constant2'
    //   Product: '<S157>/DProd Out'
    //
    //  About '<S160>/Tsamp':
    //   y = u * K where K = 1 / ( w * Ts )
    rtb_DataTypeConversion_h = s_lat * 0.02F * 100.0F;

    // Sum: '<S174>/Sum' incorporates:
    //   Constant: '<S22>/Constant'
    //   Delay: '<S158>/UD'
    //   DiscreteIntegrator: '<S165>/Integrator'
    //   Product: '<S170>/PProd Out'
    //   Sum: '<S158>/Diff'
    rtb_Reshape_h_idx_0 = (s_lat * 0.5F + rtDW.Integrator_DSTATE_b) +
      (rtb_DataTypeConversion_h - rtDW.UD_DSTATE_m);

    // Switch: '<Root>/switch_motor_out'
    if (rtb_Switch_lt) {
      // Product: '<S4>/Multiply' incorporates:
      //   Math: '<S4>/Transpose'
      //   Reshape: '<S4>/Reshape'
      for (idx = 0; idx < 8; idx++) {
        rtb_Reshape_h_tmp = idx << 2;
        rtb_Reshape_h_0 = rtConstB.Transpose[rtb_Reshape_h_tmp + 3] *
          rtb_Reshape_h_idx_0 + (rtConstB.Transpose[rtb_Reshape_h_tmp + 2] *
          rtb_stab_pitch_rate_saturation + (rtConstB.Transpose[rtb_Reshape_h_tmp
          + 1] * rtb_Switch_i + rtConstB.Transpose[rtb_Reshape_h_tmp] *
          rtb_Tsamp_a));

        // Saturate: '<S4>/Saturation' incorporates:
        //   Math: '<S4>/Transpose'
        //   Reshape: '<S4>/Reshape'
        if (rtb_Reshape_h_0 <= 0.15F) {
          rtb_switch_motor_out[idx] = 0.15F;
        } else {
          rtb_switch_motor_out[idx] = rtb_Reshape_h_0;
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
    //   BusCreator: '<S462>/Bus Creator'
    //   Constant: '<S2>/consumed_mah'
    //   Constant: '<S2>/current_ma'
    //   Constant: '<S2>/remaining_prcnt'
    //   Constant: '<S2>/remaining_time_s'
    //   Constant: '<S2>/voltage_v'
    //   Constant: '<S462>/Constant'
    //   Constant: '<S462>/Constant1'
    //   Constant: '<S463>/Constant'
    //   DataTypeConversion: '<Root>/Cast To Single'
    //   DataTypeConversion: '<Root>/Data Type Conversion'
    //   DataTypeConversion: '<Root>/Data Type Conversion1'
    //   DataTypeConversion: '<S463>/Data Type Conversion'
    //   Gain: '<Root>/Gain'
    //   Gain: '<S463>/Gain'
    //   Inport: '<Root>/Navigation Filter Data'
    //   SignalConversion generated from: '<S10>/Bus Creator'
    //   SignalConversion generated from: '<S462>/Bus Creator'
    //   Sum: '<S463>/Sum'
    //
    ctrl->motors_enabled = rtb_Switch_lt;
    ctrl->waypoint_reached = rtDW.reached;
    ctrl->mode = rtb_DataTypeConversion6;
    ctrl->throttle_cmd_prcnt = 100.0F * rtb_Tsamp_a;
    ctrl->aux[0] = rtb_Tsamp_a;
    ctrl->aux[1] = rtb_Switch_i;
    ctrl->aux[2] = rtb_roll_angle_cmd_rad;
    ctrl->aux[3] = nav.roll_rad;
    ctrl->aux[4] = rtb_stab_pitch_rate_saturation;
    ctrl->aux[5] = rtb_pitch_angle_cmd_rad;
    ctrl->aux[6] = nav.pitch_rad;
    ctrl->aux[7] = rtb_Reshape_h_idx_0;
    ctrl->aux[8] = c_lon;
    ctrl->aux[9] = nav.gyro_radps[2];
    ctrl->aux[10] = rtb_AND_n;
    ctrl->aux[11] = rtb_AND;
    ctrl->aux[12] = rtDW.cur_target_pos_m[0];
    ctrl->aux[13] = rtDW.cur_target_pos_m[1];
    ctrl->aux[14] = rtDW.cur_target_pos_m[2];
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
    for (idx = 0; idx < 16; idx++) {
      ctrl->sbus.cnt[idx] = 0;
    }

    for (idx = 0; idx < 8; idx++) {
      rtb_roll_angle_cmd_rad = rtb_switch_motor_out[idx];
      ctrl->pwm.cnt[idx] = static_cast<int16_T>(std::floor(1000.0F *
        rtb_roll_angle_cmd_rad + 1000.0));
      ctrl->pwm.cmd[idx] = rtb_roll_angle_cmd_rad;
      ctrl->analog.val[idx] = 0.0F;
    }

    ctrl->battery.voltage_v = 1.0F;
    ctrl->battery.current_ma = 1.0F;
    ctrl->battery.consumed_mah = 1.0F;
    ctrl->battery.remaining_prcnt = 1.0F;
    ctrl->battery.remaining_time_s = 1.0F;

    // End of Outport: '<Root>/VMS Data'

    // Product: '<S162>/IProd Out' incorporates:
    //   Constant: '<S22>/Constant1'
    s_lat *= 0.05F;

    // Gain: '<S48>/ZeroGain'
    c_lon = 0.0F * rtb_Cos_f;

    // DeadZone: '<S50>/DeadZone'
    if (rtb_Cos_f >= (rtMinusInfF)) {
      rtb_Cos_f = 0.0F;
    }

    // End of DeadZone: '<S50>/DeadZone'

    // Product: '<S56>/IProd Out' incorporates:
    //   Constant: '<S20>/Constant1'
    rtb_Subtract_f_idx_1 *= 0.04F;

    // Gain: '<S101>/ZeroGain'
    rtb_Tsamp_a = 0.0F * rtb_yaw;

    // DeadZone: '<S103>/DeadZone'
    if (rtb_yaw >= (rtMinusInfF)) {
      rtb_yaw = 0.0F;
    }

    // End of DeadZone: '<S103>/DeadZone'

    // Product: '<S109>/IProd Out' incorporates:
    //   Constant: '<S21>/Constant1'
    s_lon *= 0.04F;

    // Signum: '<S101>/SignPreIntegrator'
    if (s_lon < 0.0F) {
      rtb_roll_angle_cmd_rad = -1.0F;
    } else if (s_lon > 0.0F) {
      rtb_roll_angle_cmd_rad = 1.0F;
    } else if (s_lon == 0.0F) {
      rtb_roll_angle_cmd_rad = 0.0F;
    } else {
      rtb_roll_angle_cmd_rad = (rtNaNF);
    }

    // End of Signum: '<S101>/SignPreIntegrator'

    // Switch: '<S101>/Switch' incorporates:
    //   Constant: '<S101>/Constant1'
    //   DataTypeConversion: '<S101>/DataTypeConv2'
    //   Logic: '<S101>/AND3'
    //   RelationalOperator: '<S101>/Equal1'
    //   RelationalOperator: '<S101>/NotEqual'
    if ((rtb_Tsamp_a != rtb_yaw) && (0 == static_cast<int8_T>
         (rtb_roll_angle_cmd_rad))) {
      s_lon = 0.0F;
    }

    // End of Switch: '<S101>/Switch'

    // Update for DiscreteIntegrator: '<S112>/Integrator'
    rtDW.Integrator_DSTATE += 0.01F * s_lon;

    // Update for Delay: '<S105>/UD'
    rtDW.UD_DSTATE = rtb_roll;

    // Signum: '<S48>/SignPreIntegrator'
    if (rtb_Subtract_f_idx_1 < 0.0F) {
      rtb_roll_angle_cmd_rad = -1.0F;
    } else if (rtb_Subtract_f_idx_1 > 0.0F) {
      rtb_roll_angle_cmd_rad = 1.0F;
    } else if (rtb_Subtract_f_idx_1 == 0.0F) {
      rtb_roll_angle_cmd_rad = 0.0F;
    } else {
      rtb_roll_angle_cmd_rad = (rtNaNF);
    }

    // End of Signum: '<S48>/SignPreIntegrator'

    // Switch: '<S48>/Switch' incorporates:
    //   Constant: '<S48>/Constant1'
    //   DataTypeConversion: '<S48>/DataTypeConv2'
    //   Logic: '<S48>/AND3'
    //   RelationalOperator: '<S48>/Equal1'
    //   RelationalOperator: '<S48>/NotEqual'
    if ((c_lon != rtb_Cos_f) && (0 == static_cast<int8_T>(rtb_roll_angle_cmd_rad)))
    {
      rtb_Subtract_f_idx_1 = 0.0F;
    }

    // End of Switch: '<S48>/Switch'

    // Update for DiscreteIntegrator: '<S59>/Integrator'
    rtDW.Integrator_DSTATE_l += 0.01F * rtb_Subtract_f_idx_1;

    // Update for Delay: '<S52>/UD'
    rtDW.UD_DSTATE_f = rtb_Gain_h;

    // DeadZone: '<S156>/DeadZone'
    if (rtb_Reshape_h_idx_0 >= (rtMinusInfF)) {
      rtb_roll_angle_cmd_rad = 0.0F;
    } else {
      rtb_roll_angle_cmd_rad = (rtNaNF);
    }

    // End of DeadZone: '<S156>/DeadZone'

    // Signum: '<S154>/SignPreIntegrator'
    if (s_lat < 0.0F) {
      c_lon = -1.0F;
    } else if (s_lat > 0.0F) {
      c_lon = 1.0F;
    } else if (s_lat == 0.0F) {
      c_lon = 0.0F;
    } else {
      c_lon = (rtNaNF);
    }

    // End of Signum: '<S154>/SignPreIntegrator'

    // Switch: '<S154>/Switch' incorporates:
    //   Constant: '<S154>/Constant1'
    //   DataTypeConversion: '<S154>/DataTypeConv2'
    //   Gain: '<S154>/ZeroGain'
    //   Logic: '<S154>/AND3'
    //   RelationalOperator: '<S154>/Equal1'
    //   RelationalOperator: '<S154>/NotEqual'
    if ((0.0F * rtb_Reshape_h_idx_0 != rtb_roll_angle_cmd_rad) && (0 ==
         static_cast<int8_T>(c_lon))) {
      s_lat = 0.0F;
    }

    // End of Switch: '<S154>/Switch'

    // Update for DiscreteIntegrator: '<S165>/Integrator'
    rtDW.Integrator_DSTATE_b += 0.01F * s_lat;

    // Update for Delay: '<S158>/UD'
    rtDW.UD_DSTATE_m = rtb_DataTypeConversion_h;
  }

  // Model initialize function
  void Autocode::initialize()
  {
    // Registration code

    // initialize non-finites
    rt_InitInfAndNaN(sizeof(real_T));

    // SystemInitialize for Enabled SubSystem: '<Root>/WAYPOINT CONTROLLER'
    // InitializeConditions for UnitDelay: '<S631>/Delay Input1'
    //
    //  Block description for '<S631>/Delay Input1':
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
