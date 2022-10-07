//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// File: autocode.h
//
// Code generated for Simulink model 'super'.
//
// Model version                  : 4.129
// Simulink Coder version         : 9.7 (R2022a) 13-Nov-2021
// C/C++ source code generated on : Fri Oct  7 08:30:17 2022
//
#ifndef RTW_HEADER_autocode_h_
#define RTW_HEADER_autocode_h_
#include "rtwtypes.h"
#include "flight/global_defs.h"
#include <stddef.h>
#include "zero_crossing_types.h"

// Model Code Variants
#ifndef DEFINED_TYPEDEF_FOR_struct_bJj7DeK7VQpDAMomUA5NpE_
#define DEFINED_TYPEDEF_FOR_struct_bJj7DeK7VQpDAMomUA5NpE_

struct struct_bJj7DeK7VQpDAMomUA5NpE
{
    real32_T ve_z_cmd_mps;
    real32_T vb_x_cmd_mps;
    real32_T vb_y_cmd_mps;
    real32_T yaw_rate_cmd_radps;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_struct_zAK2rPkfaw9DhLoh4VRxkC_
#define DEFINED_TYPEDEF_FOR_struct_zAK2rPkfaw9DhLoh4VRxkC_

struct struct_zAK2rPkfaw9DhLoh4VRxkC
{
    real32_T throttle_cc;
    real32_T pitch_angle_cmd_rad;
    real32_T roll_angle_cmd_rad;
    real32_T yaw_rate_cmd_radps;
};

#endif

#ifndef DEFINED_TYPEDEF_FOR_struct_HxmtTSEWrXcusUFt5FJmmH_
#define DEFINED_TYPEDEF_FOR_struct_HxmtTSEWrXcusUFt5FJmmH_

struct struct_HxmtTSEWrXcusUFt5FJmmH
{
    real32_T cur_target_pos_m[3];
    int16_T current_waypoint;
    boolean_T enable;
};

#endif

extern "C" {
    static real_T rtGetInf(void);
    static real32_T rtGetInfF(void);
    static real_T rtGetMinusInf(void);
    static real32_T rtGetMinusInfF(void);
}                                      // extern "C"
    extern "C"
{
    static real_T rtGetNaN(void);
    static real32_T rtGetNaNF(void);
}                                      // extern "C"

extern "C" {
    extern real_T rtInf;
    extern real_T rtMinusInf;
    extern real_T rtNaN;
    extern real32_T rtInfF;
    extern real32_T rtMinusInfF;
    extern real32_T rtNaNF;
    static void rt_InitInfAndNaN(size_t realSize);
    static boolean_T rtIsInf(real_T value);
    static boolean_T rtIsInfF(real32_T value);
    static boolean_T rtIsNaN(real_T value);
    static boolean_T rtIsNaNF(real32_T value);
    struct BigEndianIEEEDouble {
        struct {
            uint32_T wordH;
            uint32_T wordL;
        } words;
    };

    struct LittleEndianIEEEDouble {
        struct {
            uint32_T wordL;
            uint32_T wordH;
        } words;
    };

    struct IEEESingle {
        union {
            real32_T wordLreal;
            uint32_T wordLuint;
        } wordL;
    };
}                                      // extern "C"
    // Class declaration for model super
    namespace bfs
{
    class Autocode
    {
        // public data and function members
      public:
        // Block signals and states (default storage) for system '<Root>'
        struct D_Work {
            real_T motor_arm_ramp_integrator_DSTAT;// '<S23>/motor_arm_ramp_integrator' 
            real_T UnitDelay_DSTATE;   // '<S692>/Unit Delay'
            real_T UnitDelay_DSTATE_m; // '<S685>/Unit Delay'
            real32_T cur_target_pos_m[3];
            real32_T cur_target_pos[3];// '<S474>/determine_current_tar_pos'
            real32_T prev_target_pos[3];// '<S647>/determine_prev_tar_pos'
            real32_T vb_xy[2];         // '<S484>/Product'
            real32_T Switch[2];        // '<S8>/Switch'
            real32_T DiscreteTimeIntegrator;// '<S705>/Discrete-Time Integrator' 
            real32_T cur_target_heading_rad;// '<S474>/determine_target'
            real32_T max_v_z_mps;      // '<S474>/determine_target'
            real32_T max_v_hor_mps;    // '<S474>/determine_target'
            real32_T Switch2;          // '<S580>/Switch2'
            real32_T Saturation;       // '<S590>/Saturation'
            real32_T throttle_cc;
            // '<S9>/BusConversion_InsertedFor_angle_ctrl_input_at_inport_0'
            real32_T pitch_angle_cmd_rad;// '<S9>/Gain1'
            real32_T roll_angle_cmd_rad;// '<S9>/Gain2'
            real32_T yaw_rate_cmd_radps;// '<S9>/Gain3'
            real32_T Switch2_h;        // '<S461>/Switch2'
            real32_T yaw_rate_cmd_radps_c;// '<S8>/Constant3'
            real32_T vb_x_cmd_mps;     // '<S7>/Gain1'
            real32_T vb_y_cmd_mps;     // '<S7>/Gain2'
            real32_T yaw_rate_cmd_radps_a;// '<S7>/Gain3'
            real32_T vb_x_cmd_mps_d;   // '<S6>/Gain1'
            real32_T Gain;             // '<S357>/Gain'
            real32_T vb_y_cmd_mps_f;   // '<S6>/Gain2'
            real32_T yaw_rate_cmd_radps_p;// '<S6>/Gain3'
            real32_T Gain_a;           // '<S196>/Gain'
            real32_T Saturation_n;     // '<S197>/Saturation'
            real32_T yaw_rate_cmd_radps_c5;
                    // '<S5>/BusConversion_InsertedFor_Command out_at_inport_0'
            real32_T Saturation_d;     // '<S194>/Saturation'
            real32_T vb_x_cmd_mps_o;
                       // '<S3>/BusConversion_InsertedFor_land_cmd_at_inport_0'
            real32_T Switch_h;         // '<S190>/Switch'
            real32_T vb_y_cmd_mps_l;
                       // '<S3>/BusConversion_InsertedFor_land_cmd_at_inport_0'
            real32_T yaw_rate_cmd_radps_c53;
                       // '<S3>/BusConversion_InsertedFor_land_cmd_at_inport_0'
            real32_T Integrator_DSTATE;// '<S118>/Integrator'
            real32_T Integrator_DSTATE_l;// '<S64>/Integrator'
            real32_T Integrator_DSTATE_b;// '<S173>/Integrator'
            real32_T UD_DSTATE;        // '<S135>/UD'
            real32_T DiscreteTimeIntegrator_DSTATE;// '<S2>/Discrete-Time Integrator' 
            real32_T DiscreteTimeIntegrator_DSTATE_k;// '<S705>/Discrete-Time Integrator' 
            real32_T Integrator_DSTATE_e;// '<S572>/Integrator'
            real32_T UD_DSTATE_h;      // '<S591>/UD'
            real32_T Integrator_DSTATE_bm;// '<S627>/Integrator'
            real32_T Integrator_DSTATE_c;// '<S234>/Integrator'
            real32_T UD_DSTATE_k;      // '<S227>/UD'
            real32_T Integrator_DSTATE_n;// '<S287>/Integrator'
            real32_T UD_DSTATE_a;      // '<S280>/UD'
            real32_T Integrator_DSTATE_cr;// '<S340>/Integrator'
            real32_T UD_DSTATE_c;      // '<S333>/UD'
            int16_T current_waypoint;
                           // '<S11>/BusConversion_InsertedFor_dbg_at_inport_0'
            int16_T DelayInput1_DSTATE;// '<S645>/Delay Input1'
            int8_T sub_mode;           // '<S697>/determine_wp_submode'
            int8_T sub_mode_d;         // '<S696>/determine_fast_rtl_mode'
            boolean_T yaw_arm;         // '<S706>/yaw_arm'
            boolean_T Compare;         // '<S693>/Compare'
            boolean_T Compare_d;       // '<S686>/Compare'
            boolean_T enable;
            boolean_T reached;         // '<S475>/check_wp_reached'
            boolean_T EnabledSubsystem_MODE;// '<S680>/Enabled Subsystem'
            boolean_T auto_disarm_MODE;// '<S21>/auto_disarm'
            boolean_T disarmmotor_MODE;// '<S675>/disarm motor'
            boolean_T disarmmotor_MODE_k;// '<S681>/disarm motor'
        };

        // Zero-crossing (trigger) state
        struct PrevZCSigStates {
            ZCSigState manual_arming_Trig_ZCE;// '<S680>/manual_arming'
        };

        // Invariant block signals (default storage)
        struct ConstBlockIO {
            real32_T Transpose[32];    // '<S4>/Transpose'
            real32_T ramp_time_intergratorsignal;// '<S23>/ramp_time_intergrator signal' 
            real32_T Gain1;            // '<S23>/Gain1'
            real32_T Gain;             // '<S366>/Gain'
        };

        // Constant parameters (default storage)
        struct ConstParam {
            // Expression: [172, 172, 172, 172, 172, 172]
            //  Referenced by: '<S13>/Constant'

            real_T Constant_Value_f[6];

            // Computed Parameter: Constant_Value_i
            //  Referenced by: '<S22>/Constant'

            real32_T Constant_Value_i[8];
        };

        // model initialize function
        void initialize();

        // model step function
        void Run(const SysData &sys, const SensorData &sensor, const NavData &
                 nav, const TelemData &telem, VmsData *ctrl);

        // Constructor
        Autocode();

        // Destructor
        ~Autocode();

        // private data and function members
      private:
        // Block states
        D_Work rtDWork;

        // Triggered events
        PrevZCSigStates rtPrevZCSigState;

        // private member function(s) for subsystem '<S656>/remap'
        static void remap(real32_T rtu_raw_in, real32_T rtu_in_min, real32_T
                          rtu_in_max, real32_T rtu_out_min, real32_T rtu_out_max,
                          real32_T *rty_norm_out);

        // private member function(s) for subsystem '<Root>'
        void cosd(real32_T *x);
        void sind(real32_T *x);
        void lla_to_ECEF(const real32_T lla[3], real32_T ecef_pos[3]);
    };
}

extern const bfs::Autocode::ConstBlockIO rtConstB;// constant block i/o

// Constant parameters (default storage)
extern const bfs::Autocode::ConstParam rtConstP;

//-
//  These blocks were eliminated from the model due to optimizations:
//
//  Block '<S135>/Data Type Duplicate' : Unused code path elimination
//  Block '<S192>/Data Type Duplicate' : Unused code path elimination
//  Block '<S192>/Data Type Propagation' : Unused code path elimination
//  Block '<S227>/DTDup' : Unused code path elimination
//  Block '<S280>/DTDup' : Unused code path elimination
//  Block '<S5>/Scope' : Unused code path elimination
//  Block '<S333>/DTDup' : Unused code path elimination
//  Block '<S361>/Compare' : Unused code path elimination
//  Block '<S361>/Constant' : Unused code path elimination
//  Block '<S362>/Compare' : Unused code path elimination
//  Block '<S362>/Constant' : Unused code path elimination
//  Block '<S360>/Constant' : Unused code path elimination
//  Block '<S360>/Constant1' : Unused code path elimination
//  Block '<S360>/Double' : Unused code path elimination
//  Block '<S360>/Normalize at Zero' : Unused code path elimination
//  Block '<S360>/Product' : Unused code path elimination
//  Block '<S360>/Product1' : Unused code path elimination
//  Block '<S360>/Sum' : Unused code path elimination
//  Block '<S360>/Sum1' : Unused code path elimination
//  Block '<S360>/v_z_cmd (-1 to 1)' : Unused code path elimination
//  Block '<S409>/Data Type Duplicate' : Unused code path elimination
//  Block '<S409>/Data Type Propagation' : Unused code path elimination
//  Block '<S365>/x_pos_tracking' : Unused code path elimination
//  Block '<S365>/y_pos_tracking' : Unused code path elimination
//  Block '<S461>/Data Type Duplicate' : Unused code path elimination
//  Block '<S461>/Data Type Propagation' : Unused code path elimination
//  Block '<S479>/Abs' : Unused code path elimination
//  Block '<S479>/Constant4' : Unused code path elimination
//  Block '<S479>/Constant5' : Unused code path elimination
//  Block '<S479>/Divide' : Unused code path elimination
//  Block '<S524>/Data Type Duplicate' : Unused code path elimination
//  Block '<S524>/Data Type Propagation' : Unused code path elimination
//  Block '<S479>/Subtract' : Unused code path elimination
//  Block '<S483>/Abs' : Unused code path elimination
//  Block '<S534>/Compare' : Unused code path elimination
//  Block '<S534>/Constant' : Unused code path elimination
//  Block '<S483>/Constant' : Unused code path elimination
//  Block '<S483>/Product' : Unused code path elimination
//  Block '<S483>/Sign' : Unused code path elimination
//  Block '<S483>/Subtract' : Unused code path elimination
//  Block '<S483>/Subtract1' : Unused code path elimination
//  Block '<S483>/Switch' : Unused code path elimination
//  Block '<S580>/Data Type Duplicate' : Unused code path elimination
//  Block '<S580>/Data Type Propagation' : Unused code path elimination
//  Block '<S591>/Data Type Duplicate' : Unused code path elimination
//  Block '<S13>/Scope' : Unused code path elimination
//  Block '<S677>/Compare' : Unused code path elimination
//  Block '<S677>/Constant' : Unused code path elimination
//  Block '<S71>/Saturation' : Eliminated Saturate block
//  Block '<S125>/Saturation' : Eliminated Saturate block
//  Block '<S180>/Saturation' : Eliminated Saturate block
//  Block '<Root>/Cast To Single' : Eliminate redundant data type conversion
//  Block '<Root>/Cast To Single1' : Eliminate redundant data type conversion
//  Block '<Root>/Data Type Conversion2' : Eliminate redundant data type conversion
//  Block '<Root>/Data Type Conversion3' : Eliminate redundant data type conversion
//  Block '<Root>/Data Type Conversion4' : Eliminate redundant data type conversion
//  Block '<S241>/Saturation' : Eliminated Saturate block
//  Block '<S294>/Saturation' : Eliminated Saturate block
//  Block '<S347>/Saturation' : Eliminated Saturate block
//  Block '<S471>/Cast To Boolean' : Eliminate redundant data type conversion
//  Block '<S471>/Cast To Boolean1' : Eliminate redundant data type conversion
//  Block '<S471>/Cast To Single' : Eliminate redundant data type conversion
//  Block '<S471>/Cast To Single1' : Eliminate redundant data type conversion
//  Block '<S634>/Saturation' : Eliminated Saturate block
//  Block '<S27>/Constant1' : Unused code path elimination
//  Block '<S28>/Constant1' : Unused code path elimination
//  Block '<S29>/Constant1' : Unused code path elimination
//  Block '<S81>/Constant1' : Unused code path elimination
//  Block '<S82>/Constant1' : Unused code path elimination
//  Block '<S83>/Constant1' : Unused code path elimination
//  Block '<S136>/Constant1' : Unused code path elimination
//  Block '<S137>/Constant1' : Unused code path elimination
//  Block '<S138>/Constant1' : Unused code path elimination


//-
//  The generated code includes comments that allow you to trace directly
//  back to the appropriate location in the model.  The basic format
//  is <system>/block_name, where system is the system number (uniquely
//  assigned by Simulink) and block_name is the name of the block.
//
//  Use the MATLAB hilite_system command to trace the generated code back
//  to the model.  For example,
//
//  hilite_system('<S3>')    - opens system 3
//  hilite_system('<S3>/Kp') - opens and selects block Kp which resides in S3
//
//  Here is the system hierarchy for this model
//
//  '<Root>' : 'super'
//  '<S1>'   : 'super/ANGLE CONTROLLER'
//  '<S2>'   : 'super/Battery data'
//  '<S3>'   : 'super/LAND CONTROLLER'
//  '<S4>'   : 'super/Motor Mixing Algorithm'
//  '<S5>'   : 'super/POS_HOLD CONTROLLER'
//  '<S6>'   : 'super/Pos_Hold_input_conversion'
//  '<S7>'   : 'super/Pos_Hold_input_conversion2'
//  '<S8>'   : 'super/RTL CONTROLLER'
//  '<S9>'   : 'super/Stab_input_conversion'
//  '<S10>'  : 'super/To VMS Data'
//  '<S11>'  : 'super/WAYPOINT CONTROLLER'
//  '<S12>'  : 'super/add_auxilary_cmd'
//  '<S13>'  : 'super/cmd to raw pwm'
//  '<S14>'  : 'super/command selection'
//  '<S15>'  : 'super/compare_to_land'
//  '<S16>'  : 'super/compare_to_pos_hold'
//  '<S17>'  : 'super/compare_to_rtl'
//  '<S18>'  : 'super/compare_to_stab'
//  '<S19>'  : 'super/compare_to_stab1'
//  '<S20>'  : 'super/compare_to_wp'
//  '<S21>'  : 'super/determine arm and mode selection'
//  '<S22>'  : 'super/emergency_stop_system'
//  '<S23>'  : 'super/initial_motor_ramp'
//  '<S24>'  : 'super/ANGLE CONTROLLER/Pitch Controller'
//  '<S25>'  : 'super/ANGLE CONTROLLER/Roll Controller'
//  '<S26>'  : 'super/ANGLE CONTROLLER/Yaw Rate Controller'
//  '<S27>'  : 'super/ANGLE CONTROLLER/Pitch Controller/Gain Selector'
//  '<S28>'  : 'super/ANGLE CONTROLLER/Pitch Controller/Gain Selector1'
//  '<S29>'  : 'super/ANGLE CONTROLLER/Pitch Controller/Gain Selector2'
//  '<S30>'  : 'super/ANGLE CONTROLLER/Pitch Controller/stab_pitch_PID'
//  '<S31>'  : 'super/ANGLE CONTROLLER/Pitch Controller/stab_pitch_PID/Anti-windup'
//  '<S32>'  : 'super/ANGLE CONTROLLER/Pitch Controller/stab_pitch_PID/D Gain'
//  '<S33>'  : 'super/ANGLE CONTROLLER/Pitch Controller/stab_pitch_PID/Filter'
//  '<S34>'  : 'super/ANGLE CONTROLLER/Pitch Controller/stab_pitch_PID/Filter ICs'
//  '<S35>'  : 'super/ANGLE CONTROLLER/Pitch Controller/stab_pitch_PID/I Gain'
//  '<S36>'  : 'super/ANGLE CONTROLLER/Pitch Controller/stab_pitch_PID/Ideal P Gain'
//  '<S37>'  : 'super/ANGLE CONTROLLER/Pitch Controller/stab_pitch_PID/Ideal P Gain Fdbk'
//  '<S38>'  : 'super/ANGLE CONTROLLER/Pitch Controller/stab_pitch_PID/Integrator'
//  '<S39>'  : 'super/ANGLE CONTROLLER/Pitch Controller/stab_pitch_PID/Integrator ICs'
//  '<S40>'  : 'super/ANGLE CONTROLLER/Pitch Controller/stab_pitch_PID/N Copy'
//  '<S41>'  : 'super/ANGLE CONTROLLER/Pitch Controller/stab_pitch_PID/N Gain'
//  '<S42>'  : 'super/ANGLE CONTROLLER/Pitch Controller/stab_pitch_PID/P Copy'
//  '<S43>'  : 'super/ANGLE CONTROLLER/Pitch Controller/stab_pitch_PID/Parallel P Gain'
//  '<S44>'  : 'super/ANGLE CONTROLLER/Pitch Controller/stab_pitch_PID/Reset Signal'
//  '<S45>'  : 'super/ANGLE CONTROLLER/Pitch Controller/stab_pitch_PID/Saturation'
//  '<S46>'  : 'super/ANGLE CONTROLLER/Pitch Controller/stab_pitch_PID/Saturation Fdbk'
//  '<S47>'  : 'super/ANGLE CONTROLLER/Pitch Controller/stab_pitch_PID/Sum'
//  '<S48>'  : 'super/ANGLE CONTROLLER/Pitch Controller/stab_pitch_PID/Sum Fdbk'
//  '<S49>'  : 'super/ANGLE CONTROLLER/Pitch Controller/stab_pitch_PID/Tracking Mode'
//  '<S50>'  : 'super/ANGLE CONTROLLER/Pitch Controller/stab_pitch_PID/Tracking Mode Sum'
//  '<S51>'  : 'super/ANGLE CONTROLLER/Pitch Controller/stab_pitch_PID/Tsamp - Integral'
//  '<S52>'  : 'super/ANGLE CONTROLLER/Pitch Controller/stab_pitch_PID/Tsamp - Ngain'
//  '<S53>'  : 'super/ANGLE CONTROLLER/Pitch Controller/stab_pitch_PID/postSat Signal'
//  '<S54>'  : 'super/ANGLE CONTROLLER/Pitch Controller/stab_pitch_PID/preSat Signal'
//  '<S55>'  : 'super/ANGLE CONTROLLER/Pitch Controller/stab_pitch_PID/Anti-windup/Disc. Clamping Parallel'
//  '<S56>'  : 'super/ANGLE CONTROLLER/Pitch Controller/stab_pitch_PID/Anti-windup/Disc. Clamping Parallel/Dead Zone'
//  '<S57>'  : 'super/ANGLE CONTROLLER/Pitch Controller/stab_pitch_PID/Anti-windup/Disc. Clamping Parallel/Dead Zone/Enabled'
//  '<S58>'  : 'super/ANGLE CONTROLLER/Pitch Controller/stab_pitch_PID/D Gain/Disabled'
//  '<S59>'  : 'super/ANGLE CONTROLLER/Pitch Controller/stab_pitch_PID/Filter/Disabled'
//  '<S60>'  : 'super/ANGLE CONTROLLER/Pitch Controller/stab_pitch_PID/Filter ICs/Disabled'
//  '<S61>'  : 'super/ANGLE CONTROLLER/Pitch Controller/stab_pitch_PID/I Gain/External Parameters'
//  '<S62>'  : 'super/ANGLE CONTROLLER/Pitch Controller/stab_pitch_PID/Ideal P Gain/Passthrough'
//  '<S63>'  : 'super/ANGLE CONTROLLER/Pitch Controller/stab_pitch_PID/Ideal P Gain Fdbk/Disabled'
//  '<S64>'  : 'super/ANGLE CONTROLLER/Pitch Controller/stab_pitch_PID/Integrator/Discrete'
//  '<S65>'  : 'super/ANGLE CONTROLLER/Pitch Controller/stab_pitch_PID/Integrator ICs/Internal IC'
//  '<S66>'  : 'super/ANGLE CONTROLLER/Pitch Controller/stab_pitch_PID/N Copy/Disabled wSignal Specification'
//  '<S67>'  : 'super/ANGLE CONTROLLER/Pitch Controller/stab_pitch_PID/N Gain/Disabled'
//  '<S68>'  : 'super/ANGLE CONTROLLER/Pitch Controller/stab_pitch_PID/P Copy/Disabled'
//  '<S69>'  : 'super/ANGLE CONTROLLER/Pitch Controller/stab_pitch_PID/Parallel P Gain/External Parameters'
//  '<S70>'  : 'super/ANGLE CONTROLLER/Pitch Controller/stab_pitch_PID/Reset Signal/Disabled'
//  '<S71>'  : 'super/ANGLE CONTROLLER/Pitch Controller/stab_pitch_PID/Saturation/Enabled'
//  '<S72>'  : 'super/ANGLE CONTROLLER/Pitch Controller/stab_pitch_PID/Saturation Fdbk/Disabled'
//  '<S73>'  : 'super/ANGLE CONTROLLER/Pitch Controller/stab_pitch_PID/Sum/Sum_PI'
//  '<S74>'  : 'super/ANGLE CONTROLLER/Pitch Controller/stab_pitch_PID/Sum Fdbk/Disabled'
//  '<S75>'  : 'super/ANGLE CONTROLLER/Pitch Controller/stab_pitch_PID/Tracking Mode/Disabled'
//  '<S76>'  : 'super/ANGLE CONTROLLER/Pitch Controller/stab_pitch_PID/Tracking Mode Sum/Passthrough'
//  '<S77>'  : 'super/ANGLE CONTROLLER/Pitch Controller/stab_pitch_PID/Tsamp - Integral/Passthrough'
//  '<S78>'  : 'super/ANGLE CONTROLLER/Pitch Controller/stab_pitch_PID/Tsamp - Ngain/Passthrough'
//  '<S79>'  : 'super/ANGLE CONTROLLER/Pitch Controller/stab_pitch_PID/postSat Signal/Forward_Path'
//  '<S80>'  : 'super/ANGLE CONTROLLER/Pitch Controller/stab_pitch_PID/preSat Signal/Forward_Path'
//  '<S81>'  : 'super/ANGLE CONTROLLER/Roll Controller/Gain Selector'
//  '<S82>'  : 'super/ANGLE CONTROLLER/Roll Controller/Gain Selector1'
//  '<S83>'  : 'super/ANGLE CONTROLLER/Roll Controller/Gain Selector2'
//  '<S84>'  : 'super/ANGLE CONTROLLER/Roll Controller/stab_roll_PID'
//  '<S85>'  : 'super/ANGLE CONTROLLER/Roll Controller/stab_roll_PID/Anti-windup'
//  '<S86>'  : 'super/ANGLE CONTROLLER/Roll Controller/stab_roll_PID/D Gain'
//  '<S87>'  : 'super/ANGLE CONTROLLER/Roll Controller/stab_roll_PID/Filter'
//  '<S88>'  : 'super/ANGLE CONTROLLER/Roll Controller/stab_roll_PID/Filter ICs'
//  '<S89>'  : 'super/ANGLE CONTROLLER/Roll Controller/stab_roll_PID/I Gain'
//  '<S90>'  : 'super/ANGLE CONTROLLER/Roll Controller/stab_roll_PID/Ideal P Gain'
//  '<S91>'  : 'super/ANGLE CONTROLLER/Roll Controller/stab_roll_PID/Ideal P Gain Fdbk'
//  '<S92>'  : 'super/ANGLE CONTROLLER/Roll Controller/stab_roll_PID/Integrator'
//  '<S93>'  : 'super/ANGLE CONTROLLER/Roll Controller/stab_roll_PID/Integrator ICs'
//  '<S94>'  : 'super/ANGLE CONTROLLER/Roll Controller/stab_roll_PID/N Copy'
//  '<S95>'  : 'super/ANGLE CONTROLLER/Roll Controller/stab_roll_PID/N Gain'
//  '<S96>'  : 'super/ANGLE CONTROLLER/Roll Controller/stab_roll_PID/P Copy'
//  '<S97>'  : 'super/ANGLE CONTROLLER/Roll Controller/stab_roll_PID/Parallel P Gain'
//  '<S98>'  : 'super/ANGLE CONTROLLER/Roll Controller/stab_roll_PID/Reset Signal'
//  '<S99>'  : 'super/ANGLE CONTROLLER/Roll Controller/stab_roll_PID/Saturation'
//  '<S100>' : 'super/ANGLE CONTROLLER/Roll Controller/stab_roll_PID/Saturation Fdbk'
//  '<S101>' : 'super/ANGLE CONTROLLER/Roll Controller/stab_roll_PID/Sum'
//  '<S102>' : 'super/ANGLE CONTROLLER/Roll Controller/stab_roll_PID/Sum Fdbk'
//  '<S103>' : 'super/ANGLE CONTROLLER/Roll Controller/stab_roll_PID/Tracking Mode'
//  '<S104>' : 'super/ANGLE CONTROLLER/Roll Controller/stab_roll_PID/Tracking Mode Sum'
//  '<S105>' : 'super/ANGLE CONTROLLER/Roll Controller/stab_roll_PID/Tsamp - Integral'
//  '<S106>' : 'super/ANGLE CONTROLLER/Roll Controller/stab_roll_PID/Tsamp - Ngain'
//  '<S107>' : 'super/ANGLE CONTROLLER/Roll Controller/stab_roll_PID/postSat Signal'
//  '<S108>' : 'super/ANGLE CONTROLLER/Roll Controller/stab_roll_PID/preSat Signal'
//  '<S109>' : 'super/ANGLE CONTROLLER/Roll Controller/stab_roll_PID/Anti-windup/Disc. Clamping Parallel'
//  '<S110>' : 'super/ANGLE CONTROLLER/Roll Controller/stab_roll_PID/Anti-windup/Disc. Clamping Parallel/Dead Zone'
//  '<S111>' : 'super/ANGLE CONTROLLER/Roll Controller/stab_roll_PID/Anti-windup/Disc. Clamping Parallel/Dead Zone/Enabled'
//  '<S112>' : 'super/ANGLE CONTROLLER/Roll Controller/stab_roll_PID/D Gain/Disabled'
//  '<S113>' : 'super/ANGLE CONTROLLER/Roll Controller/stab_roll_PID/Filter/Disabled'
//  '<S114>' : 'super/ANGLE CONTROLLER/Roll Controller/stab_roll_PID/Filter ICs/Disabled'
//  '<S115>' : 'super/ANGLE CONTROLLER/Roll Controller/stab_roll_PID/I Gain/External Parameters'
//  '<S116>' : 'super/ANGLE CONTROLLER/Roll Controller/stab_roll_PID/Ideal P Gain/Passthrough'
//  '<S117>' : 'super/ANGLE CONTROLLER/Roll Controller/stab_roll_PID/Ideal P Gain Fdbk/Disabled'
//  '<S118>' : 'super/ANGLE CONTROLLER/Roll Controller/stab_roll_PID/Integrator/Discrete'
//  '<S119>' : 'super/ANGLE CONTROLLER/Roll Controller/stab_roll_PID/Integrator ICs/Internal IC'
//  '<S120>' : 'super/ANGLE CONTROLLER/Roll Controller/stab_roll_PID/N Copy/Disabled wSignal Specification'
//  '<S121>' : 'super/ANGLE CONTROLLER/Roll Controller/stab_roll_PID/N Gain/Disabled'
//  '<S122>' : 'super/ANGLE CONTROLLER/Roll Controller/stab_roll_PID/P Copy/Disabled'
//  '<S123>' : 'super/ANGLE CONTROLLER/Roll Controller/stab_roll_PID/Parallel P Gain/External Parameters'
//  '<S124>' : 'super/ANGLE CONTROLLER/Roll Controller/stab_roll_PID/Reset Signal/Disabled'
//  '<S125>' : 'super/ANGLE CONTROLLER/Roll Controller/stab_roll_PID/Saturation/Enabled'
//  '<S126>' : 'super/ANGLE CONTROLLER/Roll Controller/stab_roll_PID/Saturation Fdbk/Disabled'
//  '<S127>' : 'super/ANGLE CONTROLLER/Roll Controller/stab_roll_PID/Sum/Sum_PI'
//  '<S128>' : 'super/ANGLE CONTROLLER/Roll Controller/stab_roll_PID/Sum Fdbk/Disabled'
//  '<S129>' : 'super/ANGLE CONTROLLER/Roll Controller/stab_roll_PID/Tracking Mode/Disabled'
//  '<S130>' : 'super/ANGLE CONTROLLER/Roll Controller/stab_roll_PID/Tracking Mode Sum/Passthrough'
//  '<S131>' : 'super/ANGLE CONTROLLER/Roll Controller/stab_roll_PID/Tsamp - Integral/Passthrough'
//  '<S132>' : 'super/ANGLE CONTROLLER/Roll Controller/stab_roll_PID/Tsamp - Ngain/Passthrough'
//  '<S133>' : 'super/ANGLE CONTROLLER/Roll Controller/stab_roll_PID/postSat Signal/Forward_Path'
//  '<S134>' : 'super/ANGLE CONTROLLER/Roll Controller/stab_roll_PID/preSat Signal/Forward_Path'
//  '<S135>' : 'super/ANGLE CONTROLLER/Yaw Rate Controller/Discrete Derivative'
//  '<S136>' : 'super/ANGLE CONTROLLER/Yaw Rate Controller/Gain Selector'
//  '<S137>' : 'super/ANGLE CONTROLLER/Yaw Rate Controller/Gain Selector1'
//  '<S138>' : 'super/ANGLE CONTROLLER/Yaw Rate Controller/Gain Selector2'
//  '<S139>' : 'super/ANGLE CONTROLLER/Yaw Rate Controller/stab_yaw_rate_PID'
//  '<S140>' : 'super/ANGLE CONTROLLER/Yaw Rate Controller/stab_yaw_rate_PID/Anti-windup'
//  '<S141>' : 'super/ANGLE CONTROLLER/Yaw Rate Controller/stab_yaw_rate_PID/D Gain'
//  '<S142>' : 'super/ANGLE CONTROLLER/Yaw Rate Controller/stab_yaw_rate_PID/Filter'
//  '<S143>' : 'super/ANGLE CONTROLLER/Yaw Rate Controller/stab_yaw_rate_PID/Filter ICs'
//  '<S144>' : 'super/ANGLE CONTROLLER/Yaw Rate Controller/stab_yaw_rate_PID/I Gain'
//  '<S145>' : 'super/ANGLE CONTROLLER/Yaw Rate Controller/stab_yaw_rate_PID/Ideal P Gain'
//  '<S146>' : 'super/ANGLE CONTROLLER/Yaw Rate Controller/stab_yaw_rate_PID/Ideal P Gain Fdbk'
//  '<S147>' : 'super/ANGLE CONTROLLER/Yaw Rate Controller/stab_yaw_rate_PID/Integrator'
//  '<S148>' : 'super/ANGLE CONTROLLER/Yaw Rate Controller/stab_yaw_rate_PID/Integrator ICs'
//  '<S149>' : 'super/ANGLE CONTROLLER/Yaw Rate Controller/stab_yaw_rate_PID/N Copy'
//  '<S150>' : 'super/ANGLE CONTROLLER/Yaw Rate Controller/stab_yaw_rate_PID/N Gain'
//  '<S151>' : 'super/ANGLE CONTROLLER/Yaw Rate Controller/stab_yaw_rate_PID/P Copy'
//  '<S152>' : 'super/ANGLE CONTROLLER/Yaw Rate Controller/stab_yaw_rate_PID/Parallel P Gain'
//  '<S153>' : 'super/ANGLE CONTROLLER/Yaw Rate Controller/stab_yaw_rate_PID/Reset Signal'
//  '<S154>' : 'super/ANGLE CONTROLLER/Yaw Rate Controller/stab_yaw_rate_PID/Saturation'
//  '<S155>' : 'super/ANGLE CONTROLLER/Yaw Rate Controller/stab_yaw_rate_PID/Saturation Fdbk'
//  '<S156>' : 'super/ANGLE CONTROLLER/Yaw Rate Controller/stab_yaw_rate_PID/Sum'
//  '<S157>' : 'super/ANGLE CONTROLLER/Yaw Rate Controller/stab_yaw_rate_PID/Sum Fdbk'
//  '<S158>' : 'super/ANGLE CONTROLLER/Yaw Rate Controller/stab_yaw_rate_PID/Tracking Mode'
//  '<S159>' : 'super/ANGLE CONTROLLER/Yaw Rate Controller/stab_yaw_rate_PID/Tracking Mode Sum'
//  '<S160>' : 'super/ANGLE CONTROLLER/Yaw Rate Controller/stab_yaw_rate_PID/Tsamp - Integral'
//  '<S161>' : 'super/ANGLE CONTROLLER/Yaw Rate Controller/stab_yaw_rate_PID/Tsamp - Ngain'
//  '<S162>' : 'super/ANGLE CONTROLLER/Yaw Rate Controller/stab_yaw_rate_PID/postSat Signal'
//  '<S163>' : 'super/ANGLE CONTROLLER/Yaw Rate Controller/stab_yaw_rate_PID/preSat Signal'
//  '<S164>' : 'super/ANGLE CONTROLLER/Yaw Rate Controller/stab_yaw_rate_PID/Anti-windup/Disc. Clamping Parallel'
//  '<S165>' : 'super/ANGLE CONTROLLER/Yaw Rate Controller/stab_yaw_rate_PID/Anti-windup/Disc. Clamping Parallel/Dead Zone'
//  '<S166>' : 'super/ANGLE CONTROLLER/Yaw Rate Controller/stab_yaw_rate_PID/Anti-windup/Disc. Clamping Parallel/Dead Zone/Enabled'
//  '<S167>' : 'super/ANGLE CONTROLLER/Yaw Rate Controller/stab_yaw_rate_PID/D Gain/Disabled'
//  '<S168>' : 'super/ANGLE CONTROLLER/Yaw Rate Controller/stab_yaw_rate_PID/Filter/Disabled'
//  '<S169>' : 'super/ANGLE CONTROLLER/Yaw Rate Controller/stab_yaw_rate_PID/Filter ICs/Disabled'
//  '<S170>' : 'super/ANGLE CONTROLLER/Yaw Rate Controller/stab_yaw_rate_PID/I Gain/External Parameters'
//  '<S171>' : 'super/ANGLE CONTROLLER/Yaw Rate Controller/stab_yaw_rate_PID/Ideal P Gain/Passthrough'
//  '<S172>' : 'super/ANGLE CONTROLLER/Yaw Rate Controller/stab_yaw_rate_PID/Ideal P Gain Fdbk/Disabled'
//  '<S173>' : 'super/ANGLE CONTROLLER/Yaw Rate Controller/stab_yaw_rate_PID/Integrator/Discrete'
//  '<S174>' : 'super/ANGLE CONTROLLER/Yaw Rate Controller/stab_yaw_rate_PID/Integrator ICs/Internal IC'
//  '<S175>' : 'super/ANGLE CONTROLLER/Yaw Rate Controller/stab_yaw_rate_PID/N Copy/Disabled wSignal Specification'
//  '<S176>' : 'super/ANGLE CONTROLLER/Yaw Rate Controller/stab_yaw_rate_PID/N Gain/Disabled'
//  '<S177>' : 'super/ANGLE CONTROLLER/Yaw Rate Controller/stab_yaw_rate_PID/P Copy/Disabled'
//  '<S178>' : 'super/ANGLE CONTROLLER/Yaw Rate Controller/stab_yaw_rate_PID/Parallel P Gain/External Parameters'
//  '<S179>' : 'super/ANGLE CONTROLLER/Yaw Rate Controller/stab_yaw_rate_PID/Reset Signal/Disabled'
//  '<S180>' : 'super/ANGLE CONTROLLER/Yaw Rate Controller/stab_yaw_rate_PID/Saturation/Enabled'
//  '<S181>' : 'super/ANGLE CONTROLLER/Yaw Rate Controller/stab_yaw_rate_PID/Saturation Fdbk/Disabled'
//  '<S182>' : 'super/ANGLE CONTROLLER/Yaw Rate Controller/stab_yaw_rate_PID/Sum/Sum_PI'
//  '<S183>' : 'super/ANGLE CONTROLLER/Yaw Rate Controller/stab_yaw_rate_PID/Sum Fdbk/Disabled'
//  '<S184>' : 'super/ANGLE CONTROLLER/Yaw Rate Controller/stab_yaw_rate_PID/Tracking Mode/Disabled'
//  '<S185>' : 'super/ANGLE CONTROLLER/Yaw Rate Controller/stab_yaw_rate_PID/Tracking Mode Sum/Passthrough'
//  '<S186>' : 'super/ANGLE CONTROLLER/Yaw Rate Controller/stab_yaw_rate_PID/Tsamp - Integral/Passthrough'
//  '<S187>' : 'super/ANGLE CONTROLLER/Yaw Rate Controller/stab_yaw_rate_PID/Tsamp - Ngain/Passthrough'
//  '<S188>' : 'super/ANGLE CONTROLLER/Yaw Rate Controller/stab_yaw_rate_PID/postSat Signal/Forward_Path'
//  '<S189>' : 'super/ANGLE CONTROLLER/Yaw Rate Controller/stab_yaw_rate_PID/preSat Signal/Forward_Path'
//  '<S190>' : 'super/LAND CONTROLLER/Vertical speed controller'
//  '<S191>' : 'super/LAND CONTROLLER/Vertical speed controller/Compare To Constant'
//  '<S192>' : 'super/Motor Mixing Algorithm/Saturation Dynamic'
//  '<S193>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller'
//  '<S194>' : 'super/POS_HOLD CONTROLLER/Vertical speed controller'
//  '<S195>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/2D rotation from NED_xy to body_xy'
//  '<S196>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem'
//  '<S197>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem1'
//  '<S198>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem/Pitch angle controller'
//  '<S199>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem/Pitch angle controller/Anti-windup'
//  '<S200>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem/Pitch angle controller/D Gain'
//  '<S201>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem/Pitch angle controller/Filter'
//  '<S202>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem/Pitch angle controller/Filter ICs'
//  '<S203>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem/Pitch angle controller/I Gain'
//  '<S204>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem/Pitch angle controller/Ideal P Gain'
//  '<S205>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem/Pitch angle controller/Ideal P Gain Fdbk'
//  '<S206>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem/Pitch angle controller/Integrator'
//  '<S207>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem/Pitch angle controller/Integrator ICs'
//  '<S208>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem/Pitch angle controller/N Copy'
//  '<S209>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem/Pitch angle controller/N Gain'
//  '<S210>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem/Pitch angle controller/P Copy'
//  '<S211>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem/Pitch angle controller/Parallel P Gain'
//  '<S212>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem/Pitch angle controller/Reset Signal'
//  '<S213>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem/Pitch angle controller/Saturation'
//  '<S214>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem/Pitch angle controller/Saturation Fdbk'
//  '<S215>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem/Pitch angle controller/Sum'
//  '<S216>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem/Pitch angle controller/Sum Fdbk'
//  '<S217>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem/Pitch angle controller/Tracking Mode'
//  '<S218>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem/Pitch angle controller/Tracking Mode Sum'
//  '<S219>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem/Pitch angle controller/Tsamp - Integral'
//  '<S220>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem/Pitch angle controller/Tsamp - Ngain'
//  '<S221>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem/Pitch angle controller/postSat Signal'
//  '<S222>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem/Pitch angle controller/preSat Signal'
//  '<S223>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem/Pitch angle controller/Anti-windup/Disc. Clamping Parallel'
//  '<S224>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem/Pitch angle controller/Anti-windup/Disc. Clamping Parallel/Dead Zone'
//  '<S225>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem/Pitch angle controller/Anti-windup/Disc. Clamping Parallel/Dead Zone/Enabled'
//  '<S226>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem/Pitch angle controller/D Gain/External Parameters'
//  '<S227>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem/Pitch angle controller/Filter/Differentiator'
//  '<S228>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem/Pitch angle controller/Filter/Differentiator/Tsamp'
//  '<S229>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem/Pitch angle controller/Filter/Differentiator/Tsamp/Internal Ts'
//  '<S230>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem/Pitch angle controller/Filter ICs/Internal IC - Differentiator'
//  '<S231>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem/Pitch angle controller/I Gain/External Parameters'
//  '<S232>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem/Pitch angle controller/Ideal P Gain/Passthrough'
//  '<S233>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem/Pitch angle controller/Ideal P Gain Fdbk/Disabled'
//  '<S234>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem/Pitch angle controller/Integrator/Discrete'
//  '<S235>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem/Pitch angle controller/Integrator ICs/Internal IC'
//  '<S236>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem/Pitch angle controller/N Copy/Disabled wSignal Specification'
//  '<S237>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem/Pitch angle controller/N Gain/Passthrough'
//  '<S238>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem/Pitch angle controller/P Copy/Disabled'
//  '<S239>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem/Pitch angle controller/Parallel P Gain/External Parameters'
//  '<S240>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem/Pitch angle controller/Reset Signal/Disabled'
//  '<S241>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem/Pitch angle controller/Saturation/Enabled'
//  '<S242>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem/Pitch angle controller/Saturation Fdbk/Disabled'
//  '<S243>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem/Pitch angle controller/Sum/Sum_PID'
//  '<S244>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem/Pitch angle controller/Sum Fdbk/Disabled'
//  '<S245>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem/Pitch angle controller/Tracking Mode/Disabled'
//  '<S246>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem/Pitch angle controller/Tracking Mode Sum/Passthrough'
//  '<S247>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem/Pitch angle controller/Tsamp - Integral/Passthrough'
//  '<S248>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem/Pitch angle controller/Tsamp - Ngain/Passthrough'
//  '<S249>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem/Pitch angle controller/postSat Signal/Forward_Path'
//  '<S250>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem/Pitch angle controller/preSat Signal/Forward_Path'
//  '<S251>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem1/Roll angle controller'
//  '<S252>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem1/Roll angle controller/Anti-windup'
//  '<S253>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem1/Roll angle controller/D Gain'
//  '<S254>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem1/Roll angle controller/Filter'
//  '<S255>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem1/Roll angle controller/Filter ICs'
//  '<S256>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem1/Roll angle controller/I Gain'
//  '<S257>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem1/Roll angle controller/Ideal P Gain'
//  '<S258>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem1/Roll angle controller/Ideal P Gain Fdbk'
//  '<S259>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem1/Roll angle controller/Integrator'
//  '<S260>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem1/Roll angle controller/Integrator ICs'
//  '<S261>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem1/Roll angle controller/N Copy'
//  '<S262>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem1/Roll angle controller/N Gain'
//  '<S263>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem1/Roll angle controller/P Copy'
//  '<S264>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem1/Roll angle controller/Parallel P Gain'
//  '<S265>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem1/Roll angle controller/Reset Signal'
//  '<S266>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem1/Roll angle controller/Saturation'
//  '<S267>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem1/Roll angle controller/Saturation Fdbk'
//  '<S268>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem1/Roll angle controller/Sum'
//  '<S269>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem1/Roll angle controller/Sum Fdbk'
//  '<S270>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem1/Roll angle controller/Tracking Mode'
//  '<S271>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem1/Roll angle controller/Tracking Mode Sum'
//  '<S272>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem1/Roll angle controller/Tsamp - Integral'
//  '<S273>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem1/Roll angle controller/Tsamp - Ngain'
//  '<S274>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem1/Roll angle controller/postSat Signal'
//  '<S275>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem1/Roll angle controller/preSat Signal'
//  '<S276>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem1/Roll angle controller/Anti-windup/Disc. Clamping Parallel'
//  '<S277>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem1/Roll angle controller/Anti-windup/Disc. Clamping Parallel/Dead Zone'
//  '<S278>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem1/Roll angle controller/Anti-windup/Disc. Clamping Parallel/Dead Zone/Enabled'
//  '<S279>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem1/Roll angle controller/D Gain/External Parameters'
//  '<S280>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem1/Roll angle controller/Filter/Differentiator'
//  '<S281>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem1/Roll angle controller/Filter/Differentiator/Tsamp'
//  '<S282>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem1/Roll angle controller/Filter/Differentiator/Tsamp/Internal Ts'
//  '<S283>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem1/Roll angle controller/Filter ICs/Internal IC - Differentiator'
//  '<S284>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem1/Roll angle controller/I Gain/External Parameters'
//  '<S285>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem1/Roll angle controller/Ideal P Gain/Passthrough'
//  '<S286>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem1/Roll angle controller/Ideal P Gain Fdbk/Disabled'
//  '<S287>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem1/Roll angle controller/Integrator/Discrete'
//  '<S288>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem1/Roll angle controller/Integrator ICs/Internal IC'
//  '<S289>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem1/Roll angle controller/N Copy/Disabled wSignal Specification'
//  '<S290>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem1/Roll angle controller/N Gain/Passthrough'
//  '<S291>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem1/Roll angle controller/P Copy/Disabled'
//  '<S292>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem1/Roll angle controller/Parallel P Gain/External Parameters'
//  '<S293>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem1/Roll angle controller/Reset Signal/Disabled'
//  '<S294>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem1/Roll angle controller/Saturation/Enabled'
//  '<S295>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem1/Roll angle controller/Saturation Fdbk/Disabled'
//  '<S296>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem1/Roll angle controller/Sum/Sum_PID'
//  '<S297>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem1/Roll angle controller/Sum Fdbk/Disabled'
//  '<S298>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem1/Roll angle controller/Tracking Mode/Disabled'
//  '<S299>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem1/Roll angle controller/Tracking Mode Sum/Passthrough'
//  '<S300>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem1/Roll angle controller/Tsamp - Integral/Passthrough'
//  '<S301>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem1/Roll angle controller/Tsamp - Ngain/Passthrough'
//  '<S302>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem1/Roll angle controller/postSat Signal/Forward_Path'
//  '<S303>' : 'super/POS_HOLD CONTROLLER/Horizontal speed controller/Subsystem1/Roll angle controller/preSat Signal/Forward_Path'
//  '<S304>' : 'super/POS_HOLD CONTROLLER/Vertical speed controller/PID Controller'
//  '<S305>' : 'super/POS_HOLD CONTROLLER/Vertical speed controller/PID Controller/Anti-windup'
//  '<S306>' : 'super/POS_HOLD CONTROLLER/Vertical speed controller/PID Controller/D Gain'
//  '<S307>' : 'super/POS_HOLD CONTROLLER/Vertical speed controller/PID Controller/Filter'
//  '<S308>' : 'super/POS_HOLD CONTROLLER/Vertical speed controller/PID Controller/Filter ICs'
//  '<S309>' : 'super/POS_HOLD CONTROLLER/Vertical speed controller/PID Controller/I Gain'
//  '<S310>' : 'super/POS_HOLD CONTROLLER/Vertical speed controller/PID Controller/Ideal P Gain'
//  '<S311>' : 'super/POS_HOLD CONTROLLER/Vertical speed controller/PID Controller/Ideal P Gain Fdbk'
//  '<S312>' : 'super/POS_HOLD CONTROLLER/Vertical speed controller/PID Controller/Integrator'
//  '<S313>' : 'super/POS_HOLD CONTROLLER/Vertical speed controller/PID Controller/Integrator ICs'
//  '<S314>' : 'super/POS_HOLD CONTROLLER/Vertical speed controller/PID Controller/N Copy'
//  '<S315>' : 'super/POS_HOLD CONTROLLER/Vertical speed controller/PID Controller/N Gain'
//  '<S316>' : 'super/POS_HOLD CONTROLLER/Vertical speed controller/PID Controller/P Copy'
//  '<S317>' : 'super/POS_HOLD CONTROLLER/Vertical speed controller/PID Controller/Parallel P Gain'
//  '<S318>' : 'super/POS_HOLD CONTROLLER/Vertical speed controller/PID Controller/Reset Signal'
//  '<S319>' : 'super/POS_HOLD CONTROLLER/Vertical speed controller/PID Controller/Saturation'
//  '<S320>' : 'super/POS_HOLD CONTROLLER/Vertical speed controller/PID Controller/Saturation Fdbk'
//  '<S321>' : 'super/POS_HOLD CONTROLLER/Vertical speed controller/PID Controller/Sum'
//  '<S322>' : 'super/POS_HOLD CONTROLLER/Vertical speed controller/PID Controller/Sum Fdbk'
//  '<S323>' : 'super/POS_HOLD CONTROLLER/Vertical speed controller/PID Controller/Tracking Mode'
//  '<S324>' : 'super/POS_HOLD CONTROLLER/Vertical speed controller/PID Controller/Tracking Mode Sum'
//  '<S325>' : 'super/POS_HOLD CONTROLLER/Vertical speed controller/PID Controller/Tsamp - Integral'
//  '<S326>' : 'super/POS_HOLD CONTROLLER/Vertical speed controller/PID Controller/Tsamp - Ngain'
//  '<S327>' : 'super/POS_HOLD CONTROLLER/Vertical speed controller/PID Controller/postSat Signal'
//  '<S328>' : 'super/POS_HOLD CONTROLLER/Vertical speed controller/PID Controller/preSat Signal'
//  '<S329>' : 'super/POS_HOLD CONTROLLER/Vertical speed controller/PID Controller/Anti-windup/Disc. Clamping Parallel'
//  '<S330>' : 'super/POS_HOLD CONTROLLER/Vertical speed controller/PID Controller/Anti-windup/Disc. Clamping Parallel/Dead Zone'
//  '<S331>' : 'super/POS_HOLD CONTROLLER/Vertical speed controller/PID Controller/Anti-windup/Disc. Clamping Parallel/Dead Zone/Enabled'
//  '<S332>' : 'super/POS_HOLD CONTROLLER/Vertical speed controller/PID Controller/D Gain/External Parameters'
//  '<S333>' : 'super/POS_HOLD CONTROLLER/Vertical speed controller/PID Controller/Filter/Differentiator'
//  '<S334>' : 'super/POS_HOLD CONTROLLER/Vertical speed controller/PID Controller/Filter/Differentiator/Tsamp'
//  '<S335>' : 'super/POS_HOLD CONTROLLER/Vertical speed controller/PID Controller/Filter/Differentiator/Tsamp/Internal Ts'
//  '<S336>' : 'super/POS_HOLD CONTROLLER/Vertical speed controller/PID Controller/Filter ICs/Internal IC - Differentiator'
//  '<S337>' : 'super/POS_HOLD CONTROLLER/Vertical speed controller/PID Controller/I Gain/External Parameters'
//  '<S338>' : 'super/POS_HOLD CONTROLLER/Vertical speed controller/PID Controller/Ideal P Gain/Passthrough'
//  '<S339>' : 'super/POS_HOLD CONTROLLER/Vertical speed controller/PID Controller/Ideal P Gain Fdbk/Disabled'
//  '<S340>' : 'super/POS_HOLD CONTROLLER/Vertical speed controller/PID Controller/Integrator/Discrete'
//  '<S341>' : 'super/POS_HOLD CONTROLLER/Vertical speed controller/PID Controller/Integrator ICs/Internal IC'
//  '<S342>' : 'super/POS_HOLD CONTROLLER/Vertical speed controller/PID Controller/N Copy/Disabled wSignal Specification'
//  '<S343>' : 'super/POS_HOLD CONTROLLER/Vertical speed controller/PID Controller/N Gain/Passthrough'
//  '<S344>' : 'super/POS_HOLD CONTROLLER/Vertical speed controller/PID Controller/P Copy/Disabled'
//  '<S345>' : 'super/POS_HOLD CONTROLLER/Vertical speed controller/PID Controller/Parallel P Gain/External Parameters'
//  '<S346>' : 'super/POS_HOLD CONTROLLER/Vertical speed controller/PID Controller/Reset Signal/Disabled'
//  '<S347>' : 'super/POS_HOLD CONTROLLER/Vertical speed controller/PID Controller/Saturation/Enabled'
//  '<S348>' : 'super/POS_HOLD CONTROLLER/Vertical speed controller/PID Controller/Saturation Fdbk/Disabled'
//  '<S349>' : 'super/POS_HOLD CONTROLLER/Vertical speed controller/PID Controller/Sum/Sum_PID'
//  '<S350>' : 'super/POS_HOLD CONTROLLER/Vertical speed controller/PID Controller/Sum Fdbk/Disabled'
//  '<S351>' : 'super/POS_HOLD CONTROLLER/Vertical speed controller/PID Controller/Tracking Mode/Disabled'
//  '<S352>' : 'super/POS_HOLD CONTROLLER/Vertical speed controller/PID Controller/Tracking Mode Sum/Passthrough'
//  '<S353>' : 'super/POS_HOLD CONTROLLER/Vertical speed controller/PID Controller/Tsamp - Integral/Passthrough'
//  '<S354>' : 'super/POS_HOLD CONTROLLER/Vertical speed controller/PID Controller/Tsamp - Ngain/Passthrough'
//  '<S355>' : 'super/POS_HOLD CONTROLLER/Vertical speed controller/PID Controller/postSat Signal/Forward_Path'
//  '<S356>' : 'super/POS_HOLD CONTROLLER/Vertical speed controller/PID Controller/preSat Signal/Forward_Path'
//  '<S357>' : 'super/Pos_Hold_input_conversion/Throttle Mapper'
//  '<S358>' : 'super/Pos_Hold_input_conversion/Throttle Mapper/Compare To Constant'
//  '<S359>' : 'super/Pos_Hold_input_conversion/Throttle Mapper/Compare To Constant1'
//  '<S360>' : 'super/Pos_Hold_input_conversion2/Throttle Mapper'
//  '<S361>' : 'super/Pos_Hold_input_conversion2/Throttle Mapper/Compare To Constant'
//  '<S362>' : 'super/Pos_Hold_input_conversion2/Throttle Mapper/Compare To Constant1'
//  '<S363>' : 'super/RTL CONTROLLER/Compare To Constant1'
//  '<S364>' : 'super/RTL CONTROLLER/Horizontal_position_controller'
//  '<S365>' : 'super/RTL CONTROLLER/horizontal_error_calculation '
//  '<S366>' : 'super/RTL CONTROLLER/vertical_RTL_position_controller'
//  '<S367>' : 'super/RTL CONTROLLER/Horizontal_position_controller/PID Controller2'
//  '<S368>' : 'super/RTL CONTROLLER/Horizontal_position_controller/convert to ve_cmd'
//  '<S369>' : 'super/RTL CONTROLLER/Horizontal_position_controller/inertial_to_body conversion'
//  '<S370>' : 'super/RTL CONTROLLER/Horizontal_position_controller/PID Controller2/Anti-windup'
//  '<S371>' : 'super/RTL CONTROLLER/Horizontal_position_controller/PID Controller2/D Gain'
//  '<S372>' : 'super/RTL CONTROLLER/Horizontal_position_controller/PID Controller2/Filter'
//  '<S373>' : 'super/RTL CONTROLLER/Horizontal_position_controller/PID Controller2/Filter ICs'
//  '<S374>' : 'super/RTL CONTROLLER/Horizontal_position_controller/PID Controller2/I Gain'
//  '<S375>' : 'super/RTL CONTROLLER/Horizontal_position_controller/PID Controller2/Ideal P Gain'
//  '<S376>' : 'super/RTL CONTROLLER/Horizontal_position_controller/PID Controller2/Ideal P Gain Fdbk'
//  '<S377>' : 'super/RTL CONTROLLER/Horizontal_position_controller/PID Controller2/Integrator'
//  '<S378>' : 'super/RTL CONTROLLER/Horizontal_position_controller/PID Controller2/Integrator ICs'
//  '<S379>' : 'super/RTL CONTROLLER/Horizontal_position_controller/PID Controller2/N Copy'
//  '<S380>' : 'super/RTL CONTROLLER/Horizontal_position_controller/PID Controller2/N Gain'
//  '<S381>' : 'super/RTL CONTROLLER/Horizontal_position_controller/PID Controller2/P Copy'
//  '<S382>' : 'super/RTL CONTROLLER/Horizontal_position_controller/PID Controller2/Parallel P Gain'
//  '<S383>' : 'super/RTL CONTROLLER/Horizontal_position_controller/PID Controller2/Reset Signal'
//  '<S384>' : 'super/RTL CONTROLLER/Horizontal_position_controller/PID Controller2/Saturation'
//  '<S385>' : 'super/RTL CONTROLLER/Horizontal_position_controller/PID Controller2/Saturation Fdbk'
//  '<S386>' : 'super/RTL CONTROLLER/Horizontal_position_controller/PID Controller2/Sum'
//  '<S387>' : 'super/RTL CONTROLLER/Horizontal_position_controller/PID Controller2/Sum Fdbk'
//  '<S388>' : 'super/RTL CONTROLLER/Horizontal_position_controller/PID Controller2/Tracking Mode'
//  '<S389>' : 'super/RTL CONTROLLER/Horizontal_position_controller/PID Controller2/Tracking Mode Sum'
//  '<S390>' : 'super/RTL CONTROLLER/Horizontal_position_controller/PID Controller2/Tsamp - Integral'
//  '<S391>' : 'super/RTL CONTROLLER/Horizontal_position_controller/PID Controller2/Tsamp - Ngain'
//  '<S392>' : 'super/RTL CONTROLLER/Horizontal_position_controller/PID Controller2/postSat Signal'
//  '<S393>' : 'super/RTL CONTROLLER/Horizontal_position_controller/PID Controller2/preSat Signal'
//  '<S394>' : 'super/RTL CONTROLLER/Horizontal_position_controller/PID Controller2/Anti-windup/Disabled'
//  '<S395>' : 'super/RTL CONTROLLER/Horizontal_position_controller/PID Controller2/D Gain/Disabled'
//  '<S396>' : 'super/RTL CONTROLLER/Horizontal_position_controller/PID Controller2/Filter/Disabled'
//  '<S397>' : 'super/RTL CONTROLLER/Horizontal_position_controller/PID Controller2/Filter ICs/Disabled'
//  '<S398>' : 'super/RTL CONTROLLER/Horizontal_position_controller/PID Controller2/I Gain/Disabled'
//  '<S399>' : 'super/RTL CONTROLLER/Horizontal_position_controller/PID Controller2/Ideal P Gain/Passthrough'
//  '<S400>' : 'super/RTL CONTROLLER/Horizontal_position_controller/PID Controller2/Ideal P Gain Fdbk/Disabled'
//  '<S401>' : 'super/RTL CONTROLLER/Horizontal_position_controller/PID Controller2/Integrator/Disabled'
//  '<S402>' : 'super/RTL CONTROLLER/Horizontal_position_controller/PID Controller2/Integrator ICs/Disabled'
//  '<S403>' : 'super/RTL CONTROLLER/Horizontal_position_controller/PID Controller2/N Copy/Disabled wSignal Specification'
//  '<S404>' : 'super/RTL CONTROLLER/Horizontal_position_controller/PID Controller2/N Gain/Disabled'
//  '<S405>' : 'super/RTL CONTROLLER/Horizontal_position_controller/PID Controller2/P Copy/Disabled'
//  '<S406>' : 'super/RTL CONTROLLER/Horizontal_position_controller/PID Controller2/Parallel P Gain/External Parameters'
//  '<S407>' : 'super/RTL CONTROLLER/Horizontal_position_controller/PID Controller2/Reset Signal/Disabled'
//  '<S408>' : 'super/RTL CONTROLLER/Horizontal_position_controller/PID Controller2/Saturation/External'
//  '<S409>' : 'super/RTL CONTROLLER/Horizontal_position_controller/PID Controller2/Saturation/External/Saturation Dynamic'
//  '<S410>' : 'super/RTL CONTROLLER/Horizontal_position_controller/PID Controller2/Saturation Fdbk/Disabled'
//  '<S411>' : 'super/RTL CONTROLLER/Horizontal_position_controller/PID Controller2/Sum/Passthrough_P'
//  '<S412>' : 'super/RTL CONTROLLER/Horizontal_position_controller/PID Controller2/Sum Fdbk/Disabled'
//  '<S413>' : 'super/RTL CONTROLLER/Horizontal_position_controller/PID Controller2/Tracking Mode/Disabled'
//  '<S414>' : 'super/RTL CONTROLLER/Horizontal_position_controller/PID Controller2/Tracking Mode Sum/Passthrough'
//  '<S415>' : 'super/RTL CONTROLLER/Horizontal_position_controller/PID Controller2/Tsamp - Integral/Disabled wSignal Specification'
//  '<S416>' : 'super/RTL CONTROLLER/Horizontal_position_controller/PID Controller2/Tsamp - Ngain/Passthrough'
//  '<S417>' : 'super/RTL CONTROLLER/Horizontal_position_controller/PID Controller2/postSat Signal/Forward_Path'
//  '<S418>' : 'super/RTL CONTROLLER/Horizontal_position_controller/PID Controller2/preSat Signal/Forward_Path'
//  '<S419>' : 'super/RTL CONTROLLER/Horizontal_position_controller/inertial_to_body conversion/2D rotation from NED_xy to body_xy'
//  '<S420>' : 'super/RTL CONTROLLER/vertical_RTL_position_controller/Compare To Constant'
//  '<S421>' : 'super/RTL CONTROLLER/vertical_RTL_position_controller/PID Controller2'
//  '<S422>' : 'super/RTL CONTROLLER/vertical_RTL_position_controller/PID Controller2/Anti-windup'
//  '<S423>' : 'super/RTL CONTROLLER/vertical_RTL_position_controller/PID Controller2/D Gain'
//  '<S424>' : 'super/RTL CONTROLLER/vertical_RTL_position_controller/PID Controller2/Filter'
//  '<S425>' : 'super/RTL CONTROLLER/vertical_RTL_position_controller/PID Controller2/Filter ICs'
//  '<S426>' : 'super/RTL CONTROLLER/vertical_RTL_position_controller/PID Controller2/I Gain'
//  '<S427>' : 'super/RTL CONTROLLER/vertical_RTL_position_controller/PID Controller2/Ideal P Gain'
//  '<S428>' : 'super/RTL CONTROLLER/vertical_RTL_position_controller/PID Controller2/Ideal P Gain Fdbk'
//  '<S429>' : 'super/RTL CONTROLLER/vertical_RTL_position_controller/PID Controller2/Integrator'
//  '<S430>' : 'super/RTL CONTROLLER/vertical_RTL_position_controller/PID Controller2/Integrator ICs'
//  '<S431>' : 'super/RTL CONTROLLER/vertical_RTL_position_controller/PID Controller2/N Copy'
//  '<S432>' : 'super/RTL CONTROLLER/vertical_RTL_position_controller/PID Controller2/N Gain'
//  '<S433>' : 'super/RTL CONTROLLER/vertical_RTL_position_controller/PID Controller2/P Copy'
//  '<S434>' : 'super/RTL CONTROLLER/vertical_RTL_position_controller/PID Controller2/Parallel P Gain'
//  '<S435>' : 'super/RTL CONTROLLER/vertical_RTL_position_controller/PID Controller2/Reset Signal'
//  '<S436>' : 'super/RTL CONTROLLER/vertical_RTL_position_controller/PID Controller2/Saturation'
//  '<S437>' : 'super/RTL CONTROLLER/vertical_RTL_position_controller/PID Controller2/Saturation Fdbk'
//  '<S438>' : 'super/RTL CONTROLLER/vertical_RTL_position_controller/PID Controller2/Sum'
//  '<S439>' : 'super/RTL CONTROLLER/vertical_RTL_position_controller/PID Controller2/Sum Fdbk'
//  '<S440>' : 'super/RTL CONTROLLER/vertical_RTL_position_controller/PID Controller2/Tracking Mode'
//  '<S441>' : 'super/RTL CONTROLLER/vertical_RTL_position_controller/PID Controller2/Tracking Mode Sum'
//  '<S442>' : 'super/RTL CONTROLLER/vertical_RTL_position_controller/PID Controller2/Tsamp - Integral'
//  '<S443>' : 'super/RTL CONTROLLER/vertical_RTL_position_controller/PID Controller2/Tsamp - Ngain'
//  '<S444>' : 'super/RTL CONTROLLER/vertical_RTL_position_controller/PID Controller2/postSat Signal'
//  '<S445>' : 'super/RTL CONTROLLER/vertical_RTL_position_controller/PID Controller2/preSat Signal'
//  '<S446>' : 'super/RTL CONTROLLER/vertical_RTL_position_controller/PID Controller2/Anti-windup/Disabled'
//  '<S447>' : 'super/RTL CONTROLLER/vertical_RTL_position_controller/PID Controller2/D Gain/Disabled'
//  '<S448>' : 'super/RTL CONTROLLER/vertical_RTL_position_controller/PID Controller2/Filter/Disabled'
//  '<S449>' : 'super/RTL CONTROLLER/vertical_RTL_position_controller/PID Controller2/Filter ICs/Disabled'
//  '<S450>' : 'super/RTL CONTROLLER/vertical_RTL_position_controller/PID Controller2/I Gain/Disabled'
//  '<S451>' : 'super/RTL CONTROLLER/vertical_RTL_position_controller/PID Controller2/Ideal P Gain/Passthrough'
//  '<S452>' : 'super/RTL CONTROLLER/vertical_RTL_position_controller/PID Controller2/Ideal P Gain Fdbk/Disabled'
//  '<S453>' : 'super/RTL CONTROLLER/vertical_RTL_position_controller/PID Controller2/Integrator/Disabled'
//  '<S454>' : 'super/RTL CONTROLLER/vertical_RTL_position_controller/PID Controller2/Integrator ICs/Disabled'
//  '<S455>' : 'super/RTL CONTROLLER/vertical_RTL_position_controller/PID Controller2/N Copy/Disabled wSignal Specification'
//  '<S456>' : 'super/RTL CONTROLLER/vertical_RTL_position_controller/PID Controller2/N Gain/Disabled'
//  '<S457>' : 'super/RTL CONTROLLER/vertical_RTL_position_controller/PID Controller2/P Copy/Disabled'
//  '<S458>' : 'super/RTL CONTROLLER/vertical_RTL_position_controller/PID Controller2/Parallel P Gain/External Parameters'
//  '<S459>' : 'super/RTL CONTROLLER/vertical_RTL_position_controller/PID Controller2/Reset Signal/Disabled'
//  '<S460>' : 'super/RTL CONTROLLER/vertical_RTL_position_controller/PID Controller2/Saturation/External'
//  '<S461>' : 'super/RTL CONTROLLER/vertical_RTL_position_controller/PID Controller2/Saturation/External/Saturation Dynamic'
//  '<S462>' : 'super/RTL CONTROLLER/vertical_RTL_position_controller/PID Controller2/Saturation Fdbk/Disabled'
//  '<S463>' : 'super/RTL CONTROLLER/vertical_RTL_position_controller/PID Controller2/Sum/Passthrough_P'
//  '<S464>' : 'super/RTL CONTROLLER/vertical_RTL_position_controller/PID Controller2/Sum Fdbk/Disabled'
//  '<S465>' : 'super/RTL CONTROLLER/vertical_RTL_position_controller/PID Controller2/Tracking Mode/Disabled'
//  '<S466>' : 'super/RTL CONTROLLER/vertical_RTL_position_controller/PID Controller2/Tracking Mode Sum/Passthrough'
//  '<S467>' : 'super/RTL CONTROLLER/vertical_RTL_position_controller/PID Controller2/Tsamp - Integral/Disabled wSignal Specification'
//  '<S468>' : 'super/RTL CONTROLLER/vertical_RTL_position_controller/PID Controller2/Tsamp - Ngain/Passthrough'
//  '<S469>' : 'super/RTL CONTROLLER/vertical_RTL_position_controller/PID Controller2/postSat Signal/Forward_Path'
//  '<S470>' : 'super/RTL CONTROLLER/vertical_RTL_position_controller/PID Controller2/preSat Signal/Forward_Path'
//  '<S471>' : 'super/To VMS Data/SBUS & AUX1'
//  '<S472>' : 'super/WAYPOINT CONTROLLER/WP_NAV'
//  '<S473>' : 'super/WAYPOINT CONTROLLER/capture rising edge'
//  '<S474>' : 'super/WAYPOINT CONTROLLER/determine target'
//  '<S475>' : 'super/WAYPOINT CONTROLLER/wp_completion_check'
//  '<S476>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Horizontal position controller'
//  '<S477>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Vertical position controller'
//  '<S478>' : 'super/WAYPOINT CONTROLLER/WP_NAV/heading_controller'
//  '<S479>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Horizontal position controller/Horizontal_position_controller'
//  '<S480>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Horizontal position controller/horizontal_error_calculation '
//  '<S481>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Horizontal position controller/Horizontal_position_controller/PID Controller2'
//  '<S482>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Horizontal position controller/Horizontal_position_controller/convert to ve_cmd'
//  '<S483>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Horizontal position controller/Horizontal_position_controller/heading_error'
//  '<S484>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Horizontal position controller/Horizontal_position_controller/inertial_to_body conversion'
//  '<S485>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Horizontal position controller/Horizontal_position_controller/PID Controller2/Anti-windup'
//  '<S486>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Horizontal position controller/Horizontal_position_controller/PID Controller2/D Gain'
//  '<S487>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Horizontal position controller/Horizontal_position_controller/PID Controller2/Filter'
//  '<S488>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Horizontal position controller/Horizontal_position_controller/PID Controller2/Filter ICs'
//  '<S489>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Horizontal position controller/Horizontal_position_controller/PID Controller2/I Gain'
//  '<S490>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Horizontal position controller/Horizontal_position_controller/PID Controller2/Ideal P Gain'
//  '<S491>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Horizontal position controller/Horizontal_position_controller/PID Controller2/Ideal P Gain Fdbk'
//  '<S492>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Horizontal position controller/Horizontal_position_controller/PID Controller2/Integrator'
//  '<S493>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Horizontal position controller/Horizontal_position_controller/PID Controller2/Integrator ICs'
//  '<S494>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Horizontal position controller/Horizontal_position_controller/PID Controller2/N Copy'
//  '<S495>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Horizontal position controller/Horizontal_position_controller/PID Controller2/N Gain'
//  '<S496>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Horizontal position controller/Horizontal_position_controller/PID Controller2/P Copy'
//  '<S497>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Horizontal position controller/Horizontal_position_controller/PID Controller2/Parallel P Gain'
//  '<S498>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Horizontal position controller/Horizontal_position_controller/PID Controller2/Reset Signal'
//  '<S499>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Horizontal position controller/Horizontal_position_controller/PID Controller2/Saturation'
//  '<S500>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Horizontal position controller/Horizontal_position_controller/PID Controller2/Saturation Fdbk'
//  '<S501>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Horizontal position controller/Horizontal_position_controller/PID Controller2/Sum'
//  '<S502>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Horizontal position controller/Horizontal_position_controller/PID Controller2/Sum Fdbk'
//  '<S503>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Horizontal position controller/Horizontal_position_controller/PID Controller2/Tracking Mode'
//  '<S504>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Horizontal position controller/Horizontal_position_controller/PID Controller2/Tracking Mode Sum'
//  '<S505>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Horizontal position controller/Horizontal_position_controller/PID Controller2/Tsamp - Integral'
//  '<S506>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Horizontal position controller/Horizontal_position_controller/PID Controller2/Tsamp - Ngain'
//  '<S507>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Horizontal position controller/Horizontal_position_controller/PID Controller2/postSat Signal'
//  '<S508>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Horizontal position controller/Horizontal_position_controller/PID Controller2/preSat Signal'
//  '<S509>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Horizontal position controller/Horizontal_position_controller/PID Controller2/Anti-windup/Disabled'
//  '<S510>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Horizontal position controller/Horizontal_position_controller/PID Controller2/D Gain/Disabled'
//  '<S511>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Horizontal position controller/Horizontal_position_controller/PID Controller2/Filter/Disabled'
//  '<S512>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Horizontal position controller/Horizontal_position_controller/PID Controller2/Filter ICs/Disabled'
//  '<S513>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Horizontal position controller/Horizontal_position_controller/PID Controller2/I Gain/Disabled'
//  '<S514>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Horizontal position controller/Horizontal_position_controller/PID Controller2/Ideal P Gain/Passthrough'
//  '<S515>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Horizontal position controller/Horizontal_position_controller/PID Controller2/Ideal P Gain Fdbk/Disabled'
//  '<S516>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Horizontal position controller/Horizontal_position_controller/PID Controller2/Integrator/Disabled'
//  '<S517>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Horizontal position controller/Horizontal_position_controller/PID Controller2/Integrator ICs/Disabled'
//  '<S518>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Horizontal position controller/Horizontal_position_controller/PID Controller2/N Copy/Disabled wSignal Specification'
//  '<S519>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Horizontal position controller/Horizontal_position_controller/PID Controller2/N Gain/Disabled'
//  '<S520>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Horizontal position controller/Horizontal_position_controller/PID Controller2/P Copy/Disabled'
//  '<S521>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Horizontal position controller/Horizontal_position_controller/PID Controller2/Parallel P Gain/External Parameters'
//  '<S522>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Horizontal position controller/Horizontal_position_controller/PID Controller2/Reset Signal/Disabled'
//  '<S523>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Horizontal position controller/Horizontal_position_controller/PID Controller2/Saturation/External'
//  '<S524>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Horizontal position controller/Horizontal_position_controller/PID Controller2/Saturation/External/Saturation Dynamic'
//  '<S525>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Horizontal position controller/Horizontal_position_controller/PID Controller2/Saturation Fdbk/Disabled'
//  '<S526>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Horizontal position controller/Horizontal_position_controller/PID Controller2/Sum/Passthrough_P'
//  '<S527>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Horizontal position controller/Horizontal_position_controller/PID Controller2/Sum Fdbk/Disabled'
//  '<S528>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Horizontal position controller/Horizontal_position_controller/PID Controller2/Tracking Mode/Disabled'
//  '<S529>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Horizontal position controller/Horizontal_position_controller/PID Controller2/Tracking Mode Sum/Passthrough'
//  '<S530>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Horizontal position controller/Horizontal_position_controller/PID Controller2/Tsamp - Integral/Disabled wSignal Specification'
//  '<S531>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Horizontal position controller/Horizontal_position_controller/PID Controller2/Tsamp - Ngain/Passthrough'
//  '<S532>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Horizontal position controller/Horizontal_position_controller/PID Controller2/postSat Signal/Forward_Path'
//  '<S533>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Horizontal position controller/Horizontal_position_controller/PID Controller2/preSat Signal/Forward_Path'
//  '<S534>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Horizontal position controller/Horizontal_position_controller/heading_error/Compare To Constant'
//  '<S535>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Horizontal position controller/Horizontal_position_controller/inertial_to_body conversion/2D rotation from NED_xy to body_xy'
//  '<S536>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Vertical position controller/vertical_position_controller'
//  '<S537>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Vertical position controller/vertical_position_controller/Altitude PID'
//  '<S538>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Vertical position controller/vertical_position_controller/Altitude PID/Anti-windup'
//  '<S539>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Vertical position controller/vertical_position_controller/Altitude PID/D Gain'
//  '<S540>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Vertical position controller/vertical_position_controller/Altitude PID/Filter'
//  '<S541>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Vertical position controller/vertical_position_controller/Altitude PID/Filter ICs'
//  '<S542>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Vertical position controller/vertical_position_controller/Altitude PID/I Gain'
//  '<S543>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Vertical position controller/vertical_position_controller/Altitude PID/Ideal P Gain'
//  '<S544>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Vertical position controller/vertical_position_controller/Altitude PID/Ideal P Gain Fdbk'
//  '<S545>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Vertical position controller/vertical_position_controller/Altitude PID/Integrator'
//  '<S546>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Vertical position controller/vertical_position_controller/Altitude PID/Integrator ICs'
//  '<S547>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Vertical position controller/vertical_position_controller/Altitude PID/N Copy'
//  '<S548>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Vertical position controller/vertical_position_controller/Altitude PID/N Gain'
//  '<S549>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Vertical position controller/vertical_position_controller/Altitude PID/P Copy'
//  '<S550>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Vertical position controller/vertical_position_controller/Altitude PID/Parallel P Gain'
//  '<S551>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Vertical position controller/vertical_position_controller/Altitude PID/Reset Signal'
//  '<S552>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Vertical position controller/vertical_position_controller/Altitude PID/Saturation'
//  '<S553>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Vertical position controller/vertical_position_controller/Altitude PID/Saturation Fdbk'
//  '<S554>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Vertical position controller/vertical_position_controller/Altitude PID/Sum'
//  '<S555>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Vertical position controller/vertical_position_controller/Altitude PID/Sum Fdbk'
//  '<S556>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Vertical position controller/vertical_position_controller/Altitude PID/Tracking Mode'
//  '<S557>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Vertical position controller/vertical_position_controller/Altitude PID/Tracking Mode Sum'
//  '<S558>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Vertical position controller/vertical_position_controller/Altitude PID/Tsamp - Integral'
//  '<S559>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Vertical position controller/vertical_position_controller/Altitude PID/Tsamp - Ngain'
//  '<S560>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Vertical position controller/vertical_position_controller/Altitude PID/postSat Signal'
//  '<S561>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Vertical position controller/vertical_position_controller/Altitude PID/preSat Signal'
//  '<S562>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Vertical position controller/vertical_position_controller/Altitude PID/Anti-windup/Disc. Clamping Ideal'
//  '<S563>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Vertical position controller/vertical_position_controller/Altitude PID/Anti-windup/Disc. Clamping Ideal/Dead Zone'
//  '<S564>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Vertical position controller/vertical_position_controller/Altitude PID/Anti-windup/Disc. Clamping Ideal/Dead Zone/External'
//  '<S565>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Vertical position controller/vertical_position_controller/Altitude PID/Anti-windup/Disc. Clamping Ideal/Dead Zone/External/Dead Zone Dynamic'
//  '<S566>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Vertical position controller/vertical_position_controller/Altitude PID/D Gain/Disabled'
//  '<S567>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Vertical position controller/vertical_position_controller/Altitude PID/Filter/Disabled'
//  '<S568>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Vertical position controller/vertical_position_controller/Altitude PID/Filter ICs/Disabled'
//  '<S569>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Vertical position controller/vertical_position_controller/Altitude PID/I Gain/External Parameters'
//  '<S570>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Vertical position controller/vertical_position_controller/Altitude PID/Ideal P Gain/External Parameters'
//  '<S571>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Vertical position controller/vertical_position_controller/Altitude PID/Ideal P Gain Fdbk/Disabled'
//  '<S572>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Vertical position controller/vertical_position_controller/Altitude PID/Integrator/Discrete'
//  '<S573>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Vertical position controller/vertical_position_controller/Altitude PID/Integrator ICs/Internal IC'
//  '<S574>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Vertical position controller/vertical_position_controller/Altitude PID/N Copy/Disabled wSignal Specification'
//  '<S575>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Vertical position controller/vertical_position_controller/Altitude PID/N Gain/Disabled'
//  '<S576>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Vertical position controller/vertical_position_controller/Altitude PID/P Copy/External Parameters Ideal'
//  '<S577>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Vertical position controller/vertical_position_controller/Altitude PID/Parallel P Gain/Passthrough'
//  '<S578>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Vertical position controller/vertical_position_controller/Altitude PID/Reset Signal/Disabled'
//  '<S579>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Vertical position controller/vertical_position_controller/Altitude PID/Saturation/External'
//  '<S580>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Vertical position controller/vertical_position_controller/Altitude PID/Saturation/External/Saturation Dynamic'
//  '<S581>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Vertical position controller/vertical_position_controller/Altitude PID/Saturation Fdbk/Disabled'
//  '<S582>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Vertical position controller/vertical_position_controller/Altitude PID/Sum/Sum_PI'
//  '<S583>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Vertical position controller/vertical_position_controller/Altitude PID/Sum Fdbk/Disabled'
//  '<S584>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Vertical position controller/vertical_position_controller/Altitude PID/Tracking Mode/Disabled'
//  '<S585>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Vertical position controller/vertical_position_controller/Altitude PID/Tracking Mode Sum/Passthrough'
//  '<S586>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Vertical position controller/vertical_position_controller/Altitude PID/Tsamp - Integral/Passthrough'
//  '<S587>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Vertical position controller/vertical_position_controller/Altitude PID/Tsamp - Ngain/Passthrough'
//  '<S588>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Vertical position controller/vertical_position_controller/Altitude PID/postSat Signal/Forward_Path'
//  '<S589>' : 'super/WAYPOINT CONTROLLER/WP_NAV/Vertical position controller/vertical_position_controller/Altitude PID/preSat Signal/Forward_Path'
//  '<S590>' : 'super/WAYPOINT CONTROLLER/WP_NAV/heading_controller/heading_controller'
//  '<S591>' : 'super/WAYPOINT CONTROLLER/WP_NAV/heading_controller/heading_controller/Discrete Derivative'
//  '<S592>' : 'super/WAYPOINT CONTROLLER/WP_NAV/heading_controller/heading_controller/PID Controller2'
//  '<S593>' : 'super/WAYPOINT CONTROLLER/WP_NAV/heading_controller/heading_controller/heading_error'
//  '<S594>' : 'super/WAYPOINT CONTROLLER/WP_NAV/heading_controller/heading_controller/PID Controller2/Anti-windup'
//  '<S595>' : 'super/WAYPOINT CONTROLLER/WP_NAV/heading_controller/heading_controller/PID Controller2/D Gain'
//  '<S596>' : 'super/WAYPOINT CONTROLLER/WP_NAV/heading_controller/heading_controller/PID Controller2/Filter'
//  '<S597>' : 'super/WAYPOINT CONTROLLER/WP_NAV/heading_controller/heading_controller/PID Controller2/Filter ICs'
//  '<S598>' : 'super/WAYPOINT CONTROLLER/WP_NAV/heading_controller/heading_controller/PID Controller2/I Gain'
//  '<S599>' : 'super/WAYPOINT CONTROLLER/WP_NAV/heading_controller/heading_controller/PID Controller2/Ideal P Gain'
//  '<S600>' : 'super/WAYPOINT CONTROLLER/WP_NAV/heading_controller/heading_controller/PID Controller2/Ideal P Gain Fdbk'
//  '<S601>' : 'super/WAYPOINT CONTROLLER/WP_NAV/heading_controller/heading_controller/PID Controller2/Integrator'
//  '<S602>' : 'super/WAYPOINT CONTROLLER/WP_NAV/heading_controller/heading_controller/PID Controller2/Integrator ICs'
//  '<S603>' : 'super/WAYPOINT CONTROLLER/WP_NAV/heading_controller/heading_controller/PID Controller2/N Copy'
//  '<S604>' : 'super/WAYPOINT CONTROLLER/WP_NAV/heading_controller/heading_controller/PID Controller2/N Gain'
//  '<S605>' : 'super/WAYPOINT CONTROLLER/WP_NAV/heading_controller/heading_controller/PID Controller2/P Copy'
//  '<S606>' : 'super/WAYPOINT CONTROLLER/WP_NAV/heading_controller/heading_controller/PID Controller2/Parallel P Gain'
//  '<S607>' : 'super/WAYPOINT CONTROLLER/WP_NAV/heading_controller/heading_controller/PID Controller2/Reset Signal'
//  '<S608>' : 'super/WAYPOINT CONTROLLER/WP_NAV/heading_controller/heading_controller/PID Controller2/Saturation'
//  '<S609>' : 'super/WAYPOINT CONTROLLER/WP_NAV/heading_controller/heading_controller/PID Controller2/Saturation Fdbk'
//  '<S610>' : 'super/WAYPOINT CONTROLLER/WP_NAV/heading_controller/heading_controller/PID Controller2/Sum'
//  '<S611>' : 'super/WAYPOINT CONTROLLER/WP_NAV/heading_controller/heading_controller/PID Controller2/Sum Fdbk'
//  '<S612>' : 'super/WAYPOINT CONTROLLER/WP_NAV/heading_controller/heading_controller/PID Controller2/Tracking Mode'
//  '<S613>' : 'super/WAYPOINT CONTROLLER/WP_NAV/heading_controller/heading_controller/PID Controller2/Tracking Mode Sum'
//  '<S614>' : 'super/WAYPOINT CONTROLLER/WP_NAV/heading_controller/heading_controller/PID Controller2/Tsamp - Integral'
//  '<S615>' : 'super/WAYPOINT CONTROLLER/WP_NAV/heading_controller/heading_controller/PID Controller2/Tsamp - Ngain'
//  '<S616>' : 'super/WAYPOINT CONTROLLER/WP_NAV/heading_controller/heading_controller/PID Controller2/postSat Signal'
//  '<S617>' : 'super/WAYPOINT CONTROLLER/WP_NAV/heading_controller/heading_controller/PID Controller2/preSat Signal'
//  '<S618>' : 'super/WAYPOINT CONTROLLER/WP_NAV/heading_controller/heading_controller/PID Controller2/Anti-windup/Disc. Clamping Ideal'
//  '<S619>' : 'super/WAYPOINT CONTROLLER/WP_NAV/heading_controller/heading_controller/PID Controller2/Anti-windup/Disc. Clamping Ideal/Dead Zone'
//  '<S620>' : 'super/WAYPOINT CONTROLLER/WP_NAV/heading_controller/heading_controller/PID Controller2/Anti-windup/Disc. Clamping Ideal/Dead Zone/Enabled'
//  '<S621>' : 'super/WAYPOINT CONTROLLER/WP_NAV/heading_controller/heading_controller/PID Controller2/D Gain/Disabled'
//  '<S622>' : 'super/WAYPOINT CONTROLLER/WP_NAV/heading_controller/heading_controller/PID Controller2/Filter/Disabled'
//  '<S623>' : 'super/WAYPOINT CONTROLLER/WP_NAV/heading_controller/heading_controller/PID Controller2/Filter ICs/Disabled'
//  '<S624>' : 'super/WAYPOINT CONTROLLER/WP_NAV/heading_controller/heading_controller/PID Controller2/I Gain/External Parameters'
//  '<S625>' : 'super/WAYPOINT CONTROLLER/WP_NAV/heading_controller/heading_controller/PID Controller2/Ideal P Gain/External Parameters'
//  '<S626>' : 'super/WAYPOINT CONTROLLER/WP_NAV/heading_controller/heading_controller/PID Controller2/Ideal P Gain Fdbk/Disabled'
//  '<S627>' : 'super/WAYPOINT CONTROLLER/WP_NAV/heading_controller/heading_controller/PID Controller2/Integrator/Discrete'
//  '<S628>' : 'super/WAYPOINT CONTROLLER/WP_NAV/heading_controller/heading_controller/PID Controller2/Integrator ICs/Internal IC'
//  '<S629>' : 'super/WAYPOINT CONTROLLER/WP_NAV/heading_controller/heading_controller/PID Controller2/N Copy/Disabled wSignal Specification'
//  '<S630>' : 'super/WAYPOINT CONTROLLER/WP_NAV/heading_controller/heading_controller/PID Controller2/N Gain/Disabled'
//  '<S631>' : 'super/WAYPOINT CONTROLLER/WP_NAV/heading_controller/heading_controller/PID Controller2/P Copy/External Parameters Ideal'
//  '<S632>' : 'super/WAYPOINT CONTROLLER/WP_NAV/heading_controller/heading_controller/PID Controller2/Parallel P Gain/Passthrough'
//  '<S633>' : 'super/WAYPOINT CONTROLLER/WP_NAV/heading_controller/heading_controller/PID Controller2/Reset Signal/Disabled'
//  '<S634>' : 'super/WAYPOINT CONTROLLER/WP_NAV/heading_controller/heading_controller/PID Controller2/Saturation/Enabled'
//  '<S635>' : 'super/WAYPOINT CONTROLLER/WP_NAV/heading_controller/heading_controller/PID Controller2/Saturation Fdbk/Disabled'
//  '<S636>' : 'super/WAYPOINT CONTROLLER/WP_NAV/heading_controller/heading_controller/PID Controller2/Sum/Sum_PI'
//  '<S637>' : 'super/WAYPOINT CONTROLLER/WP_NAV/heading_controller/heading_controller/PID Controller2/Sum Fdbk/Disabled'
//  '<S638>' : 'super/WAYPOINT CONTROLLER/WP_NAV/heading_controller/heading_controller/PID Controller2/Tracking Mode/Disabled'
//  '<S639>' : 'super/WAYPOINT CONTROLLER/WP_NAV/heading_controller/heading_controller/PID Controller2/Tracking Mode Sum/Passthrough'
//  '<S640>' : 'super/WAYPOINT CONTROLLER/WP_NAV/heading_controller/heading_controller/PID Controller2/Tsamp - Integral/Passthrough'
//  '<S641>' : 'super/WAYPOINT CONTROLLER/WP_NAV/heading_controller/heading_controller/PID Controller2/Tsamp - Ngain/Passthrough'
//  '<S642>' : 'super/WAYPOINT CONTROLLER/WP_NAV/heading_controller/heading_controller/PID Controller2/postSat Signal/Forward_Path'
//  '<S643>' : 'super/WAYPOINT CONTROLLER/WP_NAV/heading_controller/heading_controller/PID Controller2/preSat Signal/Forward_Path'
//  '<S644>' : 'super/WAYPOINT CONTROLLER/WP_NAV/heading_controller/heading_controller/heading_error/Compare To Constant'
//  '<S645>' : 'super/WAYPOINT CONTROLLER/capture rising edge/Detect Change1'
//  '<S646>' : 'super/WAYPOINT CONTROLLER/determine target/Compare To Constant'
//  '<S647>' : 'super/WAYPOINT CONTROLLER/determine target/calc_prev_target_pos'
//  '<S648>' : 'super/WAYPOINT CONTROLLER/determine target/determine_current_tar_pos'
//  '<S649>' : 'super/WAYPOINT CONTROLLER/determine target/determine_target'
//  '<S650>' : 'super/WAYPOINT CONTROLLER/determine target/calc_prev_target_pos/determine_prev_tar_pos'
//  '<S651>' : 'super/WAYPOINT CONTROLLER/wp_completion_check/check_wp_reached'
//  '<S652>' : 'super/cmd to raw pwm/engine_PWM_denormalize'
//  '<S653>' : 'super/cmd to raw pwm/motor_PWM_denormalize'
//  '<S654>' : 'super/cmd to raw pwm/engine_PWM_denormalize/remap'
//  '<S655>' : 'super/cmd to raw pwm/motor_PWM_denormalize/remap'
//  '<S656>' : 'super/command selection/e_stop_norm'
//  '<S657>' : 'super/command selection/engine_cmd_norm'
//  '<S658>' : 'super/command selection/mode_norm'
//  '<S659>' : 'super/command selection/pitch_norm_deadband'
//  '<S660>' : 'super/command selection/relay_norm'
//  '<S661>' : 'super/command selection/roll_norm_deadband'
//  '<S662>' : 'super/command selection/rtl_norm'
//  '<S663>' : 'super/command selection/throttle_norm_no_deadband'
//  '<S664>' : 'super/command selection/yaw_norm_deadband'
//  '<S665>' : 'super/command selection/e_stop_norm/remap'
//  '<S666>' : 'super/command selection/engine_cmd_norm/remap'
//  '<S667>' : 'super/command selection/mode_norm/remap'
//  '<S668>' : 'super/command selection/pitch_norm_deadband/remap_with_deadband'
//  '<S669>' : 'super/command selection/relay_norm/remap'
//  '<S670>' : 'super/command selection/roll_norm_deadband/remap_with_deadband'
//  '<S671>' : 'super/command selection/rtl_norm/remap'
//  '<S672>' : 'super/command selection/throttle_norm_no_deadband/remap'
//  '<S673>' : 'super/command selection/yaw_norm_deadband/remap_with_deadband'
//  '<S674>' : 'super/determine arm and mode selection/Failsafe_management'
//  '<S675>' : 'super/determine arm and mode selection/auto_disarm'
//  '<S676>' : 'super/determine arm and mode selection/compare_e_stop > 0.5'
//  '<S677>' : 'super/determine arm and mode selection/compare_rtl cmd > 0.5'
//  '<S678>' : 'super/determine arm and mode selection/compare_to_land'
//  '<S679>' : 'super/determine arm and mode selection/determine submode'
//  '<S680>' : 'super/determine arm and mode selection/yaw_stick_arming'
//  '<S681>' : 'super/determine arm and mode selection/Failsafe_management/Battery failsafe'
//  '<S682>' : 'super/determine arm and mode selection/Failsafe_management/Radio failsafe'
//  '<S683>' : 'super/determine arm and mode selection/Failsafe_management/Battery failsafe/Compare To Constant'
//  '<S684>' : 'super/determine arm and mode selection/Failsafe_management/Battery failsafe/Compare To Constant3'
//  '<S685>' : 'super/determine arm and mode selection/Failsafe_management/Battery failsafe/disarm motor'
//  '<S686>' : 'super/determine arm and mode selection/Failsafe_management/Battery failsafe/disarm motor/Compare To Constant2'
//  '<S687>' : 'super/determine arm and mode selection/Failsafe_management/Radio failsafe/Compare To Constant'
//  '<S688>' : 'super/determine arm and mode selection/Failsafe_management/Radio failsafe/Compare To Constant1'
//  '<S689>' : 'super/determine arm and mode selection/Failsafe_management/Radio failsafe/Compare To Constant2'
//  '<S690>' : 'super/determine arm and mode selection/auto_disarm/Compare To Constant'
//  '<S691>' : 'super/determine arm and mode selection/auto_disarm/Compare To Constant1'
//  '<S692>' : 'super/determine arm and mode selection/auto_disarm/disarm motor'
//  '<S693>' : 'super/determine arm and mode selection/auto_disarm/disarm motor/Compare To Constant2'
//  '<S694>' : 'super/determine arm and mode selection/determine submode/compare_to_rtl_mode'
//  '<S695>' : 'super/determine arm and mode selection/determine submode/compare_to_wp_mode'
//  '<S696>' : 'super/determine arm and mode selection/determine submode/rtl submodes'
//  '<S697>' : 'super/determine arm and mode selection/determine submode/waypoint submodes'
//  '<S698>' : 'super/determine arm and mode selection/determine submode/rtl submodes/determine_fast_rtl_mode'
//  '<S699>' : 'super/determine arm and mode selection/determine submode/waypoint submodes/determine_target_pos'
//  '<S700>' : 'super/determine arm and mode selection/determine submode/waypoint submodes/determine_wp_submode'
//  '<S701>' : 'super/determine arm and mode selection/yaw_stick_arming/Compare To Constant1'
//  '<S702>' : 'super/determine arm and mode selection/yaw_stick_arming/Compare To Constant2'
//  '<S703>' : 'super/determine arm and mode selection/yaw_stick_arming/Compare To Constant3'
//  '<S704>' : 'super/determine arm and mode selection/yaw_stick_arming/Compare To Constant4'
//  '<S705>' : 'super/determine arm and mode selection/yaw_stick_arming/Enabled Subsystem'
//  '<S706>' : 'super/determine arm and mode selection/yaw_stick_arming/manual_arming'
//  '<S707>' : 'super/initial_motor_ramp/Compare To Constant'


//-
//  Requirements for '<Root>': super

#endif                                 // RTW_HEADER_autocode_h_

//
// File trailer for generated code.
//
// [EOF]
//
