#ifndef FLIGHT_CODE_INCLUDE_FLIGHT_GLOBAL_DEFS_H_
#define FLIGHT_CODE_INCLUDE_FLIGHT_GLOBAL_DEFS_H_

#include <cstddef>
#include <cstdint>
#include <array>

#include "flight/hardware_defs.h"
#include "imu/imu.h"
#include "gnss/gnss.h"
#include "pres/pres.h"
#include "global_defs/global_defs.h"

/* Control sizes */
inline constexpr std::size_t NUM_AUX_VAR = 24;

/* Telem sizes */
inline constexpr std::size_t NUM_TELEM_PARAMS = 24;
#if defined(__FMU_R_V2__) || defined(__FMU_R_V2_BETA__)
inline constexpr std::size_t NUM_FLIGHT_PLAN_POINTS = 500;
inline constexpr std::size_t NUM_FENCE_POINTS = 100;
inline constexpr std::size_t NUM_RALLY_POINTS = 10;
#endif
#if defined(__FMU_R_V1__)
inline constexpr std::size_t NUM_FLIGHT_PLAN_POINTS = 100;
inline constexpr std::size_t NUM_FENCE_POINTS = 50;
inline constexpr std::size_t NUM_RALLY_POINTS = 10;
#endif


/* System data */
struct SysData {
  int32_t frame_time_us;
  #if defined(__FMU_R_V1__)
  float input_volt;
  float reg_volt;
  float pwm_volt;
  float sbus_volt;
  #endif
  int64_t sys_time_us;
};
/* Analog data */
struct AdcData {
  std::array<float, NUM_AIN_PINS> volt;
};
/* Power module data */
#if defined(__FMU_R_V2__)
struct PowerModuleData {
  float voltage_v;
  float current_v;
};
#endif
/* Inceptor data */
struct InceptorData {
  bool new_data;
  bool lost_frame;
  bool failsafe;
  bool ch17;
  bool ch18;
  std::array<int16_t, NUM_SBUS_CH> ch;
};
/* Sensor data */
struct SensorData {
  bool pitot_static_installed;
  InceptorData inceptor;
  bfs::ImuData imu;
  bfs::GnssData gnss;
  bfs::PresData static_pres;
  bfs::PresData diff_pres;
  AdcData adc;
  #if defined(__FMU_R_V2__)
  PowerModuleData power_module;
  #endif
};
/* Nav data */
struct NavData {
  bool nav_initialized;
  float pitch_rad;
  float roll_rad;
  float heading_rad;
  float alt_wgs84_m;
  float home_alt_wgs84_m;
  float alt_msl_m;
  float alt_rel_m;
  float static_pres_pa;
  float diff_pres_pa;
  float alt_pres_m;
  float ias_mps;
  float gnd_spd_mps;
  float gnd_track_rad;
  float flight_path_rad;
  std::array<float, 3> accel_bias_mps2;
  std::array<float, 3> gyro_bias_radps;
  std::array<float, 3> accel_mps2;
  std::array<float, 3> gyro_radps;
  std::array<float, 3> mag_ut;
  std::array<float, 3> ned_pos_m;
  std::array<float, 3> ned_vel_mps;
  double lat_rad;
  double lon_rad;
  double home_lat_rad;
  double home_lon_rad;
};
/* SBUS data */
struct SbusCmd {
  bool ch17;
  bool ch18;
  std::array<int16_t, NUM_SBUS_CH> cnt;
  std::array<float, NUM_SBUS_CH> cmd;
};
/* PWM data */
struct PwmCmd {
  std::array<int16_t, NUM_PWM_PINS> cnt;
  std::array<float, NUM_PWM_PINS> cmd;
};
/* Analog data */
struct AnalogData {
  std::array<float, NUM_AIN_PINS> val;
};
#if defined(__FMU_R_V2__)
/* Battery data */
struct BatteryData {
  float voltage_v;
  float current_ma;
  float consumed_mah;
  float remaining_prcnt;
  float remaining_time_s;
};
#endif
/* VMS data */
struct VmsData {
  bool motors_enabled;
  bool waypoint_reached;
  int8_t mode;
  float throttle_cmd_prcnt;
  std::array<float, NUM_AUX_VAR> aux;
  SbusCmd sbus;
  PwmCmd pwm;
  AnalogData analog;
  #if defined(__FMU_R_V2__)
  BatteryData battery;
  #endif
};
/* Telemetry data */
struct TelemData {
  bool waypoints_updated;
  bool fence_updated;
  bool rally_points_updated;
  int16_t current_waypoint;
  int16_t num_waypoints;
  int16_t num_fence_items;
  int16_t num_rally_points;
  std::array<float, NUM_TELEM_PARAMS> param;
  std::array<bfs::MissionItem, NUM_FLIGHT_PLAN_POINTS> flight_plan;
  std::array<bfs::MissionItem, NUM_FENCE_POINTS> fence;
  std::array<bfs::MissionItem, NUM_RALLY_POINTS> rally;
};

#endif  // FLIGHT_CODE_INCLUDE_FLIGHT_GLOBAL_DEFS_H_
