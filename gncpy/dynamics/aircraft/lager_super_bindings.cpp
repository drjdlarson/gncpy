#ifndef LAGER_SUPER_BINDINGS_H
#define LAGER_SUPER_BINDINGS_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/pytypes.h>
#include <pybind11/numpy.h>
#include <cstdint>

#include "autocode.h"
#include "imu/imu.h"
#include "gnss/gnss.h"
#include "pres/pres.h"
#include "global_defs/global_defs.h"


namespace py = pybind11;
using namespace bfs;

PYBIND11_MODULE(lager_super_bindings, m) {
    PYBIND11_NUMPY_DTYPE(MissionItem, autocontinue, frame, cmd, param1, param2,
                         param3, param4, x, y, z);

    m.doc() = "Wrapper for simulink autocode of LAGER's SUPER UAV control system.";

    m.attr("NUM_SBUS_CH") = NUM_SBUS_CH;

    py::enum_<GnssFix>(m, "GnssFix")
        .value("GNSS_FIX_NONE", GNSS_FIX_NONE)
        .value("GNSS_FIX_2D", GNSS_FIX_2D)
        .value("GNSS_FIX_3D", GNSS_FIX_3D)
        .value("GNSS_FIX_DGNSS", GNSS_FIX_DGNSS)
        .value("GNSS_FIX_RTK_FLOAT", GNSS_FIX_RTK_FLOAT)
        .value("GNSS_FIX_RTK_FIXED", GNSS_FIX_RTK_FIXED);

    py::class_<Autocode>(m, "Autocode")
        .def(py::init<>())
        .def("initialize", &Autocode::initialize)
        .def("step", &Autocode::Run);

    py::class_<SysData>(m, "SysData")
        .def(py::init<>())
        .def_readwrite("frame_time_us", &SysData::frame_time_us)
        .def_readwrite("sys_time_us", &SysData::sys_time_us);

    py::class_<AdcData>(m, "AdcData")
        .def(py::init<>())
        .def_property("volt", [](AdcData &dat) -> py::array {
            auto dtype = py::dtype(py::format_descriptor<float>::format());
            auto base = py::array(dtype, {NUM_AIN_PINS}, {sizeof(float)});
            return py::array(dtype, {NUM_AIN_PINS}, {sizeof(float)}, dat.volt.data(), base);
        }, [](AdcData& dat, py::array_t<float> val) {
            for(int ii = 0; ii < NUM_AIN_PINS; ii++) {
                dat.volt[ii] = *(val.data(ii));
            }
        });

    py::class_<PowerModuleData>(m, "PowerModuleData")
        .def(py::init<>())
        .def_readwrite("voltage_v", &PowerModuleData::voltage_v)
        .def_readwrite("current_v", &PowerModuleData::current_v);

    py::class_<InceptorData>(m, "InceptorData")
        .def(py::init<>())
        .def_readwrite("new_data", &InceptorData::new_data)
        .def_readwrite("lost_frame", &InceptorData::lost_frame)
        .def_readwrite("failsafe", &InceptorData::failsafe)
        .def_readwrite("ch17", &InceptorData::ch17)
        .def_readwrite("ch18", &InceptorData::ch18)
        .def_property("ch", [](InceptorData &dat) -> py::array {
            auto dtype = py::dtype(py::format_descriptor<int16_t>::format());
            auto base = py::array(dtype, {NUM_SBUS_CH}, {sizeof(int16_t)});
            return py::array(dtype, {NUM_SBUS_CH}, {sizeof(int16_t)}, dat.ch.data(), base);
        }, [](InceptorData& dat, py::array_t<int16_t> val) {
            for(int ii = 0; ii < NUM_SBUS_CH; ii++) {
                dat.ch[ii] = *(val.data(ii));
            }
        });

    py::class_<ImuData>(m, "ImuData")
        .def(py::init<>())
        .def_readwrite("new_data", &ImuData::new_data)
        .def_readwrite("healthy", &ImuData::healthy)
        .def_readwrite("die_temp_c", &ImuData::die_temp_c)
        .def_property("accel_mps2", [](ImuData &i) -> py::array {
            auto dtype = py::dtype(py::format_descriptor<float>::format());
            auto base = py::array(dtype, {3}, {sizeof(float)});
            return py::array(dtype, {3}, {sizeof(float)}, i.accel_mps2, base);
        }, [](ImuData& i, py::array_t<float> val) {
            i.accel_mps2[0] = *(val.data(0));
            i.accel_mps2[1] = *(val.data(1));
            i.accel_mps2[2] = *(val.data(2));
        })
        .def_property("gyro_radps", [](ImuData &i) -> py::array {
            auto dtype = py::dtype(py::format_descriptor<float>::format());
            auto base = py::array(dtype, {3}, {sizeof(float)});
            return py::array(dtype, {3}, {sizeof(float)}, i.gyro_radps, base);
        }, [](ImuData& i, py::array_t<float> val) {
            i.gyro_radps[0] = *(val.data(0));
            i.gyro_radps[1] = *(val.data(1));
            i.gyro_radps[2] = *(val.data(2));
        });

    /*
    py::class_<GnssRelPosData>(m, "GnssRelPosData")
        .def(py::init<>())
        .def_readwrite("valid", &GnssRelPosData::valid)
        .def_readwrite("baseline_acc_m", &GnssRelPosData::baseline_acc_m)
        .def_property("ned_pos_acc_m", [](GnssRelPosData &i) -> py::array {
            auto dtype = py::dtype(py::format_descriptor<float>::format());
            auto base = py::array(dtype, {3}, {sizeof(float)});
            return py::array(dtype, {3}, {sizeof(float)}, i.ned_pos_acc_m, base);
        }, [](GnssRelPosData& i, py::array_t<float> val) {
            i.ned_pos_acc_m[0] = *(val.data(0));
            i.ned_pos_acc_m[1] = *(val.data(1));
            i.ned_pos_acc_m[2] = *(val.data(2));
        })
        .def_readwrite("baseline_m", &GnssRelPosData::baseline_m)
        .def_property("ned_pos_m", [](GnssRelPosData &i) -> py::array {
            auto dtype = py::dtype(py::format_descriptor<float>::format());
            auto base = py::array(dtype, {3}, {sizeof(float)});
            return py::array(dtype, {3}, {sizeof(float)}, i.ned_pos_m, base);
        }, [](GnssRelPosData& i, py::array_t<float> val) {
            i.ned_pos_m[0] = *(val.data(0));
            i.ned_pos_m[1] = *(val.data(1));
            i.ned_pos_m[2] = *(val.data(2));
        });
*/

    py::class_<GnssData>(m, "GnssData")
        .def(py::init<>())
        .def_readwrite("new_data", &GnssData::new_data)
        .def_readwrite("healthy", &GnssData::healthy)
        .def_readwrite("fix", &GnssData::fix)
        .def_readwrite("num_sats", &GnssData::num_sats)
        .def_readwrite("week", &GnssData::week)
        .def_readwrite("tow_ms", &GnssData::tow_ms)
        .def_readwrite("alt_wgs84_m", &GnssData::alt_wgs84_m)
        .def_readwrite("alt_msl_m", &GnssData::alt_msl_m)
        //.def_readwrite("gdop", &GnssData::gdop)
        //.def_readwrite("pdop", &GnssData::pdop)
        //.def_readwrite("tdop", &GnssData::tdop)
        .def_readwrite("hdop", &GnssData::hdop)
        .def_readwrite("vdop", &GnssData::vdop)
        //.def_readwrite("ndop", &GnssData::ndop)
        //.def_readwrite("edop", &GnssData::edop)
        //.def_readwrite("time_acc_s", &GnssData::time_acc_s)
        .def_readwrite("track_rad", &GnssData::track_rad)
        .def_readwrite("spd_mps", &GnssData::spd_mps)
        .def_readwrite("horz_acc_m", &GnssData::horz_acc_m)
        .def_readwrite("vert_acc_m", &GnssData::vert_acc_m)
        .def_readwrite("vel_acc_mps", &GnssData::vel_acc_mps)
        .def_readwrite("track_acc_rad", &GnssData::track_acc_rad)
        .def_property("ned_vel_mps", [](GnssData &i) -> py::array {
            auto dtype = py::dtype(py::format_descriptor<float>::format());
            auto base = py::array(dtype, {3}, {sizeof(float)});
            return py::array(dtype, {3}, {sizeof(float)}, i.ned_vel_mps, base);
        }, [](GnssData& i, py::array_t<float> val) {
            i.ned_vel_mps[0] = *(val.data(0));
            i.ned_vel_mps[1] = *(val.data(1));
            i.ned_vel_mps[2] = *(val.data(2));
        })
        .def_readwrite("lat_rad", &GnssData::lat_rad)
        .def_readwrite("lon_rad", &GnssData::lon_rad);
        //.def_readwrite("rel_pos", &GnssData::rel_pos);

    py::class_<PresData>(m, "PresData")
        .def(py::init<>())
        .def_readwrite("new_data", &PresData::new_data)
        .def_readwrite("healthy", &PresData::healthy)
        .def_readwrite("pres_pa", &PresData::pres_pa)
        .def_readwrite("die_temp_c", &PresData::die_temp_c);

    py::class_<SensorData>(m, "SensorData")
        .def(py::init<>())
        .def_readwrite("pitot_static_installed",
                       &SensorData::pitot_static_installed)
        .def_readwrite("inceptor", &SensorData::inceptor)
        .def_readwrite("imu", &SensorData::imu)
        .def_readwrite("gnss", &SensorData::gnss)
        .def_readwrite("static_pres", &SensorData::static_pres)
        .def_readwrite("static_pres", &SensorData::static_pres)
        .def_readwrite("diff_pres", &SensorData::diff_pres)
        .def_readwrite("adc", &SensorData::adc)
        .def_readwrite("power_module", &SensorData::power_module);

    py::class_<NavData>(m, "NavData")
        .def(py::init<>())
        .def_readwrite("nav_initialized", &NavData::nav_initialized)
        .def_readwrite("pitch_rad", &NavData::pitch_rad)
        .def_readwrite("roll_rad", &NavData::roll_rad)
        .def_readwrite("heading_rad", &NavData::heading_rad)
        .def_readwrite("alt_wgs84_m", &NavData::alt_wgs84_m)
        .def_readwrite("home_alt_wgs84_m", &NavData::home_alt_wgs84_m)
        .def_readwrite("alt_msl_m", &NavData::alt_msl_m)
        .def_readwrite("alt_rel_m", &NavData::alt_rel_m)
        .def_readwrite("static_pres_pa", &NavData::static_pres_pa)
        .def_readwrite("diff_pres_pa", &NavData::diff_pres_pa)
        .def_readwrite("alt_pres_m", &NavData::alt_pres_m)
        .def_readwrite("ias_mps", &NavData::ias_mps)
        .def_readwrite("gnd_spd_mps", &NavData::gnd_spd_mps)
        .def_readwrite("gnd_track_rad", &NavData::gnd_track_rad)
        .def_readwrite("flight_path_rad", &NavData::flight_path_rad)
        .def_property("accel_bias_mps2", [](NavData &dat) -> py::array {
            auto dtype = py::dtype(py::format_descriptor<float>::format());
            auto base = py::array(dtype, {3}, {sizeof(float)});
            return py::array(dtype, {3}, {sizeof(float)}, dat.accel_bias_mps2.data(), base);
        }, [](NavData& dat, py::array_t<float> val) {
            for(int ii = 0; ii < 3; ii++) {
                dat.accel_bias_mps2[ii] = *(val.data(ii));
            }
        })
        .def_property("gyro_bias_radps", [](NavData &dat) -> py::array {
            auto dtype = py::dtype(py::format_descriptor<float>::format());
            auto base = py::array(dtype, {3}, {sizeof(float)});
            return py::array(dtype, {3}, {sizeof(float)}, dat.gyro_bias_radps.data(), base);
        }, [](NavData& dat, py::array_t<float> val) {
            for(int ii = 0; ii < 3; ii++) {
                dat.gyro_bias_radps[ii] = *(val.data(ii));
            }
        })
        .def_property("accel_mps2", [](NavData &dat) -> py::array {
            auto dtype = py::dtype(py::format_descriptor<float>::format());
            auto base = py::array(dtype, {3}, {sizeof(float)});
            return py::array(dtype, {3}, {sizeof(float)}, dat.accel_mps2.data(), base);
        }, [](NavData& dat, py::array_t<float> val) {
            for(int ii = 0; ii < 3; ii++) {
                dat.accel_mps2[ii] = *(val.data(ii));
            }
        })
        .def_property("gyro_radps", [](NavData &dat) -> py::array {
            auto dtype = py::dtype(py::format_descriptor<float>::format());
            auto base = py::array(dtype, {3}, {sizeof(float)});
            return py::array(dtype, {3}, {sizeof(float)}, dat.gyro_radps.data(), base);
        }, [](NavData& dat, py::array_t<float> val) {
            for(int ii = 0; ii < 3; ii++) {
                dat.gyro_radps[ii] = *(val.data(ii));
            }
        })
        .def_property("mag_ut", [](NavData &dat) -> py::array {
            auto dtype = py::dtype(py::format_descriptor<float>::format());
            auto base = py::array(dtype, {3}, {sizeof(float)});
            return py::array(dtype, {3}, {sizeof(float)}, dat.mag_ut.data(), base);
        }, [](NavData& dat, py::array_t<float> val) {
            for(int ii = 0; ii < 3; ii++) {
                dat.mag_ut[ii] = *(val.data(ii));
            }
        })
        .def_property("ned_pos_m", [](NavData &dat) -> py::array {
            auto dtype = py::dtype(py::format_descriptor<float>::format());
            auto base = py::array(dtype, {3}, {sizeof(float)});
            return py::array(dtype, {3}, {sizeof(float)}, dat.ned_pos_m.data(), base);
        }, [](NavData& dat, py::array_t<float> val) {
            for(int ii = 0; ii < 3; ii++) {
                dat.ned_pos_m[ii] = *(val.data(ii));
            }
        })
        .def_property("ned_vel_mps", [](NavData &dat) -> py::array {
            auto dtype = py::dtype(py::format_descriptor<float>::format());
            auto base = py::array(dtype, {3}, {sizeof(float)});
            return py::array(dtype, {3}, {sizeof(float)}, dat.ned_vel_mps.data(), base);
        }, [](NavData& dat, py::array_t<float> val) {
            for(int ii = 0; ii < 3; ii++) {
                dat.ned_vel_mps[ii] = *(val.data(ii));
            }
        })
        .def_readwrite("lat_rad", &NavData::lat_rad)
        .def_readwrite("lon_rad", &NavData::lon_rad)
        .def_readwrite("home_lat_rad", &NavData::home_lat_rad)
        .def_readwrite("home_lon_rad", &NavData::home_lon_rad);

    py::class_<MissionItem>(m, "MissionItem")
        .def(py::init([]() { return MissionItem(); }))
        .def_readwrite("autocontinue", &MissionItem::autocontinue)
        .def_readwrite("frame", &MissionItem::frame)
        .def_readwrite("cmd", &MissionItem::cmd)
        .def_readwrite("param1", &MissionItem::param1)
        .def_readwrite("param2", &MissionItem::param2)
        .def_readwrite("param3", &MissionItem::param3)
        .def_readwrite("param4", &MissionItem::param4)
        .def_readwrite("x", &MissionItem::x)
        .def_readwrite("y", &MissionItem::y)
        .def_readwrite("z", &MissionItem::z)
        .def("astuple",
             [](const MissionItem &self) {
                 return py::make_tuple(self.autocontinue, self.frame, self.cmd,
                                       self.param1, self.param2, self.param3,
                                       self.param4, self.x, self.y, self.z);
             })
        .def_static("fromtuple", [](const py::tuple &tup) {
            if (py::len(tup) != 10) {
                throw py::cast_error("Invalid size");
            }
            return MissionItem{tup[0].cast<bool>(),
                               tup[1].cast<uint8_t>(),
                               tup[2].cast<uint16_t>(),
                               tup[3].cast<float>(),
                               tup[4].cast<float>(),
                               tup[5].cast<float>(),
                               tup[6].cast<float>(),
                               tup[7].cast<int32_t>(),
                               tup[8].cast<int32_t>(),
                               tup[9].cast<float>()};
        });

    py::class_<TelemData>(m, "TelemData")
        .def(py::init<>())
        .def_readwrite("waypoints_updated", &TelemData::waypoints_updated)
        .def_readwrite("fence_updated", &TelemData::fence_updated)
        .def_readwrite("rally_points_updated", &TelemData::rally_points_updated)
        .def_readwrite("current_waypoint", &TelemData::current_waypoint)
        .def_readwrite("num_waypoints", &TelemData::num_waypoints)
        .def_readwrite("num_fence_items", &TelemData::num_fence_items)
        .def_readwrite("num_rally_points", &TelemData::num_rally_points)
        .def_property("param", [](TelemData &dat) -> py::array {
            auto dtype = py::dtype(py::format_descriptor<float>::format());
            auto base = py::array(dtype, {NUM_TELEM_PARAMS}, {sizeof(float)});
            return py::array(dtype, {NUM_TELEM_PARAMS}, {sizeof(float)}, dat.param.data(), base);
        }, [](TelemData& dat, py::array_t<float> val) {
            for(int ii = 0; ii < NUM_TELEM_PARAMS; ii++) {
                dat.param[ii] = *(val.data(ii));
            }
        })
        .def_property("flight_plan", [](TelemData &dat) -> py::array {
            auto arr = py::array(py::buffer_info(nullptr, sizeof(MissionItem),
                                                 py::format_descriptor<MissionItem>::format(),
                                                 1, {dat.flight_plan.size()},
                                                 {sizeof(MissionItem)}));
            auto req = arr.request();
            auto *ptr = static_cast<MissionItem *>(req.ptr);
            for(int ii = 0; ii < dat.flight_plan.size(); ii++) {
                ptr[ii] = dat.flight_plan[ii];
            }
            return arr;
            /*
            auto dtype = py::dtype(py::format_descriptor<MissionItem>::format());
            auto base = py::array(dtype, {dat.flight_plan.size()}, {sizeof(MissionItem)});
            return py::array(dtype, {dat.flight_plan.size()}, {sizeof(MissionItem)}, dat.flight_plan.data(), base);*/
            /*
            auto lst = py::list();
            for(int ii = 0; ii < dat.flight_plan.size(); ii++) {
                lst.append(dat.flight_plan[ii]);
            }
            return py::array(lst);*/
        }, [](TelemData& dat, py::list val) {
            int ii = 0;
            for(auto item : val) {
                dat.flight_plan[ii] = item.cast<MissionItem>();
                ii += 1;
            }
/*
            py::array_t<MissionItem> val_arr = py::array(val);
            for(int ii = 0; ii < (val_arr.size() > NUM_FLIGHT_PLAN_POINTS ? NUM_FLIGHT_PLAN_POINTS : val_arr.size()); ii++) {
                dat.flight_plan[ii] = *(val_arr.data(ii));
            }*/

            /*
            //auto lst = py::list();
            for(int ii = 0; ii < (val.size() > NUM_FLIGHT_PLAN_POINTS ? NUM_FLIGHT_PLAN_POINTS : val.size()); ii++) {
                dat.flight_plan[ii] = *(val.data(ii));
            }*/
            //dat.flight_plan = py::array(lst);
            /*
            for(int ii = 0; ii < NUM_FLIGHT_PLAN_POINTS > val.size() ? NUM_FLIGHT_PLAN_POINTS : val.size(); ii++) {
                dat.flight_plan[ii] = *(val.data(ii));
            }
            if(val.size() < NUM_FLIGHT_PLAN_POINTS) {
                for(int ii = val.size(); ii < NUM_FLIGHT_PLAN_POINTS; ii++) {
                    MissionItem item;
                    item.autocontinue = false;
                    item.frame = 0;
                    item.cmd = 0;
                    item.param1 = 0;
                    item.param2 = 0;
                    item.param3 = 0;
                    item.param4 = 0;
                    item.x = 0;
                    item.y = 0;
                    item.z = 0;
                    dat.flight_plan[ii] = item;

                }
            }*/
        })
        .def_readwrite("fence", &TelemData::fence)
        .def_readwrite("rally", &TelemData::rally);

    py::class_<SbusCmd>(m, "SbusCmd")
        .def(py::init<>())
        .def_readwrite("ch17", &SbusCmd::ch17)
        .def_readwrite("ch18", &SbusCmd::ch18)
        .def_property("cnt", [](SbusCmd &dat) -> py::array {
            auto dtype = py::dtype(py::format_descriptor<int16_t>::format());
            auto base = py::array(dtype, {NUM_SBUS_CH}, {sizeof(int16_t)});
            return py::array(dtype, {NUM_SBUS_CH}, {sizeof(int16_t)}, dat.cnt.data(), base);
        }, [](SbusCmd& dat, py::array_t<int16_t> val) {
            for(int ii = 0; ii < NUM_SBUS_CH; ii++) {
                dat.cnt[ii] = *(val.data(ii));
            }
        })
        .def_property("cmd", [](SbusCmd &dat) -> py::array {
            auto dtype = py::dtype(py::format_descriptor<float>::format());
            auto base = py::array(dtype, {NUM_SBUS_CH}, {sizeof(float)});
            return py::array(dtype, {NUM_SBUS_CH}, {sizeof(float)}, dat.cmd.data(), base);
        }, [](SbusCmd& dat, py::array_t<float> val) {
            for(int ii = 0; ii < NUM_SBUS_CH; ii++) {
                dat.cmd[ii] = *(val.data(ii));
            }
        });

    py::class_<PwmCmd>(m, "PwmCmd")
        .def(py::init<>())
        .def_property("cnt", [](PwmCmd &dat) -> py::array {
            auto dtype = py::dtype(py::format_descriptor<int16_t>::format());
            auto base = py::array(dtype, {NUM_PWM_PINS}, {sizeof(int16_t)});
            return py::array(dtype, {NUM_PWM_PINS}, {sizeof(int16_t)}, dat.cnt.data(), base);
        }, [](PwmCmd& dat, py::array_t<int16_t> val) {
            for(int ii = 0; ii < NUM_PWM_PINS; ii++) {
                dat.cnt[ii] = *(val.data(ii));
            }
        })
        .def_property("cmd", [](PwmCmd &dat) -> py::array {
            auto dtype = py::dtype(py::format_descriptor<float>::format());
            auto base = py::array(dtype, {NUM_PWM_PINS}, {sizeof(float)});
            return py::array(dtype, {NUM_PWM_PINS}, {sizeof(float)}, dat.cmd.data(), base);
        }, [](PwmCmd& dat, py::array_t<float> val) {
            for(int ii = 0; ii < NUM_PWM_PINS; ii++) {
                dat.cmd[ii] = *(val.data(ii));
            }
        });

    py::class_<AnalogData>(m, "AnalogData")
        .def(py::init<>())
        .def_property("val", [](AnalogData &dat) -> py::array {
            auto dtype = py::dtype(py::format_descriptor<float>::format());
            auto base = py::array(dtype, {NUM_AIN_PINS}, {sizeof(float)});
            return py::array(dtype, {NUM_AIN_PINS}, {sizeof(float)}, dat.val.data(), base);
        }, [](AnalogData& dat, py::array_t<float> val) {
            for(int ii = 0; ii < NUM_PWM_PINS; ii++) {
                dat.val[ii] = *(val.data(ii));
            }
        });

    py::class_<BatteryData>(m, "BatteryData")
        .def(py::init<>())
        .def_readwrite("voltage_v", &BatteryData::voltage_v)
        .def_readwrite("current_ma", &BatteryData::current_ma)
        .def_readwrite("consumed_mah", &BatteryData::consumed_mah)
        .def_readwrite("remaining_prcnt", &BatteryData::remaining_prcnt)
        .def_readwrite("remaining_time_s", &BatteryData::remaining_time_s);

    py::class_<VmsData>(m, "VmsData")
        .def(py::init<>())
        .def_readwrite("motors_enabled", &VmsData::motors_enabled)
        .def_readwrite("waypoint_reached", &VmsData::waypoint_reached)
        .def_readwrite("mode", &VmsData::mode)
        .def_readwrite("throttle_cmd_prcnt", &VmsData::throttle_cmd_prcnt)
        .def_property("aux", [](VmsData &dat) -> py::array {
            auto dtype = py::dtype(py::format_descriptor<float>::format());
            auto base = py::array(dtype, {NUM_TELEM_PARAMS}, {sizeof(float)});
            return py::array(dtype, {NUM_TELEM_PARAMS}, {sizeof(float)}, dat.aux.data(), base);
        }, [](VmsData& dat, py::array_t<float> val) {
            for(int ii = 0; ii < NUM_PWM_PINS; ii++) {
                dat.aux[ii] = *(val.data(ii));
            }
        })
        .def_readwrite("sbus", &VmsData::sbus)
        .def_readwrite("pwm", &VmsData::pwm)
        .def_readwrite("analog", &VmsData::analog)
        .def_readwrite("battery", &VmsData::battery);

}

# endif // LAGER_SUPER_BINDINGS_H
