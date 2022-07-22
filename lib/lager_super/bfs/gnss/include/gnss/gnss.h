/*
* Brian R Taylor
* brian.taylor@bolderflight.com
* 
* Copyright (c) 2021 Bolder Flight Systems Inc
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the “Software”), to
* deal in the Software without restriction, including without limitation the
* rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
* sell copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
* IN THE SOFTWARE.
*/

#ifndef INCLUDE_GNSS_GNSS_H_
#define INCLUDE_GNSS_GNSS_H_


namespace bfs {

enum GnssFix : int8_t {
  GNSS_FIX_NONE = 1,
  GNSS_FIX_2D = 2,
  GNSS_FIX_3D = 3,
  GNSS_FIX_DGNSS = 4,
  GNSS_FIX_RTK_FLOAT = 5,
  GNSS_FIX_RTK_FIXED = 6
};

struct GnssData {
  bool new_data;
  bool healthy;
  int8_t fix;
  int8_t num_sats;
  int16_t week;
  int32_t tow_ms;
  float alt_wgs84_m;
  float alt_msl_m;
  float hdop;
  float vdop;
  float track_rad;
  float spd_mps;
  float horz_acc_m;
  float vert_acc_m;
  float vel_acc_mps;
  float track_acc_rad;
  float ned_vel_mps[3];
  double lat_rad;
  double lon_rad;
};

}  // namespace bfs

#endif  // INCLUDE_GNSS_GNSS_H_
