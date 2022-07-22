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

#ifndef INCLUDE_GLOBAL_DEFS_GLOBAL_DEFS_H_
#define INCLUDE_GLOBAL_DEFS_GLOBAL_DEFS_H_

#include <cstdint>

namespace bfs {

/* Maximum poly_coef size */
static constexpr std::size_t MAX_POLY_COEF_SIZE = 10;

enum AircraftType : int8_t {
  FIXED_WING = 0,
  HELICOPTER = 1,
  MULTIROTOR = 2,
  VTOL = 3
};

enum AircraftState : int8_t {
  INIT = 0,
  STANDBY = 1,
  ACTIVE = 2,
  CAUTION = 3,
  EMERGENCY = 4,
  FTS = 5
};

enum AircraftMode : int8_t {
  MANUAL = 0,
  STABALIZED = 1,
  ATTITUDE = 2,
  AUTO = 3,
  TEST = 4
};

struct MissionItem {
  bool autocontinue;
  uint8_t frame;
  uint16_t cmd;
  float param1;
  float param2;
  float param3;
  float param4;
  int32_t x;
  int32_t y;
  float z;
};

}  // namespace bfs

#endif  // INCLUDE_GLOBAL_DEFS_GLOBAL_DEFS_H_
