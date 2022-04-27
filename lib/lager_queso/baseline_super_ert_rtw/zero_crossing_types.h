//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// File: zero_crossing_types.h
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

#ifndef ZERO_CROSSING_TYPES_H
#define ZERO_CROSSING_TYPES_H
#include "rtwtypes.h"

// Trigger directions: falling, either, and rising
typedef enum {
  FALLING_ZERO_CROSSING = -1,
  ANY_ZERO_CROSSING = 0,
  RISING_ZERO_CROSSING = 1
} ZCDirection;

// Previous state of a trigger signal
typedef uint8_T ZCSigState;

// Initial value of a trigger zero crossing signal
#define UNINITIALIZED_ZCSIG            0x03U
#define NEG_ZCSIG                      0x02U
#define POS_ZCSIG                      0x01U
#define ZERO_ZCSIG                     0x00U

// Current state of a trigger signal
typedef enum {
  FALLING_ZCEVENT = -1,
  NO_ZCEVENT = 0,
  RISING_ZCEVENT = 1
} ZCEventType;

#endif                                 // ZERO_CROSSING_TYPES_H

//
// File trailer for generated code.
//
// [EOF]
//
