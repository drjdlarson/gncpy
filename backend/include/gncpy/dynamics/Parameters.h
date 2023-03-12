#pragma once

namespace lager::gncpy::dynamics {

class StateParams {
public:
    virtual ~StateParams() = default;
};


class ControlParams {
public:
    virtual ~ControlParams() = default;
};


class ConstraintParams {
public:
    virtual ~ConstraintParams() = default;
};

} // namespace lager::gncpy::dynamics
