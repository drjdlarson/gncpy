#pragma once

namespace lager::gncpy::dynamics {

class StateTransParams {
public:
    virtual ~StateTransParams() = default;
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
