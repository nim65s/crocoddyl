///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CORE_STATE_HPP_
#define CORE_STATE_HPP_

#include <Eigen/Dense>


namespace crocoddyl {

template <class ConfigVectorType, class TangentVectorType>
class StateAbstract;

class StateAbstract_tpl {
 public:
    StateAbstract_tpl() {}
    virtual ~StateAbstract_tpl() {}  // = default;

    template <class ConfigVectorType, class TangentVectorType>
    TangentVectorType diff(const ConfigVectorType& x0, const ConfigVectorType& x1) {
        StateAbstract<ConfigVectorType, TangentVectorType>& self = dynamic_cast<
            StateAbstract<ConfigVectorType, TangentVectorType>& >(*this);
        return self.diff(x0, x1);
    }
};

template <class ConfigVectorType, class TangentVectorType>
class StateAbstract : public StateAbstract_tpl {
 public:
    StateAbstract(int nx, int ndx) : nx_(nx), ndx_(ndx) {}
    virtual ~StateAbstract() {}  // = default;
    // StateAbstract(const T& o) : obj(o) {std::cout << "hey" << std::endl;}
    // StateAbstract(T&& o) : obj(std::move(o)) {std::cout << "hey0" <<  o << std::endl;}
    // StateAbstract() : obj(std::move(0)) {std::cout << "hey1" << std::endl;}

    virtual TangentVectorType diff(const ConfigVectorType& x0, const ConfigVectorType& x1) = 0;
    virtual ConfigVectorType integrate(const ConfigVectorType& x0, const TangentVectorType& dx) = 0;

    // virtual ConfigVectorType Jdiff(const ConfigVectorType& x0, const ConfigVectorType& x1) = 0;

    int nx() const {
        return nx_;
    }

    int ndx() const {
        return ndx_;
    }

    void printer(const ConfigVectorType& x0, const ConfigVectorType& x1) {
        std::cout << diff(x0, x1) << std::endl;
    }

 private:
    int nx_;
    int ndx_;
};

}  // namespace crocoddyl

#endif  // CORE_STATE_HPP_
