#include <color_spinor_field.h>
#include <string.h>
#include <iostream>
#include <typeinfo>
#include <face_quda.h>

namespace quda {


  EigParam::EigParam(const EigValueSet &a) {
    a.fill(*this);
  }
  
  EigParam::EigParam(const EigVecSet &a) {
    a.fill(*this);
  }

  //Class definitions of EigValueSet
  EigValueSet::EigValueSet(const EigParam &param) : verbose(param.verbose), init(false), v(0) 
  {
    self_param = param;
    create();
  }

  EigValueSet::~EigValueSet() {
    destroy();
  }

  void EigValueSet::create() 
  {
    init = true;
  }

  void EigValueSet::destroy() {
    init = false;
  }

  EigValueSet& EigValueSet::operator=(const EigValueSet &src) {
    if (&src != this) {
      self_param = src.self_param;
      create();
    }
    return *this;
  }
  
  // Fills the param with the contents of this field
  void EigValueSet::fill(EigParam &param) const {
    param = (*this).self_param;
  }

  std::ostream& operator<<(std::ostream &out, const EigValueSet &a) {
    out << "nk = " << a.self_param.nk << std::endl;
    out << "np = " << a.self_param.np << std::endl;
    return out;
  }
 
  //Class definitions of EigVecSet
  EigVecSet::EigVecSet(const EigParam &param) : verbose(param.verbose), init(false), v(0) 
  {
    self_param = param;
    create();
  }

  EigVecSet::~EigVecSet() {
    destroy();
  }

  void EigVecSet::create() 
  {
    init = true;
  }

  void EigVecSet::destroy() {
    init = false;
  }

  EigVecSet& EigVecSet::operator=(const EigVecSet &src) {
    if (&src != this) {
      self_param = src.self_param;
      create();
    }
    return *this;
  }
  
  // Fills the param with the contents of this field
  void EigVecSet::fill(EigParam &param) const {
    param = (*this).self_param;
  }

  std::ostream& operator<<(std::ostream &out, const EigVecSet &a) {
    out << "nk = " << a.self_param.nk << std::endl;
    out << "np = " << a.self_param.np << std::endl;
    out << "fsize = " << a.self_param.fsize << std::endl;
    return out;
  }
} // namespace quda
