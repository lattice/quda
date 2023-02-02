#pragma once

#include <color_spinor_field.h>

namespace quda
{

  /**
     A struct that contains multiple p-vectors which are to be added to an output vector:
     x += alpha[i] * p[i], i = 0, 1, ..., Np - 1
  */
  struct XUpdateBatch {

    int _Np;                           /**< the number of underlying vectors */
    int _j;                            /**< the current index */
    int _next;                         /**< the next index */
    std::vector<ColorSpinorField> _ps; /**< the container for the p-vectors */
    std::vector<double> _alphas;       /**< @param _alphas the alpha's */

    XUpdateBatch() = default;

    /**
      @brief A struct that contains multiple p-vectors which are to be added to an output vector:
        x += alpha[i] * p[i], i = 0, 1, ..., Np - 1
      @param _Np the number of underlying vectors
      @param _j the current index
      @param _next the next index
      @param _ps the container for the p-vectors
      @param _alphas the alpha's
     */
    XUpdateBatch(int Np_, const ColorSpinorField &init, ColorSpinorParam csParam) :
      _Np(Np_), _j(0), _next((_j + 1) % _Np), _ps(_Np), _alphas(_Np)
    {
      for (int j = 1; j < _Np; j++) { _ps[j] = ColorSpinorField(csParam); }

      // need to make sure init field is copied in the correct precision
      if (init.Precision() != csParam.Precision()) {
        csParam.create = QUDA_COPY_FIELD_CREATE;
        csParam.field = const_cast<ColorSpinorField *>(&init);
        _ps[0] = ColorSpinorField(csParam);
      } else {
        _ps[0] = ColorSpinorField(init);
      }
    }

    /**
       @brief use the vectors currently stored and add to the given output field
       @param x the output field to add to
    */
    void accumulate_x(ColorSpinorField &x)
    {
      blas::axpy<double>({_alphas.begin(), _alphas.begin() + _j + 1}, {_ps.begin(), _ps.begin() + _j + 1}, x);
    }

    /**
       @brief Get the current vector
    */
    ColorSpinorField &get_current_field() { return _ps[_j]; }

    /**
       @brief Get the current vector
    */
    const ColorSpinorField &get_current_field() const { return _ps[_j]; }

    /**
       @brief Get the next vector
    */
    ColorSpinorField &get_next_field() { return _ps[_next]; }

    /**
       @brief Get the next vector
    */
    const ColorSpinorField &get_next_field() const { return _ps[_next]; }

    /**
      @brief return whether or not the container is full
     */
    bool is_container_full() { return (_j + 1) % _Np == 0; }

    /**
      @brief Get the current alpha
     */
    double &get_current_alpha() { return _alphas[_j]; }

    /**
      @brief Get the current alpha
     */
    const double &get_current_alpha() const { return _alphas[_j]; }

    /**
      @brief increase the counter by one (modulo _Np)
     */
    XUpdateBatch &operator++()
    {
      _j = (_j + 1) % _Np;
      _next = (_j + 1) % _Np;
      return *this;
    }

    /**
      @brief reset the counter (typically used after reliable update is performed)
     */
    void reset()
    {
      _j = 0;
      _next = (_j + 1) % _Np;
    }

    /**
      @brief reset the next counter (typically used after reliable update is performed)
     */
    void reset_next() { _next = 0; }
  };

} // namespace quda
