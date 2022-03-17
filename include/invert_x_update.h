#pragma once

#include <color_spinor_field.h>

namespace quda
{

  /**
    @brief A struct that contains multiple p-vectors which are to be added to an output vector:
      x += alpha[i] * p[i], i = 0, 1, ..., Np - 1
    @param _Np the number of underlying vectors
    @param _j the current index
    @param _next the next index
    @param _ps the container for the p-vectors
    @param _alphas the alpha's
   */

  struct XUpdateBatch {

    int _Np;

    int _j;

    int _next;

    std::vector<std::unique_ptr<ColorSpinorField>> _ps; // contains the p-vectors

    std::vector<double> _alphas;

    /**
      @brief contruct the batch: initialize the first vector and the rest separately
      @param Np the number vectors to contain
      @param init the ColorSpinorField used to initialize the first vector in the container
      @param csParam the ColorSpinorParam used to initialize the rest of the vectors
     */
    XUpdateBatch(int Np_, const ColorSpinorField &init, ColorSpinorParam csParam) :
      _Np(Np_), _j(0), _next((_j + 1) % _Np), _ps(_Np), _alphas(_Np)
    {
      for (int j = 1; j < _Np; j++) { _ps[j] = std::unique_ptr<ColorSpinorField>(new ColorSpinorField(csParam)); }

      // need to make sure init field is copied in the correct precision
      if (init.Precision() != csParam.Precision()) {
        csParam.create = QUDA_COPY_FIELD_CREATE;
        csParam.field = const_cast<ColorSpinorField *>(&init);
        _ps[0] = std::unique_ptr<ColorSpinorField>(new ColorSpinorField(csParam));
      } else {
        _ps[0] = std::unique_ptr<ColorSpinorField>(new ColorSpinorField(init));
      }
    }

    /**
      @brief return whether or not the container is full
     */
    bool is_container_full() { return (_j + 1) % _Np == 0; }

    /**
      @brief use the vectors currently stored and add to the given output field
      @param x the output field to add to
     */
    void accumulate_x(ColorSpinorField &x)
    {
      std::vector<ColorSpinorField *> vx(1, &x);
      std::vector<ColorSpinorField *> psj(_j + 1);
      for (int i = 0; i <= _j; i++) { psj[i] = _ps[i].get(); }
      blas::axpy(_alphas.data(), psj, vx);
    }

    /**
      @brief Get the current vector
     */
    ColorSpinorField &get_current_field() { return *_ps[_j]; }

    /**
      @brief Get the current vector
     */
    const ColorSpinorField &get_current_field() const { return *_ps[_j]; }

    /**
      @brief Get the next vector
     */
    ColorSpinorField &get_next_field() { return *_ps[_next]; }

    /**
      @brief Get the next vector
     */
    const ColorSpinorField &get_next_field() const { return *_ps[_next]; }

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
