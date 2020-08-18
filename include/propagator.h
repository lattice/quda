#pragma once

#include <color_spinor_field.h>

namespace quda
{

  class Propagator
  {

  private:
    std::vector<ColorSpinorField *> prop_vectors;
    bool prop_init;

  public:
    Propagator(const ColorSpinorParam &);

    /**
       @brief Creates a propagator
       @param [in] param ColorSpinorParam that defines the vectors
    */
    static Propagator *Create(const ColorSpinorParam &param);

    /**
       @brief Copies the vectors from in an input vector se to the
              vectors of the class
       @param [in] vecs The vectors to be copied
    */
    void copyVectors(const std::vector<ColorSpinorField *> &vecs);

    /**
       @brief Return a pointer to the selected vector
       @param [in] vec The requested vector
       @param [out] Pointer to the vector
    */
    ColorSpinorField *selectVector(const int vec);

    /**
       @brief Operator to assign one Propagator to another
    */
    Propagator &operator=(const Propagator &);

    /**
       @brief Class destructor
    */
    ~Propagator();
  };
} // namespace quda
