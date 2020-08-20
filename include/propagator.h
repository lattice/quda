#pragma once

#include <color_spinor_field.h>

namespace quda
{

  class Propagator
  {

  private:
    /**
       @brief Array that hold the numerical data of the propagator
    */
    void *prop_data;
    /**
       @brief Convenient container for the data
    */
    std::vector<ColorSpinorField *> prop_vectors;
    /**
       @brief Specifies if the data has been allocated
    */        
    bool prop_init;
    /**
       @brief Dimension of propagator (spin x color)
    */            
    size_t prop_dim;
    /**
       @brief Location of the propagator
    */            
    QudaFieldLocation prop_location;
    /**
       @brief Precision of the propagator
    */            
    QudaPrecision prop_precision;
        
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
       @brief Returns a void pointer to the prop_data array 
    */
    void *V();
    
    /**
       @brief Class destructor
    */
    ~Propagator();
  };
} // namespace quda
