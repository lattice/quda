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

    /**
       @breif How the propagator was created
    */    
    QudaFieldCreate prop_create;
    
  public:
    /**
       @brief Creates a propagator
       @param [in] param ColorSpinorParam that defines the vectors
    */
    Propagator(const ColorSpinorParam &);

    /**
       @brief Creates a propagator from host data
       @param [in] param ColorSpinorParam that defines the vectors
    */
    Propagator(const ColorSpinorParam &, void **);
        
    /**
       @brief Return a pointer to the selected vector
       @param [in] vec The requested vector
       @param [out] Pointer to the vector
    */
    ColorSpinorField* selectVector(const size_t vec);
    
    /**
       @brief Operator to copy one Propagator to another
    */
    Propagator& operator=(Propagator &);

    /**
       @brief Returns a pointer to the prop_vectors array 
    */
    std::vector<ColorSpinorField *> Vectors();

    
    size_t Dim() const { return prop_dim; };
    QudaFieldLocation Location() const { return prop_location; };
    QudaPrecision Precision() const { return prop_precision; };
    
    /**
       @brief Class destructor
    */
    ~Propagator();
  };
} // namespace quda
