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
       @brief Default constructor
    */
    Propagator();

    /**
       @brief Initialise a Propagator object using src as input
       @param src Propagator object from which to initialise
    */
    Propagator(const Propagator &src);
    
    /**
       @brief Creates a propagator using a ColorSpinorParam instance
       @param [in] param ColorSpinorParam that defines the vectors
    */
    Propagator(const ColorSpinorParam & param);

    /**
       @brief Creates a propagator from host data
       @param [in] param ColorSpinorParam that defines the vectors
       @param [in] data void pointers holding the vector data
    */
    Propagator(const ColorSpinorParam & param, void ** data);
            
    /**
       @brief Operator to copy one Propagator to another
       @param src Propagator object to copy
    */
    Propagator& operator=(const Propagator &src);

    /**
       @brief Returns a pointer to the prop_vectors array 
    */
    //std::vector<ColorSpinorField *> Vectors();

    
    size_t Dim() const { return prop_dim; };
    QudaFieldLocation Location() const { return prop_location; };
    QudaPrecision Precision() const { return prop_precision; };
    std::vector<ColorSpinorField *> Vectors() const {
      if (!prop_init) errorQuda("Propgator not initialised");
      if (prop_vectors.size() == 0) errorQuda("Zero sized vector set in Propagator");
      return prop_vectors;
    }    
    
    /**
       @brief Class destructor
    */
    ~Propagator();
  };
} // namespace quda
