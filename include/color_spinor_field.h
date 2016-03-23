#ifndef _COLOR_SPINOR_FIELD_H
#define _COLOR_SPINOR_FIELD_H

#include <quda_internal.h>
#include <quda.h>

#include <iostream>

#include <lattice_field.h>

namespace quda {
  struct FullClover;

  class ColorSpinorParam : public LatticeFieldParam {

  public:
    QudaFieldLocation location; // where are we storing the field (CUDA or CPU)?

    int nColor; // Number of colors of the field
    int nSpin; // =1 for staggered, =2 for coarse Dslash, =4 for 4d spinor

    QudaTwistFlavorType twistFlavor; // used by twisted mass

    QudaSiteOrder siteOrder; // defined for full fields
  
    QudaFieldOrder fieldOrder; // Float, Float2, Float4 etc.
    QudaGammaBasis gammaBasis;
    QudaFieldCreate create; // 

    QudaDWFPCType PCtype; // used to select preconditioning method in DWF 

    void *v; // pointer to field
    void *norm;
    
    //! for eigcg:
    int eigv_dim;    //number of eigenvectors
    int eigv_id;     //eigenvector index

    ColorSpinorParam(const ColorSpinorField &a);
      
  ColorSpinorParam()
    : LatticeFieldParam(), location(QUDA_INVALID_FIELD_LOCATION), nColor(0),
      nSpin(0), twistFlavor(QUDA_TWIST_INVALID), siteOrder(QUDA_INVALID_SITE_ORDER),
      fieldOrder(QUDA_INVALID_FIELD_ORDER), gammaBasis(QUDA_INVALID_GAMMA_BASIS),
      create(QUDA_INVALID_FIELD_CREATE), PCtype(QUDA_PC_INVALID),
      eigv_dim(0), eigv_id(-1) { ; }

      // used to create cpu params
  ColorSpinorParam(void *V, QudaInvertParam &inv_param, const int *X, const bool pc_solution,
		   QudaFieldLocation location=QUDA_CPU_FIELD_LOCATION)
    : LatticeFieldParam(4, X, 0, inv_param.cpu_prec), location(location), nColor(3), 
      nSpin( (inv_param.dslash_type == QUDA_ASQTAD_DSLASH ||
              inv_param.dslash_type == QUDA_STAGGERED_DSLASH) ? 1 : 4), 
      twistFlavor(inv_param.twist_flavor), siteOrder(QUDA_INVALID_SITE_ORDER), 
      fieldOrder(QUDA_INVALID_FIELD_ORDER), gammaBasis(inv_param.gamma_basis), 
      create(QUDA_REFERENCE_FIELD_CREATE),
      PCtype(((inv_param.dslash_type==QUDA_DOMAIN_WALL_4D_DSLASH)||
	      (inv_param.dslash_type==QUDA_MOBIUS_DWF_DSLASH))?QUDA_4D_PC:QUDA_5D_PC ), 
      v(V), eigv_dim(0), eigv_id(-1) {
    
        if (nDim > QUDA_MAX_DIM) errorQuda("Number of dimensions too great");
	for (int d=0; d<nDim; d++) x[d] = X[d];

	if (!pc_solution) {
	  siteSubset = QUDA_FULL_SITE_SUBSET;;
	} else {
	  x[0] /= 2; // X defined the full lattice dimensions
	  siteSubset = QUDA_PARITY_SITE_SUBSET;
	}

	if (inv_param.dslash_type == QUDA_DOMAIN_WALL_DSLASH ||
	    inv_param.dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH || 
	    inv_param.dslash_type == QUDA_MOBIUS_DWF_DSLASH) {
	  nDim++;
	  x[4] = inv_param.Ls;
	}
	else if(inv_param.dslash_type == QUDA_TWISTED_MASS_DSLASH && (twistFlavor == QUDA_TWIST_NONDEG_DOUBLET)){
	  nDim++;
	  x[4] = 2;//for two flavors
    	}

	if (inv_param.dirac_order == QUDA_INTERNAL_DIRAC_ORDER) {
	  fieldOrder = (precision == QUDA_DOUBLE_PRECISION || nSpin == 1) ? 
	    QUDA_FLOAT2_FIELD_ORDER : QUDA_FLOAT4_FIELD_ORDER; 
	  siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
	} else if (inv_param.dirac_order == QUDA_CPS_WILSON_DIRAC_ORDER) {
	  fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
	  siteOrder = QUDA_ODD_EVEN_SITE_ORDER;
	} else if (inv_param.dirac_order == QUDA_QDP_DIRAC_ORDER) {
	  fieldOrder = QUDA_SPACE_COLOR_SPIN_FIELD_ORDER;
	  siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
	} else if (inv_param.dirac_order == QUDA_DIRAC_ORDER) {
	  fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
	  siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
	} else if (inv_param.dirac_order == QUDA_QDPJIT_DIRAC_ORDER) {
	  fieldOrder = QUDA_QDPJIT_FIELD_ORDER;
	  siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
	} else {
	  errorQuda("Dirac order %d not supported", inv_param.dirac_order);
	}
      }
    
    // normally used to create cuda param from a cpu param
  ColorSpinorParam(ColorSpinorParam &cpuParam, QudaInvertParam &inv_param, 
		   QudaFieldLocation location=QUDA_CUDA_FIELD_LOCATION) 
    : LatticeFieldParam(cpuParam.nDim, cpuParam.x, inv_param.sp_pad, inv_param.cuda_prec),
      location(location), nColor(cpuParam.nColor), nSpin(cpuParam.nSpin), twistFlavor(cpuParam.twistFlavor), 
      siteOrder(QUDA_EVEN_ODD_SITE_ORDER), fieldOrder(QUDA_INVALID_FIELD_ORDER), 
      gammaBasis(nSpin == 4? QUDA_UKQCD_GAMMA_BASIS : QUDA_DEGRAND_ROSSI_GAMMA_BASIS), 
      create(QUDA_COPY_FIELD_CREATE), PCtype(cpuParam.PCtype), v(0), eigv_dim(cpuParam.eigv_dim), eigv_id(-1)
      {
	siteSubset = cpuParam.siteSubset;
	fieldOrder = (precision == QUDA_DOUBLE_PRECISION || nSpin == 1) ? 
	  QUDA_FLOAT2_FIELD_ORDER : QUDA_FLOAT4_FIELD_ORDER; 
      }

    /**
       If using CUDA native fields, this function will ensure that the
       field ordering is appropriate for the new precision setting to
       maintain this status
       @param precision New precision value
     */
    void setPrecision(QudaPrecision precision) {
      // is the current status in native field order?
      bool native = false;
      if ( ((this->precision == QUDA_DOUBLE_PRECISION || nSpin==1 || nSpin==2) &&
	    (fieldOrder == QUDA_FLOAT2_FIELD_ORDER)) ||
	   ((this->precision == QUDA_SINGLE_PRECISION || this->precision == QUDA_HALF_PRECISION) &&
	    (nSpin==4) && fieldOrder == QUDA_FLOAT4_FIELD_ORDER) ) { native = true; }
	   
      this->precision = precision;

      // if this is a native field order, let's preserve that status, else keep the same field order
      if (native) fieldOrder = (precision == QUDA_DOUBLE_PRECISION || nSpin == 1 || nSpin == 2) ?
	QUDA_FLOAT2_FIELD_ORDER : QUDA_FLOAT4_FIELD_ORDER;
    }

    void print() {
      printfQuda("nColor = %d\n", nColor);
      printfQuda("nSpin = %d\n", nSpin);
      printfQuda("twistFlavor = %d\n", twistFlavor);
      printfQuda("nDim = %d\n", nDim);
      for (int d=0; d<nDim; d++) printfQuda("x[%d] = %d\n", d, x[d]);
      printfQuda("precision = %d\n", precision);
      printfQuda("pad = %d\n", pad);
      printfQuda("siteSubset = %d\n", siteSubset);
      printfQuda("siteOrder = %d\n", siteOrder);
      printfQuda("fieldOrder = %d\n", fieldOrder);
      printfQuda("gammaBasis = %d\n", gammaBasis);
      printfQuda("create = %d\n", create);
      printfQuda("v = %lx\n", (unsigned long)v);
      printfQuda("norm = %lx\n", (unsigned long)norm);
      if(eigv_dim != 0) printfQuda("nEv = %d\n", eigv_dim);
    }

    virtual ~ColorSpinorParam() {
    }

  };

  class cpuColorSpinorField;
  class cudaColorSpinorField;

  class ColorSpinorField : public LatticeField {

  private:
    void create(int nDim, const int *x, int Nc, int Ns, QudaTwistFlavorType Twistflavor, 
		QudaPrecision precision, int pad, QudaSiteSubset subset, 
		QudaSiteOrder siteOrder, QudaFieldOrder fieldOrder, QudaGammaBasis gammaBasis,
		QudaDWFPCType PCtype, int nev = 0, int evid = -1);
    void destroy();  
    
  protected:
    bool init;
    QudaPrecision precision;

    int nColor;
    int nSpin;
  
    int nDim;
    int x[QUDA_MAX_DIM];

    int volume;
    int volumeCB;
    int pad;
    int stride;

    QudaTwistFlavorType twistFlavor;

    QudaDWFPCType PCtype; // used to select preconditioning method in DWF

    size_t real_length; // physical length only
    size_t length; // length including pads, but not ghost zone - used for BLAS

    void *v; // the field elements
    void *norm; // the normalization field
    void *ghost_field; // unique ghost zone for this instance

    //! used for eigcg:
    int eigv_dim;
    int eigv_id;
    int eigv_volume;       // volume of a single eigenvector 
    int eigv_stride;       // stride of a single eigenvector
    size_t eigv_real_length;  // physical length of a single eigenvector
    size_t eigv_length;       // length including pads (but not ghost zones)

    // multi-GPU parameters
    void* ghost[QUDA_MAX_DIM]; // pointers to the ghost regions - NULL by default
    void* ghostNorm[QUDA_MAX_DIM]; // pointers to ghost norms - NULL by default
    
    int ghostFace[QUDA_MAX_DIM];// the size of each face
    int ghostOffset[QUDA_MAX_DIM][2]; // offsets to each ghost zone
    int ghostNormOffset[QUDA_MAX_DIM][2]; // offsets to each ghost zone for norm field

    size_t ghost_length; // length of ghost zone
    size_t ghost_norm_length; // length of ghost zone for norm
    size_t total_length; // total length of spinor (physical + pad + ghost)
    size_t total_norm_length; // total length of norm

    mutable void *ghost_fixme[2*QUDA_MAX_DIM];

    size_t bytes; // size in bytes of spinor field
    size_t norm_bytes; // size in bytes of norm field
    size_t ghost_bytes; // size in bytes of the ghost field
    size_t ghost_face_bytes[QUDA_MAX_DIM];

    /*Warning: we need copies of the above params for eigenvectors*/
    //multi_GPU parameters:
    
    //ghost pointers are always for single eigenvector..
    size_t eigv_total_length;
    size_t eigv_total_norm_length;
    size_t eigv_ghost_length;
    size_t eigv_ghost_norm_length;

    size_t eigv_bytes;      // size in bytes of spinor field
    size_t eigv_norm_bytes; // makes no sense but let's keep it...

    QudaSiteSubset siteSubset;
    QudaSiteOrder siteOrder;
    QudaFieldOrder fieldOrder;
    QudaGammaBasis gammaBasis;
  
    // in the case of full fields, these are references to the even / odd sublattices
    ColorSpinorField *even;
    ColorSpinorField *odd;

    //! for eigcg:
    std::vector<ColorSpinorField*> eigenvectors;
      
    void createGhostZone();

    // resets the above attributes based on contents of param
    void reset(const ColorSpinorParam &);
    void fill(ColorSpinorParam &) const;
    static void checkField(const ColorSpinorField &, const ColorSpinorField &);
    void clearGhostPointers();

    char aux_string[TuneKey::aux_n]; // used as a label in the autotuner
    void setTuningString(); // set the vol_string and aux_string for use in tuning

  public:
    //ColorSpinorField();
    ColorSpinorField(const ColorSpinorField &);
    ColorSpinorField(const ColorSpinorParam &);

    virtual ~ColorSpinorField();

    virtual ColorSpinorField& operator=(const ColorSpinorField &);

    QudaPrecision Precision() const { return precision; }
    int Ncolor() const { return nColor; } 
    int Nspin() const { return nSpin; } 
    QudaTwistFlavorType TwistFlavor() const { return twistFlavor; }  
    int Ndim() const { return nDim; }
    const int* X() const { return x; }
    int X(int d) const { return x[d]; }
    size_t RealLength() const { return real_length; }
    size_t Length() const { return length; }
    size_t TotalLength() const { return total_length; }
    int Stride() const { return stride; }
    int Volume() const { return volume; }
    int VolumeCB() const { return siteSubset == QUDA_PARITY_SITE_SUBSET ? volume : volume / 2; }
    int Pad() const { return pad; }
    size_t Bytes() const { return bytes; }
    size_t NormBytes() const { return norm_bytes; }
    size_t GhostBytes() const { return ghost_bytes; } 
    size_t GhostNormBytes() const { return ghost_bytes; } 
    void PrintDims() const { printfQuda("dimensions=%d %d %d %d\n", x[0], x[1], x[2], x[3]); }

    const char *AuxString() const { return aux_string; }

    void* V() {return v;}
    const void* V() const {return v;}
    void* Norm(){return norm;}
    const void* Norm() const {return norm;}
    void* Ghost2() { return ghost_field; }
    const void* Ghost2() const { return ghost_field; } // v will eventually be replaced by ghost_field

    int Nface() const { return (nSpin == 1) ? 3 : 1; } // FIXME for naive staggered or other operators

    /**
       Do the exchange between neighbouring nodes of the data in
       sendbuf storing the result in recvbuf.  The arrays are ordered
       (2*dim + dir).
       @param recvbuf Packed buffer where we store the result
       @param sendbuf Packed buffer from which we're sending
       @param nFace Number of layers we are exchanging
     */
    void exchange(void **ghost, void **sendbuf, int nFace=1) const;

    /**
       This is a unified ghost exchange function for doing a complete
       halo exchange regardless of the type of field.  All dimensions
       are exchanged and no spin projection is done in the case of
       Wilson fermions.
     */
    virtual void exchangeGhost(QudaParity parity, int dagger) const = 0; 

    /**
      This function returns true if the field is stored in an internal
      field order, given the precision and the length of the spin
      dimension.
      */ 
    bool isNative() const;

    //! for eigcg only:
    int EigvDim() const { return eigv_dim; }
    int EigvId() const { return eigv_id; }
    int EigvVolume() const { return eigv_volume; }
    int EigvStride() const { return eigv_stride; }
    size_t EigvLength() const { return eigv_length; }
    size_t EigvRealLength() const { return eigv_real_length; }
    size_t EigvTotalLength() const { return eigv_total_length; }
    
    size_t EigvBytes() const { return eigv_bytes; }
    size_t EigvNormBytes() const { return eigv_norm_bytes; }
    size_t EigvGhostLength() const { return eigv_ghost_length; }

    QudaDWFPCType DWFPCtype() const { return PCtype; }

    QudaSiteSubset SiteSubset() const { return siteSubset; }
    QudaSiteOrder SiteOrder() const { return siteOrder; }
    QudaFieldOrder FieldOrder() const { return fieldOrder; }
    QudaGammaBasis GammaBasis() const { return gammaBasis; }

    size_t GhostLength() const { return ghost_length; }
    const int *GhostFace() const { return ghostFace; }  
    int GhostOffset(const int i) const { return ghostOffset[i][0]; }  
    int GhostOffset(const int i, const int j) const { return ghostOffset[i][j]; }
    int GhostNormOffset(const int i ) const { return ghostNormOffset[i][0]; }  
    int GhostNormOffset(const int i, const int j) const { return ghostNormOffset[i][j]; }
    
    void* Ghost(const int i);
    const void* Ghost(const int i) const;
    void* GhostNorm(const int i);
    const void* GhostNorm(const int i) const;

    /**
       Return array of pointers to the ghost zones (ordering dim*2+dir)
     */
    void* const* Ghost() const;

    const ColorSpinorField& Even() const;
    const ColorSpinorField& Odd() const;

    ColorSpinorField& Even();
    ColorSpinorField& Odd();

    virtual void Source(const QudaSourceType sourceType, const int st=0, const int s=0, const int c=0) = 0;

    /** 
     * Compute the n-dimensional site index given the 1-d offset index
     * @param y n-dimensional site index
     * @param i 1-dimensional site index
     */
    void LatticeIndex(int *y, int i) const;
    
    /** 
     * Compute the 1-d offset index given the n-dimensional site index
     * @param i 1-dimensional site index
     * @param y n-dimensional site index
     */
    void OffsetIndex(int &i, int *y) const;

    static ColorSpinorField* Create(const ColorSpinorParam &param);
    ColorSpinorField* CreateCoarse(const int *geoblockSize, int spinBlockSize, int Nvec, 
				   QudaFieldLocation location=QUDA_INVALID_FIELD_LOCATION);
    ColorSpinorField* CreateFine(const int *geoblockSize, int spinBlockSize, int Nvec,
				 QudaFieldLocation location=QUDA_INVALID_FIELD_LOCATION);
    
    friend std::ostream& operator<<(std::ostream &out, const ColorSpinorField &);
    friend class ColorSpinorParam;
  };

  // CUDA implementation
  class cudaColorSpinorField : public ColorSpinorField {

    friend class cpuColorSpinorField;

  private:
    bool alloc; // whether we allocated memory
    bool init;

    bool texInit; // whether a texture object has been created or not
#ifdef USE_TEXTURE_OBJECTS
    cudaTextureObject_t tex;
    cudaTextureObject_t texNorm;
    cudaTextureObject_t ghostTex;
    cudaTextureObject_t ghostTexNorm;
    void createTexObject();
    void destroyTexObject();
#endif

#ifdef GPU_COMMS  // This is a hack for half precision.
    // Since the ghost data and ghost norm data are not contiguous,
    // separate MPI calls are needed when using GPUDirect RDMA.
    void *my_fwd_norm_face[2][QUDA_MAX_DIM];
    void *my_back_norm_face[2][QUDA_MAX_DIM];
    void *from_fwd_norm_face[2][QUDA_MAX_DIM];
    void *from_back_norm_face[2][QUDA_MAX_DIM];

    MsgHandle ***mh_recv_norm_fwd[2];
    MsgHandle ***mh_recv_norm_back[2];
    MsgHandle ***mh_send_norm_fwd[2];
    MsgHandle ***mh_send_norm_back[2];
#endif

    bool reference; // whether the field is a reference or not

    static size_t ghostFaceBytes;
    static void* ghostFaceBuffer[2]; // gpu memory
    static void* fwdGhostFaceBuffer[2][QUDA_MAX_DIM]; // pointers to ghostFaceBuffer
    static void* backGhostFaceBuffer[2][QUDA_MAX_DIM]; // pointers to ghostFaceBuffer
    static int initGhostFaceBuffer;

    /** Peer-to-peer message handler for signaling event posting */
    MsgHandle* mh_send_p2p_fwd[QUDA_MAX_DIM];

    /** Peer-to-peer message handler for signaling event posting */
    MsgHandle* mh_send_p2p_back[QUDA_MAX_DIM];

    /** Peer-to-peer message handler for signaling event posting */
    MsgHandle* mh_recv_p2p_fwd[QUDA_MAX_DIM];

    /** Peer-to-peer message handler for signaling event posting */
    MsgHandle* mh_recv_p2p_back[QUDA_MAX_DIM];

    /** Buffer used by peer-to-peer message handler */
    static int buffer_send_p2p_fwd[QUDA_MAX_DIM];

    /** Buffer used by peer-to-peer message handler */
    static int buffer_recv_p2p_fwd[QUDA_MAX_DIM];

    /** Buffer used by peer-to-peer message handler */
    static int buffer_send_p2p_back[QUDA_MAX_DIM];

    /** Buffer used by peer-to-peer message handler */
    static int buffer_recv_p2p_back[QUDA_MAX_DIM];

    /** Local copy of event used for peer-to-peer synchronization */
    cudaEvent_t ipcCopyEvent[2][QUDA_MAX_DIM];

    /** Remote copy of event used for peer-to-peer synchronization */
    cudaEvent_t ipcRemoteCopyEvent[2][QUDA_MAX_DIM];

    /** Remote ghost pointer for sending ghost to */
    void* fwdGhostSendDest[QUDA_MAX_DIM];

    /** Remote ghost pointer for sending ghost to */
    void* backGhostSendDest[QUDA_MAX_DIM];

    void create(const QudaFieldCreate);
    void destroy();
    void copy(const cudaColorSpinorField &);

    void zeroPad();

    /**
      This function is responsible for calling the correct copy kernel
      given the nature of the source field and the desired destination.
      */
    void copySpinorField(const ColorSpinorField &src);

    void loadSpinorField(const ColorSpinorField &src);
    void saveSpinorField (ColorSpinorField &src) const;

    /** Whether we have initialized communication for this field */
    bool initComms;

    /** Whether we have initialized peer-to-peer communication for this field */
    bool initIPCComms;

    /** Keep track of which pinned-memory buffer we used for creating message handlers */
    size_t bufferMessageHandler;

    /** How many faces we are communicating in this communicator */
    int nFaceComms; //FIXME - currently can only support one nFace in a field at once

  public:

    static int bufferIndex;

    //cudaColorSpinorField();
    cudaColorSpinorField(const cudaColorSpinorField&);
    cudaColorSpinorField(const ColorSpinorField&, const ColorSpinorParam&);
    cudaColorSpinorField(const ColorSpinorField&);
    cudaColorSpinorField(const ColorSpinorParam&);
    virtual ~cudaColorSpinorField();

    ColorSpinorField& operator=(const ColorSpinorField &);
    cudaColorSpinorField& operator=(const cudaColorSpinorField&);
    cudaColorSpinorField& operator=(const cpuColorSpinorField&);

    void switchBufferPinned();

    /** Create the communication handlers and buffers */
    void createComms(int nFace);

    /** Destroy the communication handlers and buffers */
    void destroyComms();
    
    /** Create the inter-process communication handlers */
    void createIPCComms();

    /** Destroy the inter-process communication handlers */
    void destroyIPCComms();

    /** Handle to remote copy event used for peer-to-peer synchronization */
    const cudaEvent_t& getIPCRemoteCopyEvent(int dir, int dim) const;

    /** Allocate the ghost buffers */
    void allocateGhostBuffer(int nFace);

    /** Ghost buffer allocation for the generic packer*/
    void allocateGhostBuffer(void *send_buf[], void *recv_buf[]) const;

    /** Free statically allocated ghost buffers */
    static void freeGhostBuffer(void);

    /**
      Packs the cudaColorSpinorField's ghost zone 
      @param nFace How many faces to pack (depth)
      @param parity Parity of the field
      @param dim Labels space-time dimensions
      @param dir Pack data to send in forward of backward directions, or both
      @param dagger Whether the operator is the Hermitian conjugate or not
      @param stream Which stream to use for the kernel
      @param buffer Optional parameter where the ghost should be
      stored (default is to use cudaColorSpinorField::ghostFaceBuffer)
      @param a Twisted mass parameter (default=0)
      @param b Twisted mass parameter (default=0)
      */
    void packGhost(const int nFace, const QudaParity parity, const int dim, const QudaDirection dir, const int dagger, 
        cudaStream_t* stream, void *buffer=0, double a=0, double b=0);


    void packGhostExtended(const int nFace, const int R[], const QudaParity parity, const int dim, const QudaDirection dir, 
        const int dagger,cudaStream_t* stream, void *buffer=0);


    void packGhost(FullClover &clov, FullClover &clovInv, const int nFace, const QudaParity parity, const int dim,
		   const QudaDirection dir, const int dagger, cudaStream_t* stream, void *buffer=0, double a=0);

    /**
      Initiate the gpu to cpu send of the ghost zone (halo)
      @param ghost_spinor Where to send the ghost zone
      @param nFace Number of face to send
      @param dim The lattice dimension we are sending
      @param dir The direction (QUDA_BACKWARDS or QUDA_FORWARDS)
      @param dagger Whether the operator is daggerer or not
      @param stream The array of streams to use
      */
    void sendGhost(void *ghost_spinor, const int nFace, const int dim, const QudaDirection dir,
        const int dagger, cudaStream_t *stream);

    /**
      Initiate the cpu to gpu send of the ghost zone (halo)
      @param ghost_spinor Source of the ghost zone
      @param nFace Number of face to send
      @param dim The lattice dimension we are sending
      @param dir The direction (QUDA_BACKWARDS or QUDA_FORWARDS)
      @param dagger Whether the operator is daggerer or not
      @param stream The array of streams to use
      */
    void unpackGhost(const void* ghost_spinor, const int nFace, const int dim, 
        const QudaDirection dir, const int dagger, cudaStream_t* stream);

    /**
      Initiate the cpu to gpu copy of the extended border region
      @param ghost_spinor Source of the ghost zone
      @param parity Parity of the field
      @param nFace Number of face to send
      @param dim The lattice dimension we are sending
      @param dir The direction (QUDA_BACKWARDS or QUDA_FORWARDS)
      @param dagger Whether the operator is daggered or not
      @param stream The array of streams to use
      */
    void unpackGhostExtended(const void* ghost_spinor, const int nFace, const QudaParity parity,
        const int dim, const QudaDirection dir, const int dagger, cudaStream_t* stream);


    void streamInit(cudaStream_t *stream_p);

    void pack(int nFace, int parity, int dagger, int stream_idx, bool zeroCopyPack, 
              double a=0, double b=0);

    void pack(FullClover &clov, FullClover &clovInv, int nFace, int parity, int dagger,
	      int stream_idx, bool zeroCopyPack, double a=0);

    void pack(int nFace, int parity, int dagger, cudaStream_t *stream_p, bool zeroCopyPack,
	      double a=0, double b=0);

    void pack(FullClover &clov, FullClover &clovInv, int nFace, int parity, int dagger,
	      cudaStream_t *stream_p, bool zeroCopyPack, double a=0);

    void packExtended(const int nFace, const int R[], const int parity, const int dagger,
        const int dim,  cudaStream_t *stream_p, const bool zeroCopyPack=false);

    void gather(int nFace, int dagger, int dir, cudaStream_t *stream_p=NULL);

    void recvStart(int nFace, int dir, int dagger=0, cudaStream_t *stream_p=NULL);
    void sendStart(int nFace, int dir, int dagger=0, cudaStream_t *stream_p=NULL);
    void commsStart(int nFace, int dir, int dagger=0, cudaStream_t *stream_p=NULL);
    int commsQuery(int nFace, int dir, int dagger=0, cudaStream_t *stream_p=NULL);
    void commsWait(int nFace, int dir, int dagger=0, cudaStream_t *stream_p=NULL);

    void scatter(int nFace, int dagger, int dir, cudaStream_t *stream_p);
    void scatter(int nFace, int dagger, int dir);

    void scatterExtended(int nFace, int parity, int dagger, int dir);

    /** Helper function to determine if local-to-remote (send) peer-to-peer copy is complete */
    inline bool ipcCopyComplete(int dir, int dim);

    /** Helper function to determine if local-to-remote (receive) peer-to-peer copy is complete */
    inline bool ipcRemoteCopyComplete(int dir, int dim);

    /**
       Do a ghost exchange between neighbouring nodes.  All dimensions
       are exchanged and no spin projection is done in the case of
       Wilson fermions.
     */
    void exchangeGhost(QudaParity parity, int dagger) const;

#ifdef USE_TEXTURE_OBJECTS
    const cudaTextureObject_t& Tex() const { return tex; }
    const cudaTextureObject_t& TexNorm() const { return texNorm; }
    const cudaTextureObject_t& GhostTex() const { return ghostTex; }
    const cudaTextureObject_t& GhostTexNorm() const { return ghostTexNorm; }
#endif

    cudaColorSpinorField& Eigenvec(const int idx) const;
    void CopyEigenvecSubset(cudaColorSpinorField& dst, const int range, const int first_element=0) const;

    void zero();

    friend std::ostream& operator<<(std::ostream &out, const cudaColorSpinorField &);

    void getTexObjectInfo() const;

    void Source(const QudaSourceType sourceType, const int st=0, const int s=0, const int c=0);
  };

  // CPU implementation
  class cpuColorSpinorField : public ColorSpinorField {

    friend class cudaColorSpinorField;

  public:
    static void* fwdGhostFaceBuffer[QUDA_MAX_DIM]; //cpu memory
    static void* backGhostFaceBuffer[QUDA_MAX_DIM]; //cpu memory
    static void* fwdGhostFaceSendBuffer[QUDA_MAX_DIM]; //cpu memory
    static void* backGhostFaceSendBuffer[QUDA_MAX_DIM]; //cpu memory
    static int initGhostFaceBuffer;
    static size_t ghostFaceBytes[QUDA_MAX_DIM];

    private:
    //void *v; // the field elements
    //void *norm; // the normalization field
    bool init;
    bool reference; // whether the field is a reference or not

    void create(const QudaFieldCreate);
    void destroy();

    public:
    //cpuColorSpinorField();
    cpuColorSpinorField(const cpuColorSpinorField&);
    cpuColorSpinorField(const ColorSpinorField&);
    cpuColorSpinorField(const ColorSpinorField&, const ColorSpinorParam&);
    cpuColorSpinorField(const ColorSpinorParam&);
    virtual ~cpuColorSpinorField();

    ColorSpinorField& operator=(const ColorSpinorField &);
    cpuColorSpinorField& operator=(const cpuColorSpinorField&);
    cpuColorSpinorField& operator=(const cudaColorSpinorField&);

    void Source(const QudaSourceType sourceType, const int st=0, const int s=0, const int c=0);
    static int Compare(const cpuColorSpinorField &a, const cpuColorSpinorField &b, const int resolution=1);
    void PrintVector(unsigned int x);

    void allocateGhostBuffer(void) const;
    static void freeGhostBuffer(void);

    void packGhost(void **ghost, QudaParity parity, int dagger) const;
    void unpackGhost(void* ghost_spinor, const int dim, 
		     const QudaDirection dir, const int dagger);

    void copy(const cpuColorSpinorField&);
    void zero();

    /**
       Do a ghost exchange between neighbouring nodes.  All dimensions
       are exchanged and no spin projection is done in the case of
       Wilson fermions.
     */
    void exchangeGhost(QudaParity parity, int dagger) const;

  };

  void copyGenericColorSpinor(ColorSpinorField &dst, const ColorSpinorField &src, 
      QudaFieldLocation location, void *Dst=0, void *Src=0, 
      void *dstNorm=0, void*srcNorm=0);
  void genericSource(cpuColorSpinorField &a, QudaSourceType sourceType, int x, int s, int c);
  int genericCompare(const cpuColorSpinorField &a, const cpuColorSpinorField &b, int tol);
  void genericPrintVector(cpuColorSpinorField &a, unsigned int x);

  void exchangeExtendedGhost(cudaColorSpinorField* spinor, int R[], int parity, cudaStream_t *stream_p);
  
  void copyExtendedColorSpinor(ColorSpinorField &dst, const ColorSpinorField &src,
      QudaFieldLocation location, const int parity, void *Dst, void *Src, void *dstNorm, void *srcNorm);

  void genericPackGhost(void **ghost, const ColorSpinorField &a, const QudaParity parity, const int dagger);

} // namespace quda

#endif // _COLOR_SPINOR_FIELD_H
