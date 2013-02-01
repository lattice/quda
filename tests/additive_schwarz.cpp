// In this file, we include a bunch of unit test for the various components needed for the 
// overlapping additive schwarz preconditioning
// Need to check the positioning of the ghost fields as well as the ghost ordering
// I guess, that I have pack ghost and unpack ghost functions.
// 
// First, let's look at the position of the ghost fields
// We want to test the ghost functions


// ColorSpinorFields tests
// ColorSpinor Field layout is 
//
// even_sites, even_ghost_sites, odd_sites, odd_ghost_sites
//
// ColorSpinorFields has variables 
// real_length = volume*nColor*nSpin*2, where volume would 
// be half the lattice volume in the case of a single-parity field
// real_length therefore gives the number of real numbers needed to store the field
// it does not include any padding of the fields or the ghost zones.
//
// The length variable, on the other hand, does include padding but 
// doesn't include the ghost zones needed for communication. 
// length = (volume + pad)*nColor*nSpin*2
// Then ghost_length is the number of real numbers needed to store the ghost 
// fields. 
// ghost_length = ghostVolume*nSpin*nColor*2
// if(siteSubset == QUDA_PARITY_SITE_SUBSET) then
// ghostFace[0] = x[1]*x[2]*x[3]/2
// ghostFace[1] = x[0]*x[2]*x[3]
// ghostFace[2] = x[0]*x[1]*x[3]
// ghostFace[3] = x[0]*x[1]*x[2]
// (We assume paritioning in all directions, if the i direction is not partitioned, then ghostFace[i]=0.)
// Note the difference of a factor of 2 between the definition of ghostFace[0] 
// and the other elements of ghostFace, which is due to the fact that x[0] already
// incorporates division by 2.
// Then ghostVolume = num_faces*(ghostFace[[0] + ghostFace[1] + ghostFace[2] + ghostFace[3])
//      ghostNormVolume = num_norm_faces*(ghostFace[0] + ghostFace[1] + ghostFace[2] + ghostFace[3])
// and
// ghostOffset[0] = 0
// ghostOffset[1] = ghostOffset[0] + num_faces*ghostFace[0]
// ghostOffset[2] = ghostOffset[1] + num_faces*ghostFace[1]
// ghostOffset[3] = ghostOffset[2] + num_faces*ghostFace[2]
//
// ghostNormOffset[0] = 0
// ghostNormOffset[1] = ghostNormOffset[0] + num_norm_faces*ghostNormFace[0]
// etc.
//
// Then ghost length = ghostVolume*nColor*nSpin*2
//      ghost_norm_length = ghostNormVolume
//
// Then total_length = length + 2*ghost_length, since there are two ghost zone in a full field.
// Note that for improved staggered fermions, num_faces = num_norm_faces = 3x2 = 6, where the 
// factor of 2 comes from communicating in the forward and backward directions.
//
// Then we should have that 
// ghost[i] = ((char*)v + (length + ghostOffset[i]*nColor*nSpin*2)*precision);
// ghostNorm[i] = ((char*)norm + (stride + ghostNormOffset[i])*precision

