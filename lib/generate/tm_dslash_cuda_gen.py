# -*- coding: utf-8 -*-
import sys

### complex numbers ########################################################################

def complexify(a):
    return [complex(x) for x in a]

def complexToStr(c):
    def fltToString(a):
        if a == int(a): return `int(a)`
        else: return `a`
    
    def imToString(a):
        if a == 0: return "0i"
        elif a == -1: return "-i"
        elif a == 1: return "i"
        else: return fltToString(a)+"i"
    
    re = c.real
    im = c.imag
    if re == 0 and im == 0: return "0"
    elif re == 0: return imToString(im)
    elif im == 0: return fltToString(re)
    else:
        im_str = "-"+imToString(-im) if im < 0 else "+"+imToString(im)
        return fltToString(re)+im_str


### projector matrices ########################################################################

id = complexify([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1
])

gamma1 = complexify([
    0,  0, 0, 1j,
    0,  0, 1j, 0,
    0, -1j, 0, 0,
    -1j,  0, 0, 0
])

gamma2 = complexify([
    0, 0, 0, 1,
    0, 0, -1,  0,
    0, -1, 0,  0,
    1, 0, 0,  0
])

gamma3 = complexify([
    0, 0, 1j,  0,
    0, 0, 0, -1j,
    -1j, 0, 0,  0,
    0, 1j, 0,  0
])

gamma4 = complexify([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, -1, 0,
    0, 0, 0, -1
])

igamma5 = complexify([
    0, 0, 1j, 0,
    0, 0, 0, 1j,
    1j, 0, 0, 0,
    0, 1j, 0, 0
])


def gplus(g1, g2):
    return [x+y for (x,y) in zip(g1,g2)]

def gminus(g1, g2):
    return [x-y for (x,y) in zip(g1,g2)]

def projectorToStr(p):
    out = ""
    for i in range(0, 4):
        for j in range(0,4):
            out += complexToStr(p[4*i+j]) + " "
        out += "\n"
    return out

projectors = [
    gminus(id,gamma1), gplus(id,gamma1),
    gminus(id,gamma2), gplus(id,gamma2),
    gminus(id,gamma3), gplus(id,gamma3),
    gminus(id,gamma4), gplus(id,gamma4),
]

### code generation  ########################################################################

def indent(code):
    def indentline(line): return ("  "+line if (line.count("#", 0, 1) == 0) else line)
    return ''.join([indentline(line)+"\n" for line in code.splitlines()])

def block(code):
    return "{\n"+indent(code)+"}"

def sign(x):
    if x==1: return "+"
    elif x==-1: return "-"
    elif x==+2: return "+2*"
    elif x==-2: return "-2*"

def nthFloat4(n):
    return `(n/4)` + "." + ["x", "y", "z", "w"][n%4]

def nthFloat2(n):
    return `(n/2)` + "." + ["x", "y"][n%2]


def in_re(s, c): return "i"+`s`+`c`+"_re"
def in_im(s, c): return "i"+`s`+`c`+"_im"
def g_re(d, m, n): return ("g" if (d%2==0) else "gT")+`m`+`n`+"_re"
def g_im(d, m, n): return ("g" if (d%2==0) else "gT")+`m`+`n`+"_im"
def out1_re(s, c): return "o1_"+`s`+`c`+"_re"
def out1_im(s, c): return "o1_"+`s`+`c`+"_im"
def out2_re(s, c): return "o2_"+`s`+`c`+"_re"
def out2_im(s, c): return "o2_"+`s`+`c`+"_im"
def h1_re(h, c): return ["a","b"][h]+`c`+"_re"
def h1_im(h, c): return ["a","b"][h]+`c`+"_im"
def h2_re(h, c): return ["A","B"][h]+`c`+"_re"
def h2_im(h, c): return ["A","B"][h]+`c`+"_im"
def a_re(b, s, c): return "a"+`(s+2*b)`+`c`+"_re"
def a_im(b, s, c): return "a"+`(s+2*b)`+`c`+"_im"

def tmp_re(s, c): return "tmp"+`s`+`c`+"_re"
def tmp_im(s, c): return "tmp"+`s`+`c`+"_im"


def def_input_spinor():
    str = ""
    str += "// input spinor\n"
    str += "#ifdef SPINOR_DOUBLE\n"
    str += "#define spinorFloat double\n"
    if sharedDslash: 
        str += "#define WRITE_SPINOR_SHARED WRITE_SPINOR_SHARED_DOUBLE2\n"
        str += "#define READ_SPINOR_SHARED READ_SPINOR_SHARED_DOUBLE2\n"

    for s in range(0,4):
        for c in range(0,3):
            i = 3*s+c
            str += "#define "+in_re(s,c)+" I"+nthFloat2(2*i+0)+"\n"
            str += "#define "+in_im(s,c)+" I"+nthFloat2(2*i+1)+"\n"
    str += "#else\n"
    str += "#define spinorFloat float\n"
    if sharedDslash: 
        str += "#define WRITE_SPINOR_SHARED WRITE_SPINOR_SHARED_FLOAT4\n"
        str += "#define READ_SPINOR_SHARED READ_SPINOR_SHARED_FLOAT4\n"
    for s in range(0,4):
        for c in range(0,3):
            i = 3*s+c
            str += "#define "+in_re(s,c)+" I"+nthFloat4(2*i+0)+"\n"
            str += "#define "+in_im(s,c)+" I"+nthFloat4(2*i+1)+"\n"
    str += "#endif // SPINOR_DOUBLE\n\n"
    return str
# end def def_input_spinor


def def_gauge():
    str = "// gauge link\n"
    str += "#ifdef GAUGE_FLOAT2\n"
    for m in range(0,3):
        for n in range(0,3):
            i = 3*m+n
            str += "#define "+g_re(0,m,n)+" G"+nthFloat2(2*i+0)+"\n"
            str += "#define "+g_im(0,m,n)+" G"+nthFloat2(2*i+1)+"\n"

    str += "\n"
    str += "#else\n"
    for m in range(0,3):
        for n in range(0,3):
            i = 3*m+n
            str += "#define "+g_re(0,m,n)+" G"+nthFloat4(2*i+0)+"\n"
            str += "#define "+g_im(0,m,n)+" G"+nthFloat4(2*i+1)+"\n"

    str += "\n"
    str += "#endif // GAUGE_DOUBLE\n\n"
            
    str += "// conjugated gauge link\n"
    for m in range(0,3):
        for n in range(0,3):
            i = 3*m+n
            str += "#define "+g_re(1,m,n)+" (+"+g_re(0,n,m)+")\n"
            str += "#define "+g_im(1,m,n)+" (-"+g_im(0,n,m)+")\n"
    str += "\n"

    return str
# end def def_gauge



def def_output_spinor():
# sharedDslash = True: input spinors stored in shared memory
# sharedDslash = False: output spinors stored in shared memory
    str = "// output spinor for flavor 1\n"
    for s in range(0,4):
        for c in range(0,3):
            i = 3*s+c
            if 2*i < sharedFloatsPerFlavor and not sharedDslash:
                str += "#define "+out1_re(s,c)+" s["+`(2*i+0)`+"*SHARED_STRIDE]\n"
            else:
                str += "VOLATILE spinorFloat "+out1_re(s,c)+";\n"
            if 2*i+1 < sharedFloatsPerFlavor and not sharedDslash:
                str += "#define "+out1_im(s,c)+" s["+`(2*i+1)`+"*SHARED_STRIDE]\n"
            else:
                str += "VOLATILE spinorFloat "+out1_im(s,c)+";\n"

    str += "// output spinor for flavor 2\n"
    for s in range(0,4):
        for c in range(0,3):
            i = 3*s+c
            if 2*i < sharedFloatsPerFlavor and not sharedDslash:
                str += "#define "+out2_re(s,c)+" s["+`(2*i+0)+sharedFloatsPerFlavor`+"*SHARED_STRIDE]\n"
            else:
                str += "VOLATILE spinorFloat "+out2_re(s,c)+";\n"
            if 2*i+1 < sharedFloatsPerFlavor and not sharedDslash:
                str += "#define "+out2_im(s,c)+" s["+`(2*i+1)+sharedFloatsPerFlavor`+"*SHARED_STRIDE]\n"
            else:
                str += "VOLATILE spinorFloat "+out2_im(s,c)+";\n"
    return str
# end def def_output_spinor


def prolog():
    global arch
#WARNING: change for twisted mass!
    if dslash:
        prolog_str= ("// *** CUDA NDEG TWISTED MASS DSLASH ***\n\n" if not dagger else "// *** CUDA NDEG TWISTED MASS DSLASH DAGGER ***\n\n")
        prolog_str+= ("// Arguments (double) mu, (double)eta and (double)delta \n")
        prolog_str+= "#define SHARED_TMNDEG_FLOATS_PER_THREAD "+str(2*sharedFloatsPerFlavor)+"\n"
        prolog_str+= "#define FLAVORS 2\n\n"
    else:
        print "Undefined prolog"
        exit

    prolog_str+= (
"""
#if ((CUDA_VERSION >= 4010) && (__COMPUTE_CAPABILITY__ >= 200)) // NVVM compiler
#define VOLATILE
#else // Open64 compiler
#define VOLATILE volatile
#endif
""")

    prolog_str+= def_input_spinor()
    if dslash == True: prolog_str+= def_gauge()
    prolog_str+= def_output_spinor()

    if (sharedFloatsPerFlavor > 0):
        if (arch >= 200):
            prolog_str+= (
"""
#ifdef SPINOR_DOUBLE
#define SHARED_STRIDE 16 // to avoid bank conflicts on Fermi
#else
#define SHARED_STRIDE 32 // to avoid bank conflicts on Fermi
#endif
""")
        else:
            prolog_str+= (
"""
#ifdef SPINOR_DOUBLE
#define SHARED_STRIDE  8 // to avoid bank conflicts on G80 and GT200
#else
#define SHARED_STRIDE 16 // to avoid bank conflicts on G80 and GT200
#endif
""")


    # set the pointer if using shared memory for pseudo registers
#    if sharedFloatsPerFlavor > 0 and not sharedDslash: 
    if sharedFloatsPerFlavor > 0:
        prolog_str += (
"""
extern __shared__ char s_data[];
""")

        if dslash:
            prolog_str += (
"""
VOLATILE spinorFloat *s = (spinorFloat*)s_data + SHARED_TMNDEG_FLOATS_PER_THREAD*SHARED_STRIDE*(threadIdx.x/SHARED_STRIDE)
                                  + (threadIdx.x % SHARED_STRIDE);
""")

    if dslash:
        prolog_str += (
"""
#include "read_gauge.h"
#include "io_spinor.h"

int x1, x2, x3, x4;
int X;

#if (defined MULTI_GPU) && (DD_PREC==2) // half precision
int sp_norm_idx;
#endif // MULTI_GPU half precision

int sid;
""")

        if sharedDslash:
            prolog_str += (
"""
#ifdef MULTI_GPU
int face_idx;
if (kernel_type == INTERIOR_KERNEL) {
#endif

  // Inline by hand for the moment and assume even dimensions
  coordsFromIndex3D<EVEN_X>(X, x1, x2, x3, x4, sid, param.parity);

  // only need to check Y and Z dims currently since X and T set to match exactly
  if (x2 >= X2) return;
  if (x3 >= X3) return; 

""")
        else:
            prolog_str += (
"""
#ifdef MULTI_GPU
int face_idx;
if (kernel_type == INTERIOR_KERNEL) {
#endif

  sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= param.threads) return;

  // Inline by hand for the moment and assume even dimensions
  coordsFromIndex<EVEN_X>(X, x1, x2, x3, x4, sid, param.parity);

""")

        out = ""
        for s in range(0,4):
            for c in range(0,3):
                out += out1_re(s,c)+" = 0;  "+out1_im(s,c)+" = 0;\n"

        out += "\n"

        for s in range(0,4):
            for c in range(0,3):
                out += out2_re(s,c)+" = 0;  "+out2_im(s,c)+" = 0;\n"

        prolog_str+= indent(out)

        prolog_str+= (
"""
#ifdef MULTI_GPU
} else { // exterior kernel

  sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= param.threads) return;

  const int dim = static_cast<int>(kernel_type);
  const int face_volume = (param.threads >> 1);           // volume of one face (per flavor)
  const int face_num = (sid >= face_volume);              // is this thread updating face 0 or 1
  face_idx = sid - face_num*face_volume;        // index into the respective face

  // ghostOffset is scaled to include body (includes stride) and number of FloatN arrays (SPINOR_HOP)
  // face_idx not sid since faces are spin projected and share the same volume index (modulo UP/DOWN reading)
  //sp_idx = face_idx + param.ghostOffset[dim];

#if (DD_PREC==2) // half precision
  sp_norm_idx = sid + param.ghostNormOffset[static_cast<int>(kernel_type)];
#endif

  coordsFromFaceIndex<1>(X, sid, x1, x2, x3, x4, face_idx, face_volume, dim, face_num, param.parity);

""")

#for flavor 1:
        prolog_str+= (
"""
  {
     READ_INTERMEDIATE_SPINOR(INTERTEX, sp_stride, sid, sid);
""")

        out1 = "   "
        for s in range(0,4):
            for c in range(0,3):
                out1 += out1_re(s,c)+" = "+in_re(s,c)+";  "+out1_im(s,c)+" = "+in_im(s,c)+";\n   "
        prolog_str+= indent(out1)

#for flavor 2:
        prolog_str+= (
"""
  }
  {
     READ_INTERMEDIATE_SPINOR(INTERTEX, sp_stride, sid+fl_stride, sid+fl_stride);
""")

        out2 = "   "
        for s in range(0,4):
            for c in range(0,3):
                out2 += out2_re(s,c)+" = "+in_re(s,c)+";  "+out2_im(s,c)+" = "+in_im(s,c)+";\n   "
        prolog_str+= indent(out2)
        prolog_str+= (
"""
  }
""")


        prolog_str+= "}\n"
        prolog_str+= "#endif // MULTI_GPU\n\n\n"

    else:
        prolog_str+=(
"""
#include "io_spinor.h"

int sid = blockIdx.x*blockDim.x + threadIdx.x;
if (sid >= param.threads) return;

// read spinor from device memory
READ_SPINOR(SPINORTEX, sp_stride, sid, sid);
""")            
    return prolog_str
# end def prolog


def gen(dir, pack_only=False):
    projIdx = dir if not dagger else dir + (1 - 2*(dir%2))
    projStr = projectorToStr(projectors[projIdx])
    def proj(i,j):
        return projectors[projIdx][4*i+j]
    
    # if row(i) = (j, c), then the i'th row of the projector can be represented
    # as a multiple of the j'th row: row(i) = c row(j)
    def row(i):
        assert i==2 or i==3
        if proj(i,0) == 0j:
            return (1, proj(i,1))
        if proj(i,1) == 0j:
            return (0, proj(i,0))

#    boundary = ["x1==X1m1", "x1==0", "x2==X2m1", "x2==0", "x3==X3m1", "x3==0", "x4==X4m1", "x4==0"]
#    interior = ["x1<X1m1", "x1>0", "x2<X2m1", "x2>0", "x3<X3m1", "x3>0", "x4<X4m1", "x4>0"]
    boundary = ["x1==X1m1", "x1==0", "x2==X2m1", "x2==0", "x3==X3m1", "x3==0", "x4==X4m1", "x4==0"]
    interior = ["x1<X1m1", "x1>0", "x2<X2m1", "x2>0", "x3<X3m1", "x3>0", "x4<X4m1", "x4>0"]
    dim = ["X", "Y", "Z", "T"]

    # index of neighboring site when not on boundary
    sp_idx = ["X+1", "X-1", "X+X1", "X-X1", "X+X2X1", "X-X2X1", "X+X3X2X1", "X-X3X2X1"]

    # index of neighboring site (across boundary)
    sp_idx_wrap = ["X-X1m1", "X+X1m1", "X-X2X1mX1", "X+X2X1mX1", "X-X3X2X1mX2X1", "X+X3X2X1mX2X1",
                   "X-X4X3X2X1mX3X2X1", "X+X4X3X2X1mX3X2X1"]

    cond = ""
    cond += "#ifdef MULTI_GPU\n"
    cond += "if ( (kernel_type == INTERIOR_KERNEL && (!param.ghostDim["+`dir/2`+"] || "+interior[dir]+")) ||\n"
    cond += "     (kernel_type == EXTERIOR_KERNEL_"+dim[dir/2]+" && "+boundary[dir]+") )\n"
    cond += "#endif\n"

    str = ""
    
    projName = "P"+`dir/2`+["-","+"][projIdx%2]
    str += "// Projector "+projName+"\n"
    for l in projStr.splitlines():
        str += "// "+l+"\n"
    str += "\n"

    str += "#ifdef MULTI_GPU\n"
    str += "const int sp_idx = (kernel_type == INTERIOR_KERNEL) ? ("+boundary[dir]+" ? "+sp_idx_wrap[dir]+" : "+sp_idx[dir]+") >> 1 :\n"
    str += "  face_idx + param.ghostOffset[static_cast<int>(kernel_type)];\n"
    str += "#else\n"
    str += "const int sp_idx = ("+boundary[dir]+" ? "+sp_idx_wrap[dir]+" : "+sp_idx[dir]+") >> 1;\n"
    str += "#endif\n"

    str += "\n"
    if dir % 2 == 0:
        str += "const int ga_idx = sid;\n"
    else:
        str += "#ifdef MULTI_GPU\n"
        str += "const int ga_idx = ((kernel_type == INTERIOR_KERNEL) ? sp_idx : Vh+face_idx);\n"
        str += "#else\n"
        str += "const int ga_idx = sp_idx;\n"
        str += "#endif\n"
    str += "\n"

    # scan the projector to determine which loads are required
    row_cnt = ([0,0,0,0])
    for h in range(0,4):
        for s in range(0,4):
            re = proj(h,s).real
            im = proj(h,s).imag
            if re != 0 or im != 0:
                row_cnt[h] += 1
    row_cnt[0] += row_cnt[1]
    row_cnt[2] += row_cnt[3]

    decl_half = ""
    for h in range(0, 2):
        for c in range(0, 3):
            decl_half += "spinorFloat "+h1_re(h,c)+", "+h1_im(h,c)+";\n";
    decl_half += "\n"

    load_gauge = "// read gauge matrix from device memory\n"
    load_gauge += "READ_GAUGE_MATRIX(G, GAUGE"+`dir%2`+"TEX, "+`dir`+", ga_idx, ga_stride);\n\n"

    reconstruct_gauge = "// reconstruct gauge matrix\n"
    reconstruct_gauge += "RECONSTRUCT_GAUGE_MATRIX("+`dir`+");\n\n"

#flavor 1:
    load_flv1 = "// read flavor 1 from device memory\n"
    if row_cnt[0] == 0:
        load_flv1 += "READ_SPINOR_DOWN(SPINORTEX, sp_stride, sp_idx, sp_idx);\n"
    elif row_cnt[2] == 0:
        load_flv1 += "READ_SPINOR_UP(SPINORTEX, sp_stride, sp_idx, sp_idx);\n"
    else:
        load_flv1 += "READ_SPINOR(SPINORTEX, sp_stride, sp_idx, sp_idx);\n"
    load_flv1 += "\n"

#flavor 2:
    load_flv2 = "// read flavor 2 from device memory\n"
    if row_cnt[0] == 0:
        load_flv2 += "READ_SPINOR_DOWN(SPINORTEX, sp_stride, sp_idx+fl_stride, sp_idx+fl_stride);\n"
    elif row_cnt[2] == 0:
        load_flv2 += "READ_SPINOR_UP(SPINORTEX, sp_stride, sp_idx+fl_stride, sp_idx+fl_stride);\n"
    else:
        load_flv2 += "READ_SPINOR(SPINORTEX, sp_stride, sp_idx+fl_stride, sp_idx+fl_stride);\n"
    load_flv2 += "\n"


    load_half_cond = ""
    load_half_cond += "const int sp_stride_pad = FLAVORS*ghostFace[static_cast<int>(kernel_type)];\n"
    #load_half += "#if (DD_PREC==2) // half precision\n"
    #load_half += "const int sp_norm_idx = sid + param.ghostNormOffset[static_cast<int>(kernel_type)];\n"
    #load_half += "#endif\n"

    #if dir >= 6: load_half_cond += "const int t_proj_scale = TPROJSCALE;\n"
    load_half_cond += "\n"

    load_half_flv1 = "// read half spinor for the first flavor from device memory\n"
# we have to use the same volume index for backwards and forwards gathers
# instead of using READ_UP_SPINOR and READ_DOWN_SPINOR, just use READ_HALF_SPINOR with the appropriate shift
    if (dir+1) % 2 == 0: 
          load_half_flv1 += "READ_HALF_SPINOR(SPINORTEX, sp_stride_pad, sp_idx, sp_norm_idx);\n\n"
    else: 
#flavor offset: extra ghostFace[static_cast<int>(kernel_type)]
          load_half_flv1 += "READ_HALF_SPINOR(SPINORTEX, sp_stride_pad, sp_idx + (SPINOR_HOP/2)*sp_stride_pad, sp_norm_idx+ghostFace[static_cast<int>(kernel_type)]);\n\n"
    
    load_half_flv2 = "// read half spinor for the second flavor from device memory\n"
    load_half_flv2 += "const int fl_idx = sp_idx + ghostFace[static_cast<int>(kernel_type)];\n"
# we have to use the same volume index for backwards and forwards gathers
# instead of using READ_UP_SPINOR and READ_DOWN_SPINOR, just use READ_HALF_SPINOR with the appropriate shift
    if (dir+1) % 2 == 0: 
          load_half_flv2 += "READ_HALF_SPINOR(SPINORTEX, sp_stride_pad, fl_idx, sp_norm_idx+ghostFace[static_cast<int>(kernel_type)]);\n\n"
    else: 
#flavor offset: extra ghostFace[static_cast<int>(kernel_type)]
          load_half_flv2 += "READ_HALF_SPINOR(SPINORTEX, sp_stride_pad, fl_idx + (SPINOR_HOP/2)*sp_stride_pad, sp_norm_idx+FLAVORS*ghostFace[static_cast<int>(kernel_type)]);\n\n"


    project = "// project spinor into half spinors\n"
    for h in range(0, 2):
        for c in range(0, 3):
            strRe = ""
            strIm = ""
            for s in range(0, 4):
                re = proj(h,s).real
                im = proj(h,s).imag
                if re==0 and im==0: ()
                elif im==0:
                    strRe += sign(re)+in_re(s,c)
                    strIm += sign(re)+in_im(s,c)
                elif re==0:
                    strRe += sign(-im)+in_im(s,c)
                    strIm += sign(im)+in_re(s,c)
            if row_cnt[0] == 0: # projector defined on lower half only
                for s in range(0, 4):
                    re = proj(h+2,s).real
                    im = proj(h+2,s).imag
                    if re==0 and im==0: ()
                    elif im==0:
                        strRe += sign(re)+in_re(s,c)
                        strIm += sign(re)+in_im(s,c)
                    elif re==0:
                        strRe += sign(-im)+in_im(s,c)
                        strIm += sign(im)+in_re(s,c)
                
            project += h1_re(h,c)+" = "+strRe+";\n"
            project += h1_im(h,c)+" = "+strIm+";\n"

    write_shared = (
"""// store spinor into shared memory
WRITE_SPINOR_SHARED(threadIdx.x, threadIdx.y, threadIdx.z, i);\n
""")

    load_shared_1 = (
"""// load spinor from shared memory
int tx = (threadIdx.x > 0) ? threadIdx.x-1 : blockDim.x-1;
__syncthreads();
READ_SPINOR_SHARED(tx, threadIdx.y, threadIdx.z);\n
""")

    load_shared_2 = (
"""// load spinor from shared memory
int tx = (threadIdx.x + blockDim.x - ((x1+1)&1) ) % blockDim.x;
int ty = (threadIdx.y < blockDim.y - 1) ? threadIdx.y + 1 : 0;
READ_SPINOR_SHARED(tx, ty, threadIdx.z);\n
""")

    load_shared_3 = (
"""// load spinor from shared memory
int tx = (threadIdx.x + blockDim.x - ((x1+1)&1)) % blockDim.x;
int ty = (threadIdx.y > 0) ? threadIdx.y - 1 : blockDim.y - 1;
READ_SPINOR_SHARED(tx, ty, threadIdx.z);\n
""")

    load_shared_4 = (
"""// load spinor from shared memory
int tx = (threadIdx.x + blockDim.x - ((x1+1)&1) ) % blockDim.x;
int tz = (threadIdx.z < blockDim.z - 1) ? threadIdx.z + 1 : 0;
READ_SPINOR_SHARED(tx, threadIdx.y, tz);\n
""")

    load_shared_5 = (
"""// load spinor from shared memory
int tx = (threadIdx.x + blockDim.x - ((x1+1)&1)) % blockDim.x;
int tz = (threadIdx.z > 0) ? threadIdx.z - 1 : blockDim.z - 1;
READ_SPINOR_SHARED(tx, threadIdx.y, tz);\n
""")


    copy_half = ""
    for h in range(0, 2):
        for c in range(0, 3):
            #copy_half += h1_re(h,c)+" = "+("t_proj_scale*" if (dir >= 6) else "")+in_re(h,c)+";  "
            #copy_half += h1_im(h,c)+" = "+("t_proj_scale*" if (dir >= 6) else "")+in_im(h,c)+";\n"
            copy_half += h1_re(h,c)+" = "+in_re(h,c)+";  "
            copy_half += h1_im(h,c)+" = "+in_im(h,c)+";\n"

    copy_half += "\n"

    prep_half_cond1 =  ""
    prep_half_cond1 += "#ifdef MULTI_GPU\n"
    prep_half_cond1 += "if (kernel_type == INTERIOR_KERNEL) {\n"
    prep_half_cond1 += "#endif\n"
    prep_half_cond1 += "\n"

    prep_half_flv1 = ""
    prep_half_flv1 += indent(load_flv1)
    prep_half_flv1 += indent(project)

    prep_half_flv2 = ""
    prep_half_flv2 += indent(load_flv2)
    prep_half_flv2 += indent(project)

    prep_half_cond2 = "\n"
    prep_half_cond2 += "#ifdef MULTI_GPU\n"
    prep_half_cond2 += "} else {\n"
    prep_half_cond2 += "\n"

    prep_face_flv1 = indent(load_half_flv1)
    prep_face_flv2 = indent(load_half_flv2)

    prep_half = indent(copy_half)

    prep_half_cond3 = "}\n"
    prep_half_cond3 += "#endif // MULTI_GPU\n"
    prep_half_cond3 += "\n"
    
    ident = "// identity gauge matrix\n"
    for m in range(0,3):
        for h in range(0,2):
            ident += "spinorFloat "+h2_re(h,m)+" = " + h1_re(h,m) + "; "
            ident += "spinorFloat "+h2_im(h,m)+" = " + h1_im(h,m) + ";\n"
    ident += "\n"
    
    mult = ""
    for m in range(0,3):
        mult += "// multiply row "+`m`+"\n"
        for h in range(0,2):
            re = "spinorFloat "+h2_re(h,m)+" = 0;\n"
            im = "spinorFloat "+h2_im(h,m)+" = 0;\n"
            for c in range(0,3):
                re += h2_re(h,m) + " += " + g_re(dir,m,c) + " * "+h1_re(h,c)+";\n"
                re += h2_re(h,m) + " -= " + g_im(dir,m,c) + " * "+h1_im(h,c)+";\n"
                im += h2_im(h,m) + " += " + g_re(dir,m,c) + " * "+h1_im(h,c)+";\n"
                im += h2_im(h,m) + " += " + g_im(dir,m,c) + " * "+h1_re(h,c)+";\n"
            mult += re + im
        mult += "\n"
    
    reconstruct_flv1 = ""
    for m in range(0,3):

        for h in range(0,2):
            h_out = h
            if row_cnt[0] == 0: # projector defined on lower half only
                h_out = h+2
            reconstruct_flv1 += out1_re(h_out, m) + " += " + h2_re(h,m) + ";\n"
            reconstruct_flv1 += out1_im(h_out, m) + " += " + h2_im(h,m) + ";\n"
    
        for s in range(2,4):
            (h,c) = row(s)
            re = c.real
            im = c.imag
            if im == 0 and re == 0:
                ()
            elif im == 0:
                reconstruct_flv1 += out1_re(s, m) + " " + sign(re) + "= " + h2_re(h,m) + ";\n"
                reconstruct_flv1 += out1_im(s, m) + " " + sign(re) + "= " + h2_im(h,m) + ";\n"
            elif re == 0:
                reconstruct_flv1 += out1_re(s, m) + " " + sign(-im) + "= " + h2_im(h,m) + ";\n"
                reconstruct_flv1 += out1_im(s, m) + " " + sign(+im) + "= " + h2_re(h,m) + ";\n"
        
        reconstruct_flv1 += "\n"

    reconstruct_flv2 = ""
    for m in range(0,3):

        for h in range(0,2):
            h_out = h
            if row_cnt[0] == 0: # projector defined on lower half only
                h_out = h+2
            reconstruct_flv2 += out2_re(h_out, m) + " += " + h2_re(h,m) + ";\n"
            reconstruct_flv2 += out2_im(h_out, m) + " += " + h2_im(h,m) + ";\n"
    
        for s in range(2,4):
            (h,c) = row(s)
            re = c.real
            im = c.imag
            if im == 0 and re == 0:
                ()
            elif im == 0:
                reconstruct_flv2 += out2_re(s, m) + " " + sign(re) + "= " + h2_re(h,m) + ";\n"
                reconstruct_flv2 += out2_im(s, m) + " " + sign(re) + "= " + h2_im(h,m) + ";\n"
            elif re == 0:
                reconstruct_flv2 += out2_re(s, m) + " " + sign(-im) + "= " + h2_im(h,m) + ";\n"
                reconstruct_flv2 += out2_im(s, m) + " " + sign(+im) + "= " + h2_re(h,m) + ";\n"
        
        reconstruct_flv2 += "\n"


    if dir >= 6:
        str += decl_half
        str += "if (gauge_fixed && ga_idx < X4X3X2X1hmX3X2X1h)\n"
        str += block("{\n" + prep_half_cond1 + prep_half_flv1 + prep_half_cond2 + load_half_cond + prep_face_flv1 + prep_half + prep_half_cond3 + ident + reconstruct_flv1 + "}\n" + "{\n" + prep_half_cond1 + prep_half_flv2 + prep_half_cond2 + load_half_cond + prep_face_flv2 + prep_half + prep_half_cond3 + ident + reconstruct_flv2 + "}\n")
        str += " else "
        str += block(load_gauge + reconstruct_gauge + "{\n"+ prep_half_cond1 + prep_half_flv1 + prep_half_cond2 + load_half_cond + prep_face_flv1 + prep_half + prep_half_cond3 + mult + reconstruct_flv1 + "}\n" + "{\n"+ prep_half_cond1 + prep_half_flv2 + prep_half_cond2 + load_half_cond + prep_face_flv2 + prep_half + prep_half_cond3 + mult + reconstruct_flv2 +"}\n")
    else:
        str += decl_half + load_gauge + reconstruct_gauge 
        str +="{\n" + prep_half_cond1 + prep_half_flv1 + prep_half_cond2 + load_half_cond + prep_face_flv1 + prep_half + prep_half_cond3 + mult + reconstruct_flv1 + "}\n" 
        str +="{\n" + prep_half_cond1 + prep_half_flv2 + prep_half_cond2 + load_half_cond + prep_face_flv2 + prep_half + prep_half_cond3 + mult + reconstruct_flv2 + "}\n"     

    if pack_only:
        out = load_spinor + decl_half + project
        out = out.replace("sp_idx", "idx")
        return out
    else:
        return cond + block(str)+"\n\n"
# end def gen

#fixme!
def twisted_rotate(x):

    for h in range(0, 4):
        for c in range(0, 3):
            strRe = ""
            strIm = ""
            for s in range(0, 4):
                # identity
                re = id[4*h+s].real
                im = id[4*h+s].imag
                if re==0 and im==0: ()
                elif im==0:
                    strRe += sign(re)+out1_re(s,c)
                    strIm += sign(re)+out1_im(s,c)
                elif re==0:
                    strRe += sign(-im)+out1_im(s,c)
                    strIm += sign(im)+out1_re(s,c)
                
                # sign(x)*i*mu*gamma_5
                re = igamma5[4*h+s].real
                im = igamma5[4*h+s].imag
                if re==0 and im==0: ()
                elif im==0:
                    strRe += sign(re*x)+out1_re(s,c) + "*a"
                    strIm += sign(re*x)+out1_im(s,c) + "*a"
                elif re==0:
                    strRe += sign(-im*x)+out1_im(s,c) + "*a"
                    strIm += sign(im*x)+out1_re(s,c) + "*a"

            str = "VOLATILE spinorFloat "+tmp_re(h,c)+" = " + strRe + ";\n"
            str += "VOLATILE spinorFloat "+tmp_im(h,c)+" = " + strIm + ";\n"
        str += "\n"
    
    return str+"\n"


def twisted():
    str = ""
    str += twisted_rotate(+1)

    str += "#ifndef DSLASH_XPAY\n"
    str += "//scale by b = 1/(1 + a*a) \n"
    for s in range(0,4):
        for c in range(0,3):
            str += out1_re(s,c) + " = b*" + tmp_re(s,c) + ";\n"
            str += out1_im(s,c) + " = b*" + tmp_im(s,c) + ";\n"
    str += "#else\n"
    for s in range(0,4):
        for c in range(0,3):
            str += out1_re(s,c) + " = " + tmp_re(s,c) + ";\n"
            str += out1_im(s,c) + " = " + tmp_im(s,c) + ";\n"
    str += "#endif // DSLASH_XPAY\n"
    str += "\n"

    return block(str)+"\n"
# end def twisted


def twisted2():
    str = ""
    str += "//Perform twist rotation first:\n"
    if dagger :
       str += "//(1 + i*a*gamma_5 * tau_3 + b * tau_1)\n"
    else:
       str += "//(1 - i*a*gamma_5 * tau_3 + b * tau_1)\n"
    str += "volatile spinorFloat x1_re, x1_im, y1_re, y1_im;\n"
    str += "volatile spinorFloat x2_re, x2_im, y2_re, y2_im;\n\n"

    str += "x1_re = 0.0, x1_im = 0.0;\n"
    str += "y1_re = 0.0, y1_im = 0.0;\n"
    str += "x2_re = 0.0, x2_im = 0.0;\n"
    str += "y2_re = 0.0, y2_im = 0.0;\n\n\n"

    a1 = ""
    a2 = ""

    if dagger :
       a1 += " - a *"
       a2 += " + a *"
    else:     
       a1 += " + a *"
       a2 += " - a *"

    for c in range(0,3):
        for h in range(0,2):
	    #h, h+2
	    str += "// using o1 regs:\n"
	    str += "x1_re = " + out1_re(h,c) + a1 + out1_im(h+2,c) + ";\n"
	    str += "x1_im = " + out1_im(h,c) + a2 + out1_re(h+2,c) + ";\n"
	    str += "x2_re = " + "b * " + out1_re(h,c) + ";\n"
	    str += "x2_im = " + "b * " + out1_im(h,c) + ";\n\n"
	    str += "y1_re = " + out1_re(h+2,c) + a1 + out1_im(h,c) + ";\n"
	    str += "y1_im = " + out1_im(h+2,c) + a2 + out1_re(h,c) + ";\n"
	    str += "y2_re = " + "b * " + out1_re(h+2,c) + ";\n"
	    str += "y2_im = " + "b * " + out1_im(h+2,c) + ";\n\n\n"
	    str += "// using o2 regs:\n"
	    str += "x2_re += " + out2_re(h,c) + a2 + out2_im(h+2,c) + ";\n"
	    str += "x2_im += " + out2_im(h,c) + a1 + out2_re(h+2,c) + ";\n"
	    str += "x1_re += " + "b * " + out2_re(h,c) + ";\n"
	    str += "x1_im += " + "b * " + out2_im(h,c) + ";\n\n"
	    str += "y2_re += " + out2_re(h+2,c) + a2 + out2_im(h,c) + ";\n"
	    str += "y2_im += " + out2_im(h+2,c) + a1 + out2_re(h,c) + ";\n"
	    str += "y1_re += " + "b * " + out2_re(h+2,c) + ";\n"
	    str += "y1_im += " + "b * " + out2_im(h+2,c) + ";\n"
	    str += "\n\n"
            str += out1_re(h,c) + " = x1_re;  " + out1_im(h,c) + " = x1_im;\n"
            str += out1_re(h+2,c) + " = y1_re;  " + out1_im(h+2,c) + " = y1_im;\n"
	    str += "\n"
            str += out2_re(h,c) + " = x2_re;  " + out2_im(h,c) + " = x2_im;\n"
            str += out2_re(h+2,c) + " = y2_re;  " + out2_im(h+2,c) + " = y2_im;\n\n"

    str += "\n"

    return block(str)+"\n"
# end def twisted2


def xpay():
    str = "\n"
    str += "#ifndef DSLASH_XPAY\n"

    for s in range(0,4):
        for c in range(0,3):
            i = 3*s+c
            str += out1_re(s,c) +" *= c;\n"
            str += out1_im(s,c) +" *= c;\n"
    str += "\n"

    for s in range(0,4):
        for c in range(0,3):
            i = 3*s+c
            str += out2_re(s,c) +" *= c;\n"
            str += out2_im(s,c) +" *= c;\n"


    str += "#else\n"
    str += "int tmp = sid;\n"
    str += "{\n"
    str += "READ_ACCUM(ACCUMTEX, sp_stride)\n\n"
    str += "#ifdef SPINOR_DOUBLE\n"

    for s in range(0,4):
        for c in range(0,3):
            i = 3*s+c
            str += out1_re(s,c) +" = c*"+out1_re(s,c)+" + accum"+nthFloat2(2*i+0)+";\n"
            str += out1_im(s,c) +" = c*"+out1_im(s,c)+" + accum"+nthFloat2(2*i+1)+";\n"
    str += "#else\n"

    for s in range(0,4):
        for c in range(0,3):
            i = 3*s+c
            str += out1_re(s,c) +" = c*"+out1_re(s,c)+" + accum"+nthFloat4(2*i+0)+";\n"
            str += out1_im(s,c) +" = c*"+out1_im(s,c)+" + accum"+nthFloat4(2*i+1)+";\n"

    str += "#endif // SPINOR_DOUBLE\n\n"
    str += "}\n"
    str += "{\n"
    str += "sid += fl_stride;\n"
    str += "READ_ACCUM(ACCUMTEX, sp_stride)\n\n"
    str += "#ifdef SPINOR_DOUBLE\n"

    for s in range(0,4):
        for c in range(0,3):
            i = 3*s+c
            str += out2_re(s,c) +" = c*"+out2_re(s,c)+" + accum"+nthFloat2(2*i+0)+";\n"
            str += out2_im(s,c) +" = c*"+out2_im(s,c)+" + accum"+nthFloat2(2*i+1)+";\n"
    str += "#else\n"

    for s in range(0,4):
        for c in range(0,3):
            i = 3*s+c
            str += out2_re(s,c) +" = c*"+out2_re(s,c)+" + accum"+nthFloat4(2*i+0)+";\n"
            str += out2_im(s,c) +" = c*"+out2_im(s,c)+" + accum"+nthFloat4(2*i+1)+";\n"

    str += "#endif // SPINOR_DOUBLE\n\n"
    str += "}\n"
    str += "sid = tmp;\n"
    str += "#endif // DSLASH_XPAY\n"

    return str
# end def xpay


def epilog():
    str = ""
    if dslash:
       str += "#ifdef MULTI_GPU\n"
       str += (
"""
int incomplete = 0; // Have all 8 contributions been computed for this site?

switch(kernel_type) { // intentional fall-through
case INTERIOR_KERNEL:
  incomplete = incomplete || (param.commDim[3] && (x4==0 || x4==X4m1));
case EXTERIOR_KERNEL_T:
  incomplete = incomplete || (param.commDim[2] && (x3==0 || x3==X3m1));
case EXTERIOR_KERNEL_Z:
  incomplete = incomplete || (param.commDim[1] && (x2==0 || x2==X2m1));
case EXTERIOR_KERNEL_Y:
  incomplete = incomplete || (param.commDim[0] && (x1==0 || x1==X1m1));
}

""")    
    str += "\n"
    str += "if (!incomplete)\n"
    str += "#endif // MULTI_GPU\n"
    str += "// apply twisted mass rotation\n"
    
    str += block( "\n" + twisted2() + xpay() )
    
    str += "\n\n"
    str += "// write spinor field back to device memory\n"
    str += "WRITE_FLAVOR_SPINOR();\n\n"

    str += "// undefine to prevent warning when precision is changed\n"
    str += "#undef spinorFloat\n"
    if sharedDslash: 
        str += "#undef WRITE_SPINOR_SHARED\n"
        str += "#undef READ_SPINOR_SHARED\n"
    if sharedFloatsPerFlavor > 0: str += "#undef SHARED_STRIDE\n\n"

    if dslash:
        for m in range(0,3):
            for n in range(0,3):
                i = 3*m+n
                str += "#undef "+g_re(0,m,n)+"\n"
                str += "#undef "+g_im(0,m,n)+"\n"
        str += "\n"

    for s in range(0,4):
        for c in range(0,3):
            i = 3*s+c
            str += "#undef "+in_re(s,c)+"\n"
            str += "#undef "+in_im(s,c)+"\n"
    str += "\n"
#fixme
    for s in range(0,4):
        for c in range(0,3):
            i = 3*s+c
            if 2*i < sharedFloatsPerFlavor:
                str += "#undef "+out1_re(s,c)+"\n"
                if 2*i+1 < sharedFloatsPerFlavor:
                    str += "#undef "+out1_im(s,c)+"\n"
    str += "\n"

    str += "#undef VOLATILE\n" 

    return str
# end def epilog



def generate_dslash():
    return prolog() + gen(0) + gen(1) + gen(2) + gen(3) + gen(4) + gen(5) + gen(6) + gen(7) + epilog()

# generate Wilson-like Dslash kernels
def generate_dslash_kernels(arch):
    print "Generating dslash kernel for sm" + str(arch/10)

    global sharedFloatsPerFlavor
    global sharedDslash
    global dslash
    global dagger
    global twist

    sharedFloatsPerFlavor = 0
    if arch >= 200:
        sharedFloatsPerFlavor = 0
        #sharedDslash = True
        sharedDslash = False    
        name = "fermi"
    elif arch >= 120:
        sharedFloatsPerFlavor = 0
        sharedDslash = False
        name = "gt200"
    else:
        sharedFloatsPerFlavor = 19
        sharedDslash = False
        name = "g80"

    print "Shared floats set to " + str(sharedFloatsPerFlavor)

    dslash = True
    twist = False
    dagger = False

    twist = True
    dagger = False
    #filename = './new_tm_dslash_' + name + '_core.h'
    filename = './dslash_core/tm_ndeg_dslash_core.h'
    print sys.argv[0] + ": generating " + filename;
    f = open(filename, 'w')
    f.write(generate_dslash())
    f.close()

    dagger = True
    #filename = './new_tm_dslash_dagger_' + name + '_core.h'
    filename = './dslash_core/tm_ndeg_dslash_dagger_core.h'
    print sys.argv[0] + ": generating " + filename + "\n";
    f = open(filename, 'w')
    f.write(generate_dslash())
    f.close()

    dslash = False



dslash = False
dagger = False
twist = False
sharedFloatsPerFlavor = 0
sharedDslash = False

# generate dslash kernels
#arch = 200
#generate_dslash_kernels(arch)

arch = 200
generate_dslash_kernels(arch)

#arch = 100
#generate_dslash_kernels(arch)
