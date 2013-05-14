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
def out_re(s, c): return "o"+`s`+`c`+"_re"
def out_im(s, c): return "o"+`s`+`c`+"_im"
#def h1_re(h, c): return ["a","b"][h]+`c`+"_re"		#	ALEX	REMOVED
#def h1_im(h, c): return ["a","b"][h]+`c`+"_im"
#def h2_re(h, c): return ["A","B"][h]+`c`+"_re"
#def h2_im(h, c): return ["A","B"][h]+`c`+"_im"		#	ENDALEX	REMOVED
def a_re(b, s, c): return "a"+`(s+2*b)`+`c`+"_re"
def a_im(b, s, c): return "a"+`(s+2*b)`+`c`+"_im"

def tmp_re(s, c): return "tmp"+`s`+`c`+"_re"
def tmp_im(s, c): return "tmp"+`s`+`c`+"_im"


def def_input_spinor():		#	OK no usa half spinor
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

def def_gauge():		#	OK no usa half spinor
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


def def_gauge_old():		#	OK no usa half spinor
    str = "// gauge link\n"

    for m in range(0,3):
        for n in range(0,3):
            i = 3*m+n
            str += "#undef "+g_re(0,m,n)+"\n"
            str += "#undef "+g_im(0,m,n)+"\n"
            str += "#undef "+g_re(1,m,n)+"\n"
            str += "#undef "+g_im(1,m,n)+"\n"
    str += "\n"

    str += "#ifdef GAUGE_FLOAT2\n"
    for m in range(0,3):
        for n in range(0,3):
            i = 3*m+n
            str += "#define "+g_re(0,m,n)+" G"+nthFloat2(2*i+0)+"\n"
            str += "#define "+g_im(0,m,n)+" G"+nthFloat2(2*i+1)+"\n"

#    str += "// temporaries\n"
#    str += "#define A_re G"+nthFloat2(18)+"\n"
#    str += "#define A_im G"+nthFloat2(19)+"\n"
    str += "\n"
    str += "#else\n"
    for m in range(0,3):
        for n in range(0,3):
            i = 3*m+n
            str += "#define "+g_re(0,m,n)+" G"+nthFloat4(2*i+0)+"\n"
            str += "#define "+g_im(0,m,n)+" G"+nthFloat4(2*i+1)+"\n"

#    str += "// temporaries\n"
#    str += "#define A_re G"+nthFloat4(18)+"\n"
#    str += "#define A_im G"+nthFloat4(19)+"\n"
    str += "\n"
    str += "#endif // GAUGE_DOUBLE\n\n"
            
    str += "// conjugated gauge link\n"
    for m in range(0,3):
        for n in range(0,3):
            i = 3*m+n
            str += "#define "+g_re(1,m,n)+" (+"+g_re(0,n,m)+")\n"	#	ALEX	Midifed sign for the covariant derivative
            str += "#define "+g_im(1,m,n)+" (-"+g_im(0,n,m)+")\n"	#	ALEX	Midifed sign for the covariant derivative
    str += "\n"

    return str
# end def def_gauge


def def_output_spinor():		#	OK no usa half spinor
# sharedDslash = True: input spinors stored in shared memory
# sharedDslash = False: output spinors stored in shared memory
    str = "// output spinor\n"
    for s in range(0,4):
        for c in range(0,3):
            i = 3*s+c
            if 2*i < sharedFloats and not sharedDslash:
                str += "#define "+out_re(s,c)+" s["+`(2*i+0)`+"*SHARED_STRIDE]\n"
            else:
                str += "VOLATILE spinorFloat "+out_re(s,c)+";\n"
            if 2*i+1 < sharedFloats and not sharedDslash:
                str += "#define "+out_im(s,c)+" s["+`(2*i+1)`+"*SHARED_STRIDE]\n"
            else:
                str += "VOLATILE spinorFloat "+out_im(s,c)+";\n"
    return str
# end def def_output_spinor


def prolog():		#	OK no usa half spinor
    global arch

    prolog_str= ("// *** CUDA DSLASH ***\n\n" if not dagger else "// *** CUDA DSLASH DAGGER ***\n\n")
    prolog_str+= "#define DSLASH_SHARED_FLOATS_PER_THREAD "+str(sharedFloats)+"\n\n"

    prolog_str+= (
"""
#if ((CUDA_VERSION >= 4010) && (__COMPUTE_CAPABILITY__ >= 200)) // NVVM compiler
#define VOLATILE
#else // Open64 compiler
#define VOLATILE volatile
#endif
""")

    prolog_str+= def_input_spinor()
    prolog_str+= def_gauge()
    prolog_str+= def_output_spinor()

    if (sharedFloats > 0):
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
    if sharedFloats > 0 and not sharedDslash:
        prolog_str += (
"""
extern __shared__ char s_data[];
""")

        prolog_str += (
"""
VOLATILE spinorFloat *s = (spinorFloat*)s_data + DSLASH_SHARED_FLOATS_PER_THREAD*SHARED_STRIDE*(threadIdx.x/SHARED_STRIDE)
                                  + (threadIdx.x % SHARED_STRIDE);
""")

    prolog_str+= (
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
  //coordsFromIndex(X, x1, x2, x3, x4, sid, param.parity);

  int xt = blockIdx.x*blockDim.x + threadIdx.x;
  int aux = xt+xt;
  if (aux >= X1*X4) return;

  x4 = aux / X1;
  x1 = aux - x4*X1;

  x2 = blockIdx.y*blockDim.y + threadIdx.y;
  if (x2 >= X2) return;

  x3 = blockIdx.z*blockDim.z + threadIdx.z;
  if (x3 >= X3) return;

  x1 += (param.parity + x4 + x3 + x2) &1;
  X = ((x4*X3 + x3)*X2 + x2)*X1 + x1;
  sid = X >> 1; 

""")
    else:
        prolog_str += (
"""
sid = blockIdx.x*blockDim.x + threadIdx.x;
if (sid >= param.threads) return;

#ifdef MULTI_GPU
int face_idx;
if (kernel_type == INTERIOR_KERNEL) {
#endif

  // Inline by hand for the moment and assume even dimensions
  //coordsFromIndex(X, x1, x2, x3, x4, sid, param.parity);

  X = 2*sid;
  int aux1 = X / X1;
  x1 = X - aux1 * X1;
  int aux2 = aux1 / X2;
  x2 = aux1 - aux2 * X2;
  x4 = aux2 / X3;
  x3 = aux2 - x4 * X3;
  aux1 = (param.parity + x4 + x3 + x2) & 1;
  x1 += aux1;
  X += aux1;

""")

    out = ""
    for s in range(0,4):
        for c in range(0,3):
            out += out_re(s,c)+" = 0;  "+out_im(s,c)+" = 0;\n"
    prolog_str+= indent(out)

    prolog_str+= (
"""
#ifdef MULTI_GPU
} else { // exterior kernel

  const int dim = static_cast<int>(kernel_type);
  const int face_volume = param.threads;           // volume of one face
""")

    prolog_str+= ("  const int face_num = 1;\n" if not dagger else "  const int face_num = 0;\n")

    prolog_str+= (
"""
  face_idx = sid;               // index into the respective face

  // ghostOffset is scaled to include body (includes stride) and number of FloatN arrays (SPINOR_HOP)
  // face_idx not sid since faces are spin projected and share the same volume index (modulo UP/DOWN reading)
  //sp_idx = face_idx + param.ghostOffset[dim];

#if (DD_PREC==2) // half precision
  sp_norm_idx = sid + param.ghostNormOffset[static_cast<int>(kernel_type)];
#endif

  coordsFromFaceIndex<1>(X, sid, x1, x2, x3, x4, face_idx, face_volume, dim, face_num, param.parity);

""")

    out = ""
    for s in range(0,4):
        for c in range(0,3):
            out += out_re(s,c)+" = 0.;  "+out_im(s,c)+" = 0.;\n"
    prolog_str+= indent(out)
    prolog_str+= "}\n"
    prolog_str+= "#endif // MULTI_GPU\n\n\n"
            
    return prolog_str
# end def prolog

def gen(dir, pack_only=False):		#	 WRONG usa half spinor A SACO
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
#    for h in range(0, 2):	#	ALEX	REMOVED

#    for h in range(0, 4):	#	ALEX	Cambio a 4
#        for c in range(0, 3):
#            decl_half += "spinorFloat "+h1_re(h,c)+", "+h1_im(h,c)+";\n";	#	ENDALEX	REMOVED
    decl_half += "\n"

    load_spinor = "// read spinor from device memory\n"
#    if row_cnt[0] == 0:							#	ALEX	Hasta el de abajo
#        load_spinor += "READ_SPINOR_DOWN(SPINORTEX, sp_stride, sp_idx, sp_idx);\n"
#    elif row_cnt[2] == 0:
#        load_spinor += "READ_SPINOR_UP(SPINORTEX, sp_stride, sp_idx, sp_idx);\n"
#    else:
#         load_spinor += "READ_SPINOR(SPINORTEX, sp_stride, sp_idx, sp_idx);\n"
    load_spinor += "READ_SPINOR(SPINORTEX, sp_stride, sp_idx, sp_idx);\n"	#	ALEX	Leo el spinor completo
    load_spinor += "\n"

    load_half = ""
    load_half += "const int sp_stride_pad = ghostFace[static_cast<int>(kernel_type)];\n"
    #load_half += "#if (DD_PREC==2) // half precision\n"
    #load_half += "const int sp_norm_idx = sid + param.ghostNormOffset[static_cast<int>(kernel_type)];\n"
    #load_half += "#endif\n"

#    if dir >= 6: load_half += "const int t_proj_scale = TPROJSCALE;\n"		#	ALEX	This is useless
    load_half += "\n"
#    load_half += "// read half spinor from device memory\n"
    load_half += "// read full spinor from device memory\n"			#	ALEX	half -> full

# we have to use the same volume index for backwards and forwards gathers
# instead of using READ_UP_SPINOR and READ_DOWN_SPINOR, just use READ_HALF_SPINOR with the appropriate shift
#    if (dir+1) % 2 == 0: load_half += "READ_HALF_SPINOR(SPINORTEX, sp_stride_pad, sp_idx, sp_norm_idx);\n\n"
#    else: load_half += "READ_HALF_SPINOR(SPINORTEX, sp_stride_pad, sp_idx + (SPINOR_HOP/2)*sp_stride_pad, sp_norm_idx);\n\n"
    if (dir+1) % 2 == 0: load_half += "READ_SPINOR(SPINORTEX, sp_stride_pad, sp_idx, sp_norm_idx);\n\n"				#	ALEX	Se lee siempre el spinor completo
    else: load_half += "READ_SPINOR(SPINORTEX, sp_stride_pad, sp_idx /*+ (SPINOR_HOP)*sp_stride_pad*/, sp_norm_idx);\n\n"		#	ALEX	Se lee siempre el spinor completo (PROBLEM-> SPINOR_HOP/2??
    load_gauge = "// read gauge matrix from device memory\n"
    load_gauge += "READ_GAUGE_MATRIX(G, GAUGE"+`dir%2`+"TEX, "+`dir`+", ga_idx, ga_stride);\n\n"

    reconstruct_gauge = "// reconstruct gauge matrix\n"
    reconstruct_gauge += "RECONSTRUCT_GAUGE_MATRIX("+`dir`+");\n\n"

#    project = "// project spinor into half spinors\n"

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

    project = "// Do nothing useful with the spinors\n"		#	ALEX	Do nothing. Really. Up to ENDALEX

#    for h in range(0, 2):


#    for h in range(0, 4):		#	ALEX	REMOVED
#        for c in range(0, 3):		#	ALEX	REMOVED

#            strRe = ""
#            strIm = ""
#            for s in range(0, 4):
#                re = proj(h,s).real
#                im = proj(h,s).imag
#                if re==0 and im==0: ()
#                elif im==0:
#                    strRe += sign(re)+in_re(s,c)
#                    strIm += sign(re)+in_im(s,c)
#                elif re==0:
#                    strRe += sign(-im)+in_im(s,c)
#                    strIm += sign(im)+in_re(s,c)
#            if row_cnt[0] == 0: # projector defined on lower half only
#                for s in range(0, 4):
#                    re = proj(h+2,s).real
#                    im = proj(h+2,s).imag
#                    if re==0 and im==0: ()
#                    elif im==0:
#                        strRe += sign(re)+in_re(s,c)
#                        strIm += sign(re)+in_im(s,c)
#                    elif re==0:
#                        strRe += sign(-im)+in_im(s,c)
#                        strIm += sign(im)+in_re(s,c)
#                
#            project += h1_re(h,c)+" = "+strRe+";\n"
#            project += h1_im(h,c)+" = "+strIm+";\n"

#            project += h1_re(h,c)+" = "+in_re(h,c)+";\n"	#	ALEX	REMOVED
#            project += h1_im(h,c)+" = "+in_im(h,c)+";\n"	#	ALEX	REMOVED

    copy_half = ""
#    for h in range(0, 2):	#	ALEX	Range:2->4

#    for h in range(0, 4):	#	ALEX	REMOVED
#        for c in range(0, 3):	#	ALEX	REMOVED

#            copy_half += h1_re(h,c)+" = "+("t_proj_scale*" if (dir >= 6) else "")+in_re(h,c)+";  "
#            copy_half += h1_im(h,c)+" = "+("t_proj_scale*" if (dir >= 6) else "")+in_im(h,c)+";\n"

#            copy_half += h1_re(h,c)+" = "+in_re(h,c)+";  "	#	ALEX	REMOVED
#            copy_half += h1_im(h,c)+" = "+in_im(h,c)+";\n"	#	ALEX	REMOVED

    copy_half += "\n"
#	ENDALEX


    mult = ""
    for m in range(0,3):
        mult += "// multiply row "+`m`+"\n"
#        for h in range(0,2):
        for h in range(0,4):		#	ALEX	Range: 2->4
#            re = "spinorFloat "+h2_re(h,m)+" = 0;\n"
#            im = "spinorFloat "+h2_im(h,m)+" = 0;\n"
	    re  = ""
	    im  = ""
#	    re += "{ \n"
#	    re += "\ndouble	tempR  = 0.;\n\n"
#	    im += "\ndouble	tempI  = 0.;\n\n"
#            fn = ""
            for c in range(0,3):
#                re += "tempR += " + g_re(dir,m,c) + " * "+in_re(h,c)+";\n"
#                re += "tempR -= " + g_im(dir,m,c) + " * "+in_im(h,c)+";\n"
#                im += "tempI += " + g_re(dir,m,c) + " * "+in_im(h,c)+";\n"
#                im += "tempI += " + g_im(dir,m,c) + " * "+in_re(h,c)+";\n"

#	    if dir < 6:
#		if dir%2:
#			fn += out_re(h,m) + " += tempR*(1.+C) - tempI*S;\n"
#			fn += out_im(h,m) + " += tempI*(1.+C) + tempR*S;\n"
#		else:
#			fn += out_re(h,m) + " += tempR*(1.+C) + tempI*S;\n"
#			fn += out_im(h,m) + " += tempI*(1.+C) - tempR*S;\n"
#	    else:
#		fn += out_re(h,m) + " += tempR;\n"
#		fn += out_im(h,m) + " += tempI;\n"
	        if dagger:
		    if dir%2:
		        re += out_re(h,m) + " += " + g_re(dir,m,c) + " * "+in_re(h,c)+";\n"
	        	re += out_re(h,m) + " -= " + g_im(dir,m,c) + " * "+in_im(h,c)+";\n"
		        im += out_im(h,m) + " += " + g_re(dir,m,c) + " * "+in_im(h,c)+";\n"
        		im += out_im(h,m) + " += " + g_im(dir,m,c) + " * "+in_re(h,c)+";\n"
	        else:
		    if dir%2 == 0:
		        re += out_re(h,m) + " += " + g_re(dir,m,c) + " * "+in_re(h,c)+";\n"
        		re += out_re(h,m) + " -= " + g_im(dir,m,c) + " * "+in_im(h,c)+";\n"
		        im += out_im(h,m) + " += " + g_re(dir,m,c) + " * "+in_im(h,c)+";\n"
        		im += out_im(h,m) + " += " + g_im(dir,m,c) + " * "+in_re(h,c)+";\n"

            mult += re + im #+ fn + "\n}\n"
        mult += "\n"

    prep_half = ""
    prep_half += "#ifdef MULTI_GPU\n"
    prep_half += "if (kernel_type == INTERIOR_KERNEL) {\n"
    prep_half += "#endif\n"
    prep_half += "\n"

    if sharedDslash:
        if dir == 0:
            prep_half += indent(load_spinor)
            prep_half += indent(write_shared)
            prep_half += indent(project)
        elif dir == 1:
            prep_half += indent(load_shared_1)
            prep_half += indent(project)
        elif dir == 2:
            prep_half += indent("if (threadIdx.y == blockDim.y-1 && blockDim.y < X2 ) {\n")
            prep_half += indent(load_spinor)
            prep_half += indent(project)
            prep_half += indent("} else {")
            prep_half += indent(load_shared_2)
            prep_half += indent(project)
            prep_half += indent("}")
        elif dir == 3:
            prep_half += indent("if (threadIdx.y == 0 && blockDim.y < X2) {\n")
            prep_half += indent(load_spinor)
            prep_half += indent(project)
            prep_half += indent("} else {")
            prep_half += indent(load_shared_3)
            prep_half += indent(project)
            prep_half += indent("}")
        elif dir == 4:
            prep_half += indent("if (threadIdx.z == blockDim.z-1 && blockDim.z < X3) {\n")
            prep_half += indent(load_spinor)
            prep_half += indent(project)
            prep_half += indent("} else {")
            prep_half += indent(load_shared_4)
            prep_half += indent(project)
            prep_half += indent("}")
        elif dir == 5:
            prep_half += indent("if (threadIdx.z == 0 && blockDim.z < X3) {\n")
            prep_half += indent(load_spinor)
            prep_half += indent(project)
            prep_half += indent("} else {")
            prep_half += indent(load_shared_5)
            prep_half += indent(project)
            prep_half += indent("}")
        else:
            prep_half += indent(load_spinor)
            prep_half += indent(project)
    else:
        prep_half += indent(load_spinor)
        prep_half += indent(project)
        prep_half += reconstruct_gauge
        prep_half += mult

    prep_half += "\n"
    prep_half += "#ifdef MULTI_GPU\n"
    prep_half += "//JARLLLL \n"
    prep_half += "} else {\n"
    prep_half += "\n"
    prep_half += indent(load_half)
    prep_half += indent(reconstruct_gauge)
    prep_half += indent(mult)
    prep_half += "}\n"
    prep_half += "#endif // MULTI_GPU\n"
    prep_half += "\n"
    ident = "// identity gauge matrix\n"
    for m in range(0,3):
#        for h in range(0,2):
        for h in range(0,4):		#	ALEX	Range: 2->4
#            ident += "spinorFloat "+h2_re(h,m)+" = " + h1_re(h,m) + "; "
#            ident += "spinorFloat "+h2_im(h,m)+" = " + h1_im(h,m) + ";\n"
            ident += out_re(h,m)+" = " + in_re(h,m) + "; "
            ident += out_im(h,m)+" = " + in_im(h,m) + ";\n"
    ident += "\n"
    
    reconstruct = ""
#    for m in range(0,3):

#        for h in range(0,2):
#        for h in range(0,4):		#	ALEX	Range: 2->4	AND REMOVED
#            h_out = h
#            reconstruct += out_re(h_out, m) + " = " + h2_re(h,m) + ";\n"	#	ALEX	Corrected the output for full spinor			AND REMOVED
#            reconstruct += out_im(h_out, m) + " = " + h2_im(h,m) + ";\n"	#	ALEX	Corrected the output for full spinor, up to ENDALEX	AND REMOVED
#            if row_cnt[0] == 0: # projector defined on lower half only
#                h_out = h+2
#            reconstruct += out_re(h_out, m) + " += " + h2_re(h,m) + ";\n"
#            reconstruct += out_im(h_out, m) + " += " + h2_im(h,m) + ";\n"
#    
#        for s in range(2,4):
#            (h,c) = row(s)
#            re = c.real
#            im = c.imag
#            if im == 0 and re == 0:
#                ()
#            elif im == 0:
#                reconstruct += out_re(s, m) + " " + sign(re) + "= " + h2_re(h,m) + ";\n"
#                reconstruct += out_im(s, m) + " " + sign(re) + "= " + h2_im(h,m) + ";\n"
#            elif re == 0:
#                reconstruct += out_re(s, m) + " " + sign(-im) + "= " + h2_im(h,m) + ";\n"
#                reconstruct += out_im(s, m) + " " + sign(+im) + "= " + h2_re(h,m) + ";\n"
#	ENDALEX
#        reconstruct += "\n"
    reconstruct += "\n"

    if dir >= 6:
#        str += block(load_gauge + decl_half + prep_half + reconstruct_gauge + mult + reconstruct)	#	ALEX	Simplify
        str += block(load_gauge + decl_half + prep_half)
    else:
#        str += load_gauge + decl_half + prep_half + reconstruct_gauge + mult + reconstruct		#	ALEX	Simplify
        str += load_gauge + decl_half + prep_half
    
    if dagger:
        if dir%2 == 0:
	    nothing = ""
	    return nothing
    else:
        if dir%2:
	    nothing = ""
	    return nothing

    if pack_only:
        out = load_spinor + decl_half + project
        out = out.replace("sp_idx", "idx")
        return out
    else:
        return cond + block(str)+"\n\n"
# end def gen

def epilog():			#	OK, no usa half spinors
    str = ""

    str += "\n\n"
    str += "// write spinor field back to device memory\n"
    str += "WRITE_SPINOR(sp_stride);\n\n"

    str += "// undefine to prevent warning when precision is changed\n"
    str += "#undef spinorFloat\n"
    str += "#undef SHARED_STRIDE\n\n"

#    str += "#undef A_re\n"
#    str += "#undef A_im\n\n"

    for m in range(0,3):
        for n in range(0,3):
            i = 3*m+n
            str += "#undef "+g_re(0,m,n)+"\n"
            str += "#undef "+g_im(0,m,n)+"\n"
            str += "#undef "+g_re(1,m,n)+"\n"
            str += "#undef "+g_im(1,m,n)+"\n"
    str += "\n"

    for s in range(0,4):
        for c in range(0,3):
            i = 3*s+c
            str += "#undef "+in_re(s,c)+"\n"
            str += "#undef "+in_im(s,c)+"\n"
    str += "\n"

    for s in range(0,4):
        for c in range(0,3):
            i = 3*s+c
            if 2*i < sharedFloats:
                str += "#undef "+out_re(s,c)+"\n"
                if 2*i+1 < sharedFloats:
                    str += "#undef "+out_im(s,c)+"\n"
    str += "\n"

    return str
# end def epilog


def pack_face(facenum):		#	OK usa un poco de half spinor -> CORREGIDO
    str = "\n"
    str += "switch(dim) {\n"
    for dim in range(0,4):
        str += "case "+`dim`+":\n"
        proj = gen(2*dim+facenum, pack_only=True)
        proj += "\n"
        proj += "// write half spinor back to device memory\n"
#        proj += "WRITE_HALF_SPINOR(face_volume, face_idx);\n"
        proj += "WRITE_SPINOR(face_volume, face_idx);\n"
        str += indent(block(proj)+"\n"+"break;\n")
    str += "}\n\n"
    return str
# end def pack_face


def generate_pack():		#	OK no usa half spinor
    assert (sharedFloats == 0)
    str = ""
    str += def_input_spinor()
    str += "#include \"io_spinor.h\"\n\n"

    str += "if (face_num) "
    str += block(pack_face(1))
    str += " else "
    str += block(pack_face(0))

    str += "\n\n"
    str += "// undefine to prevent warning when precision is changed\n"
    str += "#undef spinorFloat\n"
    str += "#undef SHARED_STRIDE\n\n"

    for s in range(0,4):
        for c in range(0,3):
            i = 3*s+c
            str += "#undef "+in_re(s,c)+"\n"
            str += "#undef "+in_im(s,c)+"\n"
    str += "\n"

    return str
# end def generate_pack


def generate_dslash(direction):	#	OK no usa half spinor
    return prolog() + gen(2*direction+0) + gen(2*direction+1) + epilog()


# To fit 192 threads/SM (single precision) with 16K shared memory, set sharedFloats to 19 or smaller

sharedFloats = 0
sharedDslash = False
dslash = True
pack = False
arch = 200
name = "fermi"
if(len(sys.argv) > 1):
    if (sys.argv[1] == '--shared'):
        sharedFloats = int(sys.argv[2])
print "Shared floats set to " + str(sharedFloats);


dagger = False
print sys.argv[0] + ": generating covDev_mu0_core.h";
f = open('./covDev_mu0_core.h', 'w')
f.write(generate_dslash(0))
f.close()

dagger = True
print sys.argv[0] + ": generating covDev_mu0_dagger_core.h";
f = open('./covDev_mu0_dagger_core.h', 'w')
f.write(generate_dslash(0))
f.close()

dagger = False
print sys.argv[0] + ": generating covDev_mu1_core.h";
f = open('./covDev_mu1_core.h', 'w')
f.write(generate_dslash(1))
f.close()

dagger = True
print sys.argv[0] + ": generating covDev_mu1_dagger_core.h";
f = open('./covDev_mu1_dagger_core.h', 'w')
f.write(generate_dslash(1))
f.close()

dagger = False
print sys.argv[0] + ": generating covDev_mu2_core.h";
f = open('./covDev_mu2_core.h', 'w')
f.write(generate_dslash(2))
f.close()

dagger = True
print sys.argv[0] + ": generating covDev_mu2_dagger_core.h";
f = open('./covDev_mu2_dagger_core.h', 'w')
f.write(generate_dslash(2))
f.close()

dagger = False
print sys.argv[0] + ": generating covDev_mu3_core.h";
f = open('./covDev_mu3_core.h', 'w')
f.write(generate_dslash(3))
f.close()

dagger = True
print sys.argv[0] + ": generating covDev_mu3_dagger_core.h";
f = open('./covDev_mu3_dagger_core.h', 'w')
f.write(generate_dslash(3))
f.close()
