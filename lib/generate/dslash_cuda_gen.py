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
def h1_re(h, c): return ["a","b"][h]+`c`+"_re"
def h1_im(h, c): return ["a","b"][h]+`c`+"_im"
def h2_re(h, c): return ["A","B"][h]+`c`+"_re"
def h2_im(h, c): return ["A","B"][h]+`c`+"_im"
def c_re(b, sm, cm, sn, cn): return "c"+`(sm+2*b)`+`cm`+"_"+`(sn+2*b)`+`cn`+"_re"
def c_im(b, sm, cm, sn, cn): return "c"+`(sm+2*b)`+`cm`+"_"+`(sn+2*b)`+`cn`+"_im"
def a_re(b, s, c): return "a"+`(s+2*b)`+`c`+"_re"
def a_im(b, s, c): return "a"+`(s+2*b)`+`c`+"_im"

def tmp_re(s, c): return "tmp"+`s`+`c`+"_re"
def tmp_im(s, c): return "tmp"+`s`+`c`+"_im"


def def_input_spinor():
    str = ""
    str += "// input spinor\n"
    str += "#ifdef SPINOR_DOUBLE\n"
    str += "#define spinorFloat double\n"
    for s in range(0,4):
        for c in range(0,3):
            i = 3*s+c
            str += "#define "+in_re(s,c)+" I"+nthFloat2(2*i+0)+"\n"
            str += "#define "+in_im(s,c)+" I"+nthFloat2(2*i+1)+"\n"
    str += "\n"
    str += "#else\n"
    str += "#define spinorFloat float\n"
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

    str += "// temporaries\n"
    str += "#define A_re G"+nthFloat2(18)+"\n"
    str += "#define A_im G"+nthFloat2(19)+"\n"
    str += "\n"
    str += "#else\n"
    for m in range(0,3):
        for n in range(0,3):
            i = 3*m+n
            str += "#define "+g_re(0,m,n)+" G"+nthFloat4(2*i+0)+"\n"
            str += "#define "+g_im(0,m,n)+" G"+nthFloat4(2*i+1)+"\n"

    str += "// temporaries\n"
    str += "#define A_re G"+nthFloat4(18)+"\n"
    str += "#define A_im G"+nthFloat4(19)+"\n"
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


def def_clover():
    str = "// first chiral block of inverted clover term\n"
    str += "#ifdef CLOVER_DOUBLE\n"
    i = 0
    for m in range(0,6):
        s = m/3
        c = m%3
        str += "#define "+c_re(0,s,c,s,c)+" C"+nthFloat2(i)+"\n"
        i += 1
    for n in range(0,6):
        sn = n/3
        cn = n%3
        for m in range(n+1,6):
            sm = m/3
            cm = m%3
            str += "#define "+c_re(0,sm,cm,sn,cn)+" C"+nthFloat2(i)+"\n"
            str += "#define "+c_im(0,sm,cm,sn,cn)+" C"+nthFloat2(i+1)+"\n"
            i += 2
    str += "#else\n"
    i = 0
    for m in range(0,6):
        s = m/3
        c = m%3
        str += "#define "+c_re(0,s,c,s,c)+" C"+nthFloat4(i)+"\n"
        i += 1
    for n in range(0,6):
        sn = n/3
        cn = n%3
        for m in range(n+1,6):
            sm = m/3
            cm = m%3
            str += "#define "+c_re(0,sm,cm,sn,cn)+" C"+nthFloat4(i)+"\n"
            str += "#define "+c_im(0,sm,cm,sn,cn)+" C"+nthFloat4(i+1)+"\n"
            i += 2
    str += "#endif // CLOVER_DOUBLE\n\n"

    for n in range(0,6):
        sn = n/3
        cn = n%3
        for m in range(0,n):
            sm = m/3
            cm = m%3
            str += "#define "+c_re(0,sm,cm,sn,cn)+" (+"+c_re(0,sn,cn,sm,cm)+")\n"
            str += "#define "+c_im(0,sm,cm,sn,cn)+" (-"+c_im(0,sn,cn,sm,cm)+")\n"
    str += "\n"

    str += "// second chiral block of inverted clover term (reuses C0,...,C9)\n"
    for n in range(0,6):
        sn = n/3
        cn = n%3
        for m in range(0,6):
            sm = m/3
            cm = m%3
            str += "#define "+c_re(1,sm,cm,sn,cn)+" "+c_re(0,sm,cm,sn,cn)+"\n"
            if m != n: str += "#define "+c_im(1,sm,cm,sn,cn)+" "+c_im(0,sm,cm,sn,cn)+"\n"
    str += "\n"

    return str
# end def def_clover


def def_output_spinor():
    str = "// output spinor\n"
    for s in range(0,4):
        for c in range(0,3):
            i = 3*s+c
            if 2*i < sharedFloats:
                str += "#define "+out_re(s,c)+" s["+`(2*i+0)`+"*SHARED_STRIDE]\n"
            else:
                str += "volatile spinorFloat "+out_re(s,c)+";\n"
            if 2*i+1 < sharedFloats:
                str += "#define "+out_im(s,c)+" s["+`(2*i+1)`+"*SHARED_STRIDE]\n"
            else:
                str += "volatile spinorFloat "+out_im(s,c)+";\n"
    return str
# end def def_output_spinor


def prolog():
    str = ("// *** CUDA DSLASH ***\n\n" if not dagger else "// *** CUDA DSLASH DAGGER ***\n\n")
    str += "#define SHARED_FLOATS_PER_THREAD "+`sharedFloats`+"\n\n"

    str += def_input_spinor()
    str += def_gauge()
    if clover == True: str += def_clover()
    str += def_output_spinor()

    if sharedFloats > 0: str += (
"""
#ifdef SPINOR_DOUBLE
#if (__CUDA_ARCH__ >= 200)
#define SHARED_STRIDE 16 // to avoid bank conflicts on Fermi
#else
#define SHARED_STRIDE  8 // to avoid bank conflicts on G80 and GT200
#endif
extern __shared__ spinorFloat sd_data[];
volatile spinorFloat *s = sd_data + SHARED_FLOATS_PER_THREAD*SHARED_STRIDE*(threadIdx.x/SHARED_STRIDE)
                                  + (threadIdx.x % SHARED_STRIDE);
#else
#if (__CUDA_ARCH__ >= 200)
#define SHARED_STRIDE 32 // to avoid bank conflicts on Fermi
#else
#define SHARED_STRIDE 16 // to avoid bank conflicts on G80 and GT200
#endif
extern __shared__ spinorFloat ss_data[];
volatile spinorFloat *s = ss_data + SHARED_FLOATS_PER_THREAD*SHARED_STRIDE*(threadIdx.x/SHARED_STRIDE)
                                  + (threadIdx.x % SHARED_STRIDE);
#endif
""")

    str += (
"""
#include "read_gauge.h"
#include "read_clover.h"
#include "io_spinor.h"

int X, x1, x2, x3, x4, sp_idx;

#ifndef MULTI_GPU
#define t_proj_scale 2 // coefficient for temporal spin projection is always 2 in single-GPU mode
#else // MULTI_GPU
#if (DD_PREC==2) // half precision
int sp_norm_idx;
#else
const int &sp_norm_idx = sp_idx;
#endif
#endif // MULTI_GPU

int sid = blockIdx.x*blockDim.x + threadIdx.x;
if (sid >= param.threads) return;

#ifdef MULTI_GPU
if (param.kernel_type = INTERIOR_KERNEL) {
#endif

  coordsFromIndex(X, x1, x2, x3, x4, sid, param.parity);

""")

    out = ""
    for s in range(0,4):
        for c in range(0,3):
            out += out_re(s,c)+" = 0;  "+out_im(s,c)+" = 0;\n"
    str += indent(out)

    str += (
"""
#ifdef MULTI_GPU
} else { // exterior kernel

  const int dim = static_cast<int>(param.kernel_type);
  const int face_volume = (param.threads >> 1);
  const int face_num = (face_idx >= face_volume);
  const int face_idx = sid - face_num*face_volume;

  sp_idx = sid + param.ghostOffset[dim];

#if (DD_PREC==2) // half precision
  sp_norm_idx = sid + param.ghostNormOffset[dim];
#endif
    
  coordsFromFaceIndex<1>(X, sid, x1, x2, x3, x4, face_idx, face_volume, dim, face_num, param.parity);

  READ_SPINOR(INTERTEX, sp_stride, sid, sid);

""")

    out = ""
    for s in range(0,4):
        for c in range(0,3):
            out += out_re(s,c)+" = "+in_re(s,c)+";  "+out_im(s,c)+" = "+in_re(s,c)+";\n"
    str += indent(out)
    str += "}\n"
    str += "#endif // MULTI_GPU\n\n\n"
            
    return str
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
    cond += "if ( (param.kernel_type == INTERIOR_KERNEL && (!param.ghostDim["+`dir/2`+"] || "+interior[dir]+")) ||\n"
    cond += "     (param.kernel_type == EXTERIOR_KERNEL_"+dim[dir/2]+" && "+boundary[dir]+") )\n"
    cond += "#endif\n"

    str = ""
    
    projName = "P"+`dir/2`+["-","+"][projIdx%2]
    str += "// Projector "+projName+"\n"
    for l in projStr.splitlines():
        str += "// "+l+"\n"
    str += "\n"

    str += "#ifdef MULTI_GPU\n"
    str += "if (param.kernel_type = INTERIOR_KERNEL) {\n"
    str += "#endif\n"
    str += "  sp_idx = ("+boundary[dir]+" ? "+sp_idx_wrap[dir]+" : "+sp_idx[dir]+") >> 1;\n"
    str += "#ifdef MULTI_GPU\n"
    str += "}\n"
    str += "#endif // MULTI_GPU\n"

    str += "\n"
    if dir % 2 == 0:
        str += "int ga_idx = sid;\n"
    else:
        str += "#ifdef MULTI_GPU\n"
        str += "int ga_idx = ((param.kernel_type == INTERIOR_KERNEL) ? sp_idx : Vh+sid);\n"
        str += "#else\n"
        str += "int ga_idx = sp_idx;\n"
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

    load_spinor = ""
    if dir >= 6: load_spinor += "spinorFloat t_proj_scale = spinorFloat(2.0);\n\n"
    load_spinor += "// read spinor from device memory\n"
    if row_cnt[0] == 0:
        load_spinor += "READ_SPINOR_DOWN(SPINORTEX, sp_stride, sp_idx, sp_idx);\n"
    elif row_cnt[2] == 0:
        load_spinor += "READ_SPINOR_UP(SPINORTEX, sp_stride, sp_idx, sp_idx);\n"
    else:
        load_spinor += "READ_SPINOR(SPINORTEX, sp_stride, sp_idx, sp_idx);\n"
    load_spinor += "\n"

    load_half = ""
    load_half += "const int dim = static_cast<int>(param.kernel_type);\n"
    load_half += "const int sp_stride_pad = ghostFace[dim];\n"
    if dir >= 6: load_half += "const int t_proj_scale = TPROJSCALE;\n"
    load_half += "\n"
    load_half += "// read half spinor from device memory\n"
    load_half += "READ_HALF_SPINOR(SPINORTEX, sp_stride_pad, sp_idx, sp_norm_idx);\n\n"

    load_gauge = "// read gauge matrix from device memory\n"
    load_gauge += "READ_GAUGE_MATRIX(G, GAUGE"+`dir%2`+"TEX, "+`dir`+", ga_idx, ga_stride);\n\n"

    reconstruct_gauge = "// reconstruct gauge matrix\n"
    reconstruct_gauge += "RECONSTRUCT_GAUGE_MATRIX("+`dir`+");\n\n"

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

    copy_half = ""
    for h in range(0, 2):
        for c in range(0, 3):
            copy_half += h1_re(h,c)+" = "+("t_proj_scale*" if (dir >= 6) else "")+out_re(h,c)+";  "
            copy_half += h1_im(h,c)+" = "+("t_proj_scale*" if (dir >= 6) else "")+out_im(h,c)+";\n"
    copy_half += "\n"

    prep_half = ""
    prep_half += "#ifdef MULTI_GPU\n"
    prep_half += "if (param.kernel_type == INTERIOR_KERNEL) {\n"
    prep_half += "#endif\n"
    prep_half += "\n"
    prep_half += indent(load_spinor)
    prep_half += indent(project)
    prep_half += "\n"
    prep_half += "#ifdef MULTI_GPU\n"
    prep_half += "} else {\n"
    prep_half += "\n"
    prep_half += indent(load_half)
    prep_half += indent(copy_half)
    prep_half += "}\n"
    prep_half += "#endif // MULTI_GPU\n"
    prep_half += "\n"
    
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
    
    reconstruct = ""
    for m in range(0,3):

        for h in range(0,2):
            h_out = h
            if row_cnt[0] == 0: # projector defined on lower half only
                h_out = h+2
            reconstruct += out_re(h_out, m) + " += " + h2_re(h,m) + ";\n"
            reconstruct += out_im(h_out, m) + " += " + h2_im(h,m) + ";\n"
    
        for s in range(2,4):
            (h,c) = row(s)
            re = c.real
            im = c.imag
            if im == 0 and re == 0:
                ()
            elif im == 0:
                reconstruct += out_re(s, m) + " " + sign(re) + "= " + h2_re(h,m) + ";\n"
                reconstruct += out_im(s, m) + " " + sign(re) + "= " + h2_im(h,m) + ";\n"
            elif re == 0:
                reconstruct += out_re(s, m) + " " + sign(-im) + "= " + h2_im(h,m) + ";\n"
                reconstruct += out_im(s, m) + " " + sign(+im) + "= " + h2_re(h,m) + ";\n"
        
        reconstruct += "\n"

    if dir >= 6:
        str += "if (gauge_fixed && ga_idx < X4X3X2X1hmX3X2X1h)\n"
        str += block(decl_half + prep_half + ident + reconstruct)
        str += " else "
        str += block(load_gauge + decl_half + prep_half + reconstruct_gauge + mult + reconstruct)
    else:
        str += load_gauge + decl_half + prep_half + reconstruct_gauge + mult + reconstruct
    
    if pack_only:
        out = load_spinor + decl_half + project
        out = out.replace("sp_stride_t", "sp_stride")
        out = out.replace("sp_idx", "idx")
        out = out.replace("sp_norm_idx", "idx")
        return out
    else:
        str = str.replace("2*", "t_proj_scale*")
        return cond + block(str)+"\n\n"
# end def gen


def to_chiral_basis(c):
    str = ""
    str += "spinorFloat "+a_re(0,0,c)+" = -"+out_re(1,c)+" - "+out_re(3,c)+";\n"
    str += "spinorFloat "+a_im(0,0,c)+" = -"+out_im(1,c)+" - "+out_im(3,c)+";\n"
    str += "spinorFloat "+a_re(0,1,c)+" =  "+out_re(0,c)+" + "+out_re(2,c)+";\n"
    str += "spinorFloat "+a_im(0,1,c)+" =  "+out_im(0,c)+" + "+out_im(2,c)+";\n"
    str += "spinorFloat "+a_re(0,2,c)+" = -"+out_re(1,c)+" + "+out_re(3,c)+";\n"
    str += "spinorFloat "+a_im(0,2,c)+" = -"+out_im(1,c)+" + "+out_im(3,c)+";\n"
    str += "spinorFloat "+a_re(0,3,c)+" =  "+out_re(0,c)+" - "+out_re(2,c)+";\n"
    str += "spinorFloat "+a_im(0,3,c)+" =  "+out_im(0,c)+" - "+out_im(2,c)+";\n"
    str += "\n"

    for s in range (0,4):
        str += out_re(s,c)+" = "+a_re(0,s,c)+";  "
        str += out_im(s,c)+" = "+a_im(0,s,c)+";\n"

    return block(str)+"\n\n"
# end def to_chiral_basis


def from_chiral_basis(c): # note: factor of 1/2 is included in clover term normalization
    str = ""
    str += "spinorFloat "+a_re(0,0,c)+" =  "+out_re(1,c)+" + "+out_re(3,c)+";\n"
    str += "spinorFloat "+a_im(0,0,c)+" =  "+out_im(1,c)+" + "+out_im(3,c)+";\n"
    str += "spinorFloat "+a_re(0,1,c)+" = -"+out_re(0,c)+" - "+out_re(2,c)+";\n"
    str += "spinorFloat "+a_im(0,1,c)+" = -"+out_im(0,c)+" - "+out_im(2,c)+";\n"
    str += "spinorFloat "+a_re(0,2,c)+" =  "+out_re(1,c)+" - "+out_re(3,c)+";\n"
    str += "spinorFloat "+a_im(0,2,c)+" =  "+out_im(1,c)+" - "+out_im(3,c)+";\n"
    str += "spinorFloat "+a_re(0,3,c)+" = -"+out_re(0,c)+" + "+out_re(2,c)+";\n"
    str += "spinorFloat "+a_im(0,3,c)+" = -"+out_im(0,c)+" + "+out_im(2,c)+";\n"
    str += "\n"

    for s in range (0,4):
        str += out_re(s,c)+" = "+a_re(0,s,c)+";  "
        str += out_im(s,c)+" = "+a_im(0,s,c)+";\n"

    return block(str)+"\n\n"
# end def from_chiral_basis


def clover_mult(chi):
    str = "READ_CLOVER(CLOVERTEX, "+`chi`+")\n\n"

    for s in range (0,2):
        for c in range (0,3):
            str += "spinorFloat "+a_re(chi,s,c)+" = 0; spinorFloat "+a_im(chi,s,c)+" = 0;\n"
    str += "\n"

    for sm in range (0,2):
        for cm in range (0,3):
            for sn in range (0,2):
                for cn in range (0,3):
                    str += a_re(chi,sm,cm)+" += "+c_re(chi,sm,cm,sn,cn)+" * "+out_re(2*chi+sn,cn)+";\n"
                    if (sn != sm) or (cn != cm): 
                        str += a_re(chi,sm,cm)+" -= "+c_im(chi,sm,cm,sn,cn)+" * "+out_im(2*chi+sn,cn)+";\n"
                    #else: str += ";\n"
                    str += a_im(chi,sm,cm)+" += "+c_re(chi,sm,cm,sn,cn)+" * "+out_im(2*chi+sn,cn)+";\n"
                    if (sn != sm) or (cn != cm): 
                        str += a_im(chi,sm,cm)+" += "+c_im(chi,sm,cm,sn,cn)+" * "+out_re(2*chi+sn,cn)+";\n"
                    #else: str += ";\n"
            str += "\n"

    for s in range (0,2):
        for c in range (0,3):
            str += out_re(2*chi+s,c)+" = "+a_re(chi,s,c)+";  "
            str += out_im(2*chi+s,c)+" = "+a_im(chi,s,c)+";\n"
    str += "\n"

    return block(str)+"\n\n"
# end def clover_mult


def apply_clover():
    str = ""
    str += "#ifdef DSLASH_CLOVER\n\n"
    str += "// change to chiral basis\n"
    str += to_chiral_basis(0) + to_chiral_basis(1) + to_chiral_basis(2)
    str += "// apply first chiral block\n"
    str += clover_mult(0)
    str += "// apply second chiral block\n"
    str += clover_mult(1)
    str += "// change back from chiral basis\n"
    str += "// (note: required factor of 1/2 is included in clover term normalization)\n"
    str += from_chiral_basis(0) + from_chiral_basis(1) + from_chiral_basis(2)
    str += "#endif // DSLASH_CLOVER\n\n"

    return str
# end def clover


def twisted_rotate(x):
    str = "// apply twisted mass rotation\n"

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
                    strRe += sign(re)+out_re(s,c)
                    strIm += sign(re)+out_im(s,c)
                elif re==0:
                    strRe += sign(-im)+out_im(s,c)
                    strIm += sign(im)+out_re(s,c)
                
                # sign(x)*i*mu*gamma_5
                re = igamma5[4*h+s].real
                im = igamma5[4*h+s].imag
                if re==0 and im==0: ()
                elif im==0:
                    strRe += sign(re*x)+out_re(s,c) + "*a"
                    strIm += sign(re*x)+out_im(s,c) + "*a"
                elif re==0:
                    strRe += sign(-im*x)+out_im(s,c) + "*a"
                    strIm += sign(im*x)+out_re(s,c) + "*a"

            str += "volatile spinorFloat "+tmp_re(h,c)+" = " + strRe + ";\n"
            str += "volatile spinorFloat "+tmp_im(h,c)+" = " + strIm + ";\n"
        str += "\n"
    
    return str+"\n"


def twisted():
    str = ""
    str += twisted_rotate(+1)

    str += "#ifndef DSLASH_XPAY\n"
    str += "//scale by b = 1/(1 + a*a) \n"
    for s in range(0,4):
        for c in range(0,3):
            str += out_re(s,c) + " = b*" + tmp_re(s,c) + ";\n"
            str += out_im(s,c) + " = b*" + tmp_im(s,c) + ";\n"
    str += "#else\n"
    for s in range(0,4):
        for c in range(0,3):
            str += out_re(s,c) + " = " + tmp_re(s,c) + ";\n"
            str += out_im(s,c) + " = " + tmp_im(s,c) + ";\n"
    str += "#endif // DSLASH_XPAY\n"
    str += "\n"

    return block(str)+"\n"
# end def twisted


def xpay():
    str = ""
    str += "#ifdef DSLASH_XPAY\n\n"
    str += "READ_ACCUM(ACCUMTEX, sp_stride)\n\n"
    str += "#ifdef SPINOR_DOUBLE\n"

    for s in range(0,4):
        for c in range(0,3):
            i = 3*s+c
            if twist == False:
                str += out_re(s,c) +" = a*"+out_re(s,c)+" + accum"+nthFloat2(2*i+0)+";\n"
                str += out_im(s,c) +" = a*"+out_im(s,c)+" + accum"+nthFloat2(2*i+1)+";\n"
            else:
                str += out_re(s,c) +" = b*"+out_re(s,c)+" + accum"+nthFloat2(2*i+0)+";\n"
                str += out_im(s,c) +" = b*"+out_im(s,c)+" + accum"+nthFloat2(2*i+1)+";\n"

    str += "#else\n"

    for s in range(0,4):
        for c in range(0,3):
            i = 3*s+c
            if twist == False:
                str += out_re(s,c) +" = a*"+out_re(s,c)+" + accum"+nthFloat4(2*i+0)+";\n"
                str += out_im(s,c) +" = a*"+out_im(s,c)+" + accum"+nthFloat4(2*i+1)+";\n"
            else:
                str += out_re(s,c) +" = b*"+out_re(s,c)+" + accum"+nthFloat4(2*i+0)+";\n"
                str += out_im(s,c) +" = b*"+out_im(s,c)+" + accum"+nthFloat4(2*i+1)+";\n"

    str += "#endif // SPINOR_DOUBLE\n\n"
    str += "#endif // DSLASH_XPAY\n"

    return str
# end def xpay


def epilog():
    str = ""
    if twist:
        str += "#ifdef MULTI_GPU\n"
    else:
        str += "#if defined MULTI_GPU && (defined DSLASH_XPAY || defined DSLASH_CLOVER)\n"        
    str += (
"""
bool incomplete = false; // Have all 8 contributions been computed for this site?

switch(param.kernel_type) { // intentional fall-through
case INTERIOR_KERNEL:
  incomplete ||= commDim[0] && (x1==0 || x1==X1m1);
case EXTERIOR_KERNEL_X:
  incomplete ||= commDim[1] && (x2==0 || x2==X2m1);
case EXTERIOR_KERNEL_Y:
  incomplete ||= commDim[2] && (x3==0 || x3==X3m1);
case EXTERIOR_KERNEL_Z:
  incomplete ||= commDim[3] && (x4==0 || x4==X4m1);
}

""")    
    str += "if (!incomplete)\n"
    str += "#endif // MULTI_GPU\n"
    
    str += block( "\n" + (twisted() if twist else apply_clover()) + xpay() )
    
    str += "\n\n"
    str += "// write spinor field back to device memory\n"
    str += "WRITE_SPINOR(sp_stride);\n\n"

    str += "// undefine to prevent warning when precision is changed\n"
    str += "#undef spinorFloat\n"
    str += "#undef SHARED_STRIDE\n\n"

    str += "#undef A_re\n"
    str += "#undef A_im\n\n"

    str += "#ifdef sp_norm_idx\n"
    str += "#undef sp_norm_idx\n"
    str += "#endif\n"

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

    if clover == True:
        for m in range(0,6):
            s = m/3
            c = m%3
            str += "#undef "+c_re(0,s,c,s,c)+"\n"
            for n in range(0,6):
                sn = n/3
                cn = n%3
                for m in range(n+1,6):
                    sm = m/3
                    cm = m%3
                    str += "#undef "+c_re(0,sm,cm,sn,cn)+"\n"
                    str += "#undef "+c_im(0,sm,cm,sn,cn)+"\n"
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


def pack_face(facenum):
    str = "\n"
    str += "switch(dim) {\n"
    for dim in range(0,4):
        str += "case "+`dim`+":\n"
        proj = gen(2*dim+facenum, pack_only=True)
        proj += "\n"
        proj += "// write half spinor back to device memory\n"
        proj += "WRITE_HALF_SPINOR(face_volume, face_idx);\n"
        str += indent(block(proj)+"\n"+"break;\n")
    str += "}\n\n"
    return str
# end def pack_face


def generate_pack():
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


def generate_dslash():
    return prolog() + gen(0) + gen(1) + gen(2) + gen(3) + gen(4) + gen(5) + gen(6) + gen(7) + epilog()


# To fit 192 threads/SM (single precision) with 16K shared memory, set sharedFloats to 19 or smaller
sharedFloats = 8

twist = False
clover = True
dagger = False
print sys.argv[0] + ": generating wilson_dslash_core.h";
f = open('dslash_core/wilson_dslash_core.h', 'w')
f.write(generate_dslash())
f.close()

dagger = True
print sys.argv[0] + ": generating wilson_dslash_dagger_core.h";
f = open('dslash_core/wilson_dslash_dagger_core.h', 'w')
f.write(generate_dslash())
f.close()

twist = True
clover = False
dagger = False
print sys.argv[0] + ": generating tm_dslash_core.h";
f = open('dslash_core/tm_dslash_core.h', 'w')
f.write(generate_dslash())
f.close()

dagger = True
print sys.argv[0] + ": generating tm_dslash_dagger_core.h";
f = open('dslash_core/tm_dslash_dagger_core.h', 'w')
f.write(generate_dslash())
f.close()

sharedFloats = 0
twist = False
clover = False
dagger = False
print sys.argv[0] + ": generating wilson_pack_face_core.h";
f = open('dslash_core/wilson_pack_face_core.h', 'w')
f.write(generate_pack())
f.close()

dagger = True
print sys.argv[0] + ": generating wilson_pack_face_dagger_core.h";
f = open('dslash_core/wilson_pack_face_dagger_core.h', 'w')
f.write(generate_pack())
f.close()

#f = open('clover_core.h', 'w')
#f.write(prolog() + clover() + epilog())
#f.close()
