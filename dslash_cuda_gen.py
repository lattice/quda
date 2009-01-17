
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
#gamma4 = complexify([
#    0, 0, 1, 0,
#    0, 0, 0, 1,
#    1, 0, 0, 0,
#    0, 1, 0, 0
#])


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
    gminus(id,gamma4), gplus(id,gamma4)
]


### code generation  ########################################################################

### parameters
sharedFloats = 19
dagger = False

def block(code):
    lines = ''.join(["    "+line+"\n" for line in code.splitlines()])
    return "{\n"+lines+"}\n"

def sign(x):
    if x==1: return "+"
    elif x==-1: return "-"
    elif x==+2: return "+2*"
    elif x==-2: return "-2*"

def nthFloat4(n):
    return `(n/4)` + "." + ["x", "y", "z", "w"][n%4]

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



def prolog():
    str = []
    str.append("// *** CUDA DSLASH ***\n\n" if not dagger else "// *** CUDA DSLASH DAGGER ***\n\n")
    str.append("#define SHARED_FLOATS_PER_THREAD "+`sharedFloats`+"\n")
    str.append("#define SHARED_BYTES (BLOCK_DIM*SHARED_FLOATS_PER_THREAD*sizeof(float))\n\n")
    
    for s in range(0,4):
        for c in range(0,3):
            i = 3*s+c
            str.append("#define "+in_re(s,c)+" I"+nthFloat4(2*i+0)+"\n")
            str.append("#define "+in_im(s,c)+" I"+nthFloat4(2*i+1)+"\n")
    str.append("\n")
    for m in range(0,3):
        for n in range(0,3):
            i = 3*m+n
            str.append("#define "+g_re(0,m,n)+" G"+nthFloat4(2*i+0)+"\n")
            str.append("#define "+g_im(0,m,n)+" G"+nthFloat4(2*i+1)+"\n")
    str.append("\n")
    for m in range(0,3):
        for n in range(0,3):
            i = 3*m+n
            str.append("#define "+g_re(1,m,n)+" (+"+g_re(0,n,m)+")\n")
            str.append("#define "+g_im(1,m,n)+" (-"+g_im(0,n,m)+")\n")
    str.append("\n")

    # last two components of the 5th float4 used for temp storage
    str.append("#define A_re G"+nthFloat4(18)+"\n")
    str.append("#define A_im G"+nthFloat4(19)+"\n")    

    str.append("\n")
    for s in range(0,4):
        for c in range(0,3):
            i = 3*s+c
            if 2*i < sharedFloats:
                str.append("#define "+out_re(s,c)+" s["+`(2*i+0)`+"]\n")
            else:
                str.append("volatile float "+out_re(s,c)+";\n")
            if 2*i+1 < sharedFloats:
                str.append("#define "+out_im(s,c)+" s["+`(2*i+1)`+"]\n")
            else:
                str.append("volatile float "+out_im(s,c)+";\n")
    str.append("\n")
    
    str.append(
"""

#include "read_gauge.h"
#include "io_spinor.h"

int sid = BLOCK_DIM*blockIdx.x + threadIdx.x;
int boundaryCrossings = sid/L1h + sid/(L2*L1h) + sid/(L3*L2*L1h);
int X = 2*sid + (boundaryCrossings + oddBit) % 2;
int x4 = X/(L3*L2*L1);
int x3 = (X/(L2*L1)) % L3;
int x2 = (X/L1) % L2;
int x1 = X % L1;

""")
    
    str.append("extern __shared__ float s_data[];\n")
    str.append("volatile float *s = s_data+SHARED_FLOATS_PER_THREAD*threadIdx.x;\n\n")
    
    for s in range(0,4):
        for c in range(0,3):
            str.append(out_re(s,c) + " = " + out_im(s,c)+" = 0;\n")
    str.append("\n")
    
    return ''.join(str)
# end def prolog



def epilog():
    str = []
    str.append(
"""
#ifdef DSLASH_XPAY
    float4 accum0 = tex1Dfetch(accumTex, sid + 0*Nh);
    float4 accum1 = tex1Dfetch(accumTex, sid + 1*Nh);
    float4 accum2 = tex1Dfetch(accumTex, sid + 2*Nh);
    float4 accum3 = tex1Dfetch(accumTex, sid + 3*Nh);
    float4 accum4 = tex1Dfetch(accumTex, sid + 4*Nh);
    float4 accum5 = tex1Dfetch(accumTex, sid + 5*Nh);
""")
    
    for s in range(0,4):
        for c in range(0,3):
            i = 3*s+c
            str.append("    "+out_re(s,c) +" = a*"+out_re(s,c)+" + accum"+nthFloat4(2*i+0)+";\n")
            str.append("    "+out_im(s,c) +" = a*"+out_im(s,c)+" + accum"+nthFloat4(2*i+1)+";\n")
    str.append("#endif\n\n")
    
    str.append(
"""
    // write spinor field back to device memory
    WRITE_SPINOR();

""")
    return ''.join(str)
# end def prolog



def gen(dir):
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
    
    str = []
    
    projName = "P"+`dir/2`+["-","+"][projIdx%2]
    str.append("// Projector "+projName+"\n")
    for l in projStr.splitlines():
        str.append("// "+l+"\n")
    str.append("\n")
    
    if dir == 0: str.append("int sp_idx = ((x1==L1-1) ? X-(L1-1) : X+1) / 2;\n")
    if dir == 1: str.append("int sp_idx = ((x1==0)    ? X+(L1-1) : X-1) / 2;\n")
    if dir == 2: str.append("int sp_idx = ((x2==L2-1) ? X-(L2-1)*L1 : X+L1) / 2;\n")
    if dir == 3: str.append("int sp_idx = ((x2==0)    ? X+(L2-1)*L1 : X-L1) / 2;\n")
    if dir == 4: str.append("int sp_idx = ((x3==L3-1) ? X-(L3-1)*L2*L1 : X+L2*L1) / 2;\n")
    if dir == 5: str.append("int sp_idx = ((x3==0)    ? X+(L3-1)*L2*L1 : X-L2*L1) / 2;\n")
    if dir == 6: str.append("int sp_idx = ((x4==L4-1) ? X-(L4-1)*L3*L2*L1 : X+L3*L2*L1) / 2;\n")
    if dir == 7: str.append("int sp_idx = ((x4==0)    ? X+(L4-1)*L3*L2*L1 : X-L3*L2*L1) / 2;\n")
    
    ga_idx = "sid" if dir % 2 == 0 else "sp_idx"
    str.append("int ga_idx = "+ga_idx+";\n\n")
    
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

    load_spinor = []
    load_spinor.append("// read spinor from device memory\n")
    if row_cnt[0] == 0:
        load_spinor.append("READ_SPINOR_DOWN(spinorTex);\n\n")
    elif row_cnt[2] == 0:
        load_spinor.append("READ_SPINOR_UP(spinorTex);\n\n")
    else:
        load_spinor.append("READ_SPINOR(spinorTex);\n\n")

    load_gauge = []
    load_gauge.append("// read gauge matrix from device memory\n")
    load_gauge.append("READ_GAUGE_MATRIX(GAUGE"+`dir%2`+"TEX, "+`dir`+");\n\n")

    project = []
    project.append("// project spinor into half spinors\n")
    for h in range(0, 2):
        for c in range(0, 3):
            strRe = []
            strIm = []
            for s in range(0, 4):
                re = proj(h,s).real
                im = proj(h,s).imag
                if re==0 and im==0: ()
                elif im==0:
                    strRe.append(sign(re)+in_re(s,c))
                    strIm.append(sign(re)+in_im(s,c))
                elif re==0:
                    strRe.append(sign(-im)+in_im(s,c))
                    strIm.append(sign(im)+in_re(s,c))
            if row_cnt[0] == 0: #projector defined on lower half only
                for s in range(0, 4):
                    re = proj(h+2,s).real
                    im = proj(h+2,s).imag
                    if re==0 and im==0: ()
                    elif im==0:
                        strRe.append(sign(re)+in_re(s,c))
                        strIm.append(sign(re)+in_im(s,c))
                    elif re==0:
                        strRe.append(sign(-im)+in_im(s,c))
                        strIm.append(sign(im)+in_re(s,c))
                
            project.append("float "+h1_re(h,c)+ " = "+''.join(strRe)+";\n")
            project.append("float "+h1_im(h,c)+ " = "+''.join(strIm)+";\n")
        project.append("\n")
    
    ident = []
    ident.append("// identity gauge matrix\n")
    for m in range(0,3):
        for h in range(0,2):
            ident.append("float "+h2_re(h,m)+" = " + h1_re(h,m) + "; ")
            ident.append("float "+h2_im(h,m)+" = " + h1_im(h,m) + ";\n")
    ident.append("\n")
    
    mult = []
    for m in range(0,3):
        mult.append("// multiply row "+`m`+"\n")
        for h in range(0,2):
            re = ["float "+h2_re(h,m)+" ="]
            im = ["float "+h2_im(h,m)+" ="]
            for c in range(0,3):
                re.append(" + ("+g_re(dir,m,c)+" * "+h1_re(h,c)+" - "+g_im(dir,m,c)+" * "+h1_im(h,c)+")")
                im.append(" + ("+g_re(dir,m,c)+" * "+h1_im(h,c)+" + "+g_im(dir,m,c)+" * "+h1_re(h,c)+")")
            mult.append(''.join(re)+";\n")
            mult.append(''.join(im)+";\n")
        mult.append("\n")
    
    reconstruct = []
    for m in range(0,3):

        for h in range(0,2):
            h_out = h
            if row_cnt[0] == 0: # projector defined on lower half only
                h_out = h+2
            reconstruct.append(out_re(h_out, m) + " += " + h2_re(h,m) + ";\n")
            reconstruct.append(out_im(h_out, m) + " += " + h2_im(h,m) + ";\n")
    
        for s in range(2,4):
            (h,c) = row(s)
            re = c.real
            im = c.imag
            if im == 0 and re == 0:
                ()
            elif im == 0:
                reconstruct.append(out_re(s, m) + " " + sign(re) + "= " + h2_re(h,m) + ";\n")
                reconstruct.append(out_im(s, m) + " " + sign(re) + "= " + h2_im(h,m) + ";\n")
            elif re == 0:
                reconstruct.append(out_re(s, m) + " " + sign(-im) + "= " + h2_im(h,m) + ";\n")
                reconstruct.append(out_im(s, m) + " " + sign(+im) + "= " + h2_re(h,m) + ";\n")
        
        reconstruct.append("\n")
        
    if dir >= 6:
        str.append("if (gauge_fixed && ga_idx < (L4-1)*L1h*L2*L3) ")
        str.append(block(''.join(load_spinor) + ''.join(project) + ''.join(ident) + ''.join(reconstruct)))
        str.append("else ")
        str.append(block(''.join(load_gauge) + ''.join(load_spinor) + ''.join(project) + ''.join(mult) + ''.join(reconstruct)))
    else:
        str.append(''.join(load_gauge) + ''.join(load_spinor) + ''.join(project) + ''.join(mult) + ''.join(reconstruct))
    
    return block(''.join(str))+"\n"
# end def gen



def generate():
    return prolog() + gen(0) + gen(1) + gen(2) + gen(3) + gen(4) + gen(5) + gen(6) + gen(7) + epilog()

dagger = False
#dagger = True

print generate()
