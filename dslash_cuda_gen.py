
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
    0, 0, 0, -1,
    0, 0, 1,  0,
    0, 1, 0,  0,
    -1, 0, 0,  0
])

gamma3 = complexify([
    0, 0, 1j,  0,
    0, 0, 0, -1j,
    -1j, 0, 0,  0,
    0, 1j, 0,  0
])

gamma4 = complexify([
    0, 0, 1, 0,
    0, 0, 0, 1,
    1, 0, 0, 0,
    0, 1, 0, 0
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
// Performs the complex conjugated accumulation: a += b* c*
#define ACC_CONJ_PROD(a, b, c) \\
    a##_re += b##_re * c##_re - b##_im * c##_im, \\
    a##_im -= b##_re * c##_im + b##_im * c##_re

#define READ_GAUGE_MATRIX(gauge, dir) \\
    float4 G0 = tex1Dfetch((gauge), ga_idx + ((dir/2)*3+0)*Nh); \\
    float4 G1 = tex1Dfetch((gauge), ga_idx + ((dir/2)*3+1)*Nh); \\
    float4 G2 = tex1Dfetch((gauge), ga_idx + ((dir/2)*3+2)*Nh); \\
    float4 G3 = make_float4(0,0,0,0); \\
    float4 G4 = make_float4(0,0,0,0); \\
    ACC_CONJ_PROD(g20, +g01, +g12); \\
    ACC_CONJ_PROD(g20, -g02, +g11); \\
    ACC_CONJ_PROD(g21, +g02, +g10); \\
    ACC_CONJ_PROD(g21, -g00, +g12); \\
    ACC_CONJ_PROD(g22, +g00, +g11); \\
    ACC_CONJ_PROD(g22, -g01, +g10); \\
    float u0 = (dir < 6 ? SPATIAL_SCALING : (ga_idx >= (L4-1)*L1h*L2*L3 ? TIME_SYMMETRY : 1)); \\
    G3.x*=u0; G3.y*=u0; G3.z*=u0; G3.w*=u0; G4.x*=u0; G4.y*=u0;

#define READ_SPINOR(spinor) \\
    float4 I0 = tex1Dfetch((spinor), sp_idx + 0*Nh); \\
    float4 I1 = tex1Dfetch((spinor), sp_idx + 1*Nh); \\
    float4 I2 = tex1Dfetch((spinor), sp_idx + 2*Nh); \\
    float4 I3 = tex1Dfetch((spinor), sp_idx + 3*Nh); \\
    float4 I4 = tex1Dfetch((spinor), sp_idx + 4*Nh); \\
    float4 I5 = tex1Dfetch((spinor), sp_idx + 5*Nh);

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
#ifdef WRITE_FLOAT4
// this code exhibits a hardware bug in our C870 card
g_out[0*Nh+sid] = make_float4(o00_re, o00_im, o01_re, o01_im);
g_out[1*Nh+sid] = make_float4(o02_re, o02_im, o10_re, o10_im);
g_out[2*Nh+sid] = make_float4(o11_re, o11_im, o12_re, o12_im);
g_out[3*Nh+sid] = make_float4(o20_re, o20_im, o21_re, o21_im);
g_out[4*Nh+sid] = make_float4(o22_re, o22_im, o30_re, o30_im);
g_out[5*Nh+sid] = make_float4(o31_re, o31_im, o32_re, o32_im);
#endif

#ifdef WRITE_FLOAT1_SMEM
int t = threadIdx.x;
int B = BLOCK_DIM;
int b = blockIdx.x;
int f = SHARED_FLOATS_PER_THREAD;
__syncthreads();
for (int i = 0; i < 6; i++) // spinor indices
    for (int c = 0; c < 4; c++) // components of float4
        ((float*)g_out)[i*(Nh*4) + b*(B*4) + c*(B) + t] = s_data[(c*B/4 + t/4)*(f) + i*(4) + t%4];
#endif

#ifdef WRITE_FLOAT1_STAGGERED
// the alternative to writing float4's directly: almost as fast, a lot more confusing
int t = threadIdx.x;
int B = BLOCK_DIM;
int b = blockIdx.x;
int f = SHARED_FLOATS_PER_THREAD;
__syncthreads();
for (int i = 0; i < 4; i++) // spinor indices
    for (int c = 0; c < 4; c++) // components of float4
        ((float*)g_out)[i*(Nh*4) + b*(B*4) + c*(B) + t] = s_data[(c*B/4 + t/4)*(f) + i*(4) + t%4];
__syncthreads();
s[0] = o22_re;
s[1] = o22_im;
s[2] = o30_re;
s[3] = o30_im;
s[4] = o31_re;
s[5] = o31_im;
s[6] = o32_re;
s[7] = o32_im;
__syncthreads();
for (int i = 0; i < 2; i++)
    for (int c = 0; c < 4; c++)
        ((float*)g_out)[(i+4)*(Nh*4) + b*(B*4) + c*(B) + t] = s_data[(c*B/4 + t/4)*(f) + i*(4) + t%4];
#endif

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
    
    project = []
    project.append("// read spinor from device memory\n")
    project.append("READ_SPINOR(spinorTex);\n\n")
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
    mult.append("// read gauge matrix from device memory\n")
    mult.append("READ_GAUGE_MATRIX(gauge"+`dir%2`+"Tex, "+`dir`+");\n\n")
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
        reconstruct.append(out_re(0, m) + " += " + h2_re(0,m) + ";\n")
        reconstruct.append(out_im(0, m) + " += " + h2_im(0,m) + ";\n")
        reconstruct.append(out_re(1, m) + " += " + h2_re(1,m) + ";\n")
        reconstruct.append(out_im(1, m) + " += " + h2_im(1,m) + ";\n")
    
        for s in range(2,4):
            (h,c) = row(s)
            re = c.real
            im = c.imag
            if im == 0:
                reconstruct.append(out_re(s, m) + " " + sign(re) + "= " + h2_re(h,m) + ";\n")
                reconstruct.append(out_im(s, m) + " " + sign(re) + "= " + h2_im(h,m) + ";\n")
            elif re == 0:
                reconstruct.append(out_re(s, m) + " " + sign(-im) + "= " + h2_im(h,m) + ";\n")
                reconstruct.append(out_im(s, m) + " " + sign(+im) + "= " + h2_re(h,m) + ";\n")
        
        reconstruct.append("\n")
        
    if dir >= 6:
        str.append("if (GAUGE_FIXED && ga_idx >= L1h*L2*L3) ")
        str.append(block(''.join(project) + ''.join(ident) + ''.join(reconstruct)))
        str.append("else ")
        str.append(block(''.join(project) + ''.join(mult) + ''.join(reconstruct)))
    else:
        str.append(''.join(project) + ''.join(mult) + ''.join(reconstruct))
    
    return block(''.join(str))+"\n"
# end def gen



def generate():
    return prolog() + gen(0) + gen(1) + gen(2) + gen(3) + gen(4) + gen(5) + gen(6) + gen(7) + epilog()

# dagger = False
dagger = True
print generate()
