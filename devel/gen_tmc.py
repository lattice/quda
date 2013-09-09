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
    return "{\\\n"+indent(code)+"}\\\n"

def sign(x):
    if x==1: return "+"
    elif x==-1: return "-"
    elif x==+2: return "+2*"
    elif x==-2: return "-2*"

def nthFloat4(n):
    return `(n/4)` + "." + ["x", "y", "z", "w"][n%4]

def nthFloat2(n):
    return `(n/2)` + "." + ["x", "y"][n%2]


def c_re(b, sm, cm, sn, cn): return "c##"+`(sm+2*b)`+`cm`+"_"+`(sn+2*b)`+`cn`+"_re"
def c_im(b, sm, cm, sn, cn): return "c##"+`(sm+2*b)`+`cm`+"_"+`(sn+2*b)`+`cn`+"_im"

#invert_clover term:
def cinv_re(b, sm, cm, sn, cn): return "cinv##"+`(sm+2*b)`+`cm`+"_"+`(sn+2*b)`+`cn`+"_re"
def cinv_im(b, sm, cm, sn, cn): return "cinv##"+`(sm+2*b)`+`cm`+"_"+`(sn+2*b)`+`cn`+"_im"

def a_re(b, s, c): return "a"+`(s+2*b)`+`c`+"_re"
def a_im(b, s, c): return "a"+`(s+2*b)`+`c`+"_im"

def acc_re(s, c): return "acc"+`s`+`c`+"_re"
def acc_im(s, c): return "acc"+`s`+`c`+"_im"

def tmp_re(s, c): return "tmp"+`s`+`c`+"_re"
def tmp_im(s, c): return "tmp"+`s`+`c`+"_im"

def spinor(name, s, c, z): 
    if z==0: return name + "##" +`s`+`c`+"_re"
    else: return name + "##" +`s`+`c`+"_im"

def to_chiral_basis(v_out,v_in,c):
    str = ""
    str += "spinorFloat "+a_re(0,0,c)+" = -"+spinor(v_in,1,c,0)+" - "+spinor(v_in,3,c,0)+";\\\n"
    str += "spinorFloat "+a_im(0,0,c)+" = -"+spinor(v_in,1,c,1)+" - "+spinor(v_in,3,c,1)+";\\\n"
    str += "spinorFloat "+a_re(0,1,c)+" =  "+spinor(v_in,0,c,0)+" + "+spinor(v_in,2,c,0)+";\\\n"
    str += "spinorFloat "+a_im(0,1,c)+" =  "+spinor(v_in,0,c,1)+" + "+spinor(v_in,2,c,1)+";\\\n"
    str += "spinorFloat "+a_re(0,2,c)+" = -"+spinor(v_in,1,c,0)+" + "+spinor(v_in,3,c,0)+";\\\n"
    str += "spinorFloat "+a_im(0,2,c)+" = -"+spinor(v_in,1,c,1)+" + "+spinor(v_in,3,c,1)+";\\\n"
    str += "spinorFloat "+a_re(0,3,c)+" =  "+spinor(v_in,0,c,0)+" - "+spinor(v_in,2,c,0)+";\\\n"
    str += "spinorFloat "+a_im(0,3,c)+" =  "+spinor(v_in,0,c,1)+" - "+spinor(v_in,2,c,1)+";\\\n"
    str += "\\\n"

    for s in range (0,4):
        str += spinor(v_out,s,c,0)+" = "+a_re(0,s,c)+";  "
        str += spinor(v_out,s,c,1)+" = "+a_im(0,s,c)+";\\\n"

    return block(str)+"\\\n"
# end def to_chiral_basis

def from_chiral_basis(v_out,v_in,c): # note: factor of 1/2 is included in clover term normalization
    str = ""
    str += "spinorFloat "+a_re(0,0,c)+" =  "+spinor(v_in,1,c,0)+" + "+spinor(v_in,3,c,0)+";\\\n"
    str += "spinorFloat "+a_im(0,0,c)+" =  "+spinor(v_in,1,c,1)+" + "+spinor(v_in,3,c,1)+";\\\n"
    str += "spinorFloat "+a_re(0,1,c)+" = -"+spinor(v_in,0,c,0)+" - "+spinor(v_in,2,c,0)+";\\\n"
    str += "spinorFloat "+a_im(0,1,c)+" = -"+spinor(v_in,0,c,1)+" - "+spinor(v_in,2,c,1)+";\\\n"
    str += "spinorFloat "+a_re(0,2,c)+" =  "+spinor(v_in,1,c,0)+" - "+spinor(v_in,3,c,0)+";\\\n"
    str += "spinorFloat "+a_im(0,2,c)+" =  "+spinor(v_in,1,c,1)+" - "+spinor(v_in,3,c,1)+";\\\n"
    str += "spinorFloat "+a_re(0,3,c)+" = -"+spinor(v_in,0,c,0)+" + "+spinor(v_in,2,c,0)+";\\\n"
    str += "spinorFloat "+a_im(0,3,c)+" = -"+spinor(v_in,0,c,1)+" + "+spinor(v_in,2,c,1)+";\\\n"
    str += "\\\n"

    for s in range (0,4):
        str += spinor(v_out,s,c,0)+" = "+a_re(0,s,c)+";  "
        str += spinor(v_out,s,c,1)+" = "+a_im(0,s,c)+";\\\n"

    return block(str)+"\\\n"
# end def from_chiral_basis


def clover_mult(v_out, v_in, chi):
    str = "READ_CLOVER(TMCLOVERTEX, "+`chi`+")\\\n"

    for s in range (0,2):
        for c in range (0,3):
            str += "spinorFloat "+a_re(chi,s,c)+" = 0; spinorFloat "+a_im(chi,s,c)+" = 0;\\\n"
    str += "\\\n"

    for sm in range (0,2):
        for cm in range (0,3):
            for sn in range (0,2):
                for cn in range (0,3):
                    str += a_re(chi,sm,cm)+" += "+c_re(chi,sm,cm,sn,cn)+" * "+spinor(v_in,2*chi+sn,cn,0)+";\\\n"
                    if (sn != sm) or (cn != cm): 
                        str += a_re(chi,sm,cm)+" -= "+c_im(chi,sm,cm,sn,cn)+" * "+spinor(v_in,2*chi+sn,cn,1)+";\\\n"
                    #else: str += ";\n"
                    str += a_im(chi,sm,cm)+" += "+c_re(chi,sm,cm,sn,cn)+" * "+spinor(v_in,2*chi+sn,cn,1)+";\\\n"
                    if (sn != sm) or (cn != cm): 
                        str += a_im(chi,sm,cm)+" += "+c_im(chi,sm,cm,sn,cn)+" * "+spinor(v_in,2*chi+sn,cn,0)+";\\\n"
                    #else: str += ";\n"
            str += "\\\n"
    str += "/*apply  i*(2*kappa*mu=mubar)*gamma5*/\\\n"
    mubar_gamma5_re = (" + mubar* " if not chi else " - mubar* ")
    mubar_gamma5_im = (" - mubar* " if not chi else " + mubar* ")
    for s in range (0,2):
        for c in range (0,3):
            str += spinor(v_out,2*chi+s,c,0)+" = "+a_re(chi,s,c)+ mubar_gamma5_re +spinor(v_out,2*chi+s,c,0)+";  "
            str += spinor(v_out,2*chi+s,c,1)+" = "+a_im(chi,s,c)+ mubar_gamma5_im +spinor(v_out,2*chi+s,c,0)+";\\\n"
    str += "\\\n"

    return block(str)+"\\\n"
# end def clover_mult

def inv_clover_mult(v_out, v_in, chi):
    str = "READ_CLOVER(TMCLOVERTEX, "+`chi`+")\\\n"

    for s in range (0,2):
        for c in range (0,3):
            str += "spinorFloat "+a_re(chi,s,c)+" = 0; spinorFloat "+a_im(chi,s,c)+" = 0;\\\n"
    str += "\\\n"

    for sm in range (0,2):
        for cm in range (0,3):
            for sn in range (0,2):
                for cn in range (0,3):
                    str += a_re(chi,sm,cm)+" += "+c_re(chi,sm,cm,sn,cn)+" * "+spinor(v_in,2*chi+sn,cn,0)+";\\\n"
                    if (sn != sm) or (cn != cm): 
                        str += a_re(chi,sm,cm)+" -= "+c_im(chi,sm,cm,sn,cn)+" * "+spinor(v_in,2*chi+sn,cn,1)+";\\\n"
                    #else: str += ";\n"
                    str += a_im(chi,sm,cm)+" += "+c_re(chi,sm,cm,sn,cn)+" * "+spinor(v_in,2*chi+sn,cn,1)+";\\\n"
                    if (sn != sm) or (cn != cm): 
                        str += a_im(chi,sm,cm)+" += "+c_im(chi,sm,cm,sn,cn)+" * "+spinor(v_in,2*chi+sn,cn,0)+";\\\n"
                    #else: str += ";\n"
            str += "\\\n"
    str += "/*apply  i*(2*kappa*mu=mubar)*gamma5*/\\\n"
    mubar_gamma5_re = (" + mubar* " if not chi else " - mubar* ")
    mubar_gamma5_im = (" - mubar* " if not chi else " + mubar* ")
    for s in range (0,2):
        for c in range (0,3):
            str += spinor(v_out,2*chi+s,c,0)+" = "+a_re(chi,s,c)+ mubar_gamma5_re +spinor(v_out,2*chi+s,c,0)+";  "
            str += spinor(v_out,2*chi+s,c,1)+" = "+a_im(chi,s,c)+ mubar_gamma5_im +spinor(v_out,2*chi+s,c,0)+";\\\n"
    str += "\\\n"
    str += "/*Apply inverse clover*/\\\n"
    str += "READ_CLOVER(TM_INV_CLOVERTEX, "+`chi`+")\\\n"

    for s in range (0,2):
        for c in range (0,3):
            str += a_re(chi,s,c)+" = 0; "+a_im(chi,s,c)+" = 0;\\\n"
    str += "\\\n"

    for sm in range (0,2):
        for cm in range (0,3):
            for sn in range (0,2):
                for cn in range (0,3):
                    str += a_re(chi,sm,cm)+" += "+cinv_re(chi,sm,cm,sn,cn)+" * "+spinor(v_in,2*chi+sn,cn,0)+";\\\n"
                    if (sn != sm) or (cn != cm): 
                        str += a_re(chi,sm,cm)+" -= "+cinv_im(chi,sm,cm,sn,cn)+" * "+spinor(v_in,2*chi+sn,cn,1)+";\\\n"
                    #else: str += ";\n"
                    str += a_im(chi,sm,cm)+" += "+cinv_re(chi,sm,cm,sn,cn)+" * "+spinor(v_in,2*chi+sn,cn,1)+";\\\n"
                    if (sn != sm) or (cn != cm): 
                        str += a_im(chi,sm,cm)+" += "+cinv_im(chi,sm,cm,sn,cn)+" * "+spinor(v_in,2*chi+sn,cn,0)+";\\\n"
                    #else: str += ";\n"
            str += "\\\n"
    str += "/*store  the result*/\\\n"
    for s in range (0,2):
        for c in range (0,3):
            str += spinor(v_out,2*chi+s,c,0)+" = "+a_re(chi,s,c)+";  "
            str += spinor(v_out,2*chi+s,c,1)+" = "+a_im(chi,s,c)+";\\\n"
    str += "\\\n"

    return block(str)+"\\\n"
# end def inv_clover_mult


def make_title(title_str):
    str = ""
    str += "#define APPLY_"+title_str+"\\\n"
    return str


def apply_clover(v_out,v_in):
    str = "\\\n"
    str += "/* change to chiral basis*/\\\n"
    str += to_chiral_basis(v_out,v_in,0) + to_chiral_basis(v_out,v_in,1) + to_chiral_basis(v_out,v_in,2)
    str += "/* apply first chiral block*/\\\n"
    str += clover_mult(v_out,v_out,0)
    str += "/* apply second chiral block*/\\\n"
    str += clover_mult(v_out,v_out,1)
    str += "/* change back from chiral basis*/\\\n"
    str += "/* (note: required factor of 1/2 is included in clover term normalization)*/\\\n"
    str += from_chiral_basis(v_out,v_out,0) + from_chiral_basis(v_out,v_out,1) + from_chiral_basis(v_out,v_out,2)
    str += "\n"
    return str
# end def clover

def apply_inv_clover(v_out,v_in):
    str = "\\\n"
    str += "/* change to chiral basis*/\\\n"
    str += to_chiral_basis(v_out,v_in,0) + to_chiral_basis(v_out,v_in,1) + to_chiral_basis(v_out,v_in,2)
    str += "/* apply first chiral block*/\\\n"
    str += inv_clover_mult(v_out,v_out,0)
    str += "/* apply second chiral block*/\\\n"
    str += inv_clover_mult(v_out,v_out,1)
    str += "/* change back from chiral basis*/\\\n"
    str += "/* (note: required factor of 1/2 is included in clover term normalization)*/\\\n"
    str += from_chiral_basis(v_out,v_out,0) + from_chiral_basis(v_out,v_out,1) + from_chiral_basis(v_out,v_out,2)
    str += "\n"
    return str
# end def clover


def generate_tmclover_file():
    filename = '/dslash_core/tmc_core.h'
    print sys.argv[0] + ": generating " + filename;
    f = open(filename, 'w')
    f.write(make_title("CLOVER_TWIST(c, a, reg)") + apply_clover('reg', 'reg'))
    f.write(make_title("CLOVER_TWIST_INV(c, cinv, a, reg)") + apply_inv_clover('reg', 'reg'))
    f.close()

generate_tmclover_file()


