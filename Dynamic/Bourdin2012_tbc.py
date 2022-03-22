#Bourdin 2012
#traction bc

from dolfin import *
import numpy as np
import numpy.linalg as la
from numpy.random import rand
import h5py
import matplotlib.pyplot as plt
import os

#Define customer solver
class Problem(NonlinearProblem):
    def __init__(self, J, F, bcs):
        self.bilinear_form = J
        self.linear_form = F
        self.bcs = bcs
        NonlinearProblem.__init__(self)

    def F(self, b, x):
        assemble(self.linear_form, tensor=b)
        for bc in self.bcs:
            bc.apply(b, x)

    def J(self, A, x):
        assemble(self.bilinear_form, tensor=A)
        for bc in self.bcs:
            bc.apply(A)


class CustomSolver(NewtonSolver):
    def __init__(self):
        NewtonSolver.__init__(self, mesh.mpi_comm(),
                              PETScKrylovSolver(), PETScFactory.instance())

    def solver_setup(self, A, P, problem, iteration):
        self.linear_solver().set_operator(A)

        PETScOptions.set("ksp_type", "gmres")
        PETScOptions.set("ksp_monitor")
        PETScOptions.set("pc_type", "hypre")
        PETScOptions.set("ksp_initial_guess_nonzero", "true")
        
        self.linear_solver().set_from_options()
        
#Define function to update results
#Disp
def update(u_new,u_pre,v_pre,a_pre,beta,gamma,dt):
    #Get vectors
    u_new_vec, u_pre_vec = u_new.vector(), u_pre.vector()
    v_pre_vec, a_pre_vec = v_pre.vector(), a_pre.vector()
    
    #Update acceleration and velocity
    a_new_vec = (1.0/(2.0*beta))*( (u_new_vec - u_pre_vec - v_pre_vec * dt) / (0.5 * dt * dt) \
              - (1.0 - 2.0 * beta ) * a_pre_vec )
    v_new_vec = dt * ( (1.0-gamma) * a_pre_vec + gamma * a_new_vec ) + v_pre_vec
    
    #Update t_n <- t_n+1
    v_pre.vector()[:], a_pre.vector()[:] = v_new_vec, a_new_vec #through values
    u_pre.vector()[:] = u_new.vector()                          #through function
#PF
#def update(p_new,p_pre,pv_pre,pa_pre,beta,gamma,dt):
#    #Get vectors
#    p_new_vec, p_pre_vec = p_new.vector(), p_pre.vector()
#    pv_pre_vec, pa_pre_vec = pv_pre.vector(), pa_pre.vector()
#    
#    #Update acceleration and velocity
#    pa_new_vec = (1.0/(2.0*beta))*( (p_new_vec - p_pre_vec - pv_pre_vec * dt) / (0.5 * dt * dt) \
#              - (1.0 - 2.0 * beta ) * pa_pre_vec )
#    pv_new_vec = dt * ( (1.0-gamma) * pa_pre_vec + gamma * pa_new_vec ) + pv_pre_vec
#    
#    #Update t_n <- t_n+1
#    pv_pre.vector()[:], pa_pre.vector()[:] = pv_new_vec, pa_new_vec 
#    p_pre.vector()[:] = p_new.vector()                          
        
#Import mesh
mesh = Mesh('PFMdyn_um.xml') #_um: unformly mesh #_rc: real crack

#Define Function Space
#PF
V = FunctionSpace(mesh,'CG',1)
#Disp 
W = VectorFunctionSpace(mesh,'CG',1)
#History var
X = FunctionSpace(mesh,'CG',1) 
EPS = TensorFunctionSpace(mesh,'CG',1)

#Define Test Function
#Disp
u_new = Function(W)
v = TestFunction(W)
#PF
p_new = Function(V)
q = TestFunction(V)
#History variable
H_pre = Function(V)

#Space dim
dim = 2

#Define material parameters
Gc     = 3                                        #N/m           [Fracture toughness    ]
lo     = 2.5e-4                                   #m             [Length scale          ] 
rho_0  = 2450                                     #kg/m^-3       [Density               ]
Emodu  = 3.2e10                                   #N/m^2         [Young's Modulus       ]
nu     = 0.2                                      #Dimensionless [Possion's Ratio       ]
Lambda = Constant(Emodu*nu/((1+nu)*(1-2*nu)))     #N/m^2         [1st Lame Constant     ]
Mu     = Constant(Emodu   /(2*(1+nu)))            #N/m^2         [2nd Lame Constant     ]
K_para = Constant(Emodu   /(3*(1-2*nu)))          #N/m^2         [Bulk Modulus          ]
k = 1e-8                                         #Dimensionless  [Stress Parameter      ]
c_para = ((K_para+4*Mu/3)/rho_0)**0.5             #m/s           [Speed of pressure wave]      
                                                  #N/m^2         [Traction force        ]
C_M    = 1                                        #Dimensionless [Paramter for M        ]
traction = Expression(('A','B'),A=0.0,B=0.0,degree=2)

#def M_para(c_para,H):
#    return 1 * c_para / ( 2 * ( 4 * 3 * 2.5e-4 * H + 3 ** 2 ) ) #Hardcode

#Define parameters for generalized-alpha method
#parameters
rho_inf = 0.5
alpha_f = 1 / ( rho_inf + 1 )
alpha_m = ( 2 - rho_inf ) / ( rho_inf + 1 )
beta    = 0.25 * ( 1 + alpha_m - alpha_f ) ** 2
gamma   = 0.5  + alpha_m - alpha_f
#time/inc
dt = 1e-7
t, T = 0.0, 1e-4 #See (Kamensky, 2018) 
nt  = 0 

#Fields from previous time step (displacement, velocity, acceleration)
#Disp
u_pre,v_pre,a_pre = Function(W), Function(W), Function(W) 
#PF
#p_pre,pv_pre,pa_pre = Function(V), Function(V), Function(V)
p_pre = Function(V)
#Set initial values of p_pre be '1'
p_init = Constant(1.0)
p_pre.interpolate(p_init)

#Velocity and acceleration at t_n+1
#Disp
v_1 = (gamma/(beta*dt))*(u_new - u_pre) - (gamma/beta - 1.0)*v_pre - dt * (gamma/(2.0*beta) \
      - 1.0 )*a_pre
a_1 = (1.0/(beta*dt**2))*(u_new - u_pre - dt * v_pre ) - (1.0 / (2.0*beta) - 1.0 )*a_pre
#PF
#pv_1 = (gamma/(beta*dt))*(p_new - p_pre) - (gamma/beta - 1.0)*pv_pre - dt * (gamma/(2.0*beta) \
#      - 1.0 )*pa_pre
#pa_1 = (1.0/(beta*dt**2))*(p_new - p_pre - dt * pv_pre ) - (1.0 / (2.0*beta) - 1.0 )*pa_pre

#Velocity and acceleration at t_n+1-alpha
#Disp
a_new_alpha = a_pre + alpha_m * ( a_1   - a_pre )
v_new_alpha = v_pre + alpha_f * ( v_1   - v_pre )
u_new_alpha = u_pre + alpha_f * ( u_new - u_pre )
#PF
#pa_new_alpha = pa_pre + alpha_m * ( pa_1   - pa_pre )
#pv_new_alpha = pv_pre + alpha_f * ( pv_1   - pv_pre )
#p_new_alpha = p_pre + alpha_f * ( p_new - p_pre )

#Define phase field parameters
def g(phi):
    return ( phi ) ** 2 #Note: Follow (Bourdin,2012), c = 1 undamaged, c = 0 damaged

#Define ramp function
def ramp_p(x):
    return 0.5 * ( x + abs(x) )
def ramp_m(x):
    return 0.5 * ( x - abs(x) )

#Define constitutive functions
def epsilon(u):
    strain = sym(grad(u))
    return strain 

#Spectral decomposition
def get_eig(u):
    eps = epsilon(u)
    eigv1 = eps[0, 0]/2 + eps[1, 1]/2 - (eps[0, 0]**2 - 2*eps[0, 0]*eps[1, 1] + 4*eps[0, 1]*eps[1, 0] + eps[1, 1]**2)**0.5/2
    eigv2 = eps[0, 0]/2 + eps[1, 1]/2 + (eps[0, 0]**2 - 2*eps[0, 0]*eps[1, 1] + 4*eps[0, 1]*eps[1, 0] + eps[1, 1]**2)**0.5/2
    vec1  = ( eigv1 - eps[1,1] ) 
    vec2  = ( eigv2 - eps[1,1] ) 
    vecn1 = ( vec1**2 + eps[0,1]**2 ) ** 0.5
    vecn2 = ( vec2**2 + eps[0,1]**2 ) ** 0.5
    vec1_ = as_vector([vec1/vecn1,eps[0,1]/vecn1])
    vec2_ = as_vector([vec2/vecn2,eps[0,1]/vecn2])
    return eigv1,eigv2,vec1_,vec2_
    
#Compute positive/negative epsilon 
def eps_split(eigv1,eigv2,vec1_,vec2_):
    eigp = ramp_p(eigv1)*outer(vec1_,vec1_) + ramp_p(eigv2)*outer(vec2_,vec2_)
    eigm = ramp_m(eigv1)*outer(vec1_,vec1_) + ramp_m(eigv2)*outer(vec2_,vec2_)
    return eigp,eigm

#Define stress                           
def sigma(u,p):
    eigv1,eigv2,vec1_,vec2_ = get_eig(u)
    eps_pls,eps_mis = eps_split(eigv1,eigv2,vec1_,vec2_)
    eps_pure = epsilon(u)
    eps_tr_pls = ramp_p(tr(eps_pure))
    eps_tr_mis = ramp_m(tr(eps_pure))
    sig_pls = Lambda*eps_tr_pls*Identity(2) + 2*Mu*eps_pls
    sig_mis = Lambda*eps_tr_mis*Identity(2) + 2*Mu*eps_mis
    sig_final = ((1-k)*g(p)+k)*sig_pls + sig_mis
    return sig_final

#Compute postive strain energy 
def psi_pls(u):
    eigv1,eigv2,vec1_,vec2_ = get_eig(u)
    eps_pls,eps_mis = eps_split(eigv1,eigv2,vec1_,vec2_)
    eps_pure = epsilon(u)
    eps_tr_pls = ramp_p(tr(eps_pure))
    sE_pls = 0.5* Lambda * eps_tr_pls * eps_tr_pls + Mu * tr( eps_pls*eps_pls )        
    return sE_pls

#Compute history variable H
def H(u_new,H_pre):
    return conditional( lt( psi_pls(u_new),H_pre ), H_pre, psi_pls(u_new) )

#Define Boundary Condition Subdomain
class top(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1],0.02,DOLFIN_EPS)

class bot(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1],-0.02,DOLFIN_EPS)   

#top = CompiledSubDomain("near(x[1],  0.02) && on_boundary")
#bot = CompiledSubDomain("near(x[1], -0.02) && on_boundary")

#bctop = DirichletBC(W.sub(1),Constant(6.25e-7),top)
#bcbot = DirichletBC(W.sub(1),Constant(0.0),bot)

def crack(x):
    return abs(x[1]) < 1e-4 and x[0] <= 0.0

bc_u   = []
bc_phi = [DirichletBC(V,Constant(0.0),crack)]
#bc_phi = [] #_rc

# Create mesh function over the cell facets
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
force_top = top()
force_bot = bot()
force_top.mark(boundaries,3)
force_bot.mark(boundaries,4)
n = FacetNormal(mesh)

# Define measure for boundary condition integral
dss = ds(subdomain_data=boundaries)

#Random initial guess
num = u_new.vector().get_local().size
u_new.vector()[:] = np.random.random(num)*DOLFIN_EPS

#Define Residual 
R_du = (inner(grad(v),sigma(u_new_alpha,p_pre))      ) * dx  
R_du = R_du  +  inner(     v ,   rho_0* a_new_alpha  ) * dx                                          
R_du = R_du  -  inner(     v ,   traction            )* dss(3)                                    
R_du = R_du  -  inner(     v ,  -traction            )* dss(4)                                      

#R_dp = inner(4*lo * p_new_alpha * H(u_new,H_pre) / Gc + p_new_alpha ,q) * dx                  
#R_dp = R_dp  +  inner( 4*lo**2 * grad(p_new_alpha) , grad(q)) * dx                                   
#R_dp = R_dp  -  inner( 1 , q ) * dx                                                                
#R_dp = R_dp  +  inner( 2*lo / ( M_para(c_para,H(u_new,H_pre)) * Gc ) * pv_new_alpha  , q ) * dx       
#R_dp = R_dp  +  inner( 4*lo**2 / c_para**2 * pa_new_alpha  , q )  * dx                         

R_dp = inner( (4*lo  * H(u_new,H_pre) / Gc + 1) * p_new ,q) * dx                  
R_dp = R_dp  +  inner( 4*lo**2 * grad(p_new) , grad(q)) * dx                                   
R_dp = R_dp  -  inner( 1 , q ) * dx 

#Define quasi-static initial Residual
R_du_initial = (inner(grad(v),sigma(u_new_alpha,p_pre))      ) * dx 
R_du_initIal =  R_du_initial  -  inner(     v ,   traction   )* dss(3) 
R_du_initIal =  R_du_initial  -  inner(     v ,  -traction   )* dss(4) 

du = TrialFunction(W)
dp = TrialFunction(V)

J_du = derivative(R_du,u_new,du)

J_dp = derivative(R_dp,p_new,dp)

#Construct Nonlinear solver
prob_disp = NonlinearVariationalProblem(R_du,u_new,bc_u,J_du)
prob_phi  = NonlinearVariationalProblem(R_dp,p_new,bc_phi,J_dp)

J_du_initial = derivative(R_du_initial,u_new,du)
prob_disp_initial = NonlinearVariationalProblem(R_du_initial,u_new,bc_u,J_du_initial)
#Construct solvers
solver_disp = NonlinearVariationalSolver(prob_disp)
solver_phi  = NonlinearVariationalSolver(prob_phi )

solver_disp_initial = NonlinearVariationalSolver(prob_disp_initial)

#Set nonlinear solver parameters
newton_prm = solver_disp.parameters['newton_solver']
newton_prm['relative_tolerance'] = 1e-6
newton_prm['absolute_tolerance'] = 1e-6
newton_prm['maximum_iterations'] = 20
newton_prm['error_on_nonconvergence'] = False

newton_prm2 = solver_phi.parameters['newton_solver']
newton_prm2['relative_tolerance'] = 1e-10
newton_prm2['absolute_tolerance'] = 1e-10
newton_prm2['maximum_iterations'] = 20
newton_prm2['error_on_nonconvergence'] = False

newton_prm3 = solver_disp_initial.parameters['newton_solver']
newton_prm3['relative_tolerance'] = 1e-10
newton_prm3['absolute_tolerance'] = 1e-10
newton_prm3['maximum_iterations'] = 20
newton_prm3['error_on_nonconvergence'] = False

#Construct Nonlinear solver
prob_disp1 = Problem(J_du,R_du,bc_u)

J_du_initial = derivative(R_du_initial,u_new)
prob_disp_initial1 = Problem(J_du_initial,R_du_initial,bc_u)
#Construct solvers
custom_solver = CustomSolver()
custom_solver.parameters['relative_tolerance'] = 1e-8
custom_solver.parameters['absolute_tolerance'] = 1e-5

#Quasi-static loading
load_total = 1e6
loadinc = 1e6
nst = 1
stat_t = 0
tol = 1e-4
while stat_t < nst:

      print('stat_t=',stat_t)
      stat_t += 1
      traction.B = stat_t * loadinc
      iter = 0
     
      #solver_disp_initial.solve() #Not working
      custom_solver.solve(prob_disp_initial1,u_new.vector())
      solver_phi.solve()
        
      u_pre.assign(u_new)
      p_pre.assign(p_new)
            
      #Store phi value (for paraview)
      phi_f = File ("./ResultsCase_dyn_tbc_um/phi_st"+str(stat_t) + ".pvd")
      phi_f << p_new
      
      H_pre.assign(project(psi_pls(u_new),X))     

expr= interpolate(traction, W)
print('expr',expr.vector()[:])
#Time stepping
while t<= T:
    
    print('nt=',nt)
                  
    solver_disp.solve() #rc
    #custom_solver.solve(prob_disp1,u_new.vector())
              
    update(u_new,u_pre,v_pre,a_pre,beta,gamma,dt)
    
    solver_phi.solve()
    p_pre.assign(p_new)
    
    if t==0:
         #Store phi value (for paraview)
         disp_f = File ("./ResultsCase_dyn_tbc_um/disp"+str(nt) + ".pvd")
         disp_f << u_new
         
    if nt % 10 == 0:
         #Store phi value (for paraview)
         phi_f = File ("./ResultsCase_dyn_tbc_um/phi"+str(int(nt/10)) + ".pvd")
         phi_f << p_new
        
    H_pre.assign(project(psi_pls(u_new),X))
    
    nt = nt + 1
    t += dt
