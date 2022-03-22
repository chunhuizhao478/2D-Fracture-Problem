'''
5/13/21 Chunhui Zhao
Follow Miehe's article: A robust algorithmn...
Combine mpi (not very clear implementation) need further modification
No iteration limitation
'''
'''
Case 2       : Single edge noteched specimen under pure shear
Method       : Phase-field method
Mesh    type : Use Phi=1 to model initial crack
Element type : Linear triangular element
Author       : Chunhui Zhao
'''

from dolfin import *
import numpy as np
import numpy.linalg as la
from numpy.random import rand
import h5py
import matplotlib.pyplot as plt
import os

#Import MPI
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
amode = MPI.MODE_WRONLY|MPI.MODE_CREATE

#Import mesh
mesh = Mesh('miehe_slm.xml')

#Define Function Space
V = FunctionSpace(mesh,'CG',1)  
W = VectorFunctionSpace(mesh,'CG',1)
X = FunctionSpace(mesh,'CG',1) #Change to CG '1'
EPS = TensorFunctionSpace(mesh,'CG',1)

#Define Function
#u = Function(W)
v = TestFunction(W)

#p = Function(V)
q = TestFunction(V)

dim = 2
'''
Obtain space dimension
'''
#Define material parameters
Gc     = 2.7e-3         #kN/mm^2 [Fracture toughness]
lo     = 0.015          #mm      [Length scale      ] #test l1 = 0.015, l2 = 0.0075
co     = 1              #mm      [Scaling  paramter ]
Lambda = 121.15         #kN/mm^2 [1st Lame Constant ]
Mu     = 80.77          #kN/mm^2 [2nd Lame Constant ]
k = 10e-6
    
#Define phase field parameters
def g(phi):
    return ( 1 - phi ) ** 2
def g_prime(phi):
    return - 2 * ( 1 - phi )  
def alpha(phi):
    return phi ** 2
def alpha_prime(phi):
    return 2 * phi

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
    sig_final = (g(p)+k)*sig_pls + sig_mis
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
    #return conditional(lt(psi_pls(u_pre),psi_pls(u_new)),psi_pls(u_new),H_pre)

#Boundary conditions

top   = CompiledSubDomain("near(x[1],  0.5) && on_boundary")
bot   = CompiledSubDomain("near(x[1], -0.5) && on_boundary")
left  = CompiledSubDomain("near(x[0], -0.5) && on_boundary")
right = CompiledSubDomain("near(x[0],  0.5) && on_boundary")

def crack(x):
    return abs(x[1]) < 1e-3 and x[0] <= 0.0

load = Expression("t", t=0.0, degree=1)

bcbot1  = DirichletBC(W.sub(0),Constant(0.0),bot)
bcbot2  = DirichletBC(W.sub(1),Constant(0.0),bot)
bctop1  = DirichletBC(W.sub(0),    load     ,top) #load @ x-dir
bctop2  = DirichletBC(W.sub(1),Constant(0.0),top) #y-dir no change
bcleft  = DirichletBC(W.sub(1),Constant(0.0),left)
bcright = DirichletBC(W.sub(1),Constant(0.0),right)

bc_u   = [bcbot1,bcbot2,bctop1,bctop2,bcleft,bcright]
#bc_phi = [DirichletBC(V,Constant(1.0),crack)]
bc_phi = [] #Real crack

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
top.mark(boundaries,1)
ds = Measure("ds")(subdomain_data=boundaries)
n = FacetNormal(mesh)

#Construct Variational Form
p_pre = Function(V)
p_new = Function(V)
H_pre = Function(V)
u_pre = Function(W)
u_new = Function(W)

#Random initial guess
num = u_new.vector().get_local().size
u_new.vector()[:] = np.random.random(num)*DOLFIN_EPS

R_du = (inner(grad(v),sigma(u_new,p_new))) * dx

R_dp = (Gc*lo*inner(grad(p_new),grad(q))+((Gc/lo)+2.0*H(u_new,H_pre))\
             *inner(p_new,q)-2.0*H(u_new,H_pre)*q)*dx

du = TrialFunction(W)
dp = TrialFunction(V)

J_du = derivative(R_du,u_new,du)

J_dp = derivative(R_dp,p_new,dp)

#Construct Nonlinear solver
prob_disp = NonlinearVariationalProblem(R_du,u_new,bc_u,J_du)
prob_phi  = NonlinearVariationalProblem(R_dp,p_new,bc_phi,J_dp)

#Construct solvers
solver_disp = NonlinearVariationalSolver(prob_disp)
solver_phi  = NonlinearVariationalSolver(prob_phi )

#Set nonlinear solver parameters

newton_prm = solver_disp.parameters['newton_solver']
newton_prm['relative_tolerance'] = 1e-10
newton_prm['absolute_tolerance'] = 1e-10
newton_prm['maximum_iterations'] = 20
newton_prm['error_on_nonconvergence'] = False

newton_prm2 = solver_phi.parameters['newton_solver']
newton_prm2['relative_tolerance'] = 1e-10
newton_prm2['absolute_tolerance'] = 1e-10
newton_prm2['maximum_iterations'] = 20
newton_prm2['error_on_nonconvergence'] = False
         
#Initialization
t   = 0
u_r = 2e-2
dt  = 5e-4   #5e-4
tol = 1e-4
nt  = 0
#Define list for store the results
#with initial condition
loadinc0 = [0.0]
list0 = [0.0]

#Create new directory for storing plots
def createDir(dir_name):
    try:    
        os.mkdir(dir_name)
        return True;
    except:
        return False

createDir('Pics')

#Staggered scheme
#Time loop
while t <= 1.0:

    print('nt=',nt)
    nt = nt + 1
    t += dt
    load.t = t * u_r   #Delta_u = dt * u_r =  1 * 10 ^ (-5)
    if t >=0.4:        #Approx 500 time step (given in reference)
        dt = 5e-4    #Delta_u = dt * u_r =  1 * 10 ^ (-6)       #5e-4 Consistent with reference
    iter = 0
    err_u = 1
    err_p = 1
    
    #Step loop
    while err_u > tol or err_p > tol:
        iter += 1
        print('ITER',iter)
                          
        solver_disp.solve()
        solver_phi.solve()  
        
        u_diff = u_new.vector() - u_pre.vector()
        p_diff = p_new.vector() - p_pre.vector()
        
        err_u = norm( u_diff, 'L2' ) / ( norm( u_pre, 'L2') + 1e-10 )
        err_p = norm( p_diff, 'L2' ) / ( norm( p_pre, 'L2') + 1e-10 )
        
        print('err_u',err_u)
        print('err_p',err_p)
        
        u_pre.assign(u_new)
        p_pre.assign(p_new)
        
    print('Iterations:', iter, ', Total time', t)
    print('Iterations:', iter, ', Total time', t)
    print('Iterations:', iter, ', Total time', t)
    
    #Store phi value (for paraview)
    #phi_f = File ("./ResultsCase1_st/phi"+str(nt) + ".pvd")
    #phi_f << p_new
    
    #Compute tangential traction
    Traction = dot(sigma(u_new,p_new),n)
    fx = ( Traction[0] )* ds(1)
    Fx = assemble(fx) 
                    
    loadinc0.append(t*u_r)
    list0.append(Fx)
          
    #For quick view/check    
    if nt % 100 == 0:
       if rank == 0:
          plt.figure()
          plt.plot(loadinc0,list0,'b')
          plt.title('Pure Shear')
          plt.xlabel('Displacement[mm]')
          plt.ylabel('Force[kN]')
          plt.xlim(0.0,0.02)
          plt.ylim(0.0,0.6)
          plt.legend()
          plt.savefig('Pics/foo'+str(nt)+'.png')
       
    H_pre.assign(project(psi_pls(u_new),X))

#Final plot
plt.figure()
plt.plot(loadinc0,list0,'b')
plt.title('Pure Shear')
plt.xlabel('Displacement[mm]')
plt.ylabel('Force[kN]')
plt.xlim(0.0,0.02)
plt.ylim(0.0,0.6)
plt.legend()
plt.savefig('foo'+str(nt)+'.png')
#Store the data
data = np.column_stack([loadinc0, list0])
datafile_path = "./ForceDisp_pure_shear.txt"
np.savetxt(datafile_path , data, fmt=['%f','%f'])
print ('Simulation completed')  
