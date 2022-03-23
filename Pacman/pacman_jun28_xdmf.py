#Use xdmf
from dolfin import *
import numpy as np

#Import MPI
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
amode = MPI.MODE_WRONLY|MPI.MODE_CREATE

#Define Phase Field Optimization Solver
#Alternate-minimization
class PhaseField(OptimisationProblem):

    def __init__(self, total_energy, D_phi_TE, J_phi, phi, bc_phi):
        OptimisationProblem.__init__(self)
        self.total_energy = total_energy
        self.Dphi_total_energy = D_phi_TE
        self.J_phi=J_phi
        self.phi=phi
        self.bc_phi=bc_phi
    
    def f(self, x):
        self.phi.vector()[:] = x
        return assemble(self.total_energy)

    def F(self, b, x):
        self.phi.vector()[:] = x
        assemble(self.Dphi_total_energy, b)
        for bc in self.bc_phi:
            bc.apply(b)

    def J(self, A, x):
        self.phi.vector()[:] = x
        assemble(self.J_phi, A)
        for bc in self.bc_phi:
            bc.apply(A)

#Define circular Boundary Condition Expression
#In the initial stage, omega_bar and lmbda are not set to be variable.
#omega_bar = 0.3*PI -> lmbda = 0.58114152874109139524851504375993

class CircleUt(UserExpression):
    def __init__(self,nu,E,lmbda,tpara,*arg,**kwargs):
        super().__init__(*arg,**kwargs)
        self.nu    = nu
        self.E     = E
        self.lmbda = lmbda
        self.tpara = tpara
    
    def eval(self,value,x):
        #Cartesian cooridate -> Polar cooridate
        r = sqrt(x[0]*x[0]+x[1]*x[1])
        theta = np.arctan2(x[1], x[0])
        
        #Predefined parameters
        f     = ((1+self.lmbda)*np.sin((1+self.lmbda)*0.7*np.pi))/((1-self.lmbda)*np.sin((1-self.lmbda)*0.7*np.pi))
        F     = (2.0*np.pi)**(self.lmbda-1.0)*(np.cos((1+self.lmbda)*theta)-f*np.cos((1-self.lmbda)*theta))/(1-f)
        DF    = (2.0*np.pi)**(self.lmbda-1.0)*(np.sin((1+self.lmbda)*theta)*(1+self.lmbda)**(1.0)-f*np.sin((self.lmbda-1)*theta)*(self.lmbda-1)**(1.0))/(f-1)
        DDF   = (2.0*np.pi)**(self.lmbda-1.0)*(np.cos((1+self.lmbda)*theta)*(1+self.lmbda)**(2.0)-f*np.cos((self.lmbda-1)*theta)*(self.lmbda-1)**(2.0))/(f-1)
        DDDF  = (2.0*np.pi)**(self.lmbda-1.0)*(np.sin((1+self.lmbda)*theta)*(1+self.lmbda)**(3.0)-f*np.sin((self.lmbda-1)*theta)*(self.lmbda-1)**(3.0))/(f-1)*(-1)
        
        #Displacement at polar coordinate
        ur    = (r)**(self.lmbda)*((1-(self.nu)**(2))*DDF+(self.lmbda+1)*(1-self.nu*self.lmbda-(self.nu)**(2)*(self.lmbda+1))*F)/(E*(self.lmbda)**(2)*(self.lmbda+1))
        uthe  = (r)**(self.lmbda)*((1-(self.nu)**(2))*DDDF+(2*(1+self.nu)*(self.lmbda)**(2)+(self.lmbda+1)*(1-self.nu*self.lmbda-(self.nu)**(2)*(self.lmbda+1)))*DF)/(E*(self.lmbda)**(2)*(1-(self.lmbda)**(2)))
        
        #Polar coordiate -> Catesian coordiate
        value[0] = self.tpara*(ur*np.cos(theta)-uthe*np.sin(theta))
        value[1] = self.tpara*(ur*np.sin(theta)+uthe*np.cos(theta))
        
    def value_shape(self):
        return (2,)

#Import mesh
mesh = Mesh()
with XDMFFile("mesh.xdmf") as infile:
     infile.read(mesh)
mvc = MeshValueCollection("size_t",mesh,2)
with XDMFFile("mf.xdmf") as infile:
     infile.read(mvc,"name_to_read")
mf = cpp.mesh.MeshFunctionSizet(mesh,mvc)

#Define Function Space
V = FunctionSpace(mesh,'CG',1)  
W = VectorFunctionSpace(mesh,'CG',1)
X = FunctionSpace(mesh,'CG',1) 

#Define Function
u  = Function(W)
du = TrialFunction(W)
v = TestFunction(W)
p  = Function(V)
dp = TrialFunction(V)
q = TestFunction(V)

#Define Functions
p_pre = Function(V)
p_new = Function(V)
p_ub  = Function(V) #Upper-bound
p_diff= Function(V)
u_pre = Function(W)
u_new = Function(W)


#Define material parameters
#"Plane strain"
dim    = 2
Gc     = 1
R      = 1
nu     = 0.3
E      = 1; #E'=E E_prime = E / ( 1 - nu ** 2 )         #'Plane strain'
lo     = 0.015
Lambda = E * nu / ( ( 1 + nu ) * ( 1 - 2 * nu ) ) #'Plane strain'
Mu     = E     / ( ( 1 + nu ) * 2             )   #'Plane strain'
lmbda  = 0.58114152874109139524851504375993
tpara  = 0.0
k      = 10e-8
model_type = 'AT1'

#Define phase field parameters
def g(p):
    return ( 1 - p ) ** 2
    
#Define constitutive functions
def epsilon(u):
    return sym(grad(u))
def sigma(u,p):
    return (g(p)+k)*(2.0*Mu*epsilon(u)+Lambda*tr(epsilon(u))*Identity(len(u)))

#Define energy density
def elastic_energy_density(u,p):
    return 0.5 * inner(sigma(u,p),epsilon(u))
def dissipated_energy_density(p,model_type):
    if model_type == 'AT1':
       return (3/8) * Gc * ( p    / lo + lo * inner(grad(p),grad(p)))
    if model_type == 'AT2':
       return (1/2) * Gc * ( p**2 / lo + lo * inner(grad(p),grad(p)))

#Define energy and derivatives
elastic_energy = elastic_energy_density(u_new,p_new)*dx
dissipated_energy = dissipated_energy_density(p_new,model_type)*dx
total_energy = elastic_energy + dissipated_energy
#derivatives -> disp
D_disp_TE = derivative(total_energy,u_new,v)
J_du = derivative(D_disp_TE,u_new,du)
#derivatives -> alpha
D_phi_TE = derivative(total_energy,p_new,q) #q:testfunction
J_phi = derivative(D_phi_TE,p_new,dp) #dp:trialfunction
        
#Boundary conditions
Ut = CircleUt(nu,E,lmbda,tpara=0.0)
bc_u = DirichletBC(W,Ut,mf,5)
bc_phi = []

#Check mark boundary
#f = File("facets.pvd")
#f << boundaries

#Random initial guess
num = u_new.vector().get_local().size
u_new.vector()[:] = np.random.random(num)*DOLFIN_EPS

#Define variational Form and solvers
#Disp
prob_disp = NonlinearVariationalProblem(D_disp_TE,u_new,bc_u,J_du)
solver_disp = NonlinearVariationalSolver(prob_disp)

newton_prm = solver_disp.parameters['newton_solver']
newton_prm['relative_tolerance'] = 1e-10
newton_prm['absolute_tolerance'] = 1e-10
newton_prm['maximum_iterations'] = 10
newton_prm['error_on_nonconvergence'] = False

#Phase Field
def define_initial_guess_phi():
    return Constant(0.0)
"""
Initial guess for the damage, damage field can have damage levels smaller than this state
"""
def define_initial_phi():
    return Constant(0.0)
"""
Initial value (at each time step) for the damage, 
damage field cannot have damage levels smaller than this state
This is lower bound for damage field in alternate minimization (at each time step)
Which ensures irreversibility condition
"""
# Initialize Phase Field
#p_new.interpolate(define_initial_guess_phi())
p_pre.interpolate(define_initial_phi())
p_ub.interpolate(Constant(1.0)) 
'''Upper bound of phase field'''

solver_phi = PETScTAOSolver()
prob_phi = PhaseField(total_energy, D_phi_TE, J_phi, p_new, bc_phi)

#Time Step
#Define list for store the results
data_t = []
data_elastic = []
data_dissipated = []
data_total = []
#Parameters for Alternate Minimization
max_iterations = 100
tol = 1e-5
#Initization
time = 0
dt   = 0.02     #(See source code default_parameters.py min:0.0 max:1.0 nsteps:50)
nt   = 0
while time <= 1.0:
    if time * 2 >= 1.64 and time * 2 <= 1.76:
        dt = 5e-3
        nt = nt + 0.25
        time += dt
    else: 
        dt = 2e-2
        nt = nt + 1
        time += dt
    Ut.tpara = time * 2 #(See Tanne's article Fig3, to capture t ranging from (0,2.0))
    
    if rank == 0:
        print("Next is the # %i time step - t: %.3e" % (nt, time) )
        print("tpara is %.3e" % (Ut.tpara) )
    
    iteration = 0
    error = 1.0   #(See source code problem.py, only constraint on damage variable)
    while iteration < max_iterations and error > tol:
        solver_disp.solve()
        solver_phi.solve(prob_phi,p_new.vector(),p_pre.vector(),p_ub.vector())        
        p_diff = p_new.vector() - p_pre.vector()
        error = p_diff.norm("linf") #linf := max(abs)
        iteration += 1
        p_pre.assign(p_new)
        if rank == 0:
            print("AM: Iteration number: %i - Error: %.3e" % (iteration, error))
        
    #Store phi for paraview
    #if time * 2 >= 1.64 and time * 2 <= 1.70:
    phi_f = File ("./Results_pacman_jun28/phi"+str(nt) + ".pvd")
    phi_f << p_new
    
    #Compute energy
    elastic_energy_value = assemble(elastic_energy)
    dissipated_energy_value = assemble(dissipated_energy)
    total_energy_value = assemble(total_energy)
    if rank == 0:  
        print("This is the # %i time step - t: %.3e" % (nt, time) )
        print("tpara is %.3e" % (Ut.tpara) )
        print('elastic_energy_value=',elastic_energy_value)
        print('dissipated_energy_value=',dissipated_energy_value)
        print('total_energy_value=',total_energy_value)
        
        data_t.append(Ut.tpara)
        data_elastic.append(elastic_energy_value) 
        data_dissipated.append(dissipated_energy_value)
        data_total.append(total_energy_value)

        data = np.column_stack([data_t, data_elastic,data_dissipated,data_total])
        datafile_path = "./ForceDisp_pacman_energy.txt"
        np.savetxt(datafile_path , data, fmt=['%f','%f','%f','%f'])  
      
print('Simulation completed')
