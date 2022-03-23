from dolfin import *
import instant
CircleUt = """ #include “dolfin/fem/GenericDofMap.h” 
class CircleUt : public Expression
{
public:

  CircleUt() : Expression(2), t(0.0) {}

  void eval(Array<double>& values, const Array<double>& x) const
  { 
    //Cartesian cooridate -> Polar cooridate
    double r     = sqrt(x[0]*x[0]+x[1]*x[1]);
    double theta = atan2(x[1], x[0]);
    
    //Predefined parameters
    double nu     = 0.3;
    double E     = 1;
    double lmbda = 0.58114152874109139524851504375993;
    double pi    = 2*acos(0.0);
    
    //Displacement at polar coordinate
    double f     = ((1+lmbda)*sin((1+lmbda)*0.7*pi))/((1-lmbda)*sin((1-lmbda)*0.7*pi));
    double F     = pow(2.0*pi,lmbda-1.0)*(cos((1+lmbda)*theta)-f*cos((1-lmbda)*theta))/(1-f);
    double DF    = pow(2.0*pi,lmbda-1.0)*(sin((1+lmbda)*theta)*pow(1+lmbda,1.0)-f*sin((lmbda-1)*theta)*pow(lmbda-1,1.0))/(f-1);
    double DDF   = pow(2.0*pi,lmbda-1.0)*(cos((1+lmbda)*theta)*pow(1+lmbda,2.0)-f*cos((lmbda-1)*theta)*pow(lmbda-1,2.0))/(f-1);
    double DDDF  = pow(2.0*pi,lmbda-1.0)*(sin((1+lmbda)*theta)*pow(1+lmbda,3.0)-f*sin((lmbda-1)*theta)*pow(lmbda-1,3.0))/(f-1)*(-1);
    
    double ur    = pow(r,lmbda)*((1-pow(nu,2))*DDF+(lmbda+1)*(1-nu*lmbda-pow(nu,2)*(lmbda+1)))*F/(E*pow(lmbda,2)*(lmbda+1));
    double uthe  = pow(r,lmbda)*((1-pow(nu,2))*DDDF+(2*(1+nu)*pow(lmbda,2)+(lmbda+1)*(1-nu*lmbda-pow(nu,2)*(lmbda+1)))*DF/(E*pow(lmbda,2)*(1-pow(lmbda,2)));
    
    //Polar coordiate -> Catesian coordiate
    values[0] = t*(ur*cos(theta)-uthe*sin(theta));
    values[1] = t*(ur*sin(theta)+uthe*cos(theta));
  }

  double t;

};
"""

CircleUt = ''' #include “dolfin/fem/GenericDofMap.h” 
namespace dolfin {
   class CircleUt : public Expression 
   {
   public:
   
     Circle() : Expression(2), t(0.0), nu(0.3), E(1.0), lmbda(0.58114152874109139524851504375993), pi(2*acos(0.0)) t(0.0) {}
     
     void eval(Array<double>& values, const Array<double>& data, const ufc::cell&cell) const
     {
      //Cartesian cooridate -> Polar cooridate
      double r     = sqrt(x[0]*x[0]+x[1]*x[1]);
      double theta = atan2(x[1], x[0]);
      
      //Displacement at polar coordinate
      double f     = ((1+lmbda)*sin((1+lmbda)*0.7*pi))/((1-lmbda)*sin((1-lmbda)*0.7*pi));
      double F     = pow(2.0*pi,lmbda-1.0)*(cos((1+lmbda)*theta)-f*cos((1-lmbda)*theta))/(1-f);
      double DF    = pow(2.0*pi,lmbda-1.0)*(sin((1+lmbda)*theta)*pow(1+lmbda,1.0)-f*sin((lmbda-1)*theta)*pow(lmbda-1,1.0))/(f-1);
      double DDF   = pow(2.0*pi,lmbda-1.0)*(cos((1+lmbda)*theta)*pow(1+lmbda,2.0)-f*cos((lmbda-1)*theta)*pow(lmbda-1,2.0))/(f-1);
      double DDDF  = pow(2.0*pi,lmbda-1.0)*(sin((1+lmbda)*theta)*pow(1+lmbda,3.0)-f*sin((lmbda-1)*theta)*pow(lmbda-1,3.0))/(f-1)*(-1);
    
      double ur    = pow(r,lmbda)*((1-pow(nu,2))*DDF+(lmbda+1)*(1-nu*lmbda-pow(nu,2)*(lmbda+1)))*F/(E*pow(lmbda,2)*(lmbda+1));
      double uthe  = pow(r,lmbda)*((1-pow(nu,2))*DDDF+(2*(1+nu)*pow(lmbda,2)+(lmbda+1)*(1-nu*lmbda-pow(nu,2)*(lmbda+1)))*DF/(E*pow(lmbda,2)*(1-pow(lmbda,2)));

      //Polar coordiate -> Catesian coordiate
       values[0] = t*(ur*cos(theta)-uthe*sin(theta));
       values[1] = t*(ur*sin(theta)+uthe*cos(theta));
      }
      void update(const std::shared_ptr<const Function> x, double t)
      {
       const std::shared_ptr<const Mesh> mesh = x->function_space()->mesh(); 
       const std::shared_ptr<const GenericDofMap> dofmap = x->function_space()->dofmap();
       const uint ncells = mesh->num_cells();
       uint ndofs_per_cell; if (ncells > 0) 
        {
         CellIterator cell(*mesh); 
        }
         else
        {
         return;
        }
       }
      };
    }'''

Ut = CompiledExpression(compile_cpp_code(CircleUt), degree=3)
print(Ut.t)
