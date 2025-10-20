syms x(t)
syms y(t)
syms x2(t)
syms y2(t)
syms x3(t)
syms y3(t)
k=5;
b=0.0113;
ep=0.004;
u1=0.20;
u2=0.30;
a=0.18;
alp=77.5;
cond = x2(0) == 2;
I=0.50*(x2(0.02)-x)
I2=2.5*(x(0.05)-x2)+0.4*(x3(0.02)-x2)
I3=2.5*(x2(0.07)-x3)


ode=diff(x,t)==alp*(k*x(x+b)*(1-x)-y*x)+I
ode1=diff(y,t)==alp*(ep+((y*u1)/(x+u2)))*(-y-k*x*(x-a-1))
ode2=diff(x2,t)==alp*(k*x2(x2+b)*(1-x2)-y2*x2)+I2
ode21=diff(y2,t)==alp*(ep+((y2*u1)/(x2+u2)))*(-y-k*x2*(x2-a-1))
ode3=diff(x3,t)==alp*(k*x3(x3+b)*(1-x3)-y3*x3)+I3
ode31=diff(y3,t)==alp*(ep+((y3*u1)/(x3+u2)))*(-y-k*x3*(x3-a-1))

xSol(t)=dsolve(ode)
ySol(t) = dsolve(ode1)
x2Sol(t)=dsolve(ode2)
y2Sol(t) = dsolve(ode21)
x3Sol(t)=dsolve(ode3)
y3Sol(t) = dsolve(ode31)