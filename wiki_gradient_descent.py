# From wiki
# Finding local minimum for function 
# 12/03/2016

x_old = 0 
x_new = 6 
gamma = 0.001 # learning rate (step size)  
precision = 0.0001 # Two points are closed enough, implying oscillating at local minimum

def df(x): 
	y = 4 * x**3  - 9 * x**2
	return y 

while abs(x_new - x_old) > precision: 
	x_old = x_new
	x_new += -gamma * df(x_old)

print("The local minimum occurs at ", x_new)
