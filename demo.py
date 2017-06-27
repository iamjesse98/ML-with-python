# gradient descent --- sum of squares --- linear regression is just an ml, u can even use it in dl
# mathematically( through program ) => amount of study hour is directly proportional to marks obtained

from numpy import *

# to minimise error ( or loss ), sum of square errors
# Error(m, b) = ( 1 / N ) * Sum ( ( y(i) - ( m * x(i) + b ) ) ^ 2, i = 1..N ), squaring for positive, taking care of magnitude
def compute_error_for_given_points(b, m, points):
	totalError = 0
	for i in range(0, len(points)):
		x = points[i][0]
		y = points[i][1]
		totalError += ( y - (m * x + b)) ** 2
	return totalError / float(len(points))

def step_gradient(b_current, m_current, points, learningRate):
	# gradient descent, a tangent line, gives + / -, we use partial derivative
	b_gradient, m_gradient = 0, 0
	N = float(len(points))
	for i in range(0, len(points)):
		x = points[i][0]
		y = points[i][1]
		b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
		m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
	new_b = b_current - (learningRate * b_gradient)
	new_m = m_current - (learningRate * m_gradient)
	return [new_b, new_m]

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
	b, m = starting_b, starting_m
	for i in range(num_iterations):
		b, m = step_gradient(b, m, array(points), learning_rate) # this is where the real good things happen
	return [b, m]

def run():
	# pull ( parse ) our dataset
	points = list(map(lambda l: list(map(float, l.split(','))), open('data.csv', 'r').read().split()))
	#print(points)
	learning_rate = 0.0001 # how fast our model works, its a bell curve, its a hyperparameter
	initial_b, initial_m = 0, 0 # y = mx + b, slope - intercept
	num_iterations = 1000 # increases with dataset inputs
	print("Starting gradient descent at b = {}, m = {}, error = {}".format(initial_b, initial_m, compute_error_for_given_points(initial_b, initial_m, points)))
	print("Running...")
	[b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations) # this is where we write the logic
	print("After {} iterations b = {}, m = {}, error = {}".format(num_iterations, b, m, compute_error_for_given_points(b, m, points)))

run() # this is the main function