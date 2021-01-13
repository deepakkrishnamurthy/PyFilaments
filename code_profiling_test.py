import numpy as np
from profiler import profile


class test:

	def __init__(self):
		pass
	
	@profile(sort_by='cumulative', lines_to_print=10, strip_dirs=True)
	def test_fun(self, N):
		sum = 0
		for ii in range(N):
			sum+=(self.test_A(ii) + self.test_B(ii))

		return sum

	def test_A(self, N):
		sum = 0
		for ii in range(N):
			sum+=ii

		return sum

	def test_B(self,N):

		sum = 0
		for ii in range(N):
			sum+=ii

		return sum






A = test()

A.test_fun(10000)





