import numpy as np

ACTIONS=('U','D',"L","R")
DELTA_THRESHOLD=1e-1
GAMMA=0.9

class Grid:
	def __init__(self,rows,cols):
		self.rows=rows
		self.cols=cols

	# make field matrix

	def set(self,rewards,block=None):
		self.rewards=rewards
		self.block=block
	# set the rewards points
	def set_state(self,s):
		self.i=s[0]
		self.j=s[1]
	#set agent's location
	def current_state(self):
		return (self.i,self.j)

	def move(self,action):
		old_i=self.i
		old_j=self.j
		if action=="U":
			self.i-=1
		elif action=="D":
			self.i+=1
		elif action=="R":
			self.j+=1
		elif action=="L":
			self.j-=1

		#if out of range then back to previous value
		if self.i==self.rows:
			self.i=old_i
		if self.i<0:
			self.i=old_i
		if self.j==self.cols:
			self.j=old_j
		if self.j<0:
			self.j=old_j
			
			
# object generating function		
def standard_grid():
  grid=Grid(3,4)
  rewards={(0,3):1,(1,3):-1}
  block={(1,1):-10}
  grid.set(rewards,block)
  return grid


#print function
def print_values(V, grid):
	for i in range(grid.rows):
		print("-----"*(grid.cols+2))
		for j in range(grid.cols):
			value = V.get((i, j), 0)
			if value >= 0:
				print("%.2f | " % value, end = "")
			else:
				print("%.2f | " % value, end = "") # -ve sign takes up an extra space
		print("")

def print_policy(P, grid):
	for i in range(grid.rows):
		for j in range(grid.cols):
			action = P.get((i, j), ' ')
			print("  %s  |" % action, end = "")
		print("")
		
		

		
if __name__=='__main__':
#make object
  grid=standard_grid()
  best_policy={}
  V={}
  for i in range(grid.rows):
    for j in range(grid.cols):
      point=(i,j)
      if point not in grid.rewards.keys() :
        V[point]=0
      if point in grid.rewards:
        V[point]=grid.rewards[point]

      if point in grid.block:
        V[point]=grid.block[point]
  print_values(V,grid)
  i=0
  while True:
    maxChange=0
    for s in V.keys():
      if s not in grid.rewards.keys() and s not in grid.block.keys():    
        for a in ACTIONS:
          grid.set_state(s)
          oldValue=V[s]
          grid.move(a)
	# at this example there is no action reward 
          v=GAMMA*V[grid.current_state()]
          if v>oldValue:
            V[s]=v
            best_policy[s]=a
	# optimal bellman equation 
        maxChange=max(maxChange,np.abs(oldValue-V[s]))
        
    print('\n%i 번째 반복'%i,end='\n')
    print_values(V,grid)
    print("")
    print_policy(best_policy,grid)
    i+=1
    if maxChange<DELTA_THRESHOLD:
          break
