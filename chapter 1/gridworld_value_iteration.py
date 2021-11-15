import numpy as np


ACTIONS = ('U', 'D', 'L', 'R')
DELTA_THRESHOLD = 1e-2
GAMMA = 0.9



class Grid:
  def __init__(self,rows,cols):
    self.rows=rows
    self.cols=cols


  def set(self,rewards):
    self.rewards=rewards

  def set_state(self,s):
    self.i=s[0]
    self.j=s[1]

  def current_state(self):
    return (self.i,self.j)


  def move(self,action):
    if action=="U":
      self.i-=1
    elif action=="D":
      self.i+=1
    elif action=="R":
      self.j+=1
    elif action=="L":
      self.j-=1


    if self.i<0:
      self.i=0
    if self.i==self.rows:
      self.i=self.rows-1
    if self.j<0:
      self.j=0
    if self.j==self.cols:
      self.j=self.cols-1
    return self.rewards.get((self.i,self.j),0)





# 격자 공간과 각 상태에서 선택 가능한 행동을 정의
def standard_grid():
	grid = Grid(5, 5)
	rewards = {(0, 4): 1, (3, 4): -1,(4,4):1,(3,4):-1,(1,2):1}

	grid.set(rewards)
	return grid

def print_values(V, grid):
	for i in range(grid.cols):
		print("-"*grid.rows*6)
		for j in range(grid.cols):
			value = V.get((i, j), 0)
			if value >= 0:
				print("%.2f | " % value, end = "")
			else:
				print("%.2f | " % value, end = "") # -ve sign takes up an extra space
		print("")

def print_policy(P, grid):
	for i in range(grid.cols):
		print("-"*grid.rows*6)
		for j in range(grid.cols):
			action = P.get((i, j), ' ')
			print("  %s  |" % action, end = "")
		print("")
		
		
		
		
		

if __name__ =='__main__':
  grid=standard_grid()

  V={}
  for i in range(grid.rows):
    for j in range(grid.cols):
      point=(i,j)
      if point in grid.rewards.keys():
        V[point]=grid.rewards[point]
      else:
        V[point]=0



  i=0
  while True:
    maxChange=0
    for s in V.keys():
      if s not in grid.rewards.keys():
        oldValue=V[s]
        for a in ACTIONS:
          grid.set_state(s)
          r=grid.move(a)
          v=r+GAMMA*V[grid.current_state()]
          if v>oldValue:
            V[s]=v
        maxChange=max(maxChange,np.abs(oldValue-V[s]))
    
    print('\n%i 번째 반복'%i,end="\n")
    print_values(V,grid)
    i+=1

    if maxChange<DELTA_THRESHOLD:
      break

  best_policy={}
  for s in V.keys():
    if s not in grid.rewards.keys():
      bestAction=None
      bestValue=float('-inf')

      for a in ACTIONS:
        grid.set_state(s)
        r=grid.move(a)
        v=r+GAMMA*V[grid.current_state()]
        if v>bestValue:
          bestValue=v
          bestAction=a
      best_policy[s]=bestAction



  print_policy(best_policy,grid)
