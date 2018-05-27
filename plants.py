import numpy as np
class Plant():
	'''
	Plant class enabling to : 
	Define a new plant and update its different properties (center --> set_center or set_x then set_y, ray --> set ray)
	Add enemies or friends as a list of plants. 
	Compute distance between the plant and another(the distance is between the centers)
	Compute common surface between the plants (only for friendship for now)
	Compute derivatives for these surfaces and "spring" forces
	'''
	def __init__(self,center = [0,0],ray =0):
		self.center = np.array(center).reshape(2,1)
		self.ray    = ray
		self.name   ="Unknown plant"
		#The list of friends plants is the list of plants that have benefits for this instance of Plant(). 
		self.friendPlants = []
		self.ennemyPlants = []
		self.color        = "green"	
		self.update_value = np.array([0,0])	
		self.safety_ray   = ray/10
		self.ennemy_ray   = ray*2

	def set_center(self,center):
		self.center = np.array(center).reshape(2,1)

	def get_center(self):
		return self.center

	def set_ray(self,value):
		self.ray = value
		self.safety_ray   = value/3
		self.ennemy_ray   = value*2

	def set_x(self,x):
		self.center[0]=x

	def set_y(self,y):
		self.center[1]=y

	def add_friend(self,plant):
		if isinstance(plant,list):
			self.friendPlants += plant
		else :
			self.friendPlants += [plant]
	def add_ennemy(self,plant):
		if isinstance(plant,list):
			self.ennemyPlants += plant
		else :
			self.ennemyPlants += [plant]

	def set_name(self,name):
		self.name = name

	def set_color(self,color):
		self.color = color

	def update_center(self,LR):
		self.center += LR*self.update_value
	def distance_plants(self,P2):
		X  = self.center-P2.center
		dist = np.sqrt(np.transpose(X).dot(X))[0]
		return dist

	def derivate_diff_surf(self,P2,type_inter="friend"):
		D   = self.distance_plants(P2)
		if type_inter=="friend":
			rho = self.ray**2-P2.ray**2
			R   = self.ray
			Rp  = P2.ray
		elif type_inter == "ennemy":
			rho = self.ennemy_ray**2-P2.ennemy_ray**2
			R   = self.ennemy_ray
			Rp  = P2.ennemy_ray
		elif type_inter == "safety":
			rho = self.safety_ray**2-P2.safety_ray**2
			R   = self.safety_ray
			Rp  = P2.safety_ray
		d   = abs((rho+D**2)/(2*D))
		dp  = abs((-rho+D**2)/(2*D))
		X   = d/R
		Xp  = dp/Rp
		gauss = np.exp(-((self.center[1]-P2.center[1])**2)/(2*(Rp/3)))
		common_surf = (R**2)*np.arccos(d/R)-d*np.sqrt(R**2-d**2)+(Rp**2)*np.arccos(dp/Rp)-dp*np.sqrt(Rp**2-dp**2) 
		der_x = 2*(self.center[0]-P2.center[0])*(((D+X*R)*R*X**2)/np.sqrt(1-X**2)+((D+Xp*Rp)*Rp*Xp**2)/np.sqrt(1-Xp**2))/(D**2)
		der_y = 2*(self.center[1]-P2.center[1])*(((D+X*R)*R*X**2)/np.sqrt(1-X**2)+((D+Xp*Rp)*Rp*Xp**2)/np.sqrt(1-Xp**2))/(D**2)
		der_y += (P2.center[1]-self.center[1])*common_surf/((Rp/3)**2)
		return np.array([der_x,der_y]*gauss).reshape(2,1)


	def derivate_spring(self,P2):
		X1 = self.center[0]-P2.center[0]
		Y1 = self.center[1]-P2.center[1]
		d  = self.distance_plants(P2)
		X_comp = X1/d 
		Y_comp = Y1/d 
		der_spring = np.array([X_comp,Y_comp]) #+ np.random.normal(0,0.1,2).reshape(2,1)
		return der_spring


	def common_surf(P1,P2): 	
		#Totally deprecated, only works if the plants have the same ray
		d = self.distance_plants(P2)
		return 2*self.ray*np.arccos(d/2/self.ray) - d*np.sqrt(self.ray**2-(d/2)**2)