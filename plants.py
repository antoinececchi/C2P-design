import numpy as np
from threading import Thread
class Plant(Thread):
	'''
	Plant class enabling to :
	Define a new plant and update its different properties (center --> set_center or set_x then set_y, ray --> set ray)
	Add enemies or friends as a list of plants.
	Compute distance between the plant and another(the distance is between the centers)
	Compute common surface between the plants (only for friendship for now)
	Compute derivatives for these surfaces and "spring" forces
	'''
	#Todo : ajouter les threads (un par plante ? ici)
	def __init__(self,center = [0.0,0.0],ray =0):
		Thread.__init__(self)
		self.center = np.array(center).reshape(2,1)
		self.ray    = ray
		self.name   ="Unknown plant"
		#The list of friends plants is the list of plants that have benefits for this instance of Plant().
		self.friendPlants = []
		self.ennemyPlants = []
		self.color        = "green"
		self.update_value = np.array([0,0])
		self.safety_ray   = ray
		self.ennemy_ray   = ray
		self.family     = None


	def set_center(self,center):
		'''
		define the center thus the position of a plant
		'''
		self.center = np.array(center).reshape(2,1)

	def get_center(self):
		'''
		obtain the position of a plant
		'''
		return self.center

	def set_ray(self,value):
		'''
		define the positive interaction ray of a given plant
		'''
		self.ray = value

	def set_ennemy_ray(self,ray):
		'''
		define the negative interation ray of a plant
		'''
		self.ennemy_ray = ray
	def set_safety_ray(self,ray):
		'''
		define the minimal distance at which any other plant, ennemy or friend should
		be
		'''
		self.safety_ray = ray


	def set_x(self,x):
		'''
		another way to set the center of a plant when the user only wants the
		plant to move on the x axis
		'''
		self.center[0]=x

	def set_y(self,y):
		'''
		see set_x, enables the user to set the center along the y axis
		'''
		self.center[1]=y

	def set_family(self,family):
		'''
		Allows to set the family of a plant
		'''
		self.family = family

	def add_friend(self,plant):
		'''
		Add friends to the current insance of plant
		The argument plant can be either a list of plant objects or a plant
		If it is a list, this list will be stacked to the current list of friends.
		If it is a plant it is simply added to the list of friend plants

		'''
		if not(isinstance(plant,list)):
			plant = [plant]
		for pl in plant :
			if pl in self.friendPlants :
				print("Can't add twice the same plant in the friends list")
			elif pl in self.ennemyPlants :
				print("can't be both friend and ennemy")
			elif pl == self :
				print("Can't add the plant to its own friends")
			else:
				self.friendPlants += plant

	def add_ennemy(self,plant):
		'''
		Add ennemies to the current insance of plant
		The argument plant can be either a list of plant objects or a plant
		If it is a list, this list will be stacked to the current list of
		ennemies.
		If it is a plant it is simply added to the list of ennemy plants

		'''
		if not(isinstance(plant,list)):
			plant = [plant]
		for pl in plant :
			if pl in self.ennemyPlants :
				print("Can't add twice the same plant in the ennemy list")
			elif pl in self.friendPlants:
				print("can't be both ennemy and friend")
			elif pl == self :
				print("Can't add the plant to its own ennemies")
			else:
				self.ennemyPlants += plant


	def set_name(self,name):
		'''
		define the name of the plant
		'''
		self.name = name

	def set_color(self,color):
		'''
		Define the color of a plant, mostly useful when debugging to display a
		garden when a plant is in it
		'''
		self.color = color

	def update_center(self,LR):
		'''
		Add a value to the center of the plant, this takes the update_value
		attribute of a plant and multiplies it by a Learning Rate.
		'''
		self.center += LR*self.update_value
	def distance_plants(self,P2):
		'''
		Returns the distance between the plants P2 and self.
		'''
		X  = self.center-P2.center
		dist = np.sqrt(np.transpose(X).dot(X))[0]
		return dist

	def derivate_diff_surf(self,P2,type_inter="friend",axis = 1):
		'''
		Returns the derivative corrisponding to the variation of the surface in
		the current state of the garden for this instance of plant. P2 is a plant
		object with which the derivative will be computed. inter is the kind of
		interaction between the 2 plants. Finally, a axis argument is available to
		force the plants to align along a certain direction. A Gaussian componant is
		artificially added to the derivative to force this behaviour.
		'''
		D   = self.distance_plants(P2)
		if type_inter=="friend":
			rho = self.ray**2-P2.ray**2
			R   = self.ray
			Rp  = P2.ray
		elif type_inter == "ennemy":
			# TODO: is it really self.ennemy_ray, might have to think who is the
			# ennemy of whom
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
		# Define the gaussian component
		# TODO: check if the gaussian indeed gives the good behaviour
		gauss = np.exp(-((self.center[1]-P2.center[1])**2)/(2*(Rp/3)))
		# The surface shared by the 2 plants.
		common_surf = (R**2)*np.arccos(d/R)-d*np.sqrt(R**2-d**2)+(Rp**2)*np.arccos(dp/Rp)-dp*np.sqrt(Rp**2-dp**2)
		# The x component of the derivative
		der_x = 2*(self.center[0]-P2.center[0])*(((D+X*R)*R*X**2)/np.sqrt(1-X**2)+((D+Xp*Rp)*Rp*Xp**2)/np.sqrt(1-Xp**2))/(D**2)
		# The y component of the derivative
		der_y = 2*(self.center[1]-P2.center[1])*(((D+X*R)*R*X**2)/np.sqrt(1-X**2)+((D+Xp*Rp)*Rp*Xp**2)/np.sqrt(1-Xp**2))/(D**2)
		if axis :
			der_x += (P2.center[0]-self.center[0])*common_surf/((Rp/3)**2)
		else :
			der_y += (P2.center[1]-self.center[1])*common_surf/((Rp/3)**2)
		return np.array([der_x,der_y]*gauss).reshape(2,1)


	def derivate_spring(self,P2):
		'''
		The derivative of the spring strength added between plants to make friends
		come to one another when they don't share surface
		'''
		X1 = self.center[0]-P2.center[0]
		Y1 = self.center[1]-P2.center[1]
		d  = self.distance_plants(P2)
		X_comp = X1#/d
		Y_comp = Y1#/d
		der_spring = np.array([X_comp,Y_comp])
		return der_spring
	def derivate_alignment(self,P2,axis = 0):
		'''
		An alignment strength to make the plants (self and P2) of a given family go along the
		same line. This line is defined by the axis argument
		'''
		if axis :
			der = np.array([0.0,2*(self.center[1]-P2.center[1])[0]]).reshape(2,1)
		else :
			der = np.array([2*(self.center[0]-P2.center[0])[0],0.0]).reshape(2,1)
		return der


	def common_surf(P1,P2):
		#Totally deprecated, only works if the plants have the same ray
		d = self.distance_plants(P2)
		return 2*self.ray*np.arccos(d/2/self.ray) - d*np.sqrt(self.ray**2-(d/2)**2)
