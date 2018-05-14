import numpy as np 
import scipy
from matplotlib import pyplot as plt 
from pylab import *
import time

class Plant():
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
	def update_center(self):
		self.center += self.update_value

def derivate_surf_sameR(P1,P2):
	#TODO :  - add sqrt part + spring force
	#		 - Update this function to non similar ray cases
	#        - Put it in a class ?
	X1 = P1.center[0]-P2.center[0]
	Y1 = P1.center[1]-P2.center[1]
	d  = distance_plants(P1,P2)
	#To be coded in a better way because it is too long (add distances for instance)
	Y  = np.sqrt(P1.ray*2-(d**2)/4)/d
	sqrt_part   = -np.array([X1*(Y-Y**-1),Y1*(Y-Y**-1)])
	arccos_part = np.array([-2*P1.ray*((X1)/np.sqrt((X1**2+Y1**2-((X1**2+Y1**2)**2)/(4*P1.ray**2)))),-2*P1.ray*((Y1)/np.sqrt((X1**2+Y1**2-((X1**2+Y1**2)**2)/(4*P1.ray**2))))])
	return arccos_part.reshape(2,1)+sqrt_part.reshape(2,1)
def derivate_surf_safety(P1,P2):
	#TODO :  - add sqrt part + spring force
	#		 - Update this function to non similar ray cases
	#        - Put it in a class ?
	X1 = P1.center[0]-P2.center[0]
	Y1 = P1.center[1]-P2.center[1]
	d  = distance_plants(P1,P2)
	#To be coded in a better way because it is too long (add distances for instance)
	Y  = np.sqrt(P1.safety_ray**2-(d**2)/4)/d
	sqrt_part   = -np.array([X1*(Y-Y**-1),Y1*(Y-Y**-1)])
	arccos_part = np.array([-2*P1.safety_ray*((X1)/np.sqrt((X1**2+Y1**2-((X1**2+Y1**2)**2)/(4*P1.safety_ray**2)))),-2*P1.safety_ray*((Y1)/np.sqrt((X1**2+Y1**2-((X1**2+Y1**2)**2)/(4*P1.safety_ray**2))))])
	return arccos_part.reshape(2,1)+sqrt_part.reshape(2,1)
#def derivate_ennemy(P1,P2) : 
#	return -derivate_surf_sameR(P1,P2)
def derivate_ennemy(P1,P2):
	X1 = P1.center[0]-P2.center[0]
	Y1 = P1.center[1]-P2.center[1]
	d  = distance_plants(P1,P2)
	#To be coded in a better way because it is too long (add distances for instance)
	Y  = np.sqrt(P1.ennemy_ray**2-(d**2)/4)/d
	sqrt_part   = -np.array([X1*(Y-Y**-1),Y1*(Y-Y**-1)])
	arccos_part = np.array([-2*P1.ennemy_ray*((X1)/np.sqrt((X1**2+Y1**2-((X1**2+Y1**2)**2)/(4*P1.ennemy_ray**2)))),-2*P1.ennemy_ray*((Y1)/np.sqrt((X1**2+Y1**2-((X1**2+Y1**2)**2)/(4*P1.ennemy_ray**2))))])
	return -arccos_part.reshape(2,1)-sqrt_part.reshape(2,1)
def derivate_spring(P1,P2):
	X1 = P1.center[0]-P2.center[0]
	Y1 = P1.center[1]-P2.center[1]
	d  = distance_plants(P1,P2)
	X_comp = X1/d 
	Y_comp = Y1/d 
	der_spring = np.array([X_comp,Y_comp]) + np.random.normal(0,0.1,2).reshape(2,1)
	return der_spring

def distance_plants(P1,P2):
	X  = P1.center-P2.center
	dist = np.sqrt(np.transpose(X).dot(X))[0]
	return dist
def common_surf(P1,P2): 	
	d = distance_plants(P1,P2)
	return 2*P1.ray*np.arccos(d/2/P1.ray) - d*np.sqrt(P1.ray**2-(d/2)**2)
def derivate_diff_surf(P1,P2,type_inter="friend"):
	D   = distance_plants(P1,P2)
	if type_inter=="friend":
		rho = P1.ray**2-P2.ray**2
		R   = P1.ray
		Rp  = P2.ray
	elif type_inter == "ennemy":
		rho = P1.ennemy_ray**2-P2.ennemy_ray**2
		R   = P1.ennemy_ray
		Rp  = P2.ennemy_ray
	elif type_inter == "safety":
		rho = P1.safety_ray**2-P2.safety_ray**2
		R   = P1.safety_ray
		Rp  = P2.safety_ray
	d   = abs((rho+D**2)/(2*D))
	dp  = abs((-rho+D**2)/(2*D))
	X   = d/R
	Xp  = dp/Rp
	print(P1.center.reshape(1,2),P2.center.reshape(1,2))
	print(D)
	print(type_inter,P1.color,P2.color)
	der_x = 2*(P1.center[0]-P2.center[0])*(((D+X*R)*R*X**2)/sqrt(1-X**2)+((D+Xp*Rp)*Rp*Xp**2)/sqrt(1-Xp**2))/(D**2)
	der_y = 2*(P1.center[1]-P2.center[1])*(((D+X*R)*R*X**2)/sqrt(1-X**2)+((D+Xp*Rp)*Rp*Xp**2)/sqrt(1-Xp**2))/(D**2)
	return np.array([der_x,der_y]).reshape(2,1)
class Garden(plt.Figure):
	def __init__(self,height = 1,width = 1):
		self.height     = height
		self.width      = width
		self.plantsList = [] 
		self.nb_plants  = 0
		self.shown      = 0
	def add_plant(self,plant):
		if isinstance(plant,list):
			self.plantsList += plant
			self.nb_plants  += len(plant)
		else :
			self.plantsList += [plant]
			self.nb_plants  += 1
	def showing(self,fig,ax):
		ax.clear()
		if self.shown == 0 :
			fig.show()	
		for plant in self.plantsList : 
			c = plt.Circle((plant.center[0],plant.center[1]),plant.ray,color =plant.color)
			ax.add_artist(c)
			ax.set_xlim([0,self.width])
			ax.set_ylim([0,self.height])
			fig.canvas.draw()
		if self.shown == 0 : 
			time.sleep(1)
			self.shown = 1

		

	def optimize_garden(self,eps,LR):
		#TODO : add a check wether plants are close enough or not
		#update_center = LR*derivate_surf_sameR(self.plantsList[0],self.plantsList[1])
		sum_der = 100*eps
		fig = plt.figure(figsize=(8,8))
		ax = fig.add_subplot(111)
		ax.set_xlim([0,self.width])
		print(self.width)
		step = 0 
		while  abs(sum_der) > eps: 
			step +=1
			sum_der = 0
			for Garden_plant in self.plantsList:
				update_center = np.array([0.0,0.0]).reshape(2,1)
				#Friends
				for friend in Garden_plant.friendPlants:
					dist = distance_plants(Garden_plant,friend)
					Gr   = Garden_plant.ray
					Gf   = friend.ray
					if distance_plants(Garden_plant,friend)<friend.ray+Garden_plant.ray and dist >= Gr-Gf and dist > Gf-Gr :
						update_center   += -1*derivate_diff_surf(Garden_plant,friend)-0.1*derivate_spring(Garden_plant,friend)
						sum_der         += np.sum(update_center)
					elif distance_plants(Garden_plant,friend)<Garden_plant.ray-friend.ray or  distance_plants(Garden_plant,friend)<friend.ray-Garden_plant.ray :
						pass
					else : 
						update_center   += -0.1*derivate_spring(Garden_plant,friend)
						sum_der         += np.sum(update_center)
				#Ennemy
				for ennemy in Garden_plant.ennemyPlants:
					if distance_plants(Garden_plant,ennemy)<ennemy.ennemy_ray+Garden_plant.ennemy_ray:
						update_center   -= 0.5*derivate_diff_surf(Garden_plant,ennemy,"ennemy")
						sum_der         += np.sum(update_center)	
				#Neutral
				Garden_plant.update_value =  LR*update_center
			for plant1 in self.plantsList:
				for plant2 in self.plantsList:
					dist = distance_plants(plant1,plant2)
					Gr   = plant1.safety_ray
					Gf   = plant2.safety_ray
					if plant1 != plant2 :
						if dist<Gr+Gf and dist > Gr-Gf and dist > Gf-Gr:
							update_center   += derivate_diff_surf(plant1,plant2,"safety")
							sum_der         += np.sum(update_center)
						elif dist < Gf-Gr or dist < Gr-Gf:
							pass
					else : pass
				plant1.update_value += LR*update_center

			
				if isnan(update_center[0]):
					break
			#print(sum_der)
			for G_plant in self.plantsList:
				G_plant.update_center()
				#Make sure we stay in the garden
				if G_plant.center[0]+G_plant.ray>self.width : 
					G_plant.center[0] = self.width-G_plant.ray
				if G_plant.center[0]-G_plant.ray<0 : 
					G_plant.center[0] = G_plant.ray
				if G_plant.center[1]+G_plant.ray>self.height : 
					G_plant.center[1] = self.height-G_plant.ray
				if G_plant.center[1]-G_plant.ray<0 : 
					G_plant.center[1] = G_plant.ray
			if step%1000==0:
				self.showing(fig,ax)
		return 0




if __name__ == '__main__':
	P1 = Plant()
	P1.set_center([0.0,1.0])
	P1.set_ray(0.1)
	P1.set_color("yellow")
	P2 = Plant()
	P2.set_center([0.75,0.75])
	P2.set_ray(0.1)
	P2.set_color("red")
	P3 = Plant()
	P3.set_center([1.0,1.0])
	P3.set_ray(0.1)	
	P3.set_color("purple")
	P4 = Plant()
	P4.set_center([1.0,0.0])
	P4.set_ray(0.1)
	P5 = Plant()
	P5.set_center([0.2,0.2])
	P5.set_ray(0.1)
	P5.set_color("pink")
	P6 = Plant()
	P6.set_center([0.3,0.24])
	P6.set_ray(0.1)
	P6.set_color("orange")
	print(P2.safety_ray	)
	g = Garden(12,15)
	g.add_plant(P1)
	g.add_plant(P2)
	g.add_plant(P3)
	g.add_plant(P4)
	g.add_plant(P5)
	g.add_plant(P6)
	P1.set_name("tomate")
	P2.set_name("poivron")
	print(g.nb_plants)
	P1.add_friend(P2)
	P2.add_friend(P1)
	P1.add_friend([P2,P3])
	P3.add_ennemy(P2)
	P3.add_friend(P4)
	P1.add_ennemy(P4)
	P4.add_friend(P3)
	P4.add_ennemy([P2,P1])
	P2.add_friend(P1)
	P2.add_ennemy(P3)
	P5.add_friend([P2,P3])
	P5.add_ennemy(P4)
	P6.add_friend(P3)
	P6.add_ennemy(P1)

	#print(derivate_surf_sameR(g.plantsList[0],g.plantsList[1])+g.plantsList[0].center)
	print(P2.center,P2.name,P2.color)
	print(P1.center,P1.name,P1.color)
	t = time.time()
	g.optimize_garden(eps=1e-6,LR=5e-4)
	print(time.time()-t)
	print(P2.center,P2.name,P2.color)
	print(P1.center,P1.name,P1.color)
	plt.show()

	"""c =  plt.Circle((P1.center[0],P1.center[1]),P1.ray,color ='red')
	c1 = plt.Circle((P2.center[0],P2.center[1]),P2.ray)
	
	ax.add_artist(c)
	ax.add_artist(c1)	
	c2 = plt.Circle((P3.center[0],P3.center[1]),P3.ray,color ='green')
	ax.add_artist(c2)"""	
	#ax.set_autoscaley_on(False)
	#ax.plot(7*np.array(range(0,100)),np.array(range(0,100)))
	#g.showing(fig,ax)
	