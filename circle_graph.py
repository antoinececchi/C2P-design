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
	def update_center(self,LR):
		self.center += LR*self.update_value


def derivate_spring(P1,P2):
	X1 = P1.center[0]-P2.center[0]
	Y1 = P1.center[1]-P2.center[1]
	d  = distance_plants(P1,P2)
	X_comp = X1/d 
	Y_comp = Y1/d 
	der_spring = np.array([X_comp,Y_comp]) #+ np.random.normal(0,0.1,2).reshape(2,1)
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
	gauss = exp(-((P1.center[1]-P2.center[1])**2)/(2*(Rp/3)))
	common_surf = (R**2)*arccos(d/R)-d*np.sqrt(R**2-d**2)+(Rp**2)*arccos(dp/Rp)-dp*np.sqrt(Rp**2-dp**2) 
	der_x = 2*(P1.center[0]-P2.center[0])*(((D+X*R)*R*X**2)/sqrt(1-X**2)+((D+Xp*Rp)*Rp*Xp**2)/sqrt(1-Xp**2))/(D**2)
	der_y = 2*(P1.center[1]-P2.center[1])*(((D+X*R)*R*X**2)/sqrt(1-X**2)+((D+Xp*Rp)*Rp*Xp**2)/sqrt(1-Xp**2))/(D**2)
	der_y += (P2.center[1]-P1.center[1])*common_surf/((Rp/3)**2)
	return np.array([der_x,der_y]*gauss).reshape(2,1)
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
			print('là')
		for plant in self.plantsList : 
			c = plt.Circle((plant.center[0],plant.center[1]),plant.ray,color =plant.color)
			ax.add_artist(c)
			ax.set_xlim([0,self.width])
			ax.set_ylim([0,self.height])
			fig.canvas.draw()
		if self.shown == 0 : 
			time.sleep(1)
			self.shown = 1
	def return_centers(self):
		ret = self.plantsList[0].center
		for plant in self.plantsList : 
			if plant != self.plantsList[0]:
				ret = np.concatenate((ret,plant.center))
		return ret
	def return_ders(self):
		ret = self.plantsList[0].update_value
		for plant in self.plantsList : 
			if plant != self.plantsList[0]:
				ret = np.concatenate((ret,plant.update_value))
		return ret

		

	def optimize_garden(self,eps,LR):
		'''
		This methods enables the gradient descent for the garden
		For each plant, we compute : 
		- Derivate of the surface when plants are close enough  
		- One for friends
		- One for ennemies
		- One for all the "safety zones" of all the plants.
		Still have to find an idea to deal with the cases in which the plants "follow" one another and leave to the infiny
		'''
		sum_der = 100*eps
		fig = plt.figure(figsize=(8,8))
		ax = fig.add_subplot(111)
		ax.set_xlim([0,self.width])
		print(self.width)
		step = 0 
		t = time.time()
		while  abs(sum_der) > eps: 
			step +=1
			sum_der = 0
			der_diff = np.empty([0,0])
			for Garden_plant in self.plantsList:
				update_center = np.array([0.0,0.0]).reshape(2,1)
				#Friends
				for friend in Garden_plant.friendPlants:
					dist = distance_plants(Garden_plant,friend)
					Gr   = Garden_plant.ray
					Gf   = friend.ray
					if dist<Gf+Gr and dist >= Gr-Gf and dist > Gf-Gr :
						update_center   += -1*derivate_diff_surf(Garden_plant,friend)-0.1*derivate_spring(Garden_plant,friend)
						sum_der         += np.sum(update_center)
					elif dist<Gr-Gf or  dist<Gf-Gr:
						pass
					else : 
						update_center   += -1*derivate_spring(Garden_plant,friend)
						sum_der         += np.sum(update_center)
				#Ennemy
				for ennemy in Garden_plant.ennemyPlants:
					dist = distance_plants(Garden_plant,ennemy)
					Gr   = Garden_plant.ennemy_ray
					Gf   = ennemy.ennemy_ray
					if dist<Gf+Gr and dist >= Gr-Gf and dist > Gf-Gr :
						update_center   += 0.5*derivate_diff_surf(Garden_plant,ennemy,"ennemy")
						sum_der         += np.sum(update_center)	
				#Neutral
				for plant_safe in self.plantsList:
					dist = distance_plants(Garden_plant,plant_safe)
					Gr   = Garden_plant.safety_ray
					Gf   = plant_safe.safety_ray
					if plant_safe != Garden_plant :
						if dist<Gf+Gr and dist >= Gr-Gf and dist > Gf-Gr:
							update_center   += 10*derivate_diff_surf(Garden_plant,plant_safe,"safety")
							#un seul sum_der semble suffisant et plus juste
							sum_der         += np.sum(update_center)
				
				Garden_plant.update_value =  update_center #penser à remultiplier(mettre le LR en argument par exemple)
				if isnan(update_center[0]):
					break
				if der_diff.shape[0] :
					der_diff = np.concatenate((der_diff,update_center))
				else : 
					der_diff = update_center 
			if step > 2 : 
				center_diffs = self.return_centers()-centers_old
				der_diffs    = old_ders-der_diff
				LR           = (np.transpose(center_diffs).dot(der_diffs)/np.linalg.norm(der_diffs,2)**2)[0]
				print(LR)
				
			centers_old = self.return_centers()
			old_ders    = der_diff

			#print(sum_der)
			for G_plant in self.plantsList:
				G_plant.update_center(LR)
				if G_plant.center[0]+G_plant.ray>self.width : 
					G_plant.center[0] = self.width-G_plant.ray
				if G_plant.center[0]-G_plant.ray<0 : 
					G_plant.center[0] = G_plant.ray
				if G_plant.center[1]+G_plant.ray>self.height : 
					G_plant.center[1] = self.height-G_plant.ray
				if G_plant.center[1]-G_plant.ray<0 : 
					G_plant.center[1] = G_plant.ray

			if step%1000==0:
				LR = LR
			if step%1000==0:
				#print(sum_der)
				#print(time.time()-t)
				self.showing(fig,ax)
			
				#break

				
		return 0




if __name__ == '__main__':
	t1 = time.time()
	P1 = Plant()
	P1.set_center([0.0,1.0])
	P1.set_ray(0.1)
	P1.set_color("yellow")
	P2 = Plant()
	P2.set_center([0.75,0.75])
	P2.set_ray(0.05)
	P2.set_color("red")
	P3 = Plant()
	P3.set_center([1.0,1.5])
	P3.set_ray(0.12)	
	P3.set_color("purple")
	P4 = Plant()
	P4.set_center([2.0,0.0])
	P4.set_ray(0.1)
	P5 = Plant()
	P5.set_center([0.2,0.2])
	P5.set_ray(0.1)
	P5.set_color("pink")
	P6 = Plant()
	P6.set_center([0.3,0.24])
	P6.set_ray(0.1)
	P6.set_color("orange")
	
	g = Garden(5,5)
	g.add_plant(P1)
	g.add_plant(P2)
	g.add_plant(P3)
	g.add_plant(P4)
	g.add_plant(P5)
	g.add_plant(P6)
	P1.set_name("tomate")
	P2.set_name("poivron")
	print(g.nb_plants)
	print(g.return_centers())
	#P1.add_friend(P2)
	P2.add_friend(P1)
	P1.add_friend(P2)
	P3.add_ennemy(P2)
	P3.add_friend(P4)
	P1.add_ennemy(P4)
	P4.add_friend(P3)
	P4.add_ennemy([P2,P1])
	P2.add_friend(P1)
	P2.add_ennemy(P3)
	P5.add_friend(P2)
	P5.add_ennemy(P4)
	P6.add_friend(P3)
	P6.add_ennemy(P1)
	print(P3.friendPlants[0].color,P3.ennemyPlants[0].color)
	#print(derivate_surf_sameR(g.plantsList[0],g.plantsList[1])+g.plantsList[0].center)
	print(P2.center,P2.name,P2.color)
	print(P1.center,P1.name,P1.color)
	print(time.time()-t1)
	t2= time.time()
	g.optimize_garden(eps=1e-6,LR=1e-1)
	print(time.time()-t2)
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
	