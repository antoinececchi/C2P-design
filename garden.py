from pylab import *
import time
from plants import Plant
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
		- One for enemies
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
					dist = Garden_plant.distance_plants(friend)
					Gr   = Garden_plant.ray
					Gf   = friend.ray
					if dist<Gf+Gr and dist >= Gr-Gf and dist > Gf-Gr :
						update_center   += -1*Garden_plant.derivate_diff_surf(friend)-0.1*Garden_plant.derivate_spring(friend)
						sum_der         += np.sum(update_center)
					elif dist<Gr-Gf or  dist<Gf-Gr:
						pass
					else : 
						update_center   += -1*Garden_plant.derivate_spring(friend)
						sum_der         += np.sum(update_center)
				#Ennemy
				for ennemy in Garden_plant.ennemyPlants:
					dist = Garden_plant.distance_plants(ennemy)
					Gr   = Garden_plant.ennemy_ray
					Gf   = ennemy.ennemy_ray
					if dist<Gf+Gr and dist >= Gr-Gf and dist > Gf-Gr :
						update_center   += 0.5*Garden_plant.derivate_diff_surf(ennemy,"ennemy")
						sum_der         += np.sum(update_center)	
				#Neutral
				for plant_safe in self.plantsList:
					dist = Garden_plant.distance_plants(plant_safe)
					Gr   = Garden_plant.safety_ray
					Gf   = plant_safe.safety_ray
					if plant_safe != Garden_plant :
						if dist<Gf+Gr and dist >= Gr-Gf and dist > Gf-Gr:
							update_center   += 10*Garden_plant.derivate_diff_surf(plant_safe,"safety")
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
			if step%1==0:
				#print(sum_der)
				#print(time.time()-t)
				self.showing(fig,ax)
			
				#break

				
		return 0