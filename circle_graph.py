from matplotlib import pyplot as plt 
from pylab import *
import time
from garden import Garden
from plants import Plant
#TODO : 
#Split into several files
#Add threads to accelerate the processing 
#Find a good stopping criteria (perhaps the movement of all the plants)
#Remove the from pylab import *, quite dirty
#Que veut dire "espacer des plantes " ? Pour l'instant les zones de sécurité sont de centre à centre.


def test_garden(nb_plants,height,width):
	#Definition of the garden 
	gard = Garden(width,height)
	colors = ['red', 'green']
	fam ={}
	for color in colors:
		if color == 'red':
			fam[color]	 = 'tomato'
		elif color == 'green':
			fam[color]= 'bean'
		else : 
			fam[color] = None
	for _ in range(nb_plants):
		center = [np.random.uniform(0,width,1)[0],np.random.uniform(0,height,1)[0]]
		ray    = np.random.uniform(min(height,width)/30,min(height,width)/15,1)
		safety_ray = np.random.uniform(0.1,1,1)*ray
		ennemy_ray = np.random.uniform(1,2,1)*ray
		P = Plant(center)
		P.set_ray(ray)
		P.set_ennemy_ray(ennemy_ray)
		P.set_safety_ray(safety_ray)
		col = np.random.choice(colors,1)
		P.set_color(col[0])
		P.set_family(fam[col[0]])
		gard.add_plant(P)
	for P1 in gard.plantsList :
		for P2 in gard.plantsList :  
			friend_prob = np.random.uniform(0,1,1)
			if friend_prob > 0.85:
				P1.add_friend(P2)
			elif friend_prob < 0.15 : 
				P1.add_ennemy(P2)
			else : 
				pass
	P = Plant()
	P.set_ray(0.1)
	gard.add_plant(P)
	t = time.time()
	gard.optimize_garden(eps=1e-6,LR=1)
	tim = time.time()-t
	print(tim,gard.get_number_inter())
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
	
	g = Garden(2,2)
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
	#P6.add_friend(P3)
	#P6.add_ennemy(P1)
	print(P3.friendPlants[0].color,P3.ennemyPlants[0].color)
	#print(derivate_surf_sameR(g.plantsList[0],g.plantsList[1])+g.plantsList[0].center)
	print(P2.center,P2.name,P2.color)
	print(P1.center,P1.name,P1.color)
	g.optimize_garden(eps=1e-6,LR=1e-1)
	plt.show()
	for _ in range(5):
		test_garden(15,8,5)
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
	