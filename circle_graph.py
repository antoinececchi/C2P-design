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
	