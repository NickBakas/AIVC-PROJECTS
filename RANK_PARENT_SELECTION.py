#THIS IS AN ASSIGNMENT ON GENETIC ALGORITHMS (ROULETTE WHEEL SELECTION)
#THE CITY DISTANCES ARE POPULATED RANDOMLY

#1: Initialize a population of chromosomes.
#2: Evaluate each chromosome in the population.
#3: Create new chromosomes by mating current chromosomes.Apply crossover and mutation operators.
#4: Delete less fit members of the population.
#5: Evaluate and insert the new chromosomes in the population.
#6: If stopping criterion is satisfied, then stop and return the best chromosome; otherwise, go to step 3.
#END GA


#This was part of a Machine Learning semester project and was graded 9/10
#User is welcome to tweak: citysum, popsize, maxgen, mutated, crstart, crend
#citysum, crstart, crend, mutated are corellated



import numpy as np
import matplotlib.pyplot as plt
import random as rd

#Variables
citysum=30
popsize=100
maxgen=1000
#Initialization of variables for metrics
bestFitnessPerGeneration = np.zeros((maxgen, 1))
worstFitnessPerGeneration = np.zeros((maxgen, 1))
averageFitnessPerGeneration = np.zeros((maxgen, 1))
bestTotalDistancesPerGeneration = np.zeros((maxgen, 1))
worstTotalDistancesPerGeneration = np.zeros((maxgen, 1))
averageTotalDistancesPerGeneration = np.zeros((maxgen, 1))
#Diagonal symmetrical matrix of distances from city to city
distances = np.random.randint(1, 101, size=(citysum, citysum))
distances = np.triu(distances, 1) + np.triu(distances, 1).T
np.fill_diagonal(distances, 0)

Population = np.zeros((popsize, citysum), dtype=int)

for i in range(popsize):
    Population[i, :] = np.random.permutation(citysum) + 1
    
print('Distance Matrix:')
print(distances)
print('Population Matrix:')
print(Population)



for gen in range(1, maxgen ):
#calculate total distances for each chromosome
#Fitness is calculaed as 1/total distance   
#Indexes are used to "travel" from city to city  
        calculatedDistances = np.zeros_like(Population)
        for i in range(Population.shape[0]):
            for j in range(Population.shape[1]):
                if j != Population.shape[1] - 1:
                    cityIndex1 = Population[i, j]
                    cityIndex2 = Population[i, j + 1]
                    calculatedDistances[i, j] = distances[cityIndex1 - 1, cityIndex2 - 1]  
                else:
                    cityIndex1 = Population[i, j]
                    cityIndex2 = Population[i, 0]
                    calculatedDistances[i, j] = distances[cityIndex1 - 1, cityIndex2 - 1]
                    
        totalDistances = np.sum(calculatedDistances, axis=1)
    
    
        Fitness = np.zeros((popsize, 1))
        print(totalDistances)    
    
        for i in range(popsize):
            Fitness[i] = 1 / totalDistances[i]
    
    
            totalFitness = np.sum(Fitness)  
        
    
#Since this is a rank parent selection, the selection probability is calculated as rank/sum of ranks    
        sortedFitness = np.sort(Fitness, axis=0)  
        rank = np.zeros_like(Fitness)
        print('Fitness Matrix:')
        print(Fitness)
        
        for b in range(len(sortedFitness)):
            rank[b] = b + 1
    
        selectionProbability = rank / np.sum(rank)        
    
        SummedProbability = np.cumsum(selectionProbability)
        ranges = np.zeros((popsize, 2))
        rangeNumbers = np.arange(1, popsize + 1)
    
        for i in range(popsize):
            ranges[i, 0] = rangeNumbers[i]
            ranges[i, 1] = SummedProbability[i]
            
        chr1 =np.zeros((Population.shape[1]), dtype=int)
        chr2=np.zeros((Population.shape[1]), dtype=int)
    
        for u in range(2):
            randomFit = rd.random()
            print("Random fit:", randomFit)
            
            for i in range(len(SummedProbability) - 1):  # Adjusted loop range
                if randomFit > SummedProbability[i] and randomFit <= SummedProbability[i+1]:
                    if u==0: 
                        chr1= Population[i+1] 
                    elif u==1:
                        chr2=Population[i+1]
                    break
                       
                        
                                
                   
        print('summed probability')
        print(SummedProbability)
        print('chr1')
        print(chr1)
        print('chr2')
        print(chr2)
    
    

        #ORDER CROSSOVER OX
#switch elements between chromosomes between crst(crossover start) to crend (crossoverend)
#create new arrays full of zeros that will hold the remaining (not switched) value of each chromosome and once that is done remove the zeros
#fill the rest of chr1 and chr2 with the remaining values accordingly and in order        
        crst=10 
        crend=20

        off1 =np.zeros(len(chr1),dtype=int)
        off2 =np.zeros(len(chr1),dtype=int)
        rem1=np.zeros(len(chr1),dtype=int)
        rem2=np.zeros(len(chr1),dtype=int)

        off1[crst:crend]=chr2[crst:crend]
        off2[crst:crend]=chr1[crst:crend]
        print(off1,off2,chr1,chr2)
        print(crend-crst)


        for i in range(len(chr1)):
           if chr1[i] not in off1: 
              rem1[i]=chr1[i]
        rem1=rem1[rem1!=0]
        


        for i in range(len(chr1)):
           if chr2[i] not in off2: 
              rem2[i]=chr2[i]
        rem2=rem2[rem2!=0]

        print("rem1",rem1)
        print("rem2",rem2)


        xx=0
        xy=0
        for i in range(len(chr1)):
            if off1[i]==0:
                off1[i]=rem1[xx]
                xx+=1
                
        for i in range(len(chr2)):
            if off2[i]==0:
                off2[i]=rem2[xy]
                xy+=1   
                            
        print("chr1",chr1)
        print("chr2",chr2)
        print("off1",off1)       
        print("off2",off2)
        
        #MUTATION
#switch the order of elements within the array of offspring1 and offspring2
#mutaded choses how many times this will be performed (rate) and is advised to be held low for the protection of good genes            
        mutated=5


        for mut in range(mutated):
            mutationstart=rd.randint(0,len(chr1)-1)
            mutationend=rd.randint(0,len(chr1)-1)
            off1[mutationstart],off1[mutationend]=off1[mutationend],off1[mutationstart]
            off2[mutationstart],off2[mutationend]=off2[mutationend],off2[mutationstart]
            
        
        print("mutation")   
        print("chr1",chr1)
        print("chr2",chr2)        
        print("off1",off1)       
        print("off2",off2) 
    
    
    
        
        off1Distances = np.zeros(citysum,dtype=int)
        off2Distances = np.zeros(citysum, dtype=int)
#same distance calculation for the off1 and off2
        for i in range(citysum):
            if i < citysum - 1:
                cityIndex1 = off1[i]
                cityIndex2 = off1[i + 1]
            else:
                cityIndex1 = off1[-1]
                cityIndex2 = off1[0]
            off1Distances[i] = distances[cityIndex1 - 1, cityIndex2 - 1]
            
            
        for i in range(citysum):
            if i < citysum - 1:
                cityIndex1 = off2[i]
                cityIndex2 = off2[i + 1]
            else:
                cityIndex1 = off2[-1]
                cityIndex2 = off2[0]
            off2Distances[i] = distances[cityIndex1 - 1, cityIndex2 - 1]
    
        newOffspringFitness1 = 1 / np.sum(off1Distances)
        newOffspringFitness2 = 1 / np.sum(off2Distances)
        
        print("check",newOffspringFitness1)
        print(newOffspringFitness2)
        print(off2Distances)
               
        
        FitnessIndexes=np.argsort(Fitness,axis=0)
        sortedFitness = Fitness[FitnessIndexes]
        
        
        
#sort the fitnesses and given that the lowest of the offspring is higher than the lowest of the population replace the two lowest        
        if newOffspringFitness2>newOffspringFitness1:
            lowestoffspringFitness=newOffspringFitness1
        else:
            lowestoffspringFitness=newOffspringFitness2
        
        
        if lowestoffspringFitness>=sortedFitness[0]:
            Population[FitnessIndexes[0]]=off1
            Population[FitnessIndexes[1]]=off2
            
            
            
                
        

#repeat until maxgen
#metrics of cenvergence and total distances
        averageFitnessPerGeneration[gen] = np.mean(Fitness)
        bestFitnessPerGeneration[gen] = np.max(Fitness)
        worstFitnessPerGeneration[gen] = np.min(Fitness)
        worstTotalDistancesPerGeneration[gen] = np.max(totalDistances)
        bestTotalDistancesPerGeneration[gen] = np.min(totalDistances)
        averageTotalDistancesPerGeneration[gen] = np.mean(totalDistances)

        print('Population Matrix:')
        print(Population)
        

###plots of convergence and total distance
plt.figure()
plt.plot(range(1, maxgen + 1), bestFitnessPerGeneration, color=[1, 0.549, 0], linewidth=1)
plt.plot(range(1, maxgen + 1), averageFitnessPerGeneration, color=[0, 0, 0], linewidth=1)
plt.plot(range(1, maxgen + 1), worstFitnessPerGeneration, color=[0.3010, 0.7450, 0.9330], linewidth=1)
plt.xlabel('Generations')
plt.ylabel('Fitness')
plt.legend(['Best Fitness', 'Average Fitness', 'Worst Fitness'])
plt.title('Convergence-Fitness/Generations')
plt.grid(True)
plt.show()

plt.figure()
plt.plot(range(1, maxgen + 1), bestTotalDistancesPerGeneration, color=[1, 0.549, 0], linewidth=1)
plt.plot(range(1, maxgen + 1), averageTotalDistancesPerGeneration, color=[0, 0, 0], linewidth=1)
plt.plot(range(1, maxgen + 1), worstTotalDistancesPerGeneration, color=[0.3010, 0.7450, 0.9330], linewidth=1)
plt.xlabel('Generations')
plt.ylabel('Distances')
plt.legend(['Best Total Distances', 'Average Total Distances', 'Worst Total Distances'])
plt.title('Convergence-Distances/Generations')
plt.grid(True)
plt.show()

