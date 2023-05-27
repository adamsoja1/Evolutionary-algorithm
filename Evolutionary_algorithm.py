import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import random
import math


"""EVOLUTIONARY ALGORITHM FOR GIVEN PROBLEM"""
class Specimen:
    def __init__(self)->None:
        """Initiation of chromosome and its' standard deviation values"""
        self.chromosome =  np.random.uniform(-10,10,3)
        self.std = np.random.uniform(low=0, high=10, size=3)
        self.n = len(self.chromosome) + len(self.std)
        """tau1 and tau2"""
        self.tau = [1/math.sqrt(2*self.n),1/(math.sqrt(2*math.sqrt(self.n)))]

    def mutate(self) -> np.array:
        """This function returns mutated chromosome - does not change self.chromosome"""

        """
        Applies mutation to the given chromosome using the Evolution Strategy mutation operator
        with normally distributed random vector of mean 0 and covariance matrix P = diag(sigma^2).
        """
        ra = np.random.normal(loc=0, scale=self.std[0], size=1)
        rb = np.random.normal(loc=0, scale=self.std[1], size=1)
        rc = np.random.normal(loc=0, scale=self.std[2], size=1)
        
        ch1 = self.chromosome[0].copy() + ra
        ch2 = self.chromosome[1].copy() + rb
        ch3 = self.chromosome[2].copy() + rc
        
        sigmas = self.__mutate_sigmas()
        
        return self.__new_specimen(ch1[0],ch2[0],ch3[0],sigmas)

    def __new_specimen(self,
                       a:float,
                       b:float,
                       c:float,
                       sigmas:list)->object:
        """Generates new specimen with concrete a,b,c values 
        and its new mutated deviations  """
        
        new_specimen = Specimen()
        
        new_specimen.std = sigmas
        new_specimen.chromosome = np.array([a,b,c])
        return new_specimen
        
    def __mutate_sigmas(self)->np.array:
        r1 = np.exp(self.tau[0] * np.random.normal(0,1))
        r2 = np.exp([self.tau[1] * np.random.normal(0,1) for i in range(len(self.std))])
        new_sigma = self.std * r1 * r2
                 
        return np.array(new_sigma)
        
        
class Population:
    def __init__(self,
                 amount:int,
                 parent_offspring=False)->None:
        """Initiation of population -> list of Specimen classes (chromosomes)"""
        
        self.parent_offspring = parent_offspring
        self.parents = []
        self.offsprings = []
        for i in range(amount):
            specimen = Specimen()
            self.parents.append(specimen)

    def mutate_population(self)->None:
        """This function updates offsprings list  using Specimen mutate method"""
        if len(self.offsprings)==0:
            for i in range(len(self.parents)):
                for j in range(5):
                    self.offsprings.append(self.parents[i].mutate())

            if self.parent_offspring == True:        
                self.offsprings.extend(self.parents.copy())

                
        else:
            self.parents = self.offsprings.copy()
            self.offsprings = []

            for i in range(len(self.parents)):
                for j in range(5):
                    self.offsprings.append(self.parents[i].mutate())
 
                    
                    
            if self.parent_offspring == True:
                self.offsprings.extend(self.parents.copy())

        

                

class Evolution:
    def __init__(self,
                 path_to_model:str,
                 population_size:int)->None:
        
        self.__path_to_model = path_to_model
        self.df = self.read_model()
        self.population_size = population_size
        self.population = Population(self.population_size,True)

    def read_model(self)->pd.DataFrame:
        return pd.read_csv(f'{self.__path_to_model}', 
                           delim_whitespace=True, 
                           header=None, 
                           names=['x', 'y'])
    
    def show_model(self):
        fig, ax = plt.subplots(figsize=(20,15))
        plt.scatter(x=self.df['x'],y=self.df['y'])
        plt.show()


    def calculate_o(self,
                    specimens:list)->list:
        o_s = []
 
        for specimen in specimens:
            vector_temp = []
            a,b,c = specimen.chromosome
            for value in self.df['x']:
                y_temp = a*(value**2 - (b*np.cos((c*3.14*value))))
                vector_temp.append(y_temp)
            o_s.append(vector_temp)
        return o_s
    
    def calculate_mse(self,
                      specimens:list)->list:
        
        o_s = self.calculate_o(specimens)
        losses = []
        y_true = self.df['y']
        for i in range(len(o_s)):
            losses.append(np.abs(mean_squared_error(y_true, o_s[i])))
        return losses

    def select_best_individuals(self,
                                specimens:list)->list:
        """Returns best individuals based on their losses
        Minimal loss -> better"""
        
        losses = self.calculate_mse(specimens)
        "select indexes of best individuals"
        indices = np.argsort(losses)[:self.population_size]  
        "select best individuals based on indexes"
        offsprings = [self.population.offsprings[i] for i in indices] 
        return offsprings

    def select_best_parent(self)->list:
        population = self.population.parents
        losses = self.calculate_mse(population)
        "select indexes of best individuals"
        idx = np.argmax(losses)  
        "select best parent based on index"
        best_parent = self.population.parents[idx] 
        return best_parent
    
    def select_best_offspring(self)->list:
        population = self.population.offsprings
        losses = self.calculate_mse(population)
        "select indexes of best individuals"
        idx = np.argmax(losses)
        "select best individuals based on indexes"
        best_offspring = self.population.offsprings[idx] 
        return best_offspring
    
    def get_best_parent_offspring(self)->tuple:
        return self.select_best_parent(),self.select_best_offspring()


    def mse_parent_offspring(self)->float:
        """Takes self.get_best_parent_offspring->tuple and calculates its' mean squared error"""
        parent_offspring = self.get_best_parent_offspring()
        parent_o = self.calculate_function(parent_offspring[0].chromosome)
        offspring_o = self.calculate_function(parent_offspring[1].chromosome)
        return np.abs(mean_squared_error(parent_o, offspring_o))

    def update_population(self):
        """Updates whole algorithm -> mutates population"""
        self.population.mutate_population()
        self.population.offsprings = self.select_best_individuals(self.population.offsprings)
        print(self.calculate_one(self.population.offsprings[0].chromosome))
        print('   ')
        print(f'Params: a: {self.population.offsprings[0].chromosome[0]}, b: {self.population.offsprings[0].chromosome[1]},c: {self.population.offsprings[0].chromosome[2]}, std_a: {self.population.offsprings[0].std[0]}, std_b: {self.population.offsprings[0].std[1]},std_c: {self.population.offsprings[0].std[2]}')
        
        
    def calculate_function(self,
                           chromosome:np.array)->list:
        """Calculates o^ based on one chromosome"""
        a,b,c = chromosome
        vector_temp = []
        for value in self.df['x']:
            y_temp = a*(value**2 - (b*np.cos((c*3.14*value))))
            vector_temp.append(y_temp)
            
        return vector_temp
        
    def calculate_one(self,
                      chromosome:np.array)->list:
        """Calculates o^ based on one chromosome"""
        a,b,c = chromosome
        vector_temp = []
        for value in self.df['x']:
            y_temp = a*(value**2 - (b*np.cos((c*3.14*value))))
            vector_temp.append(y_temp)
            
        return np.abs(mean_squared_error(self.df['y'], vector_temp))
    
    def show_model_trained(self,
                           offspring:np.array)->plt.Figure:
        """Displays model on adjusted a,b,c"""
        a,b,c = offspring
        vector_temp = []
        for value in self.df['x']:
            y_temp = a*(value**2 - (b*np.cos((c*3.14*value))))
            vector_temp.append(y_temp)
        
        
        fig, ax = plt.subplots(figsize=(20,15))
        plt.plot(self.df['x'],vector_temp,color='blue')
        plt.scatter(self.df['x'],self.df['y'],color='orange')
        
        
if __name__ == '__main__':    
    
    evolution = Evolution('model1.txt', 500)
    
    for i in range(150):
        print(f'Iteration {i}: ',end='')
        evolution.update_population()   
        print(' ' * 6)
        print(f'Mse difference {i}: ',end='')
        mse = evolution.mse_parent_offspring()
        print('{:.6f}'.format(mse))
        print()
        if evolution.mse_parent_offspring() < 0.00001:
            break
        print('_'*30)    
        print('',end='\r')
        
    def show_model_trained(offspring):
        a,b,c = offspring
        vector_temp = []
        for value in evolution.df['x']:
            y_temp = a*(value**2 - (b*np.cos((c*3.14*value))))
            vector_temp.append(y_temp)
        
        
        fig, ax = plt.subplots(figsize=(20,15))
        plt.plot(evolution.df['x'],vector_temp,color='blue')
  
        

    
    evolution.show_model_trained(evolution.population.offsprings[0].chromosome)

from time import perf_counter

def results_print(sizes,approaches):
    
    for size in sizes:
        for approach in approaches:
            time_start = perf_counter()
            evolution = Evolution('model1.txt', size)
            evolution.population.parent_offspring = True
            if approach == 'Parent + offspring':
                evolution.population.parent_offspring = True
            else:
                evolution.population.parent_offspring = False
            i=0    
            while True:
                
                print(f'Iteration {i}: ',end='')
                evolution.update_population()   
                print(' ' * 6)
                print(f'Mse difference {i}: ',end='')
                mse = evolution.mse_parent_offspring()
                print('{:.6f}'.format(mse))
                print()
                if evolution.mse_parent_offspring() < 0.00001:
                    break
                print('_'*30)    
                print('',end='\r')
                i+=1 
            time_stop = perf_counter()
            time = time_stop - time_start
            with open('results.txt','a+') as file:
                file.write(f'{i},{time},{approach}, {size}, {evolution.population.offsprings[0].chromosome} ,{evolution.population.offsprings[0].std}\n')
                
                        
            

        
    
sizes = [10,50,70,100,150,200,500,1000,1500]
approaches = ['Parent + offspring','Parent,offspring']
results_print(sizes,approaches)




import pandas as pd

data = [
    (0.39747652000005473, 'Parent + offspring', 10, [6.81726118, -0.15177385, 4.47357109], [0.292105, 0.81045536, 0.64893694]),
    (1.6506575319999683, 'Parent, offspring', 10, [6.78242478, 0.2236641, -1.10969154], [2.93555997e-04, 4.30007828e-04, 8.31978050e-05]),
    (8.374242145000153, 'Parent + offspring', 50, [6.63328117, 1.91076597, 20.63692633], [1.98263494e-04, 4.13187532e-04, 8.43754090e-05]),
    (8.096537547000025, 'Parent, offspring', 50, [6.63335791, 1.91076262, -0.62677709], [1.24561823e-03, 4.89032040e-04, 3.88732062e-05]),
    (11.195644290000018, 'Parent + offspring', 70, [6.63341644, 1.91078771, -0.62677902], [1.33616916e-04, 1.25619532e-03, 6.13320028e-05]),
    (9.952839497000014, 'Parent, offspring', 70, [6.6333768, 1.91080739, -0.62678278], [8.53012565e-05, 2.88497535e-04, 3.93126558e-05]),
    (15.387736241000084, 'Parent + offspring', 100, [6.63335294, 1.91079338, 0.62678737], [4.28162398e-04, 1.85607823e-04, 2.12426456e-05]),
    (14.891080633000001, 'Parent, offspring', 100, [6.63338696, 1.91073851, 19.38336116], [2.43556034e-04, 2.41410392e-04, 1.04493422e-05]),
    (23.22253052499991, 'Parent + offspring', 150, [6.63336685, 1.91074284, 0.62678306], [9.63473317e-07, 1.52380315e-04, 1.01732405e-05]),
    (16.686700150999968, 'Parent, offspring', 150, [6.63329742, 1.91077214, 0.62678973], [5.14675582e-05, 5.63480095e-05, 5.04467473e-05]),
    (24.446906922000153, 'Parent + offspring', 200, [6.6334155, 1.91072806, 0.62678305], [4.00889766e-05, 2.67584699e-04, 7.36422159e-06]),
    (26.58770451800001, 'Parent, offspring', 200, [6.6333937, 1.91069804, -19.38336095], [1.10970294e-04, 4.92879574e-04, 3.82111490e-05]),
    (64.98132117399996, 'Parent + offspring', 500, [6.63334531, 1.91075811, 0.62678746], [1.08821763e-04, 1.04931566e-04, 1.38007187e-05]),
    (58.644564614000046, 'Parent, offspring', 500, [6.63335538, 1.91080566, 0.62678458], [4.22479810e-05, 9.44935140e-05, 3.79723061e-05]),
    (114.71774708499993, 'Parent + offspring', 1000, [6.63335933, 1.91079957, 0.62678314], [1.80662434e-04, 3.12786582e-05, 1.69839238e-05]),
    (105.93353380600001, 'Parent, offspring', 1000, [6.63335045, 1.91077694, -0.62678262], [4.05389710e-05, 5.19912408e-04, 5.07097703e-06]),
    (166.2528692410001, 'Parent + offspring', 1500, [6.63336733, 1.91075335, 0.62678332], [7.09149988e-04, 1.92657757e-03, 7.54964998e-05]),
    (156.30844559300022, 'Parent, offspring', 1500, [6.63333308, 1.91079363, -0.6267845], [9.27610009e-04, 3.38522785e-04, 1.14818368e-05])
]

df = pd.DataFrame(data, columns=['Time', 'Type', 'Population', 'Chromosome','Stds'])
print(df)


