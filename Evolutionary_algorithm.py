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
                                specimens:list)->list[list]:
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
    
    def get_best_parent_offspring(self)->tuple[list]:
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
    
    evolution = Evolution('model1.txt', 200)
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
