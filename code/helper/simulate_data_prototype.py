import numpy as np
import csv
from scipy.stats import bernoulli

def prototype_probability(x, stim):
	sens = x[0]
	w1 = x[1]
	w2 = x[2]
	w3 = x[3]
	w4 = x[4]
	w5 = x[5]
	w6 = x[6]
	weights = np.array([w1, w2, w3, w4, w5, w6])
	
	DistA = np.zeros(NumDim)
	DistB = np.zeros(NumDim)
	for ThisDim in range(NumDim):
		DistA[ThisDim] = (weights[ThisDim]) * abs((Stim[stim,ThisDim] - A_Prot[ThisDim]))
		DistB[ThisDim] = (weights[ThisDim]) * abs((Stim[stim,ThisDim] - B_Prot[ThisDim]))
		
	SimA = np.exp(-sens * sum(DistA))
	SimB = np.exp(-sens * sum(DistB))
	prob = SimA / (SimA + SimB)
	return prob

NumDim  = 6  #the number of dimenions or features
NumAStim = 7 #the number of category A exemplars
NumBStim = 7 #the number of category B exemplars
NumTrainStim = 14 #The total number of training exemaplrs
NumTotalStim = 14 #the total number of stimuli in the problem set

Stim = np.loadtxt("../6dNLSStim.txt",  delimiter=" ")
A_Prot=np.array([0,0,0,0,0,0])
B_Prot=np.array([1,1,1,1,1,1])

responses = []
results = [[], [], [], [], [], [], [], [], [], [], []]

with open("../data/experiment1_data.csv") as fp:
	reader = csv.reader(fp)
	for row in reader:
		responses.append([int(x) for x in row])

subj = 0
all_vals = []
for response in responses:
	csv_num = response[0]
	if csv_num % 2 == 0:
		continue
	print(csv_num)
	stim_list = []
	filename = '../../stimuli/stim_list_' + str(csv_num) + '.csv'
	with open(filename, 'r') as f:
		for rec in csv.reader(f, delimiter=","):
			for x in rec:
				stim_list.append(int(x))
	
	vals = [csv_num]
	stim_indices = {k: [] for k in range(14)}
	for trial in range(0, 616):
		stim_indices[stim_list[trial]].append(trial)
		prob = prototype_probability([10, 1/6, 1/6, 1/6, 1/6, 1/6, 1/6], stim_list[trial])
		X = bernoulli(prob)
		sample = int(X.rvs())
		if sample == 1:
			vals.append(1)
		else:
			vals.append(2)
	
	all_vals.append(vals)

with open("prototype_simulated_data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(all_vals)