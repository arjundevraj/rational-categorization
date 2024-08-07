# Usage: python control_models.py

import numpy as np
from scipy.optimize import minimize
import csv
import matplotlib.pyplot as plt

def constraint(x):
	return x[1]+x[2]+x[3]+x[4]+x[5]+x[6] - 1

NUM_TRIALS = 616
NUM_TRIALS_PER_SEG = 56

cons = ({'type': 'eq', 'fun': constraint})
bounds = ((0,20), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1))

# fitting the exemplar model to the data
def resid_exemp(x, args):
	data = args
	sens = x[0]
	w1 = x[1]
	w2 = x[2]
	w3 = x[3]
	w4 = x[4]
	w5 = x[5]
	w6 = x[6]
	SimA = np.zeros(NumTotalStim)
	SimB = np.zeros(NumTotalStim)
	probs = np.zeros(NumTotalStim)
	weights = np.array([w1, w2, w3, w4, w5, w6])

	for ThisStim in range(NumTotalStim): #loop through all of the stimuli
		for ThisTrainStim in range(NumAStim): #compare the stimulus to be classified with the As
			DistA = np.zeros(NumDim)
			for ThisDim in range(NumDim):
				DistA[ThisDim] = DistA[ThisDim] + (weights[ThisDim]) * abs(Stim[ThisStim,ThisDim] - Stim[ThisTrainStim,ThisDim])
			SimA[ThisStim] = SimA[ThisStim] + np.exp(-sens * sum(DistA))
		
		for ThisTrainStim in range(NumAStim, NumTotalStim): #compare the stimulus to be classified with the Bs
			DistB = np.zeros(NumDim)
			for ThisDim in range(NumDim):
				DistB[ThisDim] = DistB[ThisDim] + (weights[ThisDim]) * abs(Stim[ThisStim,ThisDim] - Stim[ThisTrainStim,ThisDim])
			SimB[ThisStim] = SimB[ThisStim] + np.exp(-sens * sum(DistB))

	for ThisStim in range(NumTotalStim):
		probs[ThisStim] = SimA[ThisStim] / (SimA[ThisStim] + SimB[ThisStim])
	
	res = np.sum(np.square(probs - data))
	return res

# fitting the prototype model to the data
def resid_proto(x, args):
	data = args
	sens = x[0]
	w1 = x[1]
	w2 = x[2]
	w3 = x[3]
	w4 = x[4]
	w5 = x[5]
	w6 = x[6]
	DistA = np.zeros(NumDim)
	DistB = np.zeros(NumDim)
	SimA = np.zeros(NumTotalStim)
	SimB = np.zeros(NumTotalStim)
	probs = np.zeros(NumTotalStim)
	weights = np.array([w1, w2, w3, w4, w5, w6])

	for ThisStim in range(NumTotalStim):
		DistA = np.zeros(NumDim)
		DistB = np.zeros(NumDim)

		for ThisDim in range(NumDim):
			DistA[ThisDim] = (weights[ThisDim]) * abs((Stim[ThisStim,ThisDim] - A_Prot[ThisDim]))
			DistB[ThisDim] = (weights[ThisDim]) * abs((Stim[ThisStim,ThisDim] - B_Prot[ThisDim]))
		
		SimA[ThisStim] = np.exp(-sens * sum(DistA))
		SimB[ThisStim] = np.exp(-sens * sum(DistB))

	for ThisStim in range(NumTotalStim):
		probs[ThisStim] = SimA[ThisStim] / (SimA[ThisStim] + SimB[ThisStim])
	
	res = np.sum(np.square(probs - data))
	return res

NumDim  = 6  #the number of dimenions or features
NumAStim = 7 #the number of category A exemplars
NumBStim = 7 #the number of category B exemplars
NumTotalStim = 14 #the total number of stimuli in the problem set

Stim = np.loadtxt("../helper/6dNLSStim_2.txt",  delimiter=" ")
A_Prot=np.array([0,0,0,0,0,0])
B_Prot=np.array([1,1,1,1,1,1])

responses = []
results_proto = [[], [], [], [], [], [], [], [], [], [], []]
results_exemp = [[], [], [], [], [], [], [], [], [], [], []]

# load all participant reponses
with open("../../data/experiment1_data.csv") as fp:
	reader = csv.reader(fp)
	for row in reader:
		responses.append([int(x) for x in row])

subj = 0
for response in responses:
	csv_num = response[0]
	# skip participants who received the experimental condition
	if csv_num % 2 != 0:
		continue
	stim_list = []
	# load the stimulus sequence
	filename = '../../stimuli/stim_list_' + str(csv_num) + '.csv'
	with open(filename, 'r') as f:
		for rec in csv.reader(f, delimiter=","):
			for x in rec:
				stim_list.append(int(x))
	
	vals = response[1:len(response)]
	# iterate trial segment by trial segment
	for start_trial in range(0, NUM_TRIALS, NUM_TRIALS_PER_SEG):
		# compute the observed probabilities
		counts = [0] * 14
		totals = [0] * 14
		for i in range(start_trial, start_trial + NUM_TRIALS_PER_SEG):
			totals[stim_list[i]] += 1
			if vals[i] == 1:
				counts[stim_list[i]] += 1
		data = np.array([count / total for count, total in zip(counts, totals)])

		# fit the models to the data
		for model in range(2):
			# run model-fitting with 10 random initial configurations
			reses = []
			for iteration in range(10):
				w1 = np.random.randint(0,101)
				w2 = np.random.randint(0,(101-w1))
				w3 = np.random.randint(0,(101-(w1+w2)))
				w4 = np.random.randint(0,(101-(w1+w2+w3)))
				w5 = np.random.randint(0,(101-(w1+w2+w3+w4)))
				w6 = 100-(w1+w2+w3+w4+w5) 
				w = np.array([w1,w2,w3,w4,w5,w6])
				np.random.shuffle(w) 
				weights = w/100.0
				params = np.insert(weights, 0, np.random.randint(0,200) / 10)
				if model == 0:
					res = minimize(resid_exemp, params, args=(data), constraints=cons, bounds=bounds)
				else:
					res = minimize(resid_proto, params, args=(data), constraints=cons, bounds=bounds)
				reses.append(res['fun'])

			# obtain the best-fit parameters from the initial random configurations
			if model == 0:
				results_exemp[start_trial // NUM_TRIALS_PER_SEG].append(min(reses))
			else:
				results_proto[start_trial // NUM_TRIALS_PER_SEG].append(min(reses))
	subj += 1
	if subj % 5 == 0:
		print(f"{subj} participant models fitted")

# Graph the results
graph_data_proto = []
errors_proto = []
for result in results_proto:
	graph_data_proto.append(sum(result)/len(result))
	errors_proto.append(np.std(result) / np.sqrt(len(result)))

graph_data_exemp = []
errors_exemp = []
for result in results_exemp:
	graph_data_exemp.append(sum(result)/len(result))
	errors_exemp.append(np.std(result) / np.sqrt(len(result)))

trial_segs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
	"font.size": 13
})

# Shade the area between y1 and y2
y1_proto = [val - err for val, err in zip(graph_data_proto, errors_proto)]
y2_proto = [val + err for val, err in zip(graph_data_proto, errors_proto)]

y1_exemp = [val - err for val, err in zip(graph_data_exemp, errors_exemp)]
y2_exemp = [val + err for val, err in zip(graph_data_exemp, errors_exemp)]

plt.fill_between(trial_segs, y1_proto, y2_proto, facecolor="blue", color='blue', alpha=0.2)
plt.plot(trial_segs, graph_data_proto,'blue', label='prototype')

plt.fill_between(trial_segs, y1_exemp, y2_exemp, facecolor="orange", color='orange', alpha=0.2)
plt.plot(trial_segs, graph_data_exemp,'orange', label='exemplar')

plt.xlabel('Trial Segment', fontsize=15)
plt.ylabel('Fit (SSE)', fontsize=15)
plt.ylim(0, 0.8)
plt.xticks(trial_segs)
plt.legend(loc='lower left')
plt.show()