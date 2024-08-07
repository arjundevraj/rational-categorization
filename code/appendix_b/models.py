import numpy as np
from scipy.optimize import minimize
import csv
import pickle
import matplotlib.pyplot as plt
import sys
import argparse
from copy import deepcopy

def constraint(x):
	return x[1]+x[2]+x[3]+x[4]+x[5]+x[6] - 1

cons = ({'type': 'eq', 'fun': constraint})

NUM_TRIALS = 616
NUM_TRIAL_SEGS = 11

bounds = ((0,20), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1))
bounds_response_scaling = ((0,20), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1), (0,10))

def resid_exemp(x, *args):
	data = args[0]
	stim_seen = args[1]
	sens = x[0]
	w1 = x[1]
	w2 = x[2]
	w3 = x[3]
	w4 = x[4]
	w5 = x[5]
	w6 = x[6]
	guessing = x[7]
	SimA = np.zeros(NumTotalStim)
	SimB = np.zeros(NumTotalStim)
	probs = np.zeros(NumTotalStim)
	weights = np.array([w1, w2, w3, w4, w5, w6])

	for ThisStim in range(NumTotalStim): #loop through all of the stimuli
		if ThisStim not in stim_seen: continue
		for ThisTrainStim in range(NumAStim): #compare the stimulus to be classified with the As
			if ThisTrainStim not in stim_seen:
				continue
			DistA = np.zeros(NumDim)
			for ThisDim in range(NumDim):
				DistA[ThisDim] = DistA[ThisDim] + (weights[ThisDim]) * abs(Stim[ThisStim,ThisDim] - Stim[ThisTrainStim,ThisDim])
			SimA[ThisStim] = SimA[ThisStim] + np.exp(-sens * sum(DistA))
		
		for ThisTrainStim in range((NumAStim), NumTrainStim): #compare the stimulus to be classified with the Bs
			if ThisTrainStim not in stim_seen:
				continue
			DistB = np.zeros(NumDim)
			for ThisDim in range(NumDim):
				DistB[ThisDim] = DistB[ThisDim] + (weights[ThisDim]) * abs(Stim[ThisStim,ThisDim] - Stim[ThisTrainStim,ThisDim])
			SimB[ThisStim] = SimB[ThisStim] + np.exp(-sens * sum(DistB))

	for ThisStim in stim_seen:
		probs[ThisStim] = (guessing / 2) + ((1 - guessing) * SimA[ThisStim] / (SimA[ThisStim] + SimB[ThisStim]))
	
	res = np.sum(np.square(probs - data))
	return res

def resid_exemp_response_scaling(x, *args):
	data = args[0]
	stim_seen = args[1]
	sens = x[0]
	w1 = x[1]
	w2 = x[2]
	w3 = x[3]
	w4 = x[4]
	w5 = x[5]
	w6 = x[6]
	guessing = x[7]
	gamma = x[8]
	SimA = np.zeros(NumTotalStim)
	SimB = np.zeros(NumTotalStim)
	probs = np.zeros(NumTotalStim)
	weights = np.array([w1, w2, w3, w4, w5, w6])

	for ThisStim in stim_seen: #loop through all of the stimuli
		for ThisTrainStim in range(NumAStim): #compare the stimulus to be classified with the As
			if ThisTrainStim not in stim_seen:
				continue
			DistA = np.zeros(NumDim)
			for ThisDim in range(NumDim):
				DistA[ThisDim] = DistA[ThisDim] + (weights[ThisDim]) * abs(Stim[ThisStim,ThisDim] - Stim[ThisTrainStim,ThisDim])
			SimA[ThisStim] = SimA[ThisStim] + np.exp(-sens * sum(DistA))
		
		for ThisTrainStim in range((NumAStim), NumTrainStim): #compare the stimulus to be classified with the Bs
			if ThisTrainStim not in stim_seen:
				continue
			DistB = np.zeros(NumDim)
			for ThisDim in range(NumDim):
				DistB[ThisDim] = DistB[ThisDim] + (weights[ThisDim]) * abs(Stim[ThisStim,ThisDim] - Stim[ThisTrainStim,ThisDim])
			SimB[ThisStim] = SimB[ThisStim] + np.exp(-sens * sum(DistB))

	for ThisStim in stim_seen:
		probs[ThisStim] = (guessing / 2) + ((1 - guessing) * (SimA[ThisStim] ** gamma) / ((SimA[ThisStim] ** gamma) + (SimB[ThisStim] ** gamma)))
	
	res = np.sum(np.square(probs - data))
	return res

def resid_proto(x, *args):
	data = args[0]
	stim_seen = args[1]
	sens = x[0]
	w1 = x[1]
	w2 = x[2]
	w3 = x[3]
	w4 = x[4]
	w5 = x[5]
	w6 = x[6]
	guessing = x[7]
	DistA = np.zeros(NumDim)
	DistB = np.zeros(NumDim)
	SimA = np.zeros(NumTotalStim)
	SimB = np.zeros(NumTotalStim)
	probs = np.zeros(NumTotalStim)
	weights = np.array([w1, w2, w3, w4, w5, w6])

	for ThisStim in stim_seen:
		DistA = np.zeros(NumDim)
		DistB = np.zeros(NumDim)

		for ThisDim in range(NumDim):
			DistA[ThisDim] = (weights[ThisDim]) * abs((Stim[ThisStim,ThisDim] - A_Prot[ThisDim]))
			DistB[ThisDim] = (weights[ThisDim]) * abs((Stim[ThisStim,ThisDim] - B_Prot[ThisDim]))
		
		SimA[ThisStim] = np.exp(-sens * sum(DistA))
		SimB[ThisStim] = np.exp(-sens * sum(DistB))

	for ThisStim in stim_seen:
		probs[ThisStim] = (guessing / 2) + ((1 - guessing) * SimA[ThisStim] / (SimA[ThisStim] + SimB[ThisStim]))

	res = np.sum(np.square(probs - data))
	return res

def resid_calc_exemp(x, stim, data, seen, last_seen=None, trial=None):
	global args
	sens = x[0]
	w1 = x[1]
	w2 = x[2]
	w3 = x[3]
	w4 = x[4]
	w5 = x[5]
	w6 = x[6]
	guessing = x[7]
	if args.response_scaling:
		gamma = x[8]
	if args.forgetting_function:
		assert last_seen is not None 
		assert trial is not None 
		beta = 1.4025  # average beta from the literature
	weights = np.array([w1, w2, w3, w4, w5, w6])
	SimA = 0
	SimB = 0
	
	for ThisTrainStim in range(NumAStim): #compare the stimulus to be classified with the As
		if ThisTrainStim not in seen:
			continue
		DistA = np.zeros(NumDim)
		for ThisDim in range(NumDim):
			DistA[ThisDim] = DistA[ThisDim] + (weights[ThisDim] * abs(Stim[stim,ThisDim] - Stim[ThisTrainStim,ThisDim]))
		if args.forgetting_function:
			assert last_seen[stim] >= 0
			SimA += (np.exp(-sens * sum(DistA)) * ((trial - last_seen[stim] + 1) ** -beta))
		else:
			SimA += np.exp(-sens * sum(DistA))
		
	for ThisTrainStim in range(NumAStim, NumTrainStim): #compare the stimulus to be classified with the Bs
		if ThisTrainStim not in seen:
			continue
		DistB = np.zeros(NumDim)
		for ThisDim in range(NumDim):
			DistB[ThisDim] = DistB[ThisDim] + (weights[ThisDim] * abs(Stim[stim,ThisDim] - Stim[ThisTrainStim,ThisDim]))
		if args.forgetting_function:
			assert last_seen[stim] >= 0
			SimB += (np.exp(-sens * sum(DistB)) * ((trial - last_seen[stim] + 1) ** -beta))
		else:
			SimB += np.exp(-sens * sum(DistB))
	
	prob = 0
	if args.response_scaling:
		prob = (guessing / 2) + ((1 - guessing) * (SimA ** gamma) / ((SimA ** gamma) + (SimB ** gamma)))	
	else:
		prob = (guessing / 2) + ((1 - guessing) * SimA / (SimA + SimB))
	return np.square((prob - data))

def resid_calc_proto(x, stim, data):
	sens = x[0]
	w1 = x[1]
	w2 = x[2]
	w3 = x[3]
	w4 = x[4]
	w5 = x[5]
	w6 = x[6]
	guessing = x[7]
	weights = np.array([w1, w2, w3, w4, w5, w6])

	DistA = np.zeros(NumDim)
	DistB = np.zeros(NumDim)
	for ThisDim in range(NumDim):
		DistA[ThisDim] = (weights[ThisDim]) * abs((Stim[stim,ThisDim] - A_Prot[ThisDim]))
		DistB[ThisDim] = (weights[ThisDim]) * abs((Stim[stim,ThisDim] - B_Prot[ThisDim]))
		
	SimA = np.exp(-sens * sum(DistA))
	SimB = np.exp(-sens * sum(DistB))
	prob = (guessing / 2) + ((1 - guessing) * SimA / (SimA + SimB))
	return np.square((prob - data))


parser = argparse.ArgumentParser(prog='prototype_exemplar_models', epilog='')
parser.add_argument('condition', choices=['control', 'experimental'], help='either "control" or "experimental"')
parser.add_argument('-r', '--response_scaling', help='whether to use response-scaling parameter', action='store_true')
parser.add_argument('-f', '--forgetting_function', help='whether to use forgetting function', action='store_true')

args = parser.parse_args()

assert args.condition == 'control' or args.condition == 'experimental'

randints = []
with open("../helper/random_integers.pickle", "rb") as f:
	randints = pickle.load(f)

NumDim  = 6  #the number of dimenions or features
NumAStim = 7 #the number of category A exemplars
NumBStim = 7 #the number of category B exemplars
NumTrainStim = 14 #The total number of training exemplars
NumTotalStim = 14 #the total number of stimuli in the problem set

Stim = np.loadtxt("../helper/6dNLSStim_2.txt",  delimiter=" ")
A_Prot=np.array([0,0,0,0,0,0])
B_Prot=np.array([1,1,1,1,1,1])

responses = []
results_exemp = [[], [], [], [], [], [], [], [], [], [], []]
results_proto = [[], [], [], [], [], [], [], [], [], [], []]

with open("../../data/experiment1_data.csv") as fp:
	reader = csv.reader(fp)
	for row in reader:
		responses.append([int(x) for x in row])

subj = 0
for response in responses:
	csv_num = response[0]
	if args.condition == 'experimental' and csv_num % 2 == 0:
		continue
	if args.condition == 'control' and csv_num % 2 != 0:
		continue
	stim_list = []
	filename = '../../stimuli/stim_list_' + str(csv_num) + '.csv'
	with open(filename, 'r') as f:
		for rec in csv.reader(f, delimiter=","):
			for x in rec:
				stim_list.append(int(x))
	
	counts = [0] * 14
	totals = [0] * 14
	vals = response[1:len(response)]
	fitted_proto = []
	fitted_exemp = []
	stim_indices = {k: [] for k in range(14)}
	first_seen = [-1] * 14
	last_seen = [-1] * 14
	last_seen_history = []
	for start_trial in range(0, 616, 56):
		stim_seen = set()
		counts = [0] * 14
		totals = [0] * 14
		for i in range(start_trial, start_trial + 56):
			if first_seen[stim_list[i]] == -1:
				first_seen[stim_list[i]] = i
			if last_seen[stim_list[i]] == -1:
				last_seen[stim_list[i]] = i
			last_seen_history.append(deepcopy(last_seen))
			last_seen[stim_list[i]] = i
			totals[stim_list[i]] += 1
			if vals[i] == 1:
				counts[stim_list[i]] += 1
			stim_seen.add(stim_list[i])
			stim_indices[stim_list[i]].append(i)
		data = np.array([count / total if total > 0 else 0 for count, total in zip(counts, totals)])
		for model in range(2):
			curr_min = sys.maxsize
			curr_argmin = []
			for iteration in range(20):
				w1 = np.random.randint(0,101)
				w2 = np.random.randint(0,(101-w1))
				w3 = np.random.randint(0,(101-(w1+w2)))
				w4 = np.random.randint(0,(101-(w1+w2+w3)))
				w5 = np.random.randint(0,(101-(w1+w2+w3+w4)))
				w6 = 100-(w1+w2+w3+w4+w5) 
				w = np.array([w1,w2,w3,w4,w5,w6])
				np.random.shuffle(w) 
				weights = w/100.0
				params = np.insert(weights, 0, np.random.randint(0,201) / 10.0)
				params = np.append(params, np.random.randint(0, 101) / 100.0)
				if model == 0:
					if args.response_scaling:
						params = np.append(params, np.random.randint(0, 101) / 10.0)
						res = minimize(resid_exemp_response_scaling, params, args=(data, stim_seen), constraints=cons, bounds=bounds_response_scaling)
					else:
						res = minimize(resid_exemp, params, args=(data, stim_seen), constraints=cons, bounds=bounds)
				else:
					res = minimize(resid_proto, params, args=(data, stim_seen), constraints=cons, bounds=bounds)
				if (res['fun'] < curr_min):
					curr_min = res['fun']
					curr_argmin = res['x']
			if model == 0:
				fitted_exemp.append(curr_argmin)
			else:
				fitted_proto.append(curr_argmin)
	bins = [[], [], [], [], [], [], [], [], [], [], []]
	for obj, trial_list in stim_indices.items():
		num_per_bin = len(trial_list) // 11
		remainder = len(trial_list) % 11
		allocations = [num_per_bin] * 11
		for j in range(0, remainder):
			allocations[randints[subj][j]] += 1
		curr_index = 0
		for alloc_index in range(0, len(allocations)):
			alloc = allocations[alloc_index]
			for num in range(0, alloc):
				bins[alloc_index].append(trial_list[curr_index])
				curr_index += 1
	bin_num = 0
	for bin in bins:
		res_exemp = 0
		res_proto = 0
		for trial in bin:
			seen = []
			for index in range(0, len(first_seen)):
				if trial >= first_seen[index]:
					seen.append(index)
			answer = 0
			if (vals[trial] == 1):
				answer = 1
			if args.forgetting_function:
				res_exemp += resid_calc_exemp(fitted_exemp[trial // 56], stim_list[trial], answer, seen, last_seen_history[trial], trial)
			else:
				res_exemp += resid_calc_exemp(fitted_exemp[trial // 56], stim_list[trial], answer, seen)
			res_proto += resid_calc_proto(fitted_proto[trial // 56], stim_list[trial], answer)
		results_exemp[bin_num].append(res_exemp / len(bin))
		results_proto[bin_num].append(res_proto /  len(bin))
		bin_num += 1
	
	subj += 1
	if subj % 5 == 0:
		print(f"{subj} participant models fitted")

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

print(graph_data_exemp)

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
plt.ylabel('Fit (MSE)', fontsize=15)
plt.ylim(0.06, 0.2)
plt.xticks(trial_segs)
plt.legend()
plt.show()