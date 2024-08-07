from copy import deepcopy
import numpy as np
from scipy.optimize import minimize
import csv
import pickle
import matplotlib.pyplot as plt
import sys
from scipy.stats import bernoulli
import argparse

def constraint(x):
	return x[6]+x[1]+x[2]+x[3]+x[4]+x[5] - 1

cons = ({'type': 'eq', 'fun': constraint})

bounds = ((0,20), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1))

all_params = []

def resid_exemp(x, args, stim_seen):
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
		if ThisStim not in stim_seen:
			SimA[ThisStim] = 0
			SimB[ThisStim] = 0
			continue

		for ThisTrainStim in range(NumAStim): #compare the stimulus to be classified with the As
			if ThisTrainStim not in stim_seen:
				continue
			DistA = np.zeros(NumDim)
			for ThisDim in range(NumDim):
				DistA[ThisDim] = DistA[ThisDim] + (weights[ThisDim]) * abs(Stim[ThisStim,ThisDim] - Stim[ThisTrainStim,ThisDim])
			SimA[ThisStim] = SimA[ThisStim] + np.exp(-sens * sum(DistA))
		
		for ThisTrainStim in range(NumAStim, NumTrainStim): #compare the stimulus to be classified with the Bs
			if ThisTrainStim not in stim_seen:
				continue
			DistB = np.zeros(NumDim)
			for ThisDim in range(NumDim):
				DistB[ThisDim] = DistB[ThisDim] + (weights[ThisDim]) * abs(Stim[ThisStim,ThisDim] - Stim[ThisTrainStim,ThisDim])
			SimB[ThisStim] = SimB[ThisStim] + np.exp(-sens * sum(DistB))

	for ThisStim in range(NumTotalStim):
		if ThisStim not in stim_seen:
			probs[ThisStim] = 0
		else:
			probs[ThisStim] =  SimA[ThisStim] / (SimA[ThisStim]  + SimB[ThisStim])
	
	res = np.sum(np.square(probs - data))
	return res


def resid_proto(x, args, stim_seen):
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
		if ThisStim not in stim_seen:
			SimA[ThisStim] = 0
			SimB[ThisStim] = 0
			continue
		DistA = np.zeros(NumDim)
		DistB = np.zeros(NumDim)

		for ThisDim in range(NumDim):
			DistA[ThisDim] = (weights[ThisDim]) * abs((Stim[ThisStim,ThisDim] - A_Prot[ThisDim]))
			DistB[ThisDim] = (weights[ThisDim]) * abs((Stim[ThisStim,ThisDim] - B_Prot[ThisDim]))
		
		SimA[ThisStim] = np.exp(-sens * sum(DistA))
		SimB[ThisStim] = np.exp(-sens * sum(DistB))

	for ThisStim in range(NumTotalStim):
		if ThisStim not in stim_seen:
			probs[ThisStim] = 0
		else:
			probs[ThisStim] = SimA[ThisStim] / (SimA[ThisStim] + SimB[ThisStim])
	
	res = np.sum(np.square(probs - data))
	return res

def gen_sample_exemp(x, stim, seen, last_seen):
	sens = x[0]
	w1 = x[1]
	w2 = x[2]
	w3 = x[3]
	w4 = x[4]
	w5 = x[5]
	w6 = x[6]
	weights = np.array([w1, w2, w3, w4, w5, w6])
	SimA = 0
	SimB = 0
	
	for ThisTrainStim in range(NumAStim): #compare the stimulus to be classified with the As
		if ThisTrainStim not in seen:
			continue
		DistA = np.zeros(NumDim)
		for ThisDim in range(NumDim):
			DistA[ThisDim] = DistA[ThisDim] + (weights[ThisDim]) * abs(Stim[stim,ThisDim] - Stim[ThisTrainStim,ThisDim])
		assert last_seen[stim] >= 0
		SimA += (np.exp(-sens * sum(DistA)))
		
	for ThisTrainStim in range(NumAStim, NumTrainStim): #compare the stimulus to be classified with the Bs
		if ThisTrainStim not in seen:
			continue
		DistB = np.zeros(NumDim)
		for ThisDim in range(NumDim):
			DistB[ThisDim] = DistB[ThisDim] + (weights[ThisDim]) * abs(Stim[stim,ThisDim] - Stim[ThisTrainStim,ThisDim])
		assert last_seen[stim] >= 0
		SimB += (np.exp(-sens * sum(DistB)))
		
	prob = (SimA / (SimA + SimB))
	return 2 - bernoulli.rvs(prob)

def gen_sample_proto(x, stim):
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
	return 2 - bernoulli.rvs(prob)

parser = argparse.ArgumentParser(prog='prototype_exemplar_models', epilog='')
parser.add_argument('condition', choices=['control', 'experimental'], help='either "control" or "experimental"')
args = parser.parse_args()

condition = sys.argv[1]
assert condition in ('control', 'experimental')

indicator = 0
if condition == 'experimental':
	indicator = 1

randints = []
with open("../helper/random_integers.pickle", "rb") as f:
	randints = pickle.load(f)

NumDim  = 6  #the number of dimenions or features
NumAStim = 7 #the number of category A exemplars
NumBStim = 7 #the number of category B exemplars
NumTrainStim = 14 #The total number of training exemaplrs
NumTotalStim = 14 #the total number of stimuli in the problem set

Stim = np.loadtxt("../helper/6dNLSStim_2.txt",  delimiter=" ")
A_Prot=np.array([0,0,0,0,0,0])
B_Prot=np.array([1,1,1,1,1,1])

responses = []
exemp_accuracy = [[[], [], [], [], [], [], [], [], [], [], [], [], [], []], [[], [], [], [], [], [], [], [], [], [], [], [], [], []]]
proto_accuracy = [[[], [], [], [], [], [], [], [], [], [], [], [], [], []], [[], [], [], [], [], [], [], [], [], [], [], [], [], []]]
empirical_accuracy = [[[], [], [], [], [], [], [], [], [], [], [], [], [], []], [[], [], [], [], [], [], [], [], [], [], [], [], [], []]]

with open("../../data/experiment1_data.csv") as fp:
	reader = csv.reader(fp)
	for row in reader:
		responses.append([int(x) for x in row])

subj = 0
for response in responses:
	csv_num = response[0]
	if (csv_num - 100) % 2 != indicator:
		continue
	stim_list = []
	filename = '../../stimuli/stim_list_' + str(csv_num) + '.csv'
	with open(filename, 'r') as f:
		for rec in csv.reader(f, delimiter=","):
			for x in rec:
				stim_list.append(int(x))
	
	vals = response[1:len(response)]
	fitted_exemp = []
	fitted_proto = []
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
			for iteration in range(0, 10):
				'''
				if start_trial not in (0, 56, 560):
					break
				'''
				w1 = np.random.randint(0,101)
				w2 = np.random.randint(0,(101-w1))
				w3 = np.random.randint(0,(101-(w1+w2)))
				w4 = np.random.randint(0,(101-(w1+w2+w3)))
				w5 = np.random.randint(0,(101-(w1+w2+w3+w4)))
				w6 = 100-(w1+w2+w3+w4+w5) 
				w = np.array([w1,w2,w3,w4,w5,w6])
				np.random.shuffle(w) 
				weights = w/100.0
				params = np.insert(weights, 0, np.random.randint(0,201) / 10)
				if model == 0:
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

	segs = [bins[1], bins[10]]
	for bin in segs:
		num1_exemp = [0] * 14
		num1_proto = [0] * 14
		num1_empir = [0] * 14
		num_total = [0] * 14
		for trial in bin:
			seen = []
			for index in range(0, len(first_seen)):
				if trial >= first_seen[index]:
					seen.append(index)
			answer = 1
			if (vals[trial] == 2):
				answer = 2
			sample_exemp = gen_sample_exemp(fitted_exemp[trial // 56], stim_list[trial], seen, last_seen_history[trial])
			sample_proto = gen_sample_proto(fitted_proto[trial // 56], stim_list[trial])
			if answer == 1:
				num1_empir[stim_list[trial]] += 1
			if sample_exemp == 1:
				num1_exemp[stim_list[trial]] += 1
			if sample_proto == 1:
				num1_proto[stim_list[trial]] += 1
			num_total[stim_list[trial]] += 1
		for stim_num in range(0, 14):
			empirical_accuracy[bin_num][stim_num].append(num1_empir[stim_num] / num_total[stim_num])
			exemp_accuracy[bin_num][stim_num].append(num1_exemp[stim_num] / num_total[stim_num])
			proto_accuracy[bin_num][stim_num].append(num1_proto[stim_num] / num_total[stim_num])
		bin_num += 1
	
	subj += 1
	if subj % 5 == 0:
		print(f"{subj} participant models fitted")

WIDTH = 0.2
plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.serif": ["Palatino"]})
for j in range(0, 2):
	prototype = []
	exemplar = []
	observed = []
	prototype_err = []
	exemplar_err = []
	observed_err = []

	for stim in range(0, 14):
		prototype.append(sum(proto_accuracy[j][stim]) / len(proto_accuracy[j][stim]))
		exemplar.append(sum(exemp_accuracy[j][stim]) / len(exemp_accuracy[j][stim]))
		observed.append(sum(empirical_accuracy[j][stim]) / len(empirical_accuracy[j][stim]))

		prototype_err.append(np.std(proto_accuracy[j][stim]) / np.sqrt(len(proto_accuracy[j][stim])))
		exemplar_err.append(np.std(exemp_accuracy[j][stim]) / np.sqrt(len(exemp_accuracy[j][stim])))
		observed_err.append(np.std(empirical_accuracy[j][stim]) / np.sqrt(len(empirical_accuracy[j][stim])))

	plt.clf()

	labels = range(1,15)
	x = np.arange(len(labels))

	fig, ax = plt.subplots()
	rects1 = ax.bar(x - WIDTH, prototype, WIDTH, yerr=prototype_err, label='Prototype')
	rects2 = ax.bar(x, exemplar, WIDTH, yerr=exemplar_err, label='Exemplar')
	rects3 = ax.bar(x + WIDTH, observed, WIDTH, yerr=observed_err, label='Observed')

	ax.set_xlabel('Stimulus')
	ax.set_ylabel('Proportion of responses for category 1')
	if j == 0:
		ax.set_title('Early trial segment')
	else:
		ax.set_title('Late trial segment')
	ax.set_xticks(x)
	ax.set_ylim([0, 1])
	ax.set_xticklabels(labels)
	ax.legend(bbox_to_anchor=(1.2, 1.05))

	fig.tight_layout()
	plt.show()