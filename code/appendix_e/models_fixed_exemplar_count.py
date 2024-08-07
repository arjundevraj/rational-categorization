import numpy as np
from scipy.optimize import minimize
import csv
import pickle
import matplotlib.pyplot as plt
import sys
import copy

def constraint(x):
	return x[1]+x[2]+x[3]+x[4]+x[5]+x[6] - 1

cons = ({'type': 'eq', 'fun': constraint})

NUM_TRIALS = 616
NUM_TRIAL_SEGS = 11

bounds = ((0,20), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1))

def resid_exemp(x, *args):
	data = args[0]
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

	for ThisStim in range(NumTrainStim):
		probs[ThisStim] = SimA[ThisStim] / (SimA[ThisStim] + SimB[ThisStim])
	
	res = np.sum(np.square(probs - data))
	return res

def resid_proto(x, *args):
	data = args[0]
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

def resid_calc_exemp(x, stim, data, seen):
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
		SimA += np.exp(-sens * sum(DistA))
		
	for ThisTrainStim in range((NumAStim), NumTrainStim): #compare the stimulus to be classified with the Bs
		if ThisTrainStim not in seen:
			continue
		DistB = np.zeros(NumDim)
		for ThisDim in range(NumDim):
			DistB[ThisDim] = DistB[ThisDim] + (weights[ThisDim]) * abs(Stim[stim,ThisDim] - Stim[ThisTrainStim,ThisDim])
		SimB += np.exp(-sens * sum(DistB))
		
	prob = SimA / (SimA + SimB)
	return np.square((prob - data))

def resid_calc_proto(x, stim, data):
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
	return np.square((prob - data))

randints = []
with open("../helper/random_integers.pickle", "rb") as f:
	randints = pickle.load(f)

NumDim  = 6  #the number of dimenions or features
NumAStim = 7 #the number of category A exemplars
NumBStim = 7 #the number of category B exemplars
NumTrainStim = 14 #The total number of training exemplars
NumTotalStim = 14 #the total number of stimuli in the problem set

Stim = np.loadtxt("../helper/6dNLSStim_4.txt",  delimiter=" ")
A_Prot=np.array([0,0,0,0,0,0])
B_Prot=np.array([1,1,1,1,1,1])

responses = []
template = [[], [], [], [], [], [], [], [], [], [], []]
results_exemp = []
results_proto = []

for i in range(NumTotalStim):
	results_exemp.append(copy.deepcopy(template))
	results_proto.append(copy.deepcopy(template))

with open("../../data/experiment2_data.csv") as fp:
	reader = csv.reader(fp)
	for row in reader:
		responses.append([int(x) for x in row])

subj = 0
for response in responses:
	csv_num = response[0]
	# Only care about the experimental condition with 4 exceptions
	if (csv_num - 100) % 6 != 5:
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
	stim_indices = {k: [] for k in range(14)}
	first_seen = [-1] * 14
	for trial in range(0, NUM_TRIALS):
		if first_seen[stim_list[trial]] == -1:
			first_seen[stim_list[trial]] = trial
		totals[stim_list[trial]] += 1
		stim_indices[stim_list[trial]].append(trial)
		if vals[trial] == 1:
			counts[stim_list[trial]] += 1
	list_data = []
	for count, total in zip(counts, totals):
		if total == 0:
			list_data.append(0)
		else:
			list_data.append(count / total)
	
	data = np.array(list_data)
	curr_min_exemp = sys.maxsize
	curr_argmin_exemp = []
	curr_min_proto = sys.maxsize
	curr_argmin_proto = []
	for model in range(2):
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
				res = minimize(resid_exemp, params, args=data, constraints=cons, bounds=bounds)
			else:
				res = minimize(resid_proto, params, args=data, constraints=cons, bounds=bounds)
			if model == 0 and (res['fun'] < curr_min_exemp):
				curr_min_exemp = res['fun']
				curr_argmin_exemp = res['x']
			elif model == 1 and (res['fun'] < curr_min_proto):
				curr_min_proto = res['fun']
				curr_argmin_proto = res['x']

	# make the 11 bins, depending on where the object is along the power law curve
	bins = [[], [], [], [], [], [], [], [], [], [], []]
	for obj, trial_list in stim_indices.items():
		num_per_bin = len(trial_list) // NUM_TRIAL_SEGS
		remainder = len(trial_list) % NUM_TRIAL_SEGS
		allocations = [num_per_bin] * NUM_TRIAL_SEGS
		for j in range(0, remainder):
			allocations[randints[subj][j]] += 1
		curr_index = 0
		for alloc_index in range(0, len(allocations)):
			alloc = allocations[alloc_index]
			for num in range(0, alloc):
				bins[alloc_index].append(trial_list[curr_index])
				curr_index += 1
	
	sorted_stimuli = sorted(enumerate(first_seen), key=lambda i: i[1])
	bin_num = 0
	for bin in bins:
		tmp_proto = [[], [], [], [], [], [], [], [], [], [], [], [], [], []]
		tmp_exemp = [[], [], [], [], [], [], [], [], [], [], [], [], [], []]
		for trial in bin:
			seen = []
			for index in range(0, len(first_seen)):
				if trial >= first_seen[index]:
					seen.append(index)
			answer = 0
			if (vals[trial] == 1):
				answer = 1
			res_exemp = resid_calc_exemp(curr_argmin_exemp, stim_list[trial], answer, seen)
			res_proto = resid_calc_proto(curr_argmin_proto, stim_list[trial], answer)
			tmp_exemp[len(seen) - 1].append(res_exemp)
			tmp_proto[len(seen) - 1].append(res_proto)
		for i in range(NumTotalStim):
			if len(tmp_proto[i]) > 0:
				results_exemp[i][bin_num].append((sum(tmp_exemp[i]) / len(tmp_exemp[i])))
				results_proto[i][bin_num].append((sum(tmp_proto[i]) / len(tmp_proto[i])))
			else:
				results_exemp[i][bin_num].append(-1)
				results_proto[i][bin_num].append(-1)
		bin_num += 1
	subj += 1
	if subj % 5 == 0:
		print(f"{subj} participant models fitted")

trial_segs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
	"font.size": 13
})

agg_proto = [[], [], [], [], [], [], [], [], [], [], []]
agg_exemp = [[], [], [], [], [], [], [], [], [], [], []]
for i in range(0, NumTotalStim):
	results_proto_cleaned = []
	for result in results_proto[i]:
		results_proto_cleaned.append([x for x in result if x > 0])
	
	results_exemp_cleaned = []
	for result in results_exemp[i]:
		results_exemp_cleaned.append([x for x in result if x > 0])

	graph_data_proto = []
	errors_proto = []
	for result in results_proto_cleaned:
		if len(result) > 0:
			graph_data_proto.append(sum(result)/len(result))
			errors_proto.append(np.std(result) / np.sqrt(len(result)))
		else:
			# this is just dummy value to make graph work -- we should ignore these points
			graph_data_proto.append(0)
			errors_proto.append(0)
	for idx, result in enumerate(results_proto_cleaned):
		agg_proto[idx].extend(result)

	graph_data_exemp = []
	errors_exemp = []
	for result in results_exemp_cleaned:
		if len(result) > 0:
			graph_data_exemp.append(sum(result)/len(result))
			errors_exemp.append(np.std(result) / np.sqrt(len(result)))
		else:
			# this is just dummy value to make graph work -- we should ignore these points
			graph_data_exemp.append(0)
			errors_exemp.append(0)
	for idx, result in enumerate(results_exemp_cleaned):
		agg_exemp[idx].extend(result)
	
	if i < 3: continue

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
	plt.title(f"{i+1} Stimuli Encountered")
	plt.ylim(0, 0.3)
	plt.xticks(trial_segs)
	plt.legend()
	plt.show()


graph_data_proto = []
errors_proto = []
for result in agg_proto:
	cleaned_result = [x for x in result if x > 0]
	graph_data_proto.append(sum(cleaned_result)/len(cleaned_result))
	errors_proto.append(np.std(cleaned_result) / np.sqrt(len(cleaned_result)))

graph_data_exemp = []
errors_exemp = []
for result in agg_exemp:
	cleaned_result = [x for x in result if x > 0]
	graph_data_exemp.append(sum(cleaned_result)/len(cleaned_result))
	errors_exemp.append(np.std(cleaned_result) / np.sqrt(len(cleaned_result)))

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
plt.title("Average")
plt.ylim(0, 0.3)
plt.xticks(trial_segs)
plt.legend()
plt.show()