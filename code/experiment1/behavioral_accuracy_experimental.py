# Usage: python behavioral_accuracy_experimental.py

import csv
import matplotlib.pyplot as plt
import numpy as np
import pickle

responses = []
results = [[], [], [], [], [], [], [], [], [], [], []]
NUM_TRIALS = 616
NUM_TRIALS_PER_BIN = 56
NUM_STIMULI = 14
NUM_TRIAL_SEG = 11

# load all participant reponses
with open("../../data/experiment1_data.csv") as fp:
	reader = csv.reader(fp)
	for row in reader:
		responses.append([int(x) for x in row])

# load the random ints used to divide up the remainder when partitioning trial segments for the experimental condition
randints = []
with open("../helper/random_integers.pickle", "rb") as f:
	randints = pickle.load(f)

subj = 0
for response in responses:
	csv_num = response[0]
	# skip participants who received the control condition
	if csv_num % 2 == 0:
		continue
	stim_list = []
	filename = '../../stimuli/stim_list_' + str(csv_num) + '.csv'
	with open(filename, 'r') as f:
		for rec in csv.reader(f, delimiter=","):
			for x in rec:
				stim_list.append(int(x))

	vals = response[1:len(response)]
	stim_indices = {k: [] for k in range(NUM_STIMULI)}
	for trial in range(0, NUM_TRIALS):
		stim_indices[stim_list[trial]].append(trial)

	# make the 11 bins, depending on where the object is along the power law curve
	bins = [[], [], [], [], [], [], [], [], [], [], []]
	for obj, trial_list in stim_indices.items():
		num_per_bin = len(trial_list) // NUM_TRIAL_SEG
		remainder = len(trial_list) % NUM_TRIAL_SEG
		allocations = [num_per_bin] * NUM_TRIAL_SEG
		for j in range(0, remainder):
			allocations[randints[subj][j]] += 1
		curr_index = 0
		for alloc_index in range(0, len(allocations)):
			alloc = allocations[alloc_index]
			for num in range(0, alloc):
				bins[alloc_index].append(trial_list[curr_index])
				curr_index += 1
	
	# compute the participant accuracy in each bin/trial segment
	bin_num = 0
	for bin in bins:
		num_correct = 0
		for trial in bin:
			if (stim_list[trial] < 7 and vals[trial] == 1) or (stim_list[trial] >= 7 and vals[trial] == 2):
				num_correct += 1
			
		results[bin_num].append(num_correct / len(bin))
		bin_num += 1

	subj += 1

# Graph the results
graph_data = []
errors = []
for result in results:
	graph_data.append(sum(result)/len(result))
	errors.append(np.std(result) / np.sqrt(len(result)))

trial_segs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
	"font.size": 13
})

# Shade the area between y1 and y2
y1 = [val - err for val, err in zip(graph_data, errors)]
y2 = [val + err for val, err in zip(graph_data, errors)]

plt.fill_between(trial_segs, y1, y2, facecolor="grey", color='grey', alpha=0.2)

plt.plot(trial_segs, graph_data,'k')

plt.xlabel('Trial Segment', fontsize=15)
plt.ylabel('Average Accuracy', fontsize=15)
plt.ylim(0.6, 0.95)
plt.xticks(trial_segs)
plt.show()