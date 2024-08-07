# Usage: python behavioral_accuracy.py

import csv
import pickle
import numpy as np
import matplotlib.pyplot as plt

responses = []
results = []
NUM_STIMULI = 14
NUM_TRIAL_SEGS = 11

# create results list for each of the 6 conditions
for i in range(0, 6):
	results.append([[], [], [], [], [], [], [], [], [], [], []])

# load the random ints used to divide up the remainder when partitioning trial segments for the experimental condition
randints = []
with open("../helper/random_integers.pickle", "rb") as f:
	randints = pickle.load(f)

exclude = [448, 534, 196, 163]  # exclude these 4 that did not complete the task properly according to Prolific (this also ensures 60 per group)

# load all participant reponses
with open("../../data/experiment2_data.csv", "r") as fp:
	reader = csv.reader(fp)
	for row in reader:
		if int(row[0]) not in exclude:
			responses.append([int(x) for x in row])

subj = [0, 0, 0, 0, 0, 0]
for response in responses:
	csv_num = response[0]
	stim_list = []
	filename = '../../stimuli/stim_list_' + str(csv_num) + '.csv'
	with open(filename, 'r') as f:
		for rec in csv.reader(f, delimiter=","):
			for x in rec:
				stim_list.append(int(x))

	vals = response[1:len(response)]
	stim_indices = {k: [] for k in range(NUM_STIMULI)}
	for trial in range(0, 616):
		stim_indices[stim_list[trial]].append(trial)

	# make the 11 bins, depending on where the object is along the power law curve
	bins = [[], [], [], [], [], [], [], [], [], [], []]
	for obj, trial_list in stim_indices.items():
		num_per_bin = len(trial_list) // NUM_TRIAL_SEGS
		remainder = len(trial_list) % NUM_TRIAL_SEGS
		allocations = [num_per_bin] * NUM_TRIAL_SEGS
		for j in range(0, remainder):
			allocations[randints[subj[(csv_num - 100) % 6]][j]] += 1
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
			
		results[(csv_num - 100) % 6][bin_num].append(num_correct / len(bin))  # (csv_num - 100) % 6 is the indicator for which of the 6 conditions participant belongs to
		bin_num += 1

	subj[(csv_num - 100) % 6] += 1

# Graph the results
graph_data = []
errors = []
for condition in results:
	graph_data_cond = []
	errors_cond = []
	for result in condition:
		graph_data_cond.append(sum(result) / len(result))
		errors_cond.append(np.std(result) / np.sqrt(len(result)))
	graph_data.append(graph_data_cond)
	errors.append(errors_cond)

trial_segs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
	"font.size": 13
})

y1 = [val - err for val, err in zip(graph_data[2], errors[2])]
y2 = [val + err for val, err in zip(graph_data[2], errors[2])]
plt.fill_between(trial_segs, y1, y2, facecolor="blue", color='blue', alpha=0.2)
plt.plot(trial_segs, graph_data[2], 'blue', label="0 exceptions")

y1 = [val - err for val, err in zip(graph_data[4], errors[4])]
y2 = [val + err for val, err in zip(graph_data[4], errors[4])]
plt.fill_between(trial_segs, y1, y2, facecolor="grey", color='grey', alpha=0.2)
plt.plot(trial_segs, graph_data[4], 'black', label="2 exceptions")

y1 = [val - err for val, err in zip(graph_data[0], errors[0])]
y2 = [val + err for val, err in zip(graph_data[0], errors[0])]
plt.fill_between(trial_segs, y1, y2, facecolor="orange", color='orange', alpha=0.2)
plt.plot(trial_segs, graph_data[0], 'orange', label="4 exceptions")

plt.legend(loc='upper left')
plt.xlabel('Trial Segment', fontsize=15)
plt.ylabel('Average Accuracy', fontsize=15)
plt.xticks(trial_segs)
plt.ylim(0.5, 1)
plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
plt.show()

y1 = [val - err for val, err in zip(graph_data[1], errors[1])]
y2 = [val + err for val, err in zip(graph_data[1], errors[1])]
plt.fill_between(trial_segs, y1, y2, facecolor="blue", color='blue', alpha=0.2)
plt.plot(trial_segs, graph_data[1], 'blue', label="0 exceptions")

y1 = [val - err for val, err in zip(graph_data[3], errors[3])]
y2 = [val + err for val, err in zip(graph_data[3], errors[3])]
plt.fill_between(trial_segs, y1, y2, facecolor="grey", color='grey', alpha=0.2)
plt.plot(trial_segs, graph_data[3], 'black', label="2 exceptions")

y1 = [val - err for val, err in zip(graph_data[5], errors[5])]
y2 = [val + err for val, err in zip(graph_data[5], errors[5])]
plt.fill_between(trial_segs, y1, y2, facecolor="orange", color='orange', alpha=0.2)
plt.plot(trial_segs, graph_data[5], 'orange', label="4 exceptions")

plt.legend(loc='lower right')
plt.xlabel('Trial Segment', fontsize=15)
plt.ylabel('Average Accuracy', fontsize=15)
plt.xticks(trial_segs)
plt.ylim(0.5, 1)
plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
plt.show()