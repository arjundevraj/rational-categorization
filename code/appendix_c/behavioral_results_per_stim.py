import pickle
import matplotlib.pyplot as plt
import numpy as np
import csv
import pickle

randints = []
with open("../helper/random_integers.pickle", "rb") as f:
	randints = pickle.load(f)

responses = []
results = [[], [], [], [], [], [], [], [], [], [], []]

with open("../../data/experiment1_data.csv") as fp:
	reader = csv.reader(fp)
	for row in reader:
		responses.append([int(x) for x in row])

subj = 0
for response in responses:
	csv_num = response[0]
	if csv_num % 2 == 0:
		continue
	stim_list = []
	filename = '../../stimuli/stim_list_' + str(csv_num) + '.csv'
	with open(filename, 'r') as f:
		for rec in csv.reader(f, delimiter=","):
			for x in rec:
				stim_list.append(int(x))

	vals = response[1:len(response)]
	stim_indices = {k: [] for k in range(14)}
	for trial in range(0, 616):
		stim_indices[stim_list[trial]].append(trial)

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
		num_correct_per_stim = [0] * 14
		num_total_per_stim = [0] * 14
		for trial in bin:
			num_total_per_stim[stim_list[trial]] += 1
			if (stim_list[trial] < 7 and vals[trial] == 1) or (stim_list[trial] >= 7 and vals[trial] == 2):
				num_correct_per_stim[stim_list[trial]] += 1
			
		results[bin_num].append([i / j for i, j in zip(num_correct_per_stim, num_total_per_stim)])
		bin_num += 1

	subj += 1

graph_data = []
errors = []
for result in results:
	stim_data = []
	stim_errors = []
	for i in range(0, 14):
		temp = [internal[i] for internal in  result]
		stim_data.append(sum(temp)/len(temp))
		stim_errors.append(np.std(temp) / np.sqrt(len(temp)))
	graph_data.append(stim_data)
	errors.append(stim_errors)

trial_segs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# plt.style.use('grayscale')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
	"font.size": 13
})

label = None
linestyle = None
for i in range(0,14):
	temp = [internal[i] for internal in graph_data]
	if i == 6 or i == 13:
		label = "exception"
		linestyle = "dashed"
	else:
		label = "non-exception"
		linestyle = "solid"
	plt.plot(trial_segs, temp, label=label, color='black', linestyle=linestyle)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.xlabel('Trial Segment', fontsize=15)
plt.ylabel('Average Accuracy', fontsize=15)
plt.xticks(trial_segs)
plt.tight_layout()
plt.show()