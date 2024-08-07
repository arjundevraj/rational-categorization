import pickle
import numpy as np

NUM_EXP_SUBJ = 60

output = []
for i in range(NUM_EXP_SUBJ):
	output.append(np.random.randint(11, size=11))

with open("random_integers_tmp.pickle", "wb") as fp:
	pickle.dump(output, fp)