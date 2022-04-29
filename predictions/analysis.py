import numpy as np
from sklearn.metrics import r2_score
import os

for i in range(200):
	file = f"analysis{str(i)}.csv"
	try:
		data = np.loadtxt(file, delimiter=',')
	except:
		continue
	
	_pred = data[:, 0]
	target = data[:, 1]
	pred = _pred[_pred>0]
	target = target[_pred>0]
	print(f"Epoch {file.strip('analysis.csv'):3s}: {np.sum(_pred<=0)} zeros, error ratio {np.mean(np.abs(pred-target)/target):.4f}, R2 {r2_score(pred, target):.4f}")