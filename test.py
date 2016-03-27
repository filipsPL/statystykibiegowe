#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import kendalltau
import seaborn as sns
sns.set(style="ticks")

rs = np.random.RandomState(11)
x = rs.gamma(2, size=1000)
y = -.5 * x + rs.normal(size=1000)

print x
print y

#sns.jointplot(x, y, kind="hex", stat_func=kendalltau, color="#4CB391")
#plt.show()