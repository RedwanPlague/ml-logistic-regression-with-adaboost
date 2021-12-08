import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

print(np.eye(5))

df = pd.DataFrame(np.random.randn(3, 4))
print(df)

sc = StandardScaler()
print(sc)

a = np.linspace(0, 2*np.pi, 100)
b = np.sin(a)
plt.plot(a, b)
plt.show()
