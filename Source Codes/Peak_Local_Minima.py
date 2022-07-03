import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

finaldata1 = []
finaldata2 = []
total = []

xs=pd.read_excel('/Users/sahanaprasad/Desktop/fft_30cm_0deg_3cm_mit_0.6mm_.xlsx')


df = pd.DataFrame(xs, columns=['data'])

# Find local peaks & local minima
df['max'] = df.data[(df.data.shift(1) < df.data) & (df.data.shift(-1) < df.data)]
df['min'] = df.data[(df.data.shift(1) > df.data) & (df.data.shift(-1) > df.data)]

# Plot results
plt.scatter(df.index, df['min'], c='r')
plt.scatter(df.index, df['max'], c='g')
plt.plot(df.index, df['data'])
plt.show()

final1 = df['max'].dropna()
final2 = df['min'].dropna()

#Convert the data to string format
finaldata1 = final1.to_string()
finaldata2 = final2.to_string()

#Print the values on terminal
print('----------Maxima----------')
print(finaldata1)
print('----------Minima----------')
print(finaldata2)
