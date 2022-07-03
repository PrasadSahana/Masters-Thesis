import matplotlib.pyplot as plt
from CNN1D_testing import accuracy
from MLP_Testing import accuracy_percent


# creating the bar plot
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
data123 = ['CNN', 'MLP']
data234 = [accuracy, accuracy_percent]

ax.bar(data123,data234)

for i in range(len(data234)):
    plt.annotate(str(data234[i])+'%', xy=(data123[i],data234[i]), ha='center', va='bottom')
    
plt.title('Obtained accuracy of algorithms')
    
plt.show()
