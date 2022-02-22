import matplotlib.pyplot as plt
# alpha = [[0.001,0.002],
#         [0.01,0.02],
#         [0.005,0.001]
#         ]
alpha = [0.1,0.5,1,2,4,8]
result = [80071.71352588024,80151.60514778423,78658.58505189412,77145.9199547423,74288.66366012223,69165.30754563105]

plt.plot(alpha,result)
plt.xlabel('Multiple of original commission rate')
plt.ylabel('Final asset (USD)')
plt.title("Final asset - commission rate")
plt.savefig('./results/sensiticity.png')