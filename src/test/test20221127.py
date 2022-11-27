import numpy as np

position_embedding =np.array([
            [pos / np.power(10000, 2. * i / 10) for i in range(10)]
            for pos in range(50)])
print(position_embedding)
pos_2i = np.sin(position_embedding[:, 0::2])  # dim 2i
pos_2i_1 = np.cos(position_embedding[:, 1::2])  # dim 2i+1
print("********"*20)
