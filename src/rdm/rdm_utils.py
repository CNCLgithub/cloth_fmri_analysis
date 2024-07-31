import numpy as np
import matplotlib.pyplot as plt

def plot_upper_triangular_rdm(labels, full_rdm, title=''):
    n = len(labels)
    indices = np.triu_indices(n, k=1)                        
    rdm = np.zeros((n, n))
    rdm[indices] = full_rdm
    plt.figure(figsize=(10, 8))
    plt.imshow(rdm, interpolation='nearest', cmap='viridis')
    plt.colorbar(label='Dissimilarity')
    plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=45, ha='right')
    plt.yticks(ticks=np.arange(len(labels)), labels=labels)
    plt.title(title)

    for i in range(len(rdm)):
        for j in range(len(rdm[i])):
            plt.text(j, i, f'{rdm[i, j]:.2f}', ha='center', va='center', color='white' if rdm[i, j] > rdm.max() / 2 else 'black')
    plt.show()
    
    
def upper_rdm(rdm):
    n = len(rdm)
    return rdm[np.triu_indices(n, k=1)]
    
    
def build_rdm(data_dict, scenes, plot=False):
    rdm = np.zeros((len(scenes), len(scenes)))
    for i in range(len(scenes)):
        for j in range(i+1, len(scenes)):
            cur_key1 = f'{scenes[i]}-{scenes[j]}'
            cur_key2 = f'{scenes[j]}-{scenes[i]}'
            cur_val = (data_dict[cur_key1] + data_dict[cur_key2])/2.0
            rdm[i][j] = cur_val
    rdm = upper_rdm(rdm)  
    if plot:
        plot_upper_triangular_rdm(scenes, rdm, '')
    return rdm



