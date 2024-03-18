import numpy as np
import matplotlib.pyplot as plt

def plotcams(P, fig):
    # Plots the principal axes for a set of cameras.
    # P is a list containing all the cameras.
    # P[i] is a 3x4 matrix representing camera i.

    c = np.zeros((4, len(P)))
    v = np.zeros((3, len(P)))

    for i in range(len(P)):
        _, _, Vt = np.linalg.svd(P[i])
        c[:, i] = Vt[-1, :]
        v[:, i] = np.sign(np.linalg.det(P[i][:, :3])) * P[i][2, :3]
        v[:, i] /= np.linalg.norm(v[:, i])

    c /= c[3, :]

    
    fig.quiver(c[0], c[1], c[2], v[0], v[1], v[2], color='r', linestyle='-', linewidth=1.5, arrow_length_ratio=0.1)
    fig.set_xlabel('X')
    fig.set_ylabel('Y')
    fig.set_zlabel('Z')