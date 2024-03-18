import numpy as np

def rq(a):
    m, n = a.shape
    e = np.eye(m)
    p = e[:, ::-1]
    
    q0, r0 = np.linalg.qr(p @ a[:, :m].T @ p)
    
    r = p @ r0.T @ p
    q = p @ q0.T @ p
    
    fix = np.diag(np.sign(np.diag(r)))
    r = r @ fix
    q = fix @ q
    
    if n > m:
        q = np.hstack((q, np.linalg.inv(r) @ a[:, m:n]))
    
    return r, q
