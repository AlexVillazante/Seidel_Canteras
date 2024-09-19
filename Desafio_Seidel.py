import numpy as np

# Definimos el número de iteraciones y el nivel de tolerancia
MAX_ITER = 100
TOL = 1e-5

# Coeficientes de las ecuaciones
A = np.array([[0.52, 0.2, 0.25],
              [0.3, 0.5, 0.2],
              [0.18, 0.2, 0.55]])

# Términos independientes
b = np.array([4800, 5810, 5690])

# Valores iniciales (pueden ser ceros o un valor estimado)
x = np.zeros(3)

def gauss_seidel(A, b, x, tol=TOL, max_iter=MAX_ITER):
    n = len(b)
    for iteration in range(max_iter):
        x_old = np.copy(x)
        
        for i in range(n):
            sum1 = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x[i] = (b[i] - sum1) / A[i][i]
        
        # Criterio de parada
        error = np.linalg.norm(x - x_old, ord=np.inf)
        if error < tol:
            print(f'Converged after {iteration+1} iterations')
            return x
    
    print(f'Did not converge after {max_iter} iterations')
    return x

# Llamamos a la función para resolver el sistema
solution = gauss_seidel(A, b, x)

# Mostramos los resultados
print(f"Solución: x1 = {solution[0]:.5f}, x2 = {solution[1]:.5f}, x3 = {solution[2]:.5f}")
