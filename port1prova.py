from pyomo.environ import *

expret={}
std={}
corr={}

with open("instances/port1.txt", "r") as f:
    lines = [l.strip() for l in f.readlines() if l.strip()]

n = int(lines[0])
assets = range(1, n + 1)

for i, line in enumerate(lines[1:1 + n], start=1):
    a, b = line.split()
    expret[i] = float(a)
    std[i] = float(b)


for line in lines[1 + n:]:
    i, j, v = line.split()
    i = int(i)
    j = int(j)
    corr[(i, j)] = float(v)

cov={}
for i in assets:
    for j in assets:
        c_ij = corr.get((i, j), corr.get((j, i)))
        cov[(i, j)] = c_ij * std[i] * std[j]

import numpy as np

cov_matrix = np.array([
    [cov[(i, j)] for j in assets]
    for i in assets
])

print("Simétrica:", np.allclose(cov_matrix, cov_matrix.T))

eigenvalues = np.linalg.eigvalsh(cov_matrix)

print("Autovalor mínimo:", eigenvalues.min())
print("Autovalor máximo:", eigenvalues.max())

def solve_for_Rmin(Rmin):
    model = ConcreteModel()

    model.I = Set(initialize=assets)

    model.x = Var(model.I, domain=NonNegativeReals)

    model.budget = Constraint(expr=sum(model.x[i] for i in model.I) == 1)
    model.return_min = Constraint(expr=sum(expret[i] * model.x[i] for i in model.I) >= Rmin)

    def risk_expr(m):
        return sum(cov[(i, j)] * m.x[i] * m.x[j] for i in m.I for j in m.I)

    model.risk = Objective(rule=risk_expr, sense=minimize)

    solver = SolverFactory("highs")
    solver.options["time_limit"] = 2
    result = solver.solve(model, tee=True)

    x_sol = {i: model.x[i].value for i in model.I}
    risk = value(model.risk)
    ret = sum(expret[i] * x_sol[i] for i in model.I)

    return x_sol, risk, ret

###################################################################################
###################################################################################

if __name__ == "__main__":
    
    import numpy as np

    

    

    
    Rmins = np.linspace(0.004, 0.0108, 100)
    my_frontier = []

    for Rmin in Rmins:

        print(f"\nResolviendo para Rmin = {Rmin}")

        x, risk, ret = solve_for_Rmin(Rmin)

        my_frontier.append((risk, ret))

        print(f"Return obtenido = {ret:.6f}")
        print(f"Riesgo = {risk:.8f}")

    import matplotlib.pyplot as plt

    risks = [risk for risk, ret in my_frontier]
    returns = [ret for risk, ret in my_frontier]

    plt.plot(risks, returns, marker="o")
    plt.xlabel("Risk (Variance)")
    plt.ylabel("Expected Return")
    plt.title("Efficient Frontier")
    plt.grid()
    plt.show()