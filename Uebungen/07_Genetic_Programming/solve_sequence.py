# adapted from https://github.com/DEAP/deap/blob/72c0bf56469781a76736aa0087424f9f5ccb443b/examples/gp/symbreg.py
#    This file is part of EAP.
#
#    EAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    EAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with EAP. If not, see <http://www.gnu.org/licenses/>.
#
# # Adapted by Thomas Bierweiler for Hochschule Karlsruhe, Data Science, DSCB450, Analyse von Prozess- und Produktdaten

# initial size of population
npop=10000
# number of generations
ngen=1000

import operator
import math
import random
import copy

import numpy

import sympy
from sympy.abc import x

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

# sequence for which we try and find the next member
seq=(237,474,711,948,234,471,708,945,231,468,705,942,228,465,702,939,225)

def convert_inverse_prim(prim, args):
    """Convert inverse prims.
    Convert inverse prims according to:
    [Dd]iv(a,b) -> Mul[a, 1/b]
    [Ss]ub(a,b) -> Add[a, -b]
    We achieve this by overwriting the corresponding format method of
    the sub and div prim.
    Parameters
    ----------
    prim : deap.gp.Terminal
        A DEAP primitive
    Returns
    -------
    :class: String
        The converted string
    """
    prim = copy.copy(prim)

    converter = {
        'sub': "Add({}, Mul(-1.0,{}))".format,
        'protectedDiv': "(({})/({}))".format,
        'mul': "Mul({},{})".format,
        'add': "Add({},{})".format,
        'pow': "Pow({},{})".format,
        'protectedModulus': "Mod({},{})".format,
    }
    prim_formatter = converter.get(prim.name, prim.format)

    return prim_formatter(*args)

def stringify_for_sympy(individual):
    """Return the expression in a human readable string.
    Parameters
    ----------
    individual : deap.gp.Individual
        A DEAP individual
    Returns
    -------
    :class: String
        The converted string
    """
    string = ""
    stack = []
    for node in individual:
        stack.append((node, []))
        while len(stack[-1][1]) == stack[-1][0].arity:
            prim, args = stack.pop()
            string = convert_inverse_prim(prim, args)
            if len(stack) == 0:
                break  # If stack is empty, all nodes should have been seen
            stack[-1][1].append(string)
    return string

# Define new functions
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

def protectedModulus(left, right):
    try:
        return left % right
    except ZeroDivisionError:
        return 1


pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
# pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedModulus, 2)
# pset.addPrimitive(operator.neg, 1)
# pset.addPrimitive(math.cos, 1)
# pset.addPrimitive(math.sin, 1)
pset.addEphemeralConstant("rand101", lambda: random.randint(1,1000))
pset.renameArguments(ARG0='x')

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def evalSymbReg(individual, points):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    # and the sequence that we search
    sqerrors=0.0
    for x in points:
        sqerrors+=(func(x)-seq[x-1])**2
    # sqerrors = ((func(x) - x**4 - x**3 - x**2 - x)**2 for x in points)
    return sqerrors / len(points),

points=[x for x in range(1,18)]
toolbox.register("evaluate", evalSymbReg, points=points)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

def main():
    random.seed(318)

    pop = toolbox.population(n=npop)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox,cxpb=0.5,mutpb=0.1,ngen=ngen, stats=mstats,
                                   halloffame=hof, verbose=True)
    # print log
    return pop, log, hof

if __name__ == "__main__":
    pop, log, hof=main()
     # print log
    print('Best results:')
    sympy_string=stringify_for_sympy(hof.items[0])
    print("Method generated by GP: {}".format(sympy_string))
    # calculate RMSE
    sqerrors=0.0
    func = toolbox.compile(expr=hof.items[0])
    sol_seq=[]
    for x in points:
        sol_seq.append(func(x))
        sqerrors+=(func(x)-seq[x-1])**2
    rmse=math.sqrt(sqerrors / len(points))
    print('RMSE: {}'.format(rmse))
    # print values of given sequence
    print('Given sequence: {}'.format(seq))
    # print values of "solution"
    print('Proposed sequence: {}'.format(sol_seq))
    # simplyfy term with symbolic toolbox
    sympy_term=sympy.simplify(sympy_string)
    print("Simplified method: {}".format(sympy_term))
