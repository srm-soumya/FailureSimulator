import numpy as np
import pandas as pd
import random
from scipy.stats import expon
import matplotlib.pyplot as plt

def D1(thres,m,Ti=0):
  t=np.linspace(1,10000,10000)
  #m=np.zeros(10000)
  Tin=int(np.ceil(Ti))
  for i in range(Tin,10000):
    z=np.random.uniform(0.001,0.01)
    if i==0:
      m[i]=0
    else:
      m[i]=m[i-1]+np.exp(z*t[i])
    if m[i]>thres*100:
      return max(np.abs(t[i]-Ti)-np.random.randn(),np.random.uniform(1,30)),m
  return t[i]+np.random.randn(),m
def D2(thres,m,Ti=0):
  t=np.linspace(1,10000,10000)
  #m=np.zeros(10000)
  Tin=int(np.ceil(Ti))
  for i in range(Tin,10000):
    z=np.random.uniform(0.001,0.01)
    if i==0:
      m[i]=0
    else:
      m[i]=m[i-1]+np.log(1+z*t[i])
    if m[i]>thres:
      return max(np.abs(t[i]-Ti)-np.random.randn(),np.random.uniform(1,30)),m
  return t[i]+np.random.randn(),m
def D3(thres,m,Ti=0):
  t=np.linspace(1,10000,10000)
  #m=np.zeros(10000)
  Tin=int(np.ceil(Ti))
  for i in range(Tin,10000):
    z0=np.random.uniform(2,25)
    z1=np.random.uniform(0.1,0.8)
    z2=np.random.uniform(0.01,0.075)
    if i==0:
      m[i]=0
    else:
      m[i]=m[i-1]+(z0+z1*t[i]+z2*(t[i]**0.75))
      #m[i]=m[i-1]+np.log(1+z*t[i])
    if m[i]>thres*100:
      return max(np.abs(t[i]-Ti)-np.random.randn(),np.random.uniform(1,30)),m
  return t[i]+np.random.randn(),m
def D4(thres,m,Ti=0):
  t=np.linspace(1,10000,10000)
  #m=np.zeros(10000)
  Tin=int(np.ceil(Ti))
  for i in range(Tin,10000):
    z0=np.random.uniform(6,10)
    z1=np.random.uniform(0.08,0.32)
    if i==0:
      m[i]=0
    else:
      m[i]=m[i-1]+(z0+z1*(t[i]**0.95))
      #m[i]=m[i-1]+np.log(1+z*t[i])
    if m[i]>thres*10:
      return max(np.abs(t[i]-Ti)-np.random.randn(),np.random.uniform(1,30)),m
  return t[i]+np.random.randn(),m
def D5(thres,m,Ti=0):
  t=np.linspace(1,10000,10000)
  #m=np.zeros(10000)
  Tin=int(np.ceil(Ti))
  for i in range(Tin,10000):
    z=np.random.uniform(0.01,0.2)
    if i ==0:
      m[i]=10000 / (z*t[i] + 1)
    else:
      m[i]=m[i-1]-(100 / (z*t[i] + 1))
    if m[i]<thres*10:
      return max(np.abs(t[i]-Ti)+np.random.randn(),np.random.uniform(1,30)),m
  return t[i]+np.random.randn(),m
d={'Degradation1': [{'id': 6, 'component_d1': 'CD1', 'threshold': '910'}],
    'Degradation2': [{'id': 3, 'component_d2': 'CD2', 'threshold': '1014'}],
    'Degradation3': [{'id': 1, 'component_d3': 'CD3', 'threshold': '950'}],
    'Degradation4': [{'id': 5, 'component_d4': 'CD4', 'threshold': '1120'}],
    'Degradation5': [{'id': 5, 'component_d5': 'CD5', 'threshold': '550'}]
   }
parameters_list = []
plotting = {}
for i in d['Degradation1']:
    parameters_list.append(['D1', i['component_d1'], i['threshold'], 0.0, 0.0, np.zeros(10000)])
for i in d['Degradation2']:
    parameters_list.append(['D2', i['component_d2'], i['threshold'], 0.0, 0.0, np.zeros(10000)])
for i in d['Degradation3']:
    parameters_list.append(['D3', i['component_d3'], i['threshold'], 0.0, 0.0, np.zeros(10000)])
for i in d['Degradation4']:
    parameters_list.append(['D4', i['component_d4'], i['threshold'], 0.0, 0.0, np.zeros(10000)])
for i in d['Degradation5']:
    parameters_list.append(['D5', i['component_d5'], i['threshold'], 0.0, 0.0, np.zeros(10000)])

# Simulate
for i in range(50):
    for i in range(len(parameters_list)):
        if parameters_list[i][0] == 'D1':
            parameters_list[i][4], parameters_list[i][5] = eval(parameters_list[i][0])(float(parameters_list[i][2]),
                                                                                       parameters_list[i][5],
                                                                                       parameters_list[i][3])
        elif parameters_list[i][0] == 'D2':
            parameters_list[i][4], parameters_list[i][5] = eval(parameters_list[i][0])(float(parameters_list[i][2]),
                                                                                       parameters_list[i][5],
                                                                                       parameters_list[i][3])
        elif parameters_list[i][0] == 'D3':
            parameters_list[i][4], parameters_list[i][5] = eval(parameters_list[i][0])(float(parameters_list[i][2]),
                                                                                       parameters_list[i][5],
                                                                                       parameters_list[i][3])
        elif parameters_list[i][0] == 'D4':
            parameters_list[i][4], parameters_list[i][5] = eval(parameters_list[i][0])(float(parameters_list[i][2]),
                                                                                       parameters_list[i][5],
                                                                                       parameters_list[i][3])
        elif parameters_list[i][0] == 'D5':
            parameters_list[i][4], parameters_list[i][5] = eval(parameters_list[i][0])(float(parameters_list[i][2]),
                                                                                       parameters_list[i][5],
                                                                                       parameters_list[i][3])
    # print(parameters_list[-1][:])
    T_fail_list = min(parameters_list, key=lambda x: x[-2])

    T_fail = T_fail_list[-2]
    comp = parameters_list.index(T_fail_list)
    print("Failed Time:", T_fail)
    print("Component:", parameters_list[comp][1])
    # print(list_of_lists)
    lowest_value = min([inner_list[-2] for inner_list in parameters_list])

    for inner_list in parameters_list:
        inner_list[-3] += lowest_value
    parameters_list[comp][-3] = 0.0

    T_fail1 = int(T_fail)
    for i in range(len(parameters_list)):
        if parameters_list[i][0] == 'D1' or parameters_list[i][0] == 'D2' or parameters_list[i][0] == 'D3' or \
                parameters_list[i][0] == 'D4' or parameters_list[i][0] == 'D5':
            if parameters_list[i][1] == parameters_list[comp][1]:
                if parameters_list[i][1] in plotting:
                    plotting[parameters_list[i][1]].append(parameters_list[i][5][:T_fail1])
                else:
                    plotting[parameters_list[i][1]] = [parameters_list[i][5][:T_fail1]]
                parameters_list[i][5] = np.zeros(10000)
            else:
                parameters_list[i][5][T_fail1 + 1:] = 0

            # print("Degrdation:",parameters_list[i][5][T_fail1-3:T_fail1+3])
    x = []
    for inner_list in parameters_list:
        x.append(inner_list[-3])
    print("TOT:", x)
    y = []
    for inner_list in parameters_list:
        y.append(inner_list[-2])
    print("ABS:", y)
    print("\n")