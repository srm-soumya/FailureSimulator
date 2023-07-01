from django.shortcuts import render, HttpResponse
from django.http import JsonResponse
import inspect
import math
from . import reliability_models  # replace with your Python file name
#from .trials import add_numbers, multiply_numbers, subtract_numbers
from Simulator.models import Weibull, BasicShock, ExtremeShock, CumulativeShock, Degradation1, Degradation2, Degradation3, Degradation4, Degradation5
import json
import time
from django.views.decorators.csrf import csrf_exempt
import numpy as np
import pandas as pd
import random
from scipy.stats import expon
from django.core.management import call_command
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import os
import uuid
from django.shortcuts import render
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import csv
import seaborn as sns
import mplcursors
from django.http import HttpResponseRedirect
import copy
from scipy.optimize import curve_fit
from django.conf.urls.static import static
data_dict = {}
n = 0
plot_data = {}
plot_data_s = {}
ttf = []
component = []
save_data_d = {}
save_data_w = {}
save_data_s = {}
dependency_dict = {}
dep = False
# Create your views here.
def index(request):

    return render(request, 'index.html')
    #return HttpResponse("This is our Simulator")
def about(request):
    return render(request, 'about.html')
def services(request):
    return HttpResponse("Services the simulator provides.")
def contact(request):
    return render(request, 'contact_us.html')
#def save_to_db():
def weibull(request):
    return HttpResponse("Eta-beta Parameters ")
#def simulate(request):
    #return render(request, 'simulate.html')
# define view function
def simulate(request):
    if request.method == 'POST':
        # get selected function name from form data
        function_name = request.POST.get('function_name')

        # call the selected function
        if function_name == 'Weibull':
            result = reliability_models.Weibull(1.5,2000)
        elif function_name == 'Basic_Shock':
            result = reliability_models.Basic_Shock()
        elif function_name == 'Extreme_Shock':
            result = reliability_models.Extreme_Shock()
        elif function_name == 'Cumulative_Shock':
            result = reliability_models.Cumulative_Shock()
        elif function_name == 'Degradation1':
            result = reliability_models.Degradation1(10, 40)
        elif function_name == 'Degradation2':
            result = reliability_models.Degradation2(10, 40)
        elif function_name == 'Degradation3':
            result = reliability_models.Degradation3(10, 40)
        elif function_name == 'Degradation4':
            result = reliability_models.Degradation4(10, 40)

        else:
            result = "Invalid function selection"

        # render the result with the template
        return render(request, 'function_result.html', {'result': result})

    # if the request method is GET, render the template with the function list
    return render(request, 'simulate.html', {'function_list': ['Weibull', 'Basic_Shock', 'Extreme_Shock'
        , 'Cumulative_Shock', 'Degradation1', 'Degradation2', 'Degradation3', 'Degradation4', 'Degradation5']})

def reset_database(request):
    call_command('flush', '--noinput')  # reset the database
    return HttpResponseRedirect(reverse('index.html'))  # redirect to the same page
def save_to_weibull(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        component_w = data['component_Weibull']
        eta = data['eta_Weibull']
        beta = data['beta_Weibull']

        weibull_model = Weibull(component_w=component_w, eta=eta, beta=beta)
        weibull_model.save()

        return JsonResponse({'message': 'Data saved successfully.'})

    return JsonResponse({'message': 'Invalid request method.'})
def save_to_basicshock(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        component_bs = data['component_bs']
        expected_arrival = data['eta_bs']
        threshold_shocks = data['threshold_bs']


        bs_model = BasicShock(component_bs=component_bs, expected_arrival=expected_arrival, threshold_shocks=threshold_shocks)
        bs_model.save()

        return JsonResponse({'message': 'Data saved successfully.'})

    return JsonResponse({'message': 'Invalid request method.'})
def save_to_extremeshock(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        component_es = data['component_es']
        expected_arrival = data['eta_es']
        threshold_shocks = data['threshold_es']
        mean_magnitude = data['mean_magnitude_es']
        std_magnitude = data['std_magnitude_es']
        threshold_magnitude = data['threshold_magnitude_es']


        es_model = ExtremeShock(component_es=component_es, expected_arrival=expected_arrival,
                                  threshold_shocks=threshold_shocks,mean_magnitude=mean_magnitude,
                                  std_magnitude=std_magnitude,threshold_magnitude=threshold_magnitude)
        es_model.save()

        return JsonResponse({'message': 'Data saved successfully.'})

    return JsonResponse({'message': 'Invalid request method.'})
def save_to_cumulativeshock(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        component_cs = data['component_cs']
        expected_arrival = data['eta_cs']
        threshold_shocks = data['threshold_cs']
        mean_magnitude = data['mean_magnitude_cs']
        std_magnitude = data['std_magnitude_cs']
        threshold_magnitude = data['threshold_magnitude_cs']


        cs_model = CumulativeShock(component_cs=component_cs, expected_arrival=expected_arrival,
                                  threshold_shocks=threshold_shocks,mean_magnitude=mean_magnitude,
                                  std_magnitude=std_magnitude, threshold_magnitude=threshold_magnitude)
        cs_model.save()

        return JsonResponse({'message': 'Data saved successfully.'})

    return JsonResponse({'message': 'Invalid request method.'})
def save_to_degradation1(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        component_d1 = data['component_d1']
        low_in = data['initial_min_d1']
        high_in = data['initial_max_d1']
        # z = data['parameter_d1']
        threshold = data['threshold_d1']
        p1 = data['p1_d1']
        p2 = data['p2_d1']
        dist = data['dist']


        d1_model = Degradation1(component_d1=component_d1, low_in=low_in, high_in=high_in, threshold=threshold, p1=p1, p2=p2, dist=dist)
        d1_model.save()

        return JsonResponse({'message': 'Data saved successfully.'})

    return JsonResponse({'message': 'Invalid request method.'})
def save_to_degradation2(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        component_d2 = data['component_d2']
        low_in = data['initial_min_d2']
        high_in = data['initial_max_d2']
        # z = data['parameter_d2']
        threshold = data['threshold_d2']
        p1 = data['p1_d2']
        p2 = data['p2_d2']
        dist = data['dist']


        d2_model = Degradation2(component_d2=component_d2, low_in=low_in, high_in=high_in, threshold=threshold, p1=p1, p2=p2, dist=dist)
        d2_model.save()

        return JsonResponse({'message': 'Data saved successfully.'})

    return JsonResponse({'message': 'Invalid request method.'})
def save_to_degradation3(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        component_d3 = data['component_d3']
        low_in = data['initial_min_d3']
        high_in = data['initial_max_d3']
        # z = data['parameter1_d4']
        # y = data['parameter2_d4']
        threshold = data['threshold_d3']
        p1 = data['p1_d3']
        p2 = data['p2_d3']
        p3 = data['p3_d3']
        p4 = data['p4_d3']
        p5 = data['p5_d3']
        p6 = data['p6_d3']
        dist = data['dist']
        try:
            d3_model = Degradation3(component_d3=component_d3, low_in=low_in, high_in=high_in, threshold=threshold, p1=p1, p2=p2, p3=p3, p4=p4, p5=p5, p6=p6, dist=dist)
            d3_model.save()
        except Exception as e:
            print(e)


        return JsonResponse({'message': 'Data saved successfully.'})

    return JsonResponse({'message': 'Invalid request method.'})
def save_to_degradation4(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        component_d4 = data['component_d4']
        low_in = data['initial_min_d4']
        high_in = data['initial_max_d4']
        # z = data['parameter1_d4']
        # y = data['parameter2_d4']
        threshold = data['threshold_d4']
        p1 = data['p1_d4']
        p2 = data['p2_d4']
        p3 = data['p3_d4']
        p4 = data['p4_d4']
        dist = data['dist']
        d4_model = Degradation4(component_d4=component_d4, low_in=low_in, high_in=high_in, threshold=threshold, p1=p1, p2=p2, p3=p3, p4=p4, dist=dist)
        d4_model.save()

        return JsonResponse({'message': 'Data saved successfully.'})

    return JsonResponse({'message': 'Invalid request method.'})
def save_to_degradation5(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        component_d5 = data['component_d5']
        low_in = data['initial_min_d5']
        high_in = data['initial_max_d5']
        # z = data['parameter1_d4']
        # y = data['parameter2_d4']
        threshold = data['threshold_d5']
        p1 = data['p1_d5']
        p2 = data['p2_d5']
        dist = data['dist']

        d5_model = Degradation5(component_d5=component_d5, low_in=low_in, high_in=high_in , threshold=threshold, p1=p1, p2=p2, dist=dist)
        d5_model.save()

        return JsonResponse({'message': 'Data saved successfully.'})

    return JsonResponse({'message': 'Invalid request method.'})
def select_function(request):
    available_functions = [func for func in dir(reliability_models) if callable(getattr(reliability_models, func))]
    if request.method == 'POST':
        selected_function = request.POST['selected_function']
        result = getattr(reliability_models, selected_function)()
        return render(request, 'result.html', {'result': result})
    return render(request, 'select_function.html', {'available_functions': available_functions})
def show_data(request):
    global data_dict
    global n
    global dep
    dep=False
    # Extract data from Weibull
    table1_data = Weibull.objects.all().values()
    data_dict['Weibull'] = list(table1_data)
    # Extract data from BasicShock
    table2_data = BasicShock.objects.all().values()
    data_dict['BasicShock'] = list(table2_data)
    # Extract data from ExtremeShock
    table3_data = ExtremeShock.objects.all().values()
    data_dict['ExtremeShock'] = list(table3_data)
    # Extract data from CumulativeShock
    table4_data = CumulativeShock.objects.all().values()
    data_dict['CumulativeShock'] = list(table4_data)
    # Extract data from Degradation1
    table5_data = Degradation1.objects.all().values()
    data_dict['Degradation1'] = list(table5_data)
    # Extract data from Degradation2
    table6_data = Degradation2.objects.all().values()
    data_dict['Degradation2'] = list(table6_data)
    # Extract data from Degradation3
    table7_data = Degradation3.objects.all().values()
    data_dict['Degradation3'] = list(table7_data)
    # Extract data from Degradation4
    table8_data = Degradation4.objects.all().values()
    data_dict['Degradation4'] = list(table8_data)
    # Extract data from Degradation5
    table9_data = Degradation5.objects.all().values()
    data_dict['Degradation5'] = list(table9_data)
    print(data_dict)
    # if request.method == 'POST':
    #     n = int(request.POST.get('my_integer'))
    return render(request, 'show_data.html', {'data_dict': data_dict})
def start(request):
    global n
    if request.method == 'POST':
        n = int(request.POST.get('my_integer'))
    return render(request, 'start.html')
def dependency(request):
    global dependency_dict
    global dep
    dep = True
    def get_headings():
        global data_dict
        d1 = data_dict
        headings = []
        for i in d1['Weibull']:
            headings.append(i['component_w'])
        for i in d1['BasicShock']:
            headings.append(i['component_bs'])
        for i in d1['ExtremeShock']:
            headings.append(i['component_es'])
        for i in d1['CumulativeShock']:
            headings.append(i['component_cs'])
        for i in d1['Degradation1']:
            headings.append(i['component_d1'])
        for i in d1['Degradation2']:
            headings.append(i['component_d2'])
        for i in d1['Degradation3']:
            headings.append(i['component_d3'])
        for i in d1['Degradation4']:
            headings.append(i['component_d4'])
        for i in d1['Degradation5']:
            headings.append(i['component_d5'])


        #headings = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        return headings

    if request.method == 'POST':
        data_array = []
        headings = get_headings()
        data = []
        for heading in headings:
            row = []
            for i in range(1, len(headings) + 1):
                field_name = f'{heading}{i}'
                value = request.POST.get(field_name)
                row.append(value)
            data.append(row)

        # Do something with the data_array, such as saving it to a dictionary for further processing.
        headings = get_headings()
        for i in range(len(headings)):
            inner_dict = {}
            for j in range(len(headings)):
                inner_dict[headings[j]] = data[i][j]
            dependency_dict[headings[i]] = inner_dict

        print(dependency_dict)

        # Get the unique column/row headings
        headings = list(dependency_dict.keys())

        # Create a CSV file and write the data
        with open('dependency_dict.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([''] + headings)  # Write the header row

            for heading in headings:
                row = [heading] + [dependency_dict[heading].get(col, '') for col in headings]
                writer.writerow(row)

        # Redirect to the same page after processing the data
        return HttpResponseRedirect(request.path)

    headings = get_headings()
    return render(request, 'dependency.html', {'headings': headings})
def process_data(request):
    global data_dict
    global n
    global plot_data
    global plot_data_s
    global ttf
    global component
    global save_data_d
    global save_data_w
    global save_data_s
    global dep
    global dependency_dict
    if isinstance(ttf, np.ndarray):
        ttf = ttf.tolist()
    print(type(ttf))
    if ttf:  # Check if the list is not empty
        ttf.clear()
    if component:  # Check if the list is not empty
        component.clear()
    #ttf = ttf.tolist()
    plotting_w = {}
    plotting_d = {}
    plotting_s = {}
    d=data_dict
    def common_causes(l):
        T = expon.rvs(scale=l)
        return T

    def Weibull_2_param(beta, eta, Ti=0, T_CC1=999999, af1=[1, 1], T_CC2=999999, af2=[1, 1], i=0):
        '''
        (beta,eta,R,T_CC1,af1,T_CC2,af2,Ti)'

        beta: Shape Parameter
        eta: Scale Parameter
        R: Expected Reliability
        Ti: Initial Working Hours  (Default=0)
        T_CC1: Time of arrival of common cause 1 (Default=9999)
        af1: Amount of effect on eta
        T_CC2: Time of arrival of common cause 2 (Default=9999)
        af2: Amount of effect on eta
        (NOTE--> There is a 50-50 probability that common cause arrives and it affects or not.)


        '''
        # np.random.seed(100)
        # eta=eta*(1-(i/50000000))
        Ri = np.exp(-((Ti / eta) ** beta))  # Considering the initial life
        R = np.random.uniform(0, 1)
        # print(R)
        R = R * Ri
        T = (eta * ((-np.log(R)) ** (1 / beta))) - Ti  # Finding the time to failure
        # return T-Ti
        # return T
        if T_CC1 != 9999 and T_CC2 != 9999:  # Checking whether both values are given by user
            T_min = min(T_CC1, T_CC2)
            T_max = max(T_CC1,
                        T_CC2)  # Finding which of the common cause arrive first and assigning the min and max T values accordingly.
            if T_min == T_CC1:
                af_min = af1
                af_max = af2
            else:
                af_min = af2
                af_max = af1
            x_min = np.random.randint(0,
                                      2)  # Generating values whether they are going to affect or not 0:not affecting, 1: affecting
            x_max = np.random.randint(0, 2)
            # x_min,x_max=1,1

            if T_min < T and T_max < T:  # Finding the updated time of failure after checking whether they arrive before failure
                if x_min == 1 and x_max == 1:
                    T = T_min + (T_max - T_min) * np.random.uniform(af_min[0], af_min[1]) + (
                                T - T_max) * np.random.uniform(af_max[0], af_max[1]) * np.random.uniform(af_min[0],
                                                                                                         af_min[1])
                elif x_min == 0 and x_max == 1:
                    T = T_max + (T - T_max) * np.random.uniform(af_max[0], af_max[1])
                elif x_min == 1 and x_max == 0:
                    T = T_min + (T - T_min) * np.random.uniform(af_min[0], af_min[1])


            elif T_min < T:
                if x_min == 1:
                    T = T_min + (T - T_min) * np.random.uniform(af_min[0], af_min[1])

        elif T_CC1 != 9999:  # Cases where only 1 common cause is there
            x = np.random.randint(0, 2)
            # x=1
            if T_CC1 < T:
                if x == 1:
                    T = T_CC1 + (T - T_CC1) * np.random.uniform(af1[0], af1[1])

        elif T_CC2 != 9999:  # Cases where only 1 common cause is there
            x = np.random.randint(0, 2)
            if T_CC2 < T:
                if x == 1:
                    T = T_CC2 + (T - T_CC2) * np.random.uniform(af2[0], af2[1])
        return T
        # return max(T-Ti,np.random.choice(np.random.uniform(20,50,size=100)))

    def basic_shock(l, Ti=0):
        T = expon.rvs(scale=l)
        return T

    def extreme_shock(l, mean, std, Ti=0):
        T = expon.rvs(scale=l)
        x = np.random.normal(loc=mean, scale=std)
        return x, T

    def cumulative_shock(l, mean, std, Ti=0):
        T = expon.rvs(scale=l)
        x = np.random.normal(loc=mean, scale=std)
        return x, T

    def D1(thres, m, low, high, p1, p2, dist, Ti=0):
        t = np.linspace(1, 10000, 10000)
        # m=np.zeros(10000)
        Tin = int(np.ceil(Ti))
        for i in range(Tin, 10000):
            # z = np.random.uniform(0.001, 0.01)
            if dist == 'Uniform':
                z = np.random.uniform(p1, p2)
            else:
                z = np.random.normal(loc=p1, scale=p2)
            if i == 0:
                m[i] = np.random.uniform(low, high)
            else:
                m[i] = m[i - 1] + np.exp(z * t[i])
            if m[i] > thres * 100:
                if t[i] - Ti <= 1:
                    return max(np.abs(t[i] - Ti) - np.random.rand(), np.random.uniform(1, 30)), m
                return t[i] - Ti - np.random.rand(), m
        return 10000 - Ti, m

    def D2(thres, m, low, high, p1, p2, dist, Ti=0):
        t = np.linspace(1, 10000, 10000)
        # m=np.zeros(10000)
        Tin = int(np.ceil(Ti))
        for i in range(Tin, 10000):
            # z = np.random.uniform(0.001, 0.01)
            if dist == 'Uniform':
                z = np.random.uniform(p1, p2)
            else:
                z = np.random.normal(loc=p1, scale=p2)
            if i == 0:
                m[i] = np.random.uniform(low, high)
            else:
                m[i] = m[i - 1] + np.log(1+z*t[i])
            if m[i] > thres * 100:
                if t[i] - Ti <= 1:
                    return max(np.abs(t[i] - Ti) - np.random.rand(), np.random.uniform(1, 30)), m
                return t[i] - Ti - np.random.rand(), m
        return 10000 - Ti, m

    def D3(thres, m, low, high, p1, p2, p3, p4, p5, p6, dist, Ti=0):
        t = np.linspace(1, 10000, 10000)
        # m=np.zeros(10000)
        Tin = int(np.ceil(Ti))
        for i in range(Tin, 10000):
            # z0 = np.random.uniform(2, 25)
            # z1 = np.random.uniform(0.1, 0.8)
            # z2 = np.random.uniform(0.01, 0.09)
            if dist == 'Uniform':
                z0 = np.random.uniform(p1, p2)
                z1 = np.random.uniform(p3, p4)
                z2 = np.random.uniform(p5, p6)
            else:
                z0 = np.random.normal(loc=p1, scale=p2)
                z1 = np.random.normal(loc=p3, scale=p4)
                z2 = np.random.normal(loc=p5, scale=p6)
            if i == 0:
                m[i] = np.random.uniform(low, high)
            else:
                m[i] = m[i - 1] + (z0 + z1 * t[i] + z2 * (t[i] ** 0.75))
                # m[i]=m[i-1]+np.log(1+z*t[i])
            if m[i] > thres * 100:
                if t[i] - Ti <= 1:
                    return max(np.abs(t[i] - Ti) - np.random.rand(), np.random.uniform(1, 30)), m
                return t[i] - Ti - np.random.rand(), m
        return 10000 - Ti, m

    def D4(thres, m, low, high, p1, p2, p3, p4, dist, Ti=0):
        t = np.linspace(1, 10000, 10000)
        # m=np.zeros(10000)
        Tin = int(np.ceil(Ti))
        for i in range(Tin, 10000):
            # z0 = np.random.uniform(6, 14)
            # z1 = np.random.uniform(0.08, 0.8)
            if dist == 'Uniform':
                z0 = np.random.uniform(p1, p2)
                z1 = np.random.uniform(p3, p4)
            else:
                z0 = np.random.normal(loc=p1, scale=p2)
                z1 = np.random.normal(loc=p3, scale=p4)
            if i == 0:
                m[i] = np.random.uniform(low, high)
            else:
                m[i] = m[i - 1] + (z0 + z1 * (t[i] ** 0.95))
                # m[i]=m[i-1]+np.log(1+z*t[i])
            if m[i] > thres * 10:
                if t[i] - Ti <= 1:
                    return max(np.abs(t[i] - Ti) - np.random.rand(), np.random.uniform(1, 30)), m
                return t[i] - Ti - np.random.rand(), m
        return 10000 - Ti, m

    def D5(thres, m, low, high, p1, p2, dist, Ti=0):
        t = np.linspace(1, 10000, 10000)
        # m=np.zeros(10000)
        Tin = int(np.ceil(Ti))
        for i in range(Tin, 10000):
            # z=np.random.uniform(0.01,0.2)
            if dist == 'Uniform':
                z = np.random.uniform(p1, p2)
            else:
                z = np.random.normal(loc=p1, scale=p2)
            if i == 0:
                m[i] = np.random.uniform(low, high)
            else:
                m[i] = m[i - 1] - (100 / (z * t[i] + 1))
            if m[i] < thres * 10:
                if t[i] - Ti <= 1:
                    return max(np.abs(t[i] - Ti) - np.random.rand(), np.random.uniform(1, 30)), m
                return t[i] - Ti - np.random.rand(), m

        return 10000 - Ti, m
    parameters_list = []
    for i in d['Weibull']:
        parameters_list.append(['Weibull_2_param', i['component_w'], i['eta'], i['beta'], 0.0, 0.0])
    for i in d['BasicShock']:
        parameters_list.append(['basic_shock', i['component_bs'], i['expected_arrival'], i['threshold_shocks'],
                                np.zeros(int(i['threshold_shocks'])), 0.0, 0.0])
    for i in d['ExtremeShock']:
        parameters_list.append(
            ['extreme_shock', i['component_es'], i['expected_arrival'], i['threshold_shocks'], i['mean_magnitude'],
             i['std_magnitude'], i['threshold_magnitude'], np.zeros(int(i['threshold_shocks'])),
             np.zeros(int(i['threshold_shocks'])), 0.0, 0.0])
    for i in d['CumulativeShock']:
        parameters_list.append(
            ['cumulative_shock', i['component_cs'], i['expected_arrival'], i['threshold_shocks'], i['mean_magnitude'],
             i['std_magnitude'], i['threshold_magnitude'], np.zeros(int(i['threshold_shocks'])),
             np.zeros(int(i['threshold_shocks'])), 0.0, 0.0])
    for i in d['Degradation1']:
        parameters_list.append(['D1', i['component_d1'], i['threshold'], np.zeros(10000), i['low_in'], i['high_in']
                                   , i['p1'], i['p2'], i['dist'], 0.0, 0.0])
    for i in d['Degradation2']:
        parameters_list.append(['D2', i['component_d2'], i['threshold'], np.zeros(10000), i['low_in'], i['high_in']
                                   , i['p1'], i['p2'], i['dist'], 0.0, 0.0])
    for i in d['Degradation3']:
        parameters_list.append(['D3', i['component_d3'], i['threshold'], np.zeros(10000), i['low_in'], i['high_in']
                                   , i['p1'], i['p2'], i['p3'], i['p4'], i['p5'], i['p6'], i['dist'], 0.0, 0.0])
    for i in d['Degradation4']:
        parameters_list.append(['D4', i['component_d4'], i['threshold'], np.zeros(10000), i['low_in'], i['high_in']
                                   , i['p1'], i['p2'], i['p3'], i['p4'], i['dist'], 0.0, 0.0])
    for i in d['Degradation5']:
        parameters_list.append(['D5', i['component_d5'], i['threshold'], np.zeros(10000), i['low_in'], i['high_in']
                                   , i['p1'], i['p2'], i['dist'], 0.0, 0.0])

    #Creating dictionary for plotting shocks
    for i in range(len(parameters_list)):
        if parameters_list[i][0] == 'basic_shock' or parameters_list[i][0] == 'extreme_shock' or parameters_list[i][
            0] == 'cumulative_shock':
            plotting_s[parameters_list[i][1]] = {}
            plotting_s[parameters_list[i][1]]["Time"] = []
            plotting_s[parameters_list[i][1]]["Magnitude"] = []
    parameters_list_ud = copy.deepcopy(parameters_list)
    # Simulate
    items = []
    # n=20
    for k in range(n):
        for i in range(len(parameters_list)):
            if parameters_list[i][0] == 'Weibull_2_param':
                parameters_list[i][5] = eval(parameters_list[i][0])(float(parameters_list[i][3]),
                                                                    float(parameters_list[i][2]),
                                                                     parameters_list[i][4])
            elif parameters_list[i][0] == 'basic_shock':
                if parameters_list[i][5] == 0.0:
                    for p in range(int(parameters_list[i][3])):
                        parameters_list[i][4][p] = eval(parameters_list[i][0])(float(parameters_list[i][2]),
                                                                               parameters_list[i][5])
                        parameters_list[i][4][p] = parameters_list[i][4][p] + parameters_list[i][4][p - 1] - \
                                                   parameters_list[i][5] if p > 0 else parameters_list[i][4][p]
                    parameters_list[i][6] = parameters_list[i][4][-1]
                    plotting_s[parameters_list[i][1]]["Time"].append(np.array(parameters_list[i][4]))
                    plotting_s[parameters_list[i][1]]["Magnitude"].append(
                        np.array([1] * int(parameters_list[i][3])))

                else:
                    parameters_list[i][6] = parameters_list[i][4][-1]
                # print(parameters_list[i][1],":",parameters_list[i][4],"\t",parameters_list[i][6])

            elif parameters_list[i][0] == 'extreme_shock':
                if parameters_list[i][9] == 0.0:
                    for p in range(int(parameters_list[i][3])):
                        parameters_list[i][7][p], parameters_list[i][8][p] = eval(parameters_list[i][0])(
                            float(parameters_list[i][2]), float(parameters_list[i][4])
                            , float(parameters_list[i][5]), parameters_list[i][9])
                        parameters_list[i][8][p] = parameters_list[i][8][p] + parameters_list[i][8][p - 1] - \
                                                   parameters_list[i][9] if p > 0 else parameters_list[i][8][p]
                    indices = np.where(parameters_list[i][7] > float(parameters_list[i][6]))[0]
                    if indices.size > 0:
                        index = indices[0]
                    else:
                        index = -1
                    parameters_list[i][10] = parameters_list[i][8][index]
                    if index != -1:
                        plotting_s[parameters_list[i][1]]["Time"].append(
                            np.array(parameters_list[i][8][:index + 1]))
                        plotting_s[parameters_list[i][1]]["Magnitude"].append(
                            np.array(parameters_list[i][7][:index + 1]))
                    else:
                        plotting_s[parameters_list[i][1]]["Time"].append(np.array(parameters_list[i][8]))
                        plotting_s[parameters_list[i][1]]["Magnitude"].append(np.array(parameters_list[i][7]))
                else:
                    indices = np.where(parameters_list[i][7] > float(parameters_list[i][6]))[0]
                    if indices.size > 0:
                        index = indices[0]
                    else:
                        index = -1
                    parameters_list[i][10] = parameters_list[i][8][index]

                # print(parameters_list[i][1],":",parameters_list[i][8],"\nMAG_ES",parameters_list[i][7],"\t",parameters_list[i][10])

            elif parameters_list[i][0] == 'cumulative_shock':
                if parameters_list[i][9] == 0.0:
                    for p in range(int(parameters_list[i][3])):
                        parameters_list[i][7][p], parameters_list[i][8][p] = eval(parameters_list[i][0])(
                            float(parameters_list[i][2]), float(parameters_list[i][4])
                            , float(parameters_list[i][5]), parameters_list[i][9])
                        parameters_list[i][7][p] = parameters_list[i][7][p] + parameters_list[i][7][
                            p - 1] if p > 0 else parameters_list[i][7][p]
                        parameters_list[i][8][p] = parameters_list[i][8][p] + parameters_list[i][8][p - 1] - \
                                                   parameters_list[i][9] if p > 0 else parameters_list[i][8][p]
                    indices = np.where(parameters_list[i][7] > float(parameters_list[i][6]))[0]
                    if indices.size > 0:
                        index = indices[0]
                    else:
                        index = -1
                    parameters_list[i][10] = parameters_list[i][8][index]
                    if index != -1:
                        plotting_s[parameters_list[i][1]]["Time"].append(
                            np.array(parameters_list[i][8][:index + 1]))
                        plotting_s[parameters_list[i][1]]["Magnitude"].append(
                            np.array(parameters_list[i][7][:index + 1]))
                    else:
                        plotting_s[parameters_list[i][1]]["Time"].append(np.array(parameters_list[i][8]))
                        plotting_s[parameters_list[i][1]]["Magnitude"].append(np.array(parameters_list[i][7]))
                else:
                    indices = np.where(parameters_list[i][7] > float(parameters_list[i][6]))[0]
                    if indices.size > 0:
                        index = indices[0]
                    else:
                        index = -1
                    parameters_list[i][10] = parameters_list[i][8][index]
            elif parameters_list[i][0] == 'D1':
                parameters_list[i][-1], parameters_list[i][3] = eval(parameters_list[i][0])(float(parameters_list[i][2]),parameters_list[i][3],
                                                                  float(parameters_list[i][4]),float(parameters_list[i][5]),float(parameters_list[i][6])
                                                                  ,float(parameters_list[i][7]),parameters_list[i][8],parameters_list[i][-2]
                                                                )
            elif parameters_list[i][0] == 'D2':
                parameters_list[i][-1], parameters_list[i][3] = eval(parameters_list[i][0])(float(parameters_list[i][2]),parameters_list[i][3],float(parameters_list[i][4]),
                                                                                float(parameters_list[i][5]),float(parameters_list[i][6]),
                                                                                float(parameters_list[i][7]),parameters_list[i][8],parameters_list[i][-2]
                                                                                    )
            elif parameters_list[i][0] == 'D3':
                parameters_list[i][-1], parameters_list[i][3] = eval(parameters_list[i][0])(float(parameters_list[i][2]),parameters_list[i][3],float(parameters_list[i][4]),
                                                                                float(parameters_list[i][5]),float(parameters_list[i][6]),
                                                                                float(parameters_list[i][7]),float(parameters_list[i][8]),
                                                                                float(parameters_list[i][9]),float(parameters_list[i][10]),
                                                                                float(parameters_list[i][11]),parameters_list[i][12],parameters_list[i][-2])
            elif parameters_list[i][0] == 'D4':
                parameters_list[i][-1], parameters_list[i][3] = eval(parameters_list[i][0])(float(parameters_list[i][2]),parameters_list[i][3],float(parameters_list[i][4]),
                                                                                float(parameters_list[i][5]),float(parameters_list[i][6]),
                                                                                float(parameters_list[i][7]),float(parameters_list[i][8]),
                                                                                float(parameters_list[i][9]),parameters_list[i][10],parameters_list[i][-2])
            elif parameters_list[i][0] == 'D5':
                parameters_list[i][-1], parameters_list[i][3] = eval(parameters_list[i][0])(float(parameters_list[i][2]),parameters_list[i][3],float(parameters_list[i][4]),
                                                                            float(parameters_list[i][5]),float(parameters_list[i][6]),
                                                                            float(parameters_list[i][7]),parameters_list[i][8],parameters_list[i][-2])
        T_fail_list = min(parameters_list, key=lambda x: x[-1])

        T_fail = T_fail_list[-1]
        for i in range(len(parameters_list)):
            if parameters_list[i][0] == 'basic_shock' or parameters_list[i][0] == 'extreme_shock' or parameters_list[i][
                0] == 'cumulative_shock':
                for p in range(int(parameters_list[i][3])):
                    parameters_list[i][-3][p] -= T_fail
        T_fail = round(T_fail, 1)
        comp = parameters_list.index(T_fail_list)
        ttf.append(T_fail)
        component.append(parameters_list[comp][1])

        #print("Next Failure Time:", T_fail)
        #print("Component:", parameters_list[comp][1])
        items.append({'Failed_Time': T_fail, 'Component': parameters_list[comp][1]})
        lowest_value = min([inner_list[-1] for inner_list in parameters_list])

        for inner_list in parameters_list:
            inner_list[-2] += lowest_value
        T_till_date = []
        for inner_list in parameters_list:
            T_till_date.append(inner_list[-2])

        # Implementing DEPENDENCIES
        if dep:
            def convert_to_float(dictionary):
                for key, value in dictionary.items():
                    if isinstance(value, dict):
                        convert_to_float(value)
                    else:
                        dictionary[key] = float(value)

            convert_to_float(dependency_dict)
            for key, value in dependency_dict[parameters_list[comp][1]].items():
                if value != 1:
                    for item in parameters_list:
                        if item[1] == key and item[0] == 'Weibull_2_param' and value > 0:
                            item[2] = float(item[2]) * (1 - float(value))
                            print(item[2])
                        elif item[1] == key and item[0] == 'basic_shock' and value > 0:
                            item[2] = float(item[2]) / (1 + float(value))
                        elif item[1] == key and item[0] == 'extreme_shock' and value > 0:
                            item[2] = float(item[2]) / (1 + float(value))
                            item[4] = float(item[4]) * (1 + float(value))
                        elif item[1] == key and item[0] == 'cumulative_shock' and value > 0:
                            item[2] = float(item[2]) / (1.2 + float(value))
                            item[4] = float(item[4]) * (1 + float(value))
                        elif item[1] == key and item[0] == 'D1' and value > 0:
                            item[2] = float(item[2]) * (1 - float(value))
                            item[6] = float(item[6]) * (1 + float(value))
                            item[7] = float(item[7]) * (1 + float(value))
                        elif item[1] == key and item[0] == 'D2' and value > 0:
                            item[2] = float(item[2]) * (1 - float(value))
                            item[6] = float(item[6]) * (1 + float(value))
                            item[7] = float(item[7]) * (1 + float(value))
                        elif item[1] == key and item[0] == 'D3' and value > 0:
                            item[2] = float(item[2]) * (1 - float(value))
                            item[6] = float(item[6]) * (1 + float(value))
                            item[7] = float(item[7]) * (1 + float(value))
                            item[8] = float(item[8]) * (1 + float(value))
                            item[9] = float(item[9]) * (1 + float(value))
                            item[10] = float(item[10]) * (1 + float(value))
                            item[11] = float(item[11]) * (1 + float(value))
                        elif item[1] == key and item[0] == 'D4' and value > 0:
                            item[2] = float(item[2]) * (1 - float(value))
                            item[6] = float(item[6]) * (1 + float(value))
                            item[7] = float(item[7]) * (1 + float(value))
                            item[8] = float(item[8]) * (1 + float(value))
                            item[9] = float(item[9]) * (1 + float(value))
                        elif item[1] == key and item[0] == 'D5' and value > 0:
                            item[2] = float(item[2]) * (1 - float(value))
                            item[6] = float(item[6]) * (1 + float(value))
                            item[7] = float(item[7]) * (1 + float(value))
                elif value == 1:
                    for item, item_ud in zip(parameters_list, parameters_list_ud):
                        if item[1] == key and item[0] == 'Weibull_2_param':
                            item[2] = copy.deepcopy(item_ud[2])
                            print(item[2])
                        elif item[1] == key and item[0] == 'basic_shock':
                            item[2] = copy.deepcopy(item_ud[2])
                        elif item[1] == key and item[0] == 'extreme_shock':
                            item[2] = copy.deepcopy(item_ud[2])
                            item[4] = copy.deepcopy(item_ud[4])
                        elif item[1] == key and item[0] == 'cumulative_shock':
                            item[2] = copy.deepcopy(item_ud[2])
                            item[4] = copy.deepcopy(item_ud[4])
                        elif item[1] == key and item[0] == 'D1':
                            item[2] = copy.deepcopy(item_ud[2])
                            item[6] = copy.deepcopy(item_ud[6])
                            item[7] = copy.deepcopy(item_ud[7])
                        elif item[1] == key and item[0] == 'D2':
                            item[2] = copy.deepcopy(item_ud[2])
                            item[6] = copy.deepcopy(item_ud[6])
                            item[7] = copy.deepcopy(item_ud[7])
                        elif item[1] == key and item[0] == 'D3':
                            item[2] = copy.deepcopy(item_ud[2])
                            item[6] = copy.deepcopy(item_ud[6])
                            item[7] = copy.deepcopy(item_ud[7])
                            item[8] = copy.deepcopy(item_ud[8])
                            item[9] = copy.deepcopy(item_ud[9])
                            item[10] = copy.deepcopy(item_ud[10])
                            item[11] = copy.deepcopy(item_ud[11])
                        elif item[1] == key and item[0] == 'D4':
                            item[2] = copy.deepcopy(item_ud[2])
                            item[6] = copy.deepcopy(item_ud[6])
                            item[7] = copy.deepcopy(item_ud[7])
                            item[8] = copy.deepcopy(item_ud[8])
                            item[9] = copy.deepcopy(item_ud[9])
                        elif item[1] == key and item[0] == 'D5':
                            item[2] = copy.deepcopy(item_ud[2])
                            item[6] = copy.deepcopy(item_ud[6])
                            item[7] = copy.deepcopy(item_ud[7])

        #FOR WEIBULL
        for i in range(len(parameters_list)):
            if parameters_list[i][0] == 'Weibull_2_param':
                if parameters_list[i][1] == parameters_list[comp][1]:
                    if parameters_list[i][1] in plotting_w:
                        plotting_w[parameters_list[i][1]].append(parameters_list[i][-2])
                        save_data_w[parameters_list[i][1]].append(parameters_list[i][-2])
                    else:
                        plotting_w[parameters_list[i][1]] = []
                        plotting_w[parameters_list[i][1]].append(parameters_list[i][-2])
                        save_data_w[parameters_list[i][1]] = []
                        save_data_w[parameters_list[i][1]].append(parameters_list[i][-2])
        parameters_list[comp][-2] = 0.0
        #FOR DEGRADATION
        for i in range(len(parameters_list)):
            if parameters_list[i][0] == 'D1' or parameters_list[i][0] == 'D2' or parameters_list[i][0] == 'D3' or \
                    parameters_list[i][0] == 'D4' or parameters_list[i][0] == 'D5':
                if parameters_list[i][1] == parameters_list[comp][1]:
                    if parameters_list[i][1] in plotting_d:
                        plotting_d[parameters_list[i][1]].append(parameters_list[i][3][:int(T_till_date[i]) + 1])
                    else:
                        plotting_d[parameters_list[i][1]] = [parameters_list[i][3][:int(T_till_date[i]) + 1]]
                    parameters_list[i][3] = np.zeros(10000)
                else:
                    parameters_list[i][3][int(T_till_date[i]) + 1:] = 0
        # x = []
        # for inner_list in parameters_list:
        #     x.append(inner_list[-2])
    #FOR DEGRADATION
    for key, value in plotting_d.items():
        x = np.array(value,dtype='object')
        x = np.concatenate(x)
        plot_data[key] = x
        save_data_d[key] = x

    # FOR WEIBULL
    for key, value in plotting_w.items():
        cumulative_sum = []
        total = 0
        for num in value:
            total += num
            cumulative_sum.append(total)
        plotting_w[key] = cumulative_sum
        save_data_w[key] = cumulative_sum
    for key in plotting_w:
        plot_data[key] = []
    # Iterate over existing_dict and apply conditions
    for key, value in plotting_w.items():
        for i in range(len(value)):
            if i == 0:
                for i in range(int(value[i])):
                    plot_data[key].append(0)
                plot_data[key].append(1)
            else:
                for i in range(int(value[i]) - int(value[i - 1]) - 1):
                    plot_data[key].append(0)
                plot_data[key].append(1)

    #FOR SHOCKS
    for i in range(len(parameters_list)):
        if parameters_list[i][0] == 'basic_shock' or parameters_list[i][0] == 'extreme_shock' or parameters_list[i][0] == 'cumulative_shock':
            plot_data_s[parameters_list[i][1]] = {}
            plot_data_s[parameters_list[i][1]]["Time"] = []
            plot_data_s[parameters_list[i][1]]["Magnitude"] = []
            plot_data_s[parameters_list[i][1]]["Shocks"] = []
            save_data_s[parameters_list[i][1]] = {}
            save_data_s[parameters_list[i][1]]["Time"] = []
            save_data_s[parameters_list[i][1]]["Magnitude"] = []
            save_data_s[parameters_list[i][1]]["Shocks"] = []
    for i in range(len(parameters_list)):
        magn = []
        time = []
        last_elements = []
        if parameters_list[i][0] == 'basic_shock' or parameters_list[i][0] == 'extreme_shock' or parameters_list[i][
            0] == 'cumulative_shock':
            for array in plotting_s[parameters_list[i][1]]['Time']:
                # last_element_array.append(array[-1])
                if time:
                    time.extend(array + time[-1])
                else:
                    time.extend(array)
                last_elements.append(time[-1])
            for array in plotting_s[parameters_list[i][1]]['Magnitude']:
                # last_element_array.append(array[-1])
                if magn:
                    magn.extend(array)
                else:
                    magn.extend(array)
            plot_data_s[parameters_list[i][1]]['Time'] = time
            plot_data_s[parameters_list[i][1]]['Magnitude'] = magn
            plot_data_s[parameters_list[i][1]]['Shocks'] = last_elements
            plot_data_s[parameters_list[i][1]]['Time'] = list(map(int, plot_data_s[parameters_list[i][1]]['Time']))
            plot_data_s[parameters_list[i][1]]['Shocks'] = list(map(int, plot_data_s[parameters_list[i][1]]['Shocks']))
            plot_data_s[parameters_list[i][1]]['Magnitude'] = [round(value, 1) for value in
                                                               plot_data_s[parameters_list[i][1]]['Magnitude']]
            save_data_s[parameters_list[i][1]]['Time'] = time
            save_data_s[parameters_list[i][1]]['Magnitude'] = magn
            save_data_s[parameters_list[i][1]]['Shocks'] = last_elements
            save_data_s[parameters_list[i][1]]['Time'] = list(map(int, save_data_s[parameters_list[i][1]]['Time']))
            save_data_s[parameters_list[i][1]]['Shocks'] = list(map(int, save_data_s[parameters_list[i][1]]['Shocks']))
            save_data_s[parameters_list[i][1]]['Magnitude'] = [round(value, 1) for value in
                                                               save_data_s[parameters_list[i][1]]['Magnitude']]
    context = {'items': items}
    return render(request, 'output1.html', context)

def plot_view(request):
    # Create your plots using Matplotlib
    global plot_data
    plots=[]
    for key, value in plot_data.items():
        y = np.array(value)
        x = np.arange(1, len(y) + 1, 1)
        fig, ax = plt.subplots()
        ax.plot(x, y, color='red', linewidth=2)  # Adjust line width and color

        ax.set_title(key, fontsize=16, fontweight='bold')  # Adjust title font size and weight
        ax.set_xlabel('Time', fontsize=12)  # Adjust label font size
        ax.set_ylabel('Y', fontsize=12)

        ax.spines['top'].set_visible(False)  # Hide the top border
        ax.spines['right'].set_visible(False)  # Hide the right border
        ax.spines['bottom'].set_linewidth(0.5)  # Adjust bottom border thickness
        ax.spines['left'].set_linewidth(0.5)  # Adjust left border thickness
        ax.xaxis.set_tick_params(width=0.5)  # Adjust x-axis tick thickness
        ax.yaxis.set_tick_params(width=0.5)  # Adjust y-axis tick thickness

        plt.tight_layout()  # Improve spacing between subplots

        # Generate a random filename for the plot
        filename = f'{uuid.uuid4()}.png'
        plot_path = os.path.join('media', 'plots', filename)

        # Create the subdirectory if it doesn't exist
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)

        # Save the plot to the file
        fig.savefig(plot_path)
        plt.close(fig)

        # Convert the PNG image to a base64-encoded string
        with open(plot_path, 'rb') as f:
            image_bytes = f.read()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        plots.append(image_base64)
    global plot_data_s
    C=[]
    X = []
    Y = []
    S = []
    for keys, values in plot_data_s.items():
        count = 1
        x = []
        y = []
        i = 0
        while count <= values['Time'][-1] or i < len(values['Time']):
            if values['Time'][i] == count:
                x.append(count)
                y.append(values['Magnitude'][i])
                i += 1
            else:
                x.append(count)
                y.append(0)
            count += 1
        X.append(x)
        Y.append(y)
        print(plot_data_s)
        S.append(values['Shocks'])
        C.append(keys)

    for i in range(len(X)):
        fig, ax = plt.subplots()
        ax.plot(X[i], Y[i], linewidth=2, color='blue')  # Adjust line width and color

        j, k = 0, 0
        while j < len(X[i]) or k < len(S[i]):
            if X[i][j] == S[i][k]:
                plt.plot(X[i][j], Y[i][j], marker='o', markersize=10, color='red')
                k += 1
            j += 1

        ax.set_title(C[i], fontsize=16, fontweight='bold')  # Adjust title font size and weight
        ax.set_xlabel('Time', fontsize=12)  # Adjust label font size
        ax.set_ylabel('Shock Magnitude', fontsize=12)

        ax.spines['top'].set_visible(False)  # Hide the top border
        ax.spines['right'].set_visible(False)  # Hide the right border
        ax.spines['bottom'].set_linewidth(0.5)  # Adjust bottom border thickness
        ax.spines['left'].set_linewidth(0.5)  # Adjust left border thickness
        ax.xaxis.set_tick_params(width=0.5)  # Adjust x-axis tick thickness
        ax.yaxis.set_tick_params(width=0.5)  # Adjust y-axis tick thickness

        plt.tight_layout()  # Improve spacing between subplots

        # Generate a random filename for the plot
        filename = f'{uuid.uuid4()}.png'
        plot_path = os.path.join('media', 'plots', filename)

        # Create the subdirectory if it doesn't exist
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)

        # Save the plot to the file
        fig.savefig(plot_path)
        plt.close(fig)

        # Convert the PNG image to a base64-encoded string
        with open(plot_path, 'rb') as f:
            image_bytes = f.read()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        plots.append(image_base64)
    return render(request, 'plots.html', {'plots': plots})

def plot_system(request):
    global ttf
    time_to_failure = np.cumsum(ttf)
    # def round_up_to_nearest_100(num):
    #     return math.ceil(num / 100) * 100
    # def NHPP(T):
    #     T_star = round_up_to_nearest_100(T[-1])
    #     ln_T = np.log(T)
    #     n = len(T)
    #     sum_T = sum(T)
    #     sum_ln_T = sum(ln_T)
    #     beta = n / ((n * np.log(T_star)) - sum_ln_T)
    #     eta = n / (T_star ** beta)
    #     return beta, eta
    # print(time_to_failure)
    # b, a = NHPP(time_to_failure)
    # y = np.linspace(0, int(time_to_failure[-1]), int(time_to_failure[-1])*10)
    # m = a * b * (y ** (b - 1))

    def model(t, A, B):
        return A * (t ** B)

    # Example data points
    t = time_to_failure.copy()  # Time values
    y = np.array(np.arange(1, len(t) + 1))  # Corresponding data points

    # Fit the data to the model
    params, _ = curve_fit(model, t, y)

    # Extract the fitted parameters
    A, B = params

    # Generate the fitted curve
    t_fit = np.linspace(t.min(), t.max(), 100)
    y_fit = model(t_fit, A, B)

    # Plot the data and the fitted curve
    # plt.scatter(t, y, label='Data')
    # plt.plot(t_fit, y_fit, label='Fit')
    # plt.xlabel('Time')
    # plt.ylabel('Number of Failures')
    # plt.legend()
    # plt.show()

    print("Fitted parameters: A =", A, "B =", B)

    # Styling
    sns.set(style='whitegrid', font_scale=1.2)
    sns.set_palette('colorblind')

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t_fit, y_fit, label='Fit')
    ax.scatter(t, y, label='Data')

    # Title and labels
    ax.set_title('System Failure NHPP')
    ax.set_xlabel('Time')
    ax.set_ylabel('Number of Failures')

    # Cursor hover-over functionality
    cursor = mplcursors.cursor(ax, hover=True)
    cursor.connect('add',
                   lambda sel: sel.annotation.set_text(f'Time: {sel.target[0]:.2f}\nIntensity: {sel.target[1]:.2f}'))

    # Show the plot

    # Generate a random filename for the plot
    filename = f'{uuid.uuid4()}.png'
    plot_path = os.path.join('media', 'plots', filename)

    # Create the subdirectory if it doesn't exist
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    # Save the plot to the file
    fig.savefig(plot_path)
    plt.close(fig)

    # Convert the PNG image to a base64-encoded string
    with open(plot_path, 'rb') as f:
        image_bytes = f.read()
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')

    # Pass the base64-encoded image string to the template context
    context = {
        'image_base64': image_base64,
        'a': A,
        'b': B
    }

    return render(request, 'plot_system.html', context)

def save_data_to_csv(request):
    global ttf, component
    time_to_failure = np.cumsum(ttf)
    comp = component.copy()

    #SAVING WEIBULL
    global save_data_w
    save_weibull=save_data_w.copy()
    final_save_w= {}
    for key, value in save_weibull.items():
        final_save_w[key] = {'Failure Time': value}
    #SAVING DEGRADATION
    new_data = {}
    for key, values in save_data_d.items():
        new_data[key] = {
            'magnitude': values,
            'time': list(range(1, len(values) + 1))
        }
    final_save_d = {}
    for key, value in new_data.items():
        new_values = {}
        for sub_key, sub_value in value.items():
            mode = "Increase"
            for i in range(len(sub_value)):
                if i % 20 == 0:
                    if sub_key not in new_values:
                        new_values[sub_key] = []
                        new_values[sub_key].append(sub_value[i])
                        if sub_key == 'magnitude':
                            if sub_value[i] > sub_value[i + 1]:
                                mode = "Decrease"
                            else:
                                mode = "Increase"
                            new_values['State'] = []
                            if mode == 'Decrease' or mode == 'Increase':
                                new_values['State'].append(0)
                    else:
                        new_values[sub_key].append(sub_value[i])
                        if sub_key == 'magnitude':
                            if (mode == "Decrease" and sub_value[i - 20] > sub_value[i]) or (
                                    mode == "Increase" and sub_value[i - 20] < sub_value[i]):
                                new_values['State'].append(0)
                            else:
                                new_values['State'].append(1)
        final_save_d[key] = new_values
    #FOR SHOCKS
    final_save_s = save_data_s.copy()
    for key in final_save_s.keys():
        shocks = final_save_s[key]['Shocks']
        final_save_s[key]['State'] = [1 if time in shocks else 0 for time in final_save_s[key]['Time']]
    for key in final_save_s:
        if 'Shocks' in final_save_s[key]:
            del final_save_s[key]['Shocks']
    #FOR SYSTEM
    final_save = {}
    time_to_failure=time_to_failure.tolist()
    final_save['Time'] = time_to_failure
    final_save['Component'] = comp
    for key, values in final_save_d.items():
        filename = key + '_deg.csv'
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(values.keys())  # Write column headers
            writer.writerows(zip(*values.values()))  # Write column values

        print(f"CSV_degrade file '{filename}' has been created.")
    for key, values in final_save_s.items():
        filename = key + '_shock.csv'
        with open(filename, 'w', newline='') as file:
            try:
                writer = csv.writer(file)
                writer.writerow(values.keys())  # Write column headers
                writer.writerows(zip(*values.values()))  # Write column values
            except Exception as e1:
                print(e1)
        print(f"CSV_shock file '{filename}' has been created.")
    for key, values in final_save_w.items():
        filename = key + '_weibull.csv'
        with open(filename, 'w', newline='') as file:
            try:
                writer = csv.writer(file)
                writer.writerow(values.keys())  # Write column headers
                writer.writerows(zip(*values.values()))  # Write column values
            except Exception as e2:
                print(e2)

        print(f"CSV_Weibull file '{filename}' has been created.")
    print(final_save)
    keys = list(final_save.keys())
    # Create a CSV file and write the data
    with open('System.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write the header row
        writer.writerow(keys)

        # Write the data rows
        for row in zip(*final_save.values()):
            writer.writerow(row)


    return render(request, 'saved.html', context={})


