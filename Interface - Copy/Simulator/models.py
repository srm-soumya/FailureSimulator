from django.db import models
from django.template.defaultfilters import slugify
from django.contrib.auth.models import User
from django.urls import reverse


# Create your models here.
# class Component(models.Model):
#     comp=models.CharField(max_length=3)
#     def __str__(self):
#         return self.comp
class Weibull(models.Model):
    component_w = models.CharField(max_length=30)
    eta = models.CharField(max_length=30)
    beta = models.CharField(max_length=30)

    def __str__(self):
        return f"{self.component_w} - {self.eta} - {self.beta} "


class BasicShock(models.Model):
    component_bs = models.CharField(max_length=30)
    expected_arrival = models.CharField(max_length=30)
    threshold_shocks = models.CharField(max_length=30)

    def __str__(self):
        return f"{self.component_bs} -{self.expected_arrival} - {self.threshold_shocks} "


class ExtremeShock(models.Model):
    component_es = models.CharField(max_length=30)
    expected_arrival = models.CharField(max_length=30)
    threshold_shocks = models.CharField(max_length=30)
    mean_magnitude = models.CharField(max_length=30)
    std_magnitude = models.CharField(max_length=30)
    threshold_magnitude = models.CharField(max_length=30)

    def __str__(self):
        return f"{self.component_es} -{self.expected_arrival} - {self.threshold_shocks} - {self.mean_magnitude} -{self.std_magnitude} -{self.threshold_magnitude} - "


class CumulativeShock(models.Model):
    component_cs = models.CharField(max_length=30)
    expected_arrival = models.CharField(max_length=30)
    threshold_shocks = models.CharField(max_length=30)
    mean_magnitude = models.CharField(max_length=30)
    std_magnitude = models.CharField(max_length=30)
    threshold_magnitude = models.CharField(max_length=30)
    def __str__(self):
        return f"{self.component_cs} -{self.expected_arrival} - {self.threshold_shocks} - {self.mean_magnitude} -{self.std_magnitude} -{self.threshold_magnitude} - "
class Degradation1(models.Model):
    component_d1 = models.CharField(max_length=30)
    # z = models.CharField(max_length=30)
    low_in = models.FloatField(default=0.0)
    high_in = models.FloatField(default=0.0)
    threshold = models.FloatField(default=0.0)
    p1 = models.FloatField(default=0.0)
    p2 = models.FloatField(default=0.0)
    dist = models.CharField(max_length=30, default='default_value')

    def __str__(self):
        return f"{self.component_d1}  - {self.low_in}  - {self.high_in}  - {self.threshold}  - {self.p1}  - {self.p2}  - {self.dist}"


class Degradation2(models.Model):
    component_d2 = models.CharField(max_length=30)
    # z = models.CharField(max_length=30)
    low_in = models.FloatField(default=0.0)
    high_in = models.FloatField(default=0.0)
    threshold = models.FloatField(default=0.0)
    p1 = models.FloatField(default=0.0)
    p2 = models.FloatField(default=0.0)
    dist = models.CharField(max_length=30, default='default_value')

    def __str__(self):
        return f"{self.component_d2}  - {self.low_in}  - {self.high_in}  - {self.threshold}  - {self.p1}  - {self.p2}  - {self.dist} "


class Degradation3(models.Model):
    component_d3 = models.CharField(max_length=30)
    # z = models.CharField(max_length=30)
    # y = models.CharField(max_length=30)
    low_in = models.FloatField(default=0.0)
    high_in = models.FloatField(default=0.0)
    threshold = models.FloatField(default=0.0)
    p1 = models.FloatField(default=0.0)
    p2 = models.FloatField(default=0.0)
    p3 = models.FloatField(default=0.0)
    p4 = models.FloatField(default=0.0)
    p5 = models.FloatField(default=0.0)
    p6 = models.FloatField(default=0.0)
    dist = models.CharField(max_length=30, default='default_value')

    def __str__(self):
        return f"{self.component_d3}  - {self.low_in}  - {self.high_in} - {self.threshold}  - {self.p1}  - {self.p2}  - {self.p3}  - {self.p4}  - {self.p5}  - {self.p6}  - {self.dist} "


class Degradation4(models.Model):
    component_d4 = models.CharField(max_length=30)
    # z = models.CharField(max_length=30)
    # y = models.CharField(max_length=30)'
    low_in = models.FloatField(default=0.0)
    high_in = models.FloatField(default=0.0)
    threshold = models.FloatField(default=0.0)
    p1 = models.FloatField(default=0.0)
    p2 = models.FloatField(default=0.0)
    p3 = models.FloatField(default=0.0)
    p4 = models.FloatField(default=0.0)
    dist = models.CharField(max_length=30, default='default_value')

    def __str__(self):
        return f"{self.component_d4}  - {self.low_in}  - {self.high_in} - {self.threshold}  - {self.p1}  - {self.p2}  - {self.p3}  - {self.p4}  - {self.dist}"

class Degradation5(models.Model):
    component_d5 = models.CharField(max_length=30)
    # z = models.CharField(max_length=30)
    # y = models.CharField(max_length=30)
    low_in = models.FloatField(default=0.0)
    high_in = models.FloatField(default=0.0)
    threshold = models.FloatField(default=0.0)
    p1 = models.FloatField(default=0.0)
    p2 = models.FloatField(default=0.0)
    dist = models.CharField(max_length=30, default='default_value')

    def __str__(self):
        return f"{self.component_d5}  - {self.low_in}  - {self.high_in} - {self.threshold}  - {self.p1}  - {self.p2}  - {self.dist} "
class MyModel(models.Model):
    input1Value = models.CharField(max_length=50)
    input2Value = models.CharField(max_length=50)

    def __str__(self):
        return f"{self.input1Value} - {self.input2Value}"
