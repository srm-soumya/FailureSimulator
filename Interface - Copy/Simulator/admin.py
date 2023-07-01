from django.contrib import admin

# Register your models here.
from Simulator.models import Weibull, BasicShock, ExtremeShock, CumulativeShock, Degradation1, Degradation2, Degradation3, Degradation4,Degradation5
admin.site.register(Weibull)
admin.site.register(BasicShock)
admin.site.register(ExtremeShock)
admin.site.register(CumulativeShock)
admin.site.register(Degradation1)
admin.site.register(Degradation2)
admin.site.register(Degradation3)
admin.site.register(Degradation4)
admin.site.register(Degradation5)

