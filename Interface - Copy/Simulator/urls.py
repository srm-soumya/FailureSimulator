from django.contrib import admin
from django.urls import path
from Simulator import views
from django.conf import settings
from django.conf.urls.static import static
urlpatterns = [
    path('', views.index, name='home'),
path('about', views.about, name='about'),
path('services', views.services, name='services'),
path('contact', views.contact, name='contact'),
path('weibull', views.weibull, name='weibull'),
path('simulate', views.simulate, name='simulate'),
path('select_function', views.select_function,name='select_function'),
path('save_to_weibull/', views.save_to_weibull, name='save_to_weibull'),
path('save_to_basicshock/', views.save_to_basicshock, name='save_to_basicshock'),
path('save_to_extremeshock/', views.save_to_extremeshock, name='save_to_extremeshock'),
path('save_to_cumulativeshock/', views.save_to_cumulativeshock, name='save_to_cumulativeshock'),
path('save_to_degradation1/', views.save_to_degradation1, name='save_to_degradation1'),
path('save_to_degradation2/', views.save_to_degradation2, name='save_to_degradation2'),
path('save_to_degradation3/', views.save_to_degradation3, name='save_to_degradation3'),
path('save_to_degradation4/', views.save_to_degradation4, name='save_to_degradation4'),
path('save_to_degradation5/', views.save_to_degradation5, name='save_to_degradation5'),
path('show_data/', views.show_data, name='show_data'),
path('start/', views.start, name='start'),
path('process_data/', views.process_data, name='process_data'),
path('reset-database/', views.reset_database, name='reset_database'),
path('plots/', views.plot_view, name='plots'),
path('plot_system/', views.plot_system, name='plot_system'),
path('save_data_to_csv/', views.save_data_to_csv, name='save_data_to_csv'),
path('dependency/', views.dependency, name='dependency'),
path('saved/', views.save_data_to_csv, name='saved')
    ]
urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
