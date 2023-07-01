# Generated by Django 4.2 on 2023-05-15 15:39

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Simulator', '0006_degradation1_dist'),
    ]

    operations = [
        migrations.AddField(
            model_name='degradation2',
            name='dist',
            field=models.CharField(default='default_value', max_length=30),
        ),
        migrations.AddField(
            model_name='degradation2',
            name='p1',
            field=models.FloatField(default=0.0),
        ),
        migrations.AddField(
            model_name='degradation2',
            name='p2',
            field=models.FloatField(default=0.0),
        ),
        migrations.AddField(
            model_name='degradation3',
            name='dist',
            field=models.CharField(default='default_value', max_length=30),
        ),
        migrations.AddField(
            model_name='degradation3',
            name='p1',
            field=models.FloatField(default=0.0),
        ),
        migrations.AddField(
            model_name='degradation3',
            name='p2',
            field=models.FloatField(default=0.0),
        ),
        migrations.AddField(
            model_name='degradation3',
            name='p3',
            field=models.FloatField(default=0.0),
        ),
        migrations.AddField(
            model_name='degradation3',
            name='p4',
            field=models.FloatField(default=0.0),
        ),
        migrations.AddField(
            model_name='degradation3',
            name='p5',
            field=models.FloatField(default=0.0),
        ),
        migrations.AddField(
            model_name='degradation3',
            name='p6',
            field=models.FloatField(default=0.0),
        ),
        migrations.AddField(
            model_name='degradation4',
            name='dist',
            field=models.CharField(default='default_value', max_length=30),
        ),
        migrations.AddField(
            model_name='degradation4',
            name='p1',
            field=models.FloatField(default=0.0),
        ),
        migrations.AddField(
            model_name='degradation4',
            name='p2',
            field=models.FloatField(default=0.0),
        ),
        migrations.AddField(
            model_name='degradation4',
            name='p3',
            field=models.FloatField(default=0.0),
        ),
        migrations.AddField(
            model_name='degradation4',
            name='p4',
            field=models.FloatField(default=0.0),
        ),
        migrations.AddField(
            model_name='degradation5',
            name='p1',
            field=models.FloatField(default=0.0),
        ),
        migrations.AddField(
            model_name='degradation5',
            name='p2',
            field=models.FloatField(default=0.0),
        ),
        migrations.AddField(
            model_name='degradation5',
            name='p3',
            field=models.FloatField(default=0.0),
        ),
        migrations.AddField(
            model_name='degradation5',
            name='p4',
            field=models.FloatField(default=0.0),
        ),
        migrations.AlterField(
            model_name='degradation1',
            name='threshold',
            field=models.FloatField(default=0.0),
        ),
        migrations.AlterField(
            model_name='degradation2',
            name='threshold',
            field=models.FloatField(default=0.0),
        ),
    ]
