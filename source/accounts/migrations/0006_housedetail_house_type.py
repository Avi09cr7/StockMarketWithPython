# Generated by Django 2.1.2 on 2018-10-30 06:48

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0005_auto_20181029_0734'),
    ]

    operations = [
        migrations.AddField(
            model_name='housedetail',
            name='house_type',
            field=models.CharField(default='', max_length=100),
            preserve_default=False,
        ),
    ]
