# Generated by Django 2.1.2 on 2018-10-29 07:31

import django.core.validators
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0003_auto_20181029_0729'),
    ]

    operations = [
        migrations.AlterField(
            model_name='housedetails',
            name='contact_no',
            field=models.CharField(max_length=10, validators=[django.core.validators.RegexValidator(message="Phone number must be entered in the format: '+999999999'. Up to 10 digits allowed.", regex='^\\+?1?\\d{6,10}$')]),
        ),
    ]