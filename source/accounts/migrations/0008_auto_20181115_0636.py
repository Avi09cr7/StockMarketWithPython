# Generated by Django 2.1.2 on 2018-11-15 06:36

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0007_auto_20181115_0601'),
    ]

    operations = [
        migrations.AddField(
            model_name='onlineusers',
            name='password',
            field=models.CharField(default=123, max_length=30),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='onlineusers',
            name='email',
            field=models.EmailField(max_length=50, unique=True),
        ),
    ]
