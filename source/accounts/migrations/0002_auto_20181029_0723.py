# Generated by Django 2.1.2 on 2018-10-29 07:23

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='HouseDetails',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('owner_name', models.CharField(max_length=100)),
                ('house_no', models.PositiveIntegerField(max_length=100)),
                ('house_name', models.CharField(max_length=100)),
                ('pincode', models.PositiveIntegerField(max_length=6)),
                ('address', models.TextField(max_length=200)),
                ('house_cost', models.PositiveIntegerField(max_length=10)),
                ('contact_no', models.PositiveIntegerField(max_length=10)),
                ('email', models.EmailField(max_length=50)),
                ('rented', models.BooleanField()),
                ('rooms', models.CharField(max_length=4)),
                ('rent_per_room', models.CharField(max_length=5)),
                ('house_image', models.FileField(upload_to='house_image')),
                ('room_image', models.FileField(upload_to='room_image')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
            ],
            options={
                'db_table': 'house_details',
            },
        ),
        migrations.RemoveField(
            model_name='uploadeddocuments',
            name='user',
        ),
        migrations.DeleteModel(
            name='UploadedDocuments',
        ),
    ]
