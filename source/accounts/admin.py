from django.contrib import admin


# admin.site.register(Author)
# admin.site.register(Book)
from django.contrib.admin import AdminSite
from django.utils.translation import ugettext_lazy



admin.site.site_header = 'Fake Note Detector'
admin.site.site_title = 'Fake Note Detector Admin'
admin.site.index_title = 'Administration'