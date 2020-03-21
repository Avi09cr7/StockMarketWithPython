from django.contrib import messages
from django.contrib.auth import login, authenticate, REDIRECT_FIELD_NAME
from django.contrib.auth.tokens import default_token_generator
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.views import (
    LogoutView as BaseLogoutView, PasswordChangeView as BasePasswordChangeView,
    PasswordResetDoneView as BasePasswordResetDoneView, PasswordResetConfirmView as BasePasswordResetConfirmView,
)
from django.shortcuts import get_object_or_404, redirect
from django.utils.crypto import get_random_string
from django.utils.decorators import method_decorator
from django.utils.http import is_safe_url
from django.utils.encoding import force_bytes
from django.utils.http import urlsafe_base64_encode
from django.utils.translation import gettext_lazy as _
from django.views.decorators.cache import never_cache
from django.views.decorators.csrf import csrf_protect
from django.views.decorators.debug import sensitive_post_parameters
from django.views.generic import View, FormView
from django.conf import settings

from .utils import (
    send_activation_email, send_reset_password_email, send_forgotten_username_email, send_activation_change_email,
)
from .forms import (
    SignInViaUsernameForm, SignInViaEmailForm, SignInViaEmailOrUsernameForm, SignUpForm,
    RestorePasswordForm, RestorePasswordViaEmailOrUsernameForm, RemindUsernameForm,
    ResendActivationCodeForm, ResendActivationCodeViaEmailForm, ChangeProfileForm, ChangeEmailForm,
)
from .models import Activation


class GuestOnlyView(View):
    def dispatch(self, request, *args, **kwargs):
        # Redirect to the index page if the user already authenticated
        if request.user.is_authenticated:
            return redirect(settings.LOGIN_REDIRECT_URL)

        return super().dispatch(request, *args, **kwargs)


class LogInView(GuestOnlyView, FormView):
    template_name = 'accounts/log_in.html'

    @staticmethod
    def get_form_class(**kwargs):
        if settings.DISABLE_USERNAME or settings.LOGIN_VIA_EMAIL:
            return SignInViaEmailForm

        if settings.LOGIN_VIA_EMAIL_OR_USERNAME:
            return SignInViaEmailOrUsernameForm

        return SignInViaUsernameForm

    @method_decorator(sensitive_post_parameters('password'))
    @method_decorator(csrf_protect)
    @method_decorator(never_cache)
    def dispatch(self, request, *args, **kwargs):
        # Sets a test cookie to make sure the user has cookies enabled
        request.session.set_test_cookie()

        return super().dispatch(request, *args, **kwargs)

    def form_valid(self, form):
        request = self.request

        # If the test cookie worked, go ahead and delete it since its no longer needed
        if request.session.test_cookie_worked():
            request.session.delete_test_cookie()

        # The default Django's "remember me" lifetime is 2 weeks and can be changed by modifying
        # the SESSION_COOKIE_AGE settings' option.
        if settings.USE_REMEMBER_ME:
            if not form.cleaned_data['remember_me']:
                request.session.set_expiry(0)

        login(request, form.user_cache)

        redirect_to = request.POST.get(REDIRECT_FIELD_NAME, request.GET.get(REDIRECT_FIELD_NAME))
        url_is_safe = is_safe_url(redirect_to, allowed_hosts=request.get_host(), require_https=request.is_secure())

        if url_is_safe:
            return redirect(redirect_to)

        return redirect(settings.LOGIN_REDIRECT_URL)


class SignUpView(GuestOnlyView, FormView):
    template_name = 'accounts/sign_up.html'
    form_class = SignUpForm

    def form_valid(self, form):
        request = self.request
        user = form.save(commit=False)

        if settings.DISABLE_USERNAME:
            # Set a temporary username
            user.username = get_random_string()
        else:
            user.username = form.cleaned_data['username']

        if settings.ENABLE_USER_ACTIVATION:
            user.is_active = False

        # Create a user record
        user.save()

        # Change the username to the "user_ID" form
        if settings.DISABLE_USERNAME:
            user.username = f'user_{user.id}'
            user.save()

        if settings.ENABLE_USER_ACTIVATION:
            code = get_random_string(20)

            act = Activation()
            act.code = code
            act.user = user
            act.save()

            send_activation_email(request, user.email, code)

            messages.success(
                request, _('You are signed up. To activate the account, follow the link sent to the mail.'))
        else:
            raw_password = form.cleaned_data['password1']

            user = authenticate(username=user.username, password=raw_password)
            login(request, user)

            messages.success(request, _('You are successfully signed up!'))

        return redirect('index')


class ActivateView(View):
    @staticmethod
    def get(request, code):
        act = get_object_or_404(Activation, code=code)

        # Activate profile
        user = act.user
        user.is_active = True
        user.save()

        # Remove the activation record
        act.delete()

        messages.success(request, _('You have successfully activated your account!'))

        return redirect('accounts:log_in')


class ResendActivationCodeView(GuestOnlyView, FormView):
    template_name = 'accounts/resend_activation_code.html'

    @staticmethod
    def get_form_class(**kwargs):
        if settings.DISABLE_USERNAME:
            return ResendActivationCodeViaEmailForm

        return ResendActivationCodeForm

    def form_valid(self, form):
        user = form.user_cache

        activation = user.activation_set.first()
        activation.delete()

        code = get_random_string(20)

        act = Activation()
        act.code = code
        act.user = user
        act.save()

        send_activation_email(self.request, user.email, code)

        messages.success(self.request, _('A new activation code has been sent to your email address.'))

        return redirect('accounts:resend_activation_code')


class RestorePasswordView(GuestOnlyView, FormView):
    template_name = 'accounts/restore_password.html'

    @staticmethod
    def get_form_class(**kwargs):
        if settings.RESTORE_PASSWORD_VIA_EMAIL_OR_USERNAME:
            return RestorePasswordViaEmailOrUsernameForm

        return RestorePasswordForm

    def form_valid(self, form):
        user = form.user_cache
        token = default_token_generator.make_token(user)
        uid = urlsafe_base64_encode(force_bytes(user.pk)).decode()

        send_reset_password_email(self.request, user.email, token, uid)

        return redirect('accounts:restore_password_done')


class ChangeProfileView(LoginRequiredMixin, FormView):
    template_name = 'accounts/profile/change_profile.html'
    form_class = ChangeProfileForm

    def get_initial(self):
        user = self.request.user
        initial = super().get_initial()
        initial['first_name'] = user.first_name
        initial['last_name'] = user.last_name
        return initial

    def form_valid(self, form):
        user = self.request.user
        user.first_name = form.cleaned_data['first_name']
        user.last_name = form.cleaned_data['last_name']
        user.save()

        messages.success(self.request, _('Profile data has been successfully updated.'))

        return redirect('accounts:change_profile')


class ChangeEmailView(LoginRequiredMixin, FormView):
    template_name = 'accounts/profile/change_email.html'
    form_class = ChangeEmailForm

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs['user'] = self.request.user
        return kwargs

    def get_initial(self):
        initial = super().get_initial()
        initial['email'] = self.request.user.email
        return initial

    def form_valid(self, form):
        user = self.request.user
        email = form.cleaned_data['email']

        if settings.ENABLE_ACTIVATION_AFTER_EMAIL_CHANGE:
            code = get_random_string(20)

            act = Activation()
            act.code = code
            act.user = user
            act.email = email
            act.save()

            send_activation_change_email(self.request, email, code)

            messages.success(self.request, _('To complete the change of email address, click on the link sent to it.'))
        else:
            user.email = email
            user.save()

            messages.success(self.request, _('Email successfully changed.'))

        return redirect('accounts:change_email')


class ChangeEmailActivateView(View):
    @staticmethod
    def get(request, code):
        act = get_object_or_404(Activation, code=code)

        # Change the email
        user = act.user
        user.email = act.email
        user.save()

        # Remove the activation record
        act.delete()

        messages.success(request, _('You have successfully changed your email!'))

        return redirect('accounts:change_email')


class RemindUsernameView(GuestOnlyView, FormView):
    template_name = 'accounts/remind_username.html'
    form_class = RemindUsernameForm

    def form_valid(self, form):
        user = form.user_cache
        send_forgotten_username_email(user.email, user.username)

        messages.success(self.request, _('Your username has been successfully sent to your email.'))

        return redirect('accounts:remind_username')


class ChangePasswordView(BasePasswordChangeView):
    template_name = 'accounts/profile/change_password.html'

    def form_valid(self, form):
        # Change the password
        user = form.save()

        # Re-authentication
        login(self.request, user)

        messages.success(self.request, _('Your password was changed.'))

        return redirect('accounts:change_password')


class RestorePasswordConfirmView(BasePasswordResetConfirmView):
    template_name = 'accounts/restore_password_confirm.html'

    def form_valid(self, form):
        # Change the password
        form.save()

        messages.success(self.request, _('Your password has been set. You may go ahead and log in now.'))

        return redirect('accounts:log_in')


class RestorePasswordDoneView(BasePasswordResetDoneView):
    template_name = 'accounts/restore_password_done.html'


class LogOutView(LoginRequiredMixin, BaseLogoutView):
    template_name = 'accounts/log_out.html'

from .forms import OnlineUsersForm
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
import json;
@csrf_exempt
def registerUser(request):
    data = {'msg':'Failed to save user'}
    if(request.method=='GET'):
        form1 = OnlineUsersForm(request.GET)

        # http://127.0.0.1:8000/accounts/registerUser?name=hello&contact_no=9540063573&latitude=28.77&longitude=77.66&email=user@gmail.com&usertype=user&password=123
        name = request.GET.get('name', '')
        form1.name=name
        #         fields = ('name', 'contact_no','latitude','longitude','email','usertype','password' )
        form1.contact_no= request.GET.get('contact_no', '')
        form1.latitude= request.GET.get('latitude', '')
        form1.longitude= request.GET.get('longitude', '')
        form1.email= request.GET.get('email', '')
        form1.usertype= request.GET.get('usertype', '')
        form1.password= request.GET.get('password', '')
        print('name', name)
        print('name', form1.latitude)
        print('name', form1.longitude)
        print('name', form1.email)
        print('name', form1.usertype)
        print('name', form1.password)
        saved=False
        if form1.is_valid():
            form1.save()
            saved = True
        if saved:

            data = {'msg': 'User Created Successfully'}
            json_data = json.dumps(data)
            # print(json_data)
        else:
            print(form1.errors)
            data = {'msg':"User Already Exsists"}
            json_data = json.dumps(data)
            print(json_data)

        return HttpResponse(json_data, content_type='application/json')
    else:
        return HttpResponse(data, content_type='application/json')

from .models import OnlineUsers
@csrf_exempt
def loginUser(request):
    data = {'msg':'Failed to login'}
    if(request.method=='GET'):

        # http://127.0.0.1:8000/accounts/registerUser?name=hello&contact_no=9540063573&latitude=28.77&longitude=77.66&email=user@gmail.com&usertype=user&password=123
        email = request.GET.get('email', '')
        password = request.GET.get('password', '')
        print('SELECT * FROM online_users where email="'+str(email)+'" AND password="'+password+'"')
        check_user=False
        user=''
        mobile=''
        for p in OnlineUsers.objects.raw('SELECT * FROM online_users where email="'+str(email)+'" AND password="'+password+'"'):
            check_user=True
            user=p.name
            mobile=p.contact_no


        if check_user:

            data = {'msg': 'Welcome '+user,'mobile':mobile}
            json_data = json.dumps(data)
            # print(json_data)
        else:

            data = {'msg':"Incorrect id password"}
            json_data = json.dumps(data)
            print(json_data)

        return HttpResponse(json_data, content_type='application/json')
    else:
        return HttpResponse(data, content_type='application/json')
from math import sin, cos, sqrt, atan2, radians

@csrf_exempt
def emergencyMsg(request):
    data = {'msg':'Failed to login'}
    if(request.method=='GET'):

        # http://127.0.0.1:8000/accounts/registerUser?name=hello&contact_no=9540063573&latitude=28.77&longitude=77.66&email=user@gmail.com&usertype=user&password=123
        email = request.GET.get('email', '')
        latitude = request.GET.get('latitude', '')
        longitude = request.GET.get('longitude', '')
        print(email)
        print(latitude)
        print(longitude)
        # latitude=28.77
        # longitude=77.77
        print('SELECT * FROM online_users where email=!"'+str(email)+'"')
        check_user=False
        user=''
        mobile=''
        jsonarray=[]
        for p in OnlineUsers.objects.raw('SELECT * FROM online_users where email!="'+str(email)+'"'):


            # approximate radius of earth in km
            R = 6373.0

            lat1 = radians(p.latitude)
            lon1 = radians(p.longitude)
            lat2 = radians(float(latitude))
            lon2 = radians(float(longitude))

            dlon = lon2 - lon1
            dlat = lat2 - lat1

            a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))

            distance = R * c
            if distance<=0.08:
                print(distance)
                check_user=True
                user=p.name
                mobile=p.contact_no
                data = {'no': mobile}
                j=json.dumps(data)
                jsonarray.append(j)



        return HttpResponse([jsonarray], content_type='application/json')
    else:
        return HttpResponse(data, content_type='application/json')

@csrf_exempt
def updateLocation(request):
    data = {'msg':'Failed to Update'}
    if(request.method=='GET'):

        # http://127.0.0.1:8000/accounts/updateLocation?email=1&latitude=28.77&longitude=77.66
        email = request.GET.get('email', '')
        latitude = request.GET.get('latitude', '')
        longitude = request.GET.get('longitude', '')
        sql='update online_users set latitude='+latitude+', longitude='+longitude+' where email="'+str(email)+'"'
        print(sql)
        # check_user=False
        # user=''
        # obj=OnlineUsers.objects.filter(email=email)
        # obj.latitude=latitude
        # obj.latitude=longitude
        # obj.save()
        OnlineUsers.objects.filter(email=email).update(latitude=latitude)
        OnlineUsers.objects.filter(email=email).update(longitude=longitude)

        data = {'msg': 'saved'}
        print(data)
        return HttpResponse(json.dumps(data), content_type='application/json')
    else:
        return HttpResponse(data, content_type='application/json')

from django.template.defaulttags import register

@register.filter(name='show_error')
def show_error(dictionary):
    try:
        return dictionary.values()[0][0]
    except (TypeError,IndexError,AttributeError):
        return 'tip: try again'

from django.shortcuts import render, redirect

from django.core.files.storage import FileSystemStorage

train_data = []
train_labels = []
test_data = []

import warnings

warnings.filterwarnings('ignore')
import csv




import numpy as np
from sklearn.svm import SVR


def predictStockPrice(request):
    if request.method == "POST":
        curdat=request.POST.get('txtdate')
        print(curdat)
        fs=FileSystemStorage()
        dates = []
        prices = []
        stockList = []
        stockList.clear()

        def get_data(filename):
            # Date,Open,High,Low,Close,Volume
            '''
            Reads data from a file (snap.csv) and adds data to
            the lists dates and prices
            '''
            # Use the with as block to open the file and assign it to csvfile
            with open(filename, 'r') as csvfile:
                # csvFileReader allows us to iterate over every row in our csv file
                csvFileReader = csv.reader(csvfile)
                next(csvFileReader)  # skipping column names
                for row in csvFileReader:

                    # print({str(row[0]), float(row[1])})
                    # print(row)
                    # stockList.append({int(row[0].split('-')[2]), float(row[1])})
                    dates.append(int(row[0].split('-')[2]))  # Only gets day of the month which is at index 0
                    prices.append(float(row[1]))  # Convert to float for more precision


            return

        def predict_price(dates, prices, x):
            print("dates" + str(dates))
            print("prices=" + str(prices))
            print("x" + str(x))
            '''
            Builds predictive model and graphs it
            This function creates 3 models, each of them will be a type of support vector machine.
            A support vector machine is a linear seperator. It takes data that is already classified and tries
            to predict a set of unclassified data.
            So if we only had two data classes it would look like this
            It will be such that the distances from the closest points in each of the two groups is farthest away.
            When we add a new data point in our graph depending on which side of the line it is we could classify it
            accordingly with the label. However, in this program we are not predicting a class label, so we don't
            need to classify instead we are predicting the next value in a series which means we want to use regression.
            SVM's can be used for regression as well. The support vector regression is a type of SVM that uses the space between
            data points as a margin of error and predicts the most likely next point in a dataset.
            The predict_prices returns predictions from each of our models

            '''
            dates = np.reshape(dates, (len(dates), 1))  # converting to matrix of n X 1
            print("dates1=" + str(dates))
            # Linear support vector regression model.
            # Takes in 3 parameters:
            # 	1. kernel: type of svm
            # 	2. C: penalty parameter of the error term
            # 	3. gamma: defines how far too far is.

            # Two things are required when using an SVR, a line with the largest minimum margin
            # and a line that correctly seperates as many instances as possible. Since we can't have both,
            # C determines how much we want the latter.

            # Next we make a polynomial SVR because in mathfolklore, the no free lunch theorum states that there are no guarantees for one optimization to work better
            # than the other. So we'll try both.

            # Finally, we create one more SVR using a radial basis function. RBF defines similarity to be the eucledian distance between two inputs
            # If both are right on top of each other, the max similarity is one, if too far it is a ze
            # svr_lin = SVR(kernel= 'linear', C= 1e3) # 1e3 denotes 1000
            # svr_poly = SVR(kernel= 'poly', C= 1e3, degree= 2)
            svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)  # defining the support vector regression models

            svr_rbf.fit(dates, prices)  # fitting the data points in the models
            # svr_lin.fit(dates, prices)
            # svr_poly.fit(dates, prices)

            # This plots the initial data points as black dots with the data label and plot
            # each of our models as well

            # plt.scatter(dates, prices, color= 'black', label= 'Data') # plotting the initial datapoints
            # The graphs are plotted with the help of SVR object in scikit-learn using the dates matrix as our parameter.
            # Each will be a distinct color and and give them a distinct label.
            # plt.plot(dates, svr_rbf.predict(dates), color= 'red', label= 'RBF model') # plotting the line made by the RBF kernel
            # plt.plot(dates,svr_lin.predict(dates), color= 'green', label= 'Linear model') # plotting the line made by linear kernel
            # plt.plot(dates,svr_poly.predict(dates), color= 'blue', label= 'Polynomial model') # plotting the line made by polynomial kernel
            # plt.xlabel('Date') # Setting the x-axis
            # plt.ylabel('Price') # Setting the y-axis
            # plt.title('Support Vector Regression') # Setting title
            # plt.legend() # Add legend
            # plt.show() # To display result on screen
            print(svr_rbf.predict(np.reshape([x], (len([x]), 1)))[0])
            return svr_rbf.predict(np.reshape([x], (len([x]), 1)))[0]  # returns predictions from each of our models

        # return svr_rbf.predict(np.reshape([x],(len([x]), 1)))[0], svr_lin.predict(np.reshape([x],(len([x]), 1)))[0], svr_poly.predict(np.reshape([x],(len([x]), 1)))[0] # returns predictions from each of our models

        get_data(fs.base_location+'/NSE-TATASTLBSL.csv')  # calling get_data method by passing the csv file to it
        predicted_price = predict_price(dates, prices, curdat)
        print('The predicted prices are:', predicted_price)
        stockList.append(['Prices','Dates'])
        count=0;
        for p in dates:
            # stockList.append({'Date':p,'Price':prices[count]})
            stockList.append([p,prices[count]])
            count=count+1
    print(stockList)
    return render(request, 'profile.html', {'prediction':predicted_price,'stockList':stockList,'dates':dates,'prices':prices})

def predictStockPriceDataset(request):
    if request.method == "POST" and request.FILES['documents']:
        curdat=request.POST.get('txtdate')
        fs = FileSystemStorage()
        # to save in db
        path_model = fs.base_location;
        print(path_model)
        fs = FileSystemStorage()
        # filename = 'model.svm'
        # model = pickle.load(open(filename, 'rb'))
        # model = models.load_model('music.h5')

        myfile = request.FILES['documents']
        temp = '1.mp3'
        fs = FileSystemStorage()
        filename_to_check = fs.save(temp, myfile)
        uploaded_file_url = fs.url(filename_to_check)

        path1 = fs.path(filename_to_check)
        print(path1)

        dates = []
        prices = []
        stockList = []
        stockList.clear()
        # pd.read_csv(fs.base_location+'/NSE-TATASTLBSL.csv')
        # with open(fs.base_location+'/NSE-TATASTLBSL.csv', 'r') as f:
        #     reader = csv.reader(f)
        #     for row in reader:
        #         try:
        #
        #             if str(row[0]).__contains__('-'):
        #                 # values={str(row[0]), float(row[1])}
        #                 print(row[0])
        #                 print(row[1])
        #                 values={row[0],row[1]}
        #                 print(values)
        #                 stockList.append(values)
        #
        #         except ValueError:
        #             print(ValueError)

        # stockList.append({'date','price'})
        def get_data(filename):
            # Date,Open,High,Low,Close,Volume
            '''
            Reads data from a file (snap.csv) and adds data to
            the lists dates and prices
            '''
            # Use the with as block to open the file and assign it to csvfile
            with open(filename, 'r') as csvfile:
                # csvFileReader allows us to iterate over every row in our csv file
                csvFileReader = csv.reader(csvfile)
                next(csvFileReader)  # skipping column names
                for row in csvFileReader:

                    # print({str(row[0]), float(row[1])})
                    # print(row)
                    # stockList.append({int(row[0].split('-')[2]), float(row[1])})
                    dates.append(int(row[0].split('-')[0]))  # Only gets day of the month which is at index 0
                    # dates.append(int(row[0].split('-')[2]))  # Only gets day of the month which is at index 0
                    prices.append(float(row[1]))  # Convert to float for more precision


            return

        def predict_price(dates, prices, x):
            print("dates" + str(dates))
            print("prices=" + str(prices))
            print("x" + str(x))
            '''
            Builds predictive model and graphs it
            This function creates 3 models, each of them will be a type of support vector machine.
            A support vector machine is a linear seperator. It takes data that is already classified and tries
            to predict a set of unclassified data.
            So if we only had two data classes it would look like this
            It will be such that the distances from the closest points in each of the two groups is farthest away.
            When we add a new data point in our graph depending on which side of the line it is we could classify it
            accordingly with the label. However, in this program we are not predicting a class label, so we don't
            need to classify instead we are predicting the next value in a series which means we want to use regression.
            SVM's can be used for regression as well. The support vector regression is a type of SVM that uses the space between
            data points as a margin of error and predicts the most likely next point in a dataset.
            The predict_prices returns predictions from each of our models

            '''
            dates = np.reshape(dates, (len(dates), 1))  # converting to matrix of n X 1
            print("dates1=" + str(dates))
            # Linear support vector regression model.
            # Takes in 3 parameters:
            # 	1. kernel: type of svm
            # 	2. C: penalty parameter of the error term
            # 	3. gamma: defines how far too far is.

            # Two things are required when using an SVR, a line with the largest minimum margin
            # and a line that correctly seperates as many instances as possible. Since we can't have both,
            # C determines how much we want the latter.

            # Next we make a polynomial SVR because in mathfolklore, the no free lunch theorum states that there are no guarantees for one optimization to work better
            # than the other. So we'll try both.

            # Finally, we create one more SVR using a radial basis function. RBF defines similarity to be the eucledian distance between two inputs
            # If both are right on top of each other, the max similarity is one, if too far it is a ze
            # svr_lin = SVR(kernel= 'linear', C= 1e3) # 1e3 denotes 1000
            # svr_poly = SVR(kernel= 'poly', C= 1e3, degree= 2)
            svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)  # defining the support vector regression models

            svr_rbf.fit(dates, prices)  # fitting the data points in the models
            # svr_lin.fit(dates, prices)
            # svr_poly.fit(dates, prices)

            # This plots the initial data points as black dots with the data label and plot
            # each of our models as well

            # plt.scatter(dates, prices, color= 'black', label= 'Data') # plotting the initial datapoints
            # The graphs are plotted with the help of SVR object in scikit-learn using the dates matrix as our parameter.
            # Each will be a distinct color and and give them a distinct label.
            # plt.plot(dates, svr_rbf.predict(dates), color= 'red', label= 'RBF model') # plotting the line made by the RBF kernel
            # plt.plot(dates,svr_lin.predict(dates), color= 'green', label= 'Linear model') # plotting the line made by linear kernel
            # plt.plot(dates,svr_poly.predict(dates), color= 'blue', label= 'Polynomial model') # plotting the line made by polynomial kernel
            # plt.xlabel('Date') # Setting the x-axis
            # plt.ylabel('Price') # Setting the y-axis
            # plt.title('Support Vector Regression') # Setting title
            # plt.legend() # Add legend
            # plt.show() # To display result on screen
            print(svr_rbf.predict(np.reshape([x], (len([x]), 1)))[0])
            return svr_rbf.predict(np.reshape([x], (len([x]), 1)))[0]  # returns predictions from each of our models

        # return svr_rbf.predict(np.reshape([x],(len([x]), 1)))[0], svr_lin.predict(np.reshape([x],(len([x]), 1)))[0], svr_poly.predict(np.reshape([x],(len([x]), 1)))[0] # returns predictions from each of our models

        get_data(path1)  # calling get_data method by passing the csv file to it
        predicted_price = predict_price(dates, prices, curdat)
        print('The predicted prices are:', predicted_price)
        stockList.append(['Prices','Dates'])
        count=0;
        for p in dates:
            # stockList.append({'Date':p,'Price':prices[count]})
            stockList.append([p,prices[count]])
            count=count+1
    print(stockList)
    return render(request, 'upload_file.html', {'prediction':predicted_price,'stockList':stockList,'dates':dates,'prices':prices})