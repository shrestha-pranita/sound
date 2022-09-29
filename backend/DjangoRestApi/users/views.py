from rest_framework.decorators import api_view
from django.http import JsonResponse
from rest_framework.parsers import JSONParser 
from rest_framework import status, response
from users.models import User
from users.serializers import UserSerializer , LoginSerializer ,RegistrationSerializer,PasswordSerializer
from django.contrib.auth.hashers import check_password
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import permission_classes
from rest_framework.generics import get_object_or_404
import jwt
from rest_framework.authtoken.models import Token
from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth import authenticate, login
from datetime import datetime

def get_tokens_for_user(user):
    refresh = RefreshToken.for_user(user)
    return str(refresh), str(refresh.access_token)

@api_view(['GET','POST'])
def login(request):
    """
    login function takes an HttpRequest object and a User object
    :param request: contains request data sent from frontend
    return:  user login
    """
    data=request.data
    serializers=LoginSerializer(data=data)
    if serializers.is_valid():
        user = authenticate(request, username=data['username'], password=data['password'])
        if user is not None:
            request.session["userid"] = user.id
            request.session.modified = True
            refresh, access = get_tokens_for_user(user)
            user_data = {
                "refresh": refresh,
                "access": access,
                "id": user.id,
            }
            user.last_login = datetime.now()
            user.save()
            Msg={"message":"Login Successfull"}
            response = JsonResponse(user_data, status=status.HTTP_200_OK)
            return response
        else: 
            return response.Response({"message": "Sorry, email or password not matched"},status=status.HTTP_400_BAD_REQUEST)            
    else:
        Msg={"error":serializers.errors }
        return response.Response(Msg,status=status.HTTP_400_BAD_REQUEST)
        

@api_view(['GET','POST'])
def register(request):
    """
    register function take the information from the users
    :param request: contains request data sent from frontend
    return:  register the  users    
    """    
    print(request.data)
    serializer=RegistrationSerializer(data= request.data) 
    if serializer.is_valid():
        serializer.save()       
        Msg={"message":"Register Successfull"}
        return response.Response(Msg,status=status.HTTP_200_OK)
    else:
        Msg={"error":serializer.errors }
        return response.Response(Msg,status=status.HTTP_400_BAD_REQUEST)


@api_view(['GET', 'POST', 'DELETE'])
def user_list(request):
    """
    user_list function fetches the list of users from database
    :param request: contains request data sent from frontend
    :return: list of user detail

    """
    if request.method == 'GET':
        try:
            users = User.objects.all()                       
            title = request.query_params.get('title', None)
            if title is not None:
                users = users.filter(title__icontains=title)           
            users_serializer = UserSerializer(users, many=True)
            return JsonResponse(users_serializer.data, safe=False)
        except: 
            return JsonResponse({'message': 'The users does not exist'}, status=status.HTTP_404_NOT_FOUND)    
    elif request.method == 'POST':
        user_data = JSONParser().parse(request)
        user_serializer = UserSerializer(data=user_data)
        if user_serializer.is_valid():
            user_serializer.save()
            return JsonResponse(user_serializer.data, status=status.HTTP_201_CREATED) 
        return JsonResponse(user_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    elif request.method == 'DELETE':
        count = User.objects.all().delete()
        return JsonResponse({'message': '{} Users were deleted successfully!'.format(count[0])}, status=status.HTTP_204_NO_CONTENT)

@api_view(['GET', 'PUT', 'DELETE'])
def user_detail(request, pk):

    """
    user_details function fetches the details of users from database
    param request: contains request data sent from frontend
    param pk: 
    return: list of user detail    
    """
    try: 
        user = User.objects.get(pk=pk) 
    except User.DoesNotExist: 
        return JsonResponse({'message': 'The user does not exist'}, status=status.HTTP_404_NOT_FOUND) 
    if request.method == 'GET': 
        user_serializer = UserSerializer(user) 
        return JsonResponse(user_serializer.data) 
    elif request.method == 'PUT': 
        user_data = JSONParser().parse(request) 
        user_serializer = UserSerializer(user, data=user_data) 
        if user_serializer.is_valid(): 
            user_serializer.save() 
            return JsonResponse(user_serializer.data) 
        return JsonResponse(user_serializer.errors, status=status.HTTP_400_BAD_REQUEST) 
    elif request.method == 'DELETE': 
        user.delete() 
        return JsonResponse({'message': 'User was deleted successfully!'}, status=status.HTTP_204_NO_CONTENT)

@api_view(['GET'])
@permission_classes((IsAuthenticated,))
def get_detail(request):
    return JsonResponse({
        'id': request.user.id,
        'is_approved': request.user.is_approved,
        'contract_signed': request.user.contract_signed,
        'username': request.user.username,
        'email': request.user.email
    })



@api_view(['GET','POST'])
def get_object(self, username):
    user = get_object_or_404(User, username=username)
    return user



@api_view(['put'])
def forgetpassword(request):
    serializer = PasswordSerializer(data=request.data)
    if serializer.is_valid():
        username = serializer.data['username']
        try: 
            user = User.objects.get(username=username)
        except User.DoesNotExist:
            return JsonResponse({"username": ["User with this username does not exist."]}, status=status.HTTP_400_BAD_REQUEST)
        new_password = serializer.data['password']
        is_same_as_old = user.check_password(new_password)
        if is_same_as_old:
                """
                old password and new passwords should not be the same
                """
                return JsonResponse({"password": ["It should be different from your last password."]},
                                status=status.HTTP_400_BAD_REQUEST)
        user.set_password(new_password)
        user.save()
        return JsonResponse({'success':True})
    return JsonResponse(serializer.errors, status=status.HTTP_400_BAD_REQUEST)  
        