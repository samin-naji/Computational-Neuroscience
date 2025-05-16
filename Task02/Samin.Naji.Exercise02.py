from datetime import datetime
blood_pressures= [105, 127, 119, 112, 121, 122, 124, 130, 133, 100]  #part 1
hospitalization_history= [False, True, False, False, True, False, True, False, False, True]
children= [0, 2, 0, 1, 3, 4, 2, 2, 1, 3]
information=[]
user=[]
for i in range(0,2):
    first_name=input("please enter your first name: ")
    last_name=input("please enter your last name: ")
    x=int(input("please enter your weight in kg "))
    date_of_birth_str =input("Enter your date of birth (dd/mm/yyyy): ")  #part 2
    date_of_birth = datetime.strptime(date_of_birth_str, "%d/%m/%Y")
    current_date = datetime.now()
    age = current_date.year - date_of_birth.year - ((current_date.month, current_date.day) < (date_of_birth.month, date_of_birth.day))
    z=float(input("please enter your height in cm: "))
    for j,s,t in zip(blood_pressures, hospitalization_history, children):
        user.append([first_name, last_name, x, age ,z, j ,s ,t])
        continue
    information.append(user)
print(information)
#part 3
information_2=tuple(tuple(row)
for row in information)
#part 4
password= '12345'
for k in range(0,4):
    user_password= input('please enter your password: ')

    if password==user_password:
        user_name=input("please enter your first name: ")
        for i in user:
            if user_name in i:
                print(f'dear {i[0]} {i[1]} your information successfully save!')
                break
        else:
            print("your name isn't saved!")   
        break
else:
    print("The system is locked and you don't have any permission to access information")



```1. Like in the previous exercise, we need to get the information about these 10 people, namely their name, surname, 
year of birth, height and weight, from the individuals themselves, and add the medical information of the individuals 
that is already in your system, such as history of hospitalization, blood pressure, number of children, etc., 
to each person's information.
2. This time we need to be more precise. Instead of the year of birth, we take the date of birth of the people, 
which includes the day, month, and year, separated by "/", and subtract today's day, month, and year to get the people's 
exact age and save those numbers (no need to calculate BMI).
3. We store people's information in a way that doesn't matter in order and is immutable.
4. We set a 4-digit password for the system and require each user to enter the password to open the system. 
Users can enter the password incorrectly up to 4 times and try again. The following steps may occur:
If the password is entered correctly, we ask for the user's name, and if the name is one of the names stored in the 
user names (part 1), it means that the person's information exists in the system, and we display a sentence for the 
person indicating that the person's information (including the first and last name of each person) has been successfully 
stored.
If the user entered the password correctly, but when we asked for their name, their name was not among the saved names, 
We tell them that their name had not been entered before.
If the user enters the wrong password more than 4 times, we tell them that the system is locked and they are not 
allowed to access the information.```
