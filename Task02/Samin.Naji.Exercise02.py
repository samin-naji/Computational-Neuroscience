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