first_names= ['samin', 'jack', 'ali', 'helen', 'maryam']  #part 1
last_names= ['naji', 'hoffmann', 'alavi', 'fischer', 'mirzaee']  
birth_dates= [2000, 1976, 2002, 1989, 2006]   
heights= [1.68, 1.80, 1.75, 1.65, 1.70]   
weights= [52, 75, 90, 45, 64]
k=-1
for i,j in zip(first_names, last_names):
    i=i.upper()
    j=j.upper()
    k+=1
    first_names[k]=i
    last_names[k]=j
information = [first_names, last_names ,heights, weights]   #part 3
blood_pressures= [105, 127, 119, 112, 121]   #part 2
hospitalization_history= [False, True, False, False, True]
children= [0, 2, 0, 1, 3]
information.append(blood_pressures)
information.append(hospitalization_history)
information.append(children)
for i in birth_dates:   #part 4
    i= 2023 - i  
information.append(birth_dates) 
BMI= []
for i,j in zip(weights, heights):   #part 5
    bmi= i/(j**2)
    BMI.append(bmi)
information.append(BMI)
for i,j in zip(first_names, last_names):   #part 6
    print(f'dear {i} {j} your information successfully save!')
print(f'your information = {information}')



```Get the information of 5 patients as follows:
1) Take the person's name, surname, year of birth, height, and weight as input from the person.
2) Add the medical information of the people that already exists in your system, such as history of hospitalization,
blood pressure, number of children, etc., to the information of each person.
3) Store the person's information in a way that the order of the information is important and the information can be changed. 
4) Calculate the person's age based on the person's year of birth and substitute" the age in their information instead of the year of birth. 
5) Calculate the person's BMI using the height and weight values ​​and "add" it to the person's previous information.
6) Finally, display a sentence to the person indicating that the person's information (including each person's first and last name) has been successfully saved.
The first and last names of individuals should be stored in uppercase letters (the individual does not follow this rule when entering information, 
but we want the information 
to be stored in all uppercase letters 
when we store it. The numbers taken from the individual should be in a way that allows numerical calculations to be performed.```
