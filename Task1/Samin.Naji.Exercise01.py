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