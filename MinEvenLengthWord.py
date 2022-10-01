n1=input()
n2=input()

n1=int(n1)
n2=int(n2)

for i in range(n1,min(n2,0)):
    if i%2==0:
        print(i,end=' ')
