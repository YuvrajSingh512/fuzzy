str = input("Enter your string: ")
words=str.split(' ')
words=[word for word in words if len(word)%2==0]

min_length=len(words[0])
ans=[]
for i in range(0,len(words)):
    if len(words[i])<min_length:
        min_length=len(words[i])
        ans=[words[i]]
    elif len(words[i])==min_length:
        ans.append(words[i])

print("Minimum length even word: ",ans)
