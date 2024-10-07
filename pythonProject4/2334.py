
def StringCount(n): #2
    try:
        n = str(n)
    except:
        return 404
    s = True
    n = n.lower()
    for i in range(len(n)):
        if n.count(n[i]) > 1:
            s = False
            break
    return s


def StringVowel(n): #1
    try:
        n = str(n)
    except:
        return 404
    A = ["a", "e", "i", "o", "u"]
    s = 0
    n = n.lower()
    for i in range(len(A)):
        s += n.count(A[i])
    return s

def CountBits(n): #3
    s = 0
    try:
        n = int(n)
    except:
        return 404
    n = int(n)
    while n > 0:
        n &= n - 1
        s += 1
    return s

def magic(n): #4
    s = 1
    count = 0
    try:
        n = int(n)
    except:
        return 404
    while n > 10:
        for a in str(n):
            s *= int(a)
        n = s
        count +=1
        s = 1
    return count
def MultiNumber(n): #6
   i = 2
   A = []
   B = []
   while i * i <= n:
       if n %i == 0:
           B.append(i)
       while n % i == 0:
           A.append(i)
           n = n / i
       i = i + 1
   if n > 1:
       A.append(n)
       B.append(n)
   res = " "
   for i in range(len(B)):
       if A.count(B[i]) == 1:
           res += str(B[i]) + " "
       else:
           res += str(B[i]) + "**" + str(A.count(B[i])) + " "
   return res

def AvergSum(n): #8
    try:
        n = int(n)
    except:
        return 404
    if n < 0:
        return 404
    n = str(n)
    s = 0
    z = 0
    if len(n) % 2 == 0:
        for i in range((len(n)//2)-1):
            s +=int(n[i])
            z +=int(n[len(n)-i-1])
    else:
        for i in range(len(n)//2):
            s += int(n[i])
            z += int(n[len(n)-i-1])

    return s == z

print(AvergSum(12312))
print(MultiNumber(9975))
#a = StringVowel("123321")
#print(a)
#a = StringCount("asda")
#print(a)
#a = CountBits(123)
#print(a)
#a = CountArg(4)
#print(a)