


def StringCount(n): #2
    if type(n) != str:
        return 404
    s = True
    n = n.lower()
    for i in range(len(n)):
        if n.count(n[i]) > 1:
            s = False
            break
    return s
def test_StringCount():
    assert StringCount('asasa') == False
    assert StringCount(123) == 404
    assert StringCount('abaleOOU') == False
    assert StringCount('aqwer') == True
def StringVowel(n): #1
    if type(n) != str:
        return 404
    A = ["a", "e", "i", "o", "u"]
    s = 0
    n = n.lower()
    for i in range(len(A)):
        s += n.count(A[i])
    return s
def test_StringVowel():
    assert StringVowel('asasa') == 3
    assert StringVowel(123) == 404
    assert StringVowel('abaleOOU') == 6
test_StringVowel
def CountBits(n): #3
    s = 0
    if type(n) != int:
        return 404
    n = int(n)
    while n > 0:
        n &= n - 1
        s += 1
    return s
def test_CountBits():
    assert CountBits('asasa') == 404
    assert CountBits(123) == 6
    assert CountBits(25) == 3
    assert CountBits(155) == 5

def magic(n): #4
    s = 1
    count = 0
    if type(n) != int:
        return 404
    while n > 10:
        for a in str(n):
            s *= int(a)
        n = s
        count +=1
        s = 1
    return count
def test_magic():
    assert magic('asasa') == 404
    assert magic(39) == 3
    assert magic(9) == 0
    assert magic(999) == 4
print(magic(9))
def MultiNumber(n): #6
   i = 2
   A = []
   B = []
   if type(n) != int:
       return 404
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
def test_MultiNumber():
    assert MultiNumber(86240) == '2**5 5 7**2 11.0 '
    assert MultiNumber(78654) == '2 3 13109.0 '
    assert MultiNumber('asd') == 404
    assert MultiNumber(999) == '2**2 5**2 '

def AvergSum(n): #8
    if type(n) != int:
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
def test_AvergSum():
    assert AvergSum(121) == True
    assert AvergSum(11211) == True
    assert AvergSum('asd') == 404
    assert AvergSum(15099) == False


def pyramid(n): #7
    if type(n) != int:
        return 404
    if n < 0:
        return 404
    k = 1
    while n > 0:
        n -= k**2
        k+=1
    if n == 0:
        return k-1
    return "it is impossible"
def test_AvergSum():
    assert pyramid(1) == 1
    assert pyramid(5) == 2
    assert pyramid('asd') == 404
    assert pyramid(6) == "it is impossible"


#a = StringVowel("123321")
#print(a)
#a = StringCount("asda")
#print(a)
#a = CountBits(123)
#print(a)
#a = CountArg(4)
#print(a)