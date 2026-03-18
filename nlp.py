#s.startswith()
#s.endswith()

text="I Love python and Iam learn it becuse ,python is strong , importance language"
tc="thon"
print(text.startswith(tc))
print(text.startswith("I"))
print(text.endswith("lan"))
end="ge"
print(text.endswith(end))
print(text.isupper())
print(text.islower())
print("python" in text)
print(text.istitle())
print(text.isalpha())
print("omar".isalpha())
text2="1234567890123123456456567567789789"
print(text2.isdigit())
gi="567890ertyuiopcxvbnm".isalnum()
print(gi)
gi2="234567890-=-!@#$%^&*"
print(gi2.isalnum())
pemi="CAPTEN : ARE YOU HERE TALK TO ME PLEASE "
print(pemi.lower())
pemi2="yes ,iam here but iwant to go faster iam not possebily for talk you "
print(pemi2.upper())
print(pemi2.title())
momo='Iam here , Iwanna to play game with you are you ok'
mn=momo.split(" ")
print(mn)
meme=["I do not",",","I will ","be ","study","for ","my university"]
res=" ".join(meme)
print(res)
q="                     LOndon            "
p="                     LOndon"
print(q.rstrip())
print(p.strip())
print(momo.find('Iwanna'))
print(momo.replace("Iam ","Iam not "))
print(momo.replace("Iwanna ","I wont "))
my_text="iam engneering in ai . i work online . ihava ambition to be afreelancer. i work for take  that"
print(my_text.splitlines())














