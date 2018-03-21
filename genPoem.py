#-*- coding: UTF-8 -*-
from testSentence import *
from testPoem import *
from tools import *

title,context = read_pingShuiYun()

complete = False
while not complete:
    first = gen_firstSentence(5)
    if len(first) > 5:
        print('字数超过')
        continue
    result = genNextSentence(first,5)
    if pattern_match(get_tone(title,context,result)) > -1 and get_rhythm(title,context,result) > -1:
        break
    else:
        print(pattern_match(get_tone(title,context,result)))
        print(get_rhythm(title,context,result))
print(result)
        
