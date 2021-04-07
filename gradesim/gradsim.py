#!/usr/bin/env python
import os
log_path='./logs/'+input('load from log file or make new one: ')
raw_grade=0
grade=0
total=0
ms=0
es=0
ps=0
grade_values={'m':50,'p':30,'e':20}
def show_file():
    with open(log_path,'r') as df:
        for i in df:
            print(i)

def read_file():
    global total,ms,es,ps,grade,raw_grade
    print('calculating grade')
    show_file()
    raw_grade=0
    grade=0
    denom=0
    ms=[0,0]
    es=[0,0]
    ps=[0,0]
    
    with open(log_path,'r') as df:
        for i in df:
            number,typeof=i.split(',')
            correct,total=number.split('/')
            correct=float(correct)
            total=float(total)
            typeof=typeof[0]
            #total+=grade_values[typeof]
            #raw_grade+=number*grade_values[typeof]
            if typeof=='m':
                ms[0]+=correct
                ms[1]+=total
            elif typeof=='e':
                es[0]+=correct
                es[1]+=total
            else:
                ps[0]+=correct
                ps[1]+=total
        for i,j in zip(['m','e','p'],[ms,es,ps]):
            if j[0]!=0:
                raw_grade+=(j[0]/j[1])*grade_values[i]
            denom+=grade_values[i]
        grade=raw_grade/denom
        #grade=((mn/md)*50+(en/ed)*20+(pn/pd)*30)/100

def is_empty(fname):
    return os.stat(fname).st_size==0
if os.path.isfile(log_path):
    if not is_empty(log_path):
        print(f'opening file {log_path}')
        read_file()
else:
    print('making new file...')
    with open(log_path,'w') as df:
        pass
m='m'
e='e'
p='p'
def rprint(*args):
    print(*[repr(i) for i in args])
def log_grade(correct,total,typeof):
    with open(log_path,'a') as df:
        df.write(f'{correct}/{total},{typeof}\n')

def temp_write(correct,total,mode):
    correct=int(correct)
    total=int(total)
    #print(ms,es,ps)
    if mode=='m':
        return (((ms[0]+correct)/(ms[1]+total))*50+(es[0]/es[1])*20+(ps[0]/ps[1])*30)/100
    elif mode==e:
        return ((ms[0]/ms[1])*50+((es[0]+correct)/(e[1]+total))*20+(ps[0]/ps[1])*30)/100
    else:
        return ((ms[0]/ms[1])*50+(es[0]/es[1])*20+((ps[0]+correct)/(ps[1]+total))*30)/100


def remove_lines(lines=1):
    with open(log_path,'w+') as df:
        ldf=list(df)
        for i in ldf[:-lines]:
            df.write(i)
def remove_targeted_lines(lines=[-1]):
    with open(log_path,'w+') as df:
        ldf=list(df)
        for i in lines:
            del ldf[i]
        for j in ldf:
            df.write(j)
def add_batch(correct,total,mode,times):
    for i in range(times):
        log_to_file(num,mode)
    return 
def wipe_file():
    with open(log_path,'w') as f:
        pass
    return 


doc='''
log_grade(total,correct,typeof)->adds line to record file in the format of grade,type
temp_write(total,correct,typeof)-> simulates a new score added to your grade
remove_lines(lines)-> removes $lines amount of lines from the file,starting from the bottom
show_file()->prints contents of record file
remove_targeted_lines(lines)-> takes list as input;removes those lines from the record file
add_batch(total,correct,mode,times)-> adds $times amount of the same grade in the same category 
read_file()-> re-does the grade calculations to account for any changes
wipe_file()->clears file
'''
print(doc)
def main():
    try:
        while 1:
            usrinput=input('>> ')
            print(eval(usrinput))
    except KeyboardInterrupt:
        print('done')
        return
main()
exit()
