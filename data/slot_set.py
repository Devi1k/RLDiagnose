#生成slot_set
#我是把slot_set打印出来然后复制到configuration.py的（是很傻逼（..）你想想别的方法吧
from configuration import requirement_weight as w
dict={}
for i in range(len(w)):
    dict = {**dict, **w[i]}
new_dict=list(dict.keys())
slot_set={}
keys=new_dict
v=0
for i in keys:
        slot_set[i] =v
        v+=1
print(slot_set)

