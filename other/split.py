#处理split.txt文件中的数据格式
import json
with open('split.txt') as f:
    lines = f.readlines()

s='service='
r='requirement='
service = []
requirement = []
for line in lines:
    line = line.strip('\n')
    service.append(line.split(":")[0])
    requirement.append(line.split(":")[1].split(" "))
print(service)
print(requirement)

json_str0 = json.dumps(s,ensure_ascii=False)
json_str1= json.dumps(service,indent=1,ensure_ascii=False)
json_str2= json.dumps(r,indent=1,ensure_ascii=False)
json_str3= json.dumps(requirement,indent=1,ensure_ascii=False)
with open("slot.json", "w",encoding='utf-8') as f:
    f.write(json_str0)
    f.write(json_str1)
    f.write(json_str2)
    f.write(json_str3)
