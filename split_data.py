import re
import pandas as pd


df = pd.read_excel('shujuk.xlsx')
process_col = df['Process']

split_data = process_col.str.split('+', expand=True)
split_data[0] = split_data[0].astype(str)
split_data[1] = split_data[1].astype(str)

# 打印第一阶段数据
# print("0:")
# print(split_data[0])
# print("1:")
# print(split_data[1])

process1 = split_data[0].str.split('/', expand=True)
process2 = split_data[1].str.split('/', expand=True)

print('p1:')
print(process1[0])
print('h1:')
print(process1[1])

t1 = process1[0].str.extract(r'(\d+)℃')
h1 = process1[1].str.extract(r'(\d+)h')
t2 = process2[0].str.extract(r'(\d+)℃')
h2 = process2[1].str.extract(r'(\d+)h')


print('t1:')
print(t1)
print('h1:')
print(h1)


# 更新DataFrame
df['Temp1'] = t1.astype(float).fillna(-1).astype(int)
df['Cooling1'] = h1.astype(float).fillna(-1).astype(int)
df['Temp2'] = t2.astype(float).fillna(-1).astype(int)
df['Cooling2'] = h2.astype(float).fillna(-1).astype(int)

# 保存更新后的Excel文件
df.to_excel('updated_data.xlsx', index=False)

exit()

# 提取第一阶段冷却时间
split_cooling = split_data[0].str.extract(r'(\d+)h')

split_data[1] = split_data[1].astype(str)
# 清除+号前的数据
split_data[1] = split_data[1].str.replace(r'\+\d+℃', '')

# 打印第一阶段数据
print("第一阶段数据:")
print(split_data[0])


# 提取第二阶段温度和冷却时间
split_temp2 = split_data[1].str.extract(r'(\d+)℃')
split_cooling2 = split_data[0].str.extract(r'(\d+)h')

# 打印第二阶段数据
print("第二阶段数据:")
print(split_data[1])

# 更新DataFrame
df['Temp1'] = split_temp.astype(float).fillna(-1).astype(int)
df['Cooling1'] = split_cooling.astype(float).fillna(-1).astype(int)
df['Temp2'] = split_temp2.astype(float).fillna(-1).astype(int)
df['Cooling2'] = split_cooling2.astype(float).fillna(-1).astype(int)

# 处理Cooling2的数据
df['Cooling2'].fillna(df['Cooling1'], inplace=True)

# 打印第一行数据
print("第一行数据:")
print(df.iloc[0])

# 保存更新后的Excel文件
df.to_excel('updated_data.xlsx', index=False)

