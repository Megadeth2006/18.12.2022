"""""
#Задача: Нейросеть решает насущную для меня делему, покупать квас или нет.
# 
#  
# Критерии(input):
# 1. Наличие денег
# 2. Как близко магазин К/Б, т.к. там самый вкусный по моему мнению квас.
# 3. Есть ли он там сейчас в наличии.
# 4. Хочу ли я пить.
# Вывод(output):
1. нет денег бан
2. далеко бан
3. нет в наличии БАААН
4. я всегда хочу пить
"""
#Итак пошёл код, для начало сделаем кейс на котором будет трениться моя нейронка


import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import Sequential
from keras.layers import Dense
from tensorflow import keras

c=np.array([
    [1,1,1,1],
   [1,1,1,0],
   [1,1,0,1],
   [1,0,1,1],
   [0,1,1,1],
   [0,0,0,0],
   [1,0,0,0],
   [0,1,1,0],
   [1,1,0,0],
   [1,0,1,0]
   ])
f=np.array([1, 1, 0.6, 0.6, 0, 0, 0, 0, 1, 0])
#ii=>0.93 - готовить дома, 0.1 < ii < 0.93 -  заказывать, ii < 0.1 - не поешь
model = keras.Sequential()
model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(0.1))

history = model.fit(c, f, epochs=600, verbose=0)

# plt.plot(history.history["loss"])
# plt.grid(True)
# plt.show()


""""Далее мы запросим данные и сделаем на их основе предсказание"""

def inputData():
  print("Ввод 1 или 0 !!!")
  a=int(input("Есть ли деньги?  "))
  b=int(input("Есть ли время на готовку?  "))
  c=int(input("Есть ли желание приготовить самостоятельно?  "))
  d=int(input("Есть ли в онлайн-магазине нужное нам блюдо?  "))

  array=[a,b,c,d]
  return array

def otv(array):
  ii=array[0][0]
  print(ii)
  if ii>=0.9:
    return "\n\nГотовь дома"
  elif ii<0.9 and ii >= 0.2:
    return "\n\nЗаказывай в интернет-магазине"
  else:
    return "\n\nПридется поесть в следующий раз"

    
reshenie=model.predict([inputData()])
print(otv(reshenie))
