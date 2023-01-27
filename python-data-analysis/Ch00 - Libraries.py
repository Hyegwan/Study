#!/usr/bin/env python
# coding: utf-8

# # Libraries for Data Analysis (Pandas, Numpy, Matplotlib)

# In[3]:


# 라이브러리 설치
get_ipython().system('pip install pandas numpy matplotlib')


# ### 판다스란
# - 파이썬에서 가장 널리 사용되는 데이터 분석 라이브러리로
# - 데이터 프레임이라는 자료구조를 사용한다
# - 데이터 프레임은 엑셀의 스프레드시트와 유사한 형태이며 파이썬으로 데이터를 쉽게 처리할 수 있도록 한다

# In[6]:


# 판다스 라이브러리 불러오기
import pandas as pd


# 데이터 프레임 생성

# In[7]:


# 판다스의 데이터 프레임 생성
names = ['Bob', 'Jessica', 'Mary', 'John', 'Mel']
births = [968, 155, 77, 578, 973]
custom = [1, 5, 25, 13, 23232]

BabyDataSet = list(zip(names, births))
df = pd.DataFrame(data = BabyDataSet, columns = ['Names', 'Births'])

# 데이터 프레임 상단 부분 출력
df.head()


# In[8]:


BabyDataSet


# 데이터 프레임 기본 정보 출력

# In[9]:


# 데이터 프레임 열 타입 정보
df.dtypes


# In[10]:


# 데이터 프레임 인덱스 정보
df.index


# In[11]:


# 데이터 프레임 열 형태 정보
df.columns


# In[13]:


# 조건을 추가하여 데이터 선택하기
df[df['Births'] > 100]


# ### 넘파이란
# - Numerical Python의 줄임말로
# - 수치 계산을 위해 만들어진 파이썬 라이브러리
# - 넘파이의 자료구조는 pandas, matplotlib 라이브러리의 기본 데이터 타입으로 사용되기도 한다
# - 배열(array)개념으로 변수를 사용하며
# - 벡터, 행렬 등의 연산을 쉽고 빠르게 수행한다

# In[15]:


# 넘파이 라이브러리 불러오기
import numpy as np


# In[17]:


# 넘파이 배열 생성
arr1 = np.arange(15).reshape(3, 5)
arr1


# <u>note</u>: 이 배열은 넘파이 배열이며, 파이썬의 기본 자료구조와는 다른 데이터 타입이다.

# In[21]:


# 0으로 채워진 넘파이 배열 생성
arr2 = np.zeros((3, 4))
arr2


# In[22]:


# 1로 채워진 넘파이 배열 생성
arr3 = np.ones((3, 4))
arr3


# In[25]:


# 넘파이 배열 생성2
arr4 = np.array([
    [1, 2, 3],
    [4, 5, 6]
], dtype = np.float64)

arr4


# In[28]:


# 넘파이 행렬 연산
arr5 = np.dot(arr4, arr1)
arr5


# ### Matplotlib이란
# - 기본적인 데이터 시각화 라이브러리
# - '%matplotlib inline' 선언으로 현재 실행중인 주피터 노트북에서 그래프를 출력할 수 있다

# In[29]:


# matplotlib 라이브러리 불러오기
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[31]:


y = df['Births']
x = df['Names']


# In[32]:


# [1] bar plot 막대 그래프
plt.bar(x, y) # 막대 그래프 객체 생성
plt.xlabel('Names') # 엑스축 제목
plt.ylabel('Births') # 와이축 제목
plt.title('Bar plot') # 그래프 제목
plt.show() # 그래프 출력


# In[33]:


# [2] scatter plot 산점도 그래프
# 랜덤 추출 시드 고정
np.random.seed(19920613) 

# 산점도 데이터 생성
x = np.arange(0.0, 100.0, 5.0) # 5간격으로 0부터 100까지 숫자 생성
y = (x * 1.5) + np.random.rand(20) * 50 # random.rand(): 넘파이 배열 타입의 난수 생성
# rand()에 숫자를 넣어주면 넣은 숫자의 길이만큼 랜덤한 숫자를 생성하고 그것을 1차원 array로 반환한다.

# 산점도 데이터 출력
plt.scatter(x, y, c="b", alpha=0.5, label="scatter point") # 산점도 그래프 객체 생성
# c: color, b: blue
# alpha: The alpha blending value, between 0 (transparent) and 1 (opaque).
# label: legend name
plt.xlabel("X") # 엑스축 제목
plt.ylabel("Y") # 와이축 제목
plt.legend(loc='upper left') # 범례 위치
plt.title('Scatter plot') # 그래프 제목
plt.show() # 그래프 출력

