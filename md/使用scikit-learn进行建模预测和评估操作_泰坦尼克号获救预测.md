

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```


```python
titanic = pd.read_csv('train.csv')
titanic.head(5)
# print(titanic.describe())
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())
print(titanic.describe())
```

           PassengerId    Survived      Pclass         Age       SibSp  \
    count   891.000000  891.000000  891.000000  891.000000  891.000000   
    mean    446.000000    0.383838    2.308642   29.361582    0.523008   
    std     257.353842    0.486592    0.836071   13.019697    1.102743   
    min       1.000000    0.000000    1.000000    0.420000    0.000000   
    25%     223.500000    0.000000    2.000000   22.000000    0.000000   
    50%     446.000000    0.000000    3.000000   28.000000    0.000000   
    75%     668.500000    1.000000    3.000000   35.000000    1.000000   
    max     891.000000    1.000000    3.000000   80.000000    8.000000   
    
                Parch        Fare  
    count  891.000000  891.000000  
    mean     0.381594   32.204208  
    std      0.806057   49.693429  
    min      0.000000    0.000000  
    25%      0.000000    7.910400  
    50%      0.000000   14.454200  
    75%      0.000000   31.000000  
    max      6.000000  512.329200  
    


```python
print(titanic['Sex'].unique())

# Replace all the occurences of male with the number 0.
# 将字符值转换成 数值
# 进行一个属性值转换
titanic.loc[titanic['Sex'] == 'male', 'Sex'] = 0
titanic.loc[titanic['Sex'] == 'female', 'Sex'] = 1
```

    ['male' 'female']
    


```python
# 登船地址
print(titanic['Embarked'].unique())
titanic['Embarked'] = titanic['Embarked'].fillna('S')
titanic.loc[titanic['Embarked'] == 'S', 'Embarked'] = 0
titanic.loc[titanic['Embarked'] == 'C', 'Embarked'] = 1
titanic.loc[titanic['Embarked'] == 'Q', 'Embarked'] = 2
```

    ['S' 'C' 'Q' nan]
    


```python
# Import the linear regression class (线性回归)
from sklearn.linear_model import LinearRegression
# Sklearn also has a helper that makes it easy to do cross validation(交叉验证)
from sklearn.cross_validation import KFold

# The Columns we'll use to predict the target
predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

# Initialize our algorithm class
alg = LinearRegression()
# Generate(生成) cross validation folds(交叉验证) for the titanic dataset.
# We set random_state to ensure we get the same splits(相同的分割) every time we run this.
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)

# 预测结果
predictions = []
# 训练集, 测试集, 交叉验证
for train, test in kf:
    # The predictors we're using the train the algorithm. 
    # Note how we only take the rows in the train folds (只在训练集中进行)
    train_predictors = (titanic[predictors].iloc[train, :])
    # The target we're using to train the algorithm
    train_target = titanic['Survived'].iloc[train]
    # Training the algorithm using the prodictors and target
    # 训练数据的 X, Y ==> 让他能进行判断的操作
    alg.fit(train_predictors, train_target)
    # we can now make predictions on the test fold
    test_predictions = alg.predict(titanic[predictors].iloc[test, :])
    predictions.append(test_predictions)
```


```python
import numpy as np

# The Predictions are in three separate numpy arrays. Concatenate them into one.
# We concatenate them on axis 0, as they only have one axis.我们将它们连接在轴0上，因为它们只有一个轴
predictions = np.concatenate(predictions, axis = 0)

# Map predictions to outcomes (only possible outcome are 1 and 0)
predictions[predictions > 0.5] = 1
predictions[predictions <= .5] = 0

# 进行评估模型
accuracy = sum(predictions[predictions == titanic['Survived']]) / len(predictions)
print(accuracy)
```

    0.783389450056
    

    D:\Anaconda3\lib\site-packages\ipykernel\__main__.py:12: FutureWarning: in the future, boolean array-likes will be handled as a boolean array index
    


```python
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression

# Initialize our algorithm
alg = LogisticRegression(random_state=1)
# Compute the accuracy score for all the cross validation folds. (计算所有交叉验证折叠的精度分数。)
# (much simpler than what we did before !)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic['Survived'], cv=3)
# Take the mean of the scores (because we have one for each fold)
print(scores.mean())
```

    0.787878787879
    

### 随机森林


```python
titanic_test = pd.read_csv('test.csv')
titanic_test['Age'] = titanic_test['Age'].fillna(titanic['Age'].median())
titanic_test['Fare'] = titanic_test['Fare'].fillna(titanic_test['Fare'].median())
titanic_test.loc[titanic_test['Sex'] == 'male', 'Sex'] = 0
titanic_test.loc[titanic_test['Sex'] == 'female', 'Sex'] = 1
titanic_test['Embarked'] = titanic_test['Embarked'].fillna('S')

titanic_test.loc[titanic_test['Embarked'] == 'S', 'Embarked'] = 0
titanic_test.loc[titanic_test['Embarked'] == 'C', 'Embarked'] = 1
titanic_test.loc[titanic_test['Embarked'] == 'Q', 'Embarked'] = 2
```


```python
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

#选中一些特征 
predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

# Initialize our algorithm with the default paramters
# random_state = 1 表示此处代码多运行几次得到的随机值都是一样的，如果不设置，两次执行的随机值是不一样的
# n_estimators  指定有多少颗决策树，树的分裂的条件是:
# min_samples_split 代表样本不停的分裂，某一个节点上的样本如果只有2个了 ，就不再继续分裂了
# min_samples_leaf 是控制叶子节点的最小个数
alg = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)

# Compute the accuracy score for all the cross validation folds (nuch simpler than what we did before)
# 进行交叉验证 
kf = cross_validation.KFold(titanic.shape[0], n_folds=3, random_state=1)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic['Survived'], cv=kf)

# Take the mean of the scores (because we have one for each fold)
print(scores.mean())
```

    0.785634118967
    


```python
# 建立100多个决策树
alg = RandomForestClassifier(random_state=1, n_estimators=100, min_samples_split=4, min_samples_leaf=2)
# Compute the accuracy score
kf = cross_validation.KFold(titanic.shape[0], n_folds=3, random_state=1)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic['Survived'], cv=kf)

# Take the mean of the scores (because we have one for each fold)
print(scores.mean())
```

    0.814814814815
    

## 关于特征提取问题 (非常关键)
- 尽可能多的提取特征
- 看不同特征的效果
- 特征提取是数据挖掘里很- 要的一部分
- 以上使用的特征都是数据里已经有的了，在真实的数据挖掘里我们常常没有合适的特征，需要我们自己取提取



```python
# Generating a familysize column
# 合并数据 ：自己生成一个特征，家庭成员的大小:兄弟姐妹+老人孩子
titanic['FamilySize'] = titanic['SibSp'] + titanic['Parch']

# The .apply method generates a new series 名字的长度(据说国外的富裕的家庭都喜欢取很长的名字)
titanic['NameLength'] = titanic['Name'].apply(lambda x: len(x))
```


```python
import re

# A function to get the title from a name
def get_title(name):
    # Use a regular expression to search for a title.
    # Titles always consist of capital and lowercase letters.
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists extract and return it.
    if title_search:
        return title_search.group(1)
    return ""
# Get all the titles and print how often each one occurs.
titles = titanic['Name'].apply(get_title)
print(pd.value_counts(titles))          # 输出看看, 相同数量的,设置相同映射

# 国外不同阶层的人都有不同的称呼
# Map each title to an integer. Some titles are very rare. and are compressed into the same codes as other
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, 
                 "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2 }
for k, v in title_mapping.items():
     #将不同的称呼替换成机器可以计算的数字
    titles[titles == k] = v
    
# Verify that we converted  everything
print(pd.value_counts(titles))

# Add in the title column
titanic['Title'] = titles
```

    Mr          517
    Miss        182
    Mrs         125
    Master       40
    Dr            7
    Rev           6
    Major         2
    Col           2
    Mlle          2
    Capt          1
    Sir           1
    Jonkheer      1
    Ms            1
    Countess      1
    Lady          1
    Don           1
    Mme           1
    Name: Name, dtype: int64
    1     517
    2     183
    3     125
    4      40
    5       7
    6       6
    7       5
    10      3
    8       3
    9       2
    Name: Name, dtype: int64
    


```python
# 进行特征选择
# 特征重要性分析
# 分析 不同特征对 最终结果的影响
# 例如 衡量age列的重要程度时，什么也不干，得到一个错误率error1，
# 加入一些噪音数据，替换原来的值(注意，此时其他列的数据不变)，又得到一个一个错误率error2
# 两个错误率的差值 可以体现这一个特征的重要性
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pylab as plt

# 选中一些特征
predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', "Embarked",
             'FamilySize', 'Title', 'NameLength']

# Perform feature selection 选择特性
selector = SelectKBest(f_classif, k = 5)
selector.fit(titanic[predictors], titanic['Survived'])

# Get the raw p-values(P 值) for each feature, and transform from p-values into scores
scores = -np.log10(selector.pvalues_)

# Plot the scores. See how "Plcass", "Sex", "Title", and "Fare" are the best ?
plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical')
plt.show()

# 通过以上的特征重要性分析, 选择出4个最重要的特性，重新进行随机森林的算法
# Pick only the four best features.
predictors = ['Pclass', 'Sex', 'Fare', 'Title']

alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=8, min_samples_leaf=4)

# 进行交叉验证
kf = cross_validation.KFold(titanic.shape[0], n_folds=3, random_state=1)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"],cv=kf)
#目前的结果是没有得到提高，本处的目的是为了练习在随机森林中的特征选择，它对于实际的数据挖掘具有重要意义
print (scores.mean())
```

    0.819304152637
    

### 集成多种算法(减少过拟合)


```python
# 在竞赛中常用的耍赖的办法:集成多种算法，取最后每种算法的平均值，来减少过拟合
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

# GradientBoostingClassifier也是一种随机森林的算法，可以集成多个弱分类器，然后变成强分类器
# The algorithm we want to ensemble
# We're using the more linear predictors for the logistic regression
# and everything with the gradient boosting
algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3),["Pclass","Sex","Age","Fare","Embarked","FamilySize","Title"]],
    [LogisticRegression(random_state=1), ["Pclass","Sex","Age","Fare","Embarked","FamilySize","Title"]]
]

# Initialize the cross validation folds
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)

predictions = []
for train, test in kf:
    train_target = titanic['Survived'].iloc[train]
    full_test_predictions = []
    # Make predictions for each algorithm on each folds
    for alg, predictors in algorithms:
        # Fit the algorithm on the training data.
        alg.fit(titanic[predictors].iloc[train, :], train_target)
        # Select and predict on the test fold.
        # The astype(float) is necessary to convert the dataframe
        test_predictions = alg.predict_proba(titanic[predictors].iloc[test, :].astype(float))[:, 1]
        full_test_predictions.append(test_predictions)
    # Use a simple ensembling scheme - just average the predictions to get the final classification
    # 两个算法, 分别算出来的 预测值, 取平均
    test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2
    # Any value over 5 is assumed to be a 1 prediction, and below 5 is a 0 prediction
    test_predictions[test_predictions <= 0.5] = 0
    test_predictions[test_predictions > .5] = 1
    predictions.append(test_predictions)
    
# Put all the predictions together into one array
predictions = np.concatenate(predictions, axis=0)

accuracy = sum(predictions[predictions == titanic['Survived']]) / len(predictions) 
print(accuracy)
```

    0.821548821549
    

    D:\Anaconda3\lib\site-packages\ipykernel\__main__.py:40: FutureWarning: in the future, boolean array-likes will be handled as a boolean array index
    


```python
titles = titanic['Name'].apply(get_title)
print(pd.value_counts(titles))          # 输出看看, 相同数量的,设置相同映射

# 国外不同阶层的人都有不同的称呼
# Map each title to an integer. Some titles are very rare. and are compressed into the same codes as other
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, 
                 "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2 }
for k, v in title_mapping.items():
     #将不同的称呼替换成机器可以计算的数字
    titles[titles == k] = v
# Add in the title column
titanic_test['Title'] = titles
print(pd.value_counts(titanic_test['Title']))

# Now, we add the family size column.
titanic_test['FamilySize'] = titanic_test['SibSp'] + titanic_test['Parch']
```

    Mr          517
    Miss        182
    Mrs         125
    Master       40
    Dr            7
    Rev           6
    Major         2
    Col           2
    Mlle          2
    Capt          1
    Sir           1
    Jonkheer      1
    Ms            1
    Countess      1
    Lady          1
    Don           1
    Mme           1
    Name: Name, dtype: int64
    1    228
    2    101
    3     58
    4     23
    6      3
    5      3
    9      1
    8      1
    Name: Title, dtype: int64
    


```python
predictors = ["Pclass","Sex","Age","Fare","Embarked","FamilySize","Title"]

algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), predictors],
    [LogisticRegression(random_state=1), ["Pclass","Sex","Age","Fare","Embarked","FamilySize","Title"]]
]

full_predictions = []
for alg, predictors in algorithms:
    # Fit the Algorithm using the full training data
    alg.fit(titanic[predictors], titanic['Survived'])
    predictions = alg.predict_proba(titanic_test[predictors].astype(float))[:, 1]
    full_predictions.append(predictions)

# 梯度提升分类器产生更好的预测
# The gradient boosting classifier generates better predictions, so we weight it high
predictions = (full_predictions[0] * 3 + full_predictions[1]) / 4
predictions
```




    array([ 0.11682907,  0.47835549,  0.3982137 ,  0.34432068,  0.24947368,
            0.14352088,  0.37781022,  0.543155  ,  0.67801349,  0.33799158,
            0.3072165 ,  0.56676874,  0.723492  ,  0.10891266,  0.87635012,
            0.87713472,  0.48843752,  0.13907787,  0.57219327,  0.5566099 ,
            0.22420867,  0.21739786,  0.89591201,  0.3889059 ,  0.87430856,
            0.30916409,  0.73864615,  0.1374645 ,  0.58920476,  0.12665716,
            0.37503058,  0.52929758,  0.52123441,  0.21339085,  0.42415294,
            0.14191048,  0.27563384,  0.29143252,  0.32545025,  0.44731459,
            0.31408338,  0.65482038,  0.09996498,  0.81521616,  0.88510666,
            0.14983414,  0.31593408,  0.33344456,  0.61574884,  0.54189554,
            0.68793882,  0.17718128,  0.85195587,  0.89220323,  0.17559064,
            0.1445565 ,  0.28817114,  0.12343847,  0.36566294,  0.9170857 ,
            0.13099157,  0.42988849,  0.12993964,  0.69295292,  0.37558181,
            0.88233584,  0.68621836,  0.28826491,  0.63337333,  0.57295383,
            0.39976126,  0.31496503,  0.3045394 ,  0.36740456,  0.73719756,
            0.41201923,  0.13014001,  0.39865651,  0.4816345 ,  0.66224133,
            0.36447486,  0.20605726,  0.56077175,  0.12105177,  0.47229864,
            0.34433881,  0.39358601,  0.29779125,  0.65489623,  0.38762121,
            0.26753289,  0.12104025,  0.68837775,  0.13014001,  0.2840624 ,
            0.12345364,  0.6007447 ,  0.14666334,  0.61395768,  0.12260778,
            0.89549828,  0.14730813,  0.13789618,  0.1226243 ,  0.31835962,
            0.13155871,  0.34368087,  0.13789618,  0.13020333,  0.45502839,
            0.14286384,  0.65490308,  0.71264396,  0.67146754,  0.86894944,
            0.13992076,  0.1180506 ,  0.36213675,  0.36668938,  0.84325029,
            0.54396489,  0.12609323,  0.7167887 ,  0.30713583,  0.13789618,
            0.61490447,  0.12608178,  0.30858902,  0.40653561,  0.13340571,
            0.12723634,  0.23322059,  0.23921867,  0.29926306,  0.09896734,
            0.12431121,  0.32778968,  0.16214093,  0.28684605,  0.12232635,
            0.23306241,  0.90529652,  0.52408542,  0.16153711,  0.42927583,
            0.10487175,  0.3364249 ,  0.32866031,  0.46618801,  0.34478751,
            0.86374489,  0.33874763,  0.10690995,  0.21485897,  0.12728529,
            0.12427865,  0.91070163,  0.32526102,  0.42927583,  0.53314235,
            0.39143644,  0.56151009,  0.52297138,  0.12096645,  0.45057218,
            0.60608807,  0.60071411,  0.12554362,  0.71298343,  0.28931189,
            0.1210188 ,  0.35589568,  0.32005039,  0.13207483,  0.13196549,
            0.59215   ,  0.89725408,  0.64299954,  0.51627853,  0.55130038,
            0.43319429,  0.33352245,  0.90587628,  0.36504198,  0.9139123 ,
            0.13602996,  0.87178869,  0.122414  ,  0.19982276,  0.1356068 ,
            0.51295102,  0.25547176,  0.46375191,  0.42713042,  0.69842416,
            0.28987418,  0.58261615,  0.3361868 ,  0.44546019,  0.50975301,
            0.34780152,  0.35109344,  0.36683604,  0.52032964,  0.16322633,
            0.60979018,  0.35999986,  0.16482586,  0.88984705,  0.12346404,
            0.1284965 ,  0.30711088,  0.24675034,  0.4133723 ,  0.18510508,
            0.5753533 ,  0.65492655,  0.21860349,  0.89250428,  0.13014001,
            0.50572769,  0.13611139,  0.52891054,  0.12700825,  0.58257812,
            0.2931972 ,  0.12518084,  0.39144083,  0.1148749 ,  0.42333728,
            0.60375528,  0.8074908 ,  0.1162268 ,  0.3334662 ,  0.34224636,
            0.31618214,  0.19365854,  0.33633969,  0.50002374,  0.70852783,
            0.86515996,  0.80108415,  0.33036563,  0.12105098,  0.10622824,
            0.50337665,  0.86400035,  0.49430688,  0.49697828,  0.61153743,
            0.37500773,  0.3866276 ,  0.39647443,  0.13354265,  0.33763824,
            0.32778001,  0.34783359,  0.32210217,  0.83005784,  0.33757958,
            0.10894951,  0.35345262,  0.48014394,  0.36095429,  0.44617229,
            0.12105177,  0.21821006,  0.1210188 ,  0.53978836,  0.33503395,
            0.34495856,  0.13789618,  0.91564002,  0.32400418,  0.32599878,
            0.85713529,  0.43616999,  0.12500111,  0.53759674,  0.47750693,
            0.28701473,  0.35055012,  0.39144083,  0.33562957,  0.36094497,
            0.10601081,  0.12099023,  0.36278084,  0.13207483,  0.32210217,
            0.61321335,  0.61059162,  0.13207483,  0.56862789,  0.12081673,
            0.12263652,  0.40995524,  0.39814248,  0.33024478,  0.32655723,
            0.30338162,  0.1754788 ,  0.12169405,  0.32644531,  0.39144083,
            0.83833526,  0.61445161,  0.65771976,  0.20916502,  0.39458229,
            0.33371629,  0.33357009,  0.327784  ,  0.34890831,  0.63237276,
            0.67393643,  0.60571209,  0.37442284,  0.30823329,  0.53296057,
            0.1226243 ,  0.13491475,  0.44037379,  0.64370031,  0.72938311,
            0.31248684,  0.56732433,  0.64772746,  0.52190705,  0.44998322,
            0.79625693,  0.32823562,  0.13207435,  0.31362164,  0.32913718,
            0.25111391,  0.15333419,  0.27359561,  0.20950799,  0.13207483,
            0.50783666,  0.3033597 ,  0.1469565 ,  0.69762559,  0.11655105,
            0.61836796,  0.44037379,  0.62462669,  0.49512259,  0.51582896,
            0.70542388,  0.16322633,  0.24472812,  0.16066607,  0.33549395,
            0.15642446,  0.8352084 ,  0.30721442,  0.33344456,  0.54116738,
            0.14820072,  0.59925644,  0.86918834,  0.13098154,  0.75333784,
            0.19982276,  0.34433945,  0.56081081,  0.8864177 ,  0.3987672 ,
            0.15319839,  0.73421636,  0.16307929,  0.13130568,  0.8578672 ,
            0.91302647,  0.48853352,  0.17002321,  0.19866957,  0.1495549 ,
            0.33344456,  0.33623101,  0.27743803,  0.59499236,  0.15905629,
            0.56377895,  0.34816891,  0.38946464,  0.14606633,  0.16117912,
            0.38579314,  0.58112929,  0.19459153,  0.39974246,  0.1466562 ,
            0.72531907,  0.33900599,  0.69213205,  0.34699844,  0.33273973,
            0.73144465,  0.12635158,  0.90891637,  0.35988715,  0.58226025,
            0.18966798,  0.15015206,  0.66185656,  0.39143315,  0.64585314,
            0.39144083,  0.72445663,  0.56933469,  0.13014001,  0.72642174,
            0.29563097,  0.34291789,  0.39821026])




```python

```


```python

```
