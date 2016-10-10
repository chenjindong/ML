import pandas as pd

data = pd.read_csv("C:\\Users\\cjd\\Desktop\\titanic\\train.csv")
train = data.drop(['PassengerId', 'Name'], axis=1)


# class_total = train.groupby('Pclass').count().Survived
# class_sruvived = train[train.Survived == 1].groupby('Pclass').count().Survived
# print(class_total)
# print(class_sruvived)
# print(class_sruvived/class_total)

# sex_total = train.groupby('Sex').count().Survived
# sex_survived = train[train.Survived == 1].groupby('Sex').count().Survived
# print(sex_total)
# print(sex_survived)
# print(sex_survived/sex_total)

age_total = [0 for i in range(9)]
age_survived = [0 for i in range(9)]
ages = train.Age
surviveds = train.Survived.tolist()
for age in ages:
    if age.isnan():
        print(age)

print(age_survived)
print(age_total)
print(age_survived/age_total)


