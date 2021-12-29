from sklearn.model_selection import train_test_split

x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

x_train, x_test = train_test_split(x, train_size=5, shuffle=True, random_state=49)

print(len(x_train),len(x_test))
print(x_train)
print(x_test)