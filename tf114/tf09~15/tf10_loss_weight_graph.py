import tensorflow as tf, matplotlib.pyplot as plt, matplotlib as mpl

x = [1,2,3]
y = [1,2,3]
w = tf.compat.v1.placeholder(tf.float32)

hypothesis = x*w

loss = tf.reduce_mean(tf.square(hypothesis-y))                     # loss, error, cost 다 같은 의미

w_history = []
loss_history = []

# print(mpl.matplotlib_fname())
# C:\ProgramData\Anaconda3\envs\tf114\lib\site-packages\matplotlib\mpl-data\matplotlibrc
# print(mpl.get_cachedir())
# C:\Users\비트캠프\.matplotlib
mpl.rcParams['font.family'] = 'NanumGothic'
mpl.rcParams['font.size'] = 15

with tf.compat.v1.Session() as sess:
    i = -48
    while True:
        i += 1
        curr_w = i
        curr_loss = sess.run(loss,feed_dict={w:curr_w})
        
        w_history.append(curr_w)
        loss_history.append(curr_loss)
        
        if i == 50:
            break

print("======================= W history =======================")
print(w_history)
print("====================== loss history =====================")
print(loss_history)

plt.plot(w_history,loss_history)
plt.xlabel("Weight")
plt.ylabel("Loss")
plt.title('웨이트-로스 그래프')
plt.show()
