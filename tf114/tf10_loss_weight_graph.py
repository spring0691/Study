import tensorflow as tf, matplotlib.pyplot as plt

x = [1,2,3]
y = [1,2,3]
w = tf.compat.v1.placeholder(tf.float32)

hypothesis = x*w

loss = tf.reduce_mean(tf.square(hypothesis-y))                     # loss, error, cost 다 같은 의미

w_history = []
loss_history = []

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

from matplotlib import font_manager, rc
font_path = "C:\Windows\Fonts/NanumBarunGothic.ttf"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

plt.plot(w_history,loss_history)
plt.xlabel("Weight")
plt.ylabel("Loss")
plt.title('웨이트-로스 그래프')
plt.show()